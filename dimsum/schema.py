import math
from collections.abc import Mapping
from typing import Set, Tuple, List, Union, Optional, Iterable
import numpy as np
import pandas as pd
import grblas
from .container import CodedArray, Flat


NULL = "∅"  # \u2205


class SchemaMismatchError(Exception):
    pass


class Dimension:
    def __init__(self, name, allowed_values, *, ordered=True):
        self.name = name
        self.ordered = ordered

        values = tuple(allowed_values)
        lookup = {v: i for i, v in enumerate(values, 1)}

        if len(values) <= 0:
            raise ValueError("allowed_values is empty")

        if len(lookup) < len(values):
            raise ValueError("duplicate values")

        if None in lookup:
            raise ValueError("`None` is not an allowable value")

        if NULL in lookup:
            raise ValueError("NULL (∅) is not an allowable value")

        # Add in NULL key (Python `None`), always at the zero bit value
        values = (NULL,) + values
        lookup[NULL] = 0

        self.values = values
        self.lookup = lookup
        self.pos2val = pd.Series(values)
        self.val2pos = pd.Series(lookup, dtype=np.uint64)
        self.num_bits = math.ceil(math.log2(len(values)))

    def __eq__(self, other):
        if type(other) is not Dimension:
            return NotImplemented
        return self.values == other.values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def encode(self, data):
        """
        Converts codes into indices

        :param data: pd.Series or iterable
        :return: pd.Series or list
        """
        if isinstance(data, pd.Series):
            return self.val2pos[data]
        else:
            return [self.lookup[x] for x in data]


class Schema(Mapping):
    def __init__(self, dimensions):
        self._dimensions = tuple(dimensions)
        self.names = tuple(dim.name for dim in dimensions)
        self._lookup = {dim.name: dim for dim in dimensions}

        if len(self._lookup) < len(self._dimensions):
            raise ValueError("duplicate dimension names")

        self.offset = {}
        self.mask = {}

        # Populate offsets and masks
        # Use reverse order to make sorting follow first indicated dimension
        bit_pos = 0
        for dim in reversed(self._dimensions):
            self.offset[dim.name] = bit_pos
            self.mask[dim.name] = (2**dim.num_bits - 1) << bit_pos
            bit_pos += dim.num_bits

        if bit_pos > 60:
            raise OverflowError(f"Number of required bits {bit_pos} exceeds the maximum of 60 allowed by GraphBLAS")
        self.total_bits = bit_pos

        # Add calendar if any dimensions are CalendarDimensions
        from . import calendar

        if any(isinstance(dim, calendar.CalendarDimension) for dim in self._dimensions):
            self.calendar = calendar.Calendar(self)

    def __len__(self):
        return len(self._dimensions)

    def __getitem__(self, key):
        if type(key) is int:
            return self._dimensions[key]
        return self._lookup[key]

    def __iter__(self):
        return iter(self._dimensions)

    def __repr__(self):
        r = ['Schema:']
        for dim in self._dimensions:
            r.append(f"  {self.mask[dim.name]:0{self.total_bits}b} {dim.name}")
        return '\n'.join(r)

    def dimension_indices(self, dim, masked=False):
        """
        Returns a new CodedArray containing all values of `dim` and the associated enumerations for each code

        >>> size = Dimension('size', ['small', 'medium', 'large'])
        >>> schema = Schema(['size', ...])
        >>> schema.dimension_indices('size')
            size  * values *
        0      ∅           0
        1  small           1
        2 medium           2
        3  large           3
        """
        if not isinstance(dim, Dimension):
            dim = self[dim]
        offset = self.offset[dim.name]
        indices = dim.val2pos.values
        codes = indices << offset
        if masked:
            indices = codes
        mask = self.dims_to_mask({dim.name})
        dtype = 'UINT64' if masked else 'INT64'
        vec = grblas.Vector.from_values(codes, indices, dtype=dtype, size=mask + 1)
        return CodedArray(Flat(vec, self, (dim.name,)))

    def encode_one(self, **values) -> int:
        code = 0
        for name, val in values.items():
            dim = self._lookup[name]
            if val is None:
                val = NULL
            index = dim.lookup[val]
            offset = self.offset[name]
            code |= index << offset
        return code

    def encode_many(self, values: pd.DataFrame) -> np.ndarray:
        """
        DataFrame headers must match Dimension names exactly
        """
        codes = np.zeros(len(values), dtype=np.uint64)
        for name in values.columns:
            vals = values[name]
            try:
                index = self._lookup[name].val2pos[vals].values
            except ValueError:
                # Most likely a None is present; convert to NULL
                vals = vals.where(pd.notna(vals), NULL)
                index = self._lookup[name].val2pos[vals].values
            offset = self.offset[name]
            codes |= index << offset
        return codes

    def decode_one(self, code, names: Optional[Tuple[str]] = None) -> dict:
        if names is None:
            names = self.names
        values = {}
        for name in names:
            dim = self._lookup[name]
            mask = self.mask[name]
            offset = self.offset[name]
            index = (code & mask) >> offset
            values[name] = dim[index]
        return values

    def decode_many(self, array: np.ndarray, names: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if names is None:
            names = self.names
        df = pd.DataFrame()
        for name in names:
            dim = self._lookup[name]
            mask = self.mask[name]
            offset = self.offset[name]
            index = (array & mask) >> offset
            df[name] = dim.pos2val[index].values
        return df

    def dims_to_mask(self, dims: Set[str] = None) -> int:
        if dims is None:
            return 2**self.total_bits

        mask = 0
        for dim in dims:
            mask |= self.mask[dim]
        return mask

    def mask_to_dims(self, mask: int) -> Set[str]:
        return {dim for dim in self.mask if mask & self.mask[dim]}

    def encode(self, data, dims: List[str] = None, value_column: str = None) -> CodedArray:
        """
        Converts `data` to a CodedArray. `data` may be one of:
        - pd.DataFrame
        - pd.Series
        - dict
        - list of lists

        For a DataFrame, dims must be specified. If more than one column remains after dims
        are accounted for, value_column must be provided. If only a single dimension is required,
        dims may be a str.

        For a Series, it must have a named Index or MultiIndex. The name or level names will be used
        as the dimension names. Providing dims or value_column is not allowed.

        For a dict, the keys must match the shape of dims. For example, if dims = ['Size', 'Shape']
        then the dict keys should look like {('Small', 'Circle'): 12.7, ...}. If only a single dimension
        is required, dims may be a str and dict keys must also be a str.

        For a list of lists, len(dims) must be one less than the length of each row. For example,
        if dims = ['Size', 'Color'], then the data should look like [['Small', 'Red', 54.8], ...].
        The value must come at the end of the row. If the row length is exactly 2, dims may be a str.

        :param data: pd.DataFrame or pd.Series or dict or list of lists
        :param dims: List[str] or str, dimensions
        :param value_column: str column header (only used for pd.DataFrame)
        :return: CodedArray
        """
        if isinstance(data, pd.Series):
            if dims is not None or value_column is not None:
                raise TypeError("`dims` and `value_column` are not allowed when providing a Series")
            return CodedArray.from_series(data, self)
        elif isinstance(data, pd.DataFrame):
            return CodedArray.from_dataframe(data, self, dims, value_column)
        elif isinstance(data, dict):
            if value_column is not None:
                raise TypeError("`value_column` is not allowed when providing a dict")
            return CodedArray.from_dict(data, self, dims)
        elif isinstance(data, (list, tuple)):
            if value_column is not None:
                raise TypeError("`value_column` is not allowed when providing a list of lists")
            return CodedArray.from_lists(data, self, dims)
        else:
            raise TypeError(f"Unexpected type for data: {type(data)}")
