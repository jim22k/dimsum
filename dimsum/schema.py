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
        valset = set(values)

        if len(valset) < len(values):
            from collections import Counter

            c = Counter(values)
            dups = {k: v for k, v in c.items() if v > 1}
            raise ValueError(f"Duplicate values: {dups}")

        if len(values) <= 0:
            raise ValueError("allowed_values is empty")

        if None in valset:
            raise ValueError("`None` is not an allowable value")

        if NULL in valset:
            raise ValueError("NULL (∅) is not an allowable value")

        # Add in NULL key (Python `None`), always at the zero bit value
        values = (NULL,) + values

        self.values = pd.Series(values)
        self.enums = pd.Series(np.arange(len(values), dtype=np.uint64), index=values)
        self.num_bits = math.ceil(math.log2(len(values)))

    def __eq__(self, other):
        if type(other) is not Dimension:
            return NotImplemented
        try:
            return (self.values == other.values).all()
        except ValueError:
            return False

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __repr__(self):
        return f"Dimension<{self.name}>"

    def enumerate(self, data):
        """
        Converts values into an array of indices
        """
        if not isinstance(data, pd.Series):
            data = list(data)
        return self.enums[data].values


class Schema(Mapping):
    def __init__(self, dimensions):
        self._dimensions = tuple(dimensions)
        self.names = tuple(dim.name for dim in dimensions)
        self._lookup = {dim.name: dim for dim in dimensions}

        if len(self._lookup) < len(self._dimensions):
            raise ValueError("duplicate dimension names")

        self.offset = {}
        self.bitmask = {}

        # Populate offsets and masks
        # Use reverse order to make sorting follow first indicated dimension
        bit_pos = 0
        for dim in reversed(self._dimensions):
            self.offset[dim.name] = bit_pos
            self.bitmask[dim.name] = (2**dim.num_bits - 1) << bit_pos
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
            r.append(f"  {self.bitmask[dim.name]:0{self.total_bits}b} {dim.name}")
        return '\n'.join(r)

    def dimension_enums(self, dim):
        """
        Returns a new CodedArray containing all values of `dim` and the associated enumerations for each code

        >>> size = Dimension('size', ['small', 'medium', 'large'])
        >>> schema = Schema(['size', ...])
        >>> schema.dimension_enums('size')
            size  * values *
        0      ∅           0
        1  small           1
        2 medium           2
        3  large           3
        """
        if not isinstance(dim, Dimension):
            dim = self[dim]
        offset = self.offset[dim.name]
        enums = dim.enums.values
        codes = enums << offset
        mask = self.bitmask[dim.name]
        vec = grblas.Vector.from_values(codes, enums, dtype='INT64', size=mask + 1)
        return CodedArray(Flat(vec, self, (dim.name,)))

    def encode(self, **values) -> int:
        """
        Pass dim1=value, dim2=value, ... to build the associated code
        """
        code = 0
        for name, val in values.items():
            dim = self._lookup[name]
            if val is None:
                val = NULL
            index = int(dim.enums[val])
            offset = self.offset[name]
            code |= index << offset
        return code

    def decode(self, code, names: Optional[Tuple[str]] = None) -> dict:
        """
        Builds a dict of dim_name: value from a code.

        Passing names will limit to output to only those dimensions.
        """
        if names is None:
            names = self.names
        values = {}
        for name in names:
            dim = self._lookup[name]
            mask = self.bitmask[name]
            offset = self.offset[name]
            index = (code & mask) >> offset
            values[name] = dim[index]
        return values

    def _encode_from_df(self, values: pd.DataFrame) -> np.ndarray:
        """
        DataFrame headers must match Dimension names exactly
        """
        codes = np.zeros(len(values), dtype=np.uint64)
        for name in values.columns:
            vals = values[name]
            try:
                index = self._lookup[name].enums[vals].values
            except ValueError:
                # Most likely a None is present; convert to NULL
                vals = vals.where(pd.notna(vals), NULL)
                index = self._lookup[name].enums[vals].values
            except KeyError:
                dim = self._lookup[name]
                missing = list(dim.enums.index.union(vals).difference(dim.enums.index))
                raise KeyError(f"{dim.name} does not have value(s): {missing}")
            offset = self.offset[name]
            codes |= index << offset
        return codes

    def _decode_to_df(self, array: np.ndarray, names: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if names is None:
            names = self.names
        df = pd.DataFrame()
        for name in names:
            dim = self._lookup[name]
            mask = self.bitmask[name]
            offset = self.offset[name]
            index = (array & mask) >> offset
            df[name] = dim.values[index].values
        return df

    def build_bitmask(self, dims: Set[str] = None) -> int:
        """
        Builds a bitmask from a set of dimensions.
        """
        if dims is None:
            return 2**self.total_bits

        mask = 0
        for dim in dims:
            mask |= self.bitmask[dim]
        return mask

    def deconstruct_bitmask(self, bitmask: int) -> Set[str]:
        """
        Returns a set of dimension names which correspond to the bitmask
        """
        return {dim for dim in self.bitmask if bitmask & self.bitmask[dim]}

    def load(self, data, dims: List[str] = None, value_column: str = None) -> CodedArray:
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
