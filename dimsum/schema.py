import math
from typing import Set, Tuple, List, Union, Optional, Iterable
import numpy as np
import pandas as pd
from .container import CodedArray


NULL_KEY = "∅"  # \u2205


class SchemaMismatchError(Exception):
    pass


class Dimension:
    def __init__(self, name, allowed_values, ordered=True):
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

        if NULL_KEY in lookup:
            raise ValueError("NULL_KEY (∅) is not an allowable value")

        # Add in NULL key (Python `None`), always at the zero bit value
        values = (NULL_KEY,) + values
        lookup[NULL_KEY] = 0

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


class Schema:
    def __init__(self, dimensions):
        self._dimensions = tuple(dimensions)
        self.names = tuple(dim.name for dim in dimensions)
        self._lookup = {dim.name: dim for dim in dimensions}

        if len(self._lookup) < len(self._dimensions):
            raise ValueError("duplicate dimension names")

        self._offset = {}
        self.mask = {}

        # Populate offsets and masks
        # Use reverse order to make sorting follow first indicated dimension
        bit_pos = 0
        for dim in reversed(self._dimensions):
            self._offset[dim.name] = bit_pos
            self.mask[dim.name] = (2**dim.num_bits - 1) << bit_pos
            bit_pos += dim.num_bits

        if bit_pos > 60:
            raise OverflowError(f"Number of required bits {bit_pos} exceeds the maximum of 60 allowed by GraphBLAS")
        self.total_bits = bit_pos

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

    def encode_one(self, **values) -> int:
        code = 0
        for name, val in values.items():
            dim = self._lookup[name]
            index = dim.lookup[val]
            offset = self._offset[name]
            code |= index << offset
        return code

    def encode_many(self, values: pd.DataFrame) -> np.ndarray:
        """
        DataFrame headers must match Dimension names exactly
        """
        codes = np.zeros(len(values), dtype=np.uint64)
        for name in values.columns:
            vals = values[name]
            index = self._lookup[name].val2pos[vals].values
            offset = self._offset[name]
            codes |= index << offset
        return codes

    def decode_one(self, code, names: Optional[Tuple[str]] = None) -> dict:
        if names is None:
            names = self.names
        values = {}
        for name in names:
            dim = self._lookup[name]
            mask = self.mask[name]
            offset = self._offset[name]
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
            offset = self._offset[name]
            index = (array & mask) >> offset
            df[name] = dim.pos2val[index].values
        return df

    def dims_to_mask(self, dims: Set[str]) -> int:
        mask = 0
        for dim in dims:
            mask |= self.mask[dim]
        return mask

    def mask_to_dims(self, mask: int) -> Set[str]:
        return {dim for dim in self.mask if mask & self.mask[dim]}

    def encode(self, df_or_series: Union[pd.DataFrame, pd.Series], dims: List[str] = None, value_column: str = None) -> CodedArray:
        """
        Converts a DataFrame or Series to a CodedArray.

        If providing a DataFrame, the dimensions and value column are required.

        If providing a Series, it must have a named Index or MultiIndex. The name or level names will be used
        as the dimension names.

        :param df_or_series: pd.DataFrame or pd.Series
        :param dims: List[str] list of column headers (only needed for DataFrame)
        :param value_column: str column header (only needed for DataFrame)
        :return: CodedArray
        """
        inp = df_or_series
        if isinstance(inp, pd.DataFrame):
            if dims is None or value_column is None:
                raise TypeError("`dims` and `value_column` are required when providing a DataFrame")
            return CodedArray.from_dataframe(inp, self, dims, value_column)
        elif isinstance(inp, pd.Series):
            if dims is not None or value_column is not None:
                raise TypeError("`dims` and `value_column` are not allowed when providing a Series")
            return CodedArray.from_series(inp, self)
        else:
            raise TypeError(f"DataFrame or Series is required, not {type(inp)}")
