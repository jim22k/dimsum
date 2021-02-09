import inspect
from typing import Set, Tuple, Optional
from .schema import Schema
import grblas
import numpy as np
import pandas as pd


class Flat:
    """
    Coded data in a flat structure, represented by a GraphBLAS Vector
    """
    def __init__(self, vector, schema: Schema, dims: Set[str]):
        self.vector = vector
        self.schema = schema
        self.dims = dims

    def __repr__(self):
        df = self._repr_helper()
        return repr(df)

    def _repr_html_(self):
        df = self._repr_helper()
        return df._repr_html_()

    def _repr_helper(self):
        ordered_dims = [n for n in self.schema._names if n in self.dims]
        index, vals = self.vector.to_values()
        df = self.schema.decode_many(index, ordered_dims)
        df["* values *"] = vals
        return df

    def _normalize_dims(self, dims) -> Set[str]:
        if type(dims) is set:
            return dims
        if isinstance(dims, str):
            return {dims}
        return set(dims)

    def _compute_missing_dims(self, subset: Set[str]) -> Set[str]:
        if isinstance(subset, str):
            subset = {subset}
        extra_dims = subset - self.dims
        if extra_dims:
            raise ValueError(f"Dimensions {extra_dims} requested, but not available in object")
        return self.dims - subset

    def pivot(self, *, left: Optional[Set[str]] = None, top: Optional[Set[str]] = None) -> "Pivot":
        # Check dimensions
        if left is None and top is None:
            raise TypeError("Must provide either left or top dimensions")
        elif left is not None:
            left = self._normalize_dims(left)
            top = self._compute_missing_dims(left)
        elif top is not None:
            top = self._normalize_dims(top)
            left = self._compute_missing_dims(top)
        else:
            left = self._normalize_dims(left)
            top = self._normalize_dims(top)
            top_verify, left = self._compute_missing_dims(left)
            if top_verify != top:
                raise ValueError("Union of left and top must equal the dimensions in the object")

        # Perform pivot
        index, vals = self.vector.to_values()
        rows = index & self.schema.dims_to_mask(left)
        cols = index & self.schema.dims_to_mask(top)
        matrix = grblas.Matrix.from_values(rows, cols, vals)
        return Pivot(matrix, self.schema, left, top)


class Pivot:
    """
    Coded data in a pivoted structure, represented by a GraphBLAS Matrix
    """
    def __init__(self, matrix, schema: Schema, left: Set[str], top: Set[str]):
        self.matrix = matrix
        self.schema = schema
        self.left = left
        self.top = top

    def __repr__(self):
        df = self._repr_helper()
        return repr(df)

    def _repr_html_(self):
        df = self._repr_helper()
        return df._repr_html_()

    def _repr_helper(self):
        left_dims = [n for n in self.schema._names if n in self.left]
        top_dims = [n for n in self.schema._names if n in self.top]
        rows, cols, vals = self.matrix.to_values()
        row_unique, row_reverse = np.unique(rows, return_inverse=True)
        col_unique, col_reverse = np.unique(cols, return_inverse=True)
        row_index = self.schema.decode_many(row_unique, left_dims).set_index(left_dims).index
        col_index = self.schema.decode_many(col_unique, top_dims).set_index(top_dims).index
        df = pd.DataFrame(index=row_index, columns=col_index)
        df.values[row_reverse, col_reverse] = vals
        df = df.where(pd.notnull(df), "")
        return df

    def flatten(self) -> Flat:
        rows, cols, vals = self.matrix.to_values()
        index = rows | cols
        vector = grblas.Vector.from_values(index, vals)
        combo_dims = self.left | self.top
        return Flat(vector, self.schema, combo_dims)

    op_default = inspect.signature(grblas.Matrix.reduce_rows).parameters['op'].default
    def reduce_rows(self, op=op_default):
        vector = self.matrix.reduce_rows(op).new()
        return Flat(vector, self.schema, self.left)

    op_default = inspect.signature(grblas.Matrix.reduce_columns).parameters['op'].default
    def reduce_columns(self, op=op_default):
        vector = self.matrix.reduce_columns(op).new()
        return Flat(vector, self.schema, self.top)

    del op_default
