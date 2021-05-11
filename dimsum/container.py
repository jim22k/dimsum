import inspect
from typing import Set, List, Iterable, Optional
import grblas
import numpy as np
import numpy.lib.mixins
import pandas as pd
from numbers import Number
from itertools import product
from . import oputils


def _normalize_dims(dims) -> Set[str]:
    if type(dims) is set:
        return dims
    if isinstance(dims, str):
        return {dims}
    return set(dims)


def _compute_missing_dims(dims: Set[str], subset: Set[str]) -> Set[str]:
    if isinstance(subset, str):
        subset = {subset}
    extra_dims = subset - dims
    if extra_dims:
        raise ValueError(f"Dimensions {extra_dims} not available in {dims}")
    return dims - subset


_reduce_op_default = inspect.signature(grblas.Matrix.reduce_rows).parameters['op'].default


class Flat:
    """
    Coded data in a flat structure, represented by a GraphBLAS Vector
    """
    def __init__(self, vector, schema: "Schema", dims: Iterable[str]):
        self.vector = vector
        self.schema = schema
        self.dims = set(dims)

    def __repr__(self):
        df = self.to_dataframe()
        return repr(df)

    def _repr_html_(self):
        df = self.to_dataframe()
        return df._repr_html_()

    def __len__(self):
        return self.vector.nvals

    @property
    def data(self):
        return self.vector

    @property
    def dims_list(self):
        """
        Returns the dimensions as a list, ordered according to the schema
        """
        return [n for n in self.schema.names if n in self.dims]

    def copy(self, dtype=None):
        if dtype is not None:
            dtype = grblas.dtypes.lookup_dtype(dtype)
        return Flat(self.vector.dup(dtype=dtype), self.schema, self.dims)

    def pivot(self, *, left: Optional[Set[str]] = None, top: Optional[Set[str]] = None) -> "Pivot":
        # Check dimensions
        if left is None and top is None:
            raise TypeError("Must provide either left or top dimensions")
        elif left is not None:
            left = _normalize_dims(left)
            top = _compute_missing_dims(self.dims, left)
        elif top is not None:
            top = _normalize_dims(top)
            left = _compute_missing_dims(self.dims, top)
        else:
            left = _normalize_dims(left)
            top = _normalize_dims(top)
            top_verify = _compute_missing_dims(self.dims, left)
            if top_verify != top:
                raise ValueError("Union of left and top must equal the total dimensions in the object")

        if not left:
            raise ValueError("left dimensions are empty")
        if not top:
            raise ValueError("top dimensions are empty")

        # Perform pivot
        left_mask = self.schema.build_bitmask(left)
        top_mask = self.schema.build_bitmask(top)
        index, vals = self.vector.to_values()
        rows = index & left_mask
        cols = index & top_mask
        matrix = grblas.Matrix.from_values(rows, cols, vals, nrows=left_mask + 1, ncols=top_mask + 1)
        return Pivot(matrix, self.schema, left, top)

    def reduce(self, op=_reduce_op_default):
        return self.vector.reduce(op).value

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, schema: "Schema", dims: List[str], value_column: str = None) -> "Flat":
        """
        Converts a DataFrame to a Flat by indicating the dimensions and value column

        value_column is optional if len(df.columns) == len(dims) + 1, meaning only one column is left after
        accounting for the dimension columns.

        :param df: pd.DataFrame
        :param schema: Schema
        :param dims: List[str] list of column headers
        :param value_column: str column header (optional)
        :return: Flat
        """
        index = schema._encode_from_df(df[dims])
        if value_column is None:
            remaining_cols = df.columns.difference(dims)
            if len(remaining_cols) > 1:
                raise TypeError("multiple non-dimension columns exist, must provide value_column")
            elif len(remaining_cols) == 0:
                raise TypeError("no value column found")
            value_column = remaining_cols[0]
        vals = df[value_column].values
        dim_mask = schema.build_bitmask(dims)
        vec = grblas.Vector.from_values(index, vals, size=dim_mask + 1)
        return cls(vec, schema, dims)

    @classmethod
    def from_series(cls, s: pd.Series, schema: "Schema") -> "Flat":
        """
        The Series must have a named index or MultiIndex. The name or level names will be used
        as the dimension names.

        :param s: pd.Series
        :param schema: Schema
        :return: Flat
        """
        if isinstance(s.index, pd.MultiIndex):
            dims = s.index.names
        else:
            if not s.index.name:
                err_msg = (
                    "Series index does not have a name. Unable to infer dimension.\n"
                    "When creating the series, ensure the index has a name:\n"
                    "s = pd.Series([1, 2, 3], index=pd.Index(['S', 'M', 'L'], name='size'))"
                )
                raise TypeError(err_msg)
            dims = [s.index.name]
        df = s.to_frame("* value *").reset_index()
        return cls.from_dataframe(df, schema, dims, "* value *")

    @classmethod
    def from_dict(cls, d: dict, schema: "Schema", dims: List[str]) -> "Flat":
        lst = []
        if isinstance(dims, str):
            for k, v in d.items():
                lst.append([k, v])
        else:
            for k, v in d.items():
                lst.append([*k, v])
        return cls.from_lists(lst, schema, dims)

    @classmethod
    def from_lists(cls, lst: List, schema: "Schema", dims: List[str]) -> "Flat":
        if isinstance(dims, str):
            dims = [dims]
        elif not isinstance(dims, list):
            dims = list(dims)
        df = pd.DataFrame(lst, columns=dims + ["* value *"])
        return cls.from_dataframe(df, schema, dims, "* value *")

    def to_series(self) -> pd.Series:
        """
        Converts the Flat into a Series with a named index or a MultiIndex

        :return: pd.Series
        """
        df = self.to_dataframe("* values *")
        dims = self.dims_list
        if len(dims) == 1:
            dims = dims[0]
        return df.set_index(dims)["* values *"]

    def to_dataframe(self, value_column="* values *") -> pd.DataFrame:
        """
        Converts the Flat into a DataFrame

        :param value_column: str name of column containing the values
        :return: pd.DataFrame
        """
        index, vals = self.vector.to_values()
        df = self.schema._decode_to_df(index, self.dims_list)
        df[value_column] = vals
        return df


class Pivot:
    """
    Coded data in a pivoted structure, represented by a GraphBLAS Matrix
    """
    def __init__(self, matrix, schema: "Schema", left: Set[str], top: Set[str]):
        self.matrix = matrix
        self.schema = schema
        self.left = set(left)
        self.top = set(top)

    def __repr__(self):
        with pd.option_context('display.multi_sparse', False):
            df = self.to_dataframe()
            return repr(df)

    def _repr_html_(self):
        with pd.option_context('display.multi_sparse', False):
            df = self.to_dataframe()
            return df._repr_html_()

    def __len__(self):
        return self.matrix.nvals

    @property
    def data(self):
        return self.matrix

    def copy(self, dtype=None):
        if dtype is not None:
            dtype = grblas.dtypes.lookup_dtype(dtype)
        return Pivot(self.matrix.dup(dtype=dtype), self.schema, self.left, self.top)

    def flatten(self) -> Flat:
        rows, cols, vals = self.matrix.to_values()
        index = rows | cols
        combo_dims = self.left | self.top
        dim_mask = self.schema.build_bitmask(combo_dims)
        vector = grblas.Vector.from_values(index, vals, size=dim_mask + 1)
        return Flat(vector, self.schema, combo_dims)

    def pivot(self, *, left: Optional[Set[str]] = None, top: Optional[Set[str]] = None) -> "Pivot":
        combo_dims = self.left | self.top

        # Check dimensions
        if left is None and top is None:
            raise TypeError("Must provide either left or top dimensions")
        elif left is not None:
            left = _normalize_dims(left)
            top = _compute_missing_dims(combo_dims, left)
        elif top is not None:
            top = _normalize_dims(top)
            left = _compute_missing_dims(combo_dims, top)
        else:
            left = _normalize_dims(left)
            top = _normalize_dims(top)
            top_verify = _compute_missing_dims(combo_dims, left)
            if top_verify != top:
                raise ValueError("Union of left and top must equal the total dimensions in the object")

        if not left:
            raise ValueError("left dimensions are empty; maybe you need to flatten() instead")
        if not top:
            raise ValueError("top dimensions are empty; maybe you need to flatten() instead")

        if left == self.left and top == self.top:
            return self

        # Perform pivot
        left_mask = self.schema.build_bitmask(left)
        top_mask = self.schema.build_bitmask(top)
        orig_rows, orig_cols, vals = self.matrix.to_values()
        index = orig_rows | orig_cols
        rows = index & left_mask
        cols = index & top_mask
        matrix = grblas.Matrix.from_values(rows, cols, vals, nrows=left_mask + 1, ncols=top_mask + 1)
        return Pivot(matrix, self.schema, left, top)

    def reduce_rows(self, op=_reduce_op_default):
        vector = self.matrix.reduce_rows(op).new()
        return Flat(vector, self.schema, self.left)

    def reduce_columns(self, op=_reduce_op_default):
        vector = self.matrix.reduce_columns(op).new()
        return Flat(vector, self.schema, self.top)

    def reduce(self, op=_reduce_op_default):
        return self.matrix.reduce_scalar(op).value

    def to_dataframe(self):
        left_dims = [n for n in self.schema.names if n in self.left]
        top_dims = [n for n in self.schema.names if n in self.top]
        rows, cols, vals = self.matrix.to_values()
        row_unique, row_reverse = np.unique(rows, return_inverse=True)
        col_unique, col_reverse = np.unique(cols, return_inverse=True)
        row_index = self.schema._decode_to_df(row_unique, left_dims).set_index(left_dims).index
        col_index = self.schema._decode_to_df(col_unique, top_dims).set_index(top_dims).index
        df = pd.DataFrame(index=row_index, columns=col_index)
        df.values[row_reverse, col_reverse] = vals
        df = df.where(pd.notnull(df), "")
        return df


class CodedArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, flat_or_pivot):
        if type(flat_or_pivot) not in (Flat, Pivot):
            raise TypeError(f"Flat or Pivot required, not {type(flat_or_pivot)}")
        self.obj = flat_or_pivot

    def __repr__(self):
        return self.obj.__repr__()

    def _repr_html_(self):
        return self.obj._repr_html_()

    def __len__(self):
        return len(self.obj)

    @property
    def schema(self):
        return self.obj.schema

    @property
    def dims(self):
        if type(self.obj) is Flat:
            return self.obj.dims
        return self.obj.left | self.obj.top

    @property
    def left(self):
        if type(self.obj) is Flat:
            raise TypeError("Unpivoted CodedArray does not have left dimensions")
        return self.obj.left

    @property
    def top(self):
        if type(self.obj) is Flat:
            raise TypeError("Unpivoted CodedArray does not have top dimensions")
        return self.obj.top

    @property
    def shape(self):
        if type(self.obj) is Flat:
            return (len(self.obj.dims),)
        return len(self.obj.left), len(self.obj.top)

    @property
    def is_pivoted(self):
        return type(self.obj) is Pivot

    @property
    def dtype(self):
        if type(self.obj) is Flat:
            return self.obj.vector.dtype
        return self.obj.matrix.dtype

    def astype(self, dtype):
        new_dtype = grblas.dtypes.lookup_dtype(dtype)
        if new_dtype == self.dtype:
            return self

        if type(self.obj) is Flat:
            obj = Flat(self.obj.vector.dup(dtype=new_dtype), self.obj.schema, self.obj.dims)
        else:
            obj = Pivot(self.obj.matrix.dup(dtype=new_dtype), self.obj.schema, self.obj.left, self.obj.top)
        return CodedArray(obj)

    @property
    def X(self):
        return ExpandingCodedArray(self)

    @classmethod
    def from_series(cls, s: pd.Series, schema: "Schema") -> "CodedArray":
        """
        Converts a Series to a CodedArray.

        The Series must have a named index or MultiIndex. The name or level names will be used
        as the dimension names.

        :param s: pd.Series
        :param schema: Schema
        :return: CodedArray
        """
        return CodedArray(Flat.from_series(s, schema))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, schema: "Schema", dims: List[str], value_column: str = None) -> "CodedArray":
        """
        Converts a DataFrame to a CodedArray by indicating the dimensions and value column

        :param df: pd.DataFrame
        :param schema: Schema
        :param dims: List[str] list of column headers
        :param value_column: str column header
        :return: CodedArray
        """
        return CodedArray(Flat.from_dataframe(df, schema, dims, value_column))

    @classmethod
    def from_dict(cls, d: dict, schema: "Schema", dims: List[str]) -> "CodedArray":
        return CodedArray(Flat.from_dict(d, schema, dims))

    @classmethod
    def from_lists(cls, lst: List, schema: "Schema", dims: List[str]) -> "CodedArray":
        return CodedArray(Flat.from_lists(lst, schema, dims))

    def to_series(self) -> pd.Series:
        """
        Converts the CodedArray into a Series with a named index or a MultiIndex

        :return: pd.Series
        """
        flat = self.obj if type(self.obj) is Flat else self.obj.flatten()
        return flat.to_series()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the CodedArray into a DataFrame

        :param value_column: str name of column containing the values
        :return: pd.DataFrame
        """
        return self.obj.to_dataframe()

    # TODO: add to_dict, to_lists; rearrange where `schema` is in the from_* methods

    def flatten(self) -> "CodedArray":
        if type(self.obj) is Flat:
            return self
        return CodedArray(self.obj.flatten())

    def pivot(self, left: Optional[Set[str]] = None, top: Optional[Set[str]] = None) -> "CodedArray":
        return CodedArray(self.obj.pivot(left=left, top=top))

    def codes(self, *dims):
        """
        Returns the numeric code for every element, masked by dims
        If no dims are provided, the full code is returned
        """
        obj = self.flatten().obj
        if not dims:
            dims = obj.dims
        index, _ = obj.vector.to_values()
        mask = obj.schema.build_bitmask(dims)
        codes = index & mask
        vec = grblas.Vector.from_values(index, codes, dtype='INT64', size=obj.vector.size)
        return CodedArray(Flat(vec, obj.schema, obj.dims))

    def match(self, **kwargs):
        """
        Returns a boolean CodedArray indicating whether each element matches this code.

        If multiple dimensions are included, the set logic is AND.
        If multiple values are included for a single dimension, the set logic is OR.

        For example,
          `x.has_code(size='small', shape=('circle', 'triangle'))`
        will return
          True for small circles and small triangles
          False for everything else
        """
        obj = self.flatten().obj
        schema = obj.schema
        dims = list(kwargs.keys())
        mask = schema.build_bitmask(dims)
        index, _ = obj.vector.to_values()
        codes = index & mask
        # Normalize input values
        normalized = []
        for dim_name in dims:
            dim = schema[dim_name]
            offset = schema.offset[dim_name]
            vals = kwargs[dim_name]
            if not isinstance(vals, (list, tuple, set)):
                vals = [vals]
            else:
                vals = list(vals)
            vals = dim.enums[vals].values << offset
            normalized.append(vals)
        # Test combo pairs
        matches = np.zeros_like(codes, dtype=bool)
        for combo in product(*normalized):
            # Sum up the combo because individual dimension codes have already been bit shifted
            match_code = sum(combo)
            matches |= codes == match_code
        vec = grblas.Vector.from_values(index, matches, dtype=bool, size=obj.vector.size)
        return CodedArray(Flat(vec, obj.schema, obj.dims))

    def shift(self, dim: str, amount=1, *, agg=grblas.monoid.plus):
        """
        Shifts the code associated with dim by amount. Any data which shifts beyond the allowable
        values for dim will be aggregated into the NULL value for dim. If multiple elements are shifted
        into the same code, they will also be aggregated together using `agg`.

        When providing a single integer for the amount, values with dim=NULL will *not* be shifted.
        When providing the amount as a CodedArray, a non-zero shift value is allowed for dim=NULL.

        Example:
        >>> quarters = Dimension('Q', ['2021-Q1', '2021-Q2', '2021-Q3', '2021-Q4'])
        >>> data = schema.load(
            [['sprocket', '2021-Q1', 12.5],
             ['sprocket', '2021-Q2', 17.2],
             ['sprocket', '2021-Q3', 11.7],
             ['sprocket', '2021-Q4', 14.6]],
            dims=['widget', 'Q']
        )
        # Supply chain disruption affects 2021-Q2; must move that back to Q3
        >>> shifter = schema.load({'2021-Q2': 1}, 'Q')
        # Expand shifter to tag everything not specified as having a shift of 0 (i.e. unaffected)
        >>> new_data = data.shift('Q', shifter.X[0])
        >>> new_data
            widget        Q  * value *
        0 sprocket  2021-Q1       12.5
        1 sprocket  2021-Q3       29.9
        2 sprocket  2021-Q4       14.6

        :param dim: dimension name or dimension object
        :param amount: integer or CodedArray of ints
        :param agg: aggregator to use when shifting multiple elements into the same code (default: plus)
        :return: CodedArray
        """
        from . import alignment
        from .schema import Dimension

        amount_fill = None

        if type(amount) is ExpandingCodedArray:
            amount_fill = amount.fill_value
            amount = amount.coded_array

        if type(self.obj) is Pivot:
            top_dims = self.obj.top
            input = self.flatten().obj
        else:
            top_dims = None
            input = self.obj

        if type(dim) is not Dimension:
            dim = input.schema[dim]
        if not dim.ordered:
            raise TypeError(f"Dimension {dim.name} must be ordered")
        dim_name = dim.name
        offset = input.schema.offset[dim_name]
        mask = input.schema.build_bitmask({dim_name})
        max_code = int(dim.enums.max())

        if type(amount) is CodedArray:
            amount_orig = amount
            amount = amount.flatten().obj
            input, amount = alignment.align_flats(input, amount, bfill=amount_fill)
            if type(input) is Pivot:
                input = input.flatten()
                amount = amount.flatten()
            # Ensure we aren't mutating the original inputs
            ivec = input.vector if input.vector is not self.obj.data else input.vector.dup()
            avec = amount.vector if amount.vector is not amount_orig.obj.data else amount.vector.dup()
            # Force exact alignment by performing ewise_mult on each
            ivec << grblas.binary.first(ivec & avec)
            avec << grblas.binary.second(ivec & avec)
            index, vals = ivec.to_values()
            codes = (index & mask) >> offset
            codes = codes.astype(np.int64)
            shift_idx, shift_vals = avec.to_values()
            assert (index == shift_idx).all(), 'different indexes'
            codes += shift_vals.astype(int)
        else:
            index, vals = input.vector.to_values()
            codes = (index & mask) >> offset
            codes = codes.astype(np.int64)
            nulls = codes == 0
            codes[~nulls] += amount

        # Adjust out-of-range codes to NULL
        out_of_range = (codes < 0) | (codes > max_code)
        codes[out_of_range] = 0
        codes = codes.astype(np.uint64)
        # Assign shifted codes back into index
        index &= ~np.uint64(mask)
        index |= (codes << offset)
        # Build output from shifted indices, aggregating duplicates in the process
        vector = grblas.Vector.from_values(index, vals, dup_op=agg)
        output = Flat(vector, input.schema, input.dims)
        if top_dims is not None:
            output = output.pivot(top=top_dims)
        return CodedArray(output)

    def _apply(self, op, *, left=None, right=None, inplace=False):
        if type(self.obj) is Flat:
            vec = self.obj.vector.apply(op, left=left, right=right)
            if inplace:
                self.obj.vector << vec
                vec = self.obj.vector
            else:
                vec = vec.new()
            result = Flat(vec, self.obj.schema, self.obj.dims)
        else:
            mat = self.obj.matrix.apply(op, left=left, right=right)
            if inplace:
                self.obj.matrix << mat
                mat = self.obj.matrix
            else:
                mat = mat.new()
            result = Pivot(mat, self.obj.schema, self.obj.left, self.obj.top)
        return CodedArray(result)

    def filter(self, cond, *, _afill=None):
        """
        Return a new CodedArray containing only those elements where `cond` is True.
        """
        from . import alignment

        bfill = None
        if type(cond) is ExpandingCodedArray:
            bfill = cond.fill_value
            cond = cond.coded_array

        if cond.dtype != 'BOOL':
            raise TypeError(f'filter condition must been boolean, not {cond.obj.dtype}')
        if cond.dims - self.dims:
            raise TypeError(f'Dimensions of filter condition must be a subset. Invalid dims: {cond.dims - self.dims}')

        x, y = alignment.align(self.obj, cond.obj, afill=_afill, bfill=bfill)
        if type(x) is Flat:
            vec = grblas.Vector.new(x.vector.dtype, x.vector.size)
            vec(mask=y.vector.V) << x.vector
            result = Flat(vec, x.schema, x.dims)
        else:
            mat = grblas.Matrix.new(x.matrix.dtype, x.matrix.nrows, x.matrix.ncols)
            mat(mask=y.matrix.V) << x.matrix
            result = Pivot(mat, x.schema, x.left, x.top)
        return CodedArray(result)

    def reduce_rows(self, op=_reduce_op_default):
        """
        Perform a row-wise reduction. Only the left dimensions will remain in the result.
        """
        if type(self.obj) is not Pivot:
            raise TypeError("reduce_rows can only be used for pivoted CodedArrays")
        return CodedArray(self.obj.reduce_rows(op))

    def reduce_columns(self, op=_reduce_op_default):
        """
        Perform a column-wise reduction. Only the top dimensions will remain in the result.
        """
        if type(self.obj) is not Pivot:
            raise TypeError("reduce_columns can only be used for pivoted CodedArrays")
        return CodedArray(self.obj.reduce_columns(op))

    def reduce(self, op=_reduce_op_default):
        """
        Perform a full reduction to a scalar value
        """
        return self.obj.reduce(op)

    def align(self, other):
        """
        Aligns `self` with `other`.

        To expand the shape to match `other`, use `my_obj.X[fill_val].align(other)`.
        To avoid losing any mismatching items, use `my_obj.align(other.X)`

        Returns the aligned version of `self`
        """
        from .methods import align

        ret, _ = align(self, other)
        return ret

    def cross_align(self, other, dims=None):
        """
        Performs cross-alignment with `other` by first removing `dims` from `other`
        and then aligning with the now disjoint dimensions.

        If dims are not provided, all matching dimensions between `self` and `other`
        are used.
        If dims are provided, they must exist in `other`.
        """
        if type(other) is ExpandingCodedArray:
            fill_value = other.fill_value
            other = other.coded_array
        else:
            fill_value = None

        if dims is None:
            dims = self.obj.dims & other.obj.dims
        elif dims - other.obj.dims:
            raise TypeError(f'Provided dims not found in other: {dims - other.obj.dims}')

        if dims == other.obj.dims:
            # Other will become a scalar, so nothing to align with
            return self

        # Remove dims from other
        gutted = CodedArray(other.obj.pivot(top=dims).reduce_rows(grblas.monoid.any))

        if fill_value is not None:
            gutted = ExpandingCodedArray(gutted)

        return self.align(gutted)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return CodedArray._ufunc_handler(ufunc, method, *inputs, **kwargs)

    @staticmethod
    def _ufunc_handler(ufunc, method, *inputs, **kwargs):
        func_name = ufunc.__name__
        if func_name in oputils._binary_op_lookup:
            gbfunc = oputils._binary_op_lookup[func_name]
        else:
            try:
                gbfunc = getattr(grblas.op.numpy, func_name)
            except AttributeError:
                raise NotImplemented
        if method == '__call__':
            nscalar = 0
            narray = 0
            args = []
            fills = []
            for input in inputs:
                if isinstance(input, Number):
                    nscalar += 1
                    args.append(input)
                    fills.append(None)
                elif type(input) is CodedArray:
                    narray += 1
                    args.append(input)
                    fills.append(None)
                elif type(input) is ExpandingCodedArray:
                    narray += 1
                    args.append(input.coded_array)
                    fills.append(input.fill_value)
                else:
                    return NotImplemented
            if kwargs:
                raise NotImplementedError(f'These arguments are not handled currently: {kwargs.keys()}')

            if nscalar + narray != ufunc.nin:
                raise TypeError(f"{func_name} expects {ufunc.nin} inputs, received {nscalar + narray}")
            if ufunc.nin == 1:
                # ex. np.sin(a) or np.sin(a.X)
                assert nscalar == 0, f"unexpected input to {func_name}"
                assert narray == 1, f"unexpected input to {func_name}"
                ret = args[0]._apply(gbfunc)
                if fills[0] is not None:
                    return ExpandingCodedArray(ret, fill_value=fills[0])
                return ret
            elif nscalar > 0:
                # ex. a + 3 or 2 * a.X
                assert nscalar == 1, f"unexpected input to {func_name}"
                assert narray == 1, f"unexpected input to {func_name}"
                if isinstance(args[0], Number):
                    ret = args[1]._apply(gbfunc, left=args[0])
                    fill = fills[1]
                else:
                    ret = args[0]._apply(gbfunc, right=args[1])
                    fill = fills[0]
                if fill is not None:
                    return ExpandingCodedArray(ret, fill_value=fill)
                return ret
            else:
                # ex. a + b or np.arctan2(a, b.X)
                assert narray == 2, f"unexpected input to {func_name}"
                from . import alignment

                arrays = [arg.obj for arg in args]
                result = alignment.align(*arrays, gbfunc, *fills)
                return CodedArray(result)
        elif method == 'reduce':
            # TODO: support reduce for pivoted input
            raise NotImplemented
        elif method == 'accumulate':
            # TODO: possibly support accumulate for pivoted input
            raise NotImplemented
        else:
            return NotImplemented


class ExpandingCodedArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, coded_array: CodedArray, fill_value:Number = 0):
        self.coded_array = coded_array
        if not isinstance(fill_value, Number):
            raise TypeError(f"fill_value must be a scalar number, not {type(fill_value)}")
        self.fill_value = fill_value

    def __getitem__(self, fill_value):
        if isinstance(fill_value, (CodedArray, ExpandingCodedArray)):
            # An ExpandingCodedArray with a fill value of a CodedArray has all the information
            # needed to fully realize itself back into a CodedArray
            from .alignment import _fill_like

            # Set the fill value to be whatever the values are in the other object
            self.fill_value = _fill_like

            return self.align(fill_value)
        elif isinstance(fill_value, Number):
            # Return a new ExpandingCodedArray, this time with the fill value included
            return ExpandingCodedArray(self.coded_array, fill_value=fill_value)
        else:
            raise TypeError(f"Illegal type for fill value: {type(fill_value)}")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return CodedArray._ufunc_handler(ufunc, method, *inputs, **kwargs)

    # This is useful if `cond` contains codes not found in the coded_array
    def filter(self, cond):
        return self.coded_array.filter(cond, _afill=self.fill_value)

    def align(self, other):
        from .methods import align

        ret, _ = align(self, other)
        return ret
