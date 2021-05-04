import grblas
from numbers import Number
from .container import Flat, Pivot, CodedArray, ExpandingCodedArray


__all__ = ['align', 'where', 'full_like', 'ones_like', 'zeros_like']


def align(x, y, exact=True):
    """
    Aligns two CodedArrays or their expanding forms

    Exact alignment means the returned objects will have exactly the same shape and number of elements.
    With exact alignment, elements of x or y will be dropped if they do not have a matching code in the
    other object and the other object is not expanding (ex. y.X).
    With exact_alignment turned off, no elements from x or y will be lost in the alignment, but the final
    shapes and number of elements may differ.

    :param x: CodedArray or ExpandingCodedArray
    :param y: CodedArray or ExpandingCodedArray
    :param exact: (default True) whether to force exact alignment
    :return: Tuple[CodedArray, CodedArray]
    """
    from .alignment import align

    xfill = None
    if type(x) is ExpandingCodedArray:
        xfill = x.fill_value
        x = x.coded_array

    yfill = None
    if type(y) is ExpandingCodedArray:
        yfill = y.fill_value
        y = y.coded_array

    ax, ay = align(x.obj, y.obj, afill=xfill, bfill=yfill)

    if exact:
        if xfill is None:
            ay.data(ax.data.S, replace=True) << ay.data
        if yfill is None:
            ax.data(ay.data.S, replace=True) << ax.data

    return CodedArray(ax), CodedArray(ay)


def where(cond, true_vals, false_vals=None):
    """
    Use `true_vals` where `cond` is True, `false_vals` where `cond` is False
    to build a merged dataset. The result will be empty where `cond` is empty.

    `false_vals` are optional and if not provided, this will behave nearly the same as
    `true_vals.filter(cond)`.

    `cond` can also be a boolean scalar, in which case either true_vals or false_vals
    will be returned as-is.

    :param cond: boolean CodedArray or scalar
    :param true_vals: CodedArray or scalar
    :param false_vals: CodedArray or scalar (optional)
    :return: CodedArray or scalar
    """
    if type(cond) is bool:
        return true_vals if cond else false_vals

    # Get values from true_vals where cond==True
    if true_vals is None:
        raise TypeError('true_vals cannot be None')
    elif isinstance(true_vals, Number):
        tmp_cond = cond  # need tmp_cond to avoid mutating cond which is used below in the false block
        if type(cond) is ExpandingCodedArray:
            # Expanding a condition against a scalar makes no sense, so ignore
            tmp_cond = cond.coded_array
        result = tmp_cond.obj.copy(type(true_vals))
        result.data(result.data.V, replace=True) << true_vals
        true_merge = CodedArray(result)
    else:
        true_merge = true_vals.filter(cond)

    # Get values from false_vals where cond==False
    # It is okay to mutate cond in this block because no further usages exist below
    if false_vals is None:
        return true_merge
    elif isinstance(false_vals, Number):
        if type(cond) is ExpandingCodedArray:
            # Expanding a condition against a scalar makes no sense, so ignore
            cond = cond.coded_array
        result = cond.obj.copy(type(false_vals))
        result.data << grblas.op.lnot(result.data)  # invert boolean to be used as a negative mask
        result.data(result.data.V, replace=True) << false_vals
        false_merge = CodedArray(result)
    else:
        # Invert cond (including the fill value) so we can use filter normally
        cond = ~cond
        if type(cond) is ExpandingCodedArray:
            cond.fill_value = not cond.fill_value
        false_merge = false_vals.filter(cond)

    # There should be no overlap, so add using outer-join to combine
    return true_merge.X + false_merge.X


def full_like(obj, fill_value, dtype=None):
    """
    Return a new CodedArray with all values filled in with `fill_value`.

    If dtype is not provided, it inherits the same dtype as `obj`.

    :param obj: CodedArray
    :param fill_value: scalar
    :param dtype: dtype, default is same as obj's dtype
    :return: CodedArray
    """
    # Grab the Flat or Pivot out of the CodedArray
    obj = obj.obj

    if dtype is None:
        dtype = obj.data.dtype

    if type(obj) is Flat:
        vec = obj.vector.dup(dtype=dtype)
        vec(vec.S) << fill_value
        result = Flat(vec, obj.schema, obj.dims)
    else:
        mat = obj.matrix.dup(dtype=dtype)
        mat(mat.S) << fill_value
        result = Pivot(mat, obj.schema, obj.left, obj.top)
    return CodedArray(result)


def zeros_like(obj, dtype=None):
    return full_like(obj, 0, dtype=dtype)


def ones_like(obj, dtype=None):
    return full_like(obj, 1, dtype=dtype)
