import grblas
from numbers import Number
from .container import Flat, Pivot, CodedArray, ExpandingCodedArray


__all__ = ['where']


def where(cond, true_vals, false_vals):
    """
    Use `true_vals` where `cond` is True, `false_vals` where `cond` is False
    to build a merged dataset. The result will be empty where `cond` is empty.

    If no false_vals are available and the goal is to only return values from
    `true_vals` where `cond` is True, use `true_vals.filter(cond)` instead.

    `cond` can also be a boolean scalar, in which case either true_vals or false_vals
    will be returned as-is.

    :param cond: boolean CodedArray or scalar
    :param true_vals: CodedArray or scalar
    :param false_vals: CodedArray or scalar
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
        raise TypeError('false_vals cannot be None')
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
