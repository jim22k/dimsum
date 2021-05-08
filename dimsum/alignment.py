import grblas
import numba
import numpy as np
from typing import Union, Tuple
from .container import Flat, Pivot
from .schema import SchemaMismatchError
from .oputils import jitted_op


class SizeMismatchError(Exception):
    pass


# Sentinel to indicate the fill values come from the object to which we are aligning
_fill_like = object()


def align(a: Union[Flat, Pivot], b: Union[Flat, Pivot], op=None, afill=None, bfill=None):
    """
    Dispatched to align_pivots if both a and b and Pivots.
    Otherwise dispatches to align_flats, converting any Pivots to Flats using .flatten()
    """
    if a.schema is not b.schema:
        raise SchemaMismatchError("Objects have different schemas")

    if afill is not None and afill is b:
        afill = _fill_like
    if bfill is not None and bfill is a:
        bfill = _fill_like

    a_type = type(a)
    b_type = type(b)
    if a_type is Pivot and b_type is Pivot:
        return align_pivots(a, b, op=op, afill=afill, bfill=bfill)
    elif a_type is Flat and b_type is Flat:
        return align_flats(a, b, op=op, afill=afill, bfill=bfill)
    elif a_type is Pivot:
        return align_flats(a.flatten(), b, op=op, afill=afill, bfill=bfill)
    else:
        return align_flats(a, b.flatten(), op=op, afill=afill, bfill=bfill)


def align_flats(a: Flat, b: Flat, op=None, afill=None, bfill=None):
    """
    Aligns two Flats, returning two Pivots with matching left and top dimensions.

    If a and b are already aligned, returns Flats instead of Pivots.

    If op is provided, returns a single Pivot. Otherwise returns a 2-tuple of Pivots

    afill and bfill are used to determine the kind of alignment
    afill=None, bfill=None -> inner join
    afill!=None, bfill=None -> left join
    afill=None, bfill!=None -> right join
    afill!=None, bfill!=None -> outer join

    :param a: Flat
    :param b: Flat
    :param op: grblas.BinaryOp (default None)
    :param afill: scalar or Flat (default None)
    :param bfill: scalar or Flat (default None)
    :return: Pivot or (Pivot, Pivot)
    """
    if a.schema is not b.schema:
        raise SchemaMismatchError("Objects have different schemas")

    if afill is not None and afill is b:
        afill = _fill_like
    if bfill is not None and bfill is a:
        bfill = _fill_like

    # Determine which object is a subset of the other, or if they are fully disjoint
    mismatched_dims = a.dims ^ b.dims
    if not mismatched_dims:
        result = _already_aligned_flats(a, b, op, afill, bfill)
    elif a.dims - b.dims == mismatched_dims:  # b is the subset
        a = a.pivot(top=mismatched_dims)
        result = _align_subset(a, b, op, afill, bfill)
    elif b.dims - a.dims == mismatched_dims:  # a is the subset
        b = b.pivot(top=mismatched_dims)
        result = _align_subset(b, a, op, bfill, afill, reversed=True)
    else:  # disjoint
        matched_dims = a.dims & b.dims
        if matched_dims:  # partial disjoint
            a = a.pivot(left=matched_dims)
            b = b.pivot(left=matched_dims)
            result = _align_partial_disjoint(a, b, op, afill, bfill)
        else:  # full disjoint
            result = _align_fully_disjoint(a, b, op)
    return result


def align_pivots(a: Pivot, b: Pivot, op=None, afill=None, bfill=None):
    """
    Aligns two Pivots, returning two Pivots with matching left and top dimensions.

    If op is provided, returns a single Pivot. Otherwise returns a 2-tuple of Pivots

    afill and bfill are used to determine the kind of alignment
    afill=None, bfill=None -> inner join
    afill!=None, bfill=None -> left join
    afill=None, bfill!=None -> right join
    afill!=None, bfill!=None -> outer join

    :param a: Pivot
    :param b: Pivot
    :param op: grblas.BinaryOp (default None)
    :param afill: scalar or Flat (default None)
    :param bfill: scalar or Flat (default None)
    :return: Pivot or (Pivot, Pivot)
    """
    if a.schema is not b.schema:
        raise SchemaMismatchError("Objects have different schemas")

    if afill is not None and afill is b:
        afill = _fill_like
    if bfill is not None and bfill is a:
        bfill = _fill_like

    # Determine which object is a subset of the other, or if they are fully disjoint
    a_dims = a.left | a.top
    b_dims = b.left | b.top
    mismatched_dims = a_dims ^ b_dims
    if not mismatched_dims:
        result = _already_aligned_pivots(a, b.pivot(left=a.left), op, afill, bfill)
    elif a_dims - b_dims == mismatched_dims:  # b is the subset
        a = a.pivot(top=mismatched_dims)
        result = _align_subset(a, b.flatten(), op, afill, bfill)
    elif b_dims - a_dims == mismatched_dims:  # a is the subset
        b = b.pivot(top=mismatched_dims)
        result = _align_subset(b, a.flatten(), op, bfill, afill, reversed=True)
    else:  # disjoint
        matched_dims = a_dims & b_dims
        if matched_dims:  # partial disjoint
            a = a.pivot(left=matched_dims)
            b = b.pivot(left=matched_dims)
            result = _align_partial_disjoint(a, b, op, afill, bfill)
        else:  # full disjoint
            result = _align_fully_disjoint(a.flatten(), b.flatten(), op)
    return result


def _already_aligned_flats(a: Flat, b: Flat, op=None, afill=None, bfill=None) -> Union[Flat, Tuple[Flat, Flat]]:
    """
    a.dims must equal b.dims
    """
    assert a.dims == b.dims, f"Mismatching dimensions {a.dims ^ b.dims}"
    # Create a2 and b2 as expanded, filled vectors
    a2, b2 = a.vector, b.vector
    if afill is _fill_like:
        a2 = a2.dup()
        a2(~a2.S) << b2
    elif afill is not None:
        a2 = grblas.Vector.new(a2.dtype, size=a2.size)
        a2(b2.S) << afill
        a2(a.vector.S) << a.vector

    if bfill is _fill_like:
        b2 = b2.dup()
        b2(~b2.S) << a2
    elif bfill is not None:
        b2 = grblas.Vector.new(b2.dtype, size=b2.size)
        b2(a2.S) << bfill
        b2(b.vector.S) << b.vector

    # Handle op
    if op is None:
        return Flat(a2, a.schema, a.dims), Flat(b2, b.schema, b.dims)
    else:
        result = a2.ewise_mult(b2, op=op)
        # Save result over a2, but don't modify the original input
        if afill is None:
            a2 = result.new()
        else:
            a2 << result
        return Flat(a2, a.schema, a.dims)


def _already_aligned_pivots(a: Pivot, b: Pivot, op=None, afill=None, bfill=None) -> Union[Pivot, Tuple[Pivot, Pivot]]:
    """
    a.left must equal b.left
    a.top must equal b.top
    """
    assert a.left == b.left, f"Mismatching left dimensions {a.left ^ b.left}"
    assert a.top == b.top, f"Mismatching top dimensions {a.top ^ b.top}"
    # Create a2 and b2 as expanded, filled matrices
    a2, b2 = a.matrix, b.matrix
    if afill is _fill_like:
        a2 = a2.dup()
        a2(~a2.S) << b2
    elif afill is not None:
        a2 = grblas.Matrix.new(a2.dtype, nrows=a2.nrows, ncols=a2.ncols)
        a2(b2.S) << afill
        a2(a.matrix.S) << a.matrix

    if bfill is _fill_like:
        b2 = b2.dup()
        b2(~b2.S) << a2
    elif bfill is not None:
        b2 = grblas.Matrix.new(b2.dtype, nrows=b2.nrows, ncols=b2.ncols)
        b2(a2.S) << bfill
        b2(b.matrix.S) << b.matrix

    # Handle op
    if op is None:
        return Pivot(a2, a.schema, a.left, a.top), Pivot(b2, b.schema, b.left, b.top)
    else:
        result = a2.ewise_mult(b2, op=op)
        # Save result over a2, but don't modify the original input
        if afill is None:
            a2 = result.new()
        else:
            a2 << result
        return Pivot(a2, a.schema, a.left, a.top)


def _align_subset(x: Pivot, sub: Flat, op=None, afill=None, bfill=None, reversed=False) -> Union[Pivot, Tuple[Pivot, Pivot]]:
    """
    x must have mismatched dims on top
    sub must have dims exactly matching x.left
    """
    x2 = x.matrix
    size = sub.vector.size
    if x2.nrows != size:
        raise SizeMismatchError(f"nrows {x2.nrows} != size {size}")
    # Convert sub's values into the diagonal of a matrix
    index, vals = sub.vector.to_values()
    diag = grblas.Matrix.from_values(index, index, vals, nrows=size, ncols=size)
    # Multiply the diagonal matrix by the shape of x (any_first will only take values from diag)
    # This performs a broadcast of sub's values to the corresponding locations in x
    y2 = diag.mxm(x2, grblas.semiring.any_first).new()
    # mxm is an intersection operation, so mismatched codes are missing in m_broadcast
    if op is None or afill is not None:
        # Check if sub contained more rows than are present in m_broadcast
        v_x = y2.reduce_rows(grblas.monoid.any).new()
        if v_x.nvals < sub.vector.nvals:
            # Find mismatched codes and add them in with the NULL
            v_x(~v_x.S, replace=True)[:] << sub.vector
            # Update y2 with values lost from mxm
            y2[:, 0] << v_x  # Column 0 is the code for all_dims == NULL
            if afill is not None:
                # Fill corresponding elements of x2 if afill
                if afill is not _fill_like:
                    v_x(v_x.S) << afill
                x2 = x2.dup()
                x2(v_x.S)[:, 0] << v_x
    if bfill is _fill_like:
        y2(~y2.S) << x2
    elif bfill is not None:
        ybackup = y2
        y2 = grblas.Matrix.new(y2.dtype, nrows=y2.nrows, ncols=y2.ncols)
        y2(x2.S) << bfill
        y2(ybackup.S) << ybackup
    # Handle op
    if op is None:
        x = Pivot(x2, x.schema, x.left, x.top)
        y = Pivot(y2, x.schema, x.left, x.top)
        return (y, x) if reversed else (x, y)
    else:
        result = y2.ewise_mult(x2, op=op) if reversed else x2.ewise_mult(y2, op=op)
        # Save result over x2, but don't modify the original input
        if afill is None:
            x2 = result.new()
        else:
            x2 << result
        return Pivot(x2, x.schema, x.left, x.top)


def _align_fully_disjoint(x: Flat, y: Flat, op=None) -> Union[Pivot, Tuple[Pivot, Pivot]]:
    """
    x.dims must have no overlap with y.dims
    """
    xm = grblas.Matrix.new(x.vector.dtype, x.vector.size, 1)
    xm[:, 0] << x.vector
    ym = grblas.Matrix.new(y.vector.dtype, y.vector.size, 1)
    ym[:, 0] << y.vector
    # Perform the cross-joins. Values from only a single input are used per calculation
    xr = xm.mxm(ym.T, grblas.semiring.any_first)
    yr = xm.mxm(ym.T, grblas.semiring.any_second)
    if op is None:
        return (
            Pivot(xr.new(), x.schema, left=x.dims, top=y.dims),
            Pivot(yr.new(), x.schema, left=x.dims, top=y.dims)
        )
    else:
        result = xr.new()
        result(accum=op) << yr
        return Pivot(result, x.schema, left=x.dims, top=y.dims)


def _align_partial_disjoint(x: Pivot, y: Pivot, op=None, afill=None, bfill=None) -> Union[Pivot, Tuple[Pivot, Pivot]]:
    """
    x.left must match y.left
    x.top must have no overlap with y.top
    """
    assert x.left == y.left
    matched_dims = x.left
    mismatched_dims = x.top | y.top
    top_mask = x.schema.build_bitmask(mismatched_dims)

    # Compute the size and offsets of the cross join computation
    x1 = x.matrix.apply(grblas.unary.one).new().reduce_rows(grblas.monoid.plus['INT64']).new()
    y1 = y.matrix.apply(grblas.unary.one).new().reduce_rows(grblas.monoid.plus['INT64']).new()
    combo = x1.ewise_add(y1, grblas.monoid.times).new()
    # Mask back into x1 and y1 to contain only what applies to each (unless filling to match)
    if op is None:
        xmask = x1.S if afill is None else None
        ymask = y1.S if bfill is None else None
        x1(mask=xmask) << combo
        y1(mask=ymask) << combo
        x1_size = int(x1.reduce().value)
        y1_size = int(y1.reduce().value)
    else:  # op is provided, will only have a single return object
        # Trim x1 to final size, then compute result_size
        if afill is None and bfill is None:  # intersecting values only
            x1 << x1.ewise_mult(y1, grblas.monoid.times)
        elif afill is None:  # size same as x
            x1(x1.S, replace=True) << combo
        elif bfill is None:  # size same as y
            x1(y1.S, replace=True) << combo
        else:
            x1 = combo
        result_size = int(x1.reduce().value)

    combo_idx, combo_offset = combo.to_values()

    # Extract input arrays in hypercsr format
    xs = x.matrix.ss.export(format='hypercsr', sort=True)
    xs_rows = xs['rows']
    xs_indptr = xs['indptr']
    xs_col_indices = xs['col_indices']
    xs_values = xs['values']
    ys = y.matrix.ss.export(format='hypercsr', sort=True)
    ys_rows = ys['rows']
    ys_indptr = ys['indptr']
    ys_col_indices = ys['col_indices']
    ys_values = ys['values']

    if op is None:
        # Build output data structures
        r1_rows = np.zeros((x1_size,), dtype=np.uint64)
        r1_cols = np.zeros((x1_size,), dtype=np.uint64)
        r1_vals = np.zeros((x1_size,), dtype=xs['values'].dtype)
        r2_rows = np.zeros((y1_size,), dtype=np.uint64)
        r2_cols = np.zeros((y1_size,), dtype=np.uint64)
        r2_vals = np.zeros((y1_size,), dtype=ys['values'].dtype)

        _align_partial_disjoint_numba(
            combo_idx,
            xs_rows, xs_indptr, xs_col_indices, xs_values,
            ys_rows, ys_indptr, ys_col_indices, ys_values,
            r1_rows, r1_cols, r1_vals,
            r2_rows, r2_cols, r2_vals,
            afill is not None, bfill is not None,
            afill if afill is not None and afill is not _fill_like else None,
            bfill if bfill is not None and bfill is not _fill_like else None
        )

        return (
            Pivot(grblas.Matrix.from_values(r1_rows, r1_cols, r1_vals, nrows=x.matrix.nrows, ncols=top_mask + 1),
                  x.schema, matched_dims, mismatched_dims),
            Pivot(grblas.Matrix.from_values(r2_rows, r2_cols, r2_vals, nrows=x.matrix.nrows, ncols=top_mask + 1),
                  x.schema, matched_dims, mismatched_dims)
        )

    else:
        unified_input_dtype = grblas.dtypes.unify(
            grblas.dtypes.lookup_dtype(xs['values'].dtype),
            grblas.dtypes.lookup_dtype(ys['values'].dtype)
        )
        output_dtype_str = op.types[grblas.dtypes.lookup_dtype(unified_input_dtype).name]

        op = jitted_op(op)

        # Build output data structures
        r_rows = np.zeros((result_size,), dtype=np.uint64)
        r_cols = np.zeros((result_size,), dtype=np.uint64)
        r_vals = np.zeros((result_size,), dtype=grblas.dtypes.lookup_dtype(output_dtype_str).np_type)

        _align_partial_disjoint_numba_op(
            op, combo_idx,
            xs_rows, xs_indptr, xs_col_indices, xs_values,
            ys_rows, ys_indptr, ys_col_indices, ys_values,
            r_rows, r_cols, r_vals,
            afill is not None, bfill is not None,
            afill if afill is not None and afill is not _fill_like else None,
            bfill if bfill is not None and bfill is not _fill_like else None
        )

        return (
            Pivot(grblas.Matrix.from_values(r_rows, r_cols, r_vals), x.schema, matched_dims, mismatched_dims)
        )


@numba.njit
def _align_partial_disjoint_numba(
        combo_idx,
        xs_rows, xs_indptr, xs_col_indices, xs_values,
        ys_rows, ys_indptr, ys_col_indices, ys_values,
        r1_rows, r1_cols, r1_vals,
        r2_rows, r2_cols, r2_vals,
        fill_x, fill_y,  # boolean
        x_fillval, y_fillval,  # scalar or None
):
    # xi/yi are the current index of xs/ys, not necessarily in sync with combo_idx due to mismatched codes
    xi = 0
    yi = 0
    xoffset = 0
    yoffset = 0
    for row in combo_idx:
        # Find xrow and yrow, if available
        xrow, yrow = -1, -1
        if xi < len(xs_rows) and xs_rows[xi] == row:
            xrow = xi
            xi += 1
        if yi < len(ys_rows) and ys_rows[yi] == row:
            yrow = yi
            yi += 1
        # Iterate over x and y indices for this row
        if xrow >= 0 and yrow >= 0:
            for xj in range(xs_indptr[xrow], xs_indptr[xrow + 1]):
                for yj in range(ys_indptr[yrow], ys_indptr[yrow + 1]):
                    r1_rows[xoffset] = row
                    r2_rows[yoffset] = row
                    col_idx = xs_col_indices[xj] + ys_col_indices[yj]
                    r1_cols[xoffset] = col_idx
                    r2_cols[yoffset] = col_idx
                    r1_vals[xoffset] = xs_values[xj]
                    r2_vals[yoffset] = ys_values[yj]
                    xoffset += 1
                    yoffset += 1
        elif xrow >= 0:
            for xj in range(xs_indptr[xrow], xs_indptr[xrow + 1]):
                r1_rows[xoffset] = row
                r1_cols[xoffset] = xs_col_indices[xj]
                r1_vals[xoffset] = xs_values[xj]
                xoffset += 1
                if fill_y:
                    r2_rows[yoffset] = row
                    r2_cols[yoffset] = xs_col_indices[xj]
                    if y_fillval is None:
                        r2_vals[yoffset] = xs_values[xj]
                    else:
                        r2_vals[yoffset] = y_fillval
                    yoffset += 1
        elif yrow >= 0:
            for yj in range(ys_indptr[yrow], ys_indptr[yrow + 1]):
                r2_rows[yoffset] = row
                r2_cols[yoffset] = ys_col_indices[yj]
                r2_vals[yoffset] = ys_values[yj]
                yoffset += 1
                if fill_x:
                    r1_rows[xoffset] = row
                    r1_cols[xoffset] = ys_col_indices[yj]
                    if x_fillval is None:
                        r1_vals[xoffset] = ys_values[yj]
                    else:
                        r1_vals[xoffset] = x_fillval
                    xoffset += 1
        else:
            raise Exception("Unhandled row")


@numba.njit
def _align_partial_disjoint_numba_op(
        op, combo_idx,
        xs_rows, xs_indptr, xs_col_indices, xs_values,
        ys_rows, ys_indptr, ys_col_indices, ys_values,
        r_rows, r_cols, r_vals,
        fill_x, fill_y,  # boolean
        x_fillval, y_fillval,  # scalar or None
):
    # xi/yi are the current index of xs/ys, not necessarily in sync with combo_idx due to mismatched codes
    xi = 0
    yi = 0
    offset = 0
    for row in combo_idx:
        # Find xrow and yrow, if available
        xrow, yrow = -1, -1
        if xi < len(xs_rows) and xs_rows[xi] == row:
            xrow = xi
            xi += 1
        if yi < len(ys_rows) and ys_rows[yi] == row:
            yrow = yi
            yi += 1
        # Iterate over x and y indices for this row
        if xrow >= 0 and yrow >= 0:
            for xj in range(xs_indptr[xrow], xs_indptr[xrow + 1]):
                for yj in range(ys_indptr[yrow], ys_indptr[yrow + 1]):
                    r_rows[offset] = row
                    col_idx = xs_col_indices[xj] + ys_col_indices[yj]
                    r_cols[offset] = col_idx
                    # Could do the computation here between r1 and r2 rather than keeping them separate
                    r_vals[offset] = op(xs_values[xj], ys_values[yj])
                    offset += 1
        elif xrow >= 0:
            if not fill_y:
                continue
            for xj in range(xs_indptr[xrow], xs_indptr[xrow + 1]):
                r_rows[offset] = row
                r_cols[offset] = xs_col_indices[xj]
                other_val = xs_values[xj] if y_fillval is None else y_fillval
                r_vals[offset] = op(xs_values[xj], other_val)
                offset += 1
        elif yrow >= 0:
            if not fill_x:
                continue
            for yj in range(ys_indptr[yrow], ys_indptr[yrow + 1]):
                r_rows[offset] = row
                r_cols[offset] = ys_col_indices[yj]
                other_val = ys_values[yj] if x_fillval is None else x_fillval
                r_vals[offset] = op(ys_values[yj], other_val)
                offset += 1
        else:
            raise Exception("Unhandled row")
