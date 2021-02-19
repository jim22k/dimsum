import grblas
import numba
import numpy as np
from .container import Flat, Pivot
from .schema import SchemaMismatchError


class AlreadyAlignedError(Exception):
    pass


class SizeMismatchError(Exception):
    pass


def align(a: Flat, b: Flat) -> (Pivot, Pivot):
    """
    Aligns two Flats, returning two Pivots with matching left and top dimensions.
    If the two input Flats already have matching dimensions, raises AlreadyAlignedError

    :param a: Flat
    :param b: Flat
    :return: (Pivot, Pivot)
    """
    if a.schema is not b.schema:
        raise SchemaMismatchError("Objects have different schemas")

    mismatched_dims = a.dims ^ b.dims
    if not mismatched_dims:
        raise AlreadyAlignedError("Inputs already have fully aligned dimensions")

    # Determine which object is a subset of the other, or if they are fully disjoint
    if a.dims - b.dims == mismatched_dims:
        # b is the subset
        a = a.pivot(top=mismatched_dims)
        b = _align_subset(a, b)
    elif b.dims - a.dims == mismatched_dims:
        # a is the subset
        b = b.pivot(top=mismatched_dims)
        a = _align_subset(b, a)
    else:
        # disjoint
        matched_dims = a.dims & b.dims
        if matched_dims:  # partial disjoint
            a = a.pivot(left=matched_dims)
            b = b.pivot(left=matched_dims)
            a, b = _align_partial_disjoint(a, b)
        else:  # full disjoint
            a, b = _align_fully_disjoint(a, b)
    return a, b


def _align_subset(x: Pivot, sub: Flat) -> Pivot:
    size = sub.vector.size
    if x.matrix.nrows != size:
        raise SizeMismatchError(f"nrows {x.matrix.nrows} != size {size}")
    # Convert sub's values into the diagonal of a matrix
    index, vals = sub.vector.to_values()
    diag = grblas.Matrix.from_values(index, index, vals, nrows=size, ncols=size)
    # Multiply the diagonal matrix by the shape of x (any_first will only take values from diag)
    # This performs a broadcast of sub's values to the corresponding locations in x
    m_broadcast = diag.mxm(x.matrix, grblas.semiring.any_first).new()  # <-- injecting a different semiring here could do the computation for us
    # mxm is an intersection operation, so mismatched codes are missing in m_broadcast
    # Check if sub contained more rows than are present in m_broadcast
    v_x = m_broadcast.reduce_rows(grblas.monoid.any).new()
    if v_x.nvals < sub.vector.nvals:
        # Find mismatched codes and add them in with the NULL_KEY
        v_x(~v_x.S, replace=True)[:] << sub.vector
        m_broadcast[:, 0] << v_x  # Column 0 is the code for all_dims == NULL_KEY
    return Pivot(m_broadcast, x.schema, x.left, x.top)


def _align_fully_disjoint(x: Flat, y: Flat) -> (Pivot, Pivot):
    xm = grblas.Matrix.new(x.vector.dtype, x.vector.size, 1)
    xm[:, 0] << x.vector
    ym = grblas.Matrix.new(y.vector.dtype, y.vector.size, 1)
    ym[:, 0] << y.vector
    # Perform the cross-joins. Values from only a single input are used per calculation
    xr = xm.mxm(ym.T, grblas.semiring.any_first).new()
    yr = xm.mxm(ym.T, grblas.semiring.any_second).new()
    return (
        Pivot(xr, x.schema, left=x.dims, top=y.dims),
        Pivot(yr, x.schema, left=x.dims, top=y.dims)
    )


def _align_partial_disjoint(x: Pivot, y: Pivot) -> (Pivot, Pivot):
    """
    Assumes left dims are matching dims
    """
    assert x.left == y.left
    matched_dims = x.left
    mismatched_dims = x.top | y.top

    # Compute the size and offsets of the cross join computation
    x1 = x.matrix.apply(grblas.unary.one).new().reduce_rows().new()
    y1 = y.matrix.apply(grblas.unary.one).new().reduce_rows().new()
    combo = x1.ewise_add(y1, grblas.monoid.times).new()
    # Mask back into x1 and y1 to contain only what applies to each
    x1(x1.S) << combo
    y1(y1.S) << combo
    x1_size = int(x1.reduce().value)
    y1_size = int(y1.reduce().value)
    # Grab indices for iteration
    x1_idx, _ = x1.to_values()
    y1_idx, _ = y1.to_values()
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
        r2_rows, r2_cols, r2_vals
    )

    return (
        Pivot(grblas.Matrix.from_values(r1_rows, r1_cols, r1_vals), x.schema, matched_dims, mismatched_dims),
        Pivot(grblas.Matrix.from_values(r2_rows, r2_cols, r2_vals), x.schema, matched_dims, mismatched_dims)
    )


@numba.njit
def _align_partial_disjoint_numba(
        combo_idx,
        xs_rows, xs_indptr, xs_col_indices, xs_values,
        ys_rows, ys_indptr, ys_col_indices, ys_values,
        r1_rows, r1_cols, r1_vals,
        r2_rows, r2_cols, r2_vals,
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
                    r1_vals[xoffset] = xs_values[
                        xj]  # Could do the computation here between r1 and r2 rather than keeping them separate
                    r2_vals[yoffset] = ys_values[yj]
                    xoffset += 1
                    yoffset += 1
        elif xrow >= 0:
            for xj in range(xs_indptr[xrow], xs_indptr[xrow + 1]):
                r1_rows[xoffset] = row
                r1_cols[xoffset] = xs_col_indices[xj]
                r1_vals[xoffset] = xs_values[xj]
                xoffset += 1
        elif yrow >= 0:
            for yj in range(ys_indptr[yrow], ys_indptr[yrow + 1]):
                r2_rows[yoffset] = row
                r2_cols[yoffset] = ys_col_indices[yj]
                r2_vals[yoffset] = ys_values[yj]
                yoffset += 1
        else:
            raise Exception("Unhandled row")
