import grblas
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
    # m_broadcast now has the same shape as x, so borrow the dims from it
    # Note: "same shape" is not quite right. The mxm is an intersection operation,
    #       so mismatched codes will be missing in the result
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
    row_degree_index, row_degree_vals = x1.ewise_mult(y1).new().to_values()
    # Compute offsets using cumsum, then roll once to align properly
    row_offsets = np.roll(np.cumsum(row_degree_vals), 1)
    # Last val (total size) is now the first value. Capture it, then zero it out.
    nvals = row_offsets[0]
    row_offsets[0] = 0

    # Extract input arrays in hypercsr format
    xs = x.matrix.ss.export(format='hypercsr', sort=True)
    ys = y.matrix.ss.export(format='hypercsr', sotr=True)

    # Build output data structures
    r_rows = np.zeros((nvals,), dtype=np.uint64)
    r_cols = np.zeros((nvals,), dtype=np.uint64)
    r1_vals = np.zeros((nvals,), dtype=xs['values'].dtype)
    r2_vals = np.zeros((nvals,), dtype=ys['values'].dtype)

    # This part should be njitted, possibly in parallel
    xmiss = 0
    ymiss = 0
    for irow, row in enumerate(row_degree_index):
        # Find xrow and yrow. Start from irow, but might need to move forward for non-overlapping rows
        while True:
            xrow = irow + xmiss
            if xs['rows'][xrow] == row:
                break
            xmiss += 1
        while True:
            yrow = irow + ymiss
            if ys['rows'][yrow] == row:
                break
            ymiss += 1
        # Iterate over x and y indices for this row
        offset = row_offsets[irow]
        for xj in range(xs['indptr'][xrow], xs['indptr'][xrow + 1]):
            for yj in range(ys['indptr'][yrow], ys['indptr'][yrow + 1]):
                r_rows[offset] = row
                r_cols[offset] = xs['col_indices'][xj] + ys['col_indices'][yj]
                r1_vals[offset] = xs['values'][xj]  # Could do the computation here between r1 and r2 rather than keeping them separate
                r2_vals[offset] = ys['values'][yj]
                offset += 1

    return (
        Pivot(grblas.Matrix.from_values(r_rows, r_cols, r1_vals), x.schema, matched_dims, mismatched_dims),
        Pivot(grblas.Matrix.from_values(r_rows, r_cols, r2_vals), x.schema, matched_dims, mismatched_dims)
    )
