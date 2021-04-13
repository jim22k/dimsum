import grblas
import numba
import numpy as np


_semiring_multiplicative_cache = {}


def _find_semiring_multiplicative(binary, isnumpy=False):
    semis = grblas.semiring.numpy if isnumpy else grblas.semiring
    for semi in dir(semis):
        try:
            binstr = semi.split('_')[1]
            if binstr == binary:
                _semiring_multiplicative_cache[binary] = getattr(semis, semi)
                break
        except IndexError:
            pass
    else:
        _semiring_multiplicative_cache[binary] = None


# NOTE: this was planned for use by align_fully_disjoint, but another approach was chosen
#       this utility may not be needed, but keep for now
def semiring_multiplicative_lookup(binary_or_monoid):
    """
    Given a binary or monoid, find any available semiring whose multiplicative piece matches
    """
    # Ensure we have the name of the binary or monoid, not the actual object
    if not isinstance(binary_or_monoid, str):
        binary_or_monoid = binary_or_monoid.name

    # Add to cache if not already present
    if binary_or_monoid not in _semiring_multiplicative_cache:
        isnumpy = binary_or_monoid.split('.')[0] == 'numpy'
        _find_semiring_multiplicative(binary_or_monoid, isnumpy)
    return _semiring_multiplicative_cache[binary_or_monoid]



_grblas_op_lookup = {
    grblas.binary.plus: np.add,
    grblas.binary.times: np.multiply,
    grblas.binary.minus: np.subtract,
    grblas.binary.truediv: np.true_divide,
    grblas.binary.floordiv: np.floor_divide,
    grblas.binary.fmod: np.mod,
}


_op_jitted = {}


def jitted_op(op):
    if not isinstance(op, np.ufunc):
        if op in _grblas_op_lookup:
            op = _grblas_op_lookup[op]
        elif hasattr(op, 'name') and op.name.startswith('numpy.'):
            op = getattr(np, op.name[6:])
        else:
            raise TypeError(f'Cannot jit compile op: {op}')

    if op not in _op_jitted:
        @numba.njit
        def jit_op_wrapper(x, y):
            return op(x, y)
        _op_jitted[op] = jit_op_wrapper

    return _op_jitted[op]


_binary_op_lookup = {
    'add': grblas.binary.plus,
    'subtract': grblas.binary.minus,
    'multiply': grblas.binary.times,
    'true_divide': grblas.binary.truediv,
    'floor_divide': grblas.binary.floordiv,
}
