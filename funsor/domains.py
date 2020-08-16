# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import operator
from functools import reduce
from weakref import WeakValueDictionary

import funsor
import funsor.ops as ops
from funsor.util import broadcast_shape, get_tracing_state, quote


class Domain(type):
    def __repr__(cls):
        return cls.__name__

    def __str__(cls):
        return cls.__name__


class RealMeta(Domain):
    def __getitem__(cls, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        # in some JAX versions, shape can be np.int64 type
        if get_tracing_state() or funsor.get_backend() == "jax":
            shape = tuple(map(int, shape))
        result = Real._type_cache.get(shape, None)
        if result is None:
            assert cls is Real
            assert all(isinstance(size, int) and size >= 0 for size in shape)
            name = "Real[{}]".format(",".join(map(str, shape)))
            result = RealMeta(name, (Real,), {"shape": shape})
            Real._type_cache[shape] = result
        return result

    @property
    def num_elements(cls):
        return reduce(operator.mul, cls.shape, 1)

    # SHIM
    @property
    def size(self):
        raise AssertionError("reals() has no .size")


class Real(type, metaclass=RealMeta):
    """
    Type of a real-valued array with known shape::

        Real[()] = Real  # scalar
        Real[8]          # vector of length 8
        Real[3,3]        # 3x3 matrix
    """
    _type_cache = WeakValueDictionary()
    shape = ()

    def __reduce__(self):
        return RealMeta, (self.shape,)

    # SHIM
    dtype = "real"


Real._type_cache[()] = Real  # Real[()] is Real.


class BintMeta(Domain):
    def __getitem__(cls, size):
        # in some JAX versions, shape can be np.int64 type
        if get_tracing_state() or funsor.get_backend() == "jax":
            size = int(size)
        result = Bint._type_cache.get(size, None)
        if result is None:
            assert cls is Bint
            assert isinstance(size, int) and size >= 0
            name = "Bint[{}]".format(size)
            result = BintMeta(name, (Bint,), {"size": size})
            Bint._type_cache[size] = result
        return result

    num_elements = 1

    def __iter__(cls):
        from funsor.terms import Number
        return (Number(i, cls.dtype) for i in range(cls.size))

    # SHIM
    @property
    def dtype(cls):
        return cls.size


class Bint(type, metaclass=BintMeta):
    """
    Factory for bounded integer types::

        Bint[5]  # integers ranging in {0,1,2,3,4}
    """
    _type_cache = WeakValueDictionary()

    def __reduce__(self):
        size = getattr(self, "size", None)
        return "Bint" if size is None else (BintMeta, (size,))

    # SHIM
    shape = ()


# SHIM
def reals(*args):
    return Real[args]


# SHIM
def bint(size):
    return Bint[size]


# SHIM
def make_domain(shape, dtype):
    return Real[shape] if dtype == "real" else Bint[dtype]


@quote.register(Domain)
def _(arg, indent, out):
    out.append((indent, repr(arg)))


def find_domain(op, *domains):
    r"""
    Finds the :class:`Domain` resulting when applying ``op`` to ``domains``.
    :param callable op: An operation.
    :param Domain \*domains: One or more input domains.
    """
    assert callable(op), op
    assert all(isinstance(arg, Domain) for arg in domains)
    if len(domains) == 1:
        dtype = domains[0].dtype
        shape = domains[0].shape
        if op is ops.log or op is ops.exp:
            dtype = 'real'
        elif isinstance(op, ops.ReshapeOp):
            shape = op.shape
        elif isinstance(op, ops.AssociativeOp):
            shape = ()
        return reals(*shape) if dtype == "real" else bint(dtype)

    lhs, rhs = domains
    if isinstance(op, ops.GetitemOp):
        dtype = lhs.dtype
        shape = lhs.shape[:op.offset] + lhs.shape[1 + op.offset:]
        return reals(*shape) if dtype == "real" else bint(dtype)
    elif op == ops.matmul:
        assert lhs.shape and rhs.shape
        if len(rhs.shape) == 1:
            assert lhs.shape[-1] == rhs.shape[-1]
            shape = lhs.shape[:-1]
        elif len(lhs.shape) == 1:
            assert lhs.shape[-1] == rhs.shape[-2]
            shape = rhs.shape[:-2] + rhs.shape[-1:]
        else:
            assert lhs.shape[-1] == rhs.shape[-2]
            shape = broadcast_shape(lhs.shape[:-1], rhs.shape[:-2] + (1,)) + rhs.shape[-1:]
        return reals(*shape)

    if lhs.dtype == 'real' or rhs.dtype == 'real':
        dtype = 'real'
    elif op in (ops.add, ops.mul, ops.pow, ops.max, ops.min):
        dtype = op(lhs.dtype - 1, rhs.dtype - 1) + 1
    elif op in (ops.and_, ops.or_, ops.xor):
        dtype = 2
    elif lhs.dtype == rhs.dtype:
        dtype = lhs.dtype
    else:
        raise NotImplementedError('TODO')

    if lhs.shape == rhs.shape:
        shape = lhs.shape
    else:
        shape = broadcast_shape(lhs.shape, rhs.shape)
    return reals(*shape) if dtype == "real" else bint(dtype)


__all__ = [
    'Domain',
    'find_domain',
    'bint',
    'reals',
]
