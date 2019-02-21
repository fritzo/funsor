
from __future__ import absolute_import, division, print_function

import itertools
import operator
from collections import OrderedDict

import pytest
import torch
from six.moves import reduce

import funsor
from funsor.testing import check_funsor
from funsor.torch import align_tensors


def test_to_funsor():
    assert isinstance(funsor.to_funsor(torch.tensor(2)), funsor.Tensor)
    assert isinstance(funsor.to_funsor(torch.tensor(2.)), funsor.Tensor)


def test_cons_hash():
    x = torch.randn(3, 3)
    assert funsor.Tensor(x) is funsor.Tensor(x)


def test_indexing():
    data = torch.randn(4, 5)
    inputs = OrderedDict([('i', funsor.ints(4)), ('j', funsor.ints(5))])
    x = funsor.Tensor(data, inputs)
    check_funsor(x, inputs, funsor.reals(), data)

    assert x() is x
    assert x(k=3) is x
    check_funsor(x(1), ['j'], [5], data[1])
    check_funsor(x(1, 2), (), (), data[1, 2])
    check_funsor(x(1, 2, k=3), (), (), data[1, 2])
    check_funsor(x(1, j=2), (), (), data[1, 2])
    check_funsor(x(1, j=2, k=3), (), (), data[1, 2])
    check_funsor(x(1, k=3), ['j'], [5], data[1])
    check_funsor(x(i=1), ('j',), (5,), data[1])
    check_funsor(x(i=1, j=2), (), (), data[1, 2])
    check_funsor(x(i=1, j=2, k=3), (), (), data[1, 2])
    check_funsor(x(i=1, k=3), ('j',), (5,), data[1])
    check_funsor(x(j=2), ('i',), (4,), data[:, 2])
    check_funsor(x(j=2, k=3), ('i',), (4,), data[:, 2])


def test_advanced_indexing():
    I, J, M, N = 4, 5, 2, 3
    x = funsor.Tensor(('i', 'j'), torch.randn(4, 5))
    m = funsor.Tensor(('m',), torch.tensor([2, 3]))
    n = funsor.Tensor(('n',), torch.tensor([0, 1, 1]))

    assert x.shape == (4, 5)

    check_funsor(x(i=m), ('j', 'm'), (J, M))
    check_funsor(x(i=m, j=n), ('m', 'n'), (M, N))
    check_funsor(x(i=m, j=n, k=m), ('m', 'n'), (M, N))
    check_funsor(x(i=m, k=m), ('j', 'm'), (J, M))
    check_funsor(x(i=n), ('j', 'n'), (J, N))
    check_funsor(x(i=n, k=m), ('j', 'n'), (J, N))
    check_funsor(x(j=m), ('i', 'm'), (I, M))
    check_funsor(x(j=m, i=n), ('m', 'n'), (M, N))
    check_funsor(x(j=m, i=n, k=m), ('m', 'n'), (M, N))
    check_funsor(x(j=m, k=m), ('i', 'm'), (I, M))
    check_funsor(x(j=n), ('i', 'n'), (I, N))
    check_funsor(x(j=n, k=m), ('i', 'n'), (I, N))
    check_funsor(x(m), ('j', 'm'), (J, M), x.data[m.data].t())
    check_funsor(x(m, j=n), ('m', 'n'), (M, N))
    check_funsor(x(m, j=n, k=m), ('m', 'n'), (M, N))
    check_funsor(x(m, k=m), ('j', 'm'), (J, M), x.data[m.data].t())
    check_funsor(x(m, n), ('m', 'n'), (M, N))
    check_funsor(x(m, n, k=m), ('m', 'n'), (M, N))
    check_funsor(x(n), ('j', 'n'), (J, N), x.data[n.data].t())
    check_funsor(x(n, k=m), ('j', 'n'), (J, N), x.data[n.data].t())
    check_funsor(x(n, m), ('m', 'n'), (M, N))
    check_funsor(x(n, m, k=m), ('m', 'n'), (M, N))

    check_funsor(x[m], ('j', 'm'), (J, M), x.data[m.data].t())
    check_funsor(x[n], ('j', 'n'), (J, N), x.data[n.data].t())
    check_funsor(x[:, m], ('i', 'm'), (I, M))
    check_funsor(x[:, n], ('i', 'n'), (I, N))
    check_funsor(x[m, n], ('m', 'n'), (M, N))
    check_funsor(x[n, m], ('m', 'n'), (M, N))


def test_ellipsis():
    data = torch.randn(3, 4, 5)
    x = funsor.Tensor(('i', 'j', 'k'), data)
    check_funsor(x, ('i', 'j', 'k'), (3, 4, 5))

    assert x[...] is x
    check_funsor(x[..., 1, 2, 3], (), (), data[1, 2, 3])
    check_funsor(x[..., 2, 3], ('i',), (3,), data[..., 2, 3])
    check_funsor(x[..., 3], ('i', 'j'), (3, 4), data[..., 3])
    check_funsor(x[1, ..., 2, 3], (), (), data[1, 2, 3])
    check_funsor(x[1, ..., 3], ('j',), (4,), data[1, ..., 3])
    check_funsor(x[1, ...], ('j', 'k'), (4, 5), data[1])
    check_funsor(x[1, 2, ..., 3], (), (), data[1, 2, 3])
    check_funsor(x[1, 2, ...], ('k',), (5,), data[1, 2])
    check_funsor(x[1, 2, 3, ...], (), (), data[1, 2, 3])


def unary_eval(symbol, x):
    if symbol in ['~', '-']:
        return eval('{} x'.format(symbol))
    return getattr(x, symbol)()


@pytest.mark.parametrize('shape', [(), (4,), (2, 3)])
@pytest.mark.parametrize('symbol', [
    '~', '-', 'abs', 'sqrt', 'exp', 'log', 'log1p',
])
def test_unary(symbol, shape):
    data = torch.rand(shape) + 0.5
    if symbol == '~':
        data = data.byte()
    expected_data = unary_eval(symbol, data)
    dims = tuple('abc'[:len(shape)])

    x = funsor.Tensor(dims, data)
    actual = unary_eval(symbol, x)
    check_funsor(actual, dims, shape, expected_data)


BINARY_OPS = [
    '+', '-', '*', '/', '**', '==', '!=', '<', '<=', '>', '>=',
    'min', 'max',
]
BOOLEAN_OPS = ['&', '|', '^']


def binary_eval(symbol, x, y):
    if symbol == 'min':
        return funsor.ops.min(x, y)
    if symbol == 'max':
        return funsor.ops.max(x, y)
    return eval('x {} y'.format(symbol))


@pytest.mark.parametrize('dims2', [(), ('a',), ('b', 'a'), ('b', 'c', 'a')])
@pytest.mark.parametrize('dims1', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS + BOOLEAN_OPS)
def test_binary_funsor_funsor(symbol, dims1, dims2):
    dims = tuple(sorted(set(dims1 + dims2)))
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape1 = tuple(sizes[d] for d in dims1)
    shape2 = tuple(sizes[d] for d in dims2)
    data1 = torch.rand(shape1) + 0.5
    data2 = torch.rand(shape2) + 0.5
    if symbol in BOOLEAN_OPS:
        data1 = data1.byte()
        data2 = data2.byte()
    dims, aligned = align_tensors(funsor.Tensor(dims1, data1),
                                  funsor.Tensor(dims2, data2))
    expected_data = binary_eval(symbol, aligned[0], aligned[1])

    x1 = funsor.Tensor(dims1, data1)
    x2 = funsor.Tensor(dims2, data2)
    actual = binary_eval(symbol, x1, x2)
    check_funsor(actual, dims, expected_data.shape, expected_data)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS)
def test_binary_funsor_scalar(symbol, dims, scalar):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data1 = torch.rand(shape) + 0.5
    expected_data = binary_eval(symbol, data1, scalar)

    x1 = funsor.Tensor(dims, data1)
    actual = binary_eval(symbol, x1, scalar)
    check_funsor(actual, dims, shape, expected_data)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS)
def test_binary_scalar_funsor(symbol, dims, scalar):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data1 = torch.rand(shape) + 0.5
    expected_data = binary_eval(symbol, scalar, data1)

    x1 = funsor.Tensor(dims, data1)
    actual = binary_eval(symbol, scalar, x1)
    check_funsor(actual, dims, shape, expected_data)


def finitary_eval(symbol, operands):
    op = getattr(operator, symbol)
    return reduce(op, operands[1:], operands[0])


@pytest.mark.parametrize('dims2', [(), ('a',), ('b', 'a'), ('b', 'c', 'a')])
@pytest.mark.parametrize('dims1', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', ["add", "mul", "and_", "or_"])
def test_finitary_funsor_funsor(symbol, dims1, dims2):
    # copied binary test to start
    dims = tuple(sorted(set(dims1 + dims2)))
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape1 = tuple(sizes[d] for d in dims1)
    shape2 = tuple(sizes[d] for d in dims2)
    data1 = torch.rand(shape1) + 0.5
    data2 = torch.rand(shape2) + 0.5
    if symbol in ("and_", "or_"):  # TODO move to registry
        data1 = data1.byte()
        data2 = data2.byte()
    dims, aligned = align_tensors(funsor.Tensor(dims1, data1),
                                  funsor.Tensor(dims2, data2))
    expected_data = finitary_eval(symbol, [aligned[0], aligned[1]])

    x1 = funsor.Tensor(dims1, data1)
    x2 = funsor.Tensor(dims2, data2)
    actual = finitary_eval(symbol, [x1, x2])
    check_funsor(actual, dims, expected_data.shape, expected_data)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', ["add", "mul"])
def test_finitary_funsor_scalar(symbol, dims, scalar):
    # copied binary test for now
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data1 = torch.rand(shape) + 0.5
    expected_data = finitary_eval(symbol, [data1, scalar])

    x1 = funsor.Tensor(dims, data1)
    actual = finitary_eval(symbol, [x1, scalar])
    check_funsor(actual, dims, shape, expected_data)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', ["add", "mul"])
def test_finitary_scalar_funsor(symbol, dims, scalar):
    # copied binary test for now
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data1 = torch.rand(shape) + 0.5
    expected_data = finitary_eval(symbol, [scalar, data1])

    x1 = funsor.Tensor(dims, data1)
    actual = finitary_eval(symbol, [scalar, x1])
    check_funsor(actual, dims, shape, expected_data)


REDUCE_OPS = ['sum', 'prod', 'logsumexp', 'all', 'any', 'min', 'max']


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('op_name', REDUCE_OPS)
def test_reduce_all(dims, op_name):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data = torch.rand(shape) + 0.5
    if op_name in ['all', 'any']:
        data = data.byte()
    if op_name == 'logsumexp':
        # work around missing torch.Tensor.logsumexp()
        expected_data = data.reshape(-1).logsumexp(0)
    else:
        expected_data = getattr(data, op_name)()

    x = funsor.Tensor(dims, data)
    actual = getattr(x, op_name)()
    check_funsor(actual, (), (), expected_data)


@pytest.mark.parametrize('dims,dims_reduced', [
    (dims, dims_reduced)
    for dims in [('a',), ('a', 'b'), ('b', 'a', 'c')]
    for num_reduced in range(len(dims) + 2)
    for dims_reduced in itertools.combinations(dims + ('z',), num_reduced)
])
@pytest.mark.parametrize('op_name', REDUCE_OPS)
def test_reduce_subset(dims, dims_reduced, op_name):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data = torch.rand(shape) + 0.5
    if op_name in ['all', 'any']:
        data = data.byte()
    x = funsor.Tensor(dims, data)
    actual = getattr(x, op_name)(dims_reduced)

    dims_reduced = set(dims_reduced) & set(dims)
    if not dims_reduced:
        assert actual is x
    else:
        if dims_reduced == set(dims):
            if op_name == 'logsumexp':
                # work around missing torch.Tensor.logsumexp()
                data = data.reshape(-1).logsumexp(0)
            else:
                data = getattr(data, op_name)()
        else:
            for pos in reversed(sorted(map(dims.index, dims_reduced))):
                if op_name in ('min', 'max'):
                    data = getattr(data, op_name)(pos)[0]
                else:
                    data = getattr(data, op_name)(pos)
        dims = tuple(d for d in dims if d not in dims_reduced)
        shape = data.shape
        check_funsor(actual, dims, data.shape, data)


def test_function_mm():

    @funsor.function(('a', 'b'), ('b', 'c'), ('a', 'c'))
    def mm(x, y):
        return torch.matmul(x, y)

    x = funsor.Tensor(('a', 'b'), torch.randn(3, 4))
    y = funsor.Tensor(('b', 'c'), torch.randn(4, 5))
    actual = mm(x, y)
    expected = funsor.Tensor(('a', 'c'), torch.matmul(x.data, y.data))
    check_funsor(actual, expected.dims, expected.shape, expected.data)


def test_lazy_eval_mm():

    @funsor.function(('a', 'b'), ('b', 'c'), ('a', 'c'))
    def mm(x, y):
        return torch.matmul(x, y)

    x_lazy = funsor.Variable('x', 'real')
    y = funsor.Tensor(('b', 'c'), torch.randn(4, 5))
    actual_lazy = mm(x_lazy, y)
    assert isinstance(actual_lazy, funsor.torch.LazyCall)

    x = funsor.Tensor(('a', 'b'), torch.randn(3, 4))
    actual = actual_lazy(x=x)
    expected = funsor.Tensor(('a', 'c'), torch.matmul(x.data, y.data))
    check_funsor(actual, expected.dims, expected.shape, expected.data)
