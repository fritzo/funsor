from __future__ import absolute_import, division, print_function

import itertools
from collections import OrderedDict

import pytest
import torch

import funsor
from funsor.domains import Domain, bint, reals
from funsor.terms import Variable
from funsor.testing import assert_close, assert_equiv, check_funsor, random_tensor
from funsor.torch import Tensor, align_tensors


@pytest.mark.parametrize('shape', [(), (4,), (3, 2)])
@pytest.mark.parametrize('dtype', [torch.float, torch.long, torch.uint8])
def test_to_funsor(shape, dtype):
    t = torch.randn(shape).type(dtype)
    f = funsor.to_funsor(t)
    assert isinstance(f, Tensor)


def test_cons_hash():
    x = torch.randn(3, 3)
    assert Tensor(x) is Tensor(x)


def test_indexing():
    data = torch.randn(4, 5)
    inputs = OrderedDict([('i', bint(4)),
                          ('j', bint(5))])
    x = Tensor(data, inputs)
    check_funsor(x, inputs, reals(), data)

    assert x() is x
    assert x(k=3) is x
    check_funsor(x(1), {'j': bint(5)}, reals(), data[1])
    check_funsor(x(1, 2), {}, reals(), data[1, 2])
    check_funsor(x(1, 2, k=3), {}, reals(), data[1, 2])
    check_funsor(x(1, j=2), {}, reals(), data[1, 2])
    check_funsor(x(1, j=2, k=3), (), reals(), data[1, 2])
    check_funsor(x(1, k=3), {'j': bint(5)}, reals(), data[1])
    check_funsor(x(i=1), {'j': bint(5)}, reals(), data[1])
    check_funsor(x(i=1, j=2), (), reals(), data[1, 2])
    check_funsor(x(i=1, j=2, k=3), (), reals(), data[1, 2])
    check_funsor(x(i=1, k=3), {'j': bint(5)}, reals(), data[1])
    check_funsor(x(j=2), {'i': bint(4)}, reals(), data[:, 2])
    check_funsor(x(j=2, k=3), {'i': bint(4)}, reals(), data[:, 2])


def test_advanced_indexing_shape():
    I, J, M, N = 4, 5, 2, 3
    x = Tensor(torch.randn(4, 5), OrderedDict([
        ('i', bint(I)),
        ('j', bint(J)),
    ]))
    m = Tensor(torch.tensor([2, 3]), OrderedDict([('m', bint(M))]))
    n = Tensor(torch.tensor([0, 1, 1]), OrderedDict([('n', bint(N))]))
    assert x.data.shape == (4, 5)

    check_funsor(x(i=m), {'j': bint(J), 'm': bint(M)}, reals())
    check_funsor(x(i=m, j=n), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(i=m, j=n, k=m), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(i=m, k=m), {'j': bint(J), 'm': bint(M)}, reals())
    check_funsor(x(i=n), {'j': bint(J), 'n': bint(N)}, reals())
    check_funsor(x(i=n, k=m), {'j': bint(J), 'n': bint(N)}, reals())
    check_funsor(x(j=m), {'i': bint(I), 'm': bint(M)}, reals())
    check_funsor(x(j=m, i=n), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(j=m, i=n, k=m), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(j=m, k=m), {'i': bint(I), 'm': bint(M)}, reals())
    check_funsor(x(j=n), {'i': bint(I), 'n': bint(N)}, reals())
    check_funsor(x(j=n, k=m), {'i': bint(I), 'n': bint(N)}, reals())
    check_funsor(x(m), {'j': bint(J), 'm': bint(M)}, reals())
    check_funsor(x(m, j=n), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(m, j=n, k=m), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(m, k=m), {'j': bint(J), 'm': bint(M)}, reals())
    check_funsor(x(m, n), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(m, n, k=m), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(n), {'j': bint(J), 'n': bint(N)}, reals())
    check_funsor(x(n, k=m), {'j': bint(J), 'n': bint(N)}, reals())
    check_funsor(x(n, m), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(n, m, k=m), {'m': bint(M), 'n': bint(N)}, reals())


@pytest.mark.parametrize('output_shape', [(), (7,), (3, 2)])
def test_advanced_indexing_tensor(output_shape):
    #      u   v
    #     / \ / \
    #    i   j   k
    #     \  |  /
    #      \ | /
    #        x
    x = Tensor(torch.randn((2, 3, 4) + output_shape), OrderedDict([
        ('i', bint(2)),
        ('j', bint(3)),
        ('k', bint(4)),
    ]))
    i = Tensor(random_tensor(2, (5,)), OrderedDict([
        ('u', bint(5)),
    ]))
    j = Tensor(random_tensor(3, (6, 5)), OrderedDict([
        ('v', bint(6)),
        ('u', bint(5)),
    ]))
    k = Tensor(random_tensor(4, (6,)), OrderedDict([
        ('v', bint(6)),
    ]))

    expected_data = torch.empty((5, 6) + output_shape)
    for u in range(5):
        for v in range(6):
            expected_data[u, v] = x.data[i.data[u], j.data[v, u], k.data[v]]
    expected = Tensor(expected_data, OrderedDict([
        ('u', bint(5)),
        ('v', bint(6)),
    ]))

    assert_equiv(expected, x(i, j, k))
    assert_equiv(expected, x(i=i, j=j, k=k))

    assert_equiv(expected, x(i=i, j=j)(k=k))
    assert_equiv(expected, x(j=j, k=k)(i=i))
    assert_equiv(expected, x(k=k, i=i)(j=j))

    assert_equiv(expected, x(i=i)(j=j, k=k))
    assert_equiv(expected, x(j=j)(k=k, i=i))
    assert_equiv(expected, x(k=k)(i=i, j=j))

    assert_equiv(expected, x(i=i)(j=j)(k=k))
    assert_equiv(expected, x(i=i)(k=k)(j=j))
    assert_equiv(expected, x(j=j)(i=i)(k=k))
    assert_equiv(expected, x(j=j)(k=k)(i=i))
    assert_equiv(expected, x(k=k)(i=i)(j=j))
    assert_equiv(expected, x(k=k)(j=j)(i=i))


@pytest.mark.parametrize('output_shape', [(), (7,), (3, 2)])
def test_advanced_indexing_lazy(output_shape):
    x = Tensor(torch.randn((2, 3, 4) + output_shape), OrderedDict([
        ('i', bint(2)),
        ('j', bint(3)),
        ('k', bint(4)),
    ]))
    u = Variable('u', bint(2))
    v = Variable('v', bint(3))
    i = 1 - u
    j = 2 - v
    k = u + v

    expected_data = torch.empty((2, 3) + output_shape)
    i_data = funsor.torch.materialize(i).data
    j_data = funsor.torch.materialize(j).data
    k_data = funsor.torch.materialize(k).data
    for u in range(2):
        for v in range(3):
            expected_data[u, v] = x.data[i_data[u], j_data[v], k_data[u, v]]
    expected = Tensor(expected_data, OrderedDict([
        ('u', bint(2)),
        ('v', bint(3)),
    ]))

    assert_equiv(expected, x(i, j, k))
    assert_equiv(expected, x(i=i, j=j, k=k))

    assert_equiv(expected, x(i=i, j=j)(k=k))
    assert_equiv(expected, x(j=j, k=k)(i=i))
    assert_equiv(expected, x(k=k, i=i)(j=j))

    assert_equiv(expected, x(i=i)(j=j, k=k))
    assert_equiv(expected, x(j=j)(k=k, i=i))
    assert_equiv(expected, x(k=k)(i=i, j=j))

    assert_equiv(expected, x(i=i)(j=j)(k=k))
    assert_equiv(expected, x(i=i)(k=k)(j=j))
    assert_equiv(expected, x(j=j)(i=i)(k=k))
    assert_equiv(expected, x(j=j)(k=k)(i=i))
    assert_equiv(expected, x(k=k)(i=i)(j=j))
    assert_equiv(expected, x(k=k)(j=j)(i=i))


def unary_eval(symbol, x):
    if symbol in ['~', '-']:
        return eval('{} x'.format(symbol))
    return getattr(x, symbol)()


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b')])
@pytest.mark.parametrize('symbol', [
    '~', '-', 'abs', 'sqrt', 'exp', 'log', 'log1p',
])
def test_unary(symbol, dims):
    sizes = {'a': 3, 'b': 4}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    dtype = 'real'
    data = torch.rand(shape) + 0.5
    if symbol == '~':
        data = data.byte()
        dtype = 2
    expected_data = unary_eval(symbol, data)

    x = Tensor(data, inputs, dtype)
    actual = unary_eval(symbol, x)
    check_funsor(actual, inputs, funsor.Domain((), dtype), expected_data)


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
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape1 = tuple(sizes[d] for d in dims1)
    shape2 = tuple(sizes[d] for d in dims2)
    inputs1 = OrderedDict((d, bint(sizes[d])) for d in dims1)
    inputs2 = OrderedDict((d, bint(sizes[d])) for d in dims2)
    data1 = torch.rand(shape1) + 0.5
    data2 = torch.rand(shape2) + 0.5
    dtype = 'real'
    if symbol in BOOLEAN_OPS:
        dtype = 2
        data1 = data1.byte()
        data2 = data2.byte()
    x1 = Tensor(data1, inputs1, dtype)
    x2 = Tensor(data2, inputs2, dtype)
    inputs, aligned = align_tensors(x1, x2)
    expected_data = binary_eval(symbol, aligned[0], aligned[1])

    actual = binary_eval(symbol, x1, x2)
    check_funsor(actual, inputs, Domain((), dtype), expected_data)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS)
def test_binary_funsor_scalar(symbol, dims, scalar):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    data1 = torch.rand(shape) + 0.5
    expected_data = binary_eval(symbol, data1, scalar)

    x1 = Tensor(data1, inputs)
    actual = binary_eval(symbol, x1, scalar)
    check_funsor(actual, inputs, reals(), expected_data)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS)
def test_binary_scalar_funsor(symbol, dims, scalar):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    data1 = torch.rand(shape) + 0.5
    expected_data = binary_eval(symbol, scalar, data1)

    x1 = Tensor(data1, inputs)
    actual = binary_eval(symbol, scalar, x1)
    check_funsor(actual, inputs, reals(), expected_data)


REDUCE_OPS = ['sum', 'prod', 'logsumexp', 'all', 'any', 'min', 'max']


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('op_name', REDUCE_OPS)
def test_reduce_all(dims, op_name):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    data = torch.rand(shape) + 0.5
    if op_name in ['all', 'any']:
        data = data.byte()
    if op_name == 'logsumexp':
        # work around missing torch.Tensor.logsumexp()
        expected_data = data.reshape(-1).logsumexp(0)
    else:
        expected_data = getattr(data, op_name)()

    x = Tensor(data, inputs)
    actual = getattr(x, op_name)()
    check_funsor(actual, {}, reals(), expected_data)


@pytest.mark.parametrize('dims,reduced_vars', [
    (dims, reduced_vars)
    for dims in [('a',), ('a', 'b'), ('b', 'a', 'c')]
    for num_reduced in range(len(dims) + 2)
    for reduced_vars in itertools.combinations(dims, num_reduced)
])
@pytest.mark.parametrize('op_name', REDUCE_OPS)
def test_reduce_subset(dims, reduced_vars, op_name):
    reduced_vars = frozenset(reduced_vars)
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    data = torch.rand(shape) + 0.5
    dtype = 'real'
    if op_name in ['all', 'any']:
        data = data.byte()
        dtype = 2
    x = Tensor(data, inputs, dtype)
    actual = getattr(x, op_name)(reduced_vars)
    expected_inputs = OrderedDict(
        (d, bint(sizes[d])) for d in dims if d not in reduced_vars)

    reduced_vars &= frozenset(dims)
    if not reduced_vars:
        assert actual is x
    else:
        if reduced_vars == frozenset(dims):
            if op_name == 'logsumexp':
                # work around missing torch.Tensor.logsumexp()
                data = data.reshape(-1).logsumexp(0)
            else:
                data = getattr(data, op_name)()
        else:
            for pos in reversed(sorted(map(dims.index, reduced_vars))):
                if op_name in ('min', 'max'):
                    data = getattr(data, op_name)(pos)[0]
                else:
                    data = getattr(data, op_name)(pos)
        check_funsor(actual, expected_inputs, Domain((), dtype))
        assert_close(actual, Tensor(data, expected_inputs, dtype),
                     atol=1e-5, rtol=1e-5)


def test_function_matmul():

    @funsor.function(reals(3, 4), reals(4, 5), reals(3, 5))
    def matmul(x, y):
        return torch.matmul(x, y)

    check_funsor(matmul, {'x': reals(3, 4), 'y': reals(4, 5)}, reals(3, 5))

    x = Tensor(torch.randn(3, 4))
    y = Tensor(torch.randn(4, 5))
    actual = matmul(x, y)
    expected_data = torch.matmul(x.data, y.data)
    check_funsor(actual, {}, reals(3, 5), expected_data)


def test_function_lazy_matmul():

    @funsor.function(reals(3, 4), reals(4, 5), reals(3, 5))
    def matmul(x, y):
        return torch.matmul(x, y)

    x_lazy = funsor.Variable('x', reals(3, 4))
    y = Tensor(torch.randn(4, 5))
    actual_lazy = matmul(x_lazy, y)
    check_funsor(actual_lazy, {'x': reals(3, 4)}, reals(3, 5))
    assert isinstance(actual_lazy, funsor.Function)

    x = Tensor(torch.randn(3, 4))
    actual = actual_lazy(x=x)
    expected_data = torch.matmul(x.data, y.data)
    check_funsor(actual, {}, reals(3, 5), expected_data)


def test_align():
    x = Tensor(torch.randn(2, 3, 4), OrderedDict([
        ('i', bint(2)),
        ('j', bint(3)),
        ('k', bint(4)),
    ]))
    y = x.align(('j', 'k', 'i'))
    assert isinstance(y, Tensor)
    assert tuple(y.inputs) == ('j', 'k', 'i')
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert x(i=i, j=j, k=k) == y(i=i, j=j, k=k)


@pytest.mark.parametrize('equation', [
    'a->a',
    'a,a->a',
    'a,b->',
    'a,b->a',
    'a,b->b',
    'a,b->ab',
    'a,b->ba',
    'ab,ba->',
    'ab,ba->a',
    'ab,ba->b',
    'ab,ba->ab',
    'ab,ba->ba',
    'ab,bc->ac',
])
def test_einsum(equation):
    sizes = dict(a=2, b=3, c=4)
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')
    tensors = [torch.randn(tuple(sizes[d] for d in dims)) for dims in inputs]
    funsors = [Tensor(x) for x in tensors]
    expected = Tensor(torch.einsum(equation, *tensors))
    actual = funsor.einsum(equation, *funsors)
    assert_close(actual, expected)
