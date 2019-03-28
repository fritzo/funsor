from __future__ import absolute_import, division, print_function

import itertools
from collections import OrderedDict

import pytest
import torch
from six.moves import reduce

import funsor.ops as ops
from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.integrate import Integrate
from funsor.joint import Joint
from funsor.montecarlo import monte_carlo_interpretation
from funsor.terms import Number, Variable
from funsor.testing import assert_close, id_from_inputs, random_gaussian, random_tensor, xfail_if_not_implemented
from funsor.torch import Tensor


@pytest.mark.parametrize('expr,expected_type', [
    ('-g1', Gaussian),
    ('g1 + 1', Joint),
    ('g1 - 1', Joint),
    ('1 + g1', Joint),
    ('g1 + shift', Joint),
    ('g1 + shift', Joint),
    ('shift + g1', Joint),
    ('shift - g1', Joint),
    ('g1 + g1', Joint),
    ('(g1 + g2 + g2) - g2', Joint),
    ('g1(i=i0)', Gaussian),
    ('g2(i=i0)', Gaussian),
    ('g1(i=i0) + g2(i=i0)', Joint),
    ('g1(i=i0) + g2', Joint),
    ('g1(x=x0)', Tensor),
    ('g2(y=y0)', Tensor),
    ('(g1 + g2)(i=i0)', Joint),
    ('(g1 + g2)(x=x0, y=y0)', Tensor),
    ('(g2 + g1)(x=x0, y=y0)', Tensor),
    ('g1.reduce(ops.logaddexp, "x")', Tensor),
    ('(g1 + g2).reduce(ops.logaddexp, "x")', Joint),
    ('(g1 + g2).reduce(ops.logaddexp, "y")', Joint),
    ('(g1 + g2).reduce(ops.logaddexp, frozenset(["x", "y"]))', Tensor),
])
def test_smoke(expr, expected_type):
    g1 = Gaussian(
        loc=torch.tensor([[0.0, 0.1, 0.2],
                          [2.0, 3.0, 4.0]]),
        precision=torch.tensor([[[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]],
                                [[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]]]),
        inputs=OrderedDict([('i', bint(2)), ('x', reals(3))]))
    assert isinstance(g1, Gaussian)

    g2 = Gaussian(
        loc=torch.tensor([[0.0, 0.1],
                          [2.0, 3.0]]),
        precision=torch.tensor([[[1.0, 0.2],
                                 [0.2, 1.0]],
                                [[1.0, 0.2],
                                 [0.2, 1.0]]]),
        inputs=OrderedDict([('i', bint(2)), ('y', reals(2))]))
    assert isinstance(g2, Gaussian)

    shift = Tensor(torch.tensor([-1., 1.]), OrderedDict([('i', bint(2))]))
    assert isinstance(shift, Tensor)

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    x0 = Tensor(torch.tensor([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

    y0 = Tensor(torch.tensor([[0.2, 0.3],
                              [0.8, 0.9]]),
                inputs=OrderedDict([('i', bint(2))]))
    assert isinstance(y0, Tensor)

    result = eval(expr)
    assert isinstance(result, expected_type)


@pytest.mark.parametrize('int_inputs', [
    {},
    {'i': bint(2)},
    {'i': bint(2), 'j': bint(3)},
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    {'x': reals()},
    {'x': reals(4)},
    {'x': reals(2, 3)},
    {'x': reals(), 'y': reals()},
    {'x': reals(2), 'y': reals(3)},
    {'x': reals(4), 'y': reals(2, 3), 'z': reals()},
], ids=id_from_inputs)
def test_eager_subs(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)

    for order in itertools.permutations(inputs):
        ground_values = {}
        dependent_values = {}
        for i, name in enumerate(order):
            upstream = OrderedDict([(k, inputs[k]) for k in order[:i] if k in int_inputs])
            value = random_tensor(upstream, inputs[name])
            ground_values[name] = value(**ground_values)
            dependent_values[name] = value

        expected = g(**ground_values)
        actual = g
        for k in reversed(order):
            with xfail_if_not_implemented():
                actual = actual(**{k: dependent_values[k]})
        assert_close(actual, expected, atol=1e-4)


def test_eager_subs_variable():
    inputs = OrderedDict([('i', bint(2)), ('x', reals()), ('y', reals(2))])
    g1 = random_gaussian(inputs)

    g2 = g1(x='z')
    assert set(g2.inputs) == {'i', 'y', 'z'}

    g2 = g1(x='y', y='x')
    assert set(g2.inputs) == {'i', 'x', 'y'}
    assert g2.inputs['x'] == reals(2)


@pytest.mark.parametrize('int_inputs', [
    {},
    {'i': bint(2)},
    {'i': bint(2), 'j': bint(3)},
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    {'x': reals()},
    {'x': reals(4)},
    {'x': reals(2, 3)},
    {'x': reals(), 'y': reals()},
    {'x': reals(2), 'y': reals(3)},
    {'x': reals(4), 'y': reals(2, 3), 'z': reals()},
], ids=id_from_inputs)
def test_add_gaussian_number(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)
    n = Number(1.234)
    values = {name: random_tensor(int_inputs, domain)
              for name, domain in real_inputs.items()}

    assert_close((g + n)(**values), g(**values) + n, atol=1e-4)
    assert_close((n + g)(**values), n + g(**values), atol=1e-4)
    assert_close((g - n)(**values), g(**values) - n, atol=1e-4)


@pytest.mark.parametrize('int_inputs', [
    {},
    {'i': bint(2)},
    {'i': bint(2), 'j': bint(3)},
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    {'x': reals()},
    {'x': reals(4)},
    {'x': reals(2, 3)},
    {'x': reals(), 'y': reals()},
    {'x': reals(2), 'y': reals(3)},
    {'x': reals(4), 'y': reals(2, 3), 'z': reals()},
], ids=id_from_inputs)
def test_add_gaussian_tensor(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)
    t = random_tensor(int_inputs, reals())
    values = {name: random_tensor(int_inputs, domain)
              for name, domain in real_inputs.items()}

    assert_close((g + t)(**values), g(**values) + t, atol=1e-4)
    assert_close((t + g)(**values), t + g(**values), atol=1e-4)
    assert_close((g - t)(**values), g(**values) - t, atol=1e-4)


@pytest.mark.parametrize('lhs_inputs', [
    {'x': reals()},
    {'y': reals(4)},
    {'z': reals(2, 3)},
    {'x': reals(), 'y': reals(4)},
    {'y': reals(4), 'z': reals(2, 3)},
], ids=id_from_inputs)
@pytest.mark.parametrize('rhs_inputs', [
    {'x': reals()},
    {'y': reals(4)},
    {'z': reals(2, 3)},
    {'x': reals(), 'y': reals(4)},
    {'y': reals(4), 'z': reals(2, 3)},
], ids=id_from_inputs)
def test_add_gaussian_gaussian(lhs_inputs, rhs_inputs):
    lhs_inputs = OrderedDict(sorted(lhs_inputs.items()))
    rhs_inputs = OrderedDict(sorted(rhs_inputs.items()))
    inputs = lhs_inputs.copy()
    inputs.update(rhs_inputs)
    int_inputs = OrderedDict((k, d) for k, d in inputs.items() if d.dtype != 'real')
    real_inputs = OrderedDict((k, d) for k, d in inputs.items() if d.dtype == 'real')

    g1 = random_gaussian(lhs_inputs)
    g2 = random_gaussian(rhs_inputs)
    values = {name: random_tensor(int_inputs, domain)
              for name, domain in real_inputs.items()}

    assert_close((g1 + g2)(**values), g1(**values) + g2(**values), atol=1e-4, rtol=None)


@pytest.mark.parametrize('inputs', [
    OrderedDict([('i', bint(2)), ('x', reals())]),
    OrderedDict([('i', bint(3)), ('x', reals())]),
    OrderedDict([('i', bint(2)), ('x', reals(2))]),
    OrderedDict([('i', bint(2)), ('x', reals()), ('y', reals())]),
    OrderedDict([('i', bint(3)), ('j', bint(4)), ('x', reals(2))]),
], ids=id_from_inputs)
def test_reduce_add(inputs):
    g = random_gaussian(inputs)
    actual = g.reduce(ops.add, 'i')

    gs = [g(i=i) for i in range(g.inputs['i'].dtype)]
    expected = reduce(ops.add, gs)
    assert_close(actual, expected)


@pytest.mark.parametrize('int_inputs', [
    {},
    {'i': bint(2)},
    {'i': bint(2), 'j': bint(3)},
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    {'x': reals(), 'y': reals()},
    {'x': reals(2), 'y': reals(3)},
    {'x': reals(4), 'y': reals(2, 3), 'z': reals()},
    {'w': reals(5), 'x': reals(4), 'y': reals(2, 3), 'z': reals()},
], ids=id_from_inputs)
def test_reduce_logsumexp(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)
    g_xy = g.reduce(ops.logaddexp, frozenset(['x', 'y']))
    assert_close(g_xy, g.reduce(ops.logaddexp, 'x').reduce(ops.logaddexp, 'y'), atol=1e-3, rtol=None)
    assert_close(g_xy, g.reduce(ops.logaddexp, 'y').reduce(ops.logaddexp, 'x'), atol=1e-3, rtol=None)


@pytest.mark.parametrize('int_inputs', [
    {},
    {'i': bint(2)},
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    {'x': reals()},
    {'x': reals(4)},
    {'x': reals(2, 3)},
], ids=id_from_inputs)
def test_integrate_variable(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    log_measure = random_gaussian(inputs)
    integrand = reduce(ops.add, [Variable(k, d) for k, d in real_inputs.items()])
    reduced_vars = frozenset(real_inputs)

    with monte_carlo_interpretation(particle=bint(100000)):
        approx = Integrate(log_measure, integrand, reduced_vars)
        assert isinstance(approx, Tensor)

    exact = Integrate(log_measure, integrand, reduced_vars)
    assert isinstance(exact, Tensor)
    assert_close(approx, exact, atol=0.1, rtol=0.1)


@pytest.mark.parametrize('int_inputs', [
    {},
    {'i': bint(2)},
    {'i': bint(2), 'j': bint(3)},
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    {'x': reals()},
    {'x': reals(2)},
    {'x': reals(), 'y': reals()},
    {'x': reals(2), 'y': reals(3)},
    {'x': reals(4), 'y': reals(2, 3)},
], ids=id_from_inputs)
def test_integrate_gaussian(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    log_measure = random_gaussian(inputs)
    integrand = random_gaussian(inputs)
    reduced_vars = frozenset(real_inputs)

    with monte_carlo_interpretation(particle=bint(10000)):
        approx = Integrate(log_measure, integrand, reduced_vars)
        assert isinstance(approx, Tensor)

    exact = Integrate(log_measure, integrand, reduced_vars)
    assert isinstance(exact, Tensor)
    assert_close(approx, exact, atol=0.1, rtol=0.1)
