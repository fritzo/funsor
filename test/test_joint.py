from collections import OrderedDict
from functools import reduce

import pytest
import torch

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.integrate import Integrate
from funsor.interpreter import interpretation
from funsor.joint import Joint
from funsor.terms import Number, Reduce, Variable, eager, moment_matching
from funsor.testing import assert_close, random_gaussian, random_tensor, xfail_if_not_implemented
from funsor.torch import Tensor


def id_from_inputs(inputs):
    if not inputs:
        return '()'
    return ','.join(k + ''.join(map(str, d.shape)) for k, d in inputs.items())


SMOKE_TESTS = [
    ('dx + dy', Joint),
    ('dx + g', Joint),
    ('dy + g', Joint),
    ('g + dx', Joint),
    ('g + dy', Joint),
    ('dx + t', Joint),
    ('dy + t', Joint),
    ('dx - t', Joint),
    ('dy - t', Joint),
    ('t + dx', Joint),
    ('t + dy', Joint),
    ('g + 1', Joint),
    ('g - 1', Joint),
    ('1 + g', Joint),
    ('g + t', Joint),
    ('g - t', Joint),
    ('t + g', Joint),
    ('t - g', Joint),
    ('(dx + dy)(i=i0)', Joint),
    ('(dx + g)(i=i0)', Joint),
    ('(dy + g)(i=i0)', Joint),
    ('(g + dx)(i=i0)', Joint),
    ('(g + dy)(i=i0)', Joint),
    ('(dx + t)(i=i0)', Joint),
    ('(dy + t)(i=i0)', Joint),
    ('(dx - t)(i=i0)', Joint),
    ('(dy - t)(i=i0)', Joint),
    ('(t + dx)(i=i0)', Joint),
    ('(t + dy)(i=i0)', Joint),
    ('(g + 1)(i=i0)', Joint),
    ('(g - 1)(i=i0)', Joint),
    ('(1 + g)(i=i0)', Joint),
    ('(g + t)(i=i0)', Joint),
    ('(g - t)(i=i0)', Joint),
    ('(t + g)(i=i0)', Joint),
    ('(dx + dy)(x=x0)', Joint),
    ('(dx + g)(x=x0)', Tensor),
    ('(dy + g)(x=x0)', Joint),
    ('(g + dx)(x=x0)', Tensor),
    ('(g + dy)(x=x0)', Joint),
    ('(dx + t)(x=x0)', Tensor),
    ('(dy + t)(x=x0)', Joint),
    ('(dx - t)(x=x0)', Tensor),
    ('(dy - t)(x=x0)', Joint),
    ('(t + dx)(x=x0)', Tensor),
    ('(t + dy)(x=x0)', Joint),
    ('(g + 1)(x=x0)', Tensor),
    ('(g - 1)(x=x0)', Tensor),
    ('(1 + g)(x=x0)', Tensor),
    ('(g + t)(x=x0)', Tensor),
    ('(g - t)(x=x0)', Tensor),
    ('(t + g)(x=x0)', Tensor),
    ('(g + g)(x=x0)', Tensor),
    ('(g + dy).reduce(ops.logaddexp, "x")', Joint),
    ('(g + dy).reduce(ops.logaddexp, "y")', Gaussian),
    ('(t + g + dy).reduce(ops.logaddexp, "x")', Joint),
    ('(t + g + dy).reduce(ops.logaddexp, "y")', Joint),
    ('(t + g).reduce(ops.logaddexp, "x")', Tensor),
]


@pytest.mark.parametrize('expr,expected_type', SMOKE_TESTS)
def test_smoke(expr, expected_type):
    dx = Delta('x', Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2))])))
    assert isinstance(dx, Delta)

    dy = Delta('y', Tensor(torch.randn(3, 4), OrderedDict([('j', bint(3))])))
    assert isinstance(dy, Delta)

    t = Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2)), ('j', bint(3))]))
    assert isinstance(t, Tensor)

    g = Gaussian(
        info_vec=torch.tensor([[0.0, 0.1, 0.2],
                               [2.0, 3.0, 4.0]]),
        precision=torch.tensor([[[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]],
                                [[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]]]),
        inputs=OrderedDict([('i', bint(2)), ('x', reals(3))]))
    assert isinstance(g, Gaussian)

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    x0 = Tensor(torch.tensor([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

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
def test_reduce_logaddexp(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    t = random_tensor(int_inputs)
    g = random_gaussian(inputs)
    truth = {name: random_tensor(int_inputs, domain) for name, domain in real_inputs.items()}

    state = 0
    state += g
    state += t
    for name, point in truth.items():
        with xfail_if_not_implemented():
            state += Delta(name, point)
    actual = state.reduce(ops.logaddexp, frozenset(truth))

    expected = t + g(**truth)
    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_reduce_logaddexp_deltas_lazy():
    a = Delta('a', Tensor(torch.randn(3, 2), OrderedDict(i=bint(3))))
    b = Delta('b', Tensor(torch.randn(3), OrderedDict(i=bint(3))))
    x = a + b
    assert isinstance(x, Joint)
    assert set(x.inputs) == {'a', 'b', 'i'}

    y = x.reduce(ops.logaddexp, 'i')
    assert isinstance(y, Reduce)
    assert set(y.inputs) == {'a', 'b'}
    assert_close(x.reduce(ops.logaddexp), y.reduce(ops.logaddexp))


def test_reduce_logaddexp_deltas_discrete_lazy():
    a = Delta('a', Tensor(torch.randn(3, 2), OrderedDict(i=bint(3))))
    b = Delta('b', Tensor(torch.randn(3), OrderedDict(i=bint(3))))
    c = Tensor(torch.randn(3), OrderedDict(i=bint(3)))
    x = a + b + c
    assert isinstance(x, Joint)
    assert set(x.inputs) == {'a', 'b', 'i'}

    y = x.reduce(ops.logaddexp, 'i')
    assert isinstance(y, Reduce)
    assert set(y.inputs) == {'a', 'b'}
    assert_close(x.reduce(ops.logaddexp), y.reduce(ops.logaddexp))


def test_reduce_logaddexp_gaussian_lazy():
    a = random_gaussian(OrderedDict(i=bint(3), a=reals(2)))
    b = random_tensor(OrderedDict(i=bint(3), b=bint(2)))
    x = a + b
    assert isinstance(x, Joint)
    assert set(x.inputs) == {'a', 'b', 'i'}

    y = x.reduce(ops.logaddexp, 'i')
    assert isinstance(y, Reduce)
    assert set(y.inputs) == {'a', 'b'}
    assert_close(x.reduce(ops.logaddexp), y.reduce(ops.logaddexp))


@pytest.mark.parametrize('inputs', [
    OrderedDict([('i', bint(2)), ('x', reals())]),
    OrderedDict([('i', bint(3)), ('x', reals())]),
    OrderedDict([('i', bint(2)), ('x', reals(2))]),
    OrderedDict([('i', bint(2)), ('x', reals()), ('y', reals())]),
    OrderedDict([('i', bint(3)), ('j', bint(4)), ('x', reals(2))]),
    OrderedDict([('j', bint(2)), ('i', bint(3)), ('k', bint(2)), ('x', reals(2))]),
], ids=id_from_inputs)
def test_reduce_add(inputs):
    int_inputs = OrderedDict((k, d) for k, d in inputs.items() if d.dtype != 'real')
    x = random_gaussian(inputs) + random_tensor(int_inputs)
    assert isinstance(x, Joint)
    actual = x.reduce(ops.add, 'i')

    xs = [x(i=i) for i in range(x.inputs['i'].dtype)]
    expected = reduce(ops.add, xs)
    assert_close(actual, expected, atol=1e-3, rtol=1e-4)


def test_reduce_moment_matching_univariate():
    int_inputs = [('i', bint(2))]
    real_inputs = [('x', reals())]
    inputs = OrderedDict(int_inputs + real_inputs)
    int_inputs = OrderedDict(int_inputs)
    real_inputs = OrderedDict(real_inputs)

    p = 0.8
    t = 1.234
    s1, s2, s3 = 2.0, 3.0, 4.0
    loc = torch.tensor([[-s1], [s1]])
    precision = torch.tensor([[[s2 ** -2]], [[s3 ** -2]]])
    info_vec = precision.matmul(loc.unsqueeze(-1)).squeeze(-1)
    discrete = Tensor(torch.tensor([1 - p, p]).log() + t, int_inputs)
    gaussian = Gaussian(info_vec, precision, inputs)
    gaussian -= gaussian.log_normalizer
    joint = discrete + gaussian
    with interpretation(moment_matching):
        actual = joint.reduce(ops.logaddexp, 'i')
    assert_close(actual.reduce(ops.logaddexp), joint.reduce(ops.logaddexp))

    expected_loc = torch.tensor([(2 * p - 1) * s1])
    expected_variance = (4 * p * (1 - p) * s1 ** 2
                         + (1 - p) * s2 ** 2
                         + p * s3 ** 2)
    expected_precision = torch.tensor([[1 / expected_variance]])
    expected_info_vec = expected_precision.matmul(expected_loc.unsqueeze(-1)).squeeze(-1)
    expected_gaussian = Gaussian(expected_info_vec, expected_precision, real_inputs)
    expected_gaussian -= expected_gaussian.log_normalizer
    expected_discrete = Tensor(torch.tensor(t))
    expected = expected_discrete + expected_gaussian
    assert_close(actual, expected, atol=1e-5, rtol=None)


def test_reduce_moment_matching_multivariate():
    int_inputs = [('i', bint(4))]
    real_inputs = [('x', reals(2))]
    inputs = OrderedDict(int_inputs + real_inputs)
    int_inputs = OrderedDict(int_inputs)
    real_inputs = OrderedDict(real_inputs)

    loc = torch.tensor([[-10., -1.],
                        [+10., -1.],
                        [+10., +1.],
                        [-10., +1.]])
    precision = torch.zeros(4, 1, 1) + torch.eye(2, 2)
    discrete = Tensor(torch.zeros(4), int_inputs)
    gaussian = Gaussian(loc, precision, inputs)
    gaussian -= gaussian.log_normalizer
    joint = discrete + gaussian
    with interpretation(moment_matching):
        actual = joint.reduce(ops.logaddexp, 'i')
    assert_close(actual.reduce(ops.logaddexp), joint.reduce(ops.logaddexp))

    expected_loc = torch.zeros(2)
    expected_covariance = torch.tensor([[101., 0.], [0., 2.]])
    expected_precision = expected_covariance.inverse()
    expected_gaussian = Gaussian(expected_loc, expected_precision, real_inputs)
    expected_gaussian -= expected_gaussian.log_normalizer
    expected_discrete = Tensor(torch.tensor(4.).log())
    expected = expected_discrete + expected_gaussian
    assert_close(actual, expected, atol=1e-5, rtol=None)


@pytest.mark.parametrize('interp', [eager, moment_matching],
                         ids=lambda f: f.__name__)
def test_reduce_moment_matching_shape(interp):
    delta = Delta('x', random_tensor(OrderedDict([('h', bint(7))])))
    discrete = random_tensor(OrderedDict(
        [('h', bint(7)), ('i', bint(6)), ('j', bint(5)), ('k', bint(4))]))
    gaussian = random_gaussian(OrderedDict(
        [('k', bint(4)), ('l', bint(3)), ('m', bint(2)), ('y', reals()), ('z', reals(2))]))
    reduced_vars = frozenset(['i', 'k', 'l'])
    real_vars = frozenset(k for k, d in gaussian.inputs.items() if d.dtype == "real")
    joint = delta + discrete + gaussian
    with interpretation(interp):
        actual = joint.reduce(ops.logaddexp, reduced_vars)
    assert set(actual.inputs) == set(joint.inputs) - reduced_vars
    assert_close(actual.reduce(ops.logaddexp, real_vars),
                 joint.reduce(ops.logaddexp, real_vars | reduced_vars))


def test_reduce_moment_matching_moments():
    gaussian = random_gaussian(OrderedDict(
        [('i', bint(2)), ('j', bint(3)), ('x', reals(2))]))
    with interpretation(moment_matching):
        actual = gaussian.reduce(ops.logaddexp, 'j')
    x = Variable('x', reals(2))

    sample_inputs = OrderedDict(s=bint(10000))
    actual_samples = actual.sample(frozenset(['x']), sample_inputs)
    expected_samples = gaussian.sample(frozenset(['j', 'x']), sample_inputs)
    assert_close(actual_samples.reduce(ops.logaddexp, 'x'),
                 expected_samples.reduce(ops.logaddexp, frozenset(['j', 'x'])))
    actual_mean = Integrate(actual_samples, x, frozenset(['x']))
    expected_mean = Integrate(expected_samples, x, frozenset(['j', 'x']))
    assert_close(actual_mean, expected_mean, atol=1e-1, rtol=1e-1)
