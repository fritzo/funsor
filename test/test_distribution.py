# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
from collections import OrderedDict
from importlib import import_module

import numpy as np
import pytest

import funsor
import funsor.ops as ops
from funsor.cnf import Contraction, GaussianMixture
from funsor.delta import Delta
from funsor.distribution import BACKEND_TO_DISTRIBUTIONS_BACKEND
from funsor.domains import bint, reals
from funsor.interpreter import interpretation, reinterpret
from funsor.integrate import Integrate
from funsor.tensor import Einsum, Tensor, align_tensors, numeric_array
from funsor.terms import Independent, Variable, eager, lazy
from funsor.testing import assert_close, check_funsor, rand, randint, randn, random_mvn, random_tensor, xfail_param
from funsor.util import get_backend

pytestmark = pytest.mark.skipif(get_backend() == "numpy",
                                reason="numpy does not have distributions backend")
if get_backend() != "numpy":
    dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    backend_dist = dist.dist

if get_backend() == "torch":
    from funsor.pyro.convert import dist_to_funsor


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('eager', [False, True])
def test_beta_density(batch_shape, eager):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals(), reals())
    def beta(concentration1, concentration0, value):
        return backend_dist.Beta(concentration1, concentration0).log_prob(value)

    check_funsor(beta, {'concentration1': reals(), 'concentration0': reals(), 'value': reals()}, reals())

    concentration1 = Tensor(ops.exp(randn(batch_shape)), inputs)
    concentration0 = Tensor(ops.exp(randn(batch_shape)), inputs)
    value = Tensor(rand(batch_shape), inputs)
    expected = beta(concentration1, concentration0, value)
    check_funsor(expected, inputs, reals())

    d = Variable('value', reals())
    actual = dist.Beta(concentration1, concentration0, value) if eager else \
        dist.Beta(concentration1, concentration0, d)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('syntax', ['eager', 'lazy', 'generic'])
def test_bernoulli_probs_density(batch_shape, syntax):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals())
    def bernoulli(probs, value):
        return backend_dist.Bernoulli(probs).log_prob(value)

    check_funsor(bernoulli, {'probs': reals(), 'value': reals()}, reals())

    probs = Tensor(rand(batch_shape), inputs)
    value = Tensor(rand(batch_shape).round(), inputs)
    expected = bernoulli(probs, value)
    check_funsor(expected, inputs, reals())

    d = Variable('value', reals())
    if syntax == 'eager':
        actual = dist.BernoulliProbs(probs, value)
    elif syntax == 'lazy':
        actual = dist.BernoulliProbs(probs, d)(value=value)
    elif syntax == 'generic':
        actual = dist.Bernoulli(probs=probs)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('syntax', ['eager', 'lazy', 'generic'])
def test_bernoulli_logits_density(batch_shape, syntax):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals())
    def bernoulli(logits, value):
        return backend_dist.Bernoulli(logits=logits).log_prob(value)

    check_funsor(bernoulli, {'logits': reals(), 'value': reals()}, reals())

    logits = Tensor(rand(batch_shape), inputs)
    value = Tensor(ops.astype(rand(batch_shape) >= 0.5, 'float'), inputs)
    expected = bernoulli(logits, value)
    check_funsor(expected, inputs, reals())

    d = Variable('value', reals())
    if syntax == 'eager':
        actual = dist.BernoulliLogits(logits, value)
    elif syntax == 'lazy':
        actual = dist.BernoulliLogits(logits, d)(value=value)
    elif syntax == 'generic':
        actual = dist.Bernoulli(logits=logits)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('eager', [False, True])
def test_binomial_density(batch_shape, eager):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))
    max_count = 10

    @funsor.function(reals(), reals(), reals(), reals())
    def binomial(total_count, probs, value):
        return backend_dist.Binomial(total_count, probs).log_prob(value)

    check_funsor(binomial, {'total_count': reals(), 'probs': reals(), 'value': reals()}, reals())

    value_data = ops.astype(random_tensor(inputs, bint(max_count)).data, 'float')
    total_count_data = value_data + ops.astype(random_tensor(inputs, bint(max_count)).data, 'float')
    value = Tensor(value_data, inputs)
    total_count = Tensor(total_count_data, inputs)
    probs = Tensor(rand(batch_shape), inputs)
    expected = binomial(total_count, probs, value)
    check_funsor(expected, inputs, reals())

    m = Variable('value', reals())
    actual = dist.Binomial(total_count, probs, value) if eager else \
        dist.Binomial(total_count, probs, m)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected, rtol=1e-5)


def test_categorical_defaults():
    probs = Variable('probs', reals(3))
    value = Variable('value', bint(3))
    assert dist.Categorical(probs) is dist.Categorical(probs, value)


@pytest.mark.parametrize('size', [4])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_categorical_density(size, batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.of_shape(reals(size), bint(size))
    def categorical(probs, value):
        return probs[value].log()

    check_funsor(categorical, {'probs': reals(size), 'value': bint(size)}, reals())

    probs_data = ops.exp(randn(batch_shape + (size,)))
    probs_data /= probs_data.sum(-1)[..., None]
    probs = Tensor(probs_data, inputs)
    value = random_tensor(inputs, bint(size))
    expected = categorical(probs, value)
    check_funsor(expected, inputs, reals())

    actual = dist.Categorical(probs, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


def test_delta_defaults():
    v = Variable('v', reals())
    log_density = Variable('log_density', reals())
    backend_dist_module = BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()]
    assert isinstance(dist.Delta(v, log_density), import_module(backend_dist_module).Delta)
    value = Variable('value', reals())
    assert dist.Delta(v, log_density, 'value') is dist.Delta(v, log_density, value)


@pytest.mark.parametrize('event_shape', [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_delta_density(batch_shape, event_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(*event_shape), reals(), reals(*event_shape), reals())
    def delta(v, log_density, value):
        eq = (v == value)
        for _ in range(len(event_shape)):
            eq = ops.all(eq, -1)
        return ops.log(ops.astype(eq, 'float32')) + log_density

    check_funsor(delta, {'v': reals(*event_shape),
                         'log_density': reals(),
                         'value': reals(*event_shape)}, reals())

    v = Tensor(randn(batch_shape + event_shape), inputs)
    log_density = Tensor(ops.exp(randn(batch_shape)), inputs)
    for value in [v, Tensor(randn(batch_shape + event_shape), inputs)]:
        expected = delta(v, log_density, value)
        check_funsor(expected, inputs, reals())

        actual = dist.Delta(v, log_density, value)
        check_funsor(actual, inputs, reals())
        assert_close(actual, expected)


def test_delta_delta():
    v = Variable('v', reals(2))
    point = Tensor(randn(2))
    log_density = Tensor(numeric_array(0.5))
    d = dist.Delta(point, log_density, v)
    assert d is Delta('v', point, log_density)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(1,), (4,), (5,)], ids=str)
def test_dirichlet_density(batch_shape, event_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(*event_shape), reals(*event_shape), reals())
    def dirichlet(concentration, value):
        return backend_dist.Dirichlet(concentration).log_prob(value)

    check_funsor(dirichlet, {'concentration': reals(*event_shape), 'value': reals(*event_shape)}, reals())

    concentration = Tensor(ops.exp(randn(batch_shape + event_shape)), inputs)
    value_data = rand(batch_shape + event_shape)
    value_data = value_data / value_data.sum(-1)[..., None]
    value = Tensor(value_data, inputs)
    expected = dirichlet(concentration, value)
    check_funsor(expected, inputs, reals())
    actual = dist.Dirichlet(concentration, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(1,), (4,), (5,)], ids=str)
@pytest.mark.xfail(raises=AttributeError, reason="DirichletMultinomial is not implemented yet in NumPyro")
def test_dirichlet_multinomial_density(batch_shape, event_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))
    max_count = 10

    @funsor.function(reals(*event_shape), reals(), reals(*event_shape), reals())
    def dirichlet_multinomial(concentration, total_count, value):
        return backend_dist.DirichletMultinomial(concentration, total_count).log_prob(value)

    check_funsor(dirichlet_multinomial, {'concentration': reals(*event_shape),
                                         'total_count': reals(),
                                         'value': reals(*event_shape)},
                 reals())

    concentration = Tensor(ops.exp(randn(batch_shape + event_shape)), inputs)
    value_data = ops.astype(randint(0, max_count, size=batch_shape + event_shape), 'float32')
    total_count_data = value_data.sum(-1) + ops.astype(randint(0, max_count, size=batch_shape), 'float32')
    value = Tensor(value_data, inputs)
    total_count = Tensor(total_count_data, inputs)
    expected = dirichlet_multinomial(concentration, total_count, value)
    check_funsor(expected, inputs, reals())
    actual = dist.DirichletMultinomial(concentration, total_count, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_lognormal_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals(), reals())
    def log_normal(loc, scale, value):
        return backend_dist.LogNormal(loc, scale).log_prob(value)

    check_funsor(log_normal, {'loc': reals(), 'scale': reals(), 'value': reals()}, reals())

    loc = Tensor(randn(batch_shape), inputs)
    scale = Tensor(ops.exp(randn(batch_shape)), inputs)
    value = Tensor(ops.exp(randn(batch_shape)), inputs)
    expected = log_normal(loc, scale, value)
    check_funsor(expected, inputs, reals())

    actual = dist.LogNormal(loc, scale, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(1,), (4,), (5,)], ids=str)
def test_multinomial_density(batch_shape, event_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))
    max_count = 10

    @funsor.function(reals(), reals(*event_shape), reals(*event_shape), reals())
    def multinomial(total_count, probs, value):
        if get_backend() == "torch":
            total_count = total_count.max().item()
        return backend_dist.Multinomial(total_count, probs).log_prob(value)

    check_funsor(multinomial, {'total_count': reals(), 'probs': reals(*event_shape), 'value': reals(*event_shape)},
                 reals())

    probs_data = rand(batch_shape + event_shape)
    probs_data = probs_data / probs_data.sum(-1)[..., None]
    probs = Tensor(probs_data, inputs)
    value_data = ops.astype(randint(0, max_count, size=batch_shape + event_shape), 'float')
    total_count_data = value_data.sum(-1)
    value = Tensor(value_data, inputs)
    total_count = Tensor(total_count_data, inputs)
    expected = multinomial(total_count, probs, value)
    check_funsor(expected, inputs, reals())
    actual = dist.Multinomial(total_count, probs, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


def test_normal_defaults():
    loc = Variable('loc', reals())
    scale = Variable('scale', reals())
    value = Variable('value', reals())
    assert dist.Normal(loc, scale) is dist.Normal(loc, scale, value)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_normal_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.of_shape(reals(), reals(), reals())
    def normal(loc, scale, value):
        return -((value - loc) ** 2) / (2 * scale ** 2) - scale.log() - math.log(math.sqrt(2 * math.pi))

    check_funsor(normal, {'loc': reals(), 'scale': reals(), 'value': reals()}, reals())

    loc = Tensor(randn(batch_shape), inputs)
    scale = Tensor(ops.exp(randn(batch_shape)), inputs)
    value = Tensor(randn(batch_shape), inputs)
    expected = normal(loc, scale, value)
    check_funsor(expected, inputs, reals())

    actual = dist.Normal(loc, scale, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_normal_gaussian_1(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = Tensor(randn(batch_shape), inputs)
    scale = Tensor(ops.exp(randn(batch_shape)), inputs)
    value = Tensor(randn(batch_shape), inputs)

    expected = dist.Normal(loc, scale, value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(loc, scale, 'value')
    assert isinstance(g, Contraction)
    actual = g(value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-4)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_normal_gaussian_2(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = Tensor(randn(batch_shape), inputs)
    scale = Tensor(ops.exp(randn(batch_shape)), inputs)
    value = Tensor(randn(batch_shape), inputs)

    expected = dist.Normal(loc, scale, value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(Variable('value', reals()), scale, loc)
    assert isinstance(g, Contraction)
    actual = g(value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-4)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_normal_gaussian_3(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = Tensor(randn(batch_shape), inputs)
    scale = Tensor(ops.exp(randn(batch_shape)), inputs)
    value = Tensor(randn(batch_shape), inputs)

    expected = dist.Normal(loc, scale, value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(Variable('loc', reals()), scale, 'value')
    assert isinstance(g, Contraction)
    actual = g(loc=loc, value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-4)


NORMAL_AFFINE_TESTS = [
    'dist.Normal(x+2, scale, y+2)',
    'dist.Normal(y, scale, x)',
    'dist.Normal(x - y, scale, 0)',
    'dist.Normal(0, scale, y - x)',
    'dist.Normal(2 * x - y, scale, x)',
    'dist.Normal(0, 1, (x - y) / scale) - scale.log()',
    'dist.Normal(2 * y, 2 * scale, 2 * x) + math.log(2)',
]


@pytest.mark.parametrize('expr', NORMAL_AFFINE_TESTS)
def test_normal_affine(expr):

    scale = Tensor(numeric_array(0.3), OrderedDict())
    x = Variable('x', reals())
    y = Variable('y', reals())

    expected = dist.Normal(x, scale, y)
    actual = eval(expr)

    assert isinstance(actual, Contraction)
    assert dict(actual.inputs) == dict(expected.inputs), (actual.inputs, expected.inputs)

    for ta, te in zip(actual.terms, expected.terms):
        assert_close(ta.align(tuple(te.inputs)), te)


def test_normal_independent():
    loc = random_tensor(OrderedDict(), reals(2))
    scale = ops.exp(random_tensor(OrderedDict(), reals(2)))
    fn = dist.Normal(loc['i'], scale['i'], value='z_i')
    assert fn.inputs['z_i'] == reals()
    d = Independent(fn, 'z', 'i', 'z_i')
    assert d.inputs['z'] == reals(2)
    rng_key = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)
    sample = d.sample(frozenset(['z']), rng_key=rng_key)
    assert isinstance(sample, Contraction)
    assert sample.inputs['z'] == reals(2)


def test_mvn_defaults():
    loc = Variable('loc', reals(3))
    scale_tril = Variable('scale', reals(3, 3))
    value = Variable('value', reals(3))
    assert dist.MultivariateNormal(loc, scale_tril) is dist.MultivariateNormal(loc, scale_tril, value)


def _random_scale_tril(shape):
    if get_backend() == "torch":
        data = randn(shape)
        return backend_dist.transforms.transform_to(backend_dist.constraints.lower_cholesky)(data)
    else:
        data = randn(shape[:-2] + (shape[-1] * (shape[-1] + 1) // 2,))
        return backend_dist.biject_to(backend_dist.constraints.lower_cholesky)(data)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_mvn_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(3), reals(3, 3), reals(3), reals())
    def mvn(loc, scale_tril, value):
        return backend_dist.MultivariateNormal(loc, scale_tril=scale_tril).log_prob(value)

    check_funsor(mvn, {'loc': reals(3), 'scale_tril': reals(3, 3), 'value': reals(3)}, reals())

    loc = Tensor(randn(batch_shape + (3,)), inputs)
    scale_tril = Tensor(_random_scale_tril(batch_shape + (3, 3)), inputs)
    value = Tensor(randn(batch_shape + (3,)), inputs)
    expected = mvn(loc, scale_tril, value)
    check_funsor(expected, inputs, reals())

    actual = dist.MultivariateNormal(loc, scale_tril, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_mvn_gaussian(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = Tensor(randn(batch_shape + (3,)), inputs)
    scale_tril = Tensor(_random_scale_tril(batch_shape + (3, 3)), inputs)
    value = Tensor(randn(batch_shape + (3,)), inputs)

    expected = dist.MultivariateNormal(loc, scale_tril, value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.MultivariateNormal(loc, scale_tril, 'value')
    assert isinstance(g, Contraction)
    actual = g(value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-3, rtol=1e-4)


def _check_mvn_affine(d1, data):
    backend_module = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    assert isinstance(d1, backend_module.MultivariateNormal)
    d2 = reinterpret(d1)
    assert issubclass(type(d2), GaussianMixture)
    actual = d2(**data)
    expected = d1(**data)
    assert_close(actual, expected)


@pytest.mark.xfail(get_backend() == 'jax', reason='dist_to_funsor for jax backend is not available yet')
def test_mvn_affine_one_var():
    x = Variable('x', reals(2))
    data = dict(x=Tensor(randn(2)))
    with interpretation(lazy):
        d = dist_to_funsor(random_mvn((), 2))
        d = d(value=2 * x + 1)
    _check_mvn_affine(d, data)


@pytest.mark.xfail(get_backend() == 'jax', reason='dist_to_funsor for jax backend is not available yet')
def test_mvn_affine_two_vars():
    x = Variable('x', reals(2))
    y = Variable('y', reals(2))
    data = dict(x=Tensor(randn(2)), y=Tensor(randn(2)))
    with interpretation(lazy):
        d = dist_to_funsor(random_mvn((), 2))
        d = d(value=x - y)
    _check_mvn_affine(d, data)


def test_mvn_affine_matmul():
    x = Variable('x', reals(2))
    y = Variable('y', reals(3))
    m = Tensor(randn(2, 3))
    data = dict(x=Tensor(randn(2)), y=Tensor(randn(3)))
    with interpretation(lazy):
        d = random_mvn((), 3)
        d = dist.MultivariateNormal(loc=y, scale_tril=d.scale_tril, value=x @ m)
    _check_mvn_affine(d, data)


@pytest.mark.xfail(get_backend() == 'jax', reason='dist_to_funsor for jax backend is not available yet')
def test_mvn_affine_matmul_sub():
    x = Variable('x', reals(2))
    y = Variable('y', reals(3))
    m = Tensor(randn(2, 3))
    data = dict(x=Tensor(randn(2)), y=Tensor(randn(3)))
    with interpretation(lazy):
        d = dist_to_funsor(random_mvn((), 3))
        d = d(value=x @ m - y)
    _check_mvn_affine(d, data)


@pytest.mark.xfail(get_backend() == 'jax', reason='dist_to_funsor for jax backend is not available yet')
def test_mvn_affine_einsum():
    c = Tensor(randn(3, 2, 2))
    x = Variable('x', reals(2, 2))
    y = Variable('y', reals())
    data = dict(x=Tensor(randn(2, 2)), y=Tensor(randn(())))
    with interpretation(lazy):
        d = dist_to_funsor(random_mvn((), 3))
        d = d(value=Einsum("abc,bc->a", c, x) + y)
    _check_mvn_affine(d, data)


@pytest.mark.xfail(get_backend() == 'jax', reason='dist_to_funsor for jax backend is not available yet')
def test_mvn_affine_getitem():
    x = Variable('x', reals(2, 2))
    data = dict(x=Tensor(randn(2, 2)))
    with interpretation(lazy):
        d = dist_to_funsor(random_mvn((), 2))
        d = d(value=x[0] - x[1])
    _check_mvn_affine(d, data)


@pytest.mark.xfail(get_backend() == 'jax', reason='dist_to_funsor for jax backend is not available yet')
def test_mvn_affine_reshape():
    x = Variable('x', reals(2, 2))
    y = Variable('y', reals(4))
    data = dict(x=Tensor(randn(2, 2)), y=Tensor(randn(4)))
    with interpretation(lazy):
        d = dist_to_funsor(random_mvn((), 4))
        d = d(value=x.reshape((4,)) - y)
    _check_mvn_affine(d, data)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('syntax', ['eager', 'lazy'])
def test_poisson_probs_density(batch_shape, syntax):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals())
    def poisson(rate, value):
        return backend_dist.Poisson(rate).log_prob(value)

    check_funsor(poisson, {'rate': reals(), 'value': reals()}, reals())

    rate = Tensor(rand(batch_shape), inputs)
    value = Tensor(ops.astype(ops.astype(ops.exp(randn(batch_shape)), 'int32'), 'float32'), inputs)
    expected = poisson(rate, value)
    check_funsor(expected, inputs, reals())

    d = Variable('value', reals())
    if syntax == 'eager':
        actual = dist.Poisson(rate, value)
    elif syntax == 'lazy':
        actual = dist.Poisson(rate, d)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('syntax', ['eager', 'lazy'])
def test_gamma_probs_density(batch_shape, syntax):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals(), reals())
    def gamma(concentration, rate, value):
        return backend_dist.Gamma(concentration, rate).log_prob(value)

    check_funsor(gamma, {'concentration': reals(), 'rate': reals(), 'value': reals()}, reals())

    concentration = Tensor(rand(batch_shape), inputs)
    rate = Tensor(rand(batch_shape), inputs)
    value = Tensor(ops.exp(randn(batch_shape)), inputs)
    expected = gamma(concentration, rate, value)
    check_funsor(expected, inputs, reals())

    d = Variable('value', reals())
    if syntax == 'eager':
        actual = dist.Gamma(concentration, rate, value)
    elif syntax == 'lazy':
        actual = dist.Gamma(concentration, rate, d)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('syntax', ['eager', 'lazy'])
@pytest.mark.xfail(raises=AttributeError, reason="VonMises is not implemented yet in NumPyro")
def test_von_mises_probs_density(batch_shape, syntax):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals(), reals())
    def von_mises(loc, concentration, value):
        return backend_dist.VonMises(loc, concentration).log_prob(value)

    check_funsor(von_mises, {'concentration': reals(), 'loc': reals(), 'value': reals()}, reals())

    concentration = Tensor(rand(batch_shape), inputs)
    loc = Tensor(rand(batch_shape), inputs)
    value = Tensor(ops.abs(randn(batch_shape)), inputs)
    expected = von_mises(loc, concentration, value)
    check_funsor(expected, inputs, reals())

    d = Variable('value', reals())
    if syntax == 'eager':
        actual = dist.VonMises(loc, concentration, value)
    elif syntax == 'lazy':
        actual = dist.VonMises(loc, concentration, d)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


def _get_stat_diff(funsor_dist_class, sample_inputs, inputs, num_samples, statistic, with_lazy, params):
    params = [Tensor(p, inputs) for p in params]
    if isinstance(with_lazy, bool):
        with interpretation(lazy if with_lazy else eager):
            funsor_dist = funsor_dist_class(*params)
    else:
        funsor_dist = funsor_dist_class(*params)

    rng_key = None if get_backend() == "torch" else np.array([0, 0], dtype=np.uint32)
    sample_value = funsor_dist.sample(frozenset(['value']), sample_inputs, rng_key=rng_key)
    expected_inputs = OrderedDict(
        tuple(sample_inputs.items()) + tuple(inputs.items()) + (('value', funsor_dist.inputs['value']),)
    )
    check_funsor(sample_value, expected_inputs, reals())

    if sample_inputs:

        actual_mean = Integrate(
            sample_value, Variable('value', funsor_dist.inputs['value']), frozenset(['value'])
        ).reduce(ops.add, frozenset(sample_inputs))

        inputs, tensors = align_tensors(*list(funsor_dist.params.values())[:-1])
        raw_dist = funsor_dist.dist_class(**dict(zip(funsor_dist._ast_fields[:-1], tensors)))
        expected_mean = Tensor(raw_dist.mean, inputs)

        if statistic == "mean":
            actual_stat, expected_stat = actual_mean, expected_mean
        elif statistic == "variance":
            actual_stat = Integrate(
                sample_value,
                (Variable('value', funsor_dist.inputs['value']) - actual_mean) ** 2,
                frozenset(['value'])
            ).reduce(ops.add, frozenset(sample_inputs))
            expected_stat = Tensor(raw_dist.variance, inputs)
        elif statistic == "entropy":
            actual_stat = -Integrate(
                sample_value, funsor_dist, frozenset(['value'])
            ).reduce(ops.add, frozenset(sample_inputs))
            expected_stat = Tensor(raw_dist.entropy(), inputs)
        else:
            raise ValueError("invalid test statistic")

        diff = actual_stat.reduce(ops.add).data - expected_stat.reduce(ops.add).data
        return diff.sum(), diff


def _check_sample(funsor_dist_class, params, sample_inputs, inputs, atol=1e-2,
                  num_samples=100000, statistic="mean", skip_grad=False, with_lazy=None):
    """utility that compares a Monte Carlo estimate of a distribution mean with the true mean"""
    samples_per_dim = int(num_samples ** (1./max(1, len(sample_inputs))))
    sample_inputs = OrderedDict((k, bint(samples_per_dim)) for k in sample_inputs)
    _get_stat_diff_fn = functools.partial(
        _get_stat_diff, funsor_dist_class, sample_inputs, inputs, num_samples, statistic, with_lazy)

    if get_backend() == "torch":
        import torch

        for param in params:
            param.requires_grad_()

        res = _get_stat_diff_fn(params)
        if sample_inputs:
            diff_sum, diff = res
            assert_close(diff, ops.new_zeros(diff, diff.shape), atol=atol, rtol=None)
            if not skip_grad:
                diff_grads = torch.autograd.grad(diff_sum, params, allow_unused=True)
                for diff_grad in diff_grads:
                    assert_close(diff_grad, ops.new_zeros(diff_grad, diff_grad.shape), atol=atol, rtol=None)
    elif get_backend() == "jax":
        import jax

        if sample_inputs:
            if skip_grad:
                _, diff = _get_stat_diff_fn(params)
                assert_close(diff, ops.new_zeros(diff, diff.shape), atol=atol, rtol=None)
            else:
                (_, diff), diff_grads = jax.value_and_grad(_get_stat_diff_fn, has_aux=True)(params)
                assert_close(diff, ops.new_zeros(diff, diff.shape), atol=atol, rtol=None)
                for diff_grad in diff_grads:
                    assert_close(diff_grad, ops.new_zeros(diff_grad, diff_grad.shape), atol=atol, rtol=None)
        else:
            _get_stat_diff_fn(params)


@pytest.mark.parametrize('sample_inputs', [(), ('ii',), ('ii', 'jj'), ('ii', 'jj', 'kk')])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('reparametrized', [True, False])
def test_gamma_sample(batch_shape, sample_inputs, reparametrized):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    concentration = rand(batch_shape)
    rate = rand(batch_shape)
    funsor_dist_class = (dist.Gamma if reparametrized else dist.NonreparameterizedGamma)
    params = (concentration, rate)

    _check_sample(funsor_dist_class, params, sample_inputs, inputs, num_samples=200000,
                  atol=5e-2 if reparametrized else 1e-1)


@pytest.mark.parametrize("with_lazy", [True, xfail_param(False, reason="missing pattern")])
@pytest.mark.parametrize('sample_inputs', [(), ('ii',), ('ii', 'jj'), ('ii', 'jj', 'kk')])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('reparametrized', [True, False])
def test_normal_sample(with_lazy, batch_shape, sample_inputs, reparametrized):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = randn(batch_shape)
    scale = rand(batch_shape)
    funsor_dist_class = (dist.Normal if reparametrized else dist.NonreparameterizedNormal)
    params = (loc, scale)

    _check_sample(funsor_dist_class, params, sample_inputs, inputs, num_samples=200000,
                  atol=1e-2 if reparametrized else 1e-1, with_lazy=with_lazy)


@pytest.mark.parametrize("with_lazy", [True, xfail_param(False, reason="missing pattern")])
@pytest.mark.parametrize('sample_inputs', [(), ('ii',), ('ii', 'jj'), ('ii', 'jj', 'kk')])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(1,), (4,), (5,)], ids=str)
def test_mvn_sample(with_lazy, batch_shape, sample_inputs, event_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = randn(batch_shape + event_shape)
    scale_tril = _random_scale_tril(batch_shape + event_shape * 2)
    funsor_dist_class = dist.MultivariateNormal
    params = (loc, scale_tril)

    _check_sample(funsor_dist_class, params, sample_inputs, inputs, atol=7e-2, num_samples=200000, with_lazy=with_lazy)


@pytest.mark.parametrize('sample_inputs', [(), ('ii',), ('ii', 'jj'), ('ii', 'jj', 'kk')])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(1,), (4,), (5,)], ids=str)
@pytest.mark.parametrize('reparametrized', [True, False])
def test_dirichlet_sample(batch_shape, sample_inputs, event_shape, reparametrized):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    concentration = ops.exp(randn(batch_shape + event_shape))
    funsor_dist_class = (dist.Dirichlet if reparametrized else dist.NonreparameterizedDirichlet)
    params = (concentration,)

    _check_sample(funsor_dist_class, params, sample_inputs, inputs, atol=1e-2 if reparametrized else 1e-1)


@pytest.mark.parametrize('sample_inputs', [(), ('ii',), ('ii', 'jj'), ('ii', 'jj', 'kk')])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_bernoullilogits_sample(batch_shape, sample_inputs):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    logits = rand(batch_shape)
    funsor_dist_class = dist.BernoulliLogits
    params = (logits,)

    _check_sample(funsor_dist_class, params, sample_inputs, inputs, atol=5e-2, num_samples=100000)


@pytest.mark.parametrize('sample_inputs', [(), ('ii',), ('ii', 'jj'), ('ii', 'jj', 'kk')])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_bernoulliprobs_sample(batch_shape, sample_inputs):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    probs = rand(batch_shape)
    funsor_dist_class = dist.BernoulliProbs
    params = (probs,)

    _check_sample(funsor_dist_class, params, sample_inputs, inputs, atol=5e-2, num_samples=100000)


@pytest.mark.parametrize("with_lazy", [True, xfail_param(False, reason="missing pattern")])
@pytest.mark.parametrize('sample_inputs', [(), ('ii',), ('ii', 'jj'), ('ii', 'jj', 'kk')])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('reparametrized', [True, False])
def test_beta_sample(with_lazy, batch_shape, sample_inputs, reparametrized):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    concentration1 = ops.exp(randn(batch_shape))
    concentration0 = ops.exp(randn(batch_shape))
    funsor_dist_class = (dist.Beta if reparametrized else dist.NonreparameterizedBeta)
    params = (concentration1, concentration0)

    _check_sample(funsor_dist_class, params, sample_inputs, inputs, atol=1e-2 if reparametrized else 1e-1,
                  statistic="variance", num_samples=100000, with_lazy=with_lazy)


@pytest.mark.parametrize("with_lazy", [True, xfail_param(False, reason="missing pattern")])
@pytest.mark.parametrize('sample_inputs', [(), ('ii',), ('ii', 'jj'), ('ii', 'jj', 'kk')])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_binomial_sample(with_lazy, batch_shape, sample_inputs):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    max_count = 10
    total_count_data = random_tensor(inputs, bint(max_count)).data
    if get_backend() == "torch":
        total_count_data = ops.astype(total_count_data, 'float')
    total_count = total_count_data
    probs = rand(batch_shape)
    funsor_dist_class = dist.Binomial
    params = (total_count, probs)

    _check_sample(funsor_dist_class, params, sample_inputs, inputs, atol=2e-2, skip_grad=True, with_lazy=with_lazy)


@pytest.mark.parametrize('sample_inputs', [(), ('ii',), ('ii', 'jj'), ('ii', 'jj', 'kk')])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_poisson_sample(batch_shape, sample_inputs):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    rate = rand(batch_shape)
    funsor_dist_class = dist.Poisson
    params = (rate,)

    _check_sample(funsor_dist_class, params, sample_inputs, inputs, skip_grad=True)
