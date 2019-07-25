from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
from six import integer_types
from six.moves import reduce

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.interpreter import interpretation
from funsor.optimizer import apply_optimizer, optimize
from funsor.sum_product import sum_product
from funsor.terms import Funsor, normalize
from funsor.torch import Tensor


def naive_contract_einsum(eqn, *terms, **kwargs):
    """
    Use for testing Contract against einsum
    """
    assert "plates" not in kwargs

    backend = kwargs.pop('backend', 'torch')
    if backend == 'torch':
        sum_op, prod_op = ops.add, ops.mul
    elif backend in ('pyro.ops.einsum.torch_log', 'pyro.ops.einsum.torch_marginal'):
        sum_op, prod_op = ops.logaddexp, ops.add
    else:
        raise ValueError("{} backend not implemented".format(backend))

    assert isinstance(eqn, str)
    assert all(isinstance(term, Funsor) for term in terms)
    inputs, output = eqn.split('->')
    inputs = inputs.split(',')
    assert len(inputs) == len(terms)
    assert len(output.split(',')) == 1
    input_dims = frozenset(d for inp in inputs for d in inp)
    output_dims = frozenset(d for d in output)
    reduced_vars = input_dims - output_dims

    with interpretation(optimize):
        return Contraction(sum_op, prod_op, reduced_vars, terms)


def naive_einsum(eqn, *terms, **kwargs):
    """
    Implements standard variable elimination.
    """
    backend = kwargs.pop('backend', 'torch')
    if backend == 'torch':
        sum_op, prod_op = ops.add, ops.mul
    elif backend in ('pyro.ops.einsum.torch_log', 'pyro.ops.einsum.torch_marginal'):
        sum_op, prod_op = ops.logaddexp, ops.add
    else:
        raise ValueError("{} backend not implemented".format(backend))

    assert isinstance(eqn, str)
    assert all(isinstance(term, Funsor) for term in terms)
    inputs, output = eqn.split('->')
    assert len(output.split(',')) == 1
    input_dims = frozenset(d for inp in inputs.split(',') for d in inp)
    output_dims = frozenset(output)
    reduce_dims = input_dims - output_dims
    return reduce(prod_op, terms).reduce(sum_op, reduce_dims)


def naive_plated_einsum(eqn, *terms, **kwargs):
    """
    Implements Tensor Variable Elimination (Algorithm 1 in [Obermeyer et al 2019])

    [Obermeyer et al 2019] Obermeyer, F., Bingham, E., Jankowiak, M., Chiu, J.,
        Pradhan, N., Rush, A., and Goodman, N.  Tensor Variable Elimination for
        Plated Factor Graphs, 2019
    """
    plates = kwargs.pop('plates', '')
    if not plates:
        return naive_einsum(eqn, *terms, **kwargs)

    backend = kwargs.pop('backend', 'torch')
    if backend == 'torch':
        sum_op, prod_op = ops.add, ops.mul
    elif backend in ('pyro.ops.einsum.torch_log', 'pyro.ops.einsum.torch_marginal'):
        sum_op, prod_op = ops.logaddexp, ops.add
    else:
        raise ValueError("{} backend not implemented".format(backend))

    assert isinstance(eqn, str)
    assert all(isinstance(term, Funsor) for term in terms)
    inputs, output = eqn.split('->')
    inputs = inputs.split(',')
    assert len(inputs) == len(terms)
    assert len(output.split(',')) == 1
    input_dims = frozenset(d for inp in inputs for d in inp)
    output_dims = frozenset(d for d in output)
    plate_dims = frozenset(plates) - output_dims
    reduce_vars = input_dims - output_dims - frozenset(plates)

    output_plates = output_dims & frozenset(plates)
    if not all(output_plates.issubset(inp) for inp in inputs):
        raise NotImplementedError("TODO")

    eliminate = plate_dims | reduce_vars
    return sum_product(sum_op, prod_op, terms, eliminate, frozenset(plates))


def einsum(eqn, *terms, **kwargs):
    r"""
    Top-level interface for optimized tensor variable elimination.

    :param str equation: An einsum equation.
    :param funsor.terms.Funsor \*terms: One or more operands.
    :param set plates: Optional keyword argument denoting which funsor
        dimensions are plate dimensions. Among all input dimensions (from
        terms): dimensions in plates but not in outputs are product-reduced;
        dimensions in neither plates nor outputs are sum-reduced.
    """
    with interpretation(normalize):
        naive_ast = naive_plated_einsum(eqn, *terms, **kwargs)
    return apply_optimizer(naive_ast)
