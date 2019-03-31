from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from six import add_metaclass

import funsor.ops as ops
from funsor.domains import Domain, reals
from funsor.integrate import Integrate, integrator
from funsor.interpreter import debug_logged
from funsor.ops import AddOp, SubOp, TransformOp
from funsor.registry import KeyedRegistry
from funsor.terms import (
    Align,
    Binary,
    Funsor,
    FunsorMeta,
    Lambda,
    Number,
    Subs,
    Unary,
    Uncurry,
    Variable,
    eager,
    to_funsor
)


class DeltaMeta(FunsorMeta):
    """
    Wrapper to fill in defaults.
    """
    def __call__(cls, name, point, log_density=0):
        point = to_funsor(point)
        log_density = to_funsor(log_density)
        return super(DeltaMeta, cls).__call__(name, point, log_density)


@add_metaclass(DeltaMeta)
class Delta(Funsor):
    """
    Normalized delta distribution binding a single variable.

    :param str name: Name of the bound variable.
    :param Funsor point: Value of the bound variable.
    :param Funsor log_density: Optional log density to be added when evaluating
        at a point. This is needed to make :class:`Delta` closed under
        differentiable substitution.
    """
    def __init__(self, name, point, log_density=0):
        assert isinstance(name, str)
        assert isinstance(point, Funsor)
        assert isinstance(log_density, Funsor)
        assert log_density.output == reals()
        inputs = OrderedDict([(name, point.output)])
        inputs.update(point.inputs)
        inputs.update(log_density.inputs)
        output = reals()
        super(Delta, self).__init__(inputs, output)
        self.name = name
        self.point = point
        self.log_density = log_density

    def eager_subs(self, subs):
        value = None
        index_part = []
        for k, v in subs:
            if k in self.inputs:
                if k == self.name:
                    value = v
                else:
                    assert self.name not in v.inputs
                    index_part.append((k, v))
        index_part = tuple(index_part)

        if index_part:
            point = Subs(self.point, index_part)
            log_density = Subs(self.log_density, index_part)
            result = Delta(self.name, point, log_density)
            if value is not None:
                result = Subs(result, ((self.name, value),))
            return result

        if value is None:
            return self

        if isinstance(value, Variable):
            return Delta(value.name, self.point, self.log_density)

        if not any(d.dtype == 'real' for side in (value, self.point)
                   for d in side.inputs.values()):
            return (value == self.point).all().log() + self.log_density

        # Try to invert the substitution.
        soln = solve(value, self.point)
        if soln is None:
            return None  # lazily substitute
        name, point, log_density = soln
        log_density += self.log_density
        return Delta(name, point, log_density)

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            if self.name in reduced_vars:
                return Number(0)  # Deltas are normalized.

        # TODO Implement ops.add to simulate .to_event().

        return None  # defer to default implementation


@eager.register(Binary, AddOp, Delta, (Funsor, Align))
def eager_add(op, lhs, rhs):
    if lhs.name in rhs.inputs:
        rhs = rhs(**{lhs.name: lhs.point})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, SubOp, Delta, (Funsor, Align))
def eager_sub(op, lhs, rhs):
    if lhs.name in rhs.inputs:
        rhs = rhs(**{lhs.name: lhs.point})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, AddOp, (Funsor, Align), Delta)
def eager_add(op, lhs, rhs):
    if rhs.name in lhs.inputs:
        lhs = lhs(**{rhs.name: rhs.point})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Uncurry, Delta, str, str)
def eager_uncurry(delta, reals_var, bint_var):
    if delta.name == reals_var:
        i = Variable(bint_var, delta.inputs[bint_var])
        point = Lambda(i, delta.point)
        log_density = delta.log_density.reduce(ops.add, bint_var)
        return Delta(delta.name, point, log_density)

    return None  # defer to default implementation


@eager.register(Integrate, Delta, Funsor, frozenset)
@integrator
def eager_integrate(delta, integrand, reduced_vars):
    assert delta.name in reduced_vars
    integrand = Subs(integrand, ((delta.name, delta.point),))
    log_measure = delta.log_density
    reduced_vars -= frozenset([delta.name])
    return Integrate(log_measure, integrand, reduced_vars)


def solve(expr, value):
    """
    Tries to solve for free inputs of an ``expr`` such that ``expr == value``,
    and computes the log-abs-det-Jacobian of the resulting substitution.

    :param Funsor expr: An expression with a free variable.
    :param Funsor value: A target value.
    :return: A tuple ``(name, point, log_abs_det_jacobian)``
    :rtype: tuple
    :raises: ValueError
    """
    assert isinstance(expr, Funsor)
    assert isinstance(value, Funsor)
    result = solve.dispatch(type(expr), *(expr._ast_values + (value,)))
    if result is None:
        raise ValueError("Cannot substitute into a Delta: {}".format(value))
    return result


_solve = KeyedRegistry(lambda *args: None)
solve.dispatch = _solve.__call__
solve.register = _solve.register


@solve.register(Variable, str, Domain, Funsor)
@debug_logged
def solve_variable(name, output, y):
    assert y.output == output
    point = y
    log_density = Number(0)
    return name, point, log_density


@solve.register(Unary, TransformOp, Funsor, Funsor)
@debug_logged
def solve_unary(op, arg, y):
    x = op.inv(y)
    name, point, log_density = solve(arg, x)
    log_density += op.log_abs_det_jacobian(x, y)
    return name, point, log_density


__all__ = [
    'Delta',
    'solve',
]
