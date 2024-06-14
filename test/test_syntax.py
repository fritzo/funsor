# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import operator
import sys

import pytest

from funsor import ops
from funsor.syntax import alpha_rename, rewrite_ops

python_version = tuple(map(int, sys.version.split()[0].split(".")[:2]))


def assert_fn_eq(actual, expected):
    assert actual.__name__ == expected.__name__
    assert actual.__doc__ == expected.__doc__
    assert actual.__module__ == expected.__module__
    assert actual.__closure__ == expected.__closure__
    assert actual.__code__.co_code == expected.__code__.co_code


@pytest.mark.skipif(python_version < (3, 9), reason="not supported")
def test_rewrite_ops():
    @rewrite_ops({"+": "sum_op", "*": "prod_op"})
    def product_rule(sum_op, prod_op, lhs, rhs, d):
        """foo"""
        return d(lhs) * rhs + lhs * d(rhs)

    actual = product_rule

    def product_rule(sum_op, prod_op, lhs, rhs, d):
        """foo"""
        return sum_op(prod_op(d(lhs), rhs), prod_op(lhs, d(rhs)))

    expected = product_rule

    assert_fn_eq(actual, expected)
    args = (max, operator.add, 1.23, 4.56, lambda x: x**2)
    assert actual(*args) == expected(*args)


@pytest.mark.skipif(python_version < (3, 9), reason="not supported")
def test_rewrite_ops_register():
    registry = set()

    def register(fn):
        assert fn.__name__ not in registry, "duplicate registry"
        registry.add(fn.__name__)
        return fn

    @register
    @rewrite_ops({"+": "sum_op", "*": "prod_op"})
    def product_rule(sum_op, prod_op, lhs, rhs, d):
        """foo"""
        return d(lhs) * rhs + lhs * d(rhs)

    assert len(registry) == 1
    registry.clear()
    actual = product_rule

    @register
    def product_rule(sum_op, prod_op, lhs, rhs, d):
        """foo"""
        return sum_op(prod_op(d(lhs), rhs), prod_op(lhs, d(rhs)))

    assert len(registry) == 1
    expected = product_rule

    assert_fn_eq(actual, expected)
    args = (max, operator.add, 1.23, 4.56, lambda x: x**2)
    assert actual(*args) == expected(*args)


@pytest.mark.skipif(python_version < (3, 9), reason="not supported")
def test_complex():
    @rewrite_ops(
        {"+": "add_op", "*": "mul_op", "-": "sub_op", "/": "div_op"},
        {"-": "neg_op"},
        {0.0: "zero"},
    )
    def foo(add_op, mul_op, x, y, z):
        sub_op = ops.BINARY_INVERSES[add_op]  # noqa: F841
        div_op = ops.BINARY_INVERSES[mul_op]  # noqa: F841
        neg_op = ops.UNARY_INVERSES[add_op]  # noqa: F841
        zero = ops.UNITS[add_op]  # noqa: F841
        return (-y) * (x + y * z - 0.0 / x)

    actual = foo

    def foo(add_op, mul_op, x, y, z):
        sub_op = ops.BINARY_INVERSES[add_op]
        div_op = ops.BINARY_INVERSES[mul_op]
        neg_op = ops.UNARY_INVERSES[add_op]
        zero = ops.UNITS[add_op]
        return mul_op(neg_op(y), sub_op(add_op(x, mul_op(y, z)), div_op(zero, x)))

    expected = foo

    assert_fn_eq(actual, expected)

    args = (ops.add, ops.mul, 1.23, 4.56, 7.89)
    assert actual(*args) == expected(*args)

    args = (ops.add, ops.add, 1.23, 4.56, 7.89)
    assert actual(*args) == expected(*args)

    args = (ops.mul, ops.mul, 1.23, 4.56, 7.89)
    assert actual(*args) == expected(*args)


def test_alpha_rename_1():
    @alpha_rename
    def fn(a, b, /, c, d=1):
        e = a + b
        return c + d + e

    actual = fn

    def fn(_bound_0, _bound_1, /, c, d=1):
        e = _bound_0 + _bound_1
        return c + d + e

    expected = fn
    assert_fn_eq(actual, expected)

    assert actual(1, 2, 3, 4) == expected(1, 2, 3, 4)


@pytest.mark.xfail(reason="failure to reproduce enclosing scope")
def test_alpha_rename_2():
    @alpha_rename(locals_=locals())
    def fn(a, b, /, c, d=1):
        e = a + b
        return c + d + e + f

    actual = fn

    def fn(_bound_0, _bound_1, /, c, d=1):
        e = _bound_0 + _bound_1
        return c + d + e + f

    expected = fn
    assert_fn_eq(actual, expected)

    f = 0
    assert actual(1, 2, 3, 4) == expected(1, 2, 3, 4)
