# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import numpy as np
import pytest

import funsor.ops as ops
from funsor.domains import Bint, Real, Reals
from funsor.interpreter import gensym, reinterpret
from funsor.terms import (
    Cat,
    Independent,
    Lambda,
    Number,
    Slice,
    Stack,
    Variable,
    reflect,
)
from funsor.testing import assert_close, check_funsor, random_tensor
from funsor.util import get_backend


def test_sample_subs_smoke():
    x = random_tensor(OrderedDict([("i", Bint[3]), ("j", Bint[2])]), Real)
    with reflect:
        z = x(i=1)
    rng_key = None if get_backend() == "torch" else np.array([0, 1], dtype=np.uint32)
    actual = z.sample(frozenset({"j"}), OrderedDict({"i": Bint[4]}), rng_key=rng_key)
    check_funsor(actual, {"j": Bint[2], "i": Bint[4]}, Real)


def test_subs_reduce():
    x = random_tensor(OrderedDict([("i", Bint[3]), ("j", Bint[2])]), Real)
    ix = random_tensor(OrderedDict([("i", Bint[3])]), Bint[2])
    ix2 = ix(i="i2")
    with reflect:
        actual = x.reduce(ops.add, frozenset({"i"}))
    actual = actual(j=ix)
    expected = x(j=ix2).reduce(ops.add, frozenset({"i"}))(i2="i")
    assert_close(actual, expected)


@pytest.mark.parametrize("lhs_vars", [(), ("i",), ("j",), ("i", "j")])
@pytest.mark.parametrize("rhs_vars", [(), ("i",), ("j",), ("i", "j")])
def test_distribute_reduce(lhs_vars, rhs_vars):
    lhs_vars, rhs_vars = frozenset(lhs_vars), frozenset(rhs_vars)
    lhs = random_tensor(OrderedDict([("i", Bint[3]), ("j", Bint[2])]), Real)
    rhs = random_tensor(OrderedDict([("i", Bint[3]), ("j", Bint[2])]), Real)

    with reflect:
        actual_lhs = lhs.reduce(ops.add, lhs_vars) if lhs_vars else lhs
        actual_rhs = rhs.reduce(ops.add, rhs_vars) if rhs_vars else rhs

    actual = reinterpret(actual_lhs * actual_rhs)

    lhs_subs = {v: gensym(v) for v in lhs_vars}
    rhs_subs = {v: gensym(v) for v in rhs_vars}
    expected = (lhs(**lhs_subs) * rhs(**rhs_subs)).reduce(
        ops.add, frozenset(lhs_subs.values()) | frozenset(rhs_subs.values())
    )

    assert_close(actual, expected)


def test_lazy_subs_type_clash():
    with reflect:
        Slice("t", 3)(t=Slice("t", 2, dtype=3)).reduce(ops.add)


@pytest.mark.parametrize("name", ["s", "t"])
def test_cat(name):
    with reflect:
        x = Stack("t", (Number(1), Number(2)))
        y = Stack("t", (Number(4), Number(8), Number(16)))
        xy = Cat(name, (x, y), "t")
        xy.reduce(ops.add)


def test_subs_lambda():
    z = Variable("z", Real)
    i = Variable("i", Bint[5])
    ix = random_tensor(OrderedDict([("i", Bint[5])]), Real)
    actual = Lambda(i, z)(z=ix)
    expected = Lambda(i(i="j"), z(z=ix))
    check_funsor(actual, expected.inputs, expected.output)
    assert_close(actual, expected)


def test_slice_lambda():
    z = Variable("z", Real)
    i = Variable("i", Bint[5])
    j = Variable("j", Bint[7])
    zi = Lambda(i, z)
    zj = Lambda(j, z)
    zij = Lambda(j, zi)
    zj2 = zij[:, i]
    check_funsor(zj2, zj.inputs, zj.output)


def test_subs_independent():
    f = Variable("x_i", Reals[4, 5]) + random_tensor(OrderedDict(i=Bint[3]))

    actual = Independent(f, "x", "i", "x_i")
    assert "i" not in actual.inputs
    assert "x_i" not in actual.inputs

    y = Variable("y", Reals[3, 4, 5])
    fsub = y + (0.0 * random_tensor(OrderedDict(i=Bint[7])))
    actual = actual(x=fsub)
    assert actual.inputs["i"] == Bint[7]

    expected = f(x_i=y["i"]).reduce(ops.add, "i")

    data = random_tensor(OrderedDict(i=Bint[7]), y.output)
    assert_close(actual(y=data), expected(y=data))


@pytest.mark.xfail(reason="Independent not quite compatible with sample")
def test_sample_independent():
    f = Variable("x_i", Reals[4, 5]) + random_tensor(OrderedDict(i=Bint[3]))
    actual = Independent(f, "x", "i", "x_i")
    assert actual.sample("i")
    assert actual.sample("j", {"i": 2})


def test_subs_already_bound():
    with reflect:
        x = Variable("x", Real)
        y1 = (2 * x)(x=3)
        y2 = y1.arg(4)
    assert y1.bound != y2.bound
