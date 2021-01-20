# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import weakref

import pytest

from funsor.distribution import BACKEND_TO_DISTRIBUTIONS_BACKEND
from funsor.util import get_backend
from funsor.ops import WrappedTransformOp


@pytest.fixture
def dist():
    try:
        module_name = BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()]
    except KeyError:
        pytest.skip(f"missing distributions module for {get_backend()}")
    return pytest.importorskip(module_name).dist


def test_transform_op_cache(dist):
    t = dist.transforms.PowerTransform(0.5)
    W = WrappedTransformOp
    assert W(t) is W(t)
    assert W(t).inv is W(t).inv
    assert W(t.inv) is W(t).inv
    assert W(t).log_abs_det_jacobian is W(t).log_abs_det_jacobian


@pytest.mark.xfail(reason="reference cycle")
def test_transform_op_gc(dist):
    op_set = weakref.WeakSet()

    t = dist.transforms.PowerTransform(0.5)
    op_set.add(WrappedTransformOp(t))
    assert len(op_set) == 1
    op_set.add(WrappedTransformOp(t))
    assert len(op_set) == 1
    del t
    for op in op_set:
        import sys
        print(sys.getrefcount(op), sys.getrefcount(op.fn))
    assert len(op_set) == 0
