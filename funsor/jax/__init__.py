# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax.core import Tracer
from jax.interpreters.xla import DeviceArray

from funsor.adjoint import adjoint_ops
from funsor.interpreter import children, recursion_reinterpret
from funsor.ops import AssociativeOp
from funsor.tensor import Tensor, tensor_to_funsor
from funsor.terms import Funsor, to_funsor
from funsor.util import quote

from . import distributions as _
from . import ops as _

del _  # flake8


@recursion_reinterpret.register(DeviceArray)
@recursion_reinterpret.register(Tracer)
def _recursion_reinterpret_ground(x):
    return x


@children.register(DeviceArray)
@children.register(Tracer)
def _children_ground(x):
    return ()


to_funsor.register(DeviceArray)(tensor_to_funsor)
to_funsor.register(Tracer)(tensor_to_funsor)


@quote.register(DeviceArray)
def _quote(x, indent, out):
    """
    Work around JAX's DeviceArray not supporting reproducible repr.
    """
    out.append(
        (indent, "np.array({}, dtype=np.{})".format(repr(x.copy().tolist()), x.dtype))
    )
