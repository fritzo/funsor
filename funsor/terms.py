# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import itertools
import numbers
import typing
import warnings
from collections import OrderedDict, namedtuple
from functools import reduce, singledispatch
from weakref import WeakValueDictionary

from multipledispatch import dispatch

import funsor.interpreter as interpreter
import funsor.ops as ops
from funsor.domains import (
    Array,
    Bint,
    BintType,
    Domain,
    Product,
    ProductDomain,
    Real,
    find_domain,
)
from funsor.interpretations import (
    Interpretation,
    die,
    eager,
    lazy,
    moment_matching,
    reflect,
    sequential,
)
from funsor.interpreter import PatternMissingError, interpret
from funsor.ops import AssociativeOp, GetitemOp, Op
from funsor.ops.builtin import normalize_ellipsis, parse_ellipsis, parse_slice
from funsor.syntax import INFIX_OPERATORS, PREFIX_OPERATORS
from funsor.typing import GenericTypeMeta, Variadic, deep_type, get_args, get_origin
from funsor.util import getargspec, lazy_property, pretty, quote, register_pprint

from . import instrument, interpreter, ops

_PREFIX = {k: v for v, k, _ in PREFIX_OPERATORS}
_INFIX = {k: v for v, k, _ in INFIX_OPERATORS}


# FIXME this can lead to linear nesting of interpretations
# when used in combination with alpha_convert and optimize.
# See failing example at https://github.com/pyro-ppl/funsor/pull/414
class SubstituteInterpretation(Interpretation):
    def __init__(self, subs, base_interpretation):
        super().__init__("subs")
        self.subs = subs
        self.base_interpretation = base_interpretation
        assert isinstance(subs, tuple)
        assert all(isinstance(v, Funsor) for k, v in subs)

    @property
    def is_total(self):
        return self.base_interpretation.is_total

    def interpret(self, cls, *args):
        with self.base_interpretation:
            expr = cls(*args)
            fresh_subs = tuple((k, v) for k, v in self.subs if k in expr.fresh)
            if fresh_subs:
                expr = instrument.debug_logged(expr.eager_subs)(fresh_subs)
            if instrument.PROFILE:
                instrument.COUNTERS["interpretation"]["substitute"] += 1
            return expr


def substitute(expr, subs):
    if isinstance(subs, (dict, OrderedDict)):
        subs = tuple(subs.items())
    support = frozenset(k for k, v in subs)

    def stop(x):
        if interpreter.is_atom(x):
            return True
        if isinstance(x, Funsor) and support.isdisjoint(x.inputs):
            return True
        return False

    if stop(expr):
        return expr

    env = interpreter.anf(expr, stop)

    with SubstituteInterpretation(subs, interpreter.get_interpretation()):
        for key, value in env.items():
            args = tuple(
                c if interpreter.is_atom(c) else env.get(c, c)
                for c in interpreter.children(value)
            )
            if isinstance(value, (tuple, frozenset)):  # TODO absorb this into interpret
                env[key] = type(value)(args)
            else:
                env[key] = type(value)(*args)
    return env[expr]


def _alpha_mangle(bound_vars):
    """
    Rename bound variables in expr to avoid conflict with any free variables.
    Returns substitution dictionary with mangled names for consumption by Funsor._alpha_convert.
    """
    return {
        name: interpreter.gensym(name.split("__BOUND_")[0] + "__BOUND_")
        for name in bound_vars
    }


_SKIP_ALPHA = False


@reflect.set_callable
def reflect(cls, *args, **kwargs):
    """
    Construct a funsor, populate ``._ast_values``, and cons hash.
    This is the only interpretation allowed to construct funsors.
    """
    if len(args) > len(cls._ast_fields):
        # handle varargs
        new_args = tuple(args[: len(cls._ast_fields) - 1]) + (
            args[len(cls._ast_fields) - 1 - len(args) :],
        )
        assert len(new_args) == len(cls._ast_fields)
        _, args = args, new_args

    cache_key = reflect.make_hash_key(cls, *args)
    if cache_key in cls._cons_cache:
        return cls._cons_cache[cache_key]

    arg_types = tuple(map(deep_type, args))
    cls_specific = get_origin(cls)[arg_types]
    result = super(FunsorMeta, cls_specific).__call__(*args)
    result._ast_values = args

    # alpha-convert eagerly upon binding any variable.
    global _SKIP_ALPHA
    if result.bound and not _SKIP_ALPHA:
        alpha_subs = _alpha_mangle(result.bound)
        try:
            # optimization: don't perform alpha-conversion again
            # when renaming subexpressions of result
            _SKIP_ALPHA = True
            alpha_mangled_args = reflect(result._alpha_convert)(alpha_subs)
        finally:
            _SKIP_ALPHA = False

        # TODO eliminate code duplication below
        # this is currently necessary because .bound is computed in __init__().
        result = super(FunsorMeta, cls_specific).__call__(*alpha_mangled_args)
        result._ast_values = alpha_mangled_args

        # we also make the old cons cache_key point to the new mangled value.
        # this guarantees that alpha-conversion only runs once for this expression.
        cls._cons_cache[cache_key] = result

        cache_key = reflect.make_hash_key(cls, *alpha_mangled_args)

    if instrument.PROFILE:
        size, depth, width = _get_ast_stats(result)
        instrument.COUNTERS["ast_size"][size] += 1
        instrument.COUNTERS["ast_depth"][depth] += 1
        classname = get_origin(cls).__name__
        instrument.COUNTERS["funsor"][classname] += 1
        instrument.COUNTERS[classname][width] += 1

    cls._cons_cache[cache_key] = result
    return result


class FunsorMeta(GenericTypeMeta):
    """
    Metaclass for Funsors to perform four independent tasks:

    1.  Fill in default kwargs and convert kwargs to args before deferring to a
        nonstandard interpretation. This allows derived metaclasses to fill in
        defaults and do type conversion, thereby simplifying logic of
        interpretations.
    2.  Ensure each Funsor class has an attribute ``._ast_fields`` describing
        its input args and each Funsor instance has an attribute
        ``._ast_values`` with values corresponding to its input args. This
        allows the instance to be reflectively reconstructed under a different
        interpretation, and is used by :func:`funsor.interpreter.reinterpret`.
    3.  Cons-hash construction, so that repeatedly calling the constructor
        with identical args will produce the same object. This enables cheap
        syntactic equality testing using the ``is`` operator, which is
        important both for hashing (e.g. for memoizing funsor functions)
        and for unit testing, since ``.__eq__()`` is overloaded with
        elementwise semantics. Cons hashing differs from memoization in that
        it incurs no memory overhead beyond the cons hash dict.
    4.  Support subtyping with parameters for pattern matching, e.g. Number[int, int].
    """

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        register_pprint(cls)
        if not cls.__args__:
            cls._ast_fields = getargspec(cls.__init__)[0][1:]
            cls._cons_cache = WeakValueDictionary()

    def __getitem__(cls, arg_types):
        if not isinstance(arg_types, tuple):
            arg_types = (arg_types,)
        assert len(arg_types) == len(
            cls._ast_fields
        ), "Must provide exactly one type per subexpression"
        return super().__getitem__(arg_types)

    def __call__(cls, *args, **kwargs):
        if cls.__args__:
            cls = cls.__origin__

        # Convert kwargs to args.
        if kwargs:
            args = list(args)
            for name in cls._ast_fields[len(args) :]:
                args.append(kwargs.pop(name))
            assert not kwargs, kwargs
            args = tuple(args)

        return interpret(cls, *args)

    @lazy_property
    def classname(cls):
        return repr(cls)


def _convert_reduced_vars(reduced_vars, inputs):
    """
    Helper to convert the reduced_vars arg of ``.reduce()`` and friends.

    :param reduced_vars:
    :type reduced_vars: str, Variable, or set or frozenset thereof.
    :returns: A frozenset of reduced variables.
    :rtype: frozenset of :class:`Variable`
    """
    # Avoid copying if arg is of correct type.
    if isinstance(reduced_vars, frozenset):
        if all(isinstance(var, Variable) for var in reduced_vars):
            return reduced_vars

    if isinstance(reduced_vars, (str, Variable)):
        reduced_vars = {reduced_vars}
    assert isinstance(reduced_vars, (frozenset, set))
    assert all(isinstance(var, (str, Variable)) for var in reduced_vars)
    return frozenset(
        Variable(var, inputs[var]) if isinstance(var, str) else var
        for var in reduced_vars
    )


class Funsor(object, metaclass=FunsorMeta):
    """
    Abstract base class for immutable functional tensors.

    Concrete derived classes must implement ``__init__()`` methods taking
    hashable ``*args`` and no optional ``**kwargs`` so as to support cons
    hashing.

    Derived classes with ``.fresh`` variables must implement an
    :meth:`eager_subs` method. Derived classes with ``.bound`` variables must
    implement an :meth:`_alpha_convert` method.

    :param OrderedDict inputs: A mapping from input name to domain.
        This can be viewed as a typed context or a mapping from
        free variables to domains.
    :param Domain output: An output domain.
    """

    def __init__(self, inputs, output, fresh=None, bound=None):
        fresh = frozenset() if fresh is None else fresh
        bound = {} if bound is None else bound
        assert isinstance(inputs, OrderedDict)
        for name, input_ in inputs.items():
            assert isinstance(name, str)
            assert isinstance(input_, Domain)
        assert isinstance(output, Domain)
        assert getattr(output, "is_concrete", True)
        assert isinstance(fresh, frozenset)
        assert isinstance(bound, dict)
        super(Funsor, self).__init__()
        self.inputs = inputs
        self.output = output
        self.fresh = fresh
        self.bound = bound

    @property
    def dtype(self):
        return self.output.dtype

    @property
    def shape(self):
        return self.output.shape

    @lazy_property
    def input_vars(self):
        return frozenset(Variable(k, v) for k, v in self.inputs.items())

    def __copy__(self):
        return self

    def __reduce__(self):
        return type(self).__origin__, self._ast_values

    def __hash__(self):
        return id(self)

    @lazy_property
    def __annotations__(self):
        type_hints = dict(self.inputs)
        type_hints["return"] = self.output
        return type_hints

    def __repr__(self):
        try:
            ast_values = self._ast_values
        except AttributeError:
            # E.g. when printing errors during __init__, before ._ast_values is set.
            return f"{type(self).__name__}(...)"
        return "{}({})".format(type(self).__name__, ", ".join(map(repr, ast_values)))

    def __str__(self):
        return "{}({})".format(
            type(self).__name__, ", ".join(map(str, self._ast_values))
        )

    def quote(self):
        return quote(self)

    def pretty(self, *args, **kwargs):
        return pretty(self, *args, **kwargs)

    def __contains__(self, item):
        raise TypeError

    def _alpha_convert(self, alpha_subs):
        """
        Rename bound variables while preserving all free variables.
        """
        # Substitute all funsor values.
        assert set(alpha_subs).issubset(self.bound)
        alpha_subs = {k: to_funsor(v, self.bound[k]) for k, v in alpha_subs.items()}
        return substitute(self._ast_values, alpha_subs)

    def __call__(self, *args, **kwargs):
        """
        Partially evaluates this funsor by substituting dimensions.
        """
        # Eagerly restrict to this funsor's inputs.
        subs = OrderedDict(zip(self.inputs, args))
        for k in self.inputs:
            if k in kwargs:
                subs[k] = kwargs[k]
        return Subs(self, tuple(subs.items()))

    def __bool__(self):
        if self.inputs or self.output.shape:
            raise ValueError(
                "bool value of Funsor with more than one value is ambiguous"
            )
        raise NotImplementedError

    def __nonzero__(self):
        return self.__bool__()

    def __len__(self):
        if not self.output.shape:
            raise ValueError("Funsor with empty shape has no len()")
        return self.output.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def item(self):
        if self.inputs or self.output.shape:
            raise ValueError(
                "only one element Funsors can be converted to Python scalars"
            )
        raise NotImplementedError

    @property
    def requires_grad(self):
        return False

    def reduce(self, op, reduced_vars=None):
        """
        Reduce along all or a subset of inputs.

        :param op: A reduction operation.
        :type op: ~funsor.ops.AssociativeOp or ~funsor.ops.ReductionOp
        :param reduced_vars: An optional input name or set of names to reduce.
            If unspecified, all inputs will be reduced.
        :type reduced_vars: str, Variable, or set or frozenset thereof.
        """
        assert isinstance(op, (AssociativeOp, ops.ReductionOp))

        # Eagerly convert reduced_vars to appropriate things.
        if reduced_vars is None:
            # Empty reduced_vars means "reduce over everything".
            reduced_vars = frozenset(Variable(k, v) for k, v in self.inputs.items())
        else:
            reduced_vars = _convert_reduced_vars(reduced_vars, self.inputs)
        assert isinstance(reduced_vars, frozenset), reduced_vars

        # Attempt to convert ReductionOp to AssociativeOp.
        if isinstance(op, ops.ReductionOp):
            if isinstance(op, ops.MeanOp):
                reduced_vars &= self.input_vars
                if not reduced_vars:
                    return self
                scale = 1 / reduce(ops.mul, [v.output.size for v in reduced_vars], 1)
                return self.reduce(ops.add, reduced_vars) * scale
            if isinstance(op, ops.VarOp):
                diff = self - self.reduce(ops.mean, reduced_vars)
                return (diff * diff).reduce(ops.mean, reduced_vars)
            if isinstance(op, ops.StdOp):
                return self.reduce(ops.var, reduced_vars).sqrt()
            raise NotImplementedError(f"Unsupported reduction op: {op}")
        assert isinstance(op, AssociativeOp)

        if not reduced_vars:
            return self
        return Reduce(op, self, reduced_vars)

    def approximate(self, op, guide, approx_vars=None):
        """
        Approximate wrt and all or a subset of inputs.

        :param AssociativeOp op: A reduction operation.
        :param Funsor guide: A guide funsor (e.g. a proposal distribution).
        :param approx_vars: An optional input name or set of names to reduce.
            If unspecified, all inputs will be reduced.
        :type approx_vars: str, Variable, or set or frozenset thereof.
        """
        assert isinstance(op, AssociativeOp)
        assert self.output == Real
        assert guide.output == self.output
        # Eagerly convert approx_vars to appropriate things.
        inputs = self.inputs.copy()
        inputs.update(guide.inputs)
        input_vars = self.input_vars | guide.input_vars
        if approx_vars is None:
            # Empty approx_vars means "approximate everything".
            approx_vars = input_vars
        else:
            approx_vars = _convert_reduced_vars(approx_vars, inputs)
            approx_vars &= input_vars  # Drop unrelated vars.
        if not approx_vars:
            return self  # exact
        return Approximate(op, self, guide, approx_vars)

    def sample(self, sampled_vars, sample_inputs=None, rng_key=None):
        """
        Create a Monte Carlo approximation to this funsor by replacing
        functions of ``sampled_vars`` with :class:`~funsor.delta.Delta` s.

        The result is a :class:`Funsor` with the same ``.inputs`` and
        ``.output`` as the original funsor (plus ``sample_inputs`` if
        provided), so that self can be replaced by the sample in expectation
        computations::

            y = x.sample(sampled_vars)
            assert y.inputs == x.inputs
            assert y.output == x.output
            exact = (x.exp() * integrand).reduce(ops.add)
            approx = (y.exp() * integrand).reduce(ops.add)

        If ``sample_inputs`` is provided, this creates a batch of samples.

        :param sampled_vars: A set of input variables to sample.
        :type sampled_vars: str, Variable, or set or frozenset thereof.
        :param OrderedDict sample_inputs: An optional mapping from variable
            name to :class:`~funsor.domains.Domain` over which samples will
            be batched.
        :param rng_key: a PRNG state to be used by JAX backend to generate
            random samples
        :type rng_key: None or JAX's random.PRNGKey
        """
        assert self.output == Real
        sampled_vars = _convert_reduced_vars(sampled_vars, self.inputs)
        sampled_vars = frozenset(v.name for v in sampled_vars)
        assert isinstance(sampled_vars, frozenset)
        if sample_inputs is None:
            sample_inputs = OrderedDict()
        assert isinstance(sample_inputs, OrderedDict)
        if sampled_vars.isdisjoint(self.inputs):
            return self

        result = instrument.debug_logged(self._sample)(
            sampled_vars, sample_inputs, rng_key
        )
        return result

    def _sample(self, sampled_vars, sample_inputs, rng_key):
        """
        Internal method to draw samples.
        This should be overridden by subclasses.
        """
        assert self.output == Real
        assert isinstance(sampled_vars, frozenset)
        assert isinstance(sample_inputs, OrderedDict)
        if sampled_vars.isdisjoint(self.inputs):
            return self
        raise ValueError("Cannot sample from a {}".format(type(self).__name__))

    def align(self, names):
        """
        Align this funsor to match given ``names``.
        This is mainly useful in preparation for extracting ``.data``
        of a :class:`funsor.tensor.Tensor`.

        :param tuple names: A tuple of strings representing all names
            but in a new order.
        :return: A permuted funsor equivalent to self.
        :rtype: Funsor
        """
        assert isinstance(names, tuple)
        if not names or names == tuple(self.inputs):
            return self
        return Align(self, names)

    def eager_subs(self, subs):
        """
        Internal substitution function. This relies on the user-facing
        :meth:`__call__` method to coerce non-Funsors to Funsors. Once all
        inputs are Funsors, :meth:`eager_subs` implementations can recurse to
        call :class:`Subs`.
        """
        return None  # defer to default implementation

    def eager_unary(self, op):
        return None  # defer to default implementation

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        if not reduced_vars:
            return self

        return None  # defer to default implementation

    def sequential_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        if not reduced_vars:
            return self

        # Try to sum out integer scalars. This is mainly useful for testing,
        # since reduction is more efficiently implemented by Tensor.
        eager_vars = []
        lazy_vars = []
        for k in reduced_vars:
            if isinstance(self.inputs[k].dtype, int) and not self.inputs[k].shape:
                eager_vars.append(k)
            else:
                lazy_vars.append(k)
        if eager_vars:
            result = None
            for values in itertools.product(*(self.inputs[k] for k in eager_vars)):
                subs = dict(zip(eager_vars, values))
                result = self(**subs) if result is None else op(result, self(**subs))
            if lazy_vars:
                result = Reduce(op, result, frozenset(lazy_vars))
            return result

        return None  # defer to default implementation

    def moment_matching_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        if not reduced_vars:
            return self

        return None  # defer to default implementation

    # The following methods conform to a standard array/tensor interface.

    def __invert__(self):
        return Unary(ops.invert, self)

    def __pos__(self):
        return Unary(ops.pos, self)

    def __neg__(self):
        return Unary(ops.neg, self)

    def abs(self):
        return Unary(ops.abs, self)

    def atanh(self):
        return Unary(ops.atanh, self)

    def sqrt(self):
        return Unary(ops.sqrt, self)

    def exp(self):
        return Unary(ops.exp, self)

    def log(self):
        return Unary(ops.log, self)

    def log1p(self):
        return Unary(ops.log1p, self)

    def sigmoid(self):
        return Unary(ops.sigmoid, self)

    def tanh(self):
        return Unary(ops.tanh, self)

    def reshape(self, shape):
        return Unary(ops.ReshapeOp(shape), self)

    # The following reductions are treated as Unary ops because they
    # reduce over output shape while preserving all inputs.
    # To reduce over inputs, instead call .reduce(op, reduced_vars).

    def all(self, axis=None, keepdims=False):
        return Unary(ops.AllOp(axis, keepdims), self)

    def any(self, axis=None, keepdims=False):
        return Unary(ops.AnyOp(axis, keepdims), self)

    def argmax(self, axis=None, keepdims=False):
        return Unary(ops.ArgmaxOp(axis, keepdims), self)

    def argmin(self, axis=None, keepdims=False):
        return Unary(ops.ArgminOp(axis, keepdims), self)

    def max(self, axis=None, keepdims=False):
        return Unary(ops.AmaxOp(axis, keepdims), self)

    def min(self, axis=None, keepdims=False):
        return Unary(ops.AminOp(axis, keepdims), self)

    def sum(self, axis=None, keepdims=False):
        return Unary(ops.SumOp(axis, keepdims), self)

    def prod(self, axis=None, keepdims=False):
        return Unary(ops.ProdOp(axis, keepdims), self)

    def logsumexp(self, axis=None, keepdims=False):
        return Unary(ops.LogsumexpOp(axis, keepdims), self)

    def mean(self, axis=None, keepdims=False):
        return Unary(ops.MeanOp(axis, keepdims), self)

    def std(self, axis=None, ddof=0, keepdims=False):
        return Unary(ops.StdOp(axis, ddof, keepdims), self)

    def var(self, axis=None, ddof=0, keepdims=False):
        return Unary(ops.VarOp(axis, ddof, keepdims), self)

    def __add__(self, other):
        return Binary(ops.add, self, to_funsor(other))

    def __radd__(self, other):
        return Binary(ops.add, self, to_funsor(other))

    def __sub__(self, other):
        return Binary(ops.sub, self, to_funsor(other))

    def __rsub__(self, other):
        return Binary(ops.sub, to_funsor(other), self)

    def __mul__(self, other):
        return Binary(ops.mul, self, to_funsor(other))

    def __rmul__(self, other):
        return Binary(ops.mul, self, to_funsor(other))

    def __truediv__(self, other):
        return Binary(ops.truediv, self, to_funsor(other))

    def __rtruediv__(self, other):
        return Binary(ops.truediv, to_funsor(other), self)

    def __floordiv__(self, other):
        return Binary(ops.floordiv, self, to_funsor(other))

    def __rfloordiv__(self, other):
        return Binary(ops.floordiv, to_funsor(other), self)

    def __matmul__(self, other):
        return Binary(ops.matmul, self, to_funsor(other))

    def __rmatmul__(self, other):
        return Binary(ops.matmul, to_funsor(other), self)

    def __mod__(self, other):
        return Binary(ops.mod, self, to_funsor(other))

    def __rmod__(self, other):
        return Binary(ops.mod, to_funsor(other), self)

    def __lshift__(self, other):
        return Binary(ops.lshift, self, to_funsor(other))

    def __rlshift__(self, other):
        return Binary(ops.lshift, to_funsor(other), self)

    def __rshift__(self, other):
        return Binary(ops.rshift, self, to_funsor(other))

    def __rrshift__(self, other):
        return Binary(ops.rshift, to_funsor(other), self)

    def __pow__(self, other):
        return Binary(ops.pow, self, to_funsor(other))

    def __rpow__(self, other):
        return Binary(ops.pow, to_funsor(other), self)

    def __and__(self, other):
        return Binary(ops.and_, self, to_funsor(other))

    def __rand__(self, other):
        return Binary(ops.and_, self, to_funsor(other))

    def __or__(self, other):
        return Binary(ops.or_, self, to_funsor(other))

    def __ror__(self, other):
        return Binary(ops.or_, self, to_funsor(other))

    def __xor__(self, other):
        return Binary(ops.xor, self, to_funsor(other))

    def __eq__(self, other):
        return Binary(ops.eq, self, to_funsor(other))

    def __ne__(self, other):
        return Binary(ops.ne, self, to_funsor(other))

    def __lt__(self, other):
        return Binary(ops.lt, self, to_funsor(other))

    def __le__(self, other):
        return Binary(ops.le, self, to_funsor(other))

    def __gt__(self, other):
        return Binary(ops.gt, self, to_funsor(other))

    def __ge__(self, other):
        return Binary(ops.ge, self, to_funsor(other))

    def __getitem__(self, other):
        """
        Helper to desugar into either ops.getitem (for advanced indexing
        involving Funsors as indices) or ops.getslice (for simple indexing
        involving only integers, slices, None, and Ellipsis).
        """
        if type(other) is not tuple:
            if isinstance(other, ops.getslice.supported_types):
                return ops.getslice(self, other)
            other = to_funsor(other, Bint[self.output.shape[0]])
            return Binary(ops.getitem, self, other)

        # Handle complex slicing operations involving no funsors.
        if all(isinstance(part, ops.getslice.supported_types) for part in other):
            return ops.getslice(self, other)

        # Handle Ellipsis slicing.
        if any(part is Ellipsis for part in other):
            left, right = parse_ellipsis(other)
            missing = len(self.output.shape) - len(left) - len(right)
            assert missing >= 0
            middle = [slice(None)] * missing
            other = tuple(left + middle + right)

        # Handle each slice separately.
        result = self
        offset = 0
        for part in other:
            if part is None:
                raise NotImplementedError("TODO")
            if isinstance(part, slice):
                if part != slice(None):
                    raise NotImplementedError("TODO support nontrivial slicing")
                offset += 1
            else:
                part = to_funsor(part, Bint[result.output.shape[offset]])
                result = Binary(GetitemOp(offset), result, part)
        return result


@quote.register(Funsor)
def _(arg, indent, out):
    name = type(arg).__name__
    if type(arg).__module__ in [
        "funsor.torch.distributions",
        "funsor.jax.distributions",
    ]:
        name = "dist." + name
    out.append((indent, name + "("))
    for value in arg._ast_values[:-1]:
        quote.inplace(value, indent + 1, out)
        i, line = out[-1]
        out[-1] = i, line + ","
    for value in arg._ast_values[-1:]:
        quote.inplace(value, indent + 1, out)
        i, line = out[-1]
        out[-1] = i, line + ")"


interpreter.children.register(Funsor)(interpreter.children_funsor)


@singledispatch
def to_funsor(x, output=None, dim_to_name=None, **kwargs):
    """
    Convert to a :class:`Funsor` .
    Only :class:`Funsor` s and scalars are accepted.

    :param x: An object.
    :param funsor.domains.Domain output: An optional output hint.
    :param OrderedDict dim_to_name: An optional mapping from negative batch dimensions to name strings.
    :return: A Funsor equivalent to ``x``.
    :rtype: Funsor
    :raises: ValueError
    """
    raise ValueError("Cannot convert to Funsor: {}".format(repr(x)))


@to_funsor.register(Funsor)
def funsor_to_funsor(x, output=None, dim_to_name=None):
    if output is not None and x.output != output:
        raise ValueError("Output mismatch: {} vs {}".format(x.output, output))
    if dim_to_name is not None:
        bint_names = {
            name for name, domain in x.inputs.items() if domain.dtype != "real"
        }
        if not bint_names.issubset(dim_to_name.values()):
            raise ValueError("Inputs mismatch: {} vs {}".format(x.inputs, dim_to_name))
    return x


@singledispatch
def to_data(x, name_to_dim=None, **kwargs):
    """
    Extract a python object from a :class:`Funsor`.

    Raises a ``ValueError`` if free variables remain or if the funsor is lazy.

    :param x: An object, possibly a :class:`Funsor`.
    :param OrderedDict name_to_dim: An optional inputs hint.
    :return: A non-funsor equivalent to ``x``.
    :raises: ValueError if any free variables remain.
    :raises: PatternMissingError if funsor is not fully evaluated.
    """
    return x


@to_data.register(Funsor)
def _to_data_funsor(x, name_to_dim=None):
    if name_to_dim is None and x.inputs:
        raise ValueError(
            "cannot convert {} to data due to lazy inputs: {}".format(
                type(x), set(x.inputs)
            )
        )
    raise PatternMissingError("cannot convert to a non-Funsor: {}".format(repr(x)))


class Variable(Funsor):
    """
    Funsor representing a single free variable.

    :param str name: A variable name.
    :param funsor.domains.Domain output: A domain.
    """

    def __init__(self, name, output):
        inputs = OrderedDict([(name, output)])
        fresh = frozenset({name})
        super(Variable, self).__init__(inputs, output, fresh)
        self.name = name

    def __repr__(self):
        return "Variable({}, {})".format(repr(self.name), repr(self.output))

    def __str__(self):
        return self.name

    def eager_subs(self, subs):
        assert len(subs) == 1 and subs[0][0] == self.name
        return subs[0][1]


@to_funsor.register(str)
def name_to_funsor(name, output=None):
    if output is None:
        raise ValueError("Missing output: {}".format(name))
    return Variable(name, output)


class SubsMeta(FunsorMeta):
    """
    Wrapper to call :func:`to_funsor` and check types.
    """

    def __call__(cls, arg, subs):
        subs = tuple(
            (k, to_funsor(v, arg.inputs[k])) for k, v in subs if k in arg.inputs
        )
        return super().__call__(arg, subs)


class Subs(Funsor, metaclass=SubsMeta):
    """
    Lazy substitution of the form ``x(u=y, v=z)``.

    :param Funsor arg: A funsor being substituted into.
    :param tuple subs: A tuple of ``(name, value)`` pairs, where ``name`` is a
        string and ``value`` can be coerced to a :class:`Funsor` via
        :func:`to_funsor`.
    """

    def __init__(self, arg, subs):
        assert isinstance(arg, Funsor)
        assert isinstance(subs, tuple)
        for key, value in subs:
            assert isinstance(key, str)
            assert key in arg.inputs
            assert isinstance(value, Funsor)
        inputs = arg.inputs.copy()
        for key, value in subs:
            del inputs[key]
        for key, value in subs:
            inputs.update(value.inputs)
        fresh = frozenset()
        bound = {key: value.output for key, value in subs}
        super(Subs, self).__init__(inputs, arg.output, fresh, bound)
        self.arg = arg
        self.subs = OrderedDict(subs)

    def __repr__(self):
        return "{}({})".format(
            repr(self.arg), ", ".join(f"{k}={repr(v)}" for k, v in self.subs.items())
        )

    def __str__(self):
        return "{}({})".format(
            str(self.arg), ", ".join(f"{k}={str(v)}" for k, v in self.subs.items())
        )

    def _alpha_convert(self, alpha_subs):
        assert set(alpha_subs).issubset(self.bound)
        alpha_subs = {
            k: to_funsor(v, self.subs[k].output) for k, v in alpha_subs.items()
        }
        arg, subs = self._ast_values
        arg = substitute(arg, alpha_subs)
        subs = tuple((str(alpha_subs.get(k, k)), v) for k, v in subs)
        return arg, subs

    def _sample(self, sampled_vars, sample_inputs, rng_key=None):
        if any(k in sample_inputs for k, v in self.subs.items()):
            raise NotImplementedError("TODO alpha-convert")
        subs_sampled_vars = set()
        for name in sampled_vars:
            if name in self.arg.inputs:
                if any(name in v.inputs for k, v in self.subs.items()):
                    raise ValueError("Cannot sample")
                subs_sampled_vars.add(name)
            else:
                for k, v in self.subs.items():
                    if name in v.inputs:
                        subs_sampled_vars.add(k)
        subs_sampled_vars = frozenset(subs_sampled_vars)
        arg = self.arg._sample(subs_sampled_vars, sample_inputs, rng_key)
        return Subs(arg, tuple(self.subs.items()))


@lazy.register(Subs, Funsor, object)
@eager.register(Subs, Funsor, object)
def eager_subs_funsor(arg, subs):
    assert isinstance(subs, tuple)
    if not any(k in arg.inputs for k, v in subs):
        return arg
    return substitute(arg, subs)


@lazy.register(Subs, Subs, object)
@eager.register(Subs, Subs, object)
def eager_subs_subs(arg, subs):
    assert isinstance(subs, tuple)
    subs = tuple((k, v) for k, v in subs if k in arg.inputs)
    if not subs:
        return arg

    # Fuse substitutions.
    fused_subs = tuple((k, Subs(v, subs)) for k, v in arg.subs.items())
    fused_subs += subs
    return Subs(arg.arg, fused_subs)


@die.register(Subs, Funsor, tuple)
def die_subs(arg, subs):
    expr = reflect.interpret(Subs, arg, subs)
    raise NotImplementedError(f"Missing pattern for {repr(expr)}")


class Unary(Funsor):
    """
    Lazy unary operation.

    :param ~funsor.ops.Op op: A unary operator.
    :param Funsor arg: An argument.
    """

    def __init__(self, op, arg):
        assert callable(op)
        assert isinstance(arg, Funsor)
        output = find_domain(op, arg.output)
        super(Unary, self).__init__(arg.inputs, output)
        self.op = op
        self.arg = arg

    def __repr__(self):
        if self.op in _PREFIX:
            return "({}{})".format(_PREFIX[self.op], repr(self.arg))
        return super().__repr__()

    def __str__(self):
        if self.op in _PREFIX:
            return "({}{})".format(_PREFIX[self.op], str(self.arg))
        return super().__str__()


@eager.register(Unary, Op, Funsor)
def eager_unary(op, arg):
    return instrument.debug_logged(arg.eager_unary)(op)


@eager.register(Unary, AssociativeOp, Funsor)
def eager_unary(op, arg):
    if not arg.output.shape:
        return arg
    return instrument.debug_logged(arg.eager_unary)(op)


@die.register(Unary, Op, Funsor)
def die_unary(op, arg):
    expr = reflect.interpret(Unary, op, arg)
    raise NotImplementedError(f"Missing pattern for {repr(expr)}")


class Binary(Funsor):
    """
    Lazy binary operation.

    :param ~funsor.ops.Op op: A binary operator.
    :param Funsor lhs: A left hand side argument.
    :param Funsor rhs: A right hand side argument.
    """

    def __init__(self, op, lhs, rhs):
        assert callable(op)
        assert isinstance(lhs, Funsor)
        assert isinstance(rhs, Funsor)
        inputs = lhs.inputs.copy()
        inputs.update(rhs.inputs)
        output = find_domain(op, lhs.output, rhs.output)
        super(Binary, self).__init__(inputs, output)
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        if self.op in _INFIX:
            return "({} {} {})".format(repr(self.lhs), _INFIX[self.op], repr(self.rhs))
        return super().__repr__()

    def __str__(self):
        if self.op in _INFIX:
            return "({} {} {})".format(str(self.lhs), _INFIX[self.op], str(self.rhs))
        return super().__str__()


@die.register(Binary, Op, Funsor, Funsor)
def die_binary(op, lhs, rhs):
    expr = reflect.interpret(Binary, op, lhs, rhs)
    raise NotImplementedError(f"Missing pattern for {repr(expr)}")


class Reduce(Funsor):
    """
    Lazy reduction over multiple variables.

    The user-facing interface is the :meth:`Funsor.reduce` method.

    :param op: An associative operator.
    :type op: ~funsor.ops.AssociativeOp
    :param funsor arg: An argument to be reduced.
    :param frozenset reduced_vars: A set of variables over which to reduce.
    """

    def __init__(self, op, arg, reduced_vars):
        assert isinstance(op, AssociativeOp)
        assert isinstance(arg, Funsor)
        assert isinstance(reduced_vars, frozenset)
        assert all(isinstance(v, Variable) for v in reduced_vars)
        reduced_names = frozenset(v.name for v in reduced_vars)
        inputs = OrderedDict(
            (k, v) for k, v in arg.inputs.items() if k not in reduced_names
        )
        output = arg.output
        fresh = frozenset()
        bound = {var.name: var.output for var in reduced_vars}
        super(Reduce, self).__init__(inputs, output, fresh, bound)
        self.op = op
        self.arg = arg
        self.reduced_vars = reduced_vars

    def __repr__(self):
        assert self.reduced_vars
        if self.reduced_vars == self.arg.input_vars:
            return f"{repr(self.arg)}.reduce({self.op.__name__})"
        rvars = [
            f'"{v.name}"' if v in self.arg.input_vars else repr(v)
            for v in self.reduced_vars
        ]
        return "{}.reduce({}, {{{}}})".format(
            repr(self.arg), self.op.__name__, ", ".join(rvars)
        )

    def __str__(self):
        assert self.reduced_vars
        if self.reduced_vars == self.arg.input_vars:
            return f"{str(self.arg)}.reduce({self.op.__name__})"
        rvars = [
            f'"{v.name}"' if v in self.arg.input_vars else repr(v)
            for v in self.reduced_vars
        ]
        return "{}.reduce({}, {{{}}})".format(
            str(self.arg), self.op.__name__, ", ".join(rvars)
        )


def _reduce_unrelated_vars(op, arg, reduced_vars):
    factor_vars = reduced_vars - arg.input_vars
    if factor_vars:
        reduced_vars = reduced_vars & arg.input_vars
        multiplicity = reduce(
            ops.mul,
            [
                v.output.size**v.output.num_elements
                for v in factor_vars
                if v.dtype != "real"
            ],
        )
        for add_op, mul_op in ops.DISTRIBUTIVE_OPS:
            if add_op is op:
                arg = mul_op(arg, multiplicity).reduce(op, reduced_vars)
                return arg, None
        raise NotImplementedError(f"Cannot reduce {op}")
    return arg, frozenset(v.name for v in reduced_vars)


@lazy.register(Reduce, AssociativeOp, Funsor, frozenset)
def lazy_reduce(op, arg, reduced_vars):
    new_arg, new_reduced_vars = _reduce_unrelated_vars(op, arg, reduced_vars)
    if new_reduced_vars is None:
        return new_arg
    if new_arg is arg:
        return None
    return new_arg.reduce(op, new_reduced_vars)


@eager.register(Reduce, AssociativeOp, Funsor, frozenset)
def eager_reduce(op, arg, reduced_vars):
    arg, reduced_vars = _reduce_unrelated_vars(op, arg, reduced_vars)
    if reduced_vars is None:
        return arg
    return instrument.debug_logged(arg.eager_reduce)(op, reduced_vars)


@sequential.register(Reduce, AssociativeOp, Funsor, frozenset)
def sequential_reduce(op, arg, reduced_vars):
    arg, reduced_vars = _reduce_unrelated_vars(op, arg, reduced_vars)
    if reduced_vars is None:
        return arg
    return instrument.debug_logged(arg.sequential_reduce)(op, reduced_vars)


@moment_matching.register(Reduce, AssociativeOp, Funsor, frozenset)
def moment_matching_reduce(op, arg, reduced_vars):
    arg, reduced_vars = _reduce_unrelated_vars(op, arg, reduced_vars)
    if reduced_vars is None:
        return arg
    return instrument.debug_logged(arg.moment_matching_reduce)(op, reduced_vars)


@die.register(Reduce, Op, Funsor, frozenset)
def die_reduce(op, arg, reduced_vars):
    expr = reflect.interpret(Reduce, op, arg, reduced_vars)
    raise NotImplementedError(f"Missing pattern for {repr(expr)}")


class Scatter(Funsor):
    """
    Transpose of structurally linear :class:`Subs`, followed by
    :class:`Reduce`.

    For injective scatter operations this should satisfy the equation::

        if destin = Scatter(op, subs, source, frozenset())
        then source = Subs(destin, subs)

    The ``reduced_vars`` is merely for computational efficiency, and could
    always be split out into a separate ``.reduce()``.  For example in the
    following equation, the left hand side uses much less memory than the
    right hand side::

        Scatter(op, subs, source, reduced_vars) ==
          Scatter(op, subs, source, frozenset()).reduce(op, reduced_vars)

    .. warning:: This is currently implemented only for injective scatter
        operations. In particular, this does not allow accumulation behavior
        like scatter-add.

    .. note:: ``Scatter(ops.add, ...)`` is the funsor analog of
        ``numpy.add.at()`` or :func:`torch.index_put` or
        :func:`jax.lax.scatter_add`. For injective substitutions,
        ``Scatter(ops.add, ...)`` is roughly equivalent to the tensor
        operation::

            result = zeros(...)  # since zero is the additive unit
            result[subs] = source

    :param AssociativeOp op: An op. The unit of this op will be used as
        default value.
    :param tuple subs: A substitution.
    :param Funsor source: A source for data to be scattered from.
    :param frozenset reduced_vars: A set of variables over which to reduce.
    """

    def __init__(self, op, subs, source, reduced_vars):
        assert isinstance(op, AssociativeOp)
        assert isinstance(subs, tuple)
        assert len(subs) == len(set(key for key, value in subs))
        assert isinstance(source, Funsor)
        assert isinstance(reduced_vars, frozenset)
        assert all(isinstance(v, Variable) for v in reduced_vars)
        reduced_names = frozenset(v.name for v in reduced_vars)

        # First compute inputs of the pure-scatter op with no reduction.
        inputs = OrderedDict()
        for key, value in subs:
            assert isinstance(key, str)
            assert isinstance(value, Funsor)
            assert key not in source.inputs
            assert key not in reduced_names
            for k, d in value.inputs.items():
                # These are "batch" inputs and should be left of subs keys.
                d2 = inputs.setdefault(k, d)
                assert d2 == d
        for k, d in source.inputs.items():
            # These are "batch" inputs and should be left of subs keys.
            d2 = inputs.setdefault(k, d)
            assert d2 == d
        for key, value in subs:
            assert key not in inputs
            # These are "event" inputs and should be right of "batch" inputs.
            inputs[key] = value.output

        # Then narrow these down to the fused scatter-reduce op.
        inputs = OrderedDict(
            (k, d) for k, d in inputs.items() if k not in reduced_names
        )
        fresh = frozenset(key for key, value in subs)
        bound = {v.name: v.output for v in reduced_vars}
        super().__init__(inputs, source.output, fresh, bound)
        self.op = op
        self.subs = subs
        self.source = source
        self.reduced_vars = reduced_vars

    def eager_subs(self, subs):
        subs = OrderedDict(subs)
        new_subs = []
        for name, sub in self.subs:
            if name in subs and isinstance(subs[name], Variable):
                new_subs.append((subs[name].name, sub))
            else:
                new_subs.append((name, sub))
        return Scatter(self.op, tuple(new_subs), self.source, self.reduced_vars)


class Approximate(Funsor):
    """
    Interpretation-specific approximation wrt a set of variables.

    The default eager interpretation should be exact.
    The user-facing interface is the :meth:`Funsor.approximate` method.

    :param op: An associative operator.
    :type op: ~funsor.ops.AssociativeOp
    :param Funsor model: An exact funsor depending on ``approx_vars``.
    :param Funsor guide: A proposal funsor guiding optional approximation.
    :param frozenset approx_vars: A set of variables over which to approximate.
    """

    def __init__(self, op, model, guide, approx_vars):
        assert isinstance(op, AssociativeOp)
        assert isinstance(model, Funsor)
        assert isinstance(guide, Funsor)
        assert model.output is guide.output
        assert isinstance(approx_vars, frozenset), approx_vars
        inputs = model.inputs.copy()
        inputs.update(guide.inputs)
        output = model.output
        fresh = frozenset(v.name for v in approx_vars)
        bound = {v.name: v.output for v in approx_vars}
        super().__init__(inputs, output, fresh, bound)
        self.op = op
        self.model = model
        self.guide = guide
        self.approx_vars = approx_vars

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {k: to_funsor(v, self.bound[k]) for k, v in alpha_subs.items()}
        op, model, guide, approx_vars = super()._alpha_convert(alpha_subs)
        approx_vars = frozenset(alpha_subs.get(var.name, var) for var in approx_vars)
        return op, model, guide, approx_vars


@eager.register(Approximate, AssociativeOp, Funsor, Funsor, frozenset)
def eager_approximate(op, model, guide, approx_vars):
    return model  # exact


class NumberMeta(FunsorMeta):
    """
    Wrapper to fill in default ``dtype``.
    """

    def __call__(cls, data, dtype=None):
        if dtype is None:
            dtype = "real"
        return super(NumberMeta, cls).__call__(data, dtype)


class Number(Funsor, metaclass=NumberMeta):
    """
    Funsor backed by a Python number.

    :param numbers.Number data: A python number.
    :param dtype: A nonnegative integer or the string "real".
    """

    def __init__(self, data, dtype=None):
        assert isinstance(data, numbers.Number)
        if isinstance(dtype, int):
            data = type(dtype)(data)
            if dtype != 2:  # booleans have bitwise interpretation
                assert 0 <= data and data < dtype
        else:
            assert isinstance(dtype, str) and dtype == "real"
            data = float(data)
        inputs = OrderedDict()
        output = Array[dtype, ()]
        super(Number, self).__init__(inputs, output)
        self.data = data

    def __repr__(self):
        if self.dtype == "real":
            return f"Number({str(self.data)})"
        else:
            return f"Number({str(self.data)}, {self.dtype})"

    def __str__(self):
        return str(self.data)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def __bool__(self):
        return bool(self.data)

    def item(self):
        return self.data

    def eager_unary(self, op):
        dtype = find_domain(op, self.output).dtype
        return Number(op(self.data), dtype)


@to_funsor.register(numbers.Number)
def number_to_funsor(x, output=None, dim_to_name=None):
    if output is None:
        return Number(x)
    if output.shape:
        raise ValueError("Cannot create Number with shape {}".format(output.shape))
    return Number(x, output.dtype)


@to_data.register(Number)
def _to_data_number(x, name_to_dim=None):
    return x.data


@eager.register(Binary, Op, Number, Number)
def eager_binary_number_number(op, lhs, rhs):
    data = op(lhs.data, rhs.data)
    output = find_domain(op, lhs.output, rhs.output)
    dtype = output.dtype
    return Number(data, dtype)


class SliceMeta(FunsorMeta):
    """
    Wrapper to fill in ``start``, ``stop``, ``step``, ``dtype`` following
    Python conventions.
    """

    def __call__(cls, name, *args, **kwargs):
        start = 0
        step = 1
        dtype = None
        if len(args) == 1:
            stop = args[0]
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 2:
            start, stop = args
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 3:
            start, stop, step = args
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 4:
            start, stop, step, dtype = args
        else:
            raise ValueError
        if step <= 0:
            raise ValueError
        stop = min(dtype, max(start, stop))
        return super().__call__(name, start, stop, step, dtype)


class Slice(Funsor, metaclass=SliceMeta):
    """
    Symbolic representation of a Python :py:class:`slice` object.

    :param str name: A name for the new slice dimension.
    :param int start:
    :param int stop:
    :param int step: Three args following :py:class:`slice` semantics.
    :param int dtype: An optional bounded integer type of this slice.
    """

    def __init__(self, name, start, stop, step, dtype):
        assert isinstance(name, str)
        assert isinstance(start, int) and start >= 0
        assert isinstance(stop, int) and stop >= start
        assert isinstance(step, int) and step > 0
        assert isinstance(dtype, int)
        size = max(0, (stop + step - 1 - start) // step)
        inputs = OrderedDict([(name, Bint[size])])
        output = Bint[dtype]
        fresh = frozenset({name})
        super().__init__(inputs, output, fresh)
        self.name = name
        self.slice = slice(start, stop, step)

    def eager_subs(self, subs):
        assert len(subs) == 1 and subs[0][0] == self.name
        index = subs[0][1]

        if isinstance(index, Variable):
            name = index.name
            return Slice(
                name, self.slice.start, self.slice.stop, self.slice.step, self.dtype
            )
        elif isinstance(index, Number):
            data = self.slice.start + self.slice.step * index.data
            return Number(data, self.output.dtype)
        elif type(index).__name__ == "Tensor":  # avoid importing funsor.tensor.Tensor
            data = self.slice.start + self.slice.step * index.data
            return type(index)(data, index.inputs, self.output.dtype)
        elif isinstance(index, Slice):
            name = index.name
            start = self.slice.start + self.slice.step * index.slice.start
            step = self.slice.step * index.slice.step
            return Slice(name, start, self.slice.stop, step, self.dtype)
        else:
            raise NotImplementedError(
                "TODO support substitution of {} into Slice".format(type(index))
            )


@to_funsor.register(slice)
def slice_to_funsor(s, output=None, dim_to_name=None):
    if not isinstance(output, BintType):
        raise ValueError("Incompatible slice output: {output}")
    start, stop, step = parse_slice(s, output.size)
    i = Variable("slice", output)
    return Lambda(i, Slice("slice", start, stop, step, output.size))


class Align(Funsor):
    """
    Lazy call to ``.align(...)``.

    :param Funsor arg: A funsor to align.
    :param tuple names: A tuple of input names whose order to follow.
    """

    def __init__(self, arg, names):
        assert isinstance(arg, Funsor)
        assert isinstance(names, tuple)
        assert all(isinstance(name, str) for name in names)
        assert all(name in arg.inputs for name in names)
        inputs = OrderedDict((name, arg.inputs[name]) for name in names)
        inputs.update(arg.inputs)
        output = arg.output
        fresh = frozenset()  # TODO get this right
        bound = {}
        super(Align, self).__init__(inputs, output, fresh, bound)
        self.arg = arg

    def align(self, names):
        return self.arg.align(names)

    def eager_unary(self, op):
        return Unary(op, self.arg)

    def eager_reduce(self, op, reduced_vars):
        return self.arg.reduce(op, reduced_vars)


@eager.register(Align, Funsor, tuple)
def eager_align(arg, names):
    if not frozenset(names) == frozenset(arg.inputs.keys()):
        # assume there's been a substitution and this align is no longer valid
        return arg
    return None


@eager.register(Binary, Op, Align, Funsor)
def eager_binary_align_funsor(op, lhs, rhs):
    return Binary(op, lhs.arg, rhs)


@eager.register(Binary, Op, Funsor, Align)
def eager_binary_funsor_align(op, lhs, rhs):
    return Binary(op, lhs, rhs.arg)


@eager.register(Binary, Op, Align, Align)
def eager_binary_align_align(op, lhs, rhs):
    return Binary(op, lhs.arg, rhs.arg)


class Finitary(Funsor):
    def __init__(self, op, args):
        assert isinstance(op, ops.Op)
        assert isinstance(args, tuple)
        assert all(isinstance(v, Funsor) for v in args)
        inputs = OrderedDict()
        for arg in args:
            inputs.update(arg.inputs)
        output = find_domain(op, tuple(arg.output for arg in args))
        super().__init__(inputs, output)
        self.op = op
        self.args = args


class Stack(Funsor):
    """
    Stack of funsors along a new input dimension.

    :param str name: The name of the new input variable along which to stack.
    :param tuple parts: A tuple of Funsors of homogenous output domain.
    """

    def __init__(self, name, parts):
        assert isinstance(name, str)
        assert isinstance(parts, tuple)
        assert parts
        assert not any(name in x.inputs for x in parts)
        assert len(set(x.output for x in parts)) == 1
        output = parts[0].output
        domain = Bint[len(parts)]
        inputs = OrderedDict([(name, domain)])
        for x in parts:
            inputs.update(x.inputs)
        fresh = frozenset({name})
        super().__init__(inputs, output, fresh)
        self.name = name
        self.parts = parts

    def eager_subs(self, subs):
        assert isinstance(subs, tuple) and len(subs) == 1 and subs[0][0] == self.name
        index = subs[0][1]

        # Try to eagerly select an index.
        if index.output == Bint[len(self.parts)]:
            if isinstance(index, Number):
                # Select a single part.
                return self.parts[index.data]
            elif isinstance(index, Variable):
                # Rename the stacking dimension.
                parts = self.parts
                return Stack(index.name, parts)
            elif isinstance(index, Slice):
                parts = self.parts[index.slice]
                return Stack(index.name, parts)
            else:
                raise NotImplementedError("TODO support advanced indexing in Stack")
        else:
            raise NotImplementedError("TODO support slicing in Stack")

    def eager_reduce(self, op, reduced_vars):
        parts = self.parts
        if self.name in reduced_vars:
            reduced_vars -= frozenset([self.name])
            if reduced_vars:
                parts = tuple(x.reduce(op, reduced_vars) for x in parts)
            return reduce(op, parts)
        parts = tuple(x.reduce(op, reduced_vars) for x in parts)
        return Stack(self.name, parts)


@eager.register(Stack, str, tuple)
def eager_stack(name, parts):
    return eager_stack_homogeneous(name, *parts)


@dispatch(str, Variadic[Funsor])
def eager_stack_homogeneous(name, *parts):
    return None  # defer to default implementation


class CatMeta(FunsorMeta):
    """
    Wrapper to fill in default value for ``part_name``.
    """

    def __call__(cls, name, parts, part_name=None):
        if part_name is None:
            part_name = name
        return super().__call__(name, parts, part_name)


class Cat(Funsor, metaclass=CatMeta):
    """
    Concatenate funsors along an existing input dimension.

    :param str name: The name of the input variable along which to concatenate.
    :param tuple parts: A tuple of Funsors of homogenous output domain.
    """

    def __init__(self, name, parts, part_name=None):
        assert isinstance(name, str)
        assert isinstance(parts, tuple)
        assert isinstance(part_name, str)
        assert parts
        for part in parts:
            assert part_name in part.inputs, (part_name, part.inputs)
        if part_name != name:
            assert not any(name in x.inputs for x in parts)
        assert len(set(x.output for x in parts)) == 1
        output = parts[0].output
        inputs = OrderedDict()
        for x in parts:
            inputs.update(x.inputs)
        del inputs[part_name]
        inputs[name] = Bint[sum(x.inputs[part_name].size for x in parts)]
        fresh = frozenset({name})
        bound = {part_name: x.inputs[part_name]}
        super().__init__(inputs, output, fresh, bound)
        self.name = name
        self.parts = parts
        self.part_name = part_name

    def _alpha_convert(self, alpha_subs):
        assert len(alpha_subs) == 1
        part_name = alpha_subs[self.part_name]
        parts = tuple(
            substitute(
                p, {self.part_name: to_funsor(part_name, p.inputs[self.part_name])}
            )
            for p in self.parts
        )
        return self.name, parts, part_name

    def eager_subs(self, subs):
        assert len(subs) == 1 and subs[0][0] == self.name
        value = subs[0][1]

        if isinstance(value, Variable):
            return Cat(value.name, self.parts, self.part_name)
        elif isinstance(value, Number):
            n = value.data
            for part in self.parts:
                size = part.inputs[self.part_name].size
                if n < size:
                    return part(**{self.part_name: n})
                n -= size
            assert False
        elif isinstance(value, Slice):
            start, stop, step = value.slice.start, value.slice.stop, value.slice.step
            new_parts = []
            pos = 0
            for part in self.parts:
                psize = part.inputs[self.part_name].size
                if step > 1:
                    pstart = ((pos - start) // step) * step - (pos - start)
                    pstart = pstart + step if pstart < 0 else pstart
                else:
                    pstart = max(start - pos, 0)
                pstop = min(pos + psize, stop) - pos

                if not (pstart >= pstop or pos >= stop or pos + psize <= start):
                    pslice = Slice(self.part_name, pstart, pstop, step, psize)
                    part = part(**{self.part_name: pslice})
                    new_parts.append(part)

                pos += psize

            return Cat(self.name, tuple(new_parts), self.part_name)
        else:
            raise NotImplementedError(
                "TODO implement Cat.eager_subs for {}".format(type(value))
            )


@eager.register(Cat, str, tuple, str)
def eager_cat(name, parts, part_name):
    if len(parts) == 1:
        return parts[0](**{part_name: name})
    return eager_cat_homogeneous(name, part_name, *parts)


@dispatch(str, str, Variadic[Funsor])
def eager_cat_homogeneous(name, part_name, *parts):
    return None  # defer to default implementation


class Lambda(Funsor):
    """
    Lazy inverse to ``ops.getitem``.

    This is useful to simulate higher-order functions of integers
    by representing those functions as arrays.

    :param Variable var: A variable to bind.
    :param funsor expr: A funsor.
    """

    def __init__(self, var, expr):
        assert isinstance(var, Variable)
        assert isinstance(var.dtype, int)
        assert isinstance(expr, Funsor)
        inputs = expr.inputs.copy()
        inputs.pop(var.name, None)
        shape = (var.dtype,) + expr.output.shape
        output = Array[expr.dtype, shape]
        fresh = frozenset()
        bound = {var.name: var.output}
        super(Lambda, self).__init__(inputs, output, fresh, bound)
        self.var = var
        self.expr = expr


@eager.register(Binary, GetitemOp, Lambda, (Funsor, Align))
def eager_getitem_lambda(op, lhs, rhs):
    offset = op.defaults["offset"]
    if offset == 0:
        return Subs(lhs.expr, ((lhs.var.name, rhs),))
    expr = GetitemOp(offset - 1)(lhs.expr, rhs)
    return Lambda(lhs.var, expr)


@eager.register(Unary, ops.GetsliceOp, Lambda)
def eager_getslice_lambda(op, x):
    index = normalize_ellipsis(op.defaults["index"], len(x.shape))
    head, tail = index[0], index[1:]
    expr = x.expr
    if head != slice(None):
        expr = expr(**{x.var.name: head})
    if tail:
        expr = ops.getslice(expr, tail)
    if x.var.name in expr.inputs:  # dim is preserved, e.g. x[1:]
        return Lambda(x.var, expr)
    else:  # dim is eliminated, e.g. x[0]
        return expr


class Independent(Funsor):
    """
    Creates an independent diagonal distribution.

    This is equivalent to substitution followed by reduction::

        f = ...  # a batched distribution
        assert f.inputs['x_i'] == Reals[4, 5]
        assert f.inputs['i'] == Bint[3]

        g = Independent(f, 'x', 'i', 'x_i')
        assert g.inputs['x'] == Reals[3, 4, 5]
        assert 'x_i' not in g.inputs
        assert 'i' not in g.inputs

        x = Variable('x', Reals[3, 4, 5])
        g == f(x_i=x['i']).reduce(ops.add, 'i')

    :param Funsor fn: A funsor.
    :param str reals_var: The name of a real-tensor input.
    :param str bint_var: The name of a new batch input of ``fn``.
    :param diag_var: The name of a smaller-shape real input of ``fn``.
    """

    def __init__(self, fn, reals_var, bint_var, diag_var):
        assert isinstance(fn, Funsor)
        assert isinstance(reals_var, str)
        assert isinstance(bint_var, str)
        assert bint_var in fn.inputs, (bint_var, fn.inputs)
        assert isinstance(fn.inputs[bint_var].dtype, int)
        assert isinstance(diag_var, str)
        assert diag_var in fn.inputs
        inputs = fn.inputs.copy()
        diag_input = inputs.pop(diag_var)
        shape = (inputs.pop(bint_var).dtype,) + diag_input.shape
        assert reals_var not in inputs
        inputs[reals_var] = Array[diag_input.dtype, shape]
        fresh = frozenset({reals_var})
        bound = {bint_var: fn.inputs[bint_var], diag_var: fn.inputs[diag_var]}
        super(Independent, self).__init__(inputs, fn.output, fresh, bound)
        self.fn = fn
        self.reals_var = reals_var
        self.bint_var = bint_var
        self.diag_var = diag_var

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {k: to_funsor(v, self.fn.inputs[k]) for k, v in alpha_subs.items()}
        fn, reals_var, bint_var, diag_var = super()._alpha_convert(alpha_subs)
        bint_var = str(alpha_subs.get(bint_var, bint_var))
        diag_var = str(alpha_subs.get(diag_var, diag_var))
        return fn, reals_var, bint_var, diag_var

    def _sample(self, sampled_vars, sample_inputs, rng_key=None):
        if self.bint_var in sampled_vars or self.bint_var in sample_inputs:
            raise NotImplementedError("TODO alpha-convert")
        sampled_vars = frozenset(
            self.diag_var if v == self.reals_var else v for v in sampled_vars
        )
        fn = self.fn._sample(sampled_vars, sample_inputs, rng_key)
        return Independent(fn, self.reals_var, self.bint_var, self.diag_var)

    def eager_subs(self, subs):
        assert len(subs) == 1 and subs[0][0] == self.reals_var
        value = subs[0][1]

        # Handle simple renaming to preserve Independent.
        if isinstance(value, Variable):
            return Independent(self.fn, value.name, self.bint_var, self.diag_var)

        # Otherwise convert to a Reduce.
        result = Subs(self.fn, ((self.diag_var, value[self.bint_var]),))
        result = result.reduce(ops.add, self.bint_var)
        return result

    def mean(self):
        raise NotImplementedError("mean() not yet implemented for Independent")

    def variance(self):
        raise NotImplementedError("variance() not yet implemented for Independent")

    def entropy(self):
        raise NotImplementedError("entropy() not yet implemented for Independent")


@eager.register(Independent, Funsor, str, str, str)
def eager_independent_trivial(fn, reals_var, bint_var, diag_var):
    # compare to Independent.eager_subs
    if diag_var not in fn.inputs:
        return fn.reduce(ops.add, bint_var)
    return None


class Tuple(Funsor):
    """
    Funsor term representing tuples of other terms of possibly heterogeneous type.
    """

    def __init__(self, args):
        assert isinstance(args, tuple)
        assert all(isinstance(arg, Funsor) for arg in args)
        inputs = OrderedDict()
        for arg in args:
            inputs.update(arg.inputs)
        output = Product[tuple(arg.output for arg in args)]
        super().__init__(inputs, output)
        self.args = args

    def __iter__(self):
        for i in range(len(self.args)):
            yield self[i]


@to_funsor.register(tuple)
def tuple_to_funsor(args, output=None, dim_to_name=None):
    if not isinstance(output, ProductDomain):
        raise NotImplementedError("TODO")
    outputs = get_args(output)
    assert len(outputs) == len(args)
    funsor_args = tuple(
        to_funsor(arg, output=arg_output, dim_to_name=dim_to_name)
        for arg, arg_output in zip(args, outputs)
    )
    return Tuple(funsor_args)


@lazy.register(Binary, GetitemOp, Tuple, Number)
@eager.register(Binary, GetitemOp, Tuple, Number)
def eager_getitem_tuple(op, lhs, rhs):
    return op(lhs.args, rhs.data)


@lazy.register(Unary, ops.GetsliceOp, Tuple)
@eager.register(Unary, ops.GetsliceOp, Tuple)
def eager_getslice_tuple(op, x):
    index = op.defaults["index"]
    if isinstance(index, tuple):
        assert len(index) == 1
        index = index[0]
    if isinstance(index, int):
        return op(x.args)
    elif isinstance(index, slice):
        return Tuple(op(x.args))
    else:
        raise ValueError(index)


def _symbolic(inputs, output, fn):
    args, vargs, kwargs, defaults = getargspec(fn)
    assert not vargs
    assert not kwargs
    names = tuple(args)
    if isinstance(inputs, dict):
        args = tuple(Variable(name, inputs[name]) for name in names if name in inputs)
    else:
        args = tuple(Variable(name, domain) for (name, domain) in zip(names, inputs))
    assert len(args) == len(inputs)
    return to_funsor(fn(*args), output).align(names)


def symbolic(*signature):
    r"""
    Decorator to construct a symbolic :class:`Funsor` with one free
    :class:`Variable` per function arg. This can be used either with explicit
    types or with type hints::

        # Using type hints:
        @symbolic
        def xpyi(x: Real, y: Reals[3], i: Bint[3]):
            return x + y[i]

        # Using explicit type annotations:
        @symbolic(Real, Reals[3], Bint[3])
        def xpyi(x: Real, y: Reals[3], i: Bint[3]):
            return x + y[i]

    :param \*signature: A sequence if input domains.
    """
    if len(signature) == 1:
        fn = signature[0]
        if callable(fn) and not isinstance(fn, Domain):
            # Usage: @symbolic
            inputs = typing.get_type_hints(fn)
            output = inputs.pop("return", None)
            return _symbolic(inputs, output, fn)
    # Usage: @symbolic(Real, Reals[3], Bint[3])
    output = None
    # FIXME: what is inputs?
    return functools.partial(_symbolic, inputs, output)


# DEPRECATED
def of_shape(*shape):
    warnings.warn("@of_shape is deprecated, use @symbolic instead", DeprecationWarning)
    return symbolic(*shape)


AstStats = namedtuple("AstStats", ("size", "depth", "width"))


# Profiling helpers
@singledispatch
def _count_funsors(x):
    return 0


@_count_funsors.register(Funsor)
def _(x):
    return 1


@_count_funsors.register(tuple)
def _(x):
    return sum(map(_count_funsors, x))


@singledispatch
def _get_ast_stats(x):
    return AstStats(1, 1, 0)


@_get_ast_stats.register(Funsor)
def _(x):
    result = getattr(x, "_ast_stats", None)
    if result is None:
        size, depth, _ = _get_ast_stats(x._ast_values)
        width = _count_funsors(x._ast_values)
        result = x._ast_stats = AstStats(size + 1, depth + 1, width)
    return result


@_get_ast_stats.register(tuple)
def _(x):
    parts = list(map(_get_ast_stats, x))
    size = sum(p.size for p in parts)
    depth = max([0] + [p.depth for p in parts])
    return AstStats(size, depth, 0)


################################################################################
# Register Ops
################################################################################


@quote.register(Variable)
@quote.register(Number)
@quote.register(Slice)
def quote_inplace_oneline(arg, indent, out):
    out.append((indent, repr(arg)))


@quote.register(Unary)
@quote.register(Binary)
@quote.register(Reduce)
@quote.register(Stack)
@quote.register(Cat)
@quote.register(Lambda)
def quote_inplace_first_arg_on_first_line(arg, indent, out):
    line = "{}({},".format(type(arg).__name__, repr(arg._ast_values[0]))
    out.append((indent, line))
    for value in arg._ast_values[1:-1]:
        quote.inplace(value, indent + 1, out)
        i, line = out[-1]
        out[-1] = i, line + ","
    for value in arg._ast_values[-1:]:
        quote.inplace(value, indent + 1, out)
        i, line = out[-1]
        out[-1] = i, line + ")"


@ops.UnaryOp.subclass_register(Funsor)
def unary_funsor(cls, arg, *args, **kwargs):
    op = cls(*args, **kwargs)
    return Unary(op, arg)


@ops.BinaryOp.subclass_register(Funsor, Funsor)
def binary_funsor_funsor(cls, lhs, rhs, *args, **kwargs):
    op = cls(*args, **kwargs)
    return Binary(op, lhs, rhs)


@ops.BinaryOp.subclass_register(object, Funsor)
def binary_object_funsor(cls, lhs, rhs, *args, **kwargs):
    op = cls(*args, **kwargs)
    lhs = to_funsor(lhs)
    return Binary(op, lhs, rhs)


@ops.BinaryOp.subclass_register(Funsor, object)
def binary_funsor_object(cls, lhs, rhs, *args, **kwargs):
    op = cls(*args, **kwargs)
    rhs = to_funsor(rhs)
    return Binary(op, lhs, rhs)


@ops.TernaryOp.subclass_register(Funsor, Funsor, Funsor)
@ops.TernaryOp.subclass_register(Funsor, Funsor, object)
@ops.TernaryOp.subclass_register(Funsor, object, object)
@ops.TernaryOp.subclass_register(object, Funsor, object)
@ops.TernaryOp.subclass_register(object, object, Funsor)
def ternary_funsor_object(cls, x, y, z, *args, **kwargs):
    op = cls(*args, **kwargs)
    x = to_funsor(x)
    y = to_funsor(y)
    z = to_funsor(z)
    return Finitary(op, (x, y, z))


# FIXME allow some non-funsors
@ops.FinitaryOp.subclass_register(typing.Tuple[Funsor, ...])
def finitary_funsor(cls, arg, *args, **kwargs):
    op = cls(*args, **kwargs)
    return Finitary(op, arg)


__all__ = [
    "Approximate",
    "Binary",
    "Cat",
    "Funsor",
    "Independent",
    "Lambda",
    "Number",
    "Reduce",
    "Scatter",
    "Stack",
    "Slice",
    "Subs",
    "Unary",
    "Variable",
    "of_shape",
    "to_data",
    "to_funsor",
]
