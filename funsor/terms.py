from __future__ import absolute_import, division, print_function

import functools
import itertools
import math
import numbers
import re
from collections import Hashable, OrderedDict
from weakref import WeakValueDictionary

from multipledispatch import dispatch
from six import add_metaclass, integer_types
from six.moves import reduce

import funsor.interpreter as interpreter
import funsor.ops as ops
from funsor.domains import Domain, bint, find_domain, reals
from funsor.interpreter import dispatched_interpretation, interpret
from funsor.ops import AssociativeOp, GetitemOp, Op
from funsor.six import getargspec, singledispatch


def reflect(cls, *args):
    """
    Construct a funsor, populate ``._ast_values``, and cons hash.
    This is the only interpretation allowed to construct funsors.
    """
    cache_key = tuple(id(arg) if not isinstance(arg, Hashable) else arg for arg in args)
    if cache_key in cls._cons_cache:
        return cls._cons_cache[cache_key]
    result = super(FunsorMeta, cls).__call__(*args)
    result._ast_values = args
    cls._cons_cache[cache_key] = result
    return result


@dispatched_interpretation
def lazy(cls, *args):
    """
    Substitute eagerly but perform ops lazily.
    """
    result = lazy.dispatch(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


@dispatched_interpretation
def eager(cls, *args):
    """
    Eagerly execute ops with known implementations.
    """
    result = eager.dispatch(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


@dispatched_interpretation
def sequential(cls, *args):
    """
    Eagerly execute ops with known implementations; additonally execute
    vectorized ops sequentially if no known vectorized implementation exists.
    """
    result = sequential.dispatch(cls, *args)
    if result is None:
        result = eager(cls, *args)
    return result


interpreter.set_interpretation(eager)  # Use eager interpretation by default.


class FunsorMeta(type):
    """
    Metaclass for Funsors to perform three independent tasks:

    1.  Fill in default kwargs and convert kwargs to args before deferring to a
        nonstandard interpretation. This allows derived metaclasses to fill in
        defaults and do type conversion, thereby simplifying logic of
        interpretations.
    2.  Ensure each Funsor class has an attribute ``._ast_fields`` describing
        its input args and each Funsor instance has an attribute ``._ast_args``
        with values corresponding to its input args. This allows the instance
        to be reflectively reconstructed under a different interpretation, and
        is used by :func:`funsor.interpreter.reinterpret`.
    3.  Cons-hash construction, so that repeatedly calling the constructor
        with identical args will product the same object. This enables cheap
        syntactic equality testing using the ``is`` operator, which is
        is important both for hashing (e.g. for memoizing funsor functions)
        and for unit testing, since ``.__eq__()`` is overloaded with
        elementwise semantics. Cons hashing differs from memoization in that
        it incurs no memory overhead beyond the cons hash dict.
    """
    def __init__(cls, name, bases, dct):
        super(FunsorMeta, cls).__init__(name, bases, dct)
        cls._ast_fields = getargspec(cls.__init__)[0][1:]
        cls._cons_cache = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        # Convert kwargs to args.
        if kwargs:
            args = list(args)
            for name in cls._ast_fields[len(args):]:
                args.append(kwargs.pop(name))
            assert not kwargs, kwargs
            args = tuple(args)

        return interpret(cls, *args)


@add_metaclass(FunsorMeta)
class Funsor(object):
    """
    Abstract base class for immutable functional tensors.

    Concrete derived classes must implement ``__init__()`` methods taking
    hashable ``*args`` and no optional ``**kwargs`` so as to support cons
    hashing.

    :param OrderedDict inputs: A mapping from input name to domain.
        This can be viewed as a typed context or a mapping from
        free variables to domains.
    :param Domain output: An output domain.
    """
    def __init__(self, inputs, output):
        assert isinstance(inputs, OrderedDict)
        for name, input_ in inputs.items():
            assert isinstance(name, str)
            assert isinstance(input_, Domain)
        assert isinstance(output, Domain)
        super(Funsor, self).__init__()
        self.inputs = inputs
        self.output = output

    @property
    def dtype(self):
        return self.output.dtype

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ', '.join(map(repr, self._ast_values)))

    def __str__(self):
        return '{}({})'.format(type(self).__name__, ', '.join(map(str, self._ast_values)))

    def _pretty(self, lines, indent=0):
        lines.append((indent, type(self).__name__))
        for arg in self._ast_values:
            if isinstance(arg, Funsor):
                arg._pretty(lines, indent + 1)
            elif type(arg) is tuple and all(isinstance(x, Funsor) for x in arg):
                lines.append((indent + 1, 'tuple'))
                for x in arg:
                    x._pretty(lines, indent + 2)
            else:
                lines.append((indent + 1, re.sub('\n\\s*', ' ', str(arg))))

    def pretty(self):
        lines = []
        self._pretty(lines)
        return '\n'.join('|   ' * indent + text for indent, text in lines)

    def __call__(self, *args, **kwargs):
        """
        Partially evaluates this funsor by substituting dimensions.
        """
        # Eagerly restrict to this funsor's inputs and convert to_funsor().
        subs = OrderedDict(zip(self.inputs, args))
        for k in self.inputs:
            if k in kwargs:
                subs[k] = kwargs[k]
        for k, v in subs.items():
            if isinstance(v, str):
                # Allow renaming of inputs via syntax x(y="z").
                v = Variable(v, self.inputs[k])
            else:
                v = to_funsor(v, self.inputs[k].dtype)
            if v.output != self.inputs[k]:
                raise TypeError('Expected substitution of {} to have type {}, but got {}'
                                .format(repr(k), v.output, self.inputs[k]))
            subs[k] = v
        return Subs(self, tuple(subs.items()))

    def __bool__(self):
        if self.inputs or self.output.shape:
            raise ValueError(
                "bool value of Funsor with more than one value is ambiguous")
        raise NotImplementedError

    def __nonzero__(self):
        return self.__bool__()

    def __len__(self):
        if not self.output.shape:
            raise ValueError('Funsor with empty shape has no len()')
        return self.output.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def item(self):
        if self.inputs or self.output.shape:
            raise ValueError(
                "only one element Funsors can be converted to Python scalars")
        raise NotImplementedError

    def reduce(self, op, reduced_vars=None):
        """
        Reduce along all or a subset of inputs.

        :param callable op: A reduction operation.
        :param reduced_vars: An optional input name or set of names to reduce.
            If unspecified, all inputs will be reduced.
        :type reduced_vars: str or frozenset
        """
        # Eagerly convert reduced_vars to appropriate things.
        if reduced_vars is None:
            # Empty reduced_vars means "reduce over everything".
            reduced_vars = frozenset(self.inputs)
        elif isinstance(reduced_vars, str):
            # A single name means "reduce over this one variable".
            reduced_vars = frozenset([reduced_vars])
        assert isinstance(reduced_vars, frozenset), reduced_vars
        if not reduced_vars:
            return self
        assert reduced_vars.issubset(self.inputs)
        return Reduce(op, self, reduced_vars)

    def sample(self, sampled_vars, sample_inputs=None):
        """
        Create a Monte Carlo approximation to this funsor by replacing
        functions of ``sampled_vars`` with :class:`~funsor.delta.Delta`s.

        The result is a :class:`Funsor` with the same ``.inputs`` and
        ``.output`` as the original funsor (plus ``sample_inputs`` if
        provided), so that self can be replaced by the sample in expectation
        computations::

            y = x.sample(sampled_vars)
            assert y.inputs == x.inputs
            assert y.output == x.output
            exact = (x.exp() * integrand).reduce(ops.add)
            approx = (y.exp() * integrand).reduce(ops.add)

        If ``sample_inputs`` is provided, this creates a batch of samples
        scaled samples.

        :param frozenset sampled_vars: A set of input variables to sample.
        :param OrderedDict sample_inputs: An optional mapping from variable
            name to :class:`~funsor.domains.Domain` over which samples will
            be batched.
        """
        assert self.output == reals()
        assert isinstance(sampled_vars, frozenset)
        if sampled_vars.isdisjoint(self.inputs):
            return self

        result = interpreter.debug_logged(self.unscaled_sample)(sampled_vars, sample_inputs)
        if sample_inputs is not None:
            log_scale = 0
            for var, domain in sample_inputs.items():
                if var in result.inputs and var not in self.inputs:
                    log_scale -= math.log(domain.dtype)
            if log_scale != 0:
                result += log_scale
        return result

    def unscaled_sample(self, sampled_vars, sample_inputs=None):
        """
        Internal method to draw an unscaled sample.
        This should be overridden by subclasses.
        """
        assert self.output == reals()
        assert isinstance(sampled_vars, frozenset)
        if sampled_vars.isdisjoint(self.inputs):
            return self
        raise TypeError("Cannot sample from a {}".format(type(self).__name__))

    def align(self, names):
        """
        Align this funsor to match given ``names``.
        This is mainly useful in preparation for extracting ``.data``
        of a :class:`funsor.torch.Tensor`.

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
        assert reduced_vars.issubset(self.inputs)  # FIXME Is this valid?
        if not reduced_vars:
            return self

        return None  # defer to default implementation

    def sequential_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)  # FIXME Is this valid?
        if not reduced_vars:
            return self

        # Try to sum out integer scalars. This is mainly useful for testing,
        # since reduction is more efficiently implemented by Tensor.
        eager_vars = []
        lazy_vars = []
        for k in reduced_vars:
            if isinstance(self.inputs[k].dtype, integer_types) and not self.inputs[k].shape:
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

    # The following methods conform to a standard array/tensor interface.

    def __invert__(self):
        return Unary(ops.invert, self)

    def __neg__(self):
        return Unary(ops.neg, self)

    def abs(self):
        return Unary(ops.abs, self)

    def sqrt(self):
        return Unary(ops.sqrt, self)

    def exp(self):
        return Unary(ops.exp, self)

    def log(self):
        return Unary(ops.log, self)

    def log1p(self):
        return Unary(ops.log1p, self)

    # The following reductions are treated as Unary ops because they
    # reduce over output shape while preserving all inputs.
    # To reduce over inputs, instead call .reduce(op, reduced_vars).

    def sum(self):
        return Unary(ops.add, self)

    def prod(self):
        return Unary(ops.mul, self)

    def logsumexp(self):
        return Unary(ops.logaddexp, self)

    def all(self):
        return Unary(ops.and_, self)

    def any(self):
        return Unary(ops.or_, self)

    def min(self):
        return Unary(ops.min, self)

    def max(self):
        return Unary(ops.max, self)

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

    def __min__(self, other):
        return Binary(ops.min, self, to_funsor(other))

    def __max__(self, other):
        return Binary(ops.max, self, to_funsor(other))

    def __getitem__(self, other):
        if type(other) is not tuple:
            other = to_funsor(other, self.output.shape[0])
            return Binary(ops.getitem, self, other)

        # Handle Ellipsis slicing.
        if any(part is Ellipsis for part in other):
            left = []
            for part in other:
                if part is Ellipsis:
                    break
                left.append(part)
            right = []
            for part in reversed(other):
                if part is Ellipsis:
                    break
                right.append(part)
            right.reverse()
            missing = len(self.output.shape) - len(left) - len(right)
            assert missing >= 0
            middle = [slice(None)] * missing
            other = tuple(left + middle + right)

        # Handle each slice separately.
        result = self
        offset = 0
        for part in other:
            if isinstance(part, slice):
                if part != slice(None):
                    raise NotImplementedError('TODO support nontrivial slicing')
                offset += 1
            else:
                part = to_funsor(part, result.output.shape[offset])
                result = Binary(GetitemOp(offset), result, part)
        return result


interpreter.reinterpret.register(Funsor)(interpreter.reinterpret_funsor)


@dispatch(object)
def to_funsor(x):
    """
    Convert to a :class:`Funsor`.
    Only :class:`Funsor`s and scalars are accepted.

    :param x: An object.
    :param dtype: An optional datatype hint (integer or the string "real").
    :return: A Funsor equivalent to ``x``.
    :rtype: Funsor
    :raises: ValueError
    """
    raise ValueError("cannot convert to Funsor: {}".format(repr(x)))


@dispatch(object, object)
def to_funsor(x, dtype):
    raise ValueError("cannot convert to Funsor: {}".format(repr(x)))


@dispatch(Funsor)
def to_funsor(x):
    return x


@dispatch(Funsor, object)
def to_funsor(x, dtype):
    if x.dtype != dtype:
        raise ValueError("dtype mismatch: {} vs {}".format(x.dtype, dtype))
    return x


@singledispatch
def to_data(x):
    """
    Extract a python object from a :class:`Funsor`.

    Raises a ``ValueError`` if free variables remain or if the funsor is lazy.

    :param x: An object, possibly a :class:`Funsor`.
    :return: A non-funsor equivalent to ``x``.
    :raises: ValueError
    """
    return x


@to_data.register(Funsor)
def _to_data_funsor(x):
    raise ValueError("cannot convert to a non-Funsor: {}".format(repr(x)))


class Variable(Funsor):
    """
    Funsor representing a single free variable.

    :param str name: A variable name.
    :param funsor.domains.Domain output: A domain.
    """
    def __init__(self, name, output):
        inputs = OrderedDict([(name, output)])
        super(Variable, self).__init__(inputs, output)
        self.name = name

    def __repr__(self):
        return "Variable({}, {})".format(repr(self.name), repr(self.output))

    def __str__(self):
        return self.name

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        for k, v in subs:
            if k == self.name:
                return v
        return self


@dispatch(str, integer_types)
def to_funsor(name, dtype):
    return Variable(name, bint(dtype))


class Subs(Funsor):
    """
    Lazy substitution of the form ``x(u=y, v=z)``.
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
        super(Subs, self).__init__(inputs, arg.output)
        self.arg = arg
        self.subs = OrderedDict(subs)

    def __repr__(self):
        return 'Subs({}, {})'.format(self.arg, self.subs)

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        old_subs = tuple((k, Subs(v, subs)) for k, v in self.subs.items())
        new_subs = tuple((k, v) for k, v in subs if k not in self.subs)
        subs = old_subs + new_subs
        return Subs(self.arg, subs) if subs else self.arg

    def unscaled_sample(self, sampled_vars, sample_inputs=None):
        if sample_inputs is not None:
            if any(k in sample_inputs for k, v in self.subs):
                raise NotImplementedError('TODO alpha-convert')
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
        arg = self.arg.unscaled_sample(subs_sampled_vars, sample_inputs)
        return Subs(arg, tuple(self.subs.items()))


@lazy.register(Subs, Funsor, object)
@eager.register(Subs, Funsor, object)
def eager_subs(arg, subs):
    assert isinstance(subs, tuple)
    if not any(k in arg.inputs for k, v in subs):
        return arg
    return interpreter.debug_logged(arg.eager_subs)(subs)


_PREFIX = {
    ops.neg: '-',
    ops.invert: '~',
}


class Unary(Funsor):
    """
    Lazy unary operation.
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
            return '{}{}'.format(_PREFIX[self.op], self.arg)
        return 'Unary({}, {})'.format(self.op.__name__, self.arg)

    def eager_subs(self, subs):
        arg = Subs(self.arg, subs)
        return Unary(self.op, arg)


@eager.register(Unary, Op, Funsor)
def eager_unary(op, arg):
    return interpreter.debug_logged(arg.eager_unary)(op)


@eager.register(Unary, AssociativeOp, Funsor)
def eager_unary(op, arg):
    if not arg.output.shape:
        return arg
    return interpreter.debug_logged(arg.eager_unary)(op)


_INFIX = {
    ops.add: '+',
    ops.sub: '-',
    ops.mul: '*',
    ops.truediv: '/',
    ops.pow: '**',
}


class Binary(Funsor):
    """
    Lazy binary operation.
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
            return '({} {} {})'.format(self.lhs, _INFIX[self.op], self.rhs)
        return 'Binary({}, {}, {})'.format(self.op.__name__, self.lhs, self.rhs)

    def eager_subs(self, subs):
        lhs = Subs(self.lhs, subs)
        rhs = Subs(self.rhs, subs)
        return Binary(self.op, lhs, rhs)


class Reduce(Funsor):
    """
    Lazy reduction over multiple variables.
    """
    def __init__(self, op, arg, reduced_vars):
        assert callable(op)
        assert isinstance(arg, Funsor)
        assert isinstance(reduced_vars, frozenset)
        inputs = OrderedDict((k, v) for k, v in arg.inputs.items() if k not in reduced_vars)
        output = arg.output
        super(Reduce, self).__init__(inputs, output)
        self.op = op
        self.arg = arg
        self.reduced_vars = reduced_vars

    def __repr__(self):
        return 'Reduce({}, {}, {})'.format(
            self.op.__name__, self.arg, self.reduced_vars)

    def eager_subs(self, subs):
        subs = tuple((k, v) for k, v in subs if k not in self.reduced_vars)
        if not all(self.reduced_vars.isdisjoint(v.inputs) for k, v in subs):
            raise NotImplementedError('TODO alpha-convert to avoid conflict')
        arg = Subs(self.arg, subs)
        return arg.reduce(self.op, self.reduced_vars)

    def eager_reduce(self, op, reduced_vars):
        if op is self.op:
            # Eagerly fuse reductions.
            assert isinstance(reduced_vars, frozenset)
            reduced_vars = reduced_vars.intersection(self.inputs) | self.reduced_vars
            return Reduce(op, self.arg, reduced_vars)
        return super(Reduce, self).reduce(op, reduced_vars)


@eager.register(Reduce, AssociativeOp, Funsor, frozenset)
def eager_reduce(op, arg, reduced_vars):
    return interpreter.debug_logged(arg.eager_reduce)(op, reduced_vars)


@sequential.register(Reduce, AssociativeOp, Funsor, frozenset)
def sequential_reduce(op, arg, reduced_vars):
    return interpreter.debug_logged(arg.sequential_reduce)(op, reduced_vars)


class NumberMeta(FunsorMeta):
    """
    Wrapper to fill in default ``dtype``.
    """
    def __call__(cls, data, dtype=None):
        if dtype is None:
            dtype = "real"
        return super(NumberMeta, cls).__call__(data, dtype)


@add_metaclass(NumberMeta)
class Number(Funsor):
    """
    Funsor backed by a Python number.

    :param numbers.Number data: A python number.
    :param dtype: A nonnegative integer or the string "real".
    """
    def __init__(self, data, dtype=None):
        assert isinstance(data, numbers.Number)
        if isinstance(dtype, integer_types):
            data = type(dtype)(data)
            if dtype != 2:  # booleans have bitwise interpretation
                assert 0 <= data and data < dtype
        else:
            assert isinstance(dtype, str) and dtype == "real"
            data = float(data)
        inputs = OrderedDict()
        output = Domain((), dtype)
        super(Number, self).__init__(inputs, output)
        self.data = data

    def __repr__(self):
        if self.dtype == "real":
            return 'Number({}, "real")'.format(repr(self.data))
        else:
            return 'Number({}, {})'.format(repr(self.data), self.dtype)

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

    def eager_subs(self, subs):
        return self

    def eager_unary(self, op):
        dtype = find_domain(op, self.output).dtype
        return Number(op(self.data), dtype)


@dispatch(numbers.Number)
def to_funsor(x):
    return Number(x)


@dispatch(numbers.Number, object)
def to_funsor(x, dtype):
    return Number(x, dtype)


@to_data.register(Number)
def _to_data_number(x):
    return x.data


@eager.register(Binary, Op, Number, Number)
def eager_binary_number_number(op, lhs, rhs):
    data = op(lhs.data, rhs.data)
    output = find_domain(op, lhs.output, rhs.output)
    dtype = output.dtype
    return Number(data, dtype)


class Align(Funsor):
    """
    Lazy call to ``.align(...)``.
    """
    def __init__(self, arg, names):
        assert isinstance(arg, Funsor)
        assert isinstance(names, tuple)
        assert all(isinstance(name, str) for name in names)
        assert all(name in arg.inputs for name in names)
        inputs = OrderedDict((name, arg.inputs[name]) for name in names)
        inputs.update(arg.inputs)
        output = arg.output
        super(Align, self).__init__(inputs, output)
        self.arg = arg

    def align(self, names):
        return self.arg.align(names)

    def eager_subs(self, subs):
        return Subs(self.arg, subs)

    def eager_unary(self, op):
        return Unary(op, self.arg)

    def eager_reduce(self, op, reduced_vars):
        return self.arg.reduce(op, reduced_vars)


@eager.register(Binary, Op, Align, Funsor)
def eager_binary_align_funsor(op, lhs, rhs):
    return Binary(op, lhs.arg, rhs)


@eager.register(Binary, Op, Funsor, Align)
def eager_binary_funsor_align(op, lhs, rhs):
    return Binary(op, lhs, rhs.arg)


@eager.register(Binary, Op, Align, Align)
def eager_binary_align_align(op, lhs, rhs):
    return Binary(op, lhs.arg, rhs.arg)


class Stack(Funsor):
    """
    Stack of funsors along a new input dimension.

    :param tuple components: A tuple of Funsors.
    :param str name: The name of the new leftmost dimension.
    """
    def __init__(self, components, name):
        assert isinstance(components, tuple)
        assert components
        assert not any(name in x.inputs for x in components)
        assert len(set(x.output for x in components)) == 1
        output = components[0].output
        domain = bint(len(components))
        inputs = OrderedDict([(name, domain)])
        for x in components:
            inputs.update(x.inputs)
        super(Stack, self).__init__(inputs, output)
        self.components = components
        self.name = name

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        if not any(k in self.inputs for k, v in subs):
            return self
        pos = None
        for i, (k, index) in enumerate(subs):
            if k == self.name:
                pos = i
                break

        if pos is None:
            # Eagerly recurse into components.
            assert not any(self.name in v.inputs for k, v in subs)
            components = tuple(Subs(x, subs) for x in self.components)
            return Stack(components, self.name)

        # Try to eagerly select an index.
        assert index.output == bint(len(self.components))
        subs = subs[:pos] + subs[1 + pos:]

        if isinstance(index, Number):
            # Select a single component.
            result = self.components[index.data]
            return Subs(result, subs)

        if isinstance(index, Variable):
            # Rename the stacking dimension.
            components = self.components
            if subs:
                components = tuple(Subs(x, subs) for x in components)
            return Stack(components, index.name)

        if not subs:
            raise NotImplementedError('TODO support advanced indexing in Stack')

        # Eagerly recurse into components but lazily substitute.
        components = tuple(Subs(x, subs) for x in self.components)
        result = Stack(components, self.name)
        return Subs(result, ((self.name, index),))

    def eager_reduce(self, op, reduced_vars):
        components = self.components
        if self.name in reduced_vars:
            reduced_vars -= frozenset([self.name])
            if reduced_vars:
                components = tuple(x.reduce(op, reduced_vars) for x in components)
            return reduce(op, components)
        components = tuple(x.reduce(op, reduced_vars) for x in components)
        return Stack(components, self.name)


def _of_shape(fn, shape):
    args, vargs, kwargs, defaults = getargspec(fn)
    assert not vargs
    assert not kwargs
    names = tuple(args)
    args = [Variable(name, size) for name, size in zip(names, shape)]
    return to_funsor(fn(*args)).align(names)


def of_shape(*shape):
    """
    Decorator to construct a :class:`Funsor` with one free :class:`Variable`
    per function arg.
    """
    return functools.partial(_of_shape, shape=shape)


################################################################################
# Register Ops
################################################################################

@ops.abs.register(Funsor)
def _abs(x):
    return Unary(ops.abs, x)


@ops.sqrt.register(Funsor)
def _sqrt(x):
    return Unary(ops.sqrt, x)


@ops.exp.register(Funsor)
def _exp(x):
    return Unary(ops.exp, x)


@ops.log.register(Funsor)
def _log(x):
    return Unary(ops.log, x)


@ops.log1p.register(Funsor)
def _log1p(x):
    return Unary(ops.log1p, x)


__all__ = [
    'Binary',
    'Funsor',
    'Number',
    'Reduce',
    'Stack',
    'Subs',
    'Unary',
    'Variable',
    'eager',
    'lazy',
    'of_shape',
    'reflect',
    'sequential',
    'to_data',
    'to_funsor',
]
