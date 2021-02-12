# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import re
import types
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from functools import singledispatch
from timeit import default_timer

import numpy as np

from funsor.domains import ArrayType
from funsor.instrument import debug_logged
from funsor.ops import Op, is_numeric_array
from funsor.registry import KeyedRegistry
from funsor.util import is_nn_module

from . import instrument

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_INTERPRETATION = None  # To be set later in funsor.terms
_USE_TCO = int(os.environ.get("FUNSOR_USE_TCO", 0))

_GENSYM_COUNTER = 0


def _classname(cls):
    return getattr(cls, "classname", cls.__name__)


class Interpreter:
    @property
    def __call__(self):
        return _INTERPRETATION


if instrument.DEBUG:

    def interpret(cls, *args):
        indent = instrument.get_indent()
        if instrument.DEBUG > 1:
            typenames = [_classname(cls)] + [_classname(type(arg)) for arg in args]
        else:
            typenames = [cls.__name__] + [type(arg).__name__ for arg in args]
        print(indent + " ".join(typenames))

        instrument.STACK_SIZE += 1
        try:
            result = _INTERPRETATION(cls, *args)
        finally:
            instrument.STACK_SIZE -= 1

        if instrument.DEBUG > 1:
            result_str = re.sub("\n", "\n          " + indent, str(result))
        else:
            result_str = type(result).__name__
        print(indent + "-> " + result_str)
        return result


else:
    interpret = Interpreter()


def set_interpretation(new):
    assert callable(new)
    global _INTERPRETATION
    _INTERPRETATION = new


@contextmanager
def interpretation(new):
    assert callable(new)
    if isinstance(new, Interpretation):
        try:
            new.__enter__()
            yield
        finally:
            new.__exit__()
    else:  # temporary backwards compatibility
        global _INTERPRETATION
        old = _INTERPRETATION
        new = InterpreterStack(new, old)
        try:
            _INTERPRETATION = new
            yield
        finally:
            _INTERPRETATION = old


@singledispatch
def recursion_reinterpret(x):
    r"""
    Overloaded reinterpretation of a deferred expression.
    This interpreter uses the Python stack and is subject to the recursion limit.

    This handles a limited class of expressions, raising
    ``ValueError`` in unhandled cases.

    :param x: An input, typically involving deferred
        :class:`~funsor.terms.Funsor` s.
    :type x: A funsor or data structure holding funsors.
    :return: A reinterpreted version of the input.
    :raises: ValueError
    """
    raise ValueError(type(x))


# We need to register this later in terms.py after declaring Funsor.
# reinterpret.register(Funsor)
@debug_logged
def reinterpret_funsor(x):
    return _INTERPRETATION(type(x), *map(recursion_reinterpret, x._ast_values))


_ground_types = (
    str,
    int,
    float,
    type,
    functools.partial,
    types.FunctionType,
    types.BuiltinFunctionType,
    ArrayType,
    Op,
    np.generic,
    np.ndarray,
    np.ufunc,
)


for t in _ground_types:

    @recursion_reinterpret.register(t)
    def recursion_reinterpret_ground(x):
        return x


@recursion_reinterpret.register(tuple)
@debug_logged
def recursion_reinterpret_tuple(x):
    return tuple(map(recursion_reinterpret, x))


@recursion_reinterpret.register(frozenset)
@debug_logged
def recursion_reinterpret_frozenset(x):
    return frozenset(map(recursion_reinterpret, x))


@recursion_reinterpret.register(dict)
@debug_logged
def recursion_reinterpret_dict(x):
    return {key: recursion_reinterpret(value) for key, value in x.items()}


@recursion_reinterpret.register(OrderedDict)
@debug_logged
def recursion_reinterpret_ordereddict(x):
    return OrderedDict((key, recursion_reinterpret(value)) for key, value in x.items())


@singledispatch
def children(x):
    raise ValueError(type(x))


# has to be registered in terms.py
def children_funsor(x):
    return x._ast_values


@children.register(tuple)
@children.register(frozenset)
def _children_tuple(x):
    return x


@children.register(dict)
@children.register(OrderedDict)
def _children_tuple(x):
    return x.values()


for t in _ground_types:

    @children.register(t)
    def _children_ground(x):
        return ()


def is_atom(x):
    if isinstance(x, (tuple, frozenset)):
        return len(x) == 0 or all(is_atom(c) for c in x)
    return isinstance(x, _ground_types) or is_numeric_array(x) or is_nn_module(x)


def gensym(x=None):
    global _GENSYM_COUNTER
    _GENSYM_COUNTER += 1
    sym = _GENSYM_COUNTER
    if x is not None:
        if isinstance(x, str):
            return x + "_" + str(sym)
        return id(x)
    return "V" + str(sym)


def stack_reinterpret(x):
    r"""
    Overloaded reinterpretation of a deferred expression.
    This interpreter uses an explicit stack and no recursion but is much slower.

    This handles a limited class of expressions, raising
    ``ValueError`` in unhandled cases.

    :param x: An input, typically involving deferred
        :class:`~funsor.terms.Funsor` s.
    :type x: A funsor or data structure holding funsors.
    :return: A reinterpreted version of the input.
    :raises: ValueError
    """
    x_name = gensym(x)
    node_vars = {x_name: x}
    node_names = {x: x_name}
    env = {}
    stack = [(x_name, x)]
    parent_to_children = OrderedDict()
    child_to_parents = OrderedDict()
    while stack:
        h_name, h = stack.pop(0)
        parent_to_children[h_name] = []
        for c in children(h):
            if c in node_names:
                c_name = node_names[c]
            else:
                c_name = gensym(c)
                node_names[c] = c_name
                node_vars[c_name] = c
                stack.append((c_name, c))
            parent_to_children.setdefault(h_name, []).append(c_name)
            child_to_parents.setdefault(c_name, []).append(h_name)

    children_counts = OrderedDict((k, len(v)) for k, v in parent_to_children.items())
    leaves = [name for name, count in children_counts.items() if count == 0]
    while leaves:
        h_name = leaves.pop(0)
        if h_name in child_to_parents:
            for parent in child_to_parents[h_name]:
                children_counts[parent] -= 1
                if children_counts[parent] == 0:
                    leaves.append(parent)

        h = node_vars[h_name]
        if is_atom(h):
            env[h_name] = h
        elif isinstance(h, (tuple, frozenset)):
            env[h_name] = type(h)(env[c_name] for c_name in parent_to_children[h_name])
        else:
            env[h_name] = _INTERPRETATION(
                type(h), *(env[c_name] for c_name in parent_to_children[h_name])
            )

    return env[x_name]


def reinterpret(x):
    r"""
    Overloaded reinterpretation of a deferred expression.

    This handles a limited class of expressions, raising
    ``ValueError`` in unhandled cases.

    :param x: An input, typically involving deferred
        :class:`~funsor.terms.Funsor` s.
    :type x: A funsor or data structure holding funsors.
    :return: A reinterpreted version of the input.
    :raises: ValueError
    """
    if _USE_TCO:
        return stack_reinterpret(x)
    else:
        return recursion_reinterpret(x)


class Interpretation:

    is_total = False

    def __enter__(self):
        global _INTERPRETATION  # TODO get rid of this when _INTERPRETATION is list
        new = self
        if not self.is_total:
            new = PrioritizedInterpretation(new, _INTERPRETATION)
        self.old = _INTERPRETATION  # TODO store in global list instead of self
        _INTERPRETATION = new  # TODO make a list: _INTERPRETATION.append(new)
        return new

    def __exit__(self, *args):
        global _INTERPRETATION  # TODO get rid of this when _INTERPRETATION is list
        _INTERPRETATION = self.old  # TODO make a list: _INTERPRETATION.pop()

    def __call__(self, cls, *args):
        raise NotImplementedError


class DispatchedInterpretation(Interpretation):
    def __init__(self, default=lambda *args: None):
        self.registry = KeyedRegistry(default=default)
        if instrument.DEBUG:
            self.register = lambda *args: lambda self: self.registry.register(*args)(
                debug_logged(self)
            )
        else:
            self.register = self.registry.register
        self.dispatch = self.registry.dispatch

    def __call__(self, cls, *args):
        return self.dispatch(cls, *args)(*args)


class PrioritizedInterpretation(Interpretation):
    @property
    def base(self):
        return self.subinterpreters[0]

    @property
    def is_total(self):
        return any(s.is_total for s in self.subinterpreters)

    def __init__(self, *subinterpreters):
        assert len(subinterpreters) >= 1
        self.subinterpreters = tuple(subinterpreters)
        if isinstance(self.subinterpreters[0], DispatchedInterpretation):
            self.register = self.subinterpreters[0].register
            self.dispatch = self.subinterpreters[0].dispatch

    def __call__(self, cls, *args):
        for subinterpreter in self.subinterpreters:
            result = subinterpreter(cls, *args)
            if result is not None:
                return result


class InterpreterStack(namedtuple("InterpreterStack", ["default", "fallback"])):
    def __call__(self, cls, *args):
        for interpreter in self:
            result = interpreter(cls, *args)
            if result is not None:
                return result


def dispatched_interpretation(fn):
    """
    Decorator to create a dispatched interpretation function.
    """
    registry = KeyedRegistry(default=lambda *args: None)

    if instrument.DEBUG or instrument.PROFILE:
        fn.register = lambda *args: lambda fn: registry.register(*args)(
            debug_logged(fn)
        )
    else:
        fn.register = registry.register

    if instrument.PROFILE:
        COUNTERS = instrument.COUNTERS

        def profiled_dispatch(*args):
            name = fn.__name__ + ".dispatch"
            start = default_timer()
            result = registry.dispatch(*args)
            COUNTERS["time"][name] += default_timer() - start
            COUNTERS["call"][name] += 1
            COUNTERS["interpretation"][fn.__name__] += 1
            return result

        fn.dispatch = profiled_dispatch
    else:
        fn.dispatch = registry.dispatch

    return fn


def _dispatched_interpretation(fn):
    """
    New version of dispatched_interpretation decorator using Interpretation classes.

    Syntax::

        @prioritized_interpretation(normalize.base, reflect)
        @dispatched_interpretation
        def eager(cls, *args):
            return None
    """
    return DispatchedInterpretation(fn)


class StatefulInterpretationMeta(type):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls.registry = KeyedRegistry(default=lambda *args: None)
        cls.dispatch = cls.registry.dispatch


class StatefulInterpretation(Interpretation, metaclass=StatefulInterpretationMeta):
    """
    Base class for interpreters with instance-dependent state or parameters.

    Example usage::

        class MyInterpretation(StatefulInterpretation):

            def __init__(self, my_param):
                self.my_param = my_param

        @MyInterpretation.register(...)
        def my_impl(interpreter_state, cls, *args):
            my_param = interpreter_state.my_param
            ...

        with interpretation(MyInterpretation(my_param=0.1)):
            ...
    """

    def __call__(self, cls, *args):
        return self.dispatch(cls, *args)(self, *args)

    if instrument.DEBUG:

        @classmethod
        def register(cls, *args):
            return lambda fn: cls.registry.register(*args)(debug_logged(fn))

    else:

        @classmethod
        def register(cls, *args):
            return cls.registry.register(*args)


class PatternMissingError(NotImplementedError):
    def __str__(self):
        return "{}\nThis is most likely due to a missing pattern.".format(
            super().__str__()
        )


__all__ = [
    "PatternMissingError",
    "StatefulInterpretation",
    "dispatched_interpretation",
    "interpret",
    "interpretation",
    "reinterpret",
    "set_interpretation",
]
