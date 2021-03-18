# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import weakref
from abc import ABC, abstractmethod
from collections.abc import Hashable
from contextlib import ContextDecorator, contextmanager
from timeit import default_timer

from . import instrument
from .interpreter import (
    get_interpretation,
    is_atom,
    pop_interpretation,
    push_interpretation,
    reinterpret,
)
from .registry import KeyedRegistry
from .util import get_backend


class Interpretation(ContextDecorator, ABC):
    """
    Abstract base class for Funsor interpretations.

    Instances may be used as context managers or decorators.

    :param str name: A name used for printing and debugging (required).
    """

    def __init__(self, name):
        self.__name__ = name
        super().__init__()

    def __repr__(self):
        return self.__name__

    is_total = False

    def __enter__(self):
        # TODO consider requiring totality:
        # assert self.is_total
        new = self
        if not self.is_total:
            new = PrioritizedInterpretation(new, get_interpretation())
        push_interpretation(new)
        return self

    def __exit__(self, *args):
        pop_interpretation()

    @property
    def subinterpretations(self):
        return (self,)

    @abstractmethod
    def interpret(self, cls, *args):
        raise NotImplementedError

    @staticmethod
    def make_hash_key(cls, *args):
        backend = get_backend()
        if backend == "jax":
            # JAX DeviceArray has .__hash__ method but raise the unhashable error there.
            from jax.interpreters.xla import DeviceArray

            return tuple(
                id(arg)
                if isinstance(arg, DeviceArray) or not isinstance(arg, Hashable)
                else arg
                for arg in args
            )
        if backend == "torch":
            # Avoid "ImportError: sys.meta_path is None" on shutdown.
            from torch import Tensor

            return tuple(
                id(arg)
                if isinstance(arg, Tensor) or not isinstance(arg, Hashable)
                else arg
                for arg in args
            )
        return tuple(id(arg) if not isinstance(arg, Hashable) else arg for arg in args)


class CallableInterpretation(Interpretation):
    """
    A simple callable interpretation.

    Example usage::

        @CallableInterpretation
        def my_interpretation(cls, *args):
            return ...

    :param callable interpret: A function implementing interpretation.
    """

    def __init__(self, interpret):
        assert callable(interpret)
        super().__init__(interpret.__name__)
        self.interpret = interpret

    def set_callable(self, interpret):
        """
        Resets the callable ``.interpret`` attribute.
        """
        assert callable(interpret)
        self.interpret = interpret
        return self

    def interpret(self, cls, *args):
        raise ValueError("interpret has not been defined")


class DispatchedInterpretation(Interpretation):
    """
    An interpretation based on pattern matching.

    Example usage::

        my_interpretation = DispatchedInterpretation("my_interpretation")

        # Register a funsor pattern and rule.
        @my_interpretation.register(...)
        def my_impl(cls, *args):
            ...

        # Use the new interpretation.
        with my_interpretation:
            ...
    """

    def __init__(self, name="dispatched"):
        super().__init__(name)
        self.registry = registry = KeyedRegistry(default=lambda *args: None)

        if instrument.DEBUG or instrument.PROFILE:
            self.register = lambda *args: lambda fn: registry.register(*args)(
                instrument.debug_logged(fn)
            )
        else:
            self.register = registry.register

        if instrument.PROFILE:
            COUNTERS = instrument.COUNTERS

            def profiled_dispatch(*args):
                name = self.__name__ + ".dispatch"
                start = default_timer()
                result = registry.dispatch(*args)
                COUNTERS["time"][name] += default_timer() - start
                COUNTERS["call"][name] += 1
                COUNTERS["interpretation"][self.__name__] += 1
                return result

            self.dispatch = profiled_dispatch
        else:
            self.dispatch = registry.dispatch

    def interpret(self, cls, *args):
        return self.dispatch(cls, *args)(*args)


class PrioritizedInterpretation(Interpretation):
    r"""
    A prioritized sequence of subinterpretations.

    To interpret ``cls(*args)``, each subinterpretation is tried until one returns
    a value other than None.

    :param \*subinterpretations: A sequence of :class:`Interpretation` s.
    """

    def __init__(self, *subinterpretations):
        subinterpretations = tuple(
            ss for s in subinterpretations for ss in s.subinterpretations
        )
        assert subinterpretations
        assert len(subinterpretations) < 10, "suspicious interpretation overflow"
        assert not any(s.is_total for s in subinterpretations[:-1])
        super().__init__("/".join(s.__name__ for s in subinterpretations))
        self._subinterpretations = subinterpretations

    @property
    def subinterpretations(self):
        return self._subinterpretations

    @property
    def is_total(self):
        return any(s.is_total for s in self._subinterpretations)

    @property
    def register(self):
        return self._subinterpretations[0].register

    @property
    def dispatch(self):
        return self._subinterpretations[0].dispatch

    def interpret(self, cls, *args):
        for s in self._subinterpretations:
            result = s.interpret(cls, *args)
            if result is not None:
                return result


class StatefulInterpretationMeta(type(ABC)):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls.registry = KeyedRegistry(default=lambda *args: None)

        if instrument.PROFILE:
            COUNTERS = instrument.COUNTERS

            def profiled_dispatch(*args):
                name = cls.__name__ + ".dispatch"
                start = default_timer()
                result = cls.registry.dispatch(*args)
                COUNTERS["time"][name] += default_timer() - start
                COUNTERS["call"][name] += 1
                COUNTERS["interpretation"][cls.__name__] += 1
                return result

            cls.dispatch = staticmethod(profiled_dispatch)
        else:
            cls.dispatch = staticmethod(cls.registry.dispatch)


class StatefulInterpretation(Interpretation, metaclass=StatefulInterpretationMeta):
    """
    Base class for interpretations with instance-dependent state or parameters.

    Example usage::

        class MyInterpretation(StatefulInterpretation):

            def __init__(self, my_param):
                self.my_param = my_param

        @MyInterpretation.register(...)
        def my_impl(interpretation_state, cls, *args):
            my_param = interpretation_state.my_param
            ...

        with MyInterpretation(my_param=0.1):
            ...
    """

    def __init__(self, name="stateful"):
        super().__init__(name)

    def interpret(self, cls, *args):
        return self.dispatch(cls, *args)(self, *args)

    if instrument.DEBUG:

        @classmethod
        def register(cls, *args):
            return lambda fn: cls.registry.register(*args)(instrument.debug_logged(fn))

    else:

        @classmethod
        def register(cls, *args):
            return cls.registry.register(*args)


class NormalizedInterpretation(Interpretation):
    def __init__(self, subinterpretation):
        super().__init__(f"Normalized({subinterpretation.__name__})")
        self.subinterpretation = subinterpretation
        self.register = self.subinterpretation.register
        self.dispatch = self.subinterpretation.dispatch
        self._cache = {}  # weakref.WeakValueDictionary()  # TODO make this work

    def interpret(self, cls, *args):
        # 1. try self.subinterpret.
        result = self.subinterpretation.interpret(cls, *args)
        if result is not None:
            return result

        # 2. normalize to a Contraction normal form (will succeed)
        # Note eager_contraction_generic_recursive() effectively fuses this
        # step with step 3 below to short-circuit some logic.
        with normalize:
            normalized_args = []
            for arg in args:
                try:
                    normalized_args.append(arg if is_atom(arg) else self._cache[arg])
                except KeyError:
                    normalized_arg = reinterpret(arg)
                    self._cache[arg] = normalized_arg
                    normalized_args.append(normalized_arg)
            normal_form = cls(*normalized_args)

        # 3. try evaluating that normal form
        with PrioritizedInterpretation(self.subinterpretation, simplify):
            # TODO use .interpret instead of reinterpret here to avoid traversal
            result = reinterpret(normal_form)
        if result is not normal_form:  # I.e. was progress made?
            return result

        # 4. if that fails, fall back to base interpretation of cls(*args)
        return None


class Memoize(Interpretation):
    """
    Exploits cons-hashing to do implicit common subexpression elimination.

    :param Interpretation base_interpretation: The interpretation to memoize.
    :param dict cache: An optional temporary cache where results will be
        memoized.
    """

    def __init__(self, base_interpretation, cache=None):
        super().__init__(f"Memoize({base_interpretation.__name__})")
        self.base_interpretation = base_interpretation
        if cache is None:
            cache = {}
        else:
            assert isinstance(cache, dict)
        self.cache = cache

    @property
    def is_total(self):
        return self.base_interpretation.is_total

    def interpret(self, cls, *args):
        key = self.make_hash_key(cls, *args)
        value = self.cache.get(key)
        if value is None:
            self.cache[key] = value = self.base_interpretation.interpret(cls, *args)
        return value


@contextmanager
def memoize(cache=None):
    """
    Context manager wrapping :class:`Memoize` and yielding the ``cache`` dict.
    """
    base_interpretation = get_interpretation()
    with Memoize(base_interpretation, cache) as interp:
        yield interp.cache


################################################################################
# Concrete interpretations.


class Simplify(DispatchedInterpretation):

    is_total = True  # because it always ends with normalize

    def interpret(self, cls, *args):
        result = super().interpret(cls, *args)
        if result is None:
            with normalize:
                result = cls(*args)
        return result


@CallableInterpretation
def reflect(cls, *args):
    raise ValueError("Should be overwritten in terms.py")


reflect.is_total = True

normalize_base = DispatchedInterpretation("normalize")
normalize = PrioritizedInterpretation(normalize_base, reflect)
"""
Normalize modulo associativity and commutativity, but do not evaluate any
numerical operations.
"""

simplify = Simplify("simplify")

lazy_base = NormalizedInterpretation(DispatchedInterpretation("lazy"))
lazy = PrioritizedInterpretation(lazy_base, reflect)
"""
Performs substitutions eagerly, but construct lazy funsors for everything else.
"""

eager_base = NormalizedInterpretation(DispatchedInterpretation("eager"))
eager = PrioritizedInterpretation(eager_base, reflect)
"""
Eager exact naive interpretation wherever possible.
"""

die = DispatchedInterpretation("die")
eager_or_die = PrioritizedInterpretation(eager_base.subinterpretation, die, reflect)

sequential_base = NormalizedInterpretation(DispatchedInterpretation("sequential"))
sequential = PrioritizedInterpretation(sequential_base, eager)
"""
Eagerly execute ops with known implementations; additonally execute
vectorized ops sequentially if no known vectorized implementation exists.
"""

moment_matching_base = NormalizedInterpretation(
    DispatchedInterpretation("moment_matching")
)
moment_matching = PrioritizedInterpretation(moment_matching_base, eager)
"""
A moment matching interpretation of :class:`Reduce` expressions. This falls
back to :class:`eager` in other cases.
"""

push_interpretation(reflect)  # Set for optional type checking.
push_interpretation(eager)  # Use eager interpretation by default.


__all__ = [
    "CallableInterpretation",
    "DispatchedInterpretation",
    "Interpretation",
    "NormalizedInterpretation",
    "PrioritizedInterpretation",
    "Memoize",
    "StatefulInterpretation",
    "die",
    "eager",
    "lazy",
    "memoize",
    "moment_matching",
    "normalize",
    "reflect",
    "sequential",
    "simplify",
]
