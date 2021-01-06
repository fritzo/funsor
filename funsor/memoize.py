# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable
from contextlib import contextmanager

import funsor.interpreter as interpreter


@contextmanager
def memoize(cache=None):
    """
    Exploit cons-hashing to do implicit common subexpression elimination
    """
    if cache is None:
        cache = {}

    @interpreter.interpretation(interpreter._INTERPRETATION)  # use base
    def memoize_interpretation(cls, *args):
        key = (cls,) + tuple(id(arg) if (type(arg).__name__ == "DeviceArray") or not isinstance(arg, Hashable)
                             else arg for arg in args)
        if key not in cache:
            cache[key] = cls(*args)
        return cache[key]

    with interpreter.interpretation(memoize_interpretation):
        yield cache


class MemoizeInterpretation(Interpretation):

    def __init__(self, base_interpretation, cache=None):
        self.base_interpretation = base_interpretation
        self.cache = {} if cache is None else cache

    @property
    def is_total(self):
        return self.base_interpretation.is_total

    def __call__(self, cls, *args):
        # FIXME recycled ids can cause incorrect cache hits
        key = (cls,) + tuple(id(arg) if (type(arg).__name__ == "DeviceArray") or not isinstance(arg, Hashable)
                             else arg for arg in args)
        if key not in self.cache:
            self.cache[key] = self.base_interpretation(cls, *args)
        return self.cache[key]
