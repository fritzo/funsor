from __future__ import absolute_import, division, print_function

import torch
from six import integer_types

from funsor.domains import Domain
from funsor.terms import Funsor
from funsor.torch import Tensor


def assert_close(actual, expected, atol=1e-6, rtol=1e-6):
    assert isinstance(actual, Funsor)
    assert isinstance(expected, Funsor)
    assert actual.inputs == expected.inputs, (actual.inputs, expected.inputs)
    assert actual.output == expected.output
    if isinstance(actual, Tensor):
        if actual.data.dtype in (torch.long, torch.uint8):
            assert (actual.data == expected.data).all()
        else:
            diff = (actual.data.detach() - expected.data.detach()).abs()
            assert diff.max() < atol
            assert (diff / (atol + expected.data.detach().abs())).max() < rtol
    else:
        raise ValueError('cannot compare objects of type {}'.format(type(actual)))


def check_funsor(x, inputs, output, data=None):
    """
    Check dims and shape modulo reordering.
    """
    assert isinstance(x, Funsor)
    assert dict(x.inputs) == dict(inputs)
    if output is not None:
        assert x.output == output
    if data is not None:
        if x.inputs == inputs:
            x_data = x.data
        else:
            x_data = x.align(tuple(inputs)).data
        if inputs or output.shape:
            assert (x_data == data).all()
        else:
            assert x_data == data


def assert_equiv(x, y):
    """
    Check that two funsors are equivalent up to permutation of inputs.
    """
    check_funsor(x, y.inputs, y.output, y.data)


def random_tensor(domain):
    """
    Creates a random :class:`torch.Tensor` suitable for a given
    :class:`~funsor.domains.Domain`.
    """
    assert isinstance(domain, Domain)
    if isinstance(domain.dtype, integer_types):
        return torch.multinomial(torch.ones(domain.dtype),
                                 domain.num_elements,
                                 replacement=True).reshape(domain.shape)
    elif domain.dtype == "real":
        return torch.randn(domain.shape)
    else:
        raise ValueError('unknown dtype: {}'.format(repr(domain.dtype)))
