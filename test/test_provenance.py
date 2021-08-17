# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pyro.ops.indexing import Vindex

from funsor.torch.provenance import ProvenanceTensor


@pytest.mark.parametrize(
    "op",
    ["log", "exp", "long"],
)
@pytest.mark.parametrize(
    "data,provenance",
    [
        (torch.tensor([1]), frozenset({"a", "b"})),
        (torch.tensor([1]), frozenset({"a"})),
    ],
)
def test_unary(op, data, provenance):
    if provenance is not None:
        data = ProvenanceTensor(data, provenance)

    expected = provenance
    actual = getattr(data, op)()._provenance
    assert actual == expected


@pytest.mark.parametrize(
    "data1,provenance1",
    [
        (torch.tensor([1]), frozenset({"a"})),
    ],
)
@pytest.mark.parametrize(
    "data2,provenance2",
    [
        (torch.tensor([2]), frozenset({"b"})),
        (torch.tensor([2]), None),
        (2, None),
    ],
)
def test_binary_add(data1, provenance1, data2, provenance2):
    if provenance1 is not None:
        data1 = ProvenanceTensor(data1, provenance1)
    if provenance2 is not None:
        data2 = ProvenanceTensor(data2, provenance2)

    expected = frozenset.union(
        *[m for m in (provenance1, provenance2) if m is not None]
    )
    actual = (data1 + data2)._provenance
    assert actual == expected


@pytest.mark.parametrize(
    "data1,provenance1",
    [
        (torch.tensor([0, 1]), frozenset({"a"})),
        (torch.tensor([0, 1]), None),
    ],
)
@pytest.mark.parametrize(
    "data2,provenance2",
    [
        (torch.tensor([0]), frozenset({"b"})),
        (torch.tensor([1]), None),
    ],
)
def test_indexing(data1, provenance1, data2, provenance2):
    if provenance1 is not None:
        data1 = ProvenanceTensor(data1, provenance1)
    if provenance2 is not None:
        data2 = ProvenanceTensor(data2, provenance2)

    expected = frozenset().union(
        *[m for m in (provenance1, provenance2) if m is not None]
    )
    actual = getattr(data1[data2], "_provenance", frozenset())
    assert actual == expected


@pytest.mark.parametrize(
    "data1,provenance1",
    [
        (torch.tensor([[0, 1], [2, 3]]), frozenset({"a"})),
        (torch.tensor([[0, 1], [2, 3]]), None),
    ],
)
@pytest.mark.parametrize(
    "data2,provenance2",
    [
        (torch.tensor([0.0, 1.0]), frozenset({"b"})),
        (torch.tensor([0.0, 1.0]), None),
    ],
)
@pytest.mark.parametrize(
    "data3,provenance3",
    [
        (torch.tensor([0, 1]), frozenset({"c"})),
        (torch.tensor([0, 1]), None),
    ],
)
def test_vindex(data1, provenance1, data2, provenance2, data3, provenance3):
    if provenance1 is not None:
        data1 = ProvenanceTensor(data1, provenance1)
    if provenance2 is not None:
        data2 = ProvenanceTensor(data2, provenance2)
    if provenance3 is not None:
        data3 = ProvenanceTensor(data3, provenance3)

    expected = frozenset().union(
        *[m for m in (provenance1, provenance2, provenance3) if m is not None]
    )
    actual = getattr(
        Vindex(data1)[data2.long().unsqueeze(-1), data3], "_provenance", frozenset()
    )
    assert actual == expected
