# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor
from funsor import ops
from funsor.adjoint import AdjointTape
from funsor.delta import Delta
from funsor.domains import Bint
from funsor.interpretations import DispatchedInterpretation
from funsor.tensor import Tensor, materialize
from funsor.terms import Approximate, Funsor, Number, Variable

answer_set = DispatchedInterpretation("answer_set")


@answer_set.register(Approximate, ops.OrOp, Funsor, Funsor, frozenset)
def answer_set_approximate(op, model, guide, approx_vars):

    # This computes all solutions in parallel, rather than streaming through
    # solutions, which has cost exponential in the number of variables.
    # Question: how might we stream?
    # Answer: Consider adding a Stream funsor. This may requires lazier
    #   computation of metatdata, e.g. a funsor.meta(-) function closer to
    #   @lazy_property. See https://github.com/pyro-ppl/funsor/issues/526
    m = materialize(model)
    assert isinstance(m, Tensor)

    # something like this:
    assignments = m.data.nonzero(as_tuple=True)
    answers = {k: Tensor(a)["answers"] for k, a in zip(m.inputs, assignments)}

    # Note a convex problem might now losslessly compress answers to a
    # convex generating set. Similarly a monotone problem might losslessly
    # compress to a set of maximal elements.

    return Delta(tuple((k, (a, Number(0.0))) for k, a in answers.items()))


def run(formula):
    with AdjointTape() as tape:
        truth = formula.reduce(ops.or_)
    print("_" * 80)
    with answer_set:
        assignment = tape.adjoint(ops.or_, ops.and_, truth, formula.input_vars)
    return assignment


def main():
    x = Variable("x", Bint[2])
    y = Variable("y", Bint[2])

    p = x | ~y
    assignment = run(p)
    print(assignment)


if __name__ == "__main__":
    funsor.set_backend("torch")
    main()
