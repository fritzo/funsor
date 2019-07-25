import collections

from multipledispatch.variadic import Variadic
from opt_einsum.paths import greedy

import funsor.interpreter as interpreter
from funsor.cnf import Contraction, anyop
from funsor.ops import DISTRIBUTIVE_OPS, AssociativeOp
from funsor.terms import Funsor, eager, normalize, reflect


@interpreter.dispatched_interpretation
def optimize(cls, *args):
    result = optimize.dispatch(cls, *args)
    if result is None:
        result = eager(cls, *args)
    return result


# TODO set a better value for this
REAL_SIZE = 3  # the "size" of a real-valued dimension passed to the path optimizer


optimize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[Funsor])(
    lambda r, b, v, *ts: optimize(Contraction, r, b, v, tuple(ts)))


@optimize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Funsor, Funsor)
@optimize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Funsor)
def eager_contract_base(red_op, bin_op, reduced_vars, *terms):
    return None  # eager.dispatch(Contraction, red_op, bin_op, reduced_vars, *terms)


@optimize.register(Contraction, AssociativeOp, AssociativeOp, frozenset, tuple)
def optimize_contract_finitary_funsor(red_op, bin_op, reduced_vars, terms):

    if red_op is anyop or bin_op is anyop or not (red_op, bin_op) in DISTRIBUTIVE_OPS:
        return None

    # build opt_einsum optimizer IR
    inputs = [frozenset(term.inputs) for term in terms]
    size_dict = {k: ((REAL_SIZE * v.num_elements) if v.dtype == 'real' else v.dtype)
                 for term in terms for k, v in term.inputs.items()}
    outputs = frozenset().union(*inputs) - reduced_vars

    # optimize path with greedy opt_einsum optimizer
    # TODO switch to new 'auto' strategy
    path = greedy(inputs, outputs, size_dict)

    # first prepare a reduce_dim counter to avoid early reduction
    reduce_dim_counter = collections.Counter()
    for input in inputs:
        reduce_dim_counter.update({d: 1 for d in input})

    operands = list(terms)
    for (a, b) in path:
        b, a = tuple(sorted((a, b), reverse=True))
        tb = operands.pop(b)
        ta = operands.pop(a)

        # don't reduce a dimension too early - keep a collections.Counter
        # and only reduce when the dimension is removed from all lhs terms in path
        reduce_dim_counter.subtract({d: 1 for d in reduced_vars & frozenset(ta.inputs.keys())})
        reduce_dim_counter.subtract({d: 1 for d in reduced_vars & frozenset(tb.inputs.keys())})

        # reduce variables that don't appear in other terms
        both_vars = frozenset(ta.inputs.keys()) | frozenset(tb.inputs.keys())
        path_end_reduced_vars = frozenset(d for d in reduced_vars & both_vars
                                          if reduce_dim_counter[d] == 0)

        # count new appearance of variables that aren't reduced
        reduce_dim_counter.update({d: 1 for d in reduced_vars & (both_vars - path_end_reduced_vars)})

        path_end = Contraction(red_op if path_end_reduced_vars else anyop, bin_op, path_end_reduced_vars, ta, tb)
        operands.append(path_end)

    # reduce any remaining dims, if necessary
    final_reduced_vars = frozenset(d for (d, count) in reduce_dim_counter.items()
                                   if count > 0) & reduced_vars
    if final_reduced_vars:
        path_end = path_end.reduce(red_op, final_reduced_vars)
    return path_end


def apply_optimizer(x):

    @interpreter.interpretation(interpreter._INTERPRETATION)
    def nested_optimize_interpreter(cls, *args):
        result = optimize.dispatch(cls, *args)
        if result is None:
            result = cls(*args)
        return result

    with interpreter.interpretation(nested_optimize_interpreter):
        return interpreter.reinterpret(x)
