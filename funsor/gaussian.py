from __future__ import absolute_import, division, print_function

import math
import sys
from collections import OrderedDict

import six
import torch
from pyro.distributions.util import broadcast_shape
from six import add_metaclass, integer_types
from six.moves import reduce

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import reals
from funsor.integrate import Integrate, integrator
from funsor.montecarlo import monte_carlo
from funsor.ops import AddOp, NegOp, SubOp
from funsor.terms import Align, Binary, Funsor, FunsorMeta, Number, Subs, Unary, Variable, eager
from funsor.torch import Tensor, align_tensor, align_tensors, materialize
from funsor.util import lazy_property


def _issubshape(subshape, supershape):
    if len(subshape) > len(supershape):
        return False
    for sub, sup in zip(reversed(subshape), reversed(supershape)):
        if sub not in (1, sup):
            return False
    return True


def _log_det_tri(x):
    return x.diagonal(dim1=-1, dim2=-2).log().sum(-1)


def _det_tri(x):
    return x.diagonal(dim1=-1, dim2=-2).prod(-1)


def _mv(mat, vec):
    return torch.matmul(mat, vec.unsqueeze(-1)).squeeze(-1)


def _vmv(mat, vec):
    """
    Computes the inner product ``<vec | mat | vec>``.
    """
    vt = vec.unsqueeze(-2)
    v = vec.unsqueeze(-1)
    result = torch.matmul(vt, torch.matmul(mat, v))
    return result.squeeze(-1).squeeze(-1)


def _trace_mm(x, y):
    """
    Computes ``trace(x @ y)``.
    """
    assert x.dim() >= 2
    assert y.dim() >= 2
    xy = x * y
    return xy.reshape(xy.shape[:-2] + (-1,)).sum(-1)


def _sym_solve_mv(mat, vec):
    r"""
    Computes ``mat \ vec`` assuming mat is symmetric and usually positive definite,
    but falling back to general pseudoinverse if positive definiteness fails.
    """
    try:
        # Attempt to use stable positive definite math.
        tri = torch.inverse(torch.cholesky(mat))
        return _mv(tri.transpose(-1, -2), _mv(tri, vec))
    except RuntimeError as e:
        if 'not positive definite' not in e.message:
            _, exc_value, traceback = sys.exc_info()
            six.reraise(RuntimeError, e, traceback)

    # Fall back to pseudoinverse.
    return _mv(torch.pinverse(mat), vec)


def _compute_offsets(inputs):
    """
    Compute offsets of real inputs into the concatenated Gaussian dims.
    This ignores all int inputs.

    :param OrderedDict inputs: A schema mapping variable name to domain.
    :return: a pair ``(offsets, total)``.
    :rtype: tuple
    """
    assert isinstance(inputs, OrderedDict)
    offsets = {}
    total = 0
    for key, domain in inputs.items():
        if domain.dtype == 'real':
            offsets[key] = total
            total += domain.num_elements
    return offsets, total


def align_gaussian(new_inputs, old):
    """
    Align data of a Gaussian distribution to a new ``inputs`` shape.
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(old, Gaussian)
    loc = old.loc
    precision = old.precision

    # Align int inputs.
    # Since these are are managed as in Tensor, we can defer to align_tensor().
    new_ints = OrderedDict((k, d) for k, d in new_inputs.items() if d.dtype != 'real')
    old_ints = OrderedDict((k, d) for k, d in old.inputs.items() if d.dtype != 'real')
    if new_ints != old_ints:
        loc = align_tensor(new_ints, Tensor(loc, old_ints))
        precision = align_tensor(new_ints, Tensor(precision, old_ints))

    # Align real inputs, which are all concatenated in the rightmost dims.
    new_offsets, new_dim = _compute_offsets(new_inputs)
    old_offsets, old_dim = _compute_offsets(old.inputs)
    assert loc.shape[-1:] == (old_dim,)
    assert precision.shape[-2:] == (old_dim, old_dim)
    if new_offsets != old_offsets:
        old_loc = loc
        old_precision = precision
        loc = old_loc.new_zeros(old_loc.shape[:-1] + (new_dim,))
        precision = old_loc.new_zeros(old_loc.shape[:-1] + (new_dim, new_dim))
        for k1, new_offset1 in new_offsets.items():
            if k1 not in old_offsets:
                continue
            offset1 = old_offsets[k1]
            num_elements1 = old.inputs[k1].num_elements
            old_slice1 = slice(offset1, offset1 + num_elements1)
            new_slice1 = slice(new_offset1, new_offset1 + num_elements1)
            loc[..., new_slice1] = old_loc[..., old_slice1]
            for k2, new_offset2 in new_offsets.items():
                if k2 not in old_offsets:
                    continue
                offset2 = old_offsets[k2]
                num_elements2 = old.inputs[k2].num_elements
                old_slice2 = slice(offset2, offset2 + num_elements2)
                new_slice2 = slice(new_offset2, new_offset2 + num_elements2)
                precision[..., new_slice1, new_slice2] = old_precision[..., old_slice1, old_slice2]

    return loc, precision


class GaussianMeta(FunsorMeta):
    """
    Wrapper to convert between OrderedDict and tuple.
    """
    def __call__(cls, loc, precision, inputs):
        if isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.items())
        assert isinstance(inputs, tuple)
        return super(GaussianMeta, cls).__call__(loc, precision, inputs)


@add_metaclass(GaussianMeta)
class Gaussian(Funsor):
    """
    Funsor representing a batched joint Gaussian distribution as a log-density
    function.

    Note that :class:`Gaussian`s are not normalized, rather they are
    canonicalized to evaluate to zero at their maximum value (at ``loc``). This
    canonical form is useful because it allows :class:`Gaussian`s with
    incomplete information, i.e. zero eigenvalues in the precision matrix.
    These incomplete distributions arise when making low-dimensional
    observations on higher dimensional hidden state.
    """
    def __init__(self, loc, precision, inputs):
        assert isinstance(loc, torch.Tensor)
        assert isinstance(precision, torch.Tensor)
        assert isinstance(inputs, tuple)
        inputs = OrderedDict(inputs)

        # Compute total dimension of all real inputs.
        dim = sum(d.num_elements for d in inputs.values() if d.dtype == 'real')
        assert dim
        assert loc.dim() >= 1 and loc.size(-1) == dim
        assert precision.dim() >= 2 and precision.shape[-2:] == (dim, dim)

        # Compute total shape of all bint inputs.
        batch_shape = tuple(d.dtype for d in inputs.values()
                            if isinstance(d.dtype, integer_types))
        assert _issubshape(loc.shape, batch_shape + (dim,))
        assert _issubshape(precision.shape, batch_shape + (dim, dim))

        output = reals()
        super(Gaussian, self).__init__(inputs, output)
        self.loc = loc
        self.precision = precision
        self.batch_shape = batch_shape
        self.event_shape = (dim,)

    def __repr__(self):
        return 'Gaussian(..., ({}))'.format(' '.join(
            '({}, {}),'.format(*kv) for kv in self.inputs.items()))

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        subs = tuple((k, materialize(v)) for k, v in subs if k in self.inputs)
        if not subs:
            return self

        # Constants and Variables are eagerly substituted;
        # everything else is lazily substituted.
        lazy_subs = tuple((k, v) for k, v in subs
                          if not isinstance(v, (Number, Tensor, Variable)))
        var_subs = tuple((k, v) for k, v in subs if isinstance(v, Variable))
        int_subs = tuple((k, v) for k, v in subs if isinstance(v, (Number, Tensor))
                         if v.dtype != 'real')
        real_subs = tuple((k, v) for k, v in subs if isinstance(v, (Number, Tensor))
                          if v.dtype == 'real')
        if not (var_subs or int_subs or real_subs):
            return None  # entirely lazy

        # First perform any variable substitutions.
        if var_subs:
            rename = {k: v.name for k, v in var_subs}
            targets = frozenset(rename.values())
            for k, v in int_subs + real_subs + lazy_subs:
                if not targets.isdisjoint(v.inputs):
                    raise NotImplementedError('TODO alpha-convert')
            inputs = OrderedDict((rename.get(k, k), d) for k, d in self.inputs.items())
            if len(inputs) != len(self.inputs):
                raise ValueError("Variable substitution name conflict")
            var_result = Gaussian(self.loc, self.precision, inputs)
            return Subs(var_result, int_subs + real_subs + lazy_subs)

        # Next perform any integer substitution, i.e. slicing into a batch.
        if int_subs:
            int_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype != 'real')
            real_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype == 'real')
            tensors = [self.loc, self.precision]
            funsors = [Subs(Tensor(x, int_inputs), int_subs) for x in tensors]
            inputs = funsors[0].inputs.copy()
            inputs.update(real_inputs)
            int_result = Gaussian(funsors[0].data, funsors[1].data, inputs)
            return Subs(int_result, real_subs + lazy_subs)

        # Try to perform a complete substitution of all real variables, resulting in a Tensor.
        real_subs = OrderedDict(subs)
        assert real_subs and not int_subs
        if all(k in real_subs for k, d in self.inputs.items() if d.dtype == 'real'):
            # Broadcast all component tensors.
            int_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype != 'real')
            tensors = [Tensor(self.loc, int_inputs),
                       Tensor(self.precision, int_inputs)]
            tensors.extend(real_subs.values())
            inputs, tensors = align_tensors(*tensors)
            batch_dim = tensors[0].dim() - 1
            batch_shape = broadcast_shape(*(x.shape[:batch_dim] for x in tensors))
            (loc, precision), values = tensors[:2], tensors[2:]

            # Form the concatenated value.
            offsets, event_size = _compute_offsets(self.inputs)
            value = loc.new_empty(batch_shape + (event_size,))
            for k, value_k in zip(real_subs, values):
                offset = offsets[k]
                value_k = value_k.reshape(value_k.shape[:batch_dim] + (-1,))
                assert value_k.size(-1) == self.inputs[k].num_elements
                value[..., offset: offset + self.inputs[k].num_elements] = value_k

            # Evaluate the non-normalized log density.
            result = -0.5 * _vmv(precision, value - loc)
            result = Tensor(result, inputs)
            assert result.output == reals()
            return Subs(result, lazy_subs)

        raise NotImplementedError('TODO implement partial substitution of real variables')

    @lazy_property
    def _log_normalizer(self):
        dim = self.loc.size(-1)
        log_det_term = _log_det_tri(torch.cholesky(self.precision))
        data = -log_det_term + 0.5 * math.log(2 * math.pi) * dim
        inputs = OrderedDict((k, v) for k, v in self.inputs.items() if v.dtype != 'real')
        return Tensor(data, inputs)

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            # Marginalize out real variables, but keep mixtures lazy.
            assert all(v in self.inputs for v in reduced_vars)
            real_vars = frozenset(k for k, d in self.inputs.items() if d.dtype == "real")
            reduced_reals = reduced_vars & real_vars
            reduced_ints = reduced_vars - real_vars
            if not reduced_reals:
                return None  # defer to default implementation

            inputs = OrderedDict((k, d) for k, d in self.inputs.items() if k not in reduced_reals)
            if reduced_reals == real_vars:
                result = self._log_normalizer
            else:
                int_inputs = OrderedDict((k, v) for k, v in inputs.items() if v.dtype != 'real')
                offsets, _ = _compute_offsets(self.inputs)
                index = []
                for key, domain in inputs.items():
                    if domain.dtype == 'real':
                        index.extend(range(offsets[key], offsets[key] + domain.num_elements))
                index = torch.tensor(index)

                loc = self.loc[..., index]
                self_scale_tri = torch.inverse(torch.cholesky(self.precision)).transpose(-1, -2)
                self_covariance = torch.matmul(self_scale_tri, self_scale_tri.transpose(-1, -2))
                covariance = self_covariance[..., index.unsqueeze(-1), index]
                scale_tri = torch.cholesky(covariance)
                inv_scale_tri = torch.inverse(scale_tri)
                precision = torch.matmul(inv_scale_tri.transpose(-1, -2), inv_scale_tri)
                reduced_dim = sum(self.inputs[k].num_elements for k in reduced_reals)
                log_det_term = _log_det_tri(self_scale_tri) - _log_det_tri(scale_tri)
                log_prob = Tensor(log_det_term + 0.5 * math.log(2 * math.pi) * reduced_dim, int_inputs)
                result = log_prob + Gaussian(loc, precision, inputs)

            return result.reduce(ops.logaddexp, reduced_ints)

        elif op is ops.add:
            raise NotImplementedError('TODO product-reduce along a plate dimension')

        return None  # defer to default implementation

    def unscaled_sample(self, sampled_vars, sample_inputs=None):
        # Sample only the real variables.
        sampled_vars = frozenset(k for k, v in self.inputs.items()
                                 if k in sampled_vars if v.dtype == 'real')
        if not sampled_vars:
            return self

        # Partition inputs into sample_inputs + int_inputs + real_inputs.
        if sample_inputs is None:
            sample_inputs = OrderedDict()
        else:
            sample_inputs = OrderedDict((k, d) for k, d in sample_inputs.items()
                                        if k not in self.inputs)
        sample_shape = tuple(int(d.dtype) for d in sample_inputs.values())
        int_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype != 'real')
        real_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype == 'real')
        inputs = sample_inputs.copy()
        inputs.update(int_inputs)

        if sampled_vars == frozenset(real_inputs):
            scale_tri = torch.inverse(torch.cholesky(self.precision)).transpose(-1, -2)
            assert self.loc.shape == scale_tri.shape[:-1]
            shape = sample_shape + self.loc.shape
            white_noise = torch.randn(shape)
            sample = self.loc + _mv(scale_tri, white_noise)
            offsets, _ = _compute_offsets(real_inputs)
            results = []
            for key, domain in real_inputs.items():
                data = sample[..., offsets[key]: offsets[key] + domain.num_elements]
                data = data.reshape(shape[:-1] + domain.shape)
                point = Tensor(data, inputs)
                assert point.output == domain
                results.append(Delta(key, point))
            results.append(self._log_normalizer)
            return reduce(ops.add, results)

        raise NotImplementedError('TODO implement partial sampling of real variables')


@eager.register(Binary, AddOp, Gaussian, Gaussian)
def eager_add_gaussian_gaussian(op, lhs, rhs):
    # Fuse two Gaussians by adding their log-densities pointwise.
    # This is similar to a Kalman filter update, but also keeps track of
    # the marginal likelihood which accumulates into a Tensor.

    # Align data.
    inputs = lhs.inputs.copy()
    inputs.update(rhs.inputs)
    int_inputs = OrderedDict((k, v) for k, v in inputs.items() if v.dtype != 'real')
    lhs_loc, lhs_precision = align_gaussian(inputs, lhs)
    rhs_loc, rhs_precision = align_gaussian(inputs, rhs)

    # Fuse aligned Gaussians.
    precision_loc = _mv(lhs_precision, lhs_loc) + _mv(rhs_precision, rhs_loc)
    precision = lhs_precision + rhs_precision
    loc = _sym_solve_mv(precision, precision_loc)
    quadratic_term = _vmv(lhs_precision, loc - lhs_loc) + _vmv(rhs_precision, loc - rhs_loc)
    likelihood = Tensor(-0.5 * quadratic_term, int_inputs)
    return likelihood + Gaussian(loc, precision, inputs)


@eager.register(Binary, SubOp, Gaussian, (Funsor, Align, Gaussian))
@eager.register(Binary, SubOp, (Funsor, Align), Gaussian)
def eager_sub(op, lhs, rhs):
    return lhs + -rhs


@eager.register(Unary, NegOp, Gaussian)
def eager_neg(op, arg):
    precision = -arg.precision
    return Gaussian(arg.loc, precision, arg.inputs)


@eager.register(Integrate, Gaussian, Variable, frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    real_vars = frozenset(k for k in reduced_vars if log_measure.inputs[k].dtype == 'real')
    if real_vars:
        assert real_vars == frozenset([integrand.name])
        data = log_measure.loc * log_measure._log_normalizer.data.exp().unsqueeze(-1)
        data = data.reshape(log_measure.loc.shape[:-1] + integrand.output.shape)
        inputs = OrderedDict((k, d) for k, d in log_measure.inputs.items() if d.dtype != 'real')
        return Tensor(data, inputs)

    return None  # defer to default implementation


@eager.register(Integrate, Gaussian, Gaussian, frozenset)
@integrator
def eager_integrate(log_measure, integrand, reduced_vars):
    real_vars = frozenset(k for k in reduced_vars if log_measure.inputs[k].dtype == 'real')
    if real_vars:

        lhs_reals = frozenset(k for k, d in log_measure.inputs.items() if d.dtype == 'real')
        rhs_reals = frozenset(k for k, d in integrand.inputs.items() if d.dtype == 'real')
        if lhs_reals == real_vars and rhs_reals <= real_vars:
            inputs = OrderedDict((k, d) for t in (log_measure, integrand)
                                 for k, d in t.inputs.items())
            lhs_loc, lhs_precision = align_gaussian(inputs, log_measure)
            rhs_loc, rhs_precision = align_gaussian(inputs, integrand)

            # Compute the expectation of a non-normalized quadratic form.
            # See "The Matrix Cookbook" (November 15, 2012) ss. 8.2.2 eq. 380.
            # http://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
            lhs_scale_tri = torch.inverse(torch.cholesky(lhs_precision)).transpose(-1, -2)
            lhs_covariance = torch.matmul(lhs_scale_tri, lhs_scale_tri.transpose(-1, -2))
            dim = lhs_loc.size(-1)
            norm = _det_tri(lhs_scale_tri) * (2 * math.pi) ** (0.5 * dim)
            data = -0.5 * norm * (_vmv(rhs_precision, lhs_loc - rhs_loc) +
                                  _trace_mm(rhs_precision, lhs_covariance))
            inputs = OrderedDict((k, d) for k, d in inputs.items() if k not in reduced_vars)
            result = Tensor(data, inputs)
            return result.reduce(ops.add, reduced_vars - real_vars)

        raise NotImplementedError('TODO implement partial integration')

    return None  # defer to default implementation


@monte_carlo.register(Integrate, Gaussian, Funsor, frozenset)
@integrator
def monte_carlo_integrate(log_measure, integrand, reduced_vars):
    real_vars = frozenset(k for k in reduced_vars if log_measure.inputs[k].dtype == 'real')
    if real_vars:
        log_measure = log_measure.sample(real_vars, monte_carlo.sample_inputs)
        reduced_vars = reduced_vars | frozenset(monte_carlo.sample_inputs)
        return Integrate(log_measure, integrand, reduced_vars)

    return None  # defer to default implementation


__all__ = [
    'Gaussian',
    'align_gaussian',
]
