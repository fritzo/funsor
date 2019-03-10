from __future__ import absolute_import, division, print_function

import argparse

import torch

import funsor
import funsor.distributions as dist
import funsor.ops as ops
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.terms import lazy


def main(args):
    # Declare parameters.
    trans_noise = torch.tensor(0.1, requires_grad=True)
    emit_noise = torch.tensor(0.5, requires_grad=True)
    params = [trans_noise, emit_noise]

    # A Gaussian HMM model.
    def model(data):
        log_prob = funsor.to_funsor(0.)

        x_curr = funsor.Tensor(torch.tensor(0.))
        for t, y in enumerate(data):
            x_prev = x_curr

            # A delayed sample statement.
            x_curr = funsor.Variable('x_{}'.format(t), funsor.reals())
            log_prob += dist.Normal(x_prev, trans_noise, value=x_curr)

            if not args.lazy and isinstance(x_prev, funsor.Variable):
                log_prob = log_prob.reduce(ops.logaddexp, x_prev.name)

            log_prob += dist.Normal(x_curr, emit_noise, value=y)

        log_prob = log_prob.reduce(ops.logaddexp)
        return log_prob

    # Train model parameters.
    print('---- training ----')
    data = torch.randn(args.time_steps)
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()
        if args.lazy:
            with interpretation(lazy):
                log_prob = apply_optimizer(model(data))
            log_prob = reinterpret(log_prob)
        else:
            log_prob = model(data)
        assert not log_prob.inputs, 'free variables remain'
        loss = -log_prob.data
        loss.backward()
        optim.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kalman filter example")
    parser.add_argument("-t", "--time-steps", default=10, type=int)
    parser.add_argument("-n", "--train-steps", default=101, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--lazy", action='store_true')
    parser.add_argument("--filter", action='store_true')
    parser.add_argument("--xfail-if-not-implemented", action='store_true')
    args = parser.parse_args()

    if args.xfail_if_not_implemented:
        try:
            main(args)
        except NotImplementedError:
            print('XFAIL')
    else:
        main(args)
