import argparse
import os
import json
import uuid
import functools

import torch

import pyro
import pyro.poutine as poutine

import funsor.ops as ops
from funsor.interpreter import interpretation
from funsor.optimizer import apply_optimizer
from funsor.sum_product import sequential_sum_product, sum_product
from funsor.terms import lazy


from model import model_sequential, model_parallel, guide_sequential
from seal_data import prepare_seal


def aic_num_parameters(model, guide=None):
    """
    hacky AIC param count that includes all parameters in the model and guide
    """

    def _size(tensor):
        """product of shape"""
        s = 1
        for d in tensor.shape:
            s = s * d
        return s

    with poutine.block(), poutine.trace(param_only=True) as param_capture:
        model()
        guide()

    return sum(_size(node["value"]) for node in param_capture.trace.nodes.values())


def sequential_loss_fn(model, guide):
    # XXX ignore guide for now
    with interpretation(lazy):
        factors = model()
        plates = frozenset(['g', 'i'])
        eliminate = frozenset().union(*(f.inputs for f in factors))
        loss = sum_product(ops.logaddexp, ops.add, factors, eliminate, plates)
    loss = apply_optimizer(loss)
    assert not loss.inputs
    return -loss.data


def parallel_loss_fn(model, guide):
    # XXX ignore guide for now
    factors = model()
    t_term, new_factors = factors[0], factors[1:]
    result = sequential_sum_product(ops.logaddexp, ops.add,
                                    t_term, "t", {"y": "y(t=1)"})
    new_factors = [result] + new_factors

    plates = frozenset(['g', 'i'])
    eliminate = frozenset().union(*(f.inputs for f in new_factors))
    with interpretation(lazy):
        loss = sum_product(ops.logaddexp, ops.add, new_factors, eliminate, plates)
    loss = apply_optimizer(loss)
    assert not loss.inputs
    return -loss.data


def run_expt(args):

    data_dir = args["folder"]
    dataset = args["dataset"]
    assert dataset == "seal", "shark not working"
    seed = args["seed"]
    optim = args["optim"]
    lr = args["learnrate"]
    timesteps = args["timesteps"]
    schedule = [] if not args["schedule"] else [int(i) for i in args["schedule"].split(",")]
    random_effects = {"group": args["group"], "individual": args["individual"]}

    pyro.enable_validation(args["validation"])
    pyro.set_rng_seed(seed)  # reproducible random effect parameter init
    if args["cuda"]:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    filename = os.path.join(data_dir, "prep_seal_data.csv")
    config = prepare_seal(filename, random_effects)
    if args["smoke"]:
        timesteps = 1
        config["sizes"]["timesteps"] = 3

    if args["length"] > 0:
        config["sizes"]["timesteps"] = args["length"]

    if not args["parallel"]:
        model = functools.partial(model_sequential, config)  # for JITing
        guide = functools.partial(guide_sequential, config)
        loss_fn = sequential_loss_fn
    else:
        model = functools.partial(model_parallel, config)  # for JITing
        guide = functools.partial(guide_sequential, config)
        loss_fn = parallel_loss_fn

    if args["jit"]:
        loss_fn = torch.jit.trace(lambda: loss_fn(model, guide), ())
    else:
        loss_fn = functools.partial(loss_fn, model, guide)

    # count the number of parameters once
    num_parameters = aic_num_parameters(model, guide)

    losses = []

    # TODO support continuous random effects with monte carlo
    assert random_effects["group"] != "continuous"
    assert random_effects["individual"] != "continuous"

    with pyro.poutine.trace(param_only=True) as param_capture:
        loss_fn()
    params = [site["value"].unconstrained() for site in param_capture.trace.nodes.values()]
    if optim == "sgd":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optim == "lbfgs":
        optimizer = torch.optim.LBFGS(params, lr=lr)
    else:
        raise ValueError("{} not supported optimizer".format(optim))

    if schedule:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=schedule, gamma=0.5)
        schedule_step_loss = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        schedule_step_loss = True

    for t in range(timesteps):
        def closure():
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        scheduler.step(loss.item() if schedule_step_loss else t)
        losses.append(loss.item())
        print("Loss: {}, AIC[{}]: ".format(loss.item(), t),
              2. * loss + 2. * num_parameters)

    aic_final = 2. * losses[-1] + 2. * num_parameters
    print("AIC final: {}".format(aic_final))

    results = {}
    results["args"] = args
    results["sizes"] = config["sizes"]
    results["likelihoods"] = losses
    results["likelihood_final"] = losses[-1]
    results["aic_final"] = aic_final
    results["aic_num_parameters"] = num_parameters

    if args["resultsdir"] is not None and os.path.exists(args["resultsdir"]):
        re_str = "g" + ("n" if args["group"] is None else "d" if args["group"] == "discrete" else "c")
        re_str += "i" + ("n" if args["individual"] is None else "d" if args["individual"] == "discrete" else "c")
        results_filename = "expt_{}_{}_{}.json".format(dataset, re_str, str(uuid.uuid4().hex)[0:5])
        with open(os.path.join(args["resultsdir"], results_filename), "w") as f:
            json.dump(results, f)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="seal", type=str)
    parser.add_argument("-g", "--group", default="none", type=str)
    parser.add_argument("-i", "--individual", default="none", type=str)
    parser.add_argument("-f", "--folder", default="./", type=str)
    parser.add_argument("-o", "--optim", default="sgd", type=str)
    parser.add_argument("-lr", "--learnrate", default=0.05, type=float)
    parser.add_argument("-t", "--timesteps", default=1000, type=int)
    parser.add_argument("-r", "--resultsdir", default="./results", type=str)
    parser.add_argument("-s", "--seed", default=101, type=int)
    parser.add_argument("-l", "--length", default=-1, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--schedule", default="", type=str)
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    if args.group == "none":
        args.group = None
    if args.individual == "none":
        args.individual = None

    run_expt(vars(args))
