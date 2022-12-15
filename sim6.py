# $ python3 -m sim6 -csteps 10 -cobs 10 -hso 0

import pyro
import torch
import pyro.distributions as pdist
from pyro.ops.indexing import Vindex
import pyro.poutine as poutine
from pyro.poutine.runtime import _PYRO_PARAM_STORE

from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.infer import config_enumerate, infer_discrete

import torch.distributions.constraints as constraints
from torch.distributions import transform_to

import matplotlib.pyplot as plt
import sys

import progresses.progress_cmd as progress_cmd
ProgressCmd = progress_cmd.ProgressCmd

# from parallel.threads import par_wrap


# FOR models definition:
def init_model0(init_data: dict, counts, dbg=False):
    '''To generate the init observable.
    Params in model0 will be fixed with that to learn. 
    '''
    if dbg:
        print("FROM init_model0 ==========")
    # three types of vagons:
    beta_vagons = init_data["beta_vagons"]
    alpha_vagons = init_data["alpha_vagons"]             
    
    # four types of boxes (mixed differently for each vagon):
    beta_boxes = init_data["beta_boxes"]
    alpha_boxes = init_data["alpha_boxes"]

    if dbg:
        print("beta_boxes", beta_boxes)

    # coins proportion for each of four box types:
    beta_coins = init_data["beta_coins"]
    alpha_coins = init_data["alpha_coins"]

    # coins either fair or not:
    beta_flips = init_data["beta_flips"]
    alpha_flips = init_data["alpha_flips"]
    
    res = model0([None]*counts, [None]*counts, [None]*counts, [None]*counts,
                 params_flips=(alpha_flips, beta_flips),
                 params_coins=(alpha_coins, beta_coins),
                 params_boxes=(alpha_boxes, beta_boxes),
                 params_vagons=(alpha_vagons, beta_vagons), dbg=dbg)
    return res


@config_enumerate(default='parallel')
def cond_model0(obs_flips, obs_coins, obs_boxes, obs_vagons, dbg=False):
    '''
    To generate conditional model. Params in that model will be generated
    randomly.
    Returning model will be used for pyro learning with guide. 
    It must replace all variables by a guide ones if they names match.
    '''
    # three types of vagons:
    adist_vagons = pdist.Uniform(0.01*torch.zeros(3), 0.5*torch.ones(3))
    bdist_vagons = pdist.Uniform(0.51*torch.ones(3), 0.99*torch.ones(3))
    alpha_vagons, beta_vagons = (0.01*torch.zeros(3), 0.99*torch.ones(3))
    # alpha_vagons, beta_vagons = (adist_vagons.sample(), bdist_vagons.sample())
    if dbg:
        print("\nalpha_vagons:")
        print(alpha_vagons)
        print("\nbeta_vagons:")
        print(beta_vagons)

    # four types of boxes (mixed differently for each vagon):
    adist_boxes = pdist.Uniform(0.01*torch.zeros((3, 4)), 0.5*torch.ones((3, 4)))
    bdist_boxes = pdist.Uniform(0.51*torch.ones((3, 4)), 0.99*torch.ones((3, 4)))
    alpha_boxes, beta_boxes = (0.01*torch.zeros((3, 4)), 0.99*torch.ones((3, 4)))
    # alpha_boxes, beta_boxes = (adist_boxes.sample(), bdist_boxes.sample())
    if dbg:
        print("\nalpha_boxes:")
        print(alpha_boxes)
        print("\nbeta_boxes:")
        print(beta_boxes)

    # coins proportion for each of four box types:
    adist_coins = pdist.Uniform(0.01*torch.zeros(4), 0.5*torch.ones(4))
    bdist_coins = pdist.Uniform(0.51*torch.ones(4), 0.99*torch.ones(4))
    alpha_coins, beta_coins = (0.01*torch.zeros(4), 0.99*torch.ones(4))
    # alpha_coins, beta_coins = (adist_coins.sample(), bdist_coins.sample())
    if dbg:
        print("\nalpha_coins:")
        print(alpha_coins)
        print("\nbeta_coins:")
        print(beta_coins)

    # coins either fair or not:
    adist_flips = pdist.Uniform(0.01*torch.zeros(2), 0.5*torch.ones(2))
    bdist_flips = pdist.Uniform(0.51*torch.ones(2), 0.99*torch.ones(2))
    alpha_flips, beta_flips = (0.01*torch.zeros(2), 0.99*torch.ones(2))
    # alpha_flips, beta_flips = (adist_flips.sample(), bdist_flips.sample())
    if dbg:
        print("\nalpha_flips:")
        print(alpha_flips)
        print("\nbeta_flips:")
        print(beta_flips)

    res = model0(obs_flips, obs_coins, obs_boxes, obs_vagons,
                 params_flips=(alpha_flips, beta_flips),
                 params_coins=(alpha_coins, beta_coins),
                 params_boxes=(alpha_boxes, beta_boxes),
                 params_vagons=(alpha_vagons, beta_vagons), dbg=dbg)
    return res


def model0(obs_flips, obs_coins, obs_boxes, obs_vagons,
           params_flips, params_coins,
           params_boxes, params_vagons, dbg=True):
    '''For generating the data. Also during the learning
    all `*_params` will be ignored by replacing it in elbo.
    See minipyro.py elbo def'''
    
    if dbg:
        print("FROM model0 =======")
    alpha_vagons, beta_vagons = params_vagons    
    vagons_param = pyro.sample(
        "param_vagons", pdist.Uniform(alpha_vagons, beta_vagons))
    # vagons_param = transform_to(constraints.interval(0.01, 0.99))(vagons_param)

    alpha_boxes, beta_boxes = params_boxes
    boxes_param = pyro.sample(
        "param_boxes", pdist.Uniform(alpha_boxes, beta_boxes))
    # boxes_param = transform_to(constraints.interval(0.01, 0.99))(boxes_param)

    alpha_coins, beta_coins = params_coins    
    coins_param = pyro.sample(
        "param_coins", pdist.Uniform(alpha_coins, beta_coins))
    # coins_param = transform_to(constraints.interval(0.01, 0.99))(coins_param)

    alpha_flips, beta_flips = params_flips
    flips_param = pyro.sample(
        "param_flips", pdist.Uniform(alpha_flips, beta_flips))
    # flips_param = transform_to(constraints.interval(0.01, 0.99))(flips_param)

    flips = []
    coins = []
    boxes = []
    vagons = []
    
    # len(obs_flips)
    for i in pyro.plate("vagons_plate", size=len(obs_vagons)):
        if dbg:
            print("vagons_param:", vagons_param)
        if obs_vagons[i] is not None:
            vagon = pyro.sample(
                "vagon_%d" % i, pdist.Categorical(vagons_param),
                obs=obs_vagons[i])
        else:
            vagon = pyro.sample(
                "vagon_%d" % i, pdist.Categorical(vagons_param))
        vagons.append(vagon)
        if dbg:
            print("vagon:", vagon)
        
        id_box = Vindex(boxes_param)[vagon.type(torch.long)]
        if dbg:
            print("boxes_param:", boxes_param)
            print("id_box: ", id_box)
        if obs_boxes[i] is not None:
            box = pyro.sample(
                "box_%d" % i, pdist.Categorical(id_box),
                obs=obs_boxes[i])
        else:
            box = pyro.sample("box_%d" % i, pdist.Categorical(id_box))
        # print("box.shape:", box.shape)
        boxes.append(box)
        # print("box: ", box)
        
        id_coin = Vindex(coins_param)[box.type(torch.long)]
        if dbg:
            print("coins_param:", coins_param)
            print("id_coin:", id_coin)
        if obs_coins[i] is not None:
            coin = pyro.sample(
                "coin_%d" % i, pdist.Bernoulli(id_coin),
                obs=obs_coins[i])
        else:
            coin = pyro.sample("coin_%d" % i, pdist.Bernoulli(id_coin))
        coins.append(coin)
        # print("coin:", coin)
        
        id_flip = Vindex(flips_param)[coin.type(torch.long)]
        if dbg:
            print("flips_param:", flips_param)
            print("id_flip:", id_flip)
        if obs_flips[i] is not None:
            flip = pyro.sample(
                "flip_%d" % i, pdist.Bernoulli(id_flip),
                obs=obs_flips[i])
            # with pyro.mask(obs_flips_mask)
            # flip = pyro.sample(
            #    "flip", pdist.Bernoulli(id_flip),
            #    obs=obs_flips.index_select(0, ind))
        else:
            flip = pyro.sample(
                "flip_%d" % i, pdist.Bernoulli(id_flip))
        flips.append(flip)
        # print("flips", flips)
    if dbg:
        print("END FROM model0 =======")
    
    return list(map(
        torch.tensor, 
        # lambda x: torch.tensor(x if len(x[0].shape) != 0 else [x]),
        (flips, coins, boxes, vagons)))


def guide_model0(obs_flips, obs_coins, obs_boxes, obs_vagons, dbg=False):
    
    if dbg:
        print("FROM guide_model0 =======")
    # import pdb; pdb.set_trace()
    alow_bound = 0.01
    ahigh_bound = 0.5
    blow_bound = 0.51
    bhigh_bound = 0.99

    # three types of vagons:
    adist_vagons = pyro.param(
        "alpha_vagons", pdist.Uniform(
            alow_bound*torch.zeros(3), ahigh_bound*torch.ones(3)).sample(),
        constraint=constraints.interval(alow_bound, ahigh_bound))
    bdist_vagons = pyro.param(
        "beta_vagons", pdist.Uniform(
            blow_bound*torch.ones(3), bhigh_bound*torch.ones(3)).sample(),
        constraint=constraints.interval(blow_bound, bhigh_bound))
    alpha_vagons, beta_vagons = (adist_vagons, bdist_vagons)
    vagons_params = pyro.sample(
        "param_vagons", pdist.Uniform(alpha_vagons, beta_vagons))
    if dbg:
        print("\nalpha_vagons:")
        print(alpha_vagons)
        print("\nbeta_vagons:")
        print(beta_vagons)
        print("\nvagons_params:")
        print(vagons_params)
        print("_PYRO_PARAM_STORE:alpha_vagons:")
        print(_PYRO_PARAM_STORE._params["alpha_vagons"])
        print("_PYRO_PARAM_STORE:alpha_vagons.unconstrained:")
        print(_PYRO_PARAM_STORE["alpha_vagons"])

        print("_PYRO_PARAM_STORE:beta_vagons:")
        print(_PYRO_PARAM_STORE._params["beta_vagons"])
        print("_PYRO_PARAM_STORE:beta_vagons.unconstrained:")
        print(_PYRO_PARAM_STORE["beta_vagons"])

    # four types of boxes (mixed differently for each vagon):
    adist_boxes = pyro.param(
        "alpha_boxes", pdist.Uniform(
            alow_bound*torch.zeros((3, 4)),
            ahigh_bound*torch.ones((3, 4))).sample(),
        constraint=constraints.interval(alow_bound, ahigh_bound))
    bdist_boxes = pyro.param(
        "beta_boxes", pdist.Uniform(
            blow_bound*torch.ones((3, 4)),
            bhigh_bound*torch.ones((3, 4))).sample(),
        constraint=constraints.interval(blow_bound, bhigh_bound))
    alpha_boxes, beta_boxes = (adist_boxes, bdist_boxes)
    boxes_params = pyro.sample(
        "param_boxes", pdist.Uniform(alpha_boxes, beta_boxes))
    if dbg:
        print("\nalpha_boxes:")
        print(alpha_boxes)
        print("\nbeta_boxes:")
        print(beta_boxes)
        print("\nboxes_params:")
        print(boxes_params)
        print("_PYRO_PARAM_STORE:alpha_boxes:")
        print(_PYRO_PARAM_STORE._params["alpha_boxes"])
        print("_PYRO_PARAM_STORE:alpha_boxes.unconstrained:")
        print(_PYRO_PARAM_STORE["alpha_boxes"])

        print("_PYRO_PARAM_STORE:beta_boxes:")
        print(_PYRO_PARAM_STORE._params["beta_boxes"])
        print("_PYRO_PARAM_STORE:beta_boxes.unconstrained:")
        print(_PYRO_PARAM_STORE["beta_boxes"])

    # coins proportion for each of four box types:
    adist_coins = pyro.param(
        "alpha_coins",
        pdist.Uniform(
            alow_bound*torch.zeros(4),
            ahigh_bound*torch.ones(4)).sample(),
        constraint=constraints.interval(alow_bound, ahigh_bound))
    bdist_coins = pyro.param(
        "beta_coins", pdist.Uniform(
            blow_bound*torch.ones(4), bhigh_bound*torch.ones(4)).sample(),
        constraint=constraints.interval(blow_bound, bhigh_bound))
    alpha_coins, beta_coins = (adist_coins, bdist_coins)
    coins_params = pyro.sample(
        "param_coins", pdist.Uniform(alpha_coins, beta_coins))
    if dbg:
        print("\nalpha_coins:")
        print(alpha_coins)
        print("\nbeta_coins:")
        print(beta_coins)
        print("\ncoins_params")
        print(coins_params)
        print("_PYRO_PARAM_STORE:alpha_coins:")
        print(_PYRO_PARAM_STORE._params["alpha_coins"])
        print("_PYRO_PARAM_STORE:alpha_coins.unconstrained:")
        print(_PYRO_PARAM_STORE["alpha_coins"])

        print("_PYRO_PARAM_STORE:beta_coins:")
        print(_PYRO_PARAM_STORE._params["beta_coins"])
        print("_PYRO_PARAM_STORE:beta_coins.unconstrained:")
        print(_PYRO_PARAM_STORE["beta_coins"])

    # coins either fair or not:
    adist_flips = pyro.param(
        "alpha_flips", pdist.Uniform(
            alow_bound*torch.zeros(2), ahigh_bound*torch.ones(2)).sample(),
        constraint=constraints.interval(alow_bound, ahigh_bound))
    bdist_flips = pyro.param(
        "beta_flips", pdist.Uniform(
            blow_bound*torch.ones(2), bhigh_bound*torch.ones(2)).sample(),
        constraint=constraints.interval(blow_bound, bhigh_bound))
    alpha_flips, beta_flips = (adist_flips, bdist_flips)

    flips_params = pyro.sample(
        "param_flips", pdist.Uniform(alpha_flips, beta_flips))

    if dbg:
        print("\nalpha_flips:")
        print(alpha_flips)
        print("\nbeta_flips:")
        print(beta_flips)
        print("\nflips_params:")
        print(flips_params)
        print("_PYRO_PARAM_STORE:alpha_flips:")
        print(_PYRO_PARAM_STORE._params["alpha_flips"])
        print("_PYRO_PARAM_STORE:alpha_flips.unconstrained:")
        print(_PYRO_PARAM_STORE["alpha_flips"])

        print("_PYRO_PARAM_STORE:beta_flips:")
        print(_PYRO_PARAM_STORE._params["beta_flips"])
        print("_PYRO_PARAM_STORE:beta_flips.unconstrained:")
        print(_PYRO_PARAM_STORE["beta_flips"])

    if dbg:
        print("END FROM guide_model0 =======")
# END FOR models definition


# FOR learning:
def train(init_data, steps_counts=3, obs_counts=10, hso=0):
    '''
    - ``hso`` -- if less then 0 use usual obs (flips only)
    (from `gen_obs_low_only`),
    if 0 use flips and boxes (with `gen_obs_flips_and_boxes`),
    if hso>0 use sparse with hso as a count for choosing 
    from same amount of flips and boxes
    (with `gen_obs_flips_and_boxes_rand`).
    '''
    if hso > 0:
        obs = gen_obs_flips_and_boxes_rand(init_data, hso, obs_counts)
        print(
            "gen_obs_flips_and_boxes(rand_count=%s, obs_counts=%s) used"
            % (hso, obs_counts))
    elif hso == 0:
        obs = gen_obs_flips_and_boxes(init_data, obs_counts)
        print("gen_obs_flips_and_boxes(obs_counts=%s) used"
              % (obs_counts))
    else:
        obs = gen_obs_low_only(init_data, obs_counts)
        print("gen_obs_low_only(obs_counts=%s) used"
              % (obs_counts))

    pyro.clear_param_store()

    # set up the optimizer
    # adam_params = {"lr": 0.0001}
    adam_params = {"lr": 0.0001, "betas": (0.90, 0.999)}
    adam_params1 = {"lr": 0.01, "betas": (0.90, 0.999)}
    optim_adam = Adam(adam_params1)
    optim_sgd1 = pyro.optim.SGD({"lr": 0.001, "momentum":0.1})
    optim_sgd2 = pyro.optim.SGD({"lr": 0.01, "momentum":0.1})
    optim_sgd3 = pyro.optim.SGD({"lr": 0.1, "momentum":0.1})

    # elbo = TraceEnum_ELBO(max_plate_nesting=1)
    # setup the inference algorithm
    # svi = SVI(cond_model0, guide_model0, optimizer, loss=elbo)
    svi = SVI(cond_model0, guide_model0, optim_sgd3, loss=Trace_ELBO())

    elbo = Trace_ELBO()
    # import pdb; pdb.set_trace()
    # loss = elbo.loss(cond_model0, guide_model0, *obs, dbg=True)
    # print("loss:", loss)
    # do gradient steps
    progress = ProgressCmd(steps_counts)
    losses = []
    for step in range(steps_counts):
        loss = svi.step(*obs, dbg=False)
        losses.append(loss)
        progress.succ(step)
        # print("loss: ", loss)
    progress.print_end()

    labels_init = ["params init"]
    print("init arams")
    for name_key in _PYRO_PARAM_STORE._params:
        pname = name_key + ": " + str(init_data[name_key])
        labels_init.append(pname)
  
    labels_out = ["params res:"]
    print("_PYRO_PARAM_STORE constrained params:")
    for name_key in _PYRO_PARAM_STORE._params:
        # return constrained param:
        pname = name_key + ": " + str(_PYRO_PARAM_STORE[name_key])
        print(pname)
        labels_out.append(pname)

    figure = plt.figure(figsize=(20, 10))
    # ax = figure.add_subplot(1, 1, 1)
    # plt.axis([0, 10, 0, 10])
    print("xlim", plt.xlim())
    print("ylim", plt.ylim())
    # ax.plot(losses)

    to_plot0 = plt.plot(losses, figure=figure)[0]
    x0, x1, y0, y1 = to_plot0.axes.axis()
    print("x0, x1, y0, y1: ", [x0, x1, y0, y1])
    plt.text(x0, 0.93*y1, "\n".join(labels_init))
    plt.text(0.7*x1, 0.93*y1, "\n".join(labels_out))
    # to_plot1 = plt.plot(losses)[0]
    # plt.plot(losses, label="\n".join(labels))
    # plt.title("losses")
    # plt.xlabel("steps")
    # to_plot0.figure.axes[0].axis([0, 1, 0, 1])
    
    '''
    plt.legend([to_plot0], [
        "\n".join(labels_init)], loc="lower left")
    plt.legend([to_plot1], [
        "\n".join(labels_out)], loc="lower right")
    '''
    outfilename = (
        "hso_steps_counts_%s_obs_counts_%s_hso_%s.svg"
        % (steps_counts, obs_counts, hso))
    plt.savefig(outfilename)
    print("saved to ", outfilename)
    plt.show()
    '''

    elbo = TraceEnum_ELBO(max_plate_nesting=0)
    # poutine.trace(guide).
    elbo.loss(model_par, guide);
    '''
# END FOR


# FOR generate hso:
def gen_obs_flips_and_boxes_rand(init_data, rand_count, counts, dbg=False):
    '''Generate flips and boxes observable with same count
    and choose `rand_count` samples from random but same
    positions for both boxes and flips.
    '''
    res_flips, res_coins, res_boxes, res_vagons = gen(init_data, counts)
    if dbg:
        print("res_flips:")
        print(res_flips)
        print("res_coins:")
        print(res_coins)
        print("res_boxes:")
        print(res_boxes)
        print("res_vagons:")
        print(res_vagons)
        
    mask = gen_mask(rand_count, counts)
    obs_flips = put_obs([None]*counts, res_flips, mask)
    obs_boxes = put_obs([None]*counts, res_boxes, mask)
    return (obs_flips, [None]*counts, obs_boxes, [None]*counts)


def gen_obs_flips_and_boxes(init_data, counts):
    '''Generate flips and boxes observable with same count'''
    res_flips, res_coins, res_boxes, res_vagons = gen(init_data, counts)
    
    return (res_flips, [None]*counts, res_boxes, [None]*counts)
    

def gen_obs_low_only(init_data, counts):
    '''Generate only low level observable'''
    res_flips, res_coins, res_boxes, res_vagons = gen(init_data, counts)
    
    return (res_flips, [None]*counts, [None]*counts, [None]*counts)


def add_obs(obs: list, res, k, counts)->list:

    '''add k observable from `res` to `obs` from random but
    according places (so `obs[i]` will be equal to `res[i]`)'''
    
    mask = gen_mask(k, counts)
    return put_obs(obs, res, mask)

    
def gen_mask(k: int, counts: int)->torch.tensor:
    '''Generate/choose k places from counts'''
 
    mask = torch.zeros(counts)
    mask[:k] = 1
    mask = mask[torch.randperm(counts)].numpy()
    mask = torch.BoolTensor(mask)
    return mask


def put_obs(obs: list, res, mask)->list:
    ''' obs[mask] = res[mask] '''
    # print("mask_flips:")
    # print(mask_flips)
    for idx, m in enumerate(mask):
        if m:
            obs[idx] = res[idx]
    return obs


def gen(init_data, counts=10):
    # obs_flips = [None]*counts
    # obs_coins = [None]*counts
    # obs_boxes = [None]*counts
    # obs_vagons = [None]*counts
    res_flips, res_coins, res_boxes, res_vagons = init_model0(
        init_data, counts)
    # res_flips, res_coins, res_boxes, res_vagons = init_model0(
    #     obs_flips, obs_coins, obs_boxes, obs_vagons)
    # model0(torch.ones(10))
    return (res_flips, res_coins, res_boxes, res_vagons)
# END FOR generate hso

# FOR tests:
def test_model0_replay(init_data):
    '''To see if params in model are the same as in the guide,
    which have according names.'''

    obs = init_model0(init_data, 10, dbg=False)
    guide_trace = poutine.trace(guide_model0).get_trace(*obs)
    print("guide param_boxes: ", guide_trace.nodes["param_boxes"]["value"])
    print("\nguide param_flips: ", guide_trace.nodes["param_flips"]["value"])
    
    cm = poutine.replay(cond_model0, trace=guide_trace)
    model_trace = poutine.trace(cm).get_trace(*obs)
    print("\nmodel param_boxes: ", model_trace.nodes["param_boxes"]["value"])
    print("\nmodel param_flips: ", model_trace.nodes["param_flips"]["value"])
    # print("\n_RETURN", model_trace.nodes["_RETURN"]["value"])


def test_model0(init_data):
    '''To see if params from cond_model0 and guide which have same name
    are different.'''

    print("from init_model0: ===========")
    obs_flips, obs_coins, obs_boxes, obs_vagons = init_model0(
        init_data, 10, dbg=True)
    
    print("\nfrom cond_model0: ===========")
    cond_model0(obs_flips, obs_coins, obs_boxes, obs_vagons, dbg=True)

    print("\nfrom guide_model0: ===========")
    guide_model0(obs_flips, obs_coins, obs_boxes, obs_vagons, dbg=True)

    
def test_gen_obs(init_data):
    '''To generate hierarchical sparse observable'''

    print("gen_obs_low_only:")
    res = gen_obs_low_only(init_data, 10)
    for obs in res:
        print(obs)

    print("\ngen_obs_flips_and_boxes:")
    res = gen_obs_flips_and_boxes(init_data, 10)
    for obs in res:
        print(obs)

    print("\ngen_obs_flips_and_boxes_rand:")
    res = gen_obs_flips_and_boxes_rand(init_data, 3, 10, dbg=True)
    for obs in res:
        print(obs)
    

def test_add_obs(init_data):
    counts = 10
    obs_flips = [None]*counts
    obs_coins = [None]*counts
    obs_boxes = [None]*counts
    obs_vagons = [None]*counts
    res_flips, res_coins, res_boxes, res_vagons = gen(init_data, counts)
    obs_flips = add_obs(obs_flips, res_flips, 3, counts)
    print("res_flips:")
    print(res_flips)
    print("obs_flips:")
    print(obs_flips)

    obs_coins = add_obs(obs_coins, res_coins, 3, counts)
    print("res_coins:")
    print(res_coins)
    print("obs_coins:")
    print(obs_coins)

    obs_boxes = add_obs(obs_boxes, res_boxes, 3, counts)
    print("res_boxes:")
    print(res_boxes)
    print("obs_boxes:")
    print(obs_boxes)

    obs_vagons = add_obs(obs_vagons, res_vagons, 3, counts)
    print("res_vagons:")
    print(res_vagons)
    print("obs_vagons:")
    print(obs_vagons)
# END FOR tests

if __name__ == "__main__":

    arg_name = "-csteps"
    if arg_name in sys.argv:
        csteps = int(sys.argv[sys.argv.index(arg_name)+1])
    else:
        csteps = 10
    print("count of steps (csteps):", csteps)

    arg_name = "-cobs"
    if arg_name in sys.argv:
        cobs = int(sys.argv[sys.argv.index(arg_name)+1])
    else:
        cobs = 10
    print("count of obs (cobs):", cobs)

    arg_name = "-hso"
    if arg_name in sys.argv:
        hso = int(sys.argv[sys.argv.index(arg_name)+1])
    else:
        hso = 0
    print("hso:", hso)

    init_data = {
        # three types of vagons:
        "beta_vagons": torch.tensor([0.3, 0.6, 0.9]),
        "alpha_vagons": torch.zeros(3),
        
        # four types of boxes (mixed differently for each vagon):
        "beta_boxes": torch.tensor([[0.7, 0.2, 0.3, 0.6],
                                    [0.2, 0.7, 0.3, 0.6],
                                    [0.3, 0.7, 0.2, 0.6]]),
        "alpha_boxes": torch.zeros((3, 4)),
        
        # coins proportion for each of four box types:
        "beta_coins": torch.tensor([0.3, 0.7, 0.4, 0.6]),
        "alpha_coins": torch.zeros(4),

        # coins either fair or not:
        "beta_flips": torch.tensor([0.11, 0.51]),
        "alpha_flips": torch.tensor([0.09, 0.49])
        }

    train(init_data, steps_counts=csteps, obs_counts=cobs, hso=hso)
    # test_model0_replay(init_data)
    # test_model0(init_data)
    # test_gen_obs(init_data)
    # test_add_obs(init_data)
