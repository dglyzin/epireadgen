import pyro
import torch
import pyro.distributions as pdist
from pyro.ops.indexing import Vindex
import pyro.poutine as poutine

import progresses.progress_cmd as progress_cmd
ProgressCmd = progress_cmd.ProgressCmd


def init_model0(counts, dbg=False):
    '''To generate the init observable.
    Params in model0 will be fixed with that to learn. 
    '''

    # three types of vagons:
    beta_vagons = torch.tensor([0.3, 0.6, 0.9])
    alpha_vagons = torch.zeros_like(beta_vagons)             
    
    # four types of boxes (mixed differently for each vagon):
    beta_boxes = torch.tensor([[0.7, 0.2, 0.3, 0.6],
                               [0.2, 0.7, 0.3, 0.6],
                               [0.3, 0.7, 0.2, 0.6]])
    alpha_boxes = torch.zeros_like(beta_boxes)
    
    # coins proportion for each of four box types:
    beta_coins = torch.tensor([0.3, 0.7, 0.4, 0.6])
    alpha_coins = torch.zeros_like(beta_coins)

    # coins either fair or not:
    beta_flips = torch.tensor([0.1])
    alpha_flips = torch.tensor([0.5])

    res = model0([None]*counts, [None]*counts, [None]*counts, [None]*counts,
                 params_flips=(alpha_flips, beta_flips),
                 params_coins=(alpha_coins, beta_coins),
                 params_boxes=(alpha_boxes, beta_boxes),
                 params_vagons=(alpha_vagons, beta_vagons))
    return res


def cond_model0(obs_flips, obs_coins, obs_boxes, obs_vagons, dbg=False):
    '''
    To generate conditional model. Params in that model will be generated
    randomly.
    Returning model will be used for pyro learning with guide. 
    It must replace all variables by a guide ones if they names match.
    '''
    # three types of vagons:
    adist_vagons = pdist.Uniform(torch.zeros(3), 0.5*torch.ones(3))
    bdist_vagons = pdist.Uniform(0.51*torch.ones(3), torch.ones(3))
    alpha_vagons, beta_vagons = (adist_vagons.sample(), bdist_vagons.sample())
    if dbg:
        print("\nalpha_vagons:")
        print(alpha_vagons)
        print("\nbeta_vagons:")
        print(beta_vagons)

    # four types of boxes (mixed differently for each vagon):
    adist_boxes = pdist.Uniform(torch.zeros((3, 4)), 0.5*torch.ones((3, 4)))
    bdist_boxes = pdist.Uniform(0.51*torch.ones((3, 4)), torch.ones((3, 4)))
    alpha_boxes, beta_boxes = (adist_boxes.sample(), bdist_boxes.sample())
    if dbg:
        print("\nalpha_boxes:")
        print(alpha_boxes)
        print("\nbeta_boxes:")
        print(beta_boxes)

    # coins proportion for each of four box types:
    adist_coins = pdist.Uniform(torch.zeros(4), 0.5*torch.ones(4))
    bdist_coins = pdist.Uniform(0.51*torch.ones(4), torch.ones(4))
    alpha_coins, beta_coins = (adist_coins.sample(), bdist_coins.sample())
    if dbg:
        print("\nalpha_coins:")
        print(alpha_coins)
        print("\nbeta_coins:")
        print(beta_coins)

    # coins either fair or not:
    adist_flips = pdist.Uniform(torch.zeros(1), 0.5*torch.ones(1))
    bdist_flips = pdist.Uniform(0.51*torch.ones(1), torch.ones(1))
    alpha_flips, beta_flips = (adist_flips.sample(), bdist_flips.sample())
    if dbg:
        print("\nalpha_flips:")
        print(alpha_flips)
        print("\nbeta_flips:")
        print(beta_flips)

    res = model0(obs_flips, obs_coins, obs_boxes, obs_vagons,
                 params_flips=(alpha_flips, beta_flips),
                 params_coins=(alpha_coins, beta_coins),
                 params_boxes=(alpha_boxes, beta_boxes),
                 params_vagons=(alpha_vagons, beta_vagons))
    return res


def model0(obs_flips, obs_coins, obs_boxes, obs_vagons,
           params_flips=None, params_coins=None,
           params_boxes=None, params_vagons=None):
    '''For generating the data. Also during the learning
    all `*_params` will be ignored by replacing it in elbo.
    See minipyro.py elbo def'''
    
    alpha_vagons, beta_vagons = params_vagons    
    vagons_param = pyro.sample(
        "param_vagons", pdist.Uniform(alpha_vagons, beta_vagons))

    alpha_boxes, beta_boxes = params_boxes
    boxes_param = pyro.sample(
        "param_boxes", pdist.Uniform(alpha_boxes, beta_boxes))
    
    alpha_coins, beta_coins = params_coins    
    coins_param = pyro.sample(
        "param_coins", pdist.Uniform(alpha_coins, beta_coins))

    alpha_flips, beta_flips = params_flips
    flips_param = pyro.sample(
        "param_flips", pdist.Uniform(alpha_flips, beta_flips))
    
    flips = []
    coins = []
    boxes = []
    vagons = []
    
    # len(obs_flips)
    for i in pyro.plate("vagons_plate", size=len(obs_vagons)):
        if obs_vagons[i] is not None:
            vagon = pyro.sample("vagon", pdist.Categorical(vagons_param),
                                obs=obs_vagons[i])
        else:
            vagon = pyro.sample("vagon", pdist.Categorical(vagons_param))
        vagons.append(vagon)
        # print("vagon:", vagon)
        
        id_box = Vindex(boxes_param)[vagon.type(torch.long)]
        # print("id_box: ", id_box)
        if obs_boxes[i] is not None:
            box = pyro.sample("box", pdist.Categorical(id_box),
                              obs=obs_boxes[i])
        else:
            box = pyro.sample("box", pdist.Categorical(id_box))
        boxes.append(box)
        # print("box: ", box)
        
        id_coin = Vindex(coins_param)[box.type(torch.long)]
        # print("id_coin:", id_coin)
        if obs_coins[i] is not None:
            coin = pyro.sample("coin", pdist.Bernoulli(id_coin),
                               obs=obs_coins[i])
        else:
            coin = pyro.sample("coin", pdist.Bernoulli(id_coin))
        coins.append(coin)
        # print("coin:", coin)
        
        id_flip = Vindex(flips_param)[coin.type(torch.long)]
        # print("id_flip:", id_flip)
        if obs_flips[i] is not None:
            flip = pyro.sample(
                "flip", pdist.Bernoulli(id_flip), obs=obs_flips[i])
            # with pyro.mask(obs_flips_mask)
            # flip = pyro.sample(
            #    "flip", pdist.Bernoulli(id_flip),
            #    obs=obs_flips.index_select(0, ind))
        else:
            flip = pyro.sample(
                "flip", pdist.Bernoulli(id_flip))
        flips.append(flip)
    return list(map(torch.tensor, (flips, coins, boxes, vagons)))


def guide_model0(obs_flips, obs_coins, obs_boxes, obs_vagons, dbg=False):
    # three types of vagons:
    adist_vagons = pdist.Uniform(torch.zeros(3), 0.5*torch.ones(3))
    bdist_vagons = pdist.Uniform(0.51*torch.ones(3), torch.ones(3))
    alpha_vagons, beta_vagons = (adist_vagons.sample(), bdist_vagons.sample())
    vagons_params = pyro.param(
        "param_vagons", pdist.Uniform(alpha_vagons, beta_vagons))
    if dbg:
        print("\nalpha_vagons:")
        print(alpha_vagons)
        print("\nbeta_vagons:")
        print(beta_vagons)
        print("\nvagons_params:")
        print(vagons_params)

    # four types of boxes (mixed differently for each vagon):
    adist_boxes = pdist.Uniform(torch.zeros((3, 4)), 0.5*torch.ones((3, 4)))
    bdist_boxes = pdist.Uniform(0.51*torch.ones((3, 4)), torch.ones((3, 4)))
    alpha_boxes, beta_boxes = (adist_boxes.sample(), bdist_boxes.sample())
    boxes_params = pyro.param(
        "param_boxes", pdist.Uniform(alpha_boxes, beta_boxes))
    if dbg:
        print("\nalpha_boxes:")
        print(alpha_boxes)
        print("\nbeta_boxes:")
        print(beta_boxes)
        print("\nboxes_params:")
        print(boxes_params)

    # coins proportion for each of four box types:
    adist_coins = pdist.Uniform(torch.zeros(4), 0.5*torch.ones(4))
    bdist_coins = pdist.Uniform(0.51*torch.ones(4), torch.ones(4))
    alpha_coins, beta_coins = (adist_coins.sample(), bdist_coins.sample())
    coins_params = pyro.param(
        "param_coins", pdist.Uniform(alpha_coins, beta_coins))
    if dbg:
        print("\nalpha_coins:")
        print(alpha_coins)
        print("\nbeta_coins:")
        print(beta_coins)
        print("\ncoins_params")
        print(coins_params)

    # coins either fair or not:
    adist_flips = pdist.Uniform(torch.zeros(1), 0.5*torch.ones(1))
    bdist_flips = pdist.Uniform(0.51*torch.ones(1), torch.ones(1))
    alpha_flips, beta_flips = (adist_flips.sample(), bdist_flips.sample())

    flips_params = pyro.param(
        "param_flips", pdist.Uniform(alpha_flips, beta_flips))

    if dbg:
        print("\nalpha_flips:")
        print(alpha_flips)
        print("\nbeta_flips:")
        print(beta_flips)
        print("\nflips_params:")
        print(flips_params)
 

def gen_obs_flips_and_boxes_rand(rand_count, counts, dbg=False):
    '''Generate flips and boxes observable with same count
    and choose `rand_count` samples from random but same
    positions for both boxes and flips.
    '''
    res_flips, res_coins, res_boxes, res_vagons = gen(counts)
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


def gen_obs_flips_and_boxes(counts):
    '''Generate flips and boxes observable with same count'''
    res_flips, res_coins, res_boxes, res_vagons = gen(counts)
    
    return (res_flips, [None]*counts, res_boxes, [None]*counts)
    

def gen_obs_low_only(counts):
    '''Generate only low level observable'''
    res_flips, res_coins, res_boxes, res_vagons = gen(counts)
    
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


def gen(counts=10):
    obs_flips = [None]*counts
    obs_coins = [None]*counts
    obs_boxes = [None]*counts
    obs_vagons = [None]*counts
    res_flips, res_coins, res_boxes, res_vagons = model0(
        obs_flips, obs_coins, obs_boxes, obs_vagons)
    # model0(torch.ones(10))
    return (res_flips, res_coins, res_boxes, res_vagons)


def test_model0_replay():
    obs = init_model0(10, dbg=True)
    guide_trace = poutine.trace(guide_model0).get_trace(*obs)
    print("guide param_boxes: ", guide_trace.nodes["param_boxes"]["value"])
    print("\nguide param_flips: ", guide_trace.nodes["param_flips"]["value"])
    
    cm = poutine.replay(cond_model0, trace=guide_trace)
    model_trace = poutine.trace(cm).get_trace(*obs)
    print("\nmodel param_boxes: ", model_trace.nodes["param_boxes"]["value"])
    print("\nmodel param_flips: ", model_trace.nodes["param_flips"]["value"])
    # print("\n_RETURN", model_trace.nodes["_RETURN"]["value"])


def test_model0():
    obs_flips, obs_coins, obs_boxes, obs_vagons = init_model0(10, dbg=True)
    cond_model0(obs_flips, obs_coins, obs_boxes, obs_vagons, dbg=True)
    guide_model0(obs_flips, obs_coins, obs_boxes, obs_vagons, dbg=True)

    
def test_gen_obs():
    print("gen_obs_low_only:")
    res = gen_obs_low_only(10)
    for obs in res:
        print(obs)

    print("\ngen_obs_flips_and_boxes:")
    res = gen_obs_flips_and_boxes(10)
    for obs in res:
        print(obs)

    print("\ngen_obs_flips_and_boxes_rand:")
    res = gen_obs_flips_and_boxes_rand(3, 10, dbg=True)
    for obs in res:
        print(obs)
    

def test_add_obs():
    counts = 10
    obs_flips = [None]*counts
    obs_coins = [None]*counts
    obs_boxes = [None]*counts
    obs_vagons = [None]*counts
    res_flips, res_coins, res_boxes, res_vagons = gen(counts)
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


if __name__ == "__main__":
    test_gen_obs()
    # test_add_obs()
