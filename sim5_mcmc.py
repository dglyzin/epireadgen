import numpy as np
import torch
import pyro
import pyro.distributions as pdist
from pyro.infer import MCMC, NUTS
from pyro.poutine.runtime import _PYRO_PARAM_STORE


from sim5 import SolverContextMessenger, CModel

# for load init values from file
from sim5 import ResultCollectorMessenger

import characteristics as chars


class CModelForMCMC(CModel):
    def __init__(self, *args, **kwargs):

        CModel.__init__(self, *args, **kwargs)

        # all this params will be same for all iterations inside the model:
        with pyro.plate("params_plate", size=2):
            a = pyro.sample("a", pdist.Uniform(0, 3))
            # torch.tensor([1.5, 0.])
            
            b = pyro.sample("b", pdist.Uniform(0, 3))
            # torch.tensor([1.5, 0.])

            d = pyro.sample("d", pdist.Uniform(0, 0.001))
            # d = torch.tensor([0.0001, 0.0001])
        self.a = a
        self.b = b
        self.d = d

    def get_params(self):
        return (self.a, self.b, self.d)


class CModelForTest(CModel):

    def __init__(self, a, b, d, *args, **kwargs):

        CModel.__init__(self, *args, **kwargs)

        self.a = a
        self.b = b
        self.d = d
    
    def get_params(self):
        return (self.a, self.b, self.d)


def model_for_mcmc(N, t0, t1, dd, ll):

    model = mk_model(
        "train",
        N, t0, t1, dd=dd, ll=ll,
        
        # remove a progress for it do not interfere with mcmc one
        cprogress=None, dbg=False)
    
    run_model(model)


def mk_model(_type, *args, **kwargs):

    if _type == "train":
        return(CModelForMCMC(*args, **kwargs))
    elif _type == "test":
        return(CModelForTest(*args, **kwargs))


def run_model(model):

    with SolverContextMessenger(
            model.g_init, model.get_neg_times) as tr0:
       
        with chars.Lyapunov(model.dd[0], model.dd[0], tr0.trace, 1) as lpv:
            model(tr0.trace)
    tr0._reset()

    # clearing old data from global _PYRO_PARAM_STORE:
    params_names = [name for name in _PYRO_PARAM_STORE._params]
    for name in params_names:
        # print("name", name)
        del _PYRO_PARAM_STORE[name]

    res = lpv.results["Lambda"][-1][lpv.results["Lambda"][-1] != -np.inf]
    # res = lpv.results["Lambda"][-1]
    # print("res:", res)
    print("mean(res):", torch.mean(res))

    # mean(\lambda) < 0 means getting information with each iteration:
    # while mean(\lambda) > 0 means loosing it:
    loss = -np.inf if torch.mean(res) < 0 else 0

    # loss = -1/torch.exp(torch.mean(res))
    # loss = -1/torch.exp(torch.mean(res)*torch.tensor(-10**6))
    
    print("loss:", loss)
    pyro.factor("loss", loss)
    # lpv.plot(all=True)
    # res = lpv.results["Lambda"][-1][lpv.results["Lambda"][-1] == -np.inf]

    '''
    if (res > 0).any():
        pyro.factor("loss", -np.inf)
    '''
    '''
    with pyro.plate("loss_plate", size=res.size()):
        loss = -1/torch.exp(res)
        pyro.factor("loss", loss)
    '''
    '''
    loss = -1/torch.exp(torch.mean(res))
    pyro.factor("loss", loss)
    '''
    return lpv
            

def load_init(in_filename, cuda=False):
    # FOR extracting init data:
    results = ResultCollectorMessenger(None, None)
    results.load(in_filename)
    N = results.results["z"]
    if cuda:
        N = N.to(torch.ones((3, 3)).device)
    for i in range(1, 3)[::-1]:
        print("N[-%d]" % i)
        print(N[-i])
    # END FOR
    
    return(N)


def train_mcmc(steps, N, t0, t1, dd, ll,
               cuda=False):
    '''
    - ``dt`` -- will be ignored for main
    (since N values used as well as t1)
    
    - ``dd`` -- tuple of space steps
    like (0.01,) or (0.01, 0.01,), ...

    - ``ll`` -- tuple of space sizes
    like (10,) or (10, 10,), ...
    '''
    if cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("torch.cuda.memory_allocated():")
        print(torch.cuda.memory_allocated())
    print("\ndevice used: ", torch.ones((3, 3)).device)
    
    pyro.clear_param_store()
   
    print("\nstarting")            
    nuts_kernel = NUTS(model_for_mcmc)
    mcmc = MCMC(
            nuts_kernel,
            num_samples=steps,
            # warmup_steps=1,
            num_chains=1,)

    mcmc.run(N, t0, t1, dd, ll)
    # mcmc.run(torch.ones(3))
    print(mcmc.summary())
    return(mcmc)


def test_model(mcmc, N, t0, t1, dd, ll):
    a = torch.mean(mcmc.get_samples()['a'], 0)
    b = torch.mean(mcmc.get_samples()['b'], 0)
    d = torch.mean(mcmc.get_samples()['d'], 0)
    print("mean(a): ", a)
    print("mean(b): ", b)
    print("mean(d): ", d)
    model = mk_model("test", a, b, d, N, t0, t1, dd=dd, ll=ll)
    lpv = run_model(model)
    lpv.plot(all=True)


if __name__ == "__main__":
    
    t0 = -1
    t1 = 0.1
    dd = (0.01, )
    ll = (10, )

    N = load_init(in_filename="N.tmp", cuda=False)
    mcmc = train_mcmc(10, N, t0, t1, dd, ll,
                      cuda=False)
    test_model(mcmc, N, t0, t1,  dd, ll,)
