import numpy as np

import sim1 as sm1
import lsim1 as lsm1
from copy import deepcopy

try:
    import pyro
    import pyro.distributions as pdist
    import torch
except ModuleNotFoundError:
    print("pyro or torch not installed!")


def forward(pt0, transition, emission, obs, fprev=None):

    # init:
    if fprev is None:
        f_t1 = pt0().to_numpy()
    else:
        f_t1 = fprev
        
    # finish:
    if len(obs) == 0:
        pf = sm1.CondProb("xt", parents=[],
                          support=[deepcopy(transition.support[-1])])
        # f_xt stand for pf({"xt": 0}):
        pf.set({}, f_t1)
        return(pf)
        # return(f_t1)

    ofirst, orest = obs[0], obs[1:]

    pe_t1 = emission({"et": ofirst})
    # print("pe_t1:")
    # print(pe_t1)
    # print("sum:")
    '''
    print(
        sum([
            transition({"xt1": xt1}).to_numpy() * f_t1[i] 
            for i, xt1 in enumerate(transition.get_var_values("xt1"))]))
    '''
    f_t = pe_t1.to_numpy() * sum([
        transition({"xt1": xt1}).to_numpy() * f_t1[i] 
        for i, xt1 in enumerate(transition.get_var_values("xt1"))])
    # print("f_t:")
    # print(f_t)

    # print("f_t norm:")
    f_t = f_t/f_t.sum()
    # print(f_t)
    return(forward(None, transition, emission, orest,
                   fprev=f_t))
    

def backward(transition, emission, obs, bprev=None):

    # init:
    if bprev is None:
        b_t = [1] * len(transition.support[-1])
    else:
        b_t = bprev
        
    # finish:
    if len(obs) == 0:
        pb = sm1.CondProb("xt", parents=[],
                          support=[deepcopy(transition.support[-1])])
        # f_xt stand for pf({"xt": 0}):
        pb.set({}, b_t)
        return(pb)
        # return(f_t1)

    orest, olast = obs[:-1], obs[-1]
    
    # here result will be a vector, dependence of xt1 (p(*|x_{t-1})):
    b_t1 = sum([emission({"et": olast, "xt": xt}).to_numpy()
                * transition({"xt": xt}).to_numpy() * b_t[i]
                for i, xt in enumerate(transition.get_var_values("xt"))])
    print("b_t1:")
    print(b_t1)
    
    return(backward(transition, emission, orest,
                    bprev=b_t1))
    

def create_model(obs):
    pt0 = sm1.CondProb("xt0", parents=[], support=[[0, 1]])
    pt0.set({}, [0.5, 0.5])
    print("pt0:")
    print(pt0)

    nodes = [pt0]
    cond = []
    for io, o in enumerate(obs):
        transition = sm1.CondProb("xt"+str(io+1), parents=["xt"+str(io)],
                                  support=[[0, 1], [0, 1]])
        transition.set({"xt"+str(io): 1}, [0.3, 0.7])
        transition.set({"xt"+str(io): 0}, [0.7, 0.3])
        # print("transition:")
        # print(transition)
        nodes.append(transition)
    
        emission = sm1.CondProb("et"+str(io+1), parents=["xt"+str(io+1)],
                                support=[[0, 1], [0, 1]])
        emission.set({"xt"+str(io+1): 1}, [0.1, 0.9])
        emission.set({"xt"+str(io+1): 0}, [0.8, 0.2])
        # print("emission:")
        # print(emission)
        nodes.append(emission)

        # here `[0]` means CondProb node value itself
        # (its parent will have index [1])
        # (result of sampling will be table value like (0, 1))
        cond.append("$et"+str(io+1)+"[0]"+"=="+str(o))
    cond = " and ".join(cond)
    print("cond: ", cond)
    
    # print("nodes[-1].sample():")
    # print(nodes[-1].sample())

    net = sm1.BayesNet(nodes)
    return(net, cond)


def test_forward(obs=[1, 1]):
    pt0 = sm1.CondProb("xt", parents=[], support=[[0, 1]])
    pt0.set({}, [0.5, 0.5])
    # print("pt0:")
    # print(pt0)

    transition = sm1.CondProb("xt", parents=["xt1"],
                              support=[[0, 1], [0, 1]])
    transition.set({"xt1": 1}, [0.3, 0.7])
    transition.set({"xt1": 0}, [0.7, 0.3])
    # print("transition:")
    # print(transition)
    
    emission = sm1.CondProb("et", parents=["xt"],
                            support=[[0, 1], [0, 1]])
    emission.set({"xt": 1}, [0.1, 0.9])
    emission.set({"xt": 0}, [0.8, 0.2])
    # print("emission:")
    # print(emission)
    
    f = forward(pt0, transition, emission, obs, None)
    # print("obs = umbrella")
    print("ft(xt|obs(k<=t)=%s):" % str(obs))
    print(f)


def test_backward(obs=[1, 1], dbg_show_model=True):
    pt0 = sm1.CondProb("xt", parents=[], support=[[0, 1]])
    pt0.set({}, [0.5, 0.5])

    if dbg_show_model:
        print("pt0:")
        print(pt0)

    # here xft1 means x_{t-1}:
    transition = sm1.CondProb("xt", parents=["xt1"],
                              support=[[0, 1], [0, 1]])
    transition.set({"xt1": 1}, [0.3, 0.7])
    transition.set({"xt1": 0}, [0.7, 0.3])
    if dbg_show_model:
        print("transition:")
        print(transition)
    
    emission = sm1.CondProb("et", parents=["xt"],
                            support=[[0, 1], [0, 1]])
    emission.set({"xt": 1}, [0.1, 0.9])
    emission.set({"xt": 0}, [0.8, 0.2])
    if dbg_show_model:
        print("emission:")
        print(emission)
    
    bt = backward(transition, emission, obs, None)
    print("bt(obs(k>t)=%s|xt):" % str(obs))
    print(bt.df.to_numpy().T[0])
    
    # check:
    fb = lambda tr, em, bs: sum([em[i]*tr[i]*bs[i] for i, _ in enumerate(tr)])
    b = [1, 1]
    for o in obs:
        b = fb(np.array([[0.8, 0.1], [0.2, 0.9]])[o],
               np.array([[0.7, 0.3], [0.3, 0.7]]), b)
    print("b accurate:")
    print(b)
    assert (bt.df.to_numpy().T[0] == b).all()
    

def test_fb0():
    '''general hmm'''

    pt0 = sm1.CondProb("xt", parents=[], support=[[0, 1, 2, 3]])
    pt0.set({}, [0.5, 0.5, 0.5, 0.5])

    transition = sm1.CondProb("xt", parents=["xt1"],
                              support=[[0, 1, 2, 3], [0, 1, 2, 3]])
    transition.set({"xt1": 0}, [0.4, 0.3, 0.2, 0.1])
    transition.set({"xt1": 1}, [0.3, 0.4, 0.2, 0.1])
    transition.set({"xt1": 2}, [0.2, 0.3, 0.4, 0.1])
    transition.set({"xt1": 3}, [0.1, 0.2, 0.3, 0.4])

    emission = sm1.CondProb("et", parents=["xt"],
                            support=[[0, 1], [0, 1, 2, 3]])
    emission.set({"xt": 0}, [0.1, 0.9])
    emission.set({"xt": 2}, [0.1, 0.9])
    emission.set({"xt": 1}, [0.8, 0.2])
    emission.set({"xt": 3}, [0.8, 0.2])

    obs = [1, 1]
    f = lsm1.forward(pt0, transition, emission, obs, None)
    print("f:")
    print(f)
    print("bt:")
    bt = lsm1.backward(transition, emission, obs, None)
    print(bt)


def test_rejection_sampler_work():
    net, cond = create_model([])
    print("net.sorted_vars:")
    print(net.sorted_vars)


def test_rejection_sampler(mode="forward", obs=[1, 1], N=3):
    '''
    TODO: continues, pyro
 
    var_to_extract="xt2" for forward (p(xt2|et1, et2)
    var_to_extract="xt0" for backward (p(xt0| et1, et2)
    in order to convert p(xt0|e1, e2) to p(e1, e2| xt0) we need a
    factor 1/p(xt0) = 1/<0.5, 0.5> i.e.:
    p(e2, e1|xt0) = p(xt0|e1, e2)*p(e1, e2)/p(xt0)

    transition constant [[0.7, 0.3], [0.3, 0.7]]
    emission constant [[0.8, 0.2], [0.1, 0.9]]
    '''
    net, cond = create_model(obs)
    print("net.sorted_vars:")
    print(net.sorted_vars)
    cond = "True" if cond == "" else cond
    print("cond:")
    print(cond)

    # print("net.prior_sample:")
    # print(net.prior_sample())
    if mode == "forward":
        var_to_extract = "xt2"
        factor = 1
    elif mode == "backward":
        var_to_extract = "xt0"
        # factor = 1/0.5
        factor = 1/2.
    else:
        raise(Exception("mode forward or backward"))
    res0 = []
    res1 = []

    gindexes = []
    allindexes = {}
    steps = int(N/100)
    for step in range(steps):
        print("step %d from %d" % (step, steps))

        # forward:
        # var_to_extract = 'xt2' 
        
        # backward:
        # var_to_extract = 'xt0'
        
        samples, indexes, labels = sm1.rejection_sampling(net, 100,
                                                          cond=cond)
        print("len(samples): ", len(samples[var_to_extract]))
        # print("indexes:")
        # print(indexes)
        gindexes.extend(indexes[var_to_extract])
        xt2 = np.array(gindexes)
        # print(xt2)
        n = len(xt2)
        print("n: ", n)
        if n > 0:
            for var in indexes:
                if var not in allindexes:
                    allindexes[var] = []
                allindexes[var].extend(indexes[var])
                xs = np.array(allindexes[var])
                print(var)
                print(np.array([len(xs[xs == 0]), len(xs[xs == 1])])/n)
        
            # print("f2=p(x2|e2, e1)(%s):" % str(obs))

            xto = np.array([len(xt2[xt2 == 0])/n,
                            len(xt2[xt2 == 1])/n])
            
            # print(xto)
            xtoo = np.array([len(xt2[xt2 == 0])/n,
                             len(xt2[xt2 == 1])/n]) * factor
            # print(xtoo)
            
            # print("xto/2.:")
            # print(xto/2.)
            # print("xto/0.5:")
            # print(xto/0.5)
            
            res0.append(xto[0])
            res1.append(xto[1])

    import matplotlib.pyplot as plt
    plt.plot(res0)
    plt.plot(res1)
    plt.show()


def test_gen():
    # obs is not important here, as well as cond:
    net, cond = create_model([1, 2])

    print("net.prior_sample():")
    print(net.prior_sample())
    

def test_enumeration_backward(obs=[1]):
    eobs = dict(("et%d" % (t+1), o) for t, o in enumerate(obs))
    print("eobs:")
    print(eobs)
    net, cond = create_model(obs)
    print("net.sorted_vars:")
    print(net.sorted_vars)
    print("backward:")
    result = sm1.enumeration_ask(net, "xt0", eobs)
    # result = sm1.enumeration_ask(net, "xt0", {"et1": 1})
    print("p(xt0|et1==%s):" % str(obs))
    print(result)


if __name__ == "__main__":

    # test_enumeration_backward(obs=[1, 0])
    # accurate p(x0|e1=1, e2=0): 0.418892  0.581108
    # test_enumeration_backward(obs=[1])
    test_rejection_sampler(mode="backward", obs=[1, 0], N=3000)
    #  result: 0.37878788 0.62121212

    # test_gen()
    # test_rejection_sampler_work()
    
    # accurate p(x0|e1=1): 0.372 0.627
    # result p(x0|e1=1): 0.35366605 0.64633395
    # accurate p(x1|e1=1) = forward(1): 0.181818 0.818182
    # result p(x1|e1=1): 0.18237831 0.81762169
    # test_rejection_sampler(mode="backward", obs=[1], N=3000)

    # test_rejection_sampler(var_to_extract="xt0", factor=0.5, obs=[0, 1], N=27000)

    # accurate: [0.18181818, 0.81818182]
    # 0.1148881239242685 0.8851118760757315
    # test_rejection_sampler(obs=[1, 1], N=27000)
    
    # test_backward(obs=[1], dbg_show_model=True)
    # print("\nbackward:")
    # test_backward(obs=[1], dbg_show_model=False)
    
    # print("\nforward:")
    # test_forward(obs=[1])
    
    # test_forward(obs=[1, 1])
    # test_forward(obs=[0, 1])
    # test_forward(obs=[0, 0, 1])
    # test_forward(obs=[1, 0, 1])
