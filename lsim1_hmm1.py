import sim1 as sm1
import lsim1 as lsm1
from copy import deepcopy


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
    
    print(
        sum([emission({"et": olast, "xt": xt}).to_numpy()
             * transition({"xt": xt}).to_numpy() * b_t[i]
             for i, xt in enumerate(transition.get_var_values("xt"))]))
    
    b_t1 = sum([emission({"et": olast, "xt": xt}).to_numpy()
                * transition({"xt": xt}).to_numpy() * b_t[i]
                for i, xt in enumerate(transition.get_var_values("xt"))])
    print("b_t1:")
    print(b_t1)
    
    return(backward(transition, emission, orest,
                    bprev=b_t1))
    

def test_forward(obs=[1, 1]):
    pt0 = sm1.CondProb("xt", parents=[], support=[[0, 1]])
    pt0.set({}, [0.5, 0.5])
    print("pt0:")
    print(pt0)

    transition = sm1.CondProb("xt", parents=["xt1"],
                              support=[[0, 1], [0, 1]])
    transition.set({"xt1": 1}, [0.3, 0.7])
    transition.set({"xt1": 0}, [0.7, 0.3])
    print("transition:")
    print(transition)
    
    emission = sm1.CondProb("et", parents=["xt"],
                            support=[[0, 1], [0, 1]])
    emission.set({"xt": 1}, [0.1, 0.9])
    emission.set({"xt": 0}, [0.8, 0.2])
    print("emission:")
    print(emission)
    
    f = forward(pt0, transition, emission, obs, None)
    print("obs = umbrella")
    print("ft(xt|obs(k<=t)=%s):" % str(obs))
    print(f)


def test_backward(obs=[1, 1]):
    pt0 = sm1.CondProb("xt", parents=[], support=[[0, 1]])
    pt0.set({}, [0.5, 0.5])
    print("pt0:")
    print(pt0)

    transition = sm1.CondProb("xt", parents=["xt1"],
                              support=[[0, 1], [0, 1]])
    transition.set({"xt1": 1}, [0.3, 0.7])
    transition.set({"xt1": 0}, [0.7, 0.3])
    print("transition:")
    print(transition)
    
    emission = sm1.CondProb("et", parents=["xt"],
                            support=[[0, 1], [0, 1]])
    emission.set({"xt": 1}, [0.1, 0.9])
    emission.set({"xt": 0}, [0.8, 0.2])
    print("emission:")
    print(emission)
    
    bt = backward(transition, emission, obs, None)
    print("bt(obs(k>t)=%s|xt):" % str(obs))
    print(bt)


if __name__ == "__main__":
    
    test_backward(obs=[1])
    # test_forward(obs=[1, 1])
    # test_forward(obs=[0, 1])
    # test_forward(obs=[0, 0, 1])
    # test_forward(obs=[1, 0, 1])
