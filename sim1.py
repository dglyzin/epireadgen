try:
    import torch
    import torch.distributions as dist
    from torch import tensor
except ModuleNotFoundError:
    print("WORNING: torch not found")

import pandas as pd
from functools import reduce
import numpy as np

try:
    import sim2
except ModuleNotFoundError:
    print("WORNING: sim2.py ignored")

import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

from string import Template


# from progresses.progress_cmd import ProgressCmd
import progresses.progress_cmd as progress_cmd
ProgressCmd = progress_cmd.ProgressCmd


class BayesNet():
    '''
    Each node in net (i.e. P(X|Z,Y)) must depend only on events
    from previus level of net (so if there is a node P(X|Y,Z)
    then there must be no P(Y|Z)).
    '''
    def __init__(self, nodes):
        self.nodes = nodes
        self.pre_sorted = self.sort([], [nodes[0].var], [0])

        # reverse false because of `enumeration_all`:
        self.pre_sorted.sort(key=lambda key: key[1], reverse=False)
        
        # remove duplicates:
        self.sorted_vars = reduce(
            lambda acc, x: acc + [x[0]] if x[0] not in acc else acc,
            self.pre_sorted, [])
        # print(self.sorted_vars)
        # print(list(map(lambda x: x.var, self.nodes)))
        
        # choose sorted nodes:
        self.sorted_nodes = [next(filter(lambda node: node.var == var,
                                         self.nodes))
                             for var in self.sorted_vars]
        self.dnodes = dict(zip(self.sorted_vars, self.sorted_nodes))
        
    def get_labels(self, events):
        return(dict([(var, list(map(str, self.dnodes[var].enumerate_support(events))))
                     for var in self.dnodes
                     if hasattr(self.dnodes[var], 'enumerate_support')]))
    
    def init_samples(self, default=lambda var: None):
        # just return empty dict
        return(dict([(var, default(var)) for var in self.sorted_vars]))

    def prior_sample(self, gevents=None):
        '''
        gevents like "a==0 and b>=0" supported only if there is 
        no more then one continues dist
        otherwise {"a":0, b:[1, 2, 3]} must be used
        '''
        
        # init samples with None (like {"a": None}):
        samples = self.init_samples()

        for var in self.sorted_vars:
            dnode = self.dnodes[var]
            # print("dnode:")
            # print(dnode)
            # print(dnode.var)
            
            # dnode.set_params(samples)

            # samples will be from previous setps:
            # (params, not parents!)
            dnode_pnames = dnode.reset_params(samples)
            # print("dnode_pnames:")
            # print(dnode_pnames)

            pevents = dict([
                (pvar, samples[pvar][self.dnodes[pvar].vars.index(pvar)])
                for pvar in dnode.parents if pvar not in dnode_pnames])
            # print("pevents:")
            # print(pevents)

            # update sample value in samples:
            samples[var] = dnode.sample(update(gevents, pevents))
            
        return(samples)

    def get_node(self, var):
        return(self.dnodes[var])

    def sort(self, sorted, order, idxs):
        '''
        Sort nodes in net according to they levels
        in it. It achived by steping from some term and
        calculating distance from for each node.
        
        Example:
        if `a` given in start in `net` like `root->a, root->b, a->c`
        then `s` for `root` will be -1, for `a`, `b` 0, for `c` +1.
        '''
        if len(order) == 0:
            return(sorted)

        first_var, rest_vars = order[0], order[1:]
        first_idx, rest_idxs = idxs[0], idxs[1:]
        
        # main task:
        sorted.append([first_var, first_idx])

        # FOR take/find node for name `first_var`:
        nodes = [node for node in self.nodes
                 if node.var == first_var]
        if len(nodes) == 0:
            # if no such node, continue with others:
            return(self.sort(sorted, rest_vars, rest_idxs))
        node = nodes[0]
        # END FOR

        # mark parents as `s-1`:
        for parent_var in node.parents:
            if parent_var not in [var for var, idx in sorted]:
                rest_vars.append(parent_var)
                rest_idxs.append(first_idx-1)

        # for finding children:
        children = [node.var for node in self.nodes
                    if first_var in node.parents]
        
        # for mark children as `s+1`:
        for child in children:
            if child not in [var for var, idx in sorted]:
                rest_vars.append(child)
                rest_idxs.append(first_idx+1)

        return(self.sort(sorted, rest_vars, rest_idxs))
            

class Probability():
    '''
    p = sim1.Probability()

    In [42]: p
    Out[42]: 
    0
    X Y    
    0 0 NaN
      1 NaN
    1 0 NaN
      1 NaN

    In [43]: p.set({"Y":1},0.7)


    In [44]: p
    Out[44]: 
           0

    X Y     
    0 0  NaN
      1  0.7
    1 0  NaN
      1  0.7

    In [45]: p({"Y":0}).sum()[0]
    Out[45]: 0.0

    In [46]: p({"Y":0})
    Out[46]: 

          0
    X Y    
    0 0 NaN
    1 0 NaN
    
    # this also work:
    p.set("Y>0", 0.1)
    p("Y>0")
    '''
    def __init__(self, vars=["X", "Y"], support=[[0, 1]]*2, params=[]):
        self.vars = vars
        self.support = support
        self.index = pd.MultiIndex.from_product(support, names=vars)
        size = reduce(lambda acc, x: acc*len(x), support, 1)
        self.size = size
        self.df = pd.DataFrame(np.ones(size)*np.nan, index=self.index)
        
        self.table_like = True

        self.params = params

    def __repr__(self):
        return(self.df.to_string())

    def __call__(self, events={}):
    
        values = self.get_values(events)
        return(self.df.loc[values, :])

    def set(self, events, ps):
        values = self.get_values(events)
        # print(values)
        self.df.loc[values, :] = np.array(ps)

    def reset_params(self, samples):
        dnode = self

        dnode_pnames = []
        for events, param_name in dnode.params:
            assert type(events) == str

            dnode_pnames.append(param_name)

            # ex: p(X|Y=0) = p(\Sigma).sample()
            dnode.set(events, samples[param_name])
            # p(X|Y!=0) = norm(1-p(\Sigma).sample())
            size = len(dnode("not " + events))
            dnode.set("not "+events, (1 - samples[param_name])/size)
        return(dnode_pnames)

    def get_var_values(self, var):
        return(self.support[self.vars.index(var)])

    def get_values(self, events, use_index=False):
        # if events like "X>0.1 & Y==0":
        if events is None:
            return(self.df.index)

        if type(events) == str:
            return(self.df.query(events).index)
            
        # if events like {"X": 0.1, "Y": 0}
        if use_index:
            values = [(events[var] if type(events[var]) == list else [events[var]])
                      if var in events
                      else self.support[idx]
                      for idx, var in enumerate(self.vars)]
            index = pd.MultiIndex.from_product(values,
                                               names=self.vars)
            # >>> p.df.loc[i, :] 
            return(index)
            # index = self.df.loc[tuple(values), :].index
            #  if slice not in values:
            # [[0], [1], [0]]

        
        values = [events[var] if var in events
                  else slice(None)
                  for idx, var in enumerate(self.vars)]
            
        # print("\nindex:")
        # print(index)
        # print("tuple(values)")
        # print(tuple(values))
        # print("self.df index:")
        # print(self.df.loc[index, :])
        # print("self.df values")
        # print(self.df.loc[tuple(values), :])
        return(tuple(values))
        
    def sample(self, events=None):
        val = self(events).to_numpy()
        idx = self.get_values(events, use_index=True)
        '''
        print(self)
        print("events:")
        print(events)
        print("idx:")
        print(idx)
        print("")
        print(self.df.loc[idx, :])
        print(self.df.loc[idx, :].index)
        print("val.flatten():")
        print(val.flatten())
        '''
        edist = sim2.Empirical(idx.to_numpy(), val.flatten())
        # print("edist.sample():")
        # print(edist.sample())
        return(edist.sample())

    def log_prob(self, events):

        return(np.log(self(events).to_numpy().flatten()))
        # value = p(events).to_numpy() 
        # idx = p.get_values(events)

        # edist = sim2.Empirical(idx.to_numpy(), val.flatten())
        # return(edist.log_prob(torch.tensor(value)))
        # ed.log_prob(torch.tensor((0.3, 0., 0.)))

    def enumerate_support(self, events):
        return(self.get_values(events, use_index=True).to_numpy())

    def get_support_index(self, sample, events, var_only=False):
        '''
        Return support index of sample for given events
        
        - ``events`` -- str or dict with events, must been
        same as had been in `sample = dist.sample(events)`

        - ``sample`` -- from `dist.sample(events)`
        (index of `self.df` table)
        
        - ``var_only`` -- for plot. if true return index only
        in accrdence with self.var range, otherwise return
        in accordence to events.

        Example:
        p_x_y = sim1.CondProb("X", parents=["Y"],
                              support=[[0, 1], [0, 1]])
        p_x_y.set({"Y": 0}, [0.1, 0.9])
        p_x_y.set({"Y": 1}, [0.5, 0.5])

        p_x_y
              0
        X Y     
        0 0  0.1
          1  0.5
        1 0  0.9
          1  0.5

        sample = p_x_y.sample()
        # (0, 1)

        p_x_y.get_support_index(sample, None)
        # 1

        p_x_y.get_support_index(sample, None, True)
        # 0

        p_x_y.get_support_index((1,1), None)
        # 3

        p_x_y.get_support_index((1,1), None, True)
        # 1
        '''
        if var_only:
            idx = self.vars.index(self.var)
            return(self.support[idx].index(sample[idx]))

        support = list(self.enumerate_support(events))
        return(support.index(sample))

    def get_support_val(self, sample, events):
        '''
        >>> get_support_val((0, 1, 0), {"X":0, Y:[0,1]})
        # return {"X":0, "Y":1, "Z":0}

        >>> get_support_val((0, 1, 0), {Y:[0,1]})
        # return {"X":0, "Y":1, "Z": 0}
        Not used, intended for extract sample names
        (for prior_sample update events from sample)
        '''
        vals = self.get_values(events, use_index=True)
        return(dict(zip(vals.names, sample)))


class CondProb(Probability):
    def __init__(self, var, parents=[], support=[[0, 1]]*2, params=[]):
        self.var = var
        self.parents = parents
        Probability.__init__(self, vars=[var]+parents,
                             support=support, params=params)
        

def enumeration_ask(bn, var, events):
    '''
    # calculate p(var|events)
    # if some of vars in events has several values
    # (like {"d":[0,1]}) calculate sum of them.

    Example:
    p(a|d=0):
    enumeration_ask(net, "a", {"d": 0})

    p(a|d>=1):
    sim1.enumeration_ask(net, "a", {"d":[1, 2, 3]})
    '''

    # p.df.loc[0]*p.df.loc[1]
    vals = bn.get_node(var).get_var_values(var)
    q = pd.DataFrame(np.zeros(len(vals)), index=vals)
    # print("q:")
    # print(q)
    for val in vals:
        # print("val:", val)
        # sum added for case when events var value is a list
        #  like {"d":[0,1]}
        q.loc[val, :] = enumeration_all(bn, bn.sorted_vars.copy(),
                                        update({var: val}, events)).sum()
        # print("a", a.sum())
        # q.loc[val, :] = a
    return(q/q.sum())


def enumeration_all(bn, vars, events):
    # calculate p(var=val|events)
    if len(vars) == 0:
        return(np.ones(1))
    y, rest = vars[0], vars[1:]
    p = bn.get_node(y)
    if y in events:
        return(p(update({y: events[y]}, parents(p, events))).to_numpy()
               * enumeration_all(bn, rest, events))
    else:
        '''
        print("sum")
        print(p.get_var_values(y))
        print([(p(update({y: y_val}, parents(p, events))).to_numpy(),
               enumeration_all(bn, rest, update({y: y_val}, events)))
               for y_val in p.get_var_values(y)])
        '''
        return(sum([p(update({y: y_val}, parents(p, events))).to_numpy()
                    * enumeration_all(bn, rest, update({y: y_val}, events))
                    for y_val in p.get_var_values(y)]))


def parents(p, events):
    # events must alredy have parents
    # because of `BayesNet.sort` had been already applied:
    return(dict([(var, events[var]) for var in p.parents]))


def update(d1, d2):
    if type(d1) == dict:
        return(dict([(key, d1[key]) for key in d1]
                    + [(key, d2[key]) for key in d2]))
    elif type(d1) == str and type(d2) == dict:
        return(d1 + reduce(
            lambda acc, x: " and " + x[0] + "==" + str(x[1]) + acc,
            d2.items(), ""))


def enumeration_join(p, var, events={}):
    # calculate p(var, events)
    # events like "X>0 & Y=1":
    # if type(events) == str:
    result = p(events).sum(level=var)
    return(result)
    
    # events like {"X": 0}:
    # assert var not in events
    # result = p(eve)
    # var_values = p.support[p.vars.index(var)]
    
    # result = sum([p(update(events, {var: val})).values
    #               for val in var_values])
    # return(result)


def make_dist(pyro_or_torch_dist, var, *args,
              parents=[], params=[], **kwgs):
    class ContDist(pyro_or_torch_dist):
        def __init__(self, var, *args, parents=[], params=[], **kwgs):
            
            pyro_or_torch_dist.__init__(self, *args, **kwgs)
            self.__name__ = pyro_or_torch_dist.__name__
            self.var = var
            self.parents = parents
            self.params = params
            
        def reset_params(self, samples):
            '''
            Support only for continues parents, whose names
            is `torch.tensor`
            '''
            # current_params = dict([(pname, getattr(self, pname))
            #                         for pname in self.arg_constraints])
            u_params = dict([(param_name, samples[param_parent_name])
                             for param_name, param_parent_name in self.params])
            dnode_pnames = [param_parent_name
                            for param_name, param_parent_name in self.params]
            
            # print("u_params:")
            # print(u_params)
            # current_params.update(u_params)
        
            for pname in u_params:
                pvalue = u_params[pname]
                setattr(self, pname, pvalue)
                # setattr(self, pname, torch.tensor(pvalue))
                
            return(dnode_pnames)

        def sample(self, events):
            # TODO: fix for multiply continues dist
            return(pyro_or_torch_dist.sample(self))
            # return(np.array(pyro_or_torch_dist.sample(self)))
        '''
        def sample(self, *args, **kwargs):
            return(np.array(pyro_or_torch_dist.sample(self, *args, **kwargs)))
        '''

        def get_support_index(self, sample, events, var_only=False):
            return(sample)

    # dist = pyro_or_torch_dist
    # dist.var = var
    # dist.parents = parents
    # dist.params = []
    return(ContDist(var, *args,
                    parents=parents, params=params, **kwgs))


def plot_results(dist, events, N=30, count=300):
    '''
    - ``N`` -- number of states
    '''
    support = list(dist.enumerate_support(events))
    result = []
    for k in range(count):
        r = dist.sample(events)
        # for test log_prob:
        p = np.exp(dist.log_prob(events))
        result.append(support.index(r))
        # r1 = list(r)
        # r1.reverse()
        # result.append(reduce(lambda acc, x: 10**(len(str(acc)))*x+acc, r1, 0))
        # result.append(r)
    
    print("r:", r)
    print("p:", p)

    # label = str(support)
    label = [str(c) for c in support]

    print("result:")
    print(result)
    print(set(result))
    print("label:")
    print(label)
    plt.hist(result, N,  # density=True,
             stacked=True,
             rwidth=0.1, label=label)
    # plt.legend(label)
    plt.show()


def plot_results1(vars, isamples):
    for var in vars:
        print("var:")
        print(var)
        var_data = np.array(isamples[var]).astype(np.float)
        # print(var_data)
        print("probs of var (bin=2):")
        var_data = torch.histc(torch.tensor(var_data), 2) 
        print(var_data/var_data.sum())

        plt.hist(isamples[var], 30,  # density=True,
                 stacked=True
                 # , rwidth=0.1, label=label
                 )
        plt.show()

    # plt.legend(label)


def rejection_sampling(net, N, cond, cprogress=ProgressCmd):
    progress = cprogress(N)
    events = {}

    samples = net.init_samples(default=lambda var: [])
    indexes = net.init_samples(default=lambda var: [])

    for step in range(N):
        sample = net.prior_sample(events)
        if cond is not None:
            if not check_cond(sample, cond):
                continue

        for var in samples:
            samples[var].append(sample[var])

            # collect indexes (int) instead of multiIndex (tuples): 
            indexes[var].append(net.dnodes[var]
                                .get_support_index(sample[var], events,
                                                   var_only=True))
        progress.succ(step)
    progress.print_end()
    # labels = net.get_labels(events)
    labels = None
    return(samples, indexes, labels)


def check_cond(samples, cond, dbg=None):
    cond = Template(cond)
    if dbg is not None:
        if "check_cond" in dbg:
            if dbg[check_cond]:
                print("samples:")
                print(samples)
                print("cond:")
                print(cond.substitute(samples))
    return(eval(cond.substitute(samples)))


def test_bayes_EM():
    '''solving 20.3.2. Russel-Norving'''
    f = lambda count, h1, h2, h3, b1: count*h1*h2*h3*0.6/(h1*h2*h3*0.6 +(1-h1)*(1-h2)*(1-h3)*0.4)*0.001 if b1 else count*(1-h1)*(1-h2)*(1-h3)*0.4/(h1*h2*h3*0.6+(1-h1)*(1-h2)*(1-h3)*0.4)*0.001   
    result = sum([f(count, h1, h2, h3, 1)
                  for count, h1, h2, h3 in [(273, 0.6, 0.6, 0.6),
                                            (79, 0.4, 0.6, 0.6),
                                            (93, 0.4, 0.6, 0.6),
                                            (100, 0.4,0.4,0.6),
                                            (104, 0.4, 0.6, 0.6),
                                            (94, 0.4, 0.4, 0.6),
                                            (90,0.4, 0.4, 0.6),
                                            (167, 0.4, 0.4, 0.4)]])
    return(result)


def learn_cs_em(observed, steps, imu, isigma, iw, cs_counts=3,):
    '''Mixture of Gaussian'''
    N = len(observed)

    # dublicate observed for each class:
    observed = torch.cat([torch.unsqueeze(observed, 0)
                          for k in range(cs_counts)], 0).T
    observed_cs = torch.cat(
        [torch.unsqueeze(torch.arange(0, cs_counts), 0)
         for k in range(N)], 0)
    # print("observed:")
    # print(observed.shape)
    # print("observed_cs:")
    # print(observed_cs.shape)

    # init values
    
    mu = imu
    # mu = torch.tensor([1., 3.5, 4.0])
    sigma = isigma
    # sigma = torch.tensor([1., 1.0, 1.0])
    w = iw
    # w = torch.tensor([0.1, 0.2, 0.7])
    gen_p_xj_ci = lambda mu, sigma: dist.Normal(mu, sigma)
    gen_p_ci = lambda w: dist.Binomial(cs_counts, w)

    mus, sigmas, wis = [], [], []

    for step in range(steps):
        dist_xj_ci = gen_p_xj_ci(mu, sigma)
        dist_ci = gen_p_ci(w)
        
        # FOR computing p_ci_xj:
        # print("test1")
        # print(dist_ci.sample())
        # print(dist_xj_ci.log_prob(observed).shape)
        # print(dist_ci.log_prob(observed_cs.float()).shape)
        
        mult = torch.exp(dist_xj_ci.log_prob(observed)
                         + dist_ci.log_prob(observed_cs.float()))
        
        # normalization constant:
        # print(mult)
        
        alpha = mult.sum(1)**(-1.)
        # print("alpha")
        # print(alpha)
        
        p_ci_xj = (alpha * mult.T).T
        # print("p_ci_xj:")
        # print(p_ci_xj.shape)
        # print(p_ci_xj)
        
        # END FOR

        ni = p_ci_xj.sum(0)
        # print("ni:")
        # print(list(ni.shape))
        # print(ni)
        
        mu = (p_ci_xj * observed).sum(0)/ni
        # print("mu:")
        # print(list(mu.shape))
        # print(mu)

        sigma = (p_ci_xj * (observed-mu)**2).sum(0)/ni
        # print("sigma:")
        # print(sigma.shape)
        # print(sigma)

        wi = ni/N
        # print("wi:")
        # print(wi.shape)
        # print(wi)
        # return
        mus.append(mu)
        sigmas.append(sigma)
        wis.append(wi)
    # plt.scatter(a,b)
    
    return(torch.cat([torch.unsqueeze(mu, 0)
                      for mu in mus], 0),
           torch.cat([torch.unsqueeze(sigma, 0)
                      for sigma in sigmas], 0),
           torch.cat([torch.unsqueeze(wi, 0)
                      for wi in wis], 0))


def test_learn_cs_em():
    steps = 4

    imu = torch.tensor([1., 2., 3.])
    isigma = torch.tensor([0.3, 0.3, 1.])
    iwis = torch.tensor([0.1, 0.4, 0.5])
    observed = sample_cs(100, imu, isigma, iwis)

    limu = torch.tensor([1., 3.5, 4.0])
    lisigma = torch.tensor([1., 1.0, 1.0])
    liwi = torch.tensor([0.1, 0.2, 0.7])
    mus, sigmas, wis = learn_cs_em(observed, steps,
                                   limu, lisigma, liwi)

    print("initial model:")
    print((imu, isigma, iwis))

    print("lerned initial model:")
    print((limu, lisigma, liwi))
    
    print("learned model:")
    print((mus[-1], sigmas[-1], wis[-1]))
    
    # FOR plot:
    fig, axs = plt.subplots(3, 3)
    names = ["mu", "sigma", "wi"]
    data = [mus, sigmas, wis]

    for i in range(3):
        name = names[i]
        for j in range(3):
            axs.flat[3*i+j].plot(data[i].T[j])
            axs.flat[3*i+j].set_title("%s %d" % (name, j))
    plt.show()
    # END FOR


def sample_cs(steps, mu, sigma, cs_probs):
    dist_ci = dist.Categorical(cs_probs)
    dist_xj_ci = dist.Normal(mu, sigma)
    cs_sampler = lambda: dist_xj_ci.sample()[dist_ci.sample()]
    return(torch.tensor([cs_sampler() for i in range(steps)]))


def test_sample_cs(mu, sigma, cs_probs):
    import matplotlib.pyplot as plt
    
    result = sample_cs(1000, mu, sigma, cs_probs)

    plt.hist(result, 70)
    plt.show()


def test_enumeration_join():
    p = Probability(vars=["a", "d"],
                    support=[[0.1, 0.3], [0, 1, 2, 3]])
    # note multiplication needed for p(a, d) = p(a|d)*p(d)
    # (p(a|d) represented in list):
    p.set({"a": 0.1}, np.array([0.729, 0.243, 0.027, 0.001])*0.7)
    p.set({"a": 0.3}, np.array([0.343, 0.441, 0.189, 0.027])*0.3)
    
    res = enumeration_join(p, "a", events={"d": 0})
    print("calculate p(a, d==0):")
    print(res)

    print("normalization (p(a|d==0) = alpha*p(a, d==0)):")
    print(res/res.sum())
 
    res = enumeration_join(p, "a", events="d>=1")
    print("calculate p(a, d>=1)")
    print(res)

    print("normalization (p(a|d>=1) = alpha*p(a, d>=1)):")
    print(res/res.sum())


def test_enumeration_ask():
    p0 = CondProb("d", parents=["a"],
                  support=[[0, 1, 2, 3], [0.1, 0.3]])
    # set P(d|a=0.1):
    p0.set({"a": 0.1}, [0.729, 0.243, 0.027, 0.001])
    p0.set({"a": 0.3}, [0.343, 0.441, 0.189, 0.027])
    print("p0(d|a):")
    print(p0)
    # print(p0({"a": 0.1}))
    
    p1 = CondProb("a", parents=[], support=[[0.1, 0.3]])
    # p1 = Probability(vars=["a"], support=[[0.1, 0.3]])
    p1.set({}, [0.7, 0.3])
    print("p1(A):")
    print(p1)

    net = BayesNet([p0, p1])
    print("p(d=0)")
    #  = p(d=0|a=0.1)p(a=0.1)+p(d=0|a=0.3)*p(a=0.3):
    result = enumeration_all(net, ["a", "d"], {"d": 0})
    # error: order incorrect:
    # result = enumeration_all(net, ["d", "a"], {})
    print('enumeration_all(net, ["a", "d"], {"d": 0}):')
    print(result)

    print("p(a|d=0)")
    result = enumeration_ask(net, "a", {"d": 0})
    print('enumeration_ask(net, "a", {"d": 0}):')
    print(result)


def test_enumeration_ask1():
    p_x_y = CondProb("X", parents=["Y"],
                     support=[[0, 1], [0, 1, 2]])
    # set P(d|a=0.1):
    p_x_y.set({"Y": 0}, [0.1, 0.9])
    p_x_y.set({"Y": 1}, [0.5, 0.5])
    p_x_y.set({"Y": 2}, [0.7, 0.3])
    print("p(X|Y):")
    print(p_x_y)
    # print(p0({"a": 0.1}))
    
    p_y_z = CondProb("Y", parents=["Z"],
                     support=[[0, 1, 2], [0, 1]])
    p_y_z.set({"Z": 0}, [0.1, 0.2, 0.7])
    p_y_z.set({"Z": 1}, [0.4, 0.5, 0.1])
    print("p(Y|Z):")
    print(p_y_z)

    p_z = CondProb("Z", parents=[], support=[[0, 1]])
    p_z.set({}, [0.7, 0.3])
    print("p(Z):")
    print(p_z)

    net = BayesNet([p_x_y, p_y_z, p_z])
    print("net.sorted_vars:")
    print(net.sorted_vars)

    print("p(Z=0)")
    result = enumeration_all(net,  ["Z", "Y", "X"], {"Z": 0})
    print('enumeration_all(net, ["Z", "Y", "X"], {"Z": 0}):')
    print(result)
    print("===========================")
    print("p(Z|X=0)")
    result = enumeration_ask(net, "Z", {"X": 0})
    print('enumeration_ask(net, "Z", {"X": 0}):')
    print(result)

    print("\naccurate_test0 p(X=0, Z=0):")
    print(p_x_y({"X": 0})*p_y_z({"Z": 0}))
    print(p_x_y({"X": 0})*p_z({"Z": 0}))
    print(p_z({"Z": 0}))
          
    print((p_x_y({"X": 0})*p_y_z({"Z": 0})*p_z({"Z": 0})).sum())
    print("accurate_test0.1 p(X=0, Z=0):")
    print(p_x_y({"X": 0, "Y": 0})*p_y_z({"Y": 0, "Z": 0})*p_z({"Z": 0})
          + p_x_y({"X": 0, "Y": 1})*p_y_z({"Y": 1, "Z": 0})*p_z({"Z": 0})
          + p_x_y({"X": 0, "Y": 2})*p_y_z({"Y": 2, "Z": 0})*p_z({"Z": 0}))
    print("accurate_test0.2 p(X=0, Z=0):")
    print(0.1*0.1*0.7+0.5*0.2*0.7+0.7**3)

    # print((p_x_y({"X": 0})*p_y_z({"Z": 0})*p_z({"Z": 0})).sum()/)
    
    print("\naccurate_test1 p(X=0):")
    print(sum([(p_x_y({"X": 0})*p_y_z({"Z": z})*p_z({"Z": z})).sum()
               for z in [0, 1]]))
    print("accurate_test1.1 p(X=0):")
    print(p_x_y({"X": 0, "Y": 0})*p_y_z({"Y": 0, "Z": 0})*p_z({"Z": 0})
          + p_x_y({"X": 0, "Y": 1})*p_y_z({"Y": 1, "Z": 0})*p_z({"Z": 0})
          + p_x_y({"X": 0, "Y": 2})*p_y_z({"Y": 2, "Z": 0})*p_z({"Z": 0})

          + p_x_y({"X": 0, "Y": 0})*p_y_z({"Y": 0, "Z": 1})*p_z({"Z": 1})
          + p_x_y({"X": 0, "Y": 1})*p_y_z({"Y": 1, "Z": 1})*p_z({"Z": 1})
          + p_x_y({"X": 0, "Y": 2})*p_y_z({"Y": 2, "Z": 1})*p_z({"Z": 1}))
    
    print("\naccurate_test p(Z=0|X=0):")
    print((p_x_y({"X": 0})*p_y_z({"Z": 0})*p_z({"Z": 0})).sum() /
          sum([(p_x_y({"X": 0})*p_y_z({"Z": z})*p_z({"Z": z})).sum()
               for z in [0, 1]]))
    # wrong: print(p_x_y({"X": 0})*p_y_z({})*p_z({}))
    # wrong: print(p_y_z({"Y": 0})*p_z({}))
    print("p(Z|X=1)")
    result = enumeration_ask(net, "Z", {"X": 1})
    print('enumeration_ask(net, "Z", {"X": 1}):')
    print(result)

    print("p(Z|X=0, Y<=1):")
    result = enumeration_ask(net, "Z", {"X": 0, "Y": [0, 1]})
    print('enumeration_ask(net, "Z", {"X": 0, "Y":[0, 1]}):')
    print(result)

    # this will not work:
    print('enumeration_ask(net, "Z", "X==0 & Y<=1"):')
    try:
        result = enumeration_ask(net, "Z", "X==0 & Y<=1")
        print(result)
    except:
        print("do not work")

    
def test_enumeration_ask2():
    p_x_y = CondProb("X", parents=["Y"],
                     support=[[0, 1], [0, 1]])
    p_x_y.set({"Y": 0}, [0.1, 0.9])
    p_x_y.set({"Y": 1}, [0.5, 0.5])
    print("p(X|Y):")
    print(p_x_y)
    
    p_y_z1_z2 = CondProb("Y", parents=["Z1", "Z2"],
                         support=[[0, 1]]*3)
    p_y_z1_z2.set({"Y": 0}, [0.1, 0.2, 0.05, 0.05])
    p_y_z1_z2.set({"Y": 1}, [0.1, 0.3, 0.1, 0.1])
    print("p(Y|Z1, Z2):")
    print(p_y_z1_z2)

    p_z1 = CondProb("Z1", parents=[], support=[[0, 1]])
    p_z1.set({}, [0.7, 0.3])
    print("p(Z1):")
    print(p_z1)

    p_z2 = CondProb("Z2", parents=[], support=[[0, 1]])
    p_z2.set({}, [0.5, 0.5])
    print("p(Z2):")
    print(p_z2)

    net = BayesNet([p_x_y, p_y_z1_z2, p_z1, p_z2])
    print("net.sorted_vars:")
    print(net.sorted_vars)

    print("p(Z1=0)")
    result = enumeration_all(net,  ["Z1", "Z2", "Y", "X"], {"Z1": 0})
    print('enumeration_all(net, ["Z1", "Z2", "Y", "X"], {"Z1": 0}):')
    print(result)
    result = enumeration_all(net,  ["Z2", "Z1", "Y", "X"], {"Z1": 0})
    print('enumeration_all(net, ["Z2", "Z1", "Y", "X"], {"Z1": 0}):')
    print(result)

    print("p(Z1=1)")
    result = enumeration_all(net,  ["Z1", "Z2", "Y", "X"], {"Z1": 1})
    print('enumeration_all(net, ["Z1", "Z2", "Y", "X"], {"Z1": 1}):')
    print(result)
    
    print("===========================")
    print("p(Z2|X=0)")
    result = enumeration_ask(net, "Z2", {"X": 0})
    print('enumeration_ask(net, "Z2", {"X": 0}):')
    print(result)


def test_prob_net():

    p0 = CondProb("X1", parents=["Y1", "Y2"], support=[[0, 1]]*3)
    p1 = CondProb("Y1", parents=["Z1", "Z2"], support=[[0, 1]]*3)
    p2 = CondProb("X2", parents=["Y1", "Y2"], support=[[0, 1]]*3)
    p3 = CondProb("Y2", parents=["Z2"])
    p4 = CondProb("T1", parents=["X1"])
    p5 = CondProb("Z1", parents=[], support=[[0, 1]])
    p6 = CondProb("Z2", parents=[], support=[[0, 1]])
    net = BayesNet([p0, p1, p2, p3, p4, p5, p6])

    print("net.sorted_vars:")
    print(net.sorted_vars)
    print("net.sorted_nodes:")
    print(list(map(lambda node: node.var, net.sorted_nodes)))
    # ['T1', 'X1', 'X2', 'X2', 'Y1', 'Y2', 'Z1', 'Z2', 'Z2']


def test_discrete_simple(events="Y==0"):
    p0 = CondProb("Y", parents=["Z1", "Z2"], support=[[0, 1]]*3)
    # \sum_{y}(p(y|z1, z2)) = 1:
    p0.set({"Y": 0}, [0.1, 0.2, 0.5, 0.4])
    p0.set({"Y": 1}, [0.9, 0.8, 0.5, 0.6])
    print("p0(%s):" % str(events))
    print(p0(events))
    print("p0.sample:")
    print(p0.sample(events))
    print("p0.log_prob:")
    print(p0.log_prob(events))
    plot_results(p0, events)
    

def test_prior():
    p = make_dist(dist.Uniform, "a", 0, 1)

    # p = make_dist(dist.Uniform(0, 1), "a")
    p0 = CondProb("Bag", parents=["a"], support=[[0, 1], ["a"]],
                  params=[("Bag==0", "a")])
    
    p1 = CondProb("Color", parents=["Bag"],
                  support=[["red", "green"], [0, 1]])
    p1.set({"Bag": 0}, [0.1, 0.5])
    p1.set({"Bag": 1}, [0.9, 0.5])

    net = BayesNet([p, p0, p1])

    print("net.sorted_vars:")
    print(net.sorted_vars)
    print("net.sorted_nodes:")
    print(list(map(lambda node: node.var, net.sorted_nodes)))
    # ['T1', 'X1', 'X2', 'X2', 'Y1', 'Y2', 'Z1', 'Z2', 'Z2']
    samples = net.prior_sample()
    print("net.prior_sample():")
    print(samples)
    print("p0:")
    print(p0)

    print('net.prior_sample():')
    samples = net.prior_sample()
    print(samples)

    print('net.prior_sample("Bag==0"):')
    samples = net.prior_sample("Bag==0")
    print(samples)
        

def test_rejection_sampler(N=3, cond="$a>=0.7"):
    '''
    Model:
    {a| a=Uniform(0, 1)}
      ->{Bag| [p(Bug=0)=1-a, p(Bug=1)=a]}
       ->{Color| [p(Color|"Bag": 0)=[0.1, 0.9],
                  p(Color|"Bag": 1) = [0.5, 0.5]]}
    Ask what p(X|cond) will be for X is some of [a, Bag, Color]
  
    Requirement:
    Make shure
    All children must use parents samples result.
    So if Bag=(bag_number, 'a'), Color must be ('green'/'red', bag_number)
  
    Tests:

    cond="$Bag==(0, 'a') and $Color==('green', 0)
    cond="$a>=0.7"    
    cond = "True"
    '''
    
    p = make_dist(dist.Uniform, "a", 0, 1)
    # p = make_dist(dist.Uniform(0, 1), "a")
    
    # P(Bag==1) = a:
    p0 = CondProb("Bag", parents=["a"], support=[[0, 1], ["a"]],
                  params=[("Bag==1", "a")])
    
    # P(Color|Bag):
    p1 = CondProb("Color", parents=["Bag"],
                  support=[["red", "green"], [0, 1]])
    p1.set({"Bag": 0}, [0.1, 0.9])
    p1.set({"Bag": 1}, [0.5, 0.5])

    net = BayesNet([p, p0, p1])

    print("net.sorted_vars:")
    print(net.sorted_vars)

    print('rejection_sampling(net, %s)' % cond)
    samples, indexes, labels = rejection_sampling(net, N,
                                                  cond=cond)
    print("len(samples)")
    print(len(samples['Bag']))
    # print(indexes)
    # print(labels)

    plot_results1(net.sorted_vars, indexes)


def test_rejection_sampler1(N=3, cond="$a>=0.7"):
    '''
    Model:
    {a| a=Uniform(0, 1)}
     ->{b| b=Uniform(a, 1)}
      ->{Bag| [p(Bug=0)=1-b, p(Bug=1)=b]}
       ->{Color| [p(Color|"Bag": 0)=[0.1, 0.9],
                  p(Color|"Bag": 1) = [0.5, 0.5]]}
    Ask what p(X|cond) will be for X is some of [a, b, Bag, Color]

    Requirement:
    Make shure
    All children must use parents smaples result.
    So if Bag=(bag_number, 'a'), Color must be ('green'/'red', bag_number)

    Tests:

    cond="$Bag==(0, 'a') and $Color==('green', 0)
    cond="$a>=0.7"    
    cond = "True"
    '''
    
    p = make_dist(dist.Uniform, "a", 0, 1)
    # p = make_dist(dist.Uniform(0, 1), "a")
    pp = make_dist(dist.Uniform, "b", 0, 1,
                   parents=["a"], params=[("low", "a")])

    # P(Bag==1) = b (and P(Bag!=1) = 1-b):
    p0 = CondProb("Bag", parents=["b"], support=[[0, 1], ["b"]],
                  params=[("Bag==1", "b")])
    
    # P(Color|Bag):
    p1 = CondProb("Color", parents=["Bag"],
                  support=[["red", "green"], [0, 1]])
    p1.set({"Bag": 0}, [0.1, 0.9])
    p1.set({"Bag": 1}, [0.5, 0.5])

    net = BayesNet([p, p0, p1, pp])

    print('rejection_sampling(net, %s)' % cond)
    samples, indexes, labels = rejection_sampling(net, N,
                                                  cond=cond)
    print("len(samples):")
    print(len(samples["Bag"]))
    # print(samples)
    # print(indexes)
    # print(labels)
    plot_results1(net.sorted_vars, indexes)


def test_rejection_sampler2(N=3, cond="$x<0.5", sigm_x=0.1,
                            sigm_y=0.1,  mu_z=0.0, sigm_z=0.1):
    '''
    {z| Normal(0, 0.1)}
     ->{y| Normal(z, 0.1)}
      ->{x| Normal(y, 0.1) and x<0.5}
    '''

    z = make_dist(dist.Normal, "z", mu_z, sigm_z)

    # bellow mu_z used only for init
    # subsequently z and y will be used instead:
    y = make_dist(dist.Normal, "y", mu_z, sigm_y,
                  parents=["z"], params=[("loc", "z")])
    x = make_dist(dist.Normal, "x", mu_z, sigm_x,
                  parents=["y"], params=[("loc", "y")])

    net = BayesNet([x, y, z])
    print('rejection_sampling(net, %s)' % cond)
    
    print("\ncond: ", cond)
    samples, indexes, labels = rejection_sampling(net, N,
                                                  cond=cond)
    print("len(samples):")
    print(len(samples["x"]))
    plot_results1(net.sorted_vars, indexes)

    
if __name__ == "__main__":

    '''
    {z| Normal(0, 0.1)}
     ->{y| Normal(z, 0.1)}
      ->{x| Normal(y, 0.1)}
    '''
    # test_rejection_sampler2(N=1700, cond="$x<=0.5",
    #                         sigm_z=0.1, sigm_y=0.1, sigm_x=0.1)
    # count of succ will be ~ 1695
    # => P(x<=0.5) ~ 1695/1700 ~ 0.9971
    # accurate: 0.9980537912897127

    '''
    {z| Normal(0, 1.0)}
     ->{y| Normal(z, 1.0)}
      ->{x| Normal(y, 1.0)}
    '''
    # test_rejection_sampler2(N=1700, cond="$x<=0.5",
    #                         sigm_z=1.0, sigm_y=1.0, sigm_x=1.0)
    # count of succ will be ~ 1069
    # => P(x<=0.5) ~ 1069/1700 ~ 0.6288
    # accurate: 0.6135849997754309

    # test_rejection_sampler2(N=17000, cond="$x>=0.5",
    #                         sigm_z=0.1, sigm_y=0.1, sigm_x=0.1)
    # count of succ will be ~ 35
    # => P(x>=0.5) ~ 35/17000 ~ 0.0020
    # accurate: 0.0019462084167189077

    '''
    {z| Normal(0.5, 1.0)}
     ->{y| Normal(z, 1.0)}
      ->{x| Normal(y, 0.1)}
    '''
    # test_rejection_sampler2(N=1700, cond="$x<0.5 and $z<0.5",
    #                         sigm_x=0.1, sigm_y=1.0, mu_z=0.5, sigm_z=1.0)
    # count of succ will be ~ 607
    # => P(x>=0.5) ~ 607/1700 ~ 0.3571
    # accurate: 0.3746040905409271

    '''
    {a| a=Uniform(0, 1)}
     ->{b| b=Uniform(a, 1)}
      ->{Bag| [p(Bug=0)=1-b, p(Bug=1)=b]}
       ->{Color| [p(Color|"Bag": 0)=[0.1, 0.9],
                  p(Color|"Bag": 1) = [0.5, 0.5]]}
    '''
    # test_rejection_sampler1(N=700, cond="$a>=0.0")
   
    '''
    {a| a=Uniform(0, 1)}
      ->{Bag| [p(Bug=0)=1-a, p(Bug=1)=a]}
       ->{Color| [p(Color|"Bag": 0)=[0.1, 0.9],
                  p(Color|"Bag": 1) = [0.5, 0.5]]}
    '''
    test_rejection_sampler(N=3, cond="$a>=0.7")
    # test_rejection_sampler(N=700, cond="$a>=0.7")
    # test_prior()
    # test_discrete_simple(events=None)
    # test_discrete_simple(events="Z1+Z2==1")
    # test_enumeration_ask2()
    # test_enumeration_ask1()
    # test_enumeration_ask()
    # test_enumeration_join()
    # test_enumeration0()
    # test_prob_net()
    # test_learn_cs_em()
    # test_sample_cs(torch.tensor([0.0, 0.5, 1.0]),
    #                torch.tensor([0.1, 0.1, 0.7]),
    #                torch.tensor([0.2, 0.2, 0.6]))
