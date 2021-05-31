import torch

import pyro
import pyro.distributions as pdist
from pyro import poutine
from torch.distributions import constraints
from pyro.ops.indexing import Vindex


import matplotlib.pyplot as plt


def model(obs):
    # all will be overwritten in guide:
    
    a = pyro.sample("a", pdist.Binomial(2, 0.3))
    d = pdist.Bernoulli(0.5)
    b = pyro.sample("b", d, obs=obs)
    # dout = pdist.Bernoulli()
    # out = pyro.sample("out", dout, obs=obs)
    return b


def model1(obs, n):
    p_a = torch.tensor(0.99)
    p_b = torch.tensor([0.1, 0.3, 0.6])
                     
    with pyro.plate("plate1", n):
        a = pyro.sample("a", pdist.Binomial(2, p_a))

        d = pdist.Bernoulli(Vindex(p_b)[a.type(torch.long)])

        # log_p will still work correctly in this case because it
        # use d.log_p, and d has probability p_b[a]
        # so p(b=0, a=1) = p(b=0|a=1)*p(a=1) = 0.7*(2*p*(1-p))
        # = 0.7*2*0.5*0.5 = 0.7*0.5 = 0.35
        b = pyro.sample("b", d)
    
    # cdist = pdist.
    return b


def model_cond1(obs):
    p_a = pyro.param("p(a=1)", torch.tensor(0.1),
                     constraint=constraints.interval(0, 0.9))
    a = pyro.sample("a", pdist.Binomial(2, p_a))

    p_b_a_0 = pyro.param("p(b=1|a=0)", torch.tensor(0.1),
                         constraint=constraints.interval(0, 0.9))
    p_b_a_1 = pyro.param("p(b=1|a=1)", torch.tensor(0.1),
                         constraint=constraints.interval(0, 0.9))
    p_b_a_2 = pyro.param("p(b=1|a=1)", torch.tensor(0.1),
                         constraint=constraints.interval(0, 0.9))

    p_b = torch.tensor([p_b_a_0, p_b_a_1, p_b_a_2])

    d = pdist.Bernoulli(p_b[a.type(torch.long)])
    
    # log_p will still work correctly in this case because it
    # use d.log_p, and d has probability p_b[a]
    # so p(b=0, a=1) = p(b=0|a=1)*p(a=1) = 0.7*(2*p*(1-p))
    # = 0.7*2*0.5*0.5 = 0.7*0.5 = 0.35
    b = pyro.sample("b", d, obs=obs)
    
    # cdist = pdist.
    return b
 

def guide(obs):
    p_a = pyro.param("p(a=1)", torch.tensor(0.1),
                     constraint=constraints.interval(0, 0.9))
    # p_a.requires_grad_()

    a = pyro.sample("a", pdist.Binomial(2, p_a))

    p_b_a_0 = pyro.param("p(b=1|a=0)", torch.tensor(0.1),
                         constraint=constraints.interval(0, 0.9))
    p_b_a_1 = pyro.param("p(b=1|a=1)", torch.tensor(0.2),
                         constraint=constraints.interval(0, 0.9))
    p_b_a_2 = pyro.param("p(b=1|a=2)", torch.tensor(0.3),
                         constraint=constraints.interval(0, 0.9))
    # p_b_a_0.requires_grad_()
    # p_b_a_1.requires_grad_()
    # p_b_a_2.requires_grad_()
    p_b = torch.tensor([p_b_a_0, p_b_a_1, p_b_a_2])

    ### d = pdist.Bernoulli(p_b[a.type(torch.long)])
    
    # log_p will still work correctly in this case because it
    # use d.log_p, and d has probability p_b[a]
    # so p(b=0, a=1) = p(b=0|a=1)*p(a=1) = 0.7*(2*p*(1-p))
    # = 0.7*2*0.5*0.5 = 0.7*0.5 = 0.35
    ### b = pyro.sample("b", d)
    
    ### return b
    # weight = pyro.sample("weight", pdist.Normal(b, 1.0))
    # return pyro.sample("measurement", pdist.Normal(weight, 0.75))


def model_cond2(obs):
    p_a = pyro.sample("p(a=1)", pdist.Uniform(0, 0.9))
    a = pyro.sample("a", pdist.Binomial(2, p_a))

    p_b_a_0 = pyro.sample("p(b=1|a=0)", pdist.Uniform(0, 0.9))
    p_b_a_1 = pyro.sample("p(b=1|a=1)", pdist.Uniform(0, 0.9))
    p_b_a_2 = pyro.sample("p(b=1|a=2)", pdist.Uniform(0, 0.9))

    p_b = torch.tensor([p_b_a_0, p_b_a_1, p_b_a_2])

    d = pdist.Bernoulli(p_b[a.type(torch.long)])
    
    # log_p will still work correctly in this case because it
    # use d.log_p, and d has probability p_b[a]
    # so p(b=0, a=1) = p(b=0|a=1)*p(a=1) = 0.7*(2*p*(1-p))
    # = 0.7*2*0.5*0.5 = 0.7*0.5 = 0.35
    b = pyro.sample("b", d, obs=obs)
    
    # cdist = pdist.
    return b


def model_cond3(obs, p_a, p_b_a_0, p_b_a_1, p_b_a_2):
    # p_a = pyro.sample("p(a=1)", pdist.Uniform(0, 0.9))
    with pyro.plate("plate2", len(obs)) as ind:
        a = pyro.sample("a", pdist.Binomial(2, p_a))

        # p_b_a_0 = pyro.sample("p(b=1|a=0)", pdist.Uniform(0, 0.9))
        # p_b_a_1 = pyro.sample("p(b=1|a=1)", pdist.Uniform(0, 0.9))
        # p_b_a_2 = pyro.sample("p(b=1|a=2)", pdist.Uniform(0, 0.9))

        p_b = torch.tensor([0.1, 0.3, 0.6])
        # p_b = torch.tensor([p_b_a_0, p_b_a_1, p_b_a_2])
        ip_b = Vindex(p_b)[a.type(torch.long)]
        d = pdist.Bernoulli(ip_b)
        # print(ip_b)
        # d = pdist.Bernoulli(p_b[a.type(torch.long)])
    
        # log_p will still work correctly in this case because it
        # use d.log_p, and d has probability p_b[a]
        # so p(b=0, a=1) = p(b=0|a=1)*p(a=1) = 0.7*(2*p*(1-p))
        # = 0.7*2*0.5*0.5 = 0.7*0.5 = 0.35
        b = pyro.sample("b", d, obs=obs.index_select(0, ind))
    
    # cdist = pdist.
    return b


def model_cond4(obs, weights=None):
    # 10 is count of hist.bins:
    pDist = ParamDist(10)
    if weights is not None:
        pDist.reset_weights(weights)

    p_a = pyro.sample("p_a", pDist)
    with pyro.plate("plate2", len(obs)) as ind:
        a = pyro.sample("a", pdist.Binomial(2, p_a))

        # p_b_a_0 = pyro.sample("p(b=1|a=0)", pdist.Uniform(0, 0.9))
        # p_b_a_1 = pyro.sample("p(b=1|a=1)", pdist.Uniform(0, 0.9))
        # p_b_a_2 = pyro.sample("p(b=1|a=2)", pdist.Uniform(0, 0.9))

        p_b = torch.tensor([0.1, 0.3, 0.6])
        # p_b = torch.tensor([p_b_a_0, p_b_a_1, p_b_a_2])
        ip_b = Vindex(p_b)[a.type(torch.long)]
        d = pdist.Bernoulli(ip_b)
        # print(ip_b)
        # d = pdist.Bernoulli(p_b[a.type(torch.long)])
    
        # log_p will still work correctly in this case because it
        # use d.log_p, and d has probability p_b[a]
        # so p(b=0, a=1) = p(b=0|a=1)*p(a=1) = 0.7*(2*p*(1-p))
        # = 0.7*2*0.5*0.5 = 0.7*0.5 = 0.35
        b = pyro.sample("b", d, obs=obs.index_select(0, ind))
    
    # cdist = pdist.
    return b


def sequential_discrete_marginal2(conditioned_model, data,
                                  weights, dbg=False):

    '''
    REF: http://pyro.ai/examples/effect_handlers.html#Example:-exact-inference-via-sequential-enumeration
    '''
    from six.moves import queue  # queue data structures
    q = queue.Queue()  # Instantiate a first-in first-out queue
    q.put(poutine.Trace())  # seed the queue with an empty trace

    # as before, we fix the values of observed
    # random variables with poutine.condition
    # assuming data is a dictionary whose keys are
    # names of sample sites in model
    # conditioned_model = poutine.condition(model, data=data)

    # we wrap the conditioned model in a poutine.queue,
    # which repeatedly pushes and pops partially
    # completed executions from a Queue()
    # to perform breadth-first enumeration over
    # the set of values of all discrete sample sites in model
    enum_model = poutine.queue(conditioned_model, queue=q)

    # actually perform the enumeration by repeatedly tracing enum_model
    # and accumulate samples and trace log-probabilities for postprocessing
    samples, log_weights = [], []

    model_samples = []
    guide_samples = []

    uniform = pdist.Uniform(0, 0.9)

    model_log_weights = []
    guide_log_weights = []
    traces = []

    aDist = ParamDist(10)
    ws = {"p_a": torch.ones(10)*0.5}
    new_ws = {}
    for key in weights:
        new_ws[key] = weights[key].detach().clone()
    # new_ws = {"p_a": torch.ones(len(weights['p_a']))}
    # new_ws = {"p_a": torch.zeros(len(weights['p_a']))*0.5}
    
    while not q.empty():
        # p_a = aDist.sample()
        # p_a_idx = aDist.get_site_idx()
        model_trace_messenger = poutine.trace(enum_model)
        # model_trace = model_trace_messenger.get_trace(data)
        model_trace = model_trace_messenger.get_trace(data, ws['p_a'])  # weights["p_a"])
        # TODO: move to trace:
        val = model_trace.nodes['p_a']['value']
        p_a_idx = model_trace.nodes['p_a']['fn'].get_site_idx(val)
        # print("p_a_idx (from get_site_idx)")
        # print(p_a_idx)
        # print("value:")
        # print(model_trace.nodes['p_a']['value'])
        # print("model_trace.nodes['p_a']['fn'].log_prob(val)")
        # print(model_trace.nodes['p_a']['fn'].log_prob(val))
        # print("sites:")
        # print(model_trace.nodes['p_a']['fn'].sites)
        '''
        print("model_trace.log_prob_sum():")
        print(model_trace.log_prob_sum())
        for param in ["p_a", "a", "b"]:
            print("param: ", param)
            pval = model_trace.nodes[param]['value']
            print("param val: ", pval)
            print("param log: ", model_trace.nodes[param]['fn'].log_prob(pval))
        '''
        new_ws["p_a"][p_a_idx] += torch.exp(model_trace.log_prob_sum())
        # print("model_trace.nodes:")
        # print(model_trace.nodes)
        # print('new_ws["p_a"]:')
        # print(new_ws["p_a"])
        # model_samples.append(model_trace.nodes['a'])
        # model_log_weights.append(model_trace.log_prob_sum())

    # print("new_ws['p_a']")
    # print(new_ws["p_a"])
    
    # FOR adjusting:
    '''
    if (new_ws["p_a"] < 0).any():
        new_ws["p_a"] = (new_ws["p_a"]
                         + torch.abs(new_ws["p_a"].min())*torch.ones(len(new_ws["p_a"]))) 
    # print("new_ws['p_a']")
    # print(new_ws["p_a"])
    '''
    # for norientation:
    # new_ws["p_a"] = torch.functional.F.normalize(new_ws["p_a"], dim=0)
    # END FOR

    if dbg:
        print("new_ws:")
        print(new_ws)
    # p(A) = sum(P(A, b))
    return(new_ws, model_trace)
    # ws["p_a"] = new_ws["p_a"].detach().copy()
    '''
    model_log_weights = torch.stack([torch.tensor(lw)
                                     for lw in model_log_weights], 0)
    model_log_weights = -1. * model_log_weights/model_log_weights.sum()
    
    
    p = torch.tensor([torch.exp(weight) for weight in model_log_weights]).sum()
    # print("p")
    # print(p)
    # p = torch.log(p)
    '''


def sequential_discrete_marginal1(conditioned_model, data,
                                  old_trace=None, step=0.1, dbg=False):

    '''
    REF: http://pyro.ai/examples/effect_handlers.html#Example:-exact-inference-via-sequential-enumeration
    '''
    from six.moves import queue  # queue data structures
    q = queue.Queue()  # Instantiate a first-in first-out queue
    q.put(poutine.Trace())  # seed the queue with an empty trace

    # as before, we fix the values of observed
    # random variables with poutine.condition
    # assuming data is a dictionary whose keys are
    # names of sample sites in model
    # conditioned_model = poutine.condition(model, data=data)

    # we wrap the conditioned model in a poutine.queue,
    # which repeatedly pushes and pops partially
    # completed executions from a Queue()
    # to perform breadth-first enumeration over
    # the set of values of all discrete sample sites in model
    enum_model = poutine.queue(conditioned_model, queue=q)

    # actually perform the enumeration by repeatedly tracing enum_model
    # and accumulate samples and trace log-probabilities for postprocessing
    samples, log_weights = [], []

    model_samples = []
    guide_samples = []

    uniform = pdist.Uniform(0, 0.9)

    model_log_weights = []
    guide_log_weights = []
    traces = []

    if old_trace is None:
        p_a = torch.tensor(0.5)  # pyro.sample("p(a=1)", pdist.Uniform(0, 0.9))
        p_b_a_0 = torch.tensor(0.5)  # pyro.sample("p(b=1|a=0)", pdist.Uniform(0, 0.9))
        p_b_a_1 = torch.tensor(0.5)  # pyro.sample("p(b=1|a=1)", pdist.Uniform(0, 0.9))
        p_b_a_2 = torch.tensor(0.5)  # pyro.sample("p(b=1|a=2)", pdist.Uniform(0, 0.9))
    else:
        p_a = old_trace["p(a=1)"]
        p_b_a_0 = old_trace["p(b=1|a=0)"]
        p_b_a_1 = old_trace["p(b=1|a=1)"]
        p_b_a_2 = old_trace["p(b=1|a=2)"]

        p_a += pyro.sample("e(a=1)", pdist.Normal(0, step))
        p_b_a_0 += pyro.sample("e(b=1|a=0)", pdist.Normal(0, step))
        p_b_a_1 += pyro.sample("e(b=1|a=1)", pdist.Normal(0, step))
        p_b_a_2 += pyro.sample("e(b=1|a=2)", pdist.Normal(0, step))

    if p_a < 0.01:
        p_a = torch.tensor(0.1)
    if p_a > 0.99:
        p_a = torch.tensor(0.9)
    if p_b_a_0 < 0.01:
        p_b_a_0 = torch.tensor(0.1)
    if p_b_a_0 > 0.99:
        p_b_a_0 = torch.tensor(0.99)
    if p_b_a_1 < 0.01:
        p_b_a_1 = torch.tensor(0.1)
    if p_b_a_1 > 0.99:
        p_b_a_1 = torch.tensor(0.99)
    if p_b_a_2 < 0.01:
        p_b_a_2 = torch.tensor(0.1)
    if p_b_a_2 > 0.99:
        p_b_a_2 = torch.tensor(0.99)
    # print("p_a")
    # print(p_a)
    while not q.empty():
        model_trace_messenger = poutine.trace(enum_model)
        # model_trace = model_trace_messenger.get_trace(data)
        model_trace = model_trace_messenger.get_trace(data, p_a, p_b_a_0,
                                                      p_b_a_1, p_b_a_2)
        model_samples.append(model_trace.nodes['a'])
        model_log_weights.append(model_trace.log_prob_sum())
    if dbg:
        print("model_samples(p(a=1)):")
        print(model_samples)
    # p(A) = sum(P(A, b))

    model_log_weights = torch.stack([torch.tensor(lw)
                                     for lw in model_log_weights], 0)
    model_log_weights = -1. * model_log_weights/model_log_weights.sum()
    
    
    p = torch.tensor([torch.exp(weight) for weight in model_log_weights]).sum()
    # print("p")
    # print(p)
    # p = torch.log(p)
    current_score = p
    
    # print("model_log_wieghts:")
    # print(model_log_weights)
    # print("current_score:")
    # print(current_score)
    # current_score = model_trace.log_prob_sum()
    # not used:
    if old_trace is None:
        # here and bellow is last trace because all samples (like p_a)
        # will be fixed during while (except discrete)
        old_trace = {}
        old_trace["p(a=1)"] = p_a
        old_trace["p(b=1|a=0)"] = p_b_a_0
        old_trace["p(b=1|a=1)"] = p_b_a_1
        old_trace["p(b=1|a=2)"] = p_b_a_2
        
        # old_trace["score"] = - 10
        old_trace["score"] = 0.0001
        
    old_score = old_trace["score"]
    # old_score = old_trace.log_prob_sum()
    # it must not decrease:
    acceptance = min(current_score/old_score, 1)
    if dbg:
        print("acceptance:")
        print(acceptance)
    
    if acceptance > 1:
        '''
        old_trace = dict((name, model_trace.nodes[name]["value"])
                         for name in old_trace
                         if name != "score")
        '''
        old_trace["p(a=1)"] = p_a
        old_trace["p(b=1|a=0)"] = p_b_a_0
        old_trace["p(b=1|a=1)"] = p_b_a_1
        old_trace["p(b=1|a=2)"] = p_b_a_2
        
        old_trace["score"] = current_score
        
    else:
        
        u = uniform.sample()
        if u <= acceptance:
            '''
            old_trace = dict((name, model_trace.nodes[name]["value"])
                             for name in old_trace
                             if name != "score")
            '''
            old_trace["p(a=1)"] = p_a
            old_trace["p(b=1|a=0)"] = p_b_a_0
            old_trace["p(b=1|a=1)"] = p_b_a_1
            old_trace["p(b=1|a=2)"] = p_b_a_2
            
            old_trace["score"] = current_score
            # old_trace = model_trace
            # old_score = current_score

    # print("old_trace")
    # print(old_trace)
    
    '''
    names = ["a", "b", "p(a=1)", "p(b=1|a=0)",
             "p(b=1|a=1)", "p(b=1|a=1)"]
    model_samples.append(dict([(model_trace.nodes[name]["name"],
                                model_trace.nodes[name]["value"])
                               for name in names
                               if name in model_trace.nodes]))
    guide_samples.append(dict([(guide_trace.nodes[name]["name"],
                                guide_trace.nodes[name]["value"])
                               for name in names
                               if name in guide_trace.nodes]))
    # model_log_weights.append(guide_trace.nodes[name]["fn"].log_prob(node["value"]).sum())
    model_log_weights.append(model_trace.log_prob_sum())
    guide_log_weights.append(guide_trace.log_prob_sum())
    '''
    return(old_trace)
    '''
    # FOR calculating gradients:
    if dbg:
        print("\nguide_samples:")
        print(guide_samples)
        print("\nmodel_samples:")
        print(model_samples)
        print("\nexp(model_log_weights):")
        print(torch.exp(torch.tensor(model_log_weights)))
    p_ = model_log_weights[1]
    # p_ = min(model_log_weights)
    # for p_ in model_log_weights:
    #     p_.backward()
    # q = 1-p
    # p_ = torch.exp(torch.tensor(model_log_weights)).sum()
    p_.requires_grad_()
    q_ = torch.ones(1) - p_
    q_.requires_grad_()
    # END FOR
    
    return(q_)
    '''

def sequential_discrete_marginal(conditioned_model, guide, data, names):

    '''
    REF: http://pyro.ai/examples/effect_handlers.html#Example:-exact-inference-via-sequential-enumeration
    '''
    from six.moves import queue  # queue data structures
    q = queue.Queue()  # Instantiate a first-in first-out queue
    q.put(poutine.Trace())  # seed the queue with an empty trace

    # as before, we fix the values of observed
    # random variables with poutine.condition
    # assuming data is a dictionary whose keys are
    # names of sample sites in model
    # conditioned_model = poutine.condition(model, data=data)

    # we wrap the conditioned model in a poutine.queue,
    # which repeatedly pushes and pops partially
    # completed executions from a Queue()
    # to perform breadth-first enumeration over
    # the set of values of all discrete sample sites in model
    enum_guide = poutine.queue(guide, queue=q)

    # actually perform the enumeration by repeatedly tracing enum_model
    # and accumulate samples and trace log-probabilities for postprocessing
    samples, log_weights = [], []

    model_samples = []
    guide_samples = []

    model_log_weights = []
    guide_log_weights = []
    traces = []
    while not q.empty():
        guide_trace_messenger = poutine.trace(enum_guide)
        guide_trace = guide_trace_messenger.get_trace(data)
        model_ = pyro.poutine.replay(conditioned_model,
                                     trace=guide_trace)
        model_trace = poutine.trace(model_).get_trace(data)
        # import pdb; pdb.set_trace()
        
        # calculating gradient:
        '''
        elbo = 0
        for node_name, node in model_trace.nodes.items():
            if node["type"] == "sample":
                elbo += node["fn"].log_prob(node["value"]).sum()
            if not node["is_observed"]:
                elbo -= guide_trace.nodes[node_name]["fn"].log_prob(node["value"]).sum()
        elbo *= -1
        
        '''
        
        model_samples.append(dict([(model_trace.nodes[name]["name"],
                                    model_trace.nodes[name]["value"])
                                   for name in names
                                   if name in model_trace.nodes]))
        guide_samples.append(dict([(guide_trace.nodes[name]["name"],
                                    guide_trace.nodes[name]["value"])
                                   for name in names
                                   if name in guide_trace.nodes]))

        model_log_weights.append(model_trace.log_prob_sum())
        guide_log_weights.append(guide_trace.log_prob_sum())
        
        # TODO:
        # use reply for each model_trace and guide_trace for get
        # model for each discrete value in order to optimize of remained continuous variables
        traces.append((model_trace, guide_trace))
        # elbos.append(elbos)
        # trace.log_prob_sum().backward()

    print("\nparams:")
    print(dict((name, pyro.param(name))
               for name in ["p(a=1)", "p(b=1|a=0)",
                            "p(b=1|a=1)", "p(b=1|a=2)"]))
    print("\nmodel_samples:")
    print(model_samples)
    print("\nguide_samples:")
    print(guide_samples)
    
    print("\n###############")

    # FOR calculating gradients:
    print("\nexp(model_log_weights):")
    print(torch.exp(torch.tensor(model_log_weights)))
    for p_ in model_log_weights:
        p_.backward()
    p_ = torch.exp(torch.tensor(model_log_weights)).sum()
    p_.requires_grad_()
    
    p_.backward()
    
    '''
    print("\nparams:")
    print(setpyro.param(name)
               for name in ["p(a=1)", "p(b=1|a=0)",
                            "p(b=1|a=1)", "p(b=1|a=2)"]))
    
    optimizer = pyro.optim.Adam({})
    optimizer.step(params)
    '''
    print("\nexp(guide_log_weights):")
    print(torch.exp(torch.tensor(guide_log_weights)))
    # END FOR

    print("\n##############")
    p_support = torch.exp(torch.tensor(model_log_weights)).sum()
    print("\nsum(exp(model_log_weights))")
    print(p_support)
    # p_support.backward()
    # we take the samples and log-joints and turn them into a histogram:
    # samples = torch.stack(samples, 0)
    # log_weights = torch.stack(log_weights, 0)
    return(traces)
    # return samples, log_weights
    
    log_weights = log_weights - pdist.util.logsumexp(log_weights, dim=0)
    return pdist.Empirical(samples, log_weights)


class Empirical():
    def __init__(self, sites, weights):
        self.sites = sites
        self.reset_weights(weights)

    def reset_weights(self, weights):
        self.weights = weights
        self.uDist = pdist.Uniform(0., self.weights.sum()-0.01)

    def get_site_idx(self, value):
        '''idx taken from self.sample'''
        return(list(self.sites).index(value))
        # return(self.idx)

    def __call__(self):
        return(self.sample())

    def sample(self):

        u = self.uDist.sample()
        p = 0
        for idx, w in enumerate(self.weights):
            p += w
            if p >= u:
                self.idx = idx
                return(self.sites[idx])
        print("err p:", p)
        print("err w:", w)
        print("err u:", u)

    def mean(self):
        ws = self.weights/self.weights.sum()
        return(torch.tensor([site*ws[idx] for idx, site in enumerate(self.sites)]).sum())
            
    def enumerate_support(self):
        return(self.sites)

    def has_enumerate_support(self):
        return(True)

    def log_prob(self, site):
        idxs = torch.nonzero(self.sites == site, as_tuple=False)[0]
        # print("idxs:")
        # print(idxs)
        # idx = self.sites.index(site)
        # print("self.weights:")
        # print(self.weights)
        w = torch.tensor([self.weights[:idx].sum() for idx in idxs])
        # print("w:")
        # print(w)
        # print("self.uDist:")
        # print(self.uDist)
        return(self.uDist.log_prob(w))


class ParamDist(Empirical):
    def __init__(self, hist_count):
        sites = torch.linspace(0, 1, hist_count)
        weights = torch.ones(hist_count)*0.5
        Empirical.__init__(self, sites, weights)


def test_paramdist():
    N = 30
    pDist = ParamDist(N)
    print("pDist.sample():")
    print(pDist.sample())

    '''
    print("\ninit:")
    plot_results(pDist, N)

    # FOR change weights:
    oDist = pdist.Binomial(N, 0.5)
    
    sites = oDist.enumerate_support()
    weights = torch.exp(oDist.log_prob(sites))

    pDist.reset_weights(weights)
    print("pDist.sample() with Binomial(N)")
    print(pDist.sample())
    # END FOR

    print("\napproximation:")
    plot_results(pDist, N)
    print("\norientation:")
    plot_results(oDist, N)
    '''

    # FOR change weights:
    p = 0.9
    n = 7
    oDist = pdist.Binomial(n, torch.ones(N)*p)
    
    weights = torch.zeros(n)
    for i in range(300):
        r = oDist.sample()
        weights += torch.histc(r, bins=n)
    weights = torch.functional.F.normalize(weights, dim=0)
    print("weights:")
    print(weights)

    pDist.reset_weights(weights)
    print("pDist.sample() with Binomial()*N")
    print(pDist.sample())
    # END FOR
   
    print("\napproximation:")
    plot_results(pDist, N, 300)

    oDist = pdist.Binomial(3, p)
    print("\noriginal:")
    plot_results(oDist, N)
    

def test_emp():
    N = 3
    oDist = pdist.Binomial(N, 0.7)
    
    sites = oDist.enumerate_support()
    weights = torch.exp(oDist.log_prob(sites))
    
    print("sites:")
    print(sites)
    print("weights:")
    print(weights)
    eDist = Empirical(sites, weights)
    r = pyro.sample("r", eDist)
    idx = eDist.get_site_idx(r)
    # r = eDist.sample()
    p = torch.exp(eDist.log_prob(r))

    print("r:", r)
    print("idx:", idx)
    print("p:", p)
        
    plot_results(eDist, N)


def plot_results(dist, N=30, count=300, support=None):
    '''
    - ``N`` -- number of states
    '''
    result = []
    for k in range(count):
        r = dist.sample()
        # for test log_prob:
        p = torch.exp(dist.log_prob(r))
        result.append(r)
    
    print("r:", r)
    print("p:", p)

    if support is None:
        support = dist.enumerate_support()
    label = str(support)
    # label = [str(c) for c in support]
    plt.hist(result, N,  # density=True,
             stacked=True,
             rwidth=0.1, label=label)
    plt.show()


def test_train_likelihood(dbg=False):
    # 100 for single p_a value (0.7)
    observed = model1([], 1)  # 700
    print("observed:")
    print(observed)
    # for data in observed:
    data = observed
    # data = [0., 1., 0.]
    weights = {"p_a": torch.ones(10)*0.5}
    res4 = []
    res7 = []
    for i in range(1):  # 300
        # print("data[i]:")
        # print(data[i])
        weights, mt = sequential_discrete_marginal2(model_cond4,
                                                    torch.tensor([data[i]]), weights,
                                                    dbg=dbg)
        res4.append(torch.functional.F.normalize(weights["p_a"], p=1, dim=0)[4])
        # res4.append(weights['p_a'][4])
        res7.append(torch.functional.F.normalize(weights["p_a"], p=1, dim=0)[7])
        # res7.append(weights['p_a'][7])
        # print("weights:")
        # print(weights)
    w = torch.functional.F.normalize(weights["p_a"], p=1, dim=0)
    print("normalize(weights['p_a']):")
    print(w)
    '''
    plt.plot(res4)
    plt.show()
    plt.plot(res7)
    plt.show()
    plt.plot(w)
    plt.show()
    '''
    print("mean:")
    print(mt.nodes["p_a"]["fn"].mean())
    


def test_train_mh(dbg=False):
    observed = model1([], 3000)
    
    results = {}
    
    trace = None
    '''
    trace = dict((name, None)
                 for name in ["p(a=1)", "p(b=1|a=0)",
                              "p(b=1|a=1)", "p(b=1|a=2)"])
    trace["score"] = 0.001
    '''
    scores = []
    
    # for data in observed:
    data = observed
    for i in range(300):
        trace = sequential_discrete_marginal1(model_cond3,
                                              data, old_trace=trace,
                                              step=0.01, dbg=dbg)
        # if dbg:
        # print("trace:")
        # print(trace)
        scores.append(trace["score"])
        # print("score")
        # print(score)
        for name in ["p(a=1)", "p(b=1|a=0)",
                     "p(b=1|a=1)", "p(b=1|a=2)"]:
            if name not in results:
                results[name] = []
            results[name].append(trace[name].detach().clone())
        # print("results")
        # print(results)
    import matplotlib.pyplot as plt

    plt.plot(scores)
    plt.show()
    fig, axs = plt.subplots(len(results), 1)
    for idx, name in enumerate(results):
        axs.flat[idx].plot(results[name])
        axs.flat[idx].set_title(name)
    plt.show()

    # if dbg:
    # print("\nresults:")
    # print(results)
    return(results)


def test_train(obs, dbg=False):
    optimizer = pyro.optim.Adam({})

    # datas = torch.zeros(7)
    # datas = torch.tensor([0., 0., 0.])
    results = {}
    for data in obs:
        with poutine.trace() as params_capture:
            # this section will only allow previus section (trace)
            # if applay_stack called by param.
            with poutine.block(hide_fn=lambda node: node["type"] != "param"):
                q = sequential_discrete_marginal1(model_cond1, guide, data, dbg=dbg)
        # for q in qs:
        q.backward()
        if dbg:
            print("q:")
            print(q)
        values = [node["name"]
                  for node in params_capture.trace.nodes.values()]
        params = [node["value"].unconstrained()
                  for node in params_capture.trace.nodes.values()]
        grads = [node["value"].unconstrained().grad
                 for node in params_capture.trace.nodes.values()]
        if dbg:
            print("\nvalues:")
            print(values)
            print("\nparams:")
            print(params)
            print("\ngrads:")
            print(grads)

        optimizer(params)
        pyro.infer.util.zero_grads(params)

        for name in ["p(a=1)", "p(b=1|a=0)",
                     "p(b=1|a=1)", "p(b=1|a=2)"]:
            if name not in results:
                results[name] = []
            results[name].append(pyro.param(name))
               
    return(results)


def test1():
    observed = model1([], 600)
    # print(observed)
    # observed = torch.ones(3)
    results = test_train(observed, dbg=True)
    print("done")

    import matplotlib.pyplot as plt

    plt.plot(observed)
    plt.show()

    fig, axs = plt.subplots(len(results), 1)
    for idx, name in enumerate(results):
        axs.flat[idx].plot(results[name])
        axs.flat[idx].set_title(name)
    plt.show()


if __name__ == "__main__":
    
    # test_train_likelihood(False)
    # test_paramdist()
    test_emp()
    # test_train_mh(False)
    # test1()
    # sequential_discrete_marginal(model1, guide, torch.tensor(0.), ["a", "b"])
    # sequential_discrete_marginal(model, guide, torch.tensor(0.), ["a", "b"])
    
