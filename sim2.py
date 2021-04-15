import torch

import pyro
import pyro.distributions as pdist
from pyro import poutine
from torch.distributions import constraints
from pyro.ops.indexing import Vindex


def model(obs):
    # all will be overwritten in guide:
    
    a = pyro.sample("a", pdist.Binomial(2, 0.3))
    d = pdist.Bernoulli(0.5)
    b = pyro.sample("b", d, obs=obs)
    # dout = pdist.Bernoulli()
    # out = pyro.sample("out", dout, obs=obs)
    return b


def model1(obs, n):
    p_a = torch.tensor(0.7)
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
    a = pyro.sample("a", pdist.Binomial(2, p_a))

    # p_b_a_0 = pyro.sample("p(b=1|a=0)", pdist.Uniform(0, 0.9))
    # p_b_a_1 = pyro.sample("p(b=1|a=1)", pdist.Uniform(0, 0.9))
    # p_b_a_2 = pyro.sample("p(b=1|a=2)", pdist.Uniform(0, 0.9))

    p_b = torch.tensor([p_b_a_0, p_b_a_1, p_b_a_2])

    d = pdist.Bernoulli(p_b[a.type(torch.long)])
    
    # log_p will still work correctly in this case because it
    # use d.log_p, and d has probability p_b[a]
    # so p(b=0, a=1) = p(b=0|a=1)*p(a=1) = 0.7*(2*p*(1-p))
    # = 0.7*2*0.5*0.5 = 0.7*0.5 = 0.35
    b = pyro.sample("b", d, obs=obs)
    
    # cdist = pdist.
    return b



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

    uniform = pdist.Uniform(0, 1)

    model_log_weights = []
    guide_log_weights = []
    traces = []

    if old_trace is None:
        p_a = pyro.sample("p(a=1)", pdist.Uniform(0, 0.9))
        p_b_a_0 = pyro.sample("p(b=1|a=0)", pdist.Uniform(0, 0.9))
        p_b_a_1 = pyro.sample("p(b=1|a=1)", pdist.Uniform(0, 0.9))
        p_b_a_2 = pyro.sample("p(b=1|a=2)", pdist.Uniform(0, 0.9))
    else:
        p_a = old_trace["p(a=1)"]
        p_b_a_0 = old_trace["p(b=1|a=0)"]
        p_b_a_1 = old_trace["p(b=1|a=1)"]
        p_b_a_2 = old_trace["p(b=1|a=2)"]

        p_a += pyro.sample("e(a=1)", pdist.Normal(0, step))
        p_b_a_0 += pyro.sample("e(b=1|a=0)", pdist.Normal(0, step))
        p_b_a_1 += pyro.sample("e(b=1|a=1)", pdist.Normal(0, step))
        p_b_a_2 += pyro.sample("e(b=1|a=2)", pdist.Normal(0, step))

        if p_a < 0:
            p_a = torch.tensor(0.0)
        if p_a > 1:
            p_a = torch.tensor(1.0)
        if p_b_a_0 < 0:
            p_b_a_0 = torch.tensor(0.0)
        if p_b_a_0 > 1:
            p_b_a_0 = torch.tensor(1.0)
        if p_b_a_1 < 0:
            p_b_a_1 = torch.tensor(0.0)
        if p_b_a_1 > 1:
            p_b_a_1 = torch.tensor(1.0)
        if p_b_a_2 < 0:
            p_b_a_2 = torch.tensor(0.0)
        if p_b_a_2 > 1:
            p_b_a_2 = torch.tensor(1.0)

    while not q.empty():
        model_trace_messenger = poutine.trace(enum_model)
        # model_trace = model_trace_messenger.get_trace(data)
        model_trace = model_trace_messenger.get_trace(data, p_a, p_b_a_0,
                                                      p_b_a_1, p_b_a_2)
        # model_samples.append(model_trace.nodes["p(a=1)"]["value"])
        model_log_weights.append(model_trace.log_prob_sum())
    if dbg:
        print("model_samples(p(a=1)):")
        # print(model_samples)
    # p(A) = sum(P(A, b))
    p = torch.tensor([torch.exp(weight) for weight in model_log_weights]).sum()
    current_score = p
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
        
        old_trace["score"] = 0.01
        
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


def test_train_mh(dbg=False):
    observed = model1([], 3700)
    
    results = {}
    
    trace = None
    '''
    trace = dict((name, None)
                 for name in ["p(a=1)", "p(b=1|a=0)",
                              "p(b=1|a=1)", "p(b=1|a=2)"])
    trace["score"] = 0.001
    '''
    scores = []
    
    for data in observed:
        trace =  sequential_discrete_marginal1(model_cond3,
                                               data, old_trace=trace,  # old_trace=trace,
                                               step=0.01, dbg=dbg)
        if dbg:
            print("trace:")
            print(trace)
        scores.append(trace["score"])
        # print("score")
        # print(score)
        for name in ["p(a=1)", "p(b=1|a=0)",
                     "p(b=1|a=1)", "p(b=1|a=2)"]:
            if name not in results:
                results[name] = []
            results[name].append(trace[name])
    
    import matplotlib.pyplot as plt

    plt.plot(scores)
    plt.show()
    fig, axs = plt.subplots(len(results), 1)
    for idx, name in enumerate(results):
        axs.flat[idx].plot(results[name])
        axs.flat[idx].set_title(name)
    plt.show()

    if dbg:
        print("\nresults:")
        print(results)
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
    
    test_train_mh()
    # test1()
    # sequential_discrete_marginal(model1, guide, torch.tensor(0.), ["a", "b"])
    # sequential_discrete_marginal(model, guide, torch.tensor(0.), ["a", "b"])
    
