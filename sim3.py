import numpy as np
import pyro
import pyro.distributions as pdist
import torch

import pandas as pd
import matplotlib.pyplot as plt

# from progresses.prgress_cmd import ProgressCmd
import progresses.progress_cmd as progress_cmd
ProgressCmd = progress_cmd.ProgressCmd

def model():
    p = pyro.sample("p", pdist.Uniform(0, 1))
    a = pyro.sample("a", pdist.Categorical(torch.tensor([p, 1-p])))
    return(a)


def fcond(trace):
    print(trace.nodes)
    
    return(False)


def model1():
    '''
    Model:
    {a| a=Uniform(0, 1)}
      ->{Bag| [p(Bug=0)=1-a, p(Bug=1)=a]}
       ->{Color| [p(Color|"Bag": 0)=[0.1, 0.9],
                  p(Color|"Bag": 1) = [0.5, 0.5]]}
    Ask what p(X|cond) will be for X is some of [a, Bag, Color]
    '''
    p = pyro.sample("p", pdist.Uniform(0, 1))
    # p(Bag|p) (note probability reverse in Categorical!):
    bag = pyro.sample("Bag", pdist.Categorical(torch.tensor([1-p, p])))
    pColor = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
    color = pyro.sample("Color", pdist.Categorical(pColor[bag.type(torch.long)]))
    return(color)


def m1_fcond(trace):
    return(trace.nodes['p']['value'] >= 0.7)


def model2():
    '''
    Model:
    {a| a=Uniform(0, 1)}
     ->{b| b=Uniform(a, 1)}
      ->{Bag| [p(Bug=0)=1-b, p(Bug=1)=b]}
       ->{Color| [p(Color|"Bag": 0)=[0.1, 0.9],
                  p(Color|"Bag": 1) = [0.5, 0.5]]}
    Ask what p(X|cond) will be for X is some of [a, b, Bag, Color]

    f = lambda x: sympy.integrate((x-a)/(1-a),(a, 0., x))
    x = np.linspace(0,1, 100)
    y = list(map(f, x))
    plt.plot(x, y)

    '''
    a = pyro.sample("a", pdist.Uniform(0, 1))
    b = pyro.sample("b", pdist.Uniform(a, 1))
    bag = pyro.sample("Bag", pdist.Categorical(torch.tensor([b, 1-b])))
    pColor = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
    color = pyro.sample("Color", pdist.Categorical(pColor[bag.type(torch.long)]))
    return(color)


def m2_fcond(trace):
    return(trace.nodes['a']['value'] >= 0.0)


class Cond(pyro.poutine.trace_messenger.TraceMessenger):

    '''Check conditions and add cond = True if succ'''

    def __init__(self, cond, *args, **kwargs):
        self.cond = cond
        self.tcond = []
        pyro.poutine.trace_messenger.TraceMessenger.__init__(self, *args, **kwargs)
        
    def __exit__(self, *args, **kwargs):
        
        self.trace.add_node("cond", name="cond",
                            value=self.cond(self.trace), f=self.cond)
        return pyro.poutine.trace_messenger.TraceMessenger.__exit__(self, *args, **kwargs)


def rejection_sampling(model, fcondition, N=3, cprogress=ProgressCmd):
    progress = cprogress(N)
    samples = []
    for step in range(N):
        with Cond(fcondition) as cstorage:
            Cond(fcondition)(model).get_trace()
        if cstorage.trace.nodes['cond']['value']:
            samples.append(cstorage.trace.copy())
        progress.succ(step)
    progress.print_end()
    return(samples)


def make_dataFrame(samples, cprogress=ProgressCmd):
    first_sample, rest_samples = samples[0], samples[1:]
    df = pd.DataFrame([[np.array(first_sample.nodes[key]['value'])
                        for key in first_sample.nodes]],
                      columns=list(first_sample.nodes))
    progress = cprogress(len(rest_samples), prefix="create df:")

    for step, sample in enumerate(rest_samples):
        df = df.append(dict([(key, np.array(sample.nodes[key]['value']))
                             for key in sample.nodes]),
                       ignore_index=True)
        progress.succ(step)
    progress.print_end()
    # print(df)
    return(df)


def plot_results(df):
    for var in df.columns:
        print("\nvar: ", var)
        var_data = df[var].to_numpy().astype(np.float)
        print("probs of var (bin=2):")
        var_data = torch.histc(torch.tensor(var_data), 2) 
        print(var_data/var_data.sum())

        plt.hist(df[var].to_numpy(), 30,  # density=True,
                 stacked=True
                 # , rwidth=0.1, label=label
             )
        plt.show()
    

def test_m0():
    print(Cond(fcond)(model).get_trace().nodes["cond"])
    

def test_m1_0(N=3):
    d = Cond(m1_fcond)(model1)
    for i in range(N):
        print(d.get_trace().nodes["cond"])


def test_m1_1(N=3):
    for i in range(N):
        with Cond(m1_fcond) as cstorage:
            # trace(model1).get_trace()
                print(Cond(m1_fcond)(model1).get_trace().nodes["cond"])
        print(cstorage.trace.nodes["cond"])


def test_rejection_sampler(N=3):
    samples = rejection_sampling(model1, m1_fcond, N)
    print([sample.nodes['cond']['value'] for sample in samples])
    print([sample.nodes['Color']['value'] for sample in samples])


def test_rejection_sampler1(model=model1, fcondition=m1_fcond, N=3):

    '''with plot and DataFrame'''

    samples = rejection_sampling(model, fcondition, N)
    df = make_dataFrame(samples)
    plot_results(df)
    return(df)


if __name__ == "__main__":
    test_rejection_sampler1(model=model2, fcondition=m2_fcond, N=700)
    # test_rejection_sampler1(model=model1, fcondition=m1_fcond, N=700)
    # test_m1_1()
