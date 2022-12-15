import pyro
import torch
import matplotlib.pyplot as plt

from pyro.poutine.messenger import Messenger


class Char(Messenger):
    
    def __init__(self, trace, step, *args, **kwargs):
        
        self.step = step

        # dict with args for return:
        self.results = {}

        # set up init values
        if trace is not None:
            for name_idx in trace.nodes:
                name, idx = name_idx.split("_")
                self.update(name, int(idx), trace.nodes[name_idx]['value'])
            self.init_count = len(trace.nodes)
        self.trace = trace
        Messenger.__init__(self, *args, **kwargs)
        
    def _pyro_post_param(self, msg):
        name_idx = msg['name']
        value = msg['value']
        name, idx = name_idx.split("_")
        idx = int(idx)
        if idx % self.step == 0:
            self.update(name, idx, value)

    def update(self, name, time, value):
        pass


class Correlation(Char):
    '''Show how much the diviation from the expectation,
    calclulated from m steps (i.e. uv[t], uv[t+m]),
    connected with each other'''

    def plot(self, all=False):
        C = self.results["C"].detach().numpy()
        if all:
            plt.plot(C[:, 0, :])
            plt.plot(C[:, 1, :])
        else:
            mid = C.shape[-1]//2
            plt.plot(C[:, 0, mid], label="U")
            plt.plot(C[:, 1, mid], label="V")
        plt.legend(loc="upper left")
        plt.show()

    def __exit__(self, *args, **kwargs):
        uv = self.cat_results()
        print("uv.shape:", uv.shape)
        if uv.shape[0] % 2 != 0:
            uv = uv[:-1]
        N = uv.shape[0]
        C = torch.zeros((N//2,)+uv.shape[1:])

        E = 1/N*(uv.sum(0))
        muv = uv - E
        C[0] = 1/N*(muv[:]*muv[:]).sum(0)
        for m in range(1, N//2):
            C[m] = 1/N*(muv[m:]*muv[:-m]).sum(0)

        self.results["C"] = C

        Messenger.__exit__(self, *args, **kwargs)

    def cat_results(self):
        # print("trace names:")
        # print([name for name in self.trace.nodes])

        return torch.cat(
            [self.trace.nodes[name]['value'].unsqueeze(0)
             for name in self.trace.nodes], 0)
        

'''
class Lyapunov(Char):
    def __init__(*args, **kwargs):
        Char.__init__(self, *args, **kwargs)

        self.
'''
