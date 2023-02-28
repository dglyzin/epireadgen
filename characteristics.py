import pyro
import torch
import numpy as np
import matplotlib.pyplot as plt

from pyro.poutine.messenger import Messenger


class Char(Messenger):
    
    def __init__(self, step, trace=None, *args, **kwargs):
        
        self.step = step

        # dict with args for return:
        if not hasattr(self, "results"):
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
            self.update_msg(msg)

    def update(self, name, time, value):
        '''for updating current characteristic at each call of pyro.param'''
        pass

    def update_msg(self, msg):
        '''for updating msg. to plot interactivly for example. '''
        pass

    def _plot(self, charname, title, all=False):
        '''
        - ``all`` - if true the value for each from a space dimention
        will be plotted, else only for a mid value.
        '''
        C = self.results[charname].detach().numpy()
        print("C.shape", C.shape)
        if all:
            # plot for all space dimentions
            plt.plot(C[:, 0, :])
            print("inf in C[:, 0, :]?:")
            print(np.isinf(C[:, 0, :]).any())
            
            plt.plot(C[:, 1, :])
            print("inf in C[:, 1, :]?:")
            print(np.isinf(C[:, 1, :]).any())
            
        else:
            mid = C.shape[-1]//2
            print("C[:, 0, mid]")
            print(C[:, 0, mid])
            print("inf in C[:, 0, mid]?:")
            print(np.isinf(C[:, 0, mid].any()))
            plt.plot(C[:, 0, mid], label="U")
            plt.plot(C[:, 1, mid], label="V")
        plt.legend(loc="upper left")
        plt.title(title)
        plt.show()


class Correlation(Char):
    '''Show how much the diviation from the expectation,
    calclulated from m steps (i.e. uv[t], uv[t+m]),
    connected with each other'''

    def plot(self, all=False):
        self._plot("C", "correlation", all=all)

    def __exit__(self, *args, **kwargs):
        uv = self.cat_results()
        print("uv.shape:", uv.shape)
        if uv.shape[0] % 2 != 0:
            uv = uv[:-1]
        N = uv.shape[0]
        C = torch.zeros((N//2,)+uv.shape[1:])

        # in all slices below u and v; space dim - all will
        # be preserved
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
        

class Lyapunov(Char):
    def __init__(self, dx, dy, *args, **kwargs):

        self.epsilon = min(dx, dy)
        self.results = {}
        self.results["Lambda"] = []
        
        Char.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        # convert list to torch:
        self.results["Lambda"] = torch.cat(list(map(
            lambda x: x.unsqueeze(0), self.results["Lambda"])))

        Messenger.__exit__(self, *args, **kwargs)

    def plot(self, all=False):
        self._plot("Lambda", "Lyapunov $\lambda$", all=all)

    def update(self, name, time, value):
        uv = value
        # print("uv.shape", uv.shape)
        if len(uv[0].shape) == 1:
            uv1, uv0 = uv[:, 1:], uv[:, :-1]
        elif len(uv[0].shape) == 2:
            uv1, uv0 = uv[:, 1:, 1:], uv[:, :-1, :-1]
        elif len(uv[0].shape) == 3:
            uv1, uv0 = uv[:, 1:, 1:, 1:], uv[:, :-1, :-1, :-1]
        duv = uv1-uv0
        print("duv:")
        print(duv)
        # print("self.epsilon", self.epsilon)
        # print("time:", time)
        l = torch.log(torch.abs(duv)/self.epsilon)/time
        # print("l", l)
        self.results["Lambda"].append(l)
    
    def update_msg(self, msg):
        if "chars" in msg:
            msg["chars"]["Lambda"] = self.results["Lambda"][-1]
        else:
            msg["chars"] = {}
            msg["chars"]["Lambda"] = self.results["Lambda"][-1]
