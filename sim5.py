# to find N (to create `N.tmp` file):
# $ python3 -m sim5 -init
# for a main:
# $ python3 -m sim5

import pyro
import torch
from torch.nn import Conv1d, Conv2d, Conv3d
import numpy as np
from collections import deque
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import _PYRO_PARAM_STORE
from pyro.params.param_store import ParamStoreDict
import matplotlib.pyplot as plt
import sys
from functools import wraps
import subprocess
import os

import progresses.progress_cmd as progress_cmd
ProgressCmd = progress_cmd.ProgressCmd

import characteristics as chars

path_video = "results"


class ResultCollectorMessenger(
        Messenger, ParamStoreDict):
    def __init__(self, trace, step, *args, video=False, cuda=False, **kwargs):
        '''
        - ``trace`` -- trace to take results from
        - ``step`` -- step throu which the results to be collected
        - ``video`` -- if true images will be created during computations in
        the `path_video` folder and no results will be stored in memory, so
        plot results["z"][-1] will return only init data
        if false then results will be collected in `self.results` and
        `self.plot_state` and `self.plot_states` could return
        the data from any recorded time step. 
        '''
        self.step = step
        self.results = {}
        self.video = video
        if video:
            self.clear_previous()
        self.cuda = cuda

        ParamStoreDict.__init__(self)

        # set up init values
        if trace is not None:
            for name in trace.nodes:
                self.add(name, trace.nodes[name]['value'])
            self.init_count = len(trace.nodes)
        Messenger.__init__(self, *args, **kwargs)

    def _pyro_post_param(self, msg):
        if self.video:
            self.save_img(msg["name"], msg["value"])
        else:
            self.add(msg["name"], msg["value"])

    def add(self, name_idx, value):
        name, idx = name_idx.split("_")
        idx = int(idx)
        if idx % self.step == 0:
            self.get_param(name_idx, value)
            if name in self.results:
                self.results[name].append(value)
            else:
                self.results[name] = [value]
    
    def __exit__(self, *args, **kwargs):
        self.cat_results()
        if self.video:
            self.save_video()

        return Messenger.__exit__(self, *args, **kwargs)

    def cat_results(self):
        res = {}
        for name in self.results:
            res[name] = torch.cat([
                t.unsqueeze(0) for t in self.results[name][:]], 0)
        self.results = res

    def load(self, filename):
        ParamStoreDict.load(self, filename)
        self._copy_params_to_results()
        self.cat_results()

    def _copy_params_to_results(self):
        results = {}
        series = [name_idx for name_idx in self._params]
        series.sort(key=lambda x: int(x.split("_")[-1]))
        # print("series")
        # print(series)
        for name_idx in series:
            name, idx = name_idx.split("_")
            idx = int(idx)
            value = self._params[name_idx]
            if name in results:
                results[name].append(value)
            else:
                results[name] = [value]
        self.results = results

    def show_results_size(self):
        print("\nlen results.keys:")
        print(len(self.results.keys()))
        for key in self.results:
            print("key size:")
            print(self.results[key].shape)
        print("\nlen _param.keys:")
        print(len(self._params.keys()))
        
    def show_results(self):
        print("\nresults.keys:")
        print(self.results.keys())
        print("\n_param.keys:")
        print(self._params.keys())
        
    def plot_state(self, t=-1):
        print("U(x, t= %d)" % t)
        print("V(x, t= %d)" % t)
        for name in self.results:
            yy = self.results[name]
            self._plot_state(yy[t], t)            
            plt.show()

    def _plot_state(self, uv, t):
        if self.cuda:
            # copy to cpu:
            uv = uv.cpu()

        uv = uv.detach().numpy()
        # print(uv[0].shape)
        # print(uv[1].shape)
        if len(uv[0].shape) == 1:
            plt.plot(uv[0], label="U(x, t= %d)" % t)
            plt.plot(uv[1], label="V(x, t= %d)" % t)
            plt.plot(np.sqrt(uv[0]**2+uv[1]**2),
                     label="$\sqrt{U^{2}+V^{2}}$")
            res = plt.legend(loc="upper left")
        elif len(uv[0].shape) == 2:
            plt.imshow(np.sqrt(uv[0]**2+uv[1]**2),
                       label="$\sqrt{U^{2}+V^{2}}$")
            res = plt.legend(loc="upper left")
        return res

    def plot_states(self):
        '''
        plot each variable and print its `l.max` value
        '''
        print("U(mid, t)")
        print("V(mid, t)")
        for name in self.results:
            Utmp = self.results[name][:, 0]
            Vtmp = self.results[name][:, 1]
            
            if self.cuda:
                # copy to cpu:
                Utmp = Utmp.cpu()
                Vtmp = Vtmp.cpu()

            U = Utmp.detach().numpy()
            V = Vtmp.detach().numpy()

            if len(U.shape) > 1:
                print("U.shape:", U.shape)
                idx_x = U.shape[-1]//2
                if len(U.shape[1:]) == 1:
                    U = U[:, idx_x]
                    V = V[:, idx_x]
                elif(len(U.shape[1:])) == 2:
                    U = U[:, idx_x, idx_x]
                    V = V[:, idx_x, idx_x]

            l = np.sqrt(U**2+V**2)
            print("min l", l.max())

            plt.plot(U, label="U(x=%s, t)" % idx_x)
            plt.plot(V, label="V(x=%s, t)" % idx_x)
            plt.plot(l, label="$U^{2}+V^{2}$")
        plt.legend(loc="upper left")
        plt.show()
    
    def save_img(self, name_idx, value):
        name, idx = name_idx.split("_")
        idx = int(idx)
        res = self._plot_state(value, idx)
        # print("figure")
        # print(res.figure)
        res.figure.savefig(os.path.join(path_video, name_idx+".png"))
        # clear scene
        res.figure.clf()

        # will be used in save_video: 
        self.var_name = name

    def save_video(self):
        assert hasattr(self, "init_count")
        command = (
            "avconv -r 5 -loglevel panic -start_number "
            + str(self.init_count) + " -i "
            + os.path.join(path_video,
                           self.get_img_filename_avconv())
            + " -pix_fmt yuv420p -b:v 1000k -c:v libx264 "
            + os.path.join(path_video, "video.mp4"))
        print("avconv command:")
        print(command)
        # PIPE = subprocess.PIPE
        # proc = subprocess.Popen(command, shell=True, stdin=PIPE,
        #                         stdout=PIPE, stderr=subprocess.STDOUT)
        # proc.wait()
        subprocess.call(command, shell=True)
        self.clear_previous(img_only=True)

    def get_img_filename_avconv(self):
        '''For createVideoFile func'''
        assert hasattr(self, "var_name")
        return(self.var_name + "_" + "%d.png")

    def clear_previous(self, img_only=False):
        
        for filename in sorted(os.listdir(path_video)):
            if filename.endswith('png') or (filename.endswith('mp4')
                                            and not img_only):
                os.remove(os.path.join(path_video, filename))
                        

class StackCleanerMessenger(Messenger):
    '''
    Clear stack (hopefuly in device side too) to preserve memory.

    Since `_pyro_post_param` method will change
    `msg` value to `Str`, it must be last in `PYRO_STACK`
    i.e. first wrapped context (last in `with` seq)
    '''
    def __init__(self, trace, TN_init, stack_size, *args, **kwargs):
        '''
        - ``trace`` -- to work with
        - ``TN_init`` -- count of init values (to be preserved)
        - ``stack_size`` -- count of values to be preserved.
        so whole stack size will be `TN_init+stack_size`.
        '''
        self.trace = trace
        # self.TN_init = TN_init
        self.stack = deque([name for name in trace.nodes])
        self.stack_size = stack_size + TN_init
        
        Messenger.__init__(self, *args, **kwargs)

    def _pyro_post_param(self, msg):
        '''Since this function will change msg value to Str,
        it must be last in PYRO_STACK'''

        self.stack.append(msg["name"])
        # print("msg['name']:", msg["name"])
        
        self.clear_stack(msg)

    def clear_stack(self, msg):
        while len(self.stack) > self.stack_size:
            name = self.stack.popleft()
            # print("stack:", self.stack)
            # print("name:", name)
            value = self.trace.nodes[name].pop("value")
            del value
            if 'args' in msg:
                value = msg.pop('args')
                del value
            if 'value' in msg:
                value = msg.pop('value')
                del value
                # param return in model
                # not used since all previous value
                # taken from the trace
                msg["value"] = "Null"
            # del msg
            value = self.trace.nodes.pop(name)
            del value
            params = _PYRO_PARAM_STORE._params
            if name in params:
                # value = params.pop(name)
                # del value
                # this trigger __delitem__ of pyro.ParamStoreDict
                # which in turn will remove from both constrained
                # and unconstrained values
                # REF: pyro/params/param_store.py
                del _PYRO_PARAM_STORE[name]
        # self.show_series()
        # print("msg")
        # print(msg)

    def show_stack(self):
        print("\nstack:")
        print(self.stack)

    def show_stack_size(self):
        print("\nstack_size:")
        print(len(self.stack))


class SolverContextMessenger(TraceMessenger):
    def __init__(self, g_init, get_init_times,
                 *args, **kwargs):

        self.g_init = g_init
        self.get_init_times = get_init_times
        TraceMessenger.__init__(self, *args, **kwargs)

    # fill init value as default minimal content:
    def __enter__(self,   *args, **kwargs):
        # initialize trace:
        init = TraceMessenger.__enter__(self, *args, **kwargs)
        
        # fill trace with init values: 
        tt0 = self.get_init_times()
        # tt0 = torch.linspace(self.t0, 0, self.TN_init)
        # for t in torch.arange(self.TN_init, dtype=torch.int):
        for idx, t in enumerate(tt0):
            name = "z_%d" % (idx)
            self.trace.add_node(name, value=self.g_init(idx, t), type="param")
        return init

    def _pyro_param(self, msg):
        '''this function will be called in `Messenger._process_message`
        automaticaly with param '''
        pass
        # print("hello from _pyro_param")
    
    def get_series(self):
        return [name for name in self.trace.nodes]
 
    def show_series(self):
        print("\nseries")
        print(self.get_series())
    
    def show_series_size(self):
        print("\nseries size")
        print(len(self.get_series()))


class Model:
    def __init__(self, t0=-1, t1=0, dt=0.01, cprogress=ProgressCmd, dbg=True):
        
        self.dbg = dbg

        # init_t = N.shape[0]
        # init_t = TN_init
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        
        self.tt0 = self.get_neg_times()
        # self.tt0 = torch.arange(t0, 0, dt)
        
        self.count_neg_t = len(self.tt0)
        if self.dbg:
            print("count_neg_t: ", self.count_neg_t)

        self.tt1 = self.get_pos_times()
        self.count_pos_t = len(self.tt1)
        if self.dbg:
            print("count_pos_t: ", self.count_pos_t)

        # self.tt0 = torch.linspace(-1, 0, init_t)
        # self.tt1 = torch.linspace(0, t1, TN)[1:]
        # self.tt = torch.cat([self.tt0, self.tt1], 0)

        self.progress_decorator(cprogress)

        # for test:
        self.omega = 1
        # self.omega = torch.ones(4)
        # self.omega = torch.ones((3, 3))
        # omega = np.ones((2,2))

    def progress_decorator(self, cprogress):
        '''To set up the progress. To handle `None` case
        (i.e. do not put `if self.progress is not None`
         each time its methed used).'''
        
        assert hasattr(self, "count_pos_t")

        _progress = None
        cprogress_succ = None
        cprogress_end = None

        global_self = self

        if cprogress is not None:
            assert hasattr(cprogress, "succ")
            assert hasattr(cprogress, "print_end")
            
            _progress = cprogress(self.count_pos_t)
            cprogress_succ = cprogress.succ 
            cprogress_end = cprogress.print_end
            
        class Progress():
            
            def __init__(self):
                if _progress is None:
                    if global_self.dbg:
                        print("warning: progress is not set!")
            
            @wraps(cprogress_succ)
            def succ(self, step):
                if _progress is not None:
                    _progress.succ(step)
                    # or cprogress_end(_progress, step)

            @wraps(cprogress_end)
            def end(self):
                if _progress is not None:
                    _progress.print_end()
                    # or cprogress_end(_progress)
        self.progress = Progress()
        
    def get_pos_times(self):
        return torch.arange(0, self.t1, self.dt)

    def get_neg_times(self):
        return torch.arange(self.t0, 0, self.dt)
        # return torch.arange(self.t0-self.dt, 0, self.dt)

    def g_init(self, idx, t):
        omega = self.omega
        t = torch.unsqueeze(t*omega, 0)
        return torch.cat([t, t], 0)

    def get_t(self, idx, delay=None):
        idx_res = idx+self.count_neg_t-1
        if delay is not None:
            # the shift of `-dt` is due to solving scheme:
            # U(t+1) = f(U(t),U(t-d))
            # so U(t) also shuld be known and hence distance
            # between t and t-d must be len(tt0)+1
            idx_res += -(int(delay/self.dt)-1)
        return idx_res
    # def get_t(self, t):
    #     idx, _ = next(filter(lambda it: it[1] >= t, enumerate(self.tt)))
    #     return idx

    def _model1(self, trace, idx_t, idx_td):
        omega = self.omega
        # print("t:", t)
        a = torch.tensor([1.5, 0.])
        b = torch.tensor([1.5, 0.])
        # idx_t = self.get_t(t)
        # idx_td = self.get_t(t-delay)
        # import pdb; pdb.set_trace()
        U, V = trace.nodes["z_%d" % idx_t]["value"]
        Ud, Vd = trace.nodes["z_%d" % (idx_td)]["value"]
        l2 = Ud**2+Vd**2
        U1 = ((a[0]-l2*b[0])*U-(a[1]-l2*b[1])*V)
        V1 = ((a[0]-l2*b[0])*V+(a[1]-l2*b[1])*U)

        return torch.cat([U1.unsqueeze(0),
                          V1.unsqueeze(0)], 0)
        # return torch.cat([(0.5*U*(1-Vd)*omega).unsqueeze(0),
        #                   (-0.5*V*(1-Ud)*omega).unsqueeze(0)], 0)

    def __call__(self, trace):
        # z_0 = pyro.param("z_0", self.g1(0))
        dt = self.dt
        # dts = np.diff(self.tt1)
        
        for idx, t in enumerate(self.tt1[:-1]):
            # print("t=", t)
            # print("idx=", idx)
            # dt = dts[idx]
            # print("dt=", dt)
            idx_t = self.get_t(idx+1)
            idx_t1 = self.get_t(idx)
            idx_dt = self.get_t(idx, 1)
            # idx_t = self.get_t(t)
            # print("get(t):", idx_t)
            z_prev = trace.nodes["z_%d" % (idx_t1)]['value']
            f = dt*self._model1(trace, idx_t1, idx_dt)
            # this line will call get_param and hence apply_stack
            # (get_param also used for set one)
            pyro.param("z_%d" % idx_t, f+z_prev)
            
            self.progress.succ(idx)
        
        self.progress.end()

        # z_prev = trace_msg.trace.nodes["z_%d" % (t-1)]["value"]
        # z = pyro.param("z_%d" % t, z_prev+1)


class CModel(Model):

    def __init__(self, N, t0=-1, t1=1, dd=(0.01, 0.01), ll=(3, 3), **kwargs):
        '''
        
        '''

        self.dd = dd
        self.ll = ll
        self.dim = len(dd)

        self.N = N
        if self.dim == 1:
            self.make_coord_space(ll[0], 1, 1)
        elif self.dim == 2:
            self.make_coord_space(ll[0], ll[1], 1)
        else:
            self.make_coord_space(ll[0], ll[1], ll[2])

        dt = torch.abs(torch.tensor(t0-0))/(N.shape[0])
        Model.__init__(self, t0, t1, dt, **kwargs)

        if self.dbg:
            print("tt0:")
            print(self.tt0[0], self.tt0[-1])
            print(len(self.tt0))
            print("tt1:")
            print(self.tt1[0], self.tt1[-1])
            print(len(self.tt1))

    # TODO: make decorator with classmethod
    def g_init(self, idx, t):
        # import pdb; pdb.set_trace()
        return self.N[idx]
    
    def get_pos_times(self):
        tt1 = Model.get_pos_times(self)
        return tt1

    def get_neg_times(self):
        # the shift of `-dt` is due to solving scheme:
        # U(t+1) = f(U(t),U(t-d))
        # so U(t) also should be known and hence distance
        # between t and t-d must be len(tt0)+1

        # return torch.linspace(self.t0-self.dt, 0, (self.N.shape[0]-1))
        tt0 = torch.arange(self.t0, 0, self.dt)
        return tt0 
        # tt0 = torch.arange(self.t0-self.dt, 0, self.dt)
        # tt0 = torch.arange(self.t0-self.dt, 0, 1/(self.N.shape[0]))

    def get_params(self):
        a = torch.tensor([1.5, 0.])
        b = torch.tensor([1.5, 0.])
        d = torch.tensor([0.0001, 0.0001])
        return (a, b, d)
 
    def _model1(self, trace, idx_t, idx_td):
        omega = self.omega
        # print("t:", t)
        a, b, d = self.get_params()

        # idx_t = self.get_t(t)
        # idx_td = self.get_t(t-delay)
        # import pdb; pdb.set_trace()
        U, V = trace.nodes["z_%d" % idx_t]["value"]
        Ud, Vd = trace.nodes["z_%d" % (idx_td)]["value"]
        l2 = Ud**2+Vd**2
        
        delta_U = self.calc_Delta(U, self.dd)
        delta_V = self.calc_Delta(V, self.dd)
        cDelta_U = d[0]*delta_U-d[1]*delta_V
        cDelta_V = d[1]*delta_U+d[0]*delta_V

        U1 = cDelta_U+(a[0]-l2*b[0])*U-(a[1]-l2*b[1])*V
        V1 = cDelta_V+(a[0]-l2*b[0])*V+(a[1]-l2*b[1])*U

        U1 = self.set_bound_cond(U1, self.dim)
        V1 = self.set_bound_cond(V1, self.dim)
        return torch.cat([U1.unsqueeze(0),
                          V1.unsqueeze(0)], 0)
    
    def make_coord_space(self, row_size=3, col_size=3, z_size=1):
        # must be called only after self.cat_results
        # extend N(t) to N(t,x) where x of  
        # space_size = 3
        # space_dim = 3
        # make_coord_space(yy1, 4,1)
        # for name in self.results:
        N = self.N
        N = torch.cat([
            N.unsqueeze(2)
            for k in range(row_size*col_size)], 2)
        if col_size != 1:
            if z_size != 1:
                space_shape = (row_size, col_size, z_size)
            else:
                space_shape = (row_size, col_size)
        else:
            space_shape = (row_size,)
        self.N = N.reshape(N.shape[:-1]+space_shape)

    def calc_Delta(self, U, dd=(0.01, 0.01)):

        dx = dd[0]
        if len(dd) == 1:
            # count of branches, layers, coords:
            k = Conv1d(1, 1, 3, padding=1, padding_mode="zeros", bias=False)
            wx = torch.tensor([[[1, -2, 1]]])
            w = wx/dx**2

        elif len(dd) == 2:
            k = Conv2d(1, 1, (3, 3), padding=1, padding_mode="zeros", bias=False)
            dy = dd[1]
        
            wx = torch.tensor([[[[0, 0, 0], [1, -2, 1], [0, 0, 0]]]])
            wy = torch.tensor([[[[0, 1, 0], [0, -2, 0], [0, 1, 0]]]])
            w = wx/dx**2+wy/dy**2
        elif len(dd) == 3:
            k = Conv3d(1, 1, (3, 3, 3), padding=1, padding_mode="zeros", bias=False)
            dy = dd[1]
            dz = dd[2]

            wx = torch.zeros((3, 3, 3))
            wx[1, 1, 1] = -2
            wx[1, 1, 0] = 1
            wx[1, 1, 2] = 1
            
            wy = torch.zeros((3, 3, 3))
            wy[1, 1, 1] = -2
            wy[1, 0, 1] = 1
            wy[1, 2, 1] = 1
            
            wz = torch.zeros((3, 3, 3))
            wz[1, 1, 1] = -2
            wz[0, 1, 1] = 1
            wz[2, 1, 1] = 1
            w = wx/dx**2+wy/dy**2+wy/dz**2

        # w = torch.tensor([[[[0, 1, 0], [1, -2, 1], [0, 1, 0]]]])
        k.weight = torch.nn.Parameter(w.type(torch.float))
        laplace = k(U.unsqueeze(0).unsqueeze(0)).detach().clone()[0][0]

        # do not give him clear borders:
        laplace = self.fix_border(laplace, len(dd))
        return(laplace)

    def fix_border(self, U, dim):
        return(self.set_bound_cond(U, dim))

    def set_bound_cond(self, U, dim):
        if dim == 1:
            U[..., -1] = U[..., 0]
        elif dim == 2:
            U[..., -1, :] = U[..., 0, :]
            U[..., :, -1] = U[..., :, 0]
        elif dim == 3:
            U[..., -1, :, :] = U[..., 0, :, :]
            U[..., :, -1, :] = U[..., :, 0, :]
            U[..., :, :, -1] = U[..., :, :, 0]
        return(U)

    
def find_N(t1=10, dt=0.01, filename="N.tmp"):
    t0 = -1.

    # step throu which the results to be collected:
    res_to_collect_step = 1  # 1

    stack_size = 2
    model = Model(t0, t1, dt)
    # model = Model(dt, t1, (0.01, ), (7,))

    with SolverContextMessenger(
            model.g_init, model.get_neg_times) as tr0:
        with ResultCollectorMessenger(tr0.trace, res_to_collect_step) as res:
            count_neg_t = model.count_neg_t+3
            print("count_neg_t:", count_neg_t)
            with StackCleanerMessenger(
                    tr0.trace, count_neg_t, stack_size) as stack:
                print("\nstack_size:")
                print(stack.stack_size)
                # print(model.tt)
                model(tr0.trace)
                tr0.show_series()
                tr0.show_series_size()
                stack.show_stack()
                stack.show_stack_size()
    print("\nlen _PYRO_PARAM_STORE._params.keys:")
    print(len(_PYRO_PARAM_STORE._params.keys()))
    print(_PYRO_PARAM_STORE._params.keys())
    
    # print("\nchange shape:")
    # this will be ignored by save/load
    # res.make_coord_space(3, 3)
    
    res.show_results_size()
    
    # print("\n res.results[z]:")
    # print(res.results['z'])
   
    print("saving/loading")
    res.save(filename)
    res.load(filename)
    # res.make_coord_space(3, 3)
    # print("\n res.results[z]:")
    # print(res.results['z'])
    
    res.show_results_size()
    
    print("\n len res.results[z]:")
    print(len(res.results['z']))
    res.plot_states()
    # print(res.results['z'][:, 0])

    # print([name for name in tr0.trace.nodes])
    # print([name for name in tr0.trace.nodes])
    # print("\n for checking stack")
    # for name in tr0.trace.nodes:
    #     print(name)
    #     print(tr0.trace.nodes[name]["value"])
    

def test_solver(t0, t1, dt, dd, ll, in_filename, out_filename,
                video=True, cuda=False,
                stack_size=2, res_to_collect_step=1):
    '''
    - ``dt`` -- will be ignored for main
    (since N values used as well as t1)
    
    - ``dd`` -- tuple of space steps
    like (0.01,) or (0.01, 0.01,), ...

    - ``ll`` -- tuple of space sizes
    like (10,) or (10, 10,), ...
    - ``video`` -- setting to true will force `ResultCollectorMessenger`
    do not collect result so `plot_state(t)` will not work for all t.
    '''
    
    if cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("torch.cuda.memory_allocated():")
        print(torch.cuda.memory_allocated())
    print("device used:")
    print(torch.ones((3, 3)).device)
        
    results = ResultCollectorMessenger(None, None)
    results.load(in_filename)
    N = results.results["z"]
    if cuda:
        N = N.to(torch.ones((3, 3)).device)
    for i in range(1, 3)[::-1]:
        print("N[-%d]" % i)
        print(N[-i])
    
    model = CModel(N, t0, t1,
                   dd=dd, ll=ll)
    # dd=(0.01, 0.01), ll=(3, 3))
    
    # res_to_collect_step = 1  # 1
    # stack_size = 2
    pyro.clear_param_store()
    print("starting")
    with SolverContextMessenger(
            model.g_init, model.get_neg_times) as tr0:
        with ResultCollectorMessenger(tr0.trace, res_to_collect_step,
                                      video=video, cuda=cuda) as res:
            count_neg_t = model.count_neg_t+3
            # count_neg_t = 10
            # print("count_neg_t:", count_neg_t)
            with chars.Correlation(trace=tr0.trace, step=1) as corr:
                with chars.Lyapunov(
                        dd[0], dd[0], trace=tr0.trace, step=1) as lpv:
                    with StackCleanerMessenger(
                            tr0.trace, count_neg_t, stack_size) as stack:
                        print("\nstack_size:")
                        print(stack.stack_size)
                        # print(model.tt)
                        model(tr0.trace)
                        # tr0.show_series()
                        tr0.show_series_size()
                        # stack.show_stack()
                        stack.show_stack_size()
    print("for Lapunov:")
    print("lpv.results:")
    print('lpv.results["Lambda"][-1].shape')
    print(lpv.results["Lambda"][-1].shape)
    print('lpv.results["Lambda"][-1]')
    print(lpv.results["Lambda"][-1])
    print('lpv.results["Lambda"][-1][lpv.results["Lambda"][-1]==-np.inf]')
    print(lpv.results["Lambda"][-1][lpv.results["Lambda"][-1] == -np.inf])
    # print(lpv.results["Lambda"])
    lpv.plot(all=True)
    print("len(lpv.results)")
    print(len(lpv.results["Lambda"]))

    print("for correlation:")
    corr.plot(all=False)
    # print("len corr.result:")
    # print(corr.results['C'])
    print("corr.init_count:")
    print(corr.init_count)
    print("corr.result.shape: ", corr.results['C'].shape)
    # print("\nlen _PYRO_PARAM_STORE._params.keys:")
    # print(len(_PYRO_PARAM_STORE._params.keys()))
    # print(_PYRO_PARAM_STORE._params.keys())
    res.show_results_size()
    # res.show_results()
    print("res.results[z][-1]:")
    print(res.results["z"][-1])
    res.plot_state()
    res.plot_states()
    if cuda:
        print("torch.cuda.memory_allocated():")
        print(torch.cuda.memory_allocated())
    
    return res


if __name__ == "__main__":
    # python3 -m sim5 -init
    # python3 -m sim5
    # if args.cuda:
    #     torch.set_default_tensor_type("torch.cuda.FloatTensor")

    name = '-init'
    if name in sys.argv:
        print("finding N:")
        # find_N(10, 0.01)
        find_N(4, 0.01, filename="N.tmp")
    else:
        print("solving main problem:")
        # test_solver(-1, 10, 0.01, (0.01, 0.01), (3, 3),
        # test_solver(-1, 100, 0.01, (0.01,), (10,),
        test_solver(-1, 1, 0.01, (0.01,), (10,),
                    in_filename="N.tmp",
                    out_filename="res.tmp", video=False)

    
