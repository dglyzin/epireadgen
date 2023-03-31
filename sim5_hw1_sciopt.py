import numpy as np
import scipy as sp
from scipy import optimize as opt
import torch
from torch.functional import F
import torch.distributions as dist
import time

import sim5_hw1 as sm1
import sim5_hw_tests as smt


def wrap(mClass):
    '''Wrap mClass with functions for `scipy.optimize`
    and they ranges formats.'''

    class Wrapper(mClass):
        
        def __init__(self, rangesA, rangesB, *args, shape=None, **kwargs):

            mClass.__init__(self, *args, **kwargs)
            self.rangesA = rangesA
            self.rangesB = rangesB
            self.shape = shape
            
        def get_funcA(self, Ay):

            def f(Ax):
                Ax = torch.tensor(Ax).type(torch.float)
                if self.shape is not None:
                    Ax = Ax.reshape(self.shape)
                if self.dbg:
                    print("f:Ax:", Ax)
                
                loss, *args = self.run_model(Ax, Ay)
                if self.dbg:
                    print("f:loss, Ax, x, y:", (loss, )+tuple(args))
        
                lossA = self.calc_lossA(loss)
                return lossA.detach().clone().numpy()
            
            return f

        def get_funcB(self, Ax):
            
            def f(Ay):
                    
                Ay = torch.tensor(Ay).type(torch.float)
                if self.shape is not None:
                    Ay = Ay.reshape(self.shape)
                if self.dbg:
                    print("f:Ay:", Ay)
                loss, *args = self.run_model(Ax, Ay)
                if self.dbg:
                    print("f:loss, Ay, x, y:", (loss, )+tuple(args))
        
                lossB = self.calc_lossB(loss)
                return lossB.detach().clone().numpy()
            
            return f

    return Wrapper


@wrap
class wSeddle(sm1.Seddle):
    '''Here One need to convert scalars to vectors for `scipy.optimize`'''
    def init_params_dist(self):
        
        self.dA = dist.Uniform(torch.zeros_like(self.a), 2*self.a)
        self.dB = dist.Uniform(torch.zeros_like(self.a), 2*self.b)

    def run_model(self, Ax, Ay):
        loss, *res = sm1.Seddle.run_model(self, Ax, Ay)
        return (loss.sum(),)+tuple(res)

    
class RunnerSeddleModel():
    '''Some common methods used for a `seddle`.'''
    def __init__(self, rangesA, rangesB, a, b):
        self.rangesA = rangesA
        self.rangesB = rangesB
        self.a = a
        self.b = b

    def get_model(self):
        rangesA = self.rangesA
        rangesB = self.rangesB
        a = self.a
        b = self.b
        model = wSeddle(rangesA, rangesB, a, b)
        model.init_params_dist()
        return model

    def get_Ay(self, model):
        # Ax = model.gen_param(model.dA)
        
        # randomly generated or pricise:
        Ay = torch.tensor([0.7])
        # Ay = model.gen_param(model.dB)
        return Ay

    def finilize(self, res_Ax, fval, model, Ay):
        '''What to do with the result.'''

        print("\nfor given Ay=", Ay)
        print("result is:")
        print("res_Ax:")
        print(res_Ax)
        print("fval:")
        print(fval)
        loss, *_ = model.run_model(torch.tensor(res_Ax), Ay)
        print("loss:", loss)


class RunnerHwModel():
    '''Some common methods used for a `hw_model`.'''

    def __init__(self, rangesA, rangesB, problem_shape, mClass=None):
        '''
        - ``mClass`` -- if None, sm1.HwModel will be used.
        '''
        self.rangesA = rangesA
        self.rangesB = rangesB
        self.problem_shape = problem_shape
        self.mClass = mClass

    def get_model(self):
        rangesA = self.rangesA
        rangesB = self.rangesB
        problem_shape = self.problem_shape

        sim_spec = smt.mk_spec_for_test0()

        if self.mClass is None:
            wHwModel = wrap(sm1.HwModel)
        else:
            wHwModel = wrap(self.mClass)

        model = wHwModel(rangesA, rangesB, sim_spec,
                         shape=problem_shape, dbg=False)
        print("model.rangesA:", model.rangesA)
        print("model.rangesB:", model.rangesB)

        model.init_params_dist()
        return model

    def get_Ay(self, model):
        # Ax = model.gen_param(model.dA)
        Ay = model.gen_param(model.dB)
        Ay[0, 0] = 1.
        Ay[1, 0] = 0.
        Ay[0, 1] = 1.
        Ay[1, 1] = 0.
        print("Ay:", Ay)

        return Ay

    def finilize(self, res_Ax, fval, model, Ay):
        '''What to do with the result.'''

        res_Ax = torch.tensor(res_Ax).reshape(model.shape).type(torch.float)
        res_Ax = F.normalize(res_Ax, p=1, dim=0)

        print("res_Ax:")
        print(res_Ax)
        print("fval:")
        print(fval)
        loss, _, x, y = model.run_model(res_Ax, Ay)
        print("loss:", loss)
        print("x:", x)
        print("y:", y)


def optimize(model_runner, get_opt):
    '''Using `get_opt` general funtion to wrapp several
    optimizers in one code.'''

    model = model_runner.get_model()
    Ay = model_runner.get_Ay(model)

    last_time = time.time()
    get_opt(model, Ay)
    print("working time:", time.time()-last_time)
    # res = opt.brute(model.get_funcA(Ay), model.rangesA, full_output=True)
    # print("res:", res)


def test_hw_shgo():
    problem_shape = (2, 2)
    problem_size = sum(problem_shape)
    rangesA = [(0, 1)]*problem_size
    rangesB = [(0, 1)]*problem_size

    model_runner = RunnerHwModel(rangesA, rangesB, problem_shape)

    def get_opt(model, Ay):
        res = opt.shgo(model.get_funcA(Ay), model.rangesA)
        print("res:", res)
        print("res.fun:", res.fun)
        print("res.x", res.x)
        model_runner.finilize(res.x, res.fun, model, Ay)

    optimize(model_runner, get_opt)
    

def test_hw_diff():
    problem_shape = (2, 2)
    problem_size = sum(problem_shape)
    rangesA = [(0, 1)]*problem_size
    rangesB = [(0, 1)]*problem_size

    model_runner = RunnerHwModel(rangesA, rangesB, problem_shape)

    def get_opt(model, Ay):
        res = opt.differential_evolution(model.get_funcA(Ay), model.rangesA)
        print("res:", res)
        print("res.fun:", res.fun)
        print("res.x", res.x)
        model_runner.finilize(res.x, res.fun, model, Ay)

    optimize(model_runner, get_opt)


def test_hw_brute():
    problem_shape = (2, 2)
    problem_size = sum(problem_shape)
    rangesA = [slice(0, 1, 0.3)]*problem_size
    rangesB = [slice(0, 1, 0.3)]*problem_size

    model_runner = RunnerHwModel(rangesA, rangesB, problem_shape)

    def get_opt(model, Ay):
        res = opt.brute(model.get_funcA(Ay), model.rangesA, full_output=True)
        res_Ax, fval, *_ = res
        model_runner.finilize(res_Ax, fval, model, Ay)

    optimize(model_runner, get_opt)


def test_seddle_brute():
    print("numpy.version:", np.version.short_version)
    print("scipy.version:", sp.version.short_version)

    a = torch.tensor([0.3])
    b = torch.tensor([0.7])
    print("a, b:", (a, b))

    problem_size = 1
    rangesA = [slice(0, 2*float(a[0]), 0.01)]*problem_size
    rangesB = [slice(0, 2*float(b[0]), 0.01)]*problem_size
    print("rangesA:", rangesA)
    print("rangesB:", rangesB)

    model_runner = RunnerSeddleModel(rangesA, rangesB, a, b)

    def get_opt(model, Ay):
        res = opt.brute(model.get_funcA(Ay), model.rangesA,
                        full_output=True)
        # res = opt.brute(model.get_funcB(Ax), model.rangesB, full_output=True)
        # res = opt.brute(model.get_funcA(Ay), model.rangesA)
        # print("res:")
        # print(res)

        res_Ay, fval, *_ = res
        model_runner.finilize(res_Ay, fval, model, Ay)
    optimize(model_runner, get_opt)


if __name__ == "__main__":
    # test_hw_shgo()
    # test_hw_diff()
    test_hw_brute()
    # test_seddle_brute()
