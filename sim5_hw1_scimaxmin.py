import numpy as np
import scipy as sp
from scipy import optimize as opt
import torch
from torch.functional import F

import sim5_hw1_sciopt as sciopt

import progresses.progress_cmd as progress_cmd
ProgressCmd = progress_cmd.ProgressCmd


class HwModel(sciopt.sm1.HwModel):
    # used to redefine the gen_params
    def gen_params(self):
        # define starting points
        Ax = torch.tensor([[0.1449, 0.9720],
                           [0.8551, 0.0280]])
        Ay = torch.tensor([[0.3485, 0.4437],
                           [0.6515, 0.5563]])
        return (Ax, Ay)


class RunnerHwModel(sciopt.RunnerHwModel):

    def prepare_param(self, Ax):
        '''From numpy.array to torch.tensor'''
        Ax = (torch.tensor(Ax).type(torch.float)
              .reshape(self.problem_shape))
        Ax = F.normalize(Ax, p=1, dim=0)
        return (Ax)

    # finilize used differently from sciopt case:
    def finalize(self, lossesA, lossesB, Ax, Ay):
        print("lossesA[-1]:", lossesA[-1])
        print("lossesB[-1]:", lossesB[-1])
        Ax = F.normalize(Ax, p=1, dim=0)
        Ay = F.normalize(Ay, p=1, dim=0)
        print("Ax:", Ax)
        print("Ay:", Ay)


class RunnerSeddleModel(sciopt.RunnerSeddleModel):

    def prepare_param(self, Ax):
        return torch.tensor(Ax)

    # finilize used differently from sciopt case:
    def finalize(self, lossesA, lossesB, Ax, Ay):
        print("lossesA[-1]:", lossesA[-1])
        print("lossesB[-1]:", lossesB[-1])
        print("Ax:", Ax)
        print("Ay:", Ay)
        

def optimize_maxmin(steps, model_runner, get_optA, get_optB,
                    dbg=True):
    model = model_runner.get_model()
    
    # gen init values:
    Ax, Ay = model.gen_params()
    print("\ninit Ax, Ay:")
    print(Ax)
    print(Ay)

    progress = ProgressCmd(steps, prefix="opt progress:")

    lossesA = []
    lossesB = []

    for step in range(steps):
        progress.succ(step)

        errB, Ay = get_optB(model, Ax)
        Ay = model_runner.prepare_param(Ay)
        errA, Ax = get_optA(model, Ay)
        Ax = model_runner.prepare_param(Ax)

        if dbg:
            print("Ax")
            print(Ax)
            print("Ay")
            print(Ay)
            print("\nerrA, errB:", errA, errB)
        lossesA.append(errA)
        lossesB.append(errB)

    progress.print_end()

    model_runner.finalize(lossesA, lossesB, Ax, Ay)


def test_hw_diff(steps, dbg=True):
    problem_shape = (2, 2)
    problem_size = sum(problem_shape)
    rangesA = [(0, 1)]*problem_size
    rangesB = [(0, 1)]*problem_size

    # replace this line by next if random init values needed:
    model_runner = RunnerHwModel(rangesA, rangesB, problem_shape,
                                 mClass=HwModel)
    # model_runner = RunnerHwModel(rangesA, rangesB, problem_shape)

    def get_optA(model, Ay):
        res = opt.differential_evolution(model.get_funcA(Ay), model.rangesA)
        res_Ax, fval = res.x, res.fun

        # loss alredy been converted to lossA in model.get_funcA:
        lossA = fval
        return lossA, res_Ax

    def get_optB(model, Ax):
        res = opt.differential_evolution(model.get_funcB(Ax), model.rangesB)
        res_Ay, fval = res.x, res.fun

        # loss alredy been converted to lossB in model.get_funcB:
        lossB = fval
        return lossB, res_Ay

    # solve the maxmin problem:
    optimize_maxmin(steps, model_runner, get_optA, get_optB, dbg=dbg)


def test_hw_brute(steps, dbg=True):
    '''Since brute is computationly hard for this problem
    very pure approximation is used here.'''

    problem_shape = (2, 2)
    problem_size = sum(problem_shape)
    rangesA = [slice(0, 1, 0.3)]*problem_size
    rangesB = [slice(0, 1, 0.3)]*problem_size

    # replace this line by next if random init values needed:
    model_runner = RunnerHwModel(rangesA, rangesB, problem_shape,
                                 mClass=HwModel)
    # model_runner = RunnerHwModel(rangesA, rangesB, problem_shape)

    def get_optA(model, Ay):
        res = opt.brute(model.get_funcA(Ay), model.rangesA,
                        full_output=True)
        res_Ax, fval, *_ = res

        # loss alredy been converted to lossA in model.get_funcA:
        lossA = fval
        return lossA, res_Ax

    def get_optB(model, Ax):
        res = opt.brute(model.get_funcB(Ax), model.rangesB,
                        full_output=True)
        res_Ay, fval, *_ = res

        # loss alredy been converted to lossB in model.get_funcB:
        lossB = fval
        return lossB, res_Ay

    # solve the maxmin problem:
    optimize_maxmin(steps, model_runner, get_optA, get_optB, dbg=dbg)


def test_seddle_brute(steps, dbg=True):

    # defining the ranges:
    a = torch.tensor([0.3])
    b = torch.tensor([0.7])
    print("a, b:", (a, b))

    problem_size = 1
    rangesA = [slice(0, 2*float(a[0]), 0.01)]*problem_size
    rangesB = [slice(0, 2*float(b[0]), 0.01)]*problem_size
    print("rangesA:", rangesA)
    print("rangesB:", rangesB)

    # defining the model:
    model_runner = RunnerSeddleModel(rangesA, rangesB, a, b)
    
    def get_optA(model, Ay):
        res = opt.brute(model.get_funcA(Ay), model.rangesA,
                        full_output=True)
        res_Ax, fval, *_ = res

        # loss alredy been converted to lossA in model.get_funcA:
        lossA = fval
        return lossA, res_Ax

    def get_optB(model, Ax):
        res = opt.brute(model.get_funcB(Ax), model.rangesB,
                        full_output=True)
        res_Ay, fval, *_ = res

        # loss alredy been converted to lossB in model.get_funcB:
        lossB = fval
        return lossB, res_Ay

    # solve the maxmin problem:
    optimize_maxmin(steps, model_runner, get_optA, get_optB, dbg=dbg)


if __name__ == "__main__":
    # test_hw_diff(3, dbg=True)
    test_hw_brute(3, dbg=True)
    # test_seddle_brute(1, dbg=False)
