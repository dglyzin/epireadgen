import torch
from torch.functional import F
from torch import optim
import torch.distributions as dist

import pyro
import pyro.distributions as pdist

import matplotlib.pyplot as plt

import sim5_hw as sm
import sim5_hw_tests as smt

import progresses.progress_cmd as progress_cmd
ProgressCmd = progress_cmd.ProgressCmd


class Seddle():
    '''Simple test for verifying the algorithms is working properly.'''

    def __init__(self, a, b, dbg=False):
        self.a = a
        self.b = b
        self.dbg = dbg
        
    def init_params_dist(self):
        self.depsilon = dist.Normal(0.01, 0.001)
    
        self.dA = dist.Uniform(0, 2*self.a)
        self.dB = dist.Uniform(0, 2*self.b)
        
    def gen_params(self):
        Ax = self.gen_param(self.dA)
        Ay = self.gen_param(self.dB)
        return (Ax, Ay)

    def gen_param(self, pdist):
        # initial guess:
        Ax = pdist.sample()
        return(Ax)
    
    def calc_lossA(self, loss):
        # minimum for A side means minimum of loss:
        return loss
    
    def calc_lossB(self, loss):
        # maximum for a B side means maximum of loss
        # and hence minimum for -loss:
        lossB = - loss
        # lossB = - loss + self.depsilon.sample()
        return lossB
    
    def run_model(self, Ax, Ay):
        loss = torch.pow((Ax-self.a), 2)/2. - torch.pow((Ay-self.b), 2)/2.
        return (loss, )


class HwModel():
        
    def __init__(self, sim_spec, dbg=False):

        # these will be given in optimization:
        sim_spec["agents"]["A"]["decision_matrix"] = None
        sim_spec["agents"]["B"]["decision_matrix"] = None

        self.sim_spec = sim_spec
        self.dbg = dbg

    def init_params_dist(self):
        
        self.depsilon = dist.Normal(0.01, 0.001)
        self.dA = dist.Uniform(torch.zeros(2, 2), torch.ones(2, 2))
        self.dB = dist.Uniform(torch.zeros(2, 2), torch.ones(2, 2))
        
    def gen_params(self):
        Ax = self.gen_param(self.dA)
        if self.dbg:
            print("init Ax:", Ax)
        Ay = self.gen_param(self.dB)
        # Ay = 0.1*torch.ones((2, 2))
        # Ay[0, 0] = 0.9
        # Ay[0, 1] = 0.9

        if self.dbg:
            print("init Ax:", Ax)
        return (Ax, Ay)

    def gen_param(self, pdist):
        # initial guess:
        Ax = pdist.sample()
        Ax = F.normalize(Ax, p=1, dim=0)
        # Ax = 0.5*torch.ones((2, 2))
        
        return(Ax)

    def calc_lossA(self, loss):
        lossA = loss
        # lossA = loss + self.depsilon.sample()  # - 0.001
        return lossA

    def calc_lossB(self, loss):
        # see model lost section
        y01 = self.sim_spec["agents"]["B"]["units"]["counts"][1]
        lossB = torch.pow(y01, 3) - loss
        # lossB = lossB - depsilonB.sample()  # - 0.001
        return lossB

    def run_model(self, Ax, Ay):
        self.sim_spec["agents"]["A"]["decision_matrix"] = Ax
        self.sim_spec["agents"]["B"]["decision_matrix"] = Ay

        # opt_mode=False since we do not need Ax.requires_grad_ inside:
        loss, Ax, Ay, x, y = model(self.sim_spec, opt_mode=False, mdbg=False)
        return (loss, Ax, Ay, x, y)


class HwModel1(HwModel):

    def gen_params(self):
        Ax = torch.tensor([[0.1449, 0.9720],
                           [0.8551, 0.0280]])
        Ay = torch.tensor([[0.3485, 0.4437],
                           [0.6515, 0.5563]])

        print("init Ax:", Ax)
        print("init Ay:", Ay)

        return (Ax, Ay)
    

def guide(model_spec, mdbg=True):
    aSpec = model_spec["agents"]["A"]
    bSpec = model_spec["agents"]["B"]
    Ax = aSpec["decision_matrix"]
    x0 = aSpec["counts"]
    y0 = bSpec["counts"]

    if Ax is None:
        # why `_T` is used here see description in `elbo_guide`:
        Ax_T_shape = (len(x0), len(y0))  # should be (len(y0), len(x0)) for Ax
        Ax_T = pyro.param("Ax_T", pdist.Uniform(0.5*torch.ones(Ax_T_shape)))
        Ax = Ax_T.T
        # since mcmc do not used guide and hence simplex constrain:
        Ax = F.normalize(Ax, p=1, dim=0)
        if mdbg:
            print("Ax:")
            print(Ax)
    # aSpec["decision_matrix"] = Ax


def model(model_spec, opt_mode=False, mdbg=True):
    '''Same as the model from sm but without ehandler'''
    T = model_spec["T"]
    dt = model_spec["dt"]
    aSpec = model_spec["agents"]["A"]
    bSpec = model_spec["agents"]["B"]
    # aSpec = mk_decisionsA_as_param(aSpec, bSpec, mdbg)

    U = model_spec["U"]
    ehandler = None
    
    x0, y0, Ua, Ub, Ax, Ay = sm.init_model(aSpec, bSpec, U, ehandler, mdbg)
        
    if opt_mode:
        # make Ax param for optimization:
        Ax.requires_grad_()

    x, y, Ax, Ay = sm.run_model(x0, y0, Ua, Ub, T, dt, Ax=Ax, Ay=Ay,
                                ehandler=ehandler,
                                mdbg=mdbg, edbg=False)

    # for a loss:
    goalA = aSpec["goal"](x, y)
    goalB = bSpec["goal"](x, y)
    factor = model_spec["init_factor"]
    for score in model_spec["scores"]:
        if score["test"](goalA, goalB):
            factor += score["factor"]
    loss = torch.pow(y[1], 3)
    # pyro.factor("err", factor) 
    
    return (loss, Ax, Ay, x, y)
    # return factor
    

def optimize(steps, model, sim_spec):
    
    lr = 0.0001

    # initial guess:
    Ax = 0.5*torch.ones((2, 2))
    Ax.requires_grad_()
    sim_spec["agents"]["A"]["decision_matrix"] = Ax

    opt = optim.SGD([Ax], lr=lr, momentum=0.9)
    # opt = optim.Adam(model.parameters())

    losses = []
    for step in range(steps):
        
        loss, Ax, x, y = model(sim_spec, opt_mode=True, mdbg=False)
        loss.backward()
        opt.step()
        # send gradients back to zeros:
        opt.zero_grad()
        ''' SGD instead of:
        with torch.no_grad():
           for p in model.parameters(): p -= p.grad * lr
           model.zero_grad()
        '''
        losses.append(loss)

    return(losses, Ax, x, y)


def optimize_maxmin(stepsA, stepsB, lrA, lrB, model,
                    switchB_used=True, dbg=True):
    '''$max_{Ax}min_{Ay} R(Ax, Ay)$

    - ``switchB_used`` -- adding randomness to B side controled
    param Ay in order to spread up searching space (support of the model 
    execution space).
    '''
    # lrA = 0.01
    # lrB = 0.01

    model.init_params_dist()

    # initial guess:
    Ax, Ay = model.gen_params()
    Ax.requires_grad_()
    optA = optim.SGD([Ax], lr=lrA, momentum=0.9)

    Ay.requires_grad_()
    optB = optim.SGD([Ay], lr=lrB, momentum=0.9)

    opt_progress = ProgressCmd(stepsA, prefix="opt progress:")
    lossesA = []
    lossesB = []
    for step in range(stepsA):
        opt_progress.succ(step)
        
        # min_{Ay} step (note detach() usage here):
        for step1 in range(stepsB):
            if switchB_used:
                switchB = dist.Bernoulli(step1/stepsB).sample().type(torch.long)
                # Ax_for_B = dB.sample() if switchB else Ax.detach()
                if switchB:
                    Ay = model.gen_param(model.dB)
                    Ay.requires_grad_()
                    optB = optim.SGD([Ay], lr=lrB, momentum=0.9)

            lossB, *_ = model.run_model(Ax.detach(), Ay)
            # do the x, y realy needed?
            lossB = model.calc_lossB(lossB)
            if dbg:
                print("lossB:", lossB)
            lossB.backward()
            # print("Ay.grad:", Ay.grad)
            optB.step()
            optB.zero_grad()

            lossesB.append(lossB.detach().clone())

        # max_{Ax} step:
        lossA, *_ = model.run_model(Ax, Ay.detach())
        # do the x, y realy needed?
        lossA = model.calc_lossA(lossA)
        # print("\nlossA:", lossA)

        lossA.backward()
        optA.step()
        optA.zero_grad()

        lossesA.append(lossA.detach().clone())
    opt_progress.print_end()
    return(lossesA, lossesB, Ax, Ay)


def test_maxmin_seddle(
        stepsA, stepsB, lrA, lrB,
        switchB_used=False, dbg=True):
    model = Seddle(0.3, 0.7, dbg=dbg)
    lossesA, lossesB, Ax, Ay = optimize_maxmin(
        stepsA, stepsB, lrA, lrB, model,
        switchB_used=switchB_used, dbg=dbg)
    
    # if dbg:
    print("optimization results:")
    print("lossesA[-1]:", lossesA[-1])
    print("lossesB[-1]:", lossesB[-1])
    print("Ax:", Ax)
    print("Ay:", Ay)
    # print("x:", x)
    # print("y:", y)

    plt.plot(lossesA)
    plt.show()
    plt.plot(lossesB)
    # plt.ylim([-1, 126])
    plt.show()

    
def test_maxmin_hw(
        stepsA, stepsB, lrA, lrB, steps_test,
        switchB_used=True, dbg=True):
    '''testing $max_{Ax}min_{Ay} R(Ax, Ay)$
    optimization.
    '''

    sim_spec = smt.mk_spec_for_test0()

    model = HwModel1(sim_spec, dbg=dbg)
    # model = HwModel(sim_spec, dbg=dbg)

    lossesA, lossesB, Ax, Ay = optimize_maxmin(
        stepsA, stepsB, lrA, lrB, model,
        switchB_used=switchB_used, dbg=dbg)

    Ax = F.normalize(Ax, p=1, dim=0)
    Ay = F.normalize(Ay, p=1, dim=0)

    loss, _, _, x, y = model.run_model(Ax, Ay)
    # if dbg:
    print("optimization results:")
    print("lossesA[-1]:", lossesA[-1])
    print("lossesB[-1]:", lossesB[-1])
    print("Ax:", Ax)
    print("Ay:", Ay)
    print("x:", x)
    print("y:", y)
    plt.plot(lossesA)
    plt.show()
    plt.plot(lossesB)
    plt.ylim([-1, 126])
    plt.show()

    # test:
    test_progress = ProgressCmd(steps_test, prefix="test progress:")
    
    losses = []
    loss_max = 0
    for step in range(steps_test):
        test_progress.succ(step)
        loss, Ax, Ay, x, y = model.run_model(Ax, None)
        if loss > loss_max:
            Ax_max = Ax
            Ay_max = Ay
            x_max = x
            y_max = y
            loss_max = loss

        losses.append(loss.detach().clone())
        # losses.append(torch.sign(loss).detach().numpy())
    test_progress.print_end()

    # if dbg:
    print("test results:")
    print("loss_max:", loss_max)

    print("Ax_max:", Ax_max)
    print("Ay_max:", Ay_max)
    print("x_max:", x_max)
    print("y_max:", y_max)
    print("losses<=0/losses:",
          len(list(filter(lambda x: x <= 0.02, losses)))/len(losses))
    plt.plot(losses)
    # plt.hist(losses)
    plt.show()


def test2(steps, dbg=True):
    sim_spec = smt.mk_spec_for_test0()
    Ay = torch.zeros((2, 2))

    # only unit 0 is important here
    Ay[0, 0] = 1.
    Ay[1, 0] = 0.5
    Ay[1, 1] = 0.5
    if dbg:
        print("Ay:", Ay)

    # Ay = F.normalize(Ay, p=1, dim=0)
    sim_spec["agents"]["B"]["decision_matrix"] = Ay
    # torch.autograd.set_detect_anomaly(True)
    losses, Ax, x, y = optimize(steps, model, sim_spec)
    if dbg:
        print("losses[-1]:", losses[-1])
        print("Ax:", Ax)
        print("x:", x)
        print("y:", y)
    plt.plot(losses)
    plt.show()

    
def test1(steps):
    
    min_loss = 100
    for step in range(steps):
        loss, Ax, x, y = test0(dbg=False)
        if min_loss >= loss:
            min_loss = loss
            min_Ax = Ax

    print("min_loss:", loss)
    print("min_Ax:", min_Ax)


def test0(dbg=True):
    sim_spec = smt.mk_spec_for_test0()
    Ay = torch.zeros((2, 2))

    # only unit 0 is important here
    Ay[0, 0] = 1.
    Ay[1, 0] = 0.5
    Ay[1, 1] = 0.5
    if dbg:
        print("Ay:", Ay)

    # Ay = F.normalize(Ay, p=1, dim=0)
    sim_spec["agents"]["B"]["decision_matrix"] = Ay

    loss, Ax, x, y = model(sim_spec, mdbg=True)
    if dbg:
        print("Ax", Ax)
        print("x:", x)
        print("y:", y)
        print("loss", loss)
    return (loss, Ax, x, y)


if __name__ == "__main__":

    # this is working correctly:
    # test_maxmin_seddle(700, 1, 0.001, 0.001, switchB_used=True, dbg=False)

    # this is working correctly:
    # test_maxmin_seddle(700, 1, 0.001, 0.001, switchB_used=False, dbg=False)

    # this is most promising test for hw_model from all below:
    # (note: it seems that pytorch maxmin implementation only work if there is
    #  some randomness):
    # (note: increasing stepsA father will give no beter result)
    test_maxmin_hw(170, 10, 0.001, 0.001, 130, switchB_used=True, dbg=True)

    # test_maxmin_hw(400, 3, 0.0001, 0.0001, 130, switchB_used=True, dbg=True)
    # test_maxmin_hw(20, 40, 0.001, 0.001, 130)
    # test_maxmin_hw(300, 10, 0.001, 0.001, 130)
    # test2(300)
    # test1(700)
    # test0()
    
