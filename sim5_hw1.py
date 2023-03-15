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


def model_init(sim_spec):
    y01 = sim_spec["agents"]["B"]["units"]["counts"][1]
    dA = dist.Uniform(torch.zeros(2, 2), torch.ones(2, 2))
    dB = dist.Uniform(torch.zeros(2, 2), torch.ones(2, 2))
    return (y01, dA, dB)


def model_init_param(dA):
    # initial guess:
    Ax = dA.sample()
    Ax = F.normalize(Ax, p=1, dim=0)
    # Ax = 0.5*torch.ones((2, 2))
    print("init Ax:", Ax)
    return(Ax)


def run_model(sim_spec, Ax, Ay):
    sim_spec["agents"]["A"]["decision_matrix"] = Ax
    sim_spec["agents"]["B"]["decision_matrix"] = Ay
    loss, _, x, y = model(sim_spec, opt_mode=False, mdbg=False)
    return (loss, x, y)

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
    
    return (loss, Ax, x, y)
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


def optimize_maxmin(stepsA, stepsB, lrA, lrB,  model, sim_spec):
    '''$max_{Ax}min_{Ay} R(Ax, Ay)$'''
    # lrA = 0.01
    # lrB = 0.01

    # see model lost sect
    y01 = sim_spec["agents"]["B"]["units"]["counts"][1]

    # initial guess:
    dA = dist.Uniform(torch.zeros(2, 2), torch.ones(2, 2))
    Ax = dA.sample()
    Ax = F.normalize(Ax, p=1, dim=0)
    # Ax = 0.5*torch.ones((2, 2))
    print("init Ax:", Ax)

    Ax.requires_grad_()
    optA = optim.SGD([Ax], lr=lrA, momentum=0.9)

    dB = dist.Uniform(torch.zeros(2, 2), torch.ones(2, 2))
    Ay = dB.sample()
    Ay = F.normalize(Ay, p=1, dim=0)
    # Ay = 0.1*torch.ones((2, 2))
    # Ay[0, 0] = 0.9
    # Ay[0, 1] = 0.9
    print("init Ay:", Ay)

    Ay.requires_grad_()
    optB = optim.SGD([Ay], lr=lrB, momentum=0.9)

    depsilon = dist.Normal(0.01, 0.001)
    depsilonA = dist.Normal(0.5, 0.1)
    opt_progress = ProgressCmd(stepsA, prefix="opt progress:")
    lossesA = []
    lossesB = []
    for step in range(stepsA):
        opt_progress.succ(step)
        
        # min_{Ay} step (note detach() usage here):
        for step1 in range(stepsB):
            switchB = dist.Bernoulli(step1/stepsB).sample().type(torch.long)
            # Ax_for_B = dB.sample() if switchB else Ax.detach()
            if switchB:
                Ay = dB.sample()
                Ay = F.normalize(Ay, p=1, dim=0)
                Ay.requires_grad_()
                optB = optim.SGD([Ay], lr=lrB, momentum=0.9)

            Ax_for_B = Ax.detach()
            sim_spec["agents"]["A"]["decision_matrix"] = Ax_for_B
            sim_spec["agents"]["B"]["decision_matrix"] = Ay
            # opt_mode=False since we do not need Ax.requires_grad_ inside:
            lossB, _, x, y = model(sim_spec, opt_mode=False, mdbg=False)
            lossB = torch.pow(y01, 3) - lossB
            # lossB = lossB - depsilonB.sample()  # - 0.001
            print("lossB:", lossB)
            lossB.backward()
            # print("Ay.grad:", Ay.grad)
            optB.step()
            optB.zero_grad()

            lossesB.append(lossB.detach().clone())

        # max_{Ax} step:
        sim_spec["agents"]["A"]["decision_matrix"] = Ax
        sim_spec["agents"]["B"]["decision_matrix"] = Ay.detach()
        lossA, _, x, y = model(sim_spec, opt_mode=False, mdbg=False)
        # print("\nlossA:", lossA)
        lossA = lossA + depsilon.sample()  # - 0.001
        lossA.backward()
        optA.step()
        optA.zero_grad()

        lossesA.append(lossA.detach().clone())
    opt_progress.print_end()
    return(lossesA, lossesB, Ax, Ay, x, y)


def test3(stepsA, stepsB, lrA, lrB, steps_test, dbg=True):
    '''testing $max_{Ax}min_{Ay} R(Ax, Ay)$
    optimization.'''

    sim_spec = smt.mk_spec_for_test0()

    # these will be given in optimization:
    sim_spec["agents"]["A"]["decision_matrix"] = None
    sim_spec["agents"]["B"]["decision_matrix"] = None
    
    lossesA, lossesB, Ax, Ay, x, y = optimize_maxmin(
        stepsA, stepsB, lrA, lrB, model, sim_spec)

    Ax = F.normalize(Ax, p=1, dim=0)
    Ay = F.normalize(Ay, p=1, dim=0)

    if dbg:
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
        sim_spec["agents"]["B"]["decision_matrix"] = None
        loss, Ax, x, y = model(sim_spec, opt_mode=True, mdbg=False)
        if loss > loss_max:
            Ax_max = Ax
            x_max = x
            y_max = y
            loss_max = loss

        losses.append(loss.detach().clone())
        # losses.append(torch.sign(loss).detach().numpy())
    test_progress.print_end()

    if dbg:
        print("test results:")
        print("loss_max:", loss_max)
        
        print("Ax_max:", Ax_max)
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

    test3(20, 40, 0.001, 0.001, 130)
    # test3(300, 10, 0.001, 0.001, 130)
    # test2(300)
    # test1(700)
    # test0()
    
