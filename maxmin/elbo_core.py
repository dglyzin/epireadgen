import numpy as np

import pyro
import pyro.distributions as pdist
from pyro.infer import SVI, Trace_ELBO
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine import trace, replay
from pyro.poutine.runtime import _PYRO_PARAM_STORE

import torch
import torch.distributions.constraints as constraints

import matplotlib.pyplot as plt

import sim5_hw1_rejsam as rejsam
import sim5_hw1_scimaxmin as scimaxmin


class SeddleElbo(rejsam.SeddleWithScores):

    def gen_params(self):
        alpha, beta = rejsam.SeddleWithScores.gen_params(self)
        return((alpha, beta), (alpha, beta))

    def calc_lossA(self, loss):
        lossA = loss-self.z_min
        # return lossA
        # return loss
        # return 0 if lossA > -0.01 else -np.inf
        lossA = torch.log((self.z_max-loss)/(self.z_max-self.z_min))
        return lossA

    def calc_lossB(self, loss):
        # lossB = self.z_max - loss
        # then 
        # lossB = torch.log((self.z_max - loss)/(self.z_max-self.z_min))
        # return lossB
        # return -loss
        # return 0 if lossB < 0.01 else -np.inf
        # this only for sampling with scores
        lossB = torch.log((loss-self.z_min)/(self.z_max-self.z_min))
        return lossB

    def get_funcA(self, alpha_init, beta):
        
        def f():
            '''A-model'''

            # since several Ax in on loop:
            # pyro.clear_param_store()
            # print("Ay given: ", Ay)
            
            '''
            Warning:

            # this line values will be replaced by the guide
            # but the dist itself will be used in elbo still!
            # so do not put unrealistic Normal(1., 0.01) here!
            Or otherwise (in case of Normal(1., 0.01)),
             the Ax1 in the model will be:
            ('Ax1', {
               'type': 'sample', 'name': 'Ax1',
               'fn': Normal(loc: 1.0, scale: 0.01),
               'value': tensor(0.6049, grad_fn=<AddBackward0>)
            
            while in the guide:

            ('Ax1', {
               'type': 'sample', 'name': 'Ax1',
                'fn': Normal(loc: 0.5999944806098938, scale: 0.01),
                'value': tensor(0.6049, grad_fn=<AddBackward0>)

            and so:

            # A rmodel_log_prob_sum: tensor(-750.7166, grad_fn=<AddBackward0>)
            # \A rguide_log_prob_sum: tensor(3.0741, grad_fn=<AddBackward0>)

            and also:

            # A rmodel.log_prob_sum-guide.log_prob_sum:
            # tensor(-753.7907, grad_fn=<SubBackward0>)
            # A elbo.loss:
            # 754.9568638801575
            
            and it will create a problem.
            This is because
            >>> dist.Normal(1.,0.01).log_prob(torch.tensor(0.6049))
            tensor(-776.8339)
            # and because factor losses will be to small (~-0.17676100134849548)
            # it grad would not change the log_prob_sum significantly
            # and hence the alpha param will almost be stationary
            # and hence Ax1 will be changed only slitely (0.6103/0.6107)
            '''
            Ax = pyro.sample("Ax1", pdist.Uniform(0., 2*self.a))
            # for reproducing the problem mentioned above:
            # Ax = pyro.sample("Ax1", pdist.Normal(1., 0.01))
            
            Ay = pyro.sample("Ay1", pdist.Normal(beta, 0.01))
            loss = self.run_model(Ax, Ay)
            
            self.lossA = self.calc_lossA(loss)
            pyro.factor("lossA", self.lossA)
        
        def g():
            '''A-guide'''
            alpha = pyro.param(
                "alpha", alpha_init,
                constraint=constraints.interval(0.01, 2*self.a))
            pyro.sample("Ax1", pdist.Normal(alpha, 0.01))
        return (f, g)

    def get_funcB(self, alpha, beta_init):

        def f():
            '''B-model'''

            # since several Ax in on loop:
            # pyro.clear_param_store()
            # print("Ay given: ", Ay)
            
            Ax = pyro.sample("Ax1", pdist.Normal(alpha, 0.01))
            
            # this line will be replaced by the guide
            # but the dist itself will be used in elbo still!
            # so do not put unrealistic Normal(1., 0.01) here!
            Ay = pyro.sample("Ay1", pdist.Uniform(0., 2*self.b))
            # for reproducing the problem mentioned above (in `get_funcA`):
            # Ay = pyro.sample("Ay1", pdist.Normal(1., 0.01))
            loss = self.run_model(Ax, Ay)
            
            self.lossB = self.calc_lossB(loss)
            pyro.factor("lossB", self.lossB)
        
        def g():
            '''B-guide'''
            beta = pyro.param(
                "beta", beta_init,
                constraint=constraints.interval(0.01, 2*self.b))
            pyro.sample("Ay1", pdist.Normal(beta, 0.01))
        return (f, g)

    def get_factor_loss(self, trace, whose_loss):
        loss_val = trace.nodes[whose_loss]['value'].clone().detach()
        loss_fn = trace.nodes[whose_loss]['fn']
        loss = loss_fn.log_prob(loss_val)
        return loss


class RunnerSeddleModelElbo(rejsam.RunnerSeddleModel):
    def get_model(self):
        a = self.a
        b = self.b
        model = SeddleElbo(a, b)
        model.init_params_dist()
        return model

    def get_replayed(self, f, g):
        guide_trace1 = trace(g).get_trace()
        replayed_model1 = replay(f, trace=guide_trace1)
        rtrace = trace(replayed_model1).get_trace()
        replayed_guide1 = replay(g, trace=guide_trace1)
        rgtrace = trace(replayed_guide1).get_trace()
        return (replayed_model1, replayed_guide1,
                guide_trace1, rtrace, rgtrace)

    def check_elbo(self, replayed_model1, replayed_guide1,
                   rtrace, rgtrace):
        rmodel_log_prob = rtrace.log_prob_sum()
        rguide_log_prob = rgtrace.log_prob_sum()
        print("\nA rmodel_log_prob:", rmodel_log_prob)
        print("\A rguide_log_prob:", rguide_log_prob)
        print("\nA rmodel.log_prob_sum-guide.log_prob_sum:")
        print(rmodel_log_prob-rguide_log_prob)

        # since Ay is different result will be different:
        elbo = Trace_ELBO()
        # elbo.loss_and_grads(model_test1, guide_test1,)
        print("\nA elbo.loss:")
        print(elbo.loss(replayed_model1, replayed_guide1))
        
        elbo1 = Trace_ELBO()
        # elbo.loss_and_grads(model_test1, guide_test1,)
        print("\nA elbo1.loss:")
        print(elbo1.loss(replayed_model1, replayed_guide1))

    def finilize_extended(self, Axs, Ays, lossesA_svi, lossesB_svi):
        tAxs = torch.cat(list(map(
            lambda a: a.unsqueeze(0), Axs)), 0)
        print("\nmean Axs:", torch.mean(tAxs, 0))
        print("var Axs:", torch.var(tAxs))

        tAys = torch.cat(list(map(
            lambda a: a.unsqueeze(0), Ays)), 0)
        print("\nmean Ays:", torch.mean(tAys, 0))
        print("var Ays:", torch.var(tAys))

        fig, axs = plt.subplots(1, 2)

        axs.flat[0].plot(Axs)
        axs.flat[0].set_title("Axs")
        axs.flat[1].plot(Ays)
        axs.flat[1].set_title("Ays")

        plt.show()

        fig, axs = plt.subplots(1, 2)

        axs.flat[0].plot(lossesA_svi)
        axs.flat[0].set_title("lossesA_svi")
        axs.flat[1].plot(lossesB_svi)
        axs.flat[1].set_title("lossesB_svi")

        plt.show()

    
def test_elbo_maxmin(steps, sampler_steps):
    model_runner = RunnerSeddleModelElbo(0.3, 0.7)
    elbo_maxmin(steps, model_runner, sampler_steps=sampler_steps)


def elbo_maxmin(
        steps, model_runner,
        sampler_steps=1, dbg=False):
    
    optim_sgd2 = pyro.optim.SGD({"lr": 0.01, "momentum": 0.1})

    Axs = []
    lossesA_svi = []

    Ays = []
    lossesB_svi = []

    def get_optA(model, alpha_beta):
        '''The `alpha` will be optimized here,
        the `beta` used only to sample Ay'''

        pyro.clear_param_store()
        
        alpha_init, beta = alpha_beta
        f, g = model.get_funcA(alpha_init, beta)

        # setup the inference algorithm
        svi = SVI(f, g, optim_sgd2, loss=Trace_ELBO())

        # do gradient steps
        for step in range(sampler_steps):
            # print("stepA:", step)
            # with trace(param_only=False) as tr:
            # with TraceMessenger() as tr:
            lossA_svi = svi.step()

            # the lossA of the whole model:
            lossesA_svi.append(float(lossA_svi))

        # since pyro.param is ParamStoreDict.get_param
        # which is not override the value if it exist alredy,
        # we can collect the trace
        # by rerunning the model
        replayed_res = model_runner.get_replayed(f, g)
        replayed_model1, replayed_guide1 = replayed_res[:2]
        gtrace, rtrace, rgtrace = replayed_res[2:]
        
        # FOR checking elbo:
        # model_runner.check_elbo(replayed_model1, replayed_guide1,
        #                         rtrace, rgtrace)
        # END FOR
        # print("\nA gtrace.nodes")
        # print(gtrace.nodes)
        # print("A rtrace.nodes")
        # print(rtrace.nodes)
        # print("A rgtrace.nodes")
        # print(rgtrace.nodes)
        
        # the lossA of the factor(lossA) only:
        lossA = model.get_factor_loss(rtrace, "lossA")

        res_Ax = rtrace.nodes["Ax1"]['value'].clone().detach()
        Axs.append(res_Ax)

        # we need only last param:
        res_alpha = _PYRO_PARAM_STORE["alpha"].clone().detach()
        # res_alpha = rtrace.nodes["alpha"]['value'].clone().detach()
        return lossA, (res_alpha, beta)

    def get_optB(model, alpha_beta):
        '''The `alpha` will be optimized here,
        the `beta` used only to sample Ay'''

        pyro.clear_param_store()
        
        alpha, beta_init = alpha_beta
        f, g = model.get_funcB(alpha, beta_init)

        # setup the inference algorithm
        # trace_loss
        svi = SVI(f, g, optim_sgd2, loss=Trace_ELBO())

        # do gradient steps
        for step in range(sampler_steps):
            # with TraceMessenger() as tr:
            # with trace(param_only=False) as tr:
            lossB_svi = svi.step()

            # the lossB of the whole model:
            lossesB_svi.append(float(lossB_svi))
    
        # since pyro.param is ParamStoreDict.get_param
        # which is not override the value
        # if it exist alredy we can collect the trace
        # by rerunning the model
        replayed_res = model_runner.get_replayed(f, g)
        rtrace = replayed_res[3]
        # TODO: use hol here:

        # the lossB of the factor(lossB) only:
        lossB = model.get_factor_loss(rtrace, "lossB")

        res_Ay = rtrace.nodes["Ay1"]['value'].clone().detach()
        Ays.append(res_Ay)

        # print("\nB guide_trace1.nodes")
        # print(guide_trace1.nodes)
        # print("B rtrace.nodes")
        # print(rtrace.nodes)
                
        # we need only last param:
        res_beta = _PYRO_PARAM_STORE["beta"].clone().detach()
        # print("res_beta:", res_beta)
        # res_beta = rtrace.nodes["beta"]['value'].clone().detach()
        return lossB, (alpha, res_beta)
        
    scimaxmin.optimize_maxmin(steps, model_runner, get_optA, get_optB, dbg=dbg)

    model_runner.finilize_extended(Axs, Ays, lossesA_svi, lossesB_svi)


if __name__ == "__main__":
    test_elbo_maxmin(40, 10)
    # done
    # lossesA[-1]: tensor(-1.7635, grad_fn=<ExpandBackward>)
    # lossesB[-1]: tensor(-0.1784, grad_fn=<ExpandBackward>)
    # mean lossesA: tensor(-1.3051, grad_fn=<MeanBackward1>)
    # mean lossesB: tensor(-0.3689, grad_fn=<MeanBackward1>)
    # this mean (alpha, beta) here:
    # (the result of both get_funcA and get_funcB)
    # Ax: tensor([0.2965, 0.7842])
    # Ay: tensor([0.2965, 0.7842])

