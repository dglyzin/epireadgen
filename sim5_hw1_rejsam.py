import pyro
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

import sim3
import sim5_hw as sm
import sim5_hw_tests as smt
import sim5_hw1 as sm1
import sim5_hw1_scimaxmin as scimaxmin


class Seddle(sm1.Seddle):

    def init_params_dist(self):
        self.depsilon = dist.Uniform(0.0, 0.01)
    
        a_min, a_max = (0., 2*self.a)
        b_min, b_max = (0., 2*self.b)
        self.dA = dist.Uniform(a_min, a_max)
        self.dB = dist.Uniform(b_min, b_max)
        
        self.z_max = self.run_model(
            torch.tensor(a_max), torch.tensor(self.b))
        self.z_min = self.run_model(
            torch.tensor(self.a), torch.tensor(b_max))

    def model_condA(self, trace):
        # print("traceA", trace.nodes)
        # print("Ax1:", trace.nodes['Ax1']['value'])
        udist = trace.nodes["lossA"]['fn']
        value = trace.nodes["lossA"]['value']
        
        # value of dist.Unit will be [] so log_prob
        # needed to extract putted by calc_lossA loss
        self.factorA = udist.log_prob(value)
        # print("self.factorA:", self.factorA)
        ## return True
        # the A side should put z to the min:
        return self.factorA <= 0+self.depsilon.sample()

    def model_condB(self, trace):
        # print("traceB", trace.nodes)
        # print("Ax1:", trace.nodes['Ay1']['value'])
        # they are not realy used
        # since rj scores usage for continues
        udist = trace.nodes["lossB"]['fn']
        value = trace.nodes["lossB"]['value']
        
        self.factorB = udist.log_prob(value)
        # print("self.factorB:", self.factorB)
        ## return True
        # the B side should put z to the max: 
        return self.factorB >= 0-self.depsilon.sample()

    def get_funcA(self, Ay):

        def f():
            # since several Ax in on loop:
            # pyro.clear_param_store()
            # print("Ay given: ", Ay)
            Ax = self.gen_param(self.dA)

            # store it for a runner.finalize in the get_opt
            # inside the optimize_maxmin
            pyro.deterministic("Ax1", Ax)
            
            loss = self.run_model(Ax, Ay)
            
            self.lossA = self.calc_lossA(loss)
            pyro.factor("lossA", self.lossA)
        return f

    def get_funcB(self, Ax):

        def f():
            # since several Ax in on loop:
            # pyro.clear_param_store()
            # print("Ax given: ", Ax)

            Ay = self.gen_param(self.dB)
            # store it for a runner.finalize in the get_opt
            # inside the optimize_maxmin
            pyro.deterministic("Ay1", Ay)

            loss = self.run_model(Ax, Ay)

            self.lossB = self.calc_lossB(loss)
            pyro.factor("lossB", self.lossB)
        return f

    def run_model(self, Ax, Ay):
        loss, *_ = sm1.Seddle.run_model(self, Ax, Ay)
        return loss

    def calc_lossA(self, loss):
        return loss

    def calc_lossB(self, loss):
        return loss


class SeddleWithScores(Seddle):
    
    # FOR rs.observe - used to observe the scores in trace
    # after a sample of the model been made:
    def observerA(self, trace):
        return self.observer(trace, "lossA")

    def observerB(self, trace):
        return self.observer(trace, "lossB")

    def observer(self, trace, var):
    
        val = trace.nodes[var]['value']
        fn = trace.nodes[var]['fn']
        # score = val
        score = fn.log_prob(val)
        
        # print("observer ", var, " score ", score)
        return(score)
    # END FOR

    def calc_lossA(self, loss):
        # exp here 
        # due to rej.samp usage of score in that case
        # score>=threshold+Uniform(0,1)
        # hence has to be in [0, 1] and converge to 1
        ## lossA = -(loss - self.z_min)
        self.lossA = torch.log((self.z_max-loss)/(self.z_max-self.z_min))
        # self.lossA = torch.exp(-(self.z_max-loss))
        self.factorA = self.lossA
        return self.lossA

    def calc_lossB(self, loss):
        # exp here 
        # due to rej.samp usage of score in that case
        # score>=threshold+Uniform(0,1)
        # hence has to be in [0, 1] and converge to 1
        ## lossB = - (self.z_max-loss)
        self.lossB = torch.log((loss-self.z_min)/(self.z_max-self.z_min))
        # self.lossB = torch.exp(-(loss-self.z_min))
        # print("lossB:", self.lossB)
        self.factorB = self.lossB
        return self.lossB

    def model_condA(self, trace):
        # they are not realy used
        # since rj scores usage for continues
        return True
        
    def model_condB(self, trace):
        # they are not realy used
        # since rj scores usage for continues
        return True


class RunnerFinalizeMean():
    def finalize(self, lossesA, lossesB, Ax, Ay):
    
        # print("lossesA:", lossesA)
        # print("lossesB:", lossesB)
        
        print("lossesA[-1]:", lossesA[-1])
        print("lossesB[-1]:", lossesB[-1])
        print("mean lossesA:", torch.mean(torch.cat(list(map(
            lambda a: a.unsqueeze(0), lossesA)), 0), 0))
        print("mean lossesB:", torch.mean(torch.cat(list(map(
            lambda a: a.unsqueeze(0), lossesB)), 0), 0))
        
        print("Ax:", Ax)
        print("Ay:", Ay)

        fig, axs = plt.subplots(1, 2)

        axs.flat[0].plot(lossesA)
        axs.flat[0].set_title("lossesA")
        axs.flat[1].plot(lossesB)
        axs.flat[1].set_title("lossesB")

        plt.show()


class RunnerSeddleModel(scimaxmin.RunnerSeddleModel, RunnerFinalizeMean):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def get_model(self):
        a = self.a
        b = self.b
        model = Seddle(a, b)
        model.init_params_dist()
        return model

    def finalize_extended(self, Axs, Ays):
       
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


class RunnerSeddleModelScore(RunnerSeddleModel):
    def get_model(self):
        a = self.a
        b = self.b
        model = SeddleWithScores(a, b)
        model.init_params_dist()
        return model
    

class HwModel(sm1.HwModel):
    '''with use of factors'''

    # used to redefine the gen_params
    '''
    def gen_params(self):
        # define starting points
        Ax = torch.tensor([[0.1449, 0.9720],
                           [0.8551, 0.0280]])
        Ay = torch.tensor([[0.3485, 0.4437],
                           [0.6515, 0.5563]])
        return (Ax, Ax.detach().clone())
        # return (Ax, Ay)
    '''
    def get_funcA(self, Ay):
        # print("\nSideA: Ay:", Ay)
        def f():
            # since several Ax in on loop:
            # pyro.clear_params_store()

            # Ax = self.gen_param("Ax", self.dA)

            # play for a side A:
            self.sim_spec["agents"]["A"]["goal"] = lambda x, y: y[1] <= 0
            self.sim_spec["agents"]["B"]["goal"] = lambda x, y: y[1] > 0

            # if goalA succ then override the factor to 0. and exit
            # if goalB succ then wait for a next step
            self.sim_spec["init_factor"] = -110.0
            self.sim_spec["scores"][0]["exit"] = True
            self.sim_spec["scores"][1]["exit"] = False

            loss, Ax1, Ay1, x, y = self.run_model(None, Ay)
            # print("\nsideA: Ax1", Ax1)
            # print("given Ay:", Ay)
            # print("lossA:", loss)
            pyro.deterministic("lossA", loss)
            pyro.deterministic("Ax1", Ax1)
            pyro.deterministic("y_1", y)
        return f

    def get_funcB(self, Ax):
        # print("Side B Ax:", Ax)
        def f():
            # play for a side B:
            self.sim_spec["agents"]["A"]["goal"] = lambda x, y: x[1] > 0
            self.sim_spec["agents"]["B"]["goal"] = lambda x, y: x[1] <= 0
            # if goalB fail then wait for a next step
            # if goalA fail then override the factor to -inf and exit
            # (the zero will still be preserved as succ in that case, 
            #  so no need for model_condB to be changed)
            self.sim_spec["init_factor"] = 0.0
            self.sim_spec["scores"][0]["exit"] = False
            self.sim_spec["scores"][1]["exit"] = True

            loss, Ax1, Ay1, x, y = self.run_model(None, Ax)
            # print("\nsideB: Ay= Ax1", Ax1)
            # print("given Ax:", Ax)
            # print("lossB", loss)
            # print("x, y:", x, y)
            pyro.deterministic("lossB", loss)
            pyro.deterministic("Ay1", Ax1)
            pyro.deterministic("y_1", x)
        return f

    def run_model(self, Ax, Ay):
        self.sim_spec["agents"]["A"]["decision_matrix"] = Ax
        self.sim_spec["agents"]["B"]["decision_matrix"] = Ay
        
        ehandler = sm.model(self.sim_spec, mdbg=False)

        observations = ehandler.get("observation")
        Ax = observations[1][-1]["value"]["Ax"]
        Ay = observations[1][-1]["value"]["Ay"]
        
        x = observations[1][-1]["value"]["x"]
        y = observations[1][-1]["value"]["y"]
        # print("y", observations[1][-1]["value"]["y"])
        
        # raise Exception("dbg")        
        factor = ehandler.trace.nodes["event_error_factor"]
        factor_value = factor["value"]
        # print("factor_value:", factor_value)
        factor_log = factor["fn"].log_prob(factor_value)
        # print("factor_log:", factor_log)
        # pyro.deterministic("loss", factor_log)
        # let get_func{A/B} decide what error to return
        return (factor_log, Ax, Ay, x, y)


class HwModelWithFactors(HwModel):
    def model_condA(self, trace):
        return self.model_cond(trace)

    def model_condB(self, trace):
        return self.model_cond(trace)

    def model_cond(self, trace):
        '''The same methods could been used
        since the sides been switched in run_model.'''

        udist = trace.nodes["event_error_factor"]['fn']
        value = trace.nodes["event_error_factor"]['value']
        factor = udist.log_prob(value)
        # print("self.factorA:", self.factorA)
        return float(factor) >= self.sim_spec["scores"][0]["factor"]
        # return trace.nodes["y_1"]["value"] <= 0


class HwModelWithoutFactors(HwModel):
    ''''without factors'''

    def model_condA(self, trace):
        factorA = trace.nodes["y_1"]["value"][1]
        return factorA <= 0

    def model_condB(self, trace):
        factorB = trace.nodes["y_1"]["value"][1]
        # TODO: test: maybe worth to relax that a bit: 
        return factorB > 0

    '''
    # alternative:
    def run_model(self, Ax, Ay):
        self.sim_spec["agents"]["A"]["decision_matrix"] = Ax
        self.sim_spec["agents"]["B"]["decision_matrix"] = Ay

        loss, Ax, Ay, x, y = sm1.model(sim_spec)
        
        pyro.deterministic("y_1", y[1])
        # raise Exception("dbg")
    '''


class HwModelWithoutFactorsScores(HwModel):
    ''''without factors'''
    # FOR rs.observe - used to observe the scores in trace
    # after a sample of the model been made:
    def observerA(self, trace):
        y = trace.nodes["y_1"]['value']
        return -y[1]/0.5

    def observerB(self, trace):
        y = trace.nodes["y_1"]['value']
        return -(0.5-y[1])/0.5

    def observer(self, trace, var):
    
        val = trace.nodes[var]['value']
        fn = trace.nodes[var]['fn']
        # score = val
        score = fn.log_prob(val)
        
        # print("observer ", var, " score ", score)
        return(score)
    # END FOR

    def model_condA(self, trace):
        # they are not realy used
        # since rj scores usage for continues
        return True
        
    def model_condB(self, trace):
        # they are not realy used
        # since rj scores usage for continues
        return True


class RunnerHwModel(scimaxmin.RunnerHwModel, RunnerFinalizeMean):
    def __init__(self, use_factors, use_score, dbg=False):
        self.use_factors = use_factors
        self.use_score = use_score
        self.dbg = dbg

    def get_model(self):
        sim_spec = smt.mk_spec_for_test0()

        if self.use_score:
            cHwModel = HwModelWithoutFactorsScores
        else:
            if self.use_factors:
                cHwModel = HwModelWithFactors
            else:
                cHwModel = HwModelWithoutFactors

        model = cHwModel(sim_spec, dbg=self.dbg)

        model.init_params_dist()
        return model

    def prepare_param(self, Ax):
        return Ax

    def finalize(self, lossesA, lossesB, Ax, Ay):
        # print("lossesA:", lossesA)
        # print("lossesB:", lossesB)
        
        print("lossesA[-1]:", lossesA[-1])
        print("lossesB[-1]:", lossesB[-1])
        print("Ax:", Ax)
        print("Ay:", Ay)

    def finalize_extended(self, Axs, Ays):
        tAxs = torch.cat(list(map(
            lambda a: a.unsqueeze(0), Axs)), 0)
        print("\nmean Axs:", torch.mean(tAxs, 0))
        print("var Axs:", torch.var(tAxs))

        tAys = torch.cat(list(map(
            lambda a: a.unsqueeze(0), Ays)), 0)
        print("\nmean Ays:", torch.mean(tAys, 0))
        print("var Ays:", torch.var(tAys))
                
        # print("tAxs:", tAxs)
        # print("tAys:", tAys)

        fig, axs = plt.subplots(2, 2)

        axs.flat[0].plot(tAxs[:, 0, 0])
        axs.flat[0].set_title("Axs[:, 0, 0]")
        axs.flat[1].plot(tAxs[:, 0, 1])
        axs.flat[1].set_title("Axs[:, 0, 1]")
        axs.flat[2].plot(tAys[:, 0, 0])
        axs.flat[2].set_title("Ays[:, 0, 0]")
        axs.flat[3].plot(tAys[:, 0, 1])
        axs.flat[3].set_title("Ays[:, 0, 1]")

        plt.show()

       
def model(sim_spec=None, **kwargs):
    ehandler = sm.model(sim_spec, **kwargs)

    '''
    observations = ehandler.get("observation")
    y = observations[1][-1]["value"]["y"]
    # print("y", observations[1][-1]["value"]["y"])
    pyro.deterministic("y_1", y[1])
    # raise Exception("dbg")
    '''


def test_maxmin_hw(
        steps, sampler_steps=1,
        use_score=False, threshold=0., max_counter=100,
        use_factors=True,
        dbg=False):
    model_runner = RunnerHwModel(
        use_factors, use_score)
    test_rs_maxmin(
        steps, model_runner,
        sampler_steps=sampler_steps,
        use_score=use_score, max_counter=max_counter, threshold=threshold,
        dbg=dbg)

    
def test_maxmin_seddle(
        steps, sampler_steps=1, use_score=False, threshold=0., max_counter=100,
        dbg=False):
    if use_score:
        model_runner = RunnerSeddleModelScore(0.3, 0.7)
    else:
        model_runner = RunnerSeddleModel(0.3, 0.7)

    test_rs_maxmin(
        steps, model_runner,
        sampler_steps=sampler_steps,
        use_score=use_score, max_counter=max_counter, threshold=threshold,
        dbg=dbg)

    
def test_rs_maxmin(
        steps, model_runner,
        sampler_steps=1,
        use_score=False, threshold=0., max_counter=100,
        dbg=False):

    '''
    - ``threshold`` -- score>=threshold+Uniform(0,1).log_prob
    or rather p(score)>=e^{threshold}*Uniform(0,1).

    - ``max_counter`` -- is max count of samples attempt before given up.
    
    '''
    Axs = []
    Ays = []

    def get_optA(model, Ay):
        observerA = model.observerA if use_score else None
    
        i = 0
        samples = []
        while len(samples) == 0:
            i += 1
            if i > max_counter:
                raise(Exception("max_counter excided in get_optA"))
            
            samples = sim3.rejection_sampling1(
                model.get_funcA(Ay), {}, model.model_condA, sampler_steps,
                use_score=use_score, threshold=threshold,
                observer=observerA, cprogress=None)

        # print("samples:")
        # print(samples)
        # print("get_optA:samples[-1].trace:", samples[-1].nodes)

        if sampler_steps > 1:
            res_Ax = torch.mean(torch.cat(list(map(
                lambda a: a.nodes["Ax1"]['value'].unsqueeze(0), samples)), 0), 0)
            lossA = torch.mean(torch.cat(list(map(
                lambda a: a.nodes["lossA"]['fn'].log_prob(a.nodes["lossA"]['value']).unsqueeze(0), samples)), 0), 0)
        
        else:
            res_Ax = samples[-1].nodes["Ax1"]['value']
            lossA = samples[-1].nodes["lossA"]['fn'].log_prob(samples[-1].nodes["lossA"]['value'])
        Axs.append(res_Ax)

        # loss alredy been converted to lossA in model.get_funcA:
        # lossA = model.factorA
        
        return lossA, res_Ax

    def get_optB(model, Ax):
        observerB = model.observerB if use_score else None

        i = 0
        samples = []
        while len(samples) == 0:
            i += 1
            if i > max_counter:
                raise(Exception("max_counter excided in get_optB"))

            samples = sim3.rejection_sampling1(
                model.get_funcB(Ax), {}, model.model_condB, sampler_steps,
                use_score=use_score, threshold=threshold,
                observer=observerB, cprogress=None)
        # print("samples:")
        # print(samples)
        # print("get_optB:samples[-1].trace:", samples[-1].nodes)
        if sampler_steps > 1:
            res_Ay = torch.mean(torch.cat(list(map(
                lambda a: a.nodes["Ay1"]['value'].unsqueeze(0), samples)), 0), 0)
            lossB = torch.mean(torch.cat(list(map(
                lambda a: a.nodes["lossB"]['fn'].log_prob(a.nodes["lossB"]['value']).unsqueeze(0), samples)), 0), 0)

        else:
            res_Ay = samples[-1].nodes["Ay1"]['value']
            lossB = samples[-1].nodes["lossB"]['fn'].log_prob(samples[-1].nodes["lossB"]['value'])
        Ays.append(res_Ay)

        # loss alredy been converted to lossB in model.get_funcB:
        # lossB = model.factorB

        return lossB, res_Ay

    try:
        scimaxmin.optimize_maxmin(
            steps, model_runner, get_optA, get_optB, dbg=dbg)

    except Exception as e:
        print("forcing exit due to error:")
        print(e)
        if "get_optA" in str(e):
            print("the cause are: A side fail to find solution (i.e. Ax)!")
        elif "get_optB" in str(e):
            print("the cause are: B side fail to find solution (i.e. Ay)!")

    model_runner.finalize_extended(Axs, Ays)


def test_rs():
    sim_spec = smt.mk_spec_for_test0()
    smt.run_test_and_show(sim_spec, mdbg=True)

    # mcmc, losses = sm.run_mcmc(1200, sim_spec, mdbg=False, edbg=False)

    losses = []
    samples = sim3.rejection_sampling1(
        model, {"sim_spec": sim_spec, "losses": losses}, model_cond, 3)
    print("samples:")
    print(samples)

    '''
    df = sim3.make_dataFrame(samples)
    sim3.plot_results1(samples, "sum")
    df[:10].sort_values("sum", ascending=True)

    sim_spec = sm.update_spec(sim_spec, mcmc, idx=-1, side="A", dbg=False)
    print("\nsolution:", sim_spec['agents']['A'])
    factors = run_tests_and_collect_factors(30, sim_spec)
    run_test_and_show(sim_spec)
    '''


if __name__ == "__main__":
    '''
    # use_score=True
    # TODO: seems not working correctly:
    test_maxmin_hw(
        30, sampler_steps=1,
        use_score=True, threshold=1.7, max_counter=3000,
        use_factors=False,
        dbg=False)
    '''
    '''
    # use_factors=False
    test_maxmin_hw(
        30, sampler_steps=1,
        use_score=False, threshold=0.0, max_counter=3000,
        use_factors=False,
        dbg=False)
    '''
    
    # use_factors=True
    test_maxmin_hw(
        30, sampler_steps=1,
        use_score=False, threshold=0.0, max_counter=3000,
        use_factors=True,
        dbg=False)
    

    # test_maxmin_seddle(70, sampler_steps=70, use_score=False, threshold=0.0, max_counter=3000, dbg=False)
    # lossesA[-1]: tensor(0.0018)
    # lossesB[-1]: tensor(-0.0013)
    # mean lossesA: tensor(0.0018)
    # mean lossesB: tensor(-0.0015)
    # Ax: tensor(0.2992)
    # Ay: tensor(0.6747)

    # mean Axs: tensor(0.2975)
    # var Axs: tensor(0.0002)
    
    # mean Ays: tensor(0.6966)
    # var Ays: tensor(0.0003)

    # test_maxmin_seddle(70, sampler_steps=70, use_score=True, threshold=0.0, max_counter=3000, dbg=False)
    # lossesA[-1]: tensor(-2.0817)
    # lossesB[-1]: tensor(-0.4801)
    # mean lossesA: tensor(-2.1176)
    # mean lossesB: tensor(-0.4554)
    # Ax: tensor(0.3666)
    # Ay: tensor(0.7130)

    # mean Axs: tensor(0.3028)
    # var Axs: tensor(0.0031)

    # mean Ays: tensor(0.7053)
    # var Ays: tensor(0.0027)

    # discrete rej.sample:
    # test_maxmin_seddle(70, use_score=False, threshold=6., max_counter=3000, dbg=False)
