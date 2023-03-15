import sim5_hw as sm

import torch
import matplotlib.pyplot as plt


def run_test_and_show(sim_spec, mdbg=False, edbg=False):
    '''Run the model with given spec.'''
    losses = []
    ehandler = sm.test_mk_model(sim_spec, losses, mdbg, edbg)

    sm.show_model_states_context(
        ehandler, ["x", "y", "Ua", "Ub", "Ax", "Ay", "events_to_check"])
    sm.show_model_states_context(
        ehandler, ["utypes"],
        observation_name="sample_space_of_a_utypes")
    sm.show_model_states_context(
            ehandler, ["ucounts"],
            observation_name="counts_of_x0")

    sm.show_model_trace(ehandler, ["x0", "Ax_T", "y0", "Ay_T"])
   

def run_tests_and_collect_factors(count, sim_spec):
    '''Run the model `count` times with given sim.'''
    factors = []
    losses = []
    for i in range(count):
        ehandler = sm.test_mk_model(sim_spec, losses, False, False)
        factor = ehandler.trace.nodes["event_error_factor"]
        factor_value = factor["value"]
        factor_log = factor["fn"].log_prob(factor_value)
        factors.append(factor_log)

    factors = torch.cat([
        factor.unsqueeze(0) for factor in factors]).type(torch.long)
    print("factors:")
    print(factors)
    print("losses:")
    print(losses)

    plt.hist(factors, bins=15)
    plt.title("factors")
    plt.show()
    return factors


def mk_spec(U, A_spec, B_spec):
    
    sim_spec = {
        "agents": {"A": A_spec, "B": B_spec},

        "T": 30,
        # than smaller it became than less influence of
        # who will attack first:
        "dt": 0.5,
        # "dt": 0.5,
        "init_factor": 0.0,
        # "init_factor": -110.0,

        "U": U,

        # choose side A (i.e. want A to win):
        "scores": [
            {
                "test": lambda goalA, goalB: goalA and not goalB,
                "once": False,
                "factor": 0,
                # will be exited once happend, factor been be overriden
                "exit": True
            },
            {
                "test": lambda goalA, goalB: not goalA and goalB,
                "once": False,
                "factor": -1000,
                # will be exited once happend, factor been be overriden
                "exit": True
            },
            {
                # it is actualy some kind of learning rate:
                "test": lambda goalA, goalB: not goalA and not goalB,
                # only once from all times will factor been used  
                "once": True,
                "factor": -100,
                # "factor": 10,
                "exit": False
            }
        ] 
    }
    return sim_spec

# ######################### tests #########################:


def mk_spec_for_test0():
    # efficiency matrix:
    # (will be transposed)
    U = 0.1*torch.ones((2, 2))
    # first effective against second:
    # U[0][1] = 0.9  # 0.7

    # second effective against first:
    # U[1][0] = 0.9

    U[0][:] = 0.5

    A_spec = {
        "decision_matrix": None,
        "units": {

            # needed only if there is neither types nor counts given:
            # "possible_types": [0, 1, 2, 3, 4],
            # "chosen_count": 3,
            "types": [0, 1],

            "counts": torch.tensor([5, 5]).type(torch.float)
            # "counts": torch.tensor([3, 3, 4]).type(torch.float)
            # maximal amount of units to sample for any type
            # must be given if "counts" not:
            # "max_count": [3, 3, 2],
            ## "min_count": 0,
            ## "max_count": 4,
        },
        # "goal": lambda x, y: (y <= 0).all(),
        "goal": lambda x, y: y[1] <= 0
        # "goal": lambda x, y: y[0] <= 0 and y[1] <= 0 and x[-1] > 2
    }

    B_spec = {
        "decision_matrix": None,
        "units": {
            "types": [0, 1],
            "counts": torch.tensor([5, 5]).type(torch.float)
        },
        "goal": lambda x, y: y[1] > 0
        # "goal": lambda x, y: x[0] <= 0
        # "goal": lambda x, y: (x <= 0).all()
        # "goal": lambda x, y: x[1] <= 0 and x[-1] <= 0,  # and y[2] >= 2
    }

    sim_spec = mk_spec(U, A_spec, B_spec)
    return sim_spec


def test0():
    '''
    U: 1>2; 1=1>2=2
    goals:
    A: B.2=0
    B: B.2!=0
    find decision_matrix
    '''
    sim_spec = mk_spec_for_test0()
    run_test_and_show(sim_spec, mdbg=True)

    mcmc, losses = sm.test_mcmc(20, sim_spec, mdbg=False, edbg=False)
    sim_spec = sm.update_spec(sim_spec, mcmc, idx=-1, side="A", dbg=False)
    print("\nsolution:", sim_spec['agents']['A'])
    factors = run_tests_and_collect_factors(30, sim_spec)


def mk_spec_for_test1():
    # efficiency matrix:
    # (will be transposed)
    U = 0.1*torch.ones((2, 2))
    # first effective against second:
    # U[0][1] = 0.9  # 0.7

    # second effective against first:
    # U[1][0] = 0.9

    U[0][:] = 0.5

    A_spec = {
        "decision_matrix": None,
        "units": {

            # needed only if there is neither types nor counts given:
            # "possible_types": [0, 1, 2, 3, 4],
            # "chosen_count": 3,
            "types": [0, 1],

            "counts": torch.tensor([7, 3]).type(torch.float),
            # "counts": torch.tensor([5, 5]).type(torch.float)
            # "counts": torch.tensor([3, 3, 4]).type(torch.float)
            # maximal amount of units to sample for any type
            # must be given if "counts" not:
            "max_count": [7, 5],
            "min_count": [5, 0],
            ## "min_count": 0,
            ## "max_count": 4,
        },
        # "goal": lambda x, y: (y <= 0).all(),
        "goal": lambda x, y: (y <= 0).all() and x.sum() <= 10  # units restriction
        # "goal": lambda x, y: y[1] <= 0 and x.sum() <= 10  # units restriction
        # "goal": lambda x, y: y[0] <= 0 and y[1] <= 0 and x[-1] > 2
    }

    B_spec = {
        "decision_matrix": None,
        "units": {
            "types": [0, 1],
            "counts": torch.tensor([5, 5]).type(torch.float)
        },
        "goal": lambda x, y: x[0] <= 0
        # "goal": lambda x, y: (x <= 0).all()
        # "goal": lambda x, y: x[1] <= 0 and x[-1] <= 0,  # and y[2] >= 2
    }

    sim_spec = mk_spec(U, A_spec, B_spec)
    return sim_spec


def test1():
    '''
    U: 1>2; 1=1>2=2
    goals:
    A: B.2=0
    B: B.2!=0
    find decision_matrix and x0
    '''
    sim_spec = mk_spec_for_test1()
    run_test_and_show(sim_spec, mdbg=True)

    mcmc, losses = sm.test_mcmc(90, sim_spec, mdbg=False, edbg=False)
    sim_spec = sm.update_spec(sim_spec, mcmc, idx=-1, side="A", dbg=False)
    print("\nsolution:", sim_spec['agents']['A'])
    factors = run_tests_and_collect_factors(30, sim_spec)



if __name__ == "__main__":
    test1()
    
