import torch
import torch.distributions as dist
import pyro
import pyro.distributions as pdist
from pyro.poutine.escape_messenger import EscapeMessenger
from pyro.poutine import NonlocalExit
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.runtime import effectful
from pyro.poutine.runtime import _PYRO_STACK

from collections import OrderedDict
from itertools import combinations

import torch.distributions.constraints as constraints
from torch.functional import F
from pyro.infer import MCMC, NUTS

import matplotlib.pyplot as plt


class EventsHandler(TraceMessenger):
    '''In order to affect scores the `factor` attribute will be given
    to context of each event, which could be changed by event registred
    handler. This factor will affect the `pyro.factor` call during exiting
    the context.

    Has predefined exit event with same name. So do not use it
    in yours handlers.
    '''
    def __init__(self, init_factor, *args, **kwargs):
        # {(ename, ehandler)}:
        self.estack = OrderedDict()
        if type(init_factor) != torch.Tensor:
            init_factor = torch.tensor(init_factor).type(torch.float)
        self.init_factor = init_factor
        self.final_factors = []

        # python self will allow to do that:
        self.eregister("exit")(self._goal_exit)
        self.eregister("force_exit")(self._force_exit)
        TraceMessenger.__init__(self, *args, **kwargs)

    def get(self, type, msgs_cond=None):
        '''To get primitives of given type from the `self.trace`
        - ``msgs_cond``-- function::msg->Bool,
        applied only if type is matched'''

        msgs = self.trace.nodes.copy()

        def test(msg):
            cond = msg["type"] == type
            if not cond:
                return False
            if msgs_cond is not None:
                cond = cond and msgs_cond(msg)
            return cond
        sites_names = list(filter(
            lambda msg_key: test(msgs[msg_key]), msgs))
            
        sites_msgs = [msgs[key] for key in sites_names]
        return (sites_names, sites_msgs)

    # def __enter__(self):
    #     return TraceMessenger.__enter__(self)

    def __exit__(self, exc_type, exc_value, traceback):
        '''The scores will be only computed here (by pyro.factor),
        at the end of the `with` statemnt.'''
        # print("exiting")

        if exc_type is None:
           
            enames, events = self.get("event")
            onames, observations = self.get("observation")
            
            if len(list(observations)) == 0:
                # the events used when there is no observation!
                onames = enames
                observations = events

            last_observation = observations[-1]
            state_context = last_observation["value"]
            if state_context["dbg"]:
                print("FROM __exit__:")
                print("events:", *[str(e["value"])+"\n" for e in events])
                print("enames:", enames)
                print("\nobservations:",
                      *[str(o["value"])+"\n" for o in observations])
                print("onames:", onames)

            score = state_context["factor"]
            if state_context["dbg"]:
                print("score:", score)
                print("END FROM __exit__:")
            self._trigger("exit", state_context, last_observation, self.trace)

        elif exc_type == NonlocalExit:
            
            msg = exc_value.site
            econtext = msg["value"]
            if econtext["dbg"]:
                print("from __exit__")
                print("exc_value:")
                print(exc_type)
                print("msg[name]: ", msg["name"])
                print("econtext: ", econtext)
                score = econtext["factor"]
                print("score:", score)
            self._trigger("exit", econtext, msg, self.trace)
            TraceMessenger.__exit__(self, None, None, None)
            # to suppress the exception:
            return True
        return TraceMessenger.__exit__(self, exc_type, exc_value, traceback)

    def _force_exit(self, econtext, msg, trace):
        
        if econtext["dbg"]:
            print("from _force_exit ehandler: exiting")

        msg["done"] = True
        msg["stop"] = True

        # import pdb; pdb.set_trace()
        def cont(m):
            # to protect memory recursion:
            econtext["msg"] = None
            # m["observation_state_context"] = econtext
            # MY:
            # print("continuation site m:", m)
            raise NonlocalExit(m)
        msg["continuation"] = cont
        # raise NonlocalExit("goal achived")
        return econtext

    def _goal_exit(self, econtext, msg, trace):
        '''Will be called at the end of `with ehandler` statements.
        Do not use observe and trigger effectful computations here,
        only pyro.sample (factor) or pyro.param (they do not been watched
        by `ehandler._process_message`).
        Added final factors to `self.final_factors` to collect the losses
        during mcmc training.
        '''
        if econtext["dbg"]:
            print("from goal_exit")
        pyro.factor("event_error_factor", econtext["factor"])
        # self.final_factors.append(econtext["factor"])
        
    def observe(self, oname, events=[], state_context={}, dbg=False):
        '''Same as the `effectful_observe` but with the keyword args given
        which required for the `effectful` decorator
        (see it source in 'pyro/poutine/runtime.py').
  
       - ``oname`` -- name of current observation.
        Will be given to msg["name"]. Each observation name will be used in
        `trace` to store its `msg`.

        - ``events`` -- events to be checked. 
        Order here is not important. Will be given to
        `observation_state_context` as "events_to_check" dict keys.

        - ``observation_state_context`` -- what will be given to
        each called event. Will be given to msg["value"].
        If has `factor` inside it will be used as `init_factor`
        '''
        observation_state_context = state_context
        events_ordered = OrderedDict()
        for ename in events:
            events_ordered[ename] = None
        observation_state_context["events_to_check"] = events_ordered
        observation_state_context["dbg"] = dbg

        # supposed that `observe` not used inside
        # (use `_observe` if needed)
        if "factor" not in observation_state_context:
            observation_state_context["factor"] = self.init_factor
        return self.effectful_observe(
            name=oname, obs=observation_state_context,
            no_context_name=oname,
            no_context_obs=observation_state_context)

    @effectful(type="observation")
    def effectful_observe(self, name=None, obs=None,
                          no_context_name=None, no_context_obs=None):
        '''
        This function will be called by an user somewhere in the model
        (to trigger the events in `obs["events_to_check"]`).
        (see also the `effectful_trigger` function description and its args)
        ''' 
        # this code will be running only outside the `with` context

        # oname = name
        ocontext = no_context_obs if obs is None else obs
        msg = {"warning": "no msg since running outside the context!"}
        trace = self.trace if hasattr(self, "trace") else {}
        
        self._observe(ocontext, msg, trace)

    def trigger(self, ename, econtext, dbg=False):
        '''Same as `effectful_trigger` but with the keyword args given
        which required for the `effectful` decorator
        (see it source in 'pyro/poutine/runtime.py').

        - ``econtext`` -- If has `factor` inside it will be used
        as `init_factor`.
        '''
        econtext["dbg"] = dbg

        # supposed that `trigger` not used inside
        # (use `_trigger` if needed)
        if "factor" not in econtext:
            econtext["factor"] = self.init_factor
        
        return self.effectful_trigger(name=ename, obs=econtext,
                                      no_context_name=ename,
                                      no_context_obs=econtext)
        
    @effectful(type="event")
    def effectful_trigger(self, name=None, obs=None,
                          no_context_name=None, no_context_obs=None):
        '''
        This function will be called by an user somewhere in the model
        (to trigger the events in `obs["events_to_check"]`).
        The `name` and `obs` will be given by `effectful`
        (as `ename` and `context` accordingly) to msg and then
        to `apply_stack` and finelly to `_process_message`
        of each Messenger in the `_PYRO_STACK`.

        The code below will only be used if this function called outside
        the context (when `_PYRO_STACK` is empty).
        
        In the present of the context the call will be same but trace and
        msg will be given in the _observe econtext argument.
        
        Introducing this effectful instead of using pyro.param allow as
        avoid intervening with some pyro.inference algorithms.

        - ``no_context_*`` -- params used in case there is no context.
        Since pyro.effectful will pop up the name and obs from given
        kwargs, and do not restor them after no_context detected,
        it run wrapped without them (bug!).
        '''
        # this code will be running only outside the `with` context

        ename = no_context_name if name is None else name 
        econtext = no_context_obs if obs is None else obs
        msg = {"warning": "no msg since running outside the context!"}
        trace = self.trace if hasattr(self, "trace") else {}

        self._trigger(ename, econtext, msg, trace)

    def _observe(self, observation_state_context, msg, trace):
        '''As opposite to the `observe` method above this function
        will be called inneraly during prociding of a `_PYRO_STACK`.
        See the `_process_message` func below.'''
        events = observation_state_context["events_to_check"]
        for ename in filter(
                lambda ename: ename in events,
                self.estack):
            self._trigger(ename, observation_state_context, msg, trace)
            
    def _trigger(self, ename, econtext, msg, trace):
        '''As opposite to the `trigger` method above this function
        will be called inneraly during prociding of a `_PYRO_STACK`.
        See the `_process_message` func below.'''

        if ename in self.estack:
            econtext = econtext if econtext is not None else {}
            if "ename" not in econtext:
                econtext["ename"] = ename
            self.estack[ename](econtext, msg, trace)
        else:
            # raise ValueError("no such event: %s" % ename)
            print("no such event: %s" % ename)

    def _process_message(self, msg):
        # each observation name will be used in trace to store its msg
        # everywhere below the python dict obj is changeable is supposed:
        # (so there is no chains like `context = trigger(context)`)
        # and hence each trigger/event handler can change the msg.
        # (so no msg.copy() used for events handlers)
        # we could call self.stack[ename](econtext) in trigger
        # func but in that case there will be no msg and trace.
        
        if msg["type"] == "event":
            ename = msg["name"]
            econtext = msg["value"]
            # to protect memory recursion:
            # msg["value"] = None

            # econtext["msg"] = msg
            # econtext["trace"] = self.trace
            self._trigger(ename, econtext, msg, self.trace)

            # just in case
            # msg["value"] = econtext
            self.trace.add_node(msg["name"], **msg.copy())

        elif msg["type"] == "observation":
            observation_state_context = msg["value"]
            # to protect memory recursion:
            # msg["value"] = None

            # observation_state_context["msg"] = msg
            # observation_state_context["trace"] = self.trace
            self._observe(observation_state_context, msg, self.trace)
            if observation_state_context["dbg"]:
                print("events:", observation_state_context["events_to_check"])
          
            # just in case
            # msg["value"] = econtext
            self.trace.add_node(msg["name"], **msg.copy())

    def eregister(self, name):
        '''register event with name ``name`` in
        ``self.estack`` or call it with args (context)
        if it already been registred.
        
        Ex::
           events = EventsHandler()
           
           @events.eregister("jump")
           def f(context):
              msg["up"] = True
        '''

        def wrapper(ehandler):
            if name not in self.estack:
                self.estack[name] = ehandler
            else:
                def trigger_event(econtext, msg, trace):
                    if "ename" not in econtext:
                        econtext["ename"] = name
                    return self.estack[name](econtext, msg, trace)
                return trigger_event
        return wrapper


def mk_ehandler(init_factor, goalAtest, goalBtest, scores):
    '''
    # Example of EventsHandler usage for simple decision problem.

    Supposing that (x0, y0) looks as:
    ::
       x0 = torch.tensor([3, 3, 0, 0, 4])
       y0 = torch.tensor([0, 1, 4,  4, 0])

    - ``init_factor`` -- will be given to as init to all in
    the `with self` context.

    - ``goal{A,B}test`` -- is func of x, y.
    - ``scores`` -- list contained from dicts of conditions
    to change the factor with, each of with contain the attributes:
       1. "test" - function :: (goalA, goalB) -> Bool 
    describing when to change facotor (True) or not (False).
       2. "once" - if test shuld be applied only once in all
    computation history (i.e. inside `with self` context).
       3. "factor" - value at which `econtext["factor"]` will be increased
    additively.
       4. "exit" - if True will exiting from farther computation
    by raising the `NonlocalExit` error.
    '''
    
    ehandler = EventsHandler(init_factor)

    @ehandler.eregister("goalA")
    def goalA(econtext, msg, trace):
        x = econtext["x"]
        y = econtext["y"]
        goal = goalAtest(x, y)
        # goal = y[1] <= 0 and y[2] <= 0 and x[-1] > 2
        # if goal:
        #     pyro.factor("Ax", +1000)
        if econtext["dbg"]:
            print("goalA:", goal)

        econtext["events_to_check"]["goalA"] = goal
        # msg[ename] = goal
        return econtext

    @ehandler.eregister("goalB")
    def goalB(econtext, msg, trace):
        '''Supposing:
        x0 = torch.tensor([3, 3, 0, 0, 4]),
        y0 =   torch.tensor([0, 1, 4,  4, 0])
        '''       
        x = econtext["x"]
        y = econtext["y"]
        goal = goalBtest(x, y)
        # goal = x[1] <= 0 and x[-1] <= 0 and y[3] >= 2
        if econtext["dbg"]:
            print("goalB:", goal)
        # if goal:
        #     pyro.factor("Ax", -1000)
        #     # factor("Ay", +1000)
        econtext["events_to_check"]["goalB"] = goal
        # msg[ename] = goal
        return econtext

    @ehandler.eregister("goalAoverB")
    def goalAoverB(econtext, msg, trace):
        '''Solver choose side A'''
        
        # oname = msg["name"]
        events_to_check = econtext["events_to_check"]
        goalAres = events_to_check["goalA"]
        goalBres = events_to_check["goalB"]
        onames, omsgs = ehandler.get(
            "observation",
            (lambda msg: "goalA" in msg["value"]["events_to_check"]
             and "goalB" in msg["value"]["events_to_check"]))

        goal_triggered = False
        for score in scores:
            # usage of trace:
            if score["once"]:
                # to check if `score["test"]` been observed
                # in computation history:
                been_observed = len(list(filter(
                    lambda msg: (score["test"](
                        msg["value"]["events_to_check"]["goalA"],
                        msg["value"]["events_to_check"]["goalB"])),
                    omsgs))) != 0
            else:
                been_observed = False
            if not been_observed:
                if score["test"](goalAres, goalBres):
                    if score["exit"]:
                        goal_triggered = True
                        econtext["factor"] = score["factor"]
                        break
                    if econtext["factor"]+score["factor"] < 0:
                        econtext["factor"] += score["factor"]
                    else:
                        econtext["factor"] = 0
        '''
        # usage of trace:
        # not goalA and not goalB been observed in computation history:
        notA_and_notB_been_observed = len(list(filter(
            lambda msg: (not msg["value"]["events_to_check"]["goalA"]
                         and not msg["value"]["events_to_check"]["goalB"]),
            omsgs))) != 0
        notA_and_B_been_observed = len(list(filter(
            lambda msg: (not msg["value"]["events_to_check"]["goalA"]
                         and msg["value"]["events_to_check"]["goalB"]),
            omsgs))) != 0
        '''
        '''
        # collect factors and apply it only to the end
        # since msg.copy used in trace.add_node(msg["value"], msg.copy())
        # (called separately in mcmc and others)
        # we need to collect factor until the end!
        goal_triggered = False
        if goalA and not goalB:
            econtext["factor"] = 0
            goal_triggered = True
        elif not goalA and goalB:
            if not notA_and_B_been_observed:
                econtext["factor"] -= 1000 
        elif not goalA and not goalB:
            if not notA_and_notB_been_observed:
                
                if econtext["factor"]+100 < 0:
                    econtext["factor"] += 100
                else:
                    econtext["factor"] = 0
            else:
                pass
                # print("notA_and_notB_been_observed!")
        # goal_triggered = False
        '''
        if econtext["dbg"]:
            print("goalAoverB:", goal_triggered)

        econtext["events_to_check"]["goalAoverB"] = goal_triggered
        return econtext

    @ehandler.eregister("exit?")
    def exit(econtext, msg, trace):
        '''always  call it in the last observation in the end of the
        model and not earlier since it event is  only one which call
        the `pyro.factor` (the `pyro.factor` must be called only
        once in the model).
        '''
        
        if econtext["events_to_check"]["goalAoverB"]:
            econtext["events_to_check"]["exit?"] = True
            if econtext["dbg"]:
                print("from exit handler: exiting")
            # 
        
            msg["done"] = True
            msg["stop"] = True

            # import pdb; pdb.set_trace()
            def cont(m):
                # to protect memory recursion:
                econtext["msg"] = None
                # m["observation_state_context"] = econtext
                # MY:
                # print("continuation site m:", m)
                raise NonlocalExit(m)
            msg["continuation"] = cont
            # raise NonlocalExit("goal achived")
        return econtext
    return ehandler


def show_model_trace(ehandler, keys):
    print("\ntrace:")
    for key in keys:
        if key in ehandler.trace.nodes:
            if "A" in key:
                print("\n"+key[:2])
                A = ehandler.trace.nodes[key]["value"]
                A = F.normalize(A.T, p=1, dim=0)
                print(A)
            else:
                print("\n"+key)
                print(ehandler.trace.nodes[key]["value"])
    print("\nevent_error_factor:")
    factor = ehandler.trace.nodes["event_error_factor"]
    factor_value = factor["value"]
    factor_log = factor["fn"].log_prob(factor_value)
    print(factor_log)
    

def show_model_states_context(
        ehandler, attrs_keys,
        state_key=-1,  observation_name=None):
    '''
    # TODO: test.
    - ``state_key`` if None, then mean values will be shown,
    if int or -1 then will show the according value.'''

    observations = ehandler.get(
        "observation", (lambda msg: msg["name"] == observation_name)
        if observation_name is not None else None)
    print("\nobservations state_context::")
    if len(observations[1]) == 0:
        print("no observations to show")
        return
    if state_key is None:
        for key in attrs_keys:
            # check if key in ["value"]
            # or directly in observation object
            if key in observations[1][-1]["value"]:
                if type(observations[1][-1]["value"][key]) == torch.Tensor:
                    print("\nmean "+key)
                    print(torch.mean(torch.cat(
                        [obs["value"][key].unsqueeze(0)
                         for obs in observations[1]]), 0))
            elif key in observations[1][-1]:
                if type(observations[1][-1][key]) == torch.Tensor:
                    print("\nmean "+key)
                    print(torch.mean(torch.cat(
                        [obs[key].squeeze(0) for obs in observations[1]]), 0))
    else:
        last_state_context = observations[1][state_key]["value"]
        for key in attrs_keys:
            if key in last_state_context:
                print("\n"+key)
                print(last_state_context[key])
            elif key in observations[1][state_key]:
                print("\n"+key)
                print(observations[1][state_key][key])
        

def collect_mcmc_result(mcmc, idx=-1, dbg=True):
    '''To collect the results of  mcmc.
    - ``idx`` -- index of result to be collected. '''

    result = mcmc.get_samples()
    test_spec = {}
    for key in result:
        if "A" in key:
            A = result[key][idx]
            A = F.normalize(A.T, p=1, dim=0)
            test_spec[key[:2]] = A
            
            if dbg:
                print("\n"+key[:2])
                print(A)
        else:
            test_spec[key] = result[key][idx]

            if dbg:
                print("\n"+key)
                print(test_spec[key])
    return test_spec


def get_utypes(sim_spec, idx):
    sample_space = list(combinations(
        sim_spec["possible_types"], sim_spec["chosen_count"]))
    return sample_space[idx]


def update_utypes(sim_spec, test_spec, side):
    '''
    Update `sim_spec` from the `test_spec`.

    - ``side`` -- either A or B.
    '''
    if ("types" not in sim_spec["agents"][side]["units"]
        or ("types" in sim_spec["agents"][side]["units"]
            and sim_spec["agents"][side]["units"]["types"] is None)):
        chosen = test_spec[
            side.lower() + "_utypes"].type(torch.long)
        sim_spec["agents"][side]["units"]["types"] = get_utypes(
            sim_spec["agents"][side]["units"], chosen)
    return sim_spec


def update_spec(sim_spec, mcmc, idx=-1, side="A", dbg=False):
    '''Update `sim_spec` from mcmc result.'''

    test_spec = collect_mcmc_result(mcmc, idx=-1, dbg=dbg)
    if dbg:
        print("\ntest_spec:")
        print(test_spec)
    sim_spec = update_utypes(sim_spec, test_spec, side)
    value = "x" if side == "A" else "y"

    sim_spec["agents"][side]["decision_matrix"] = test_spec["A"+value]
    if value+"0" in test_spec:
        sim_spec["agents"][side]["units"]["counts"] = test_spec[value+"0"]
    if dbg:
        print("updated sim_spec:")
        print(sim_spec["agents"][side])
    return sim_spec


def model(model_spec, losses=None, mdbg=False, edbg=False):
    '''Should only be run in `with ehandler` context'''
    T = model_spec["T"]
    dt = model_spec["dt"]
    aSpec = model_spec["agents"]["A"]
    bSpec = model_spec["agents"]["B"]
    U = model_spec["U"]

    init_factor = model_spec["init_factor"]
    ehandler = mk_ehandler(
        init_factor, aSpec["goal"], bSpec["goal"],
        model_spec["scores"])
    # scores = []
    with ehandler:
        x0, y0, Ua, Ub, Ax, Ay = init_model(aSpec, bSpec, U, ehandler, mdbg)
        
        run_model(x0, y0, Ua, Ub, T, dt, Ax=Ax, Ay=Ay,
                  ehandler=ehandler,
                  mdbg=mdbg, edbg=edbg)
        # all attempts of mcmc will be collected here:
        # scores.extend(filter(lambda x: x["name"]=='event_error_factor',
        #                      ehandler.get("sample")[1]))
    if losses is not None:
        losses.extend(ehandler.final_factors)
    return(ehandler)


def init_model(aSpec, bSpec, U, ehandler, mdbg):
    '''To Initialize the model from its spec.'''
    # initiate_sim:
    aUnits = aSpec["units"]
    bUnits = bSpec["units"]
    aUtypes, x0 = mk_units(aUnits, "a_utypes", "x0", ehandler)
    bUtypes, y0 = mk_units(bUnits, "b_utypes", "y0", ehandler)
    if mdbg:
        print("aUtypes:")
        print(aUtypes)
        print("x0:", x0)
        print("bUtypes:")
        print(bUtypes)
        print("y0:", y0)
        
    Ua, Ub = make_U(U, aUtypes, bUtypes, mdbg)
    Ax = aSpec["decision_matrix"]
    Ay = bSpec["decision_matrix"]

    Ax, Ay = mk_decisions(Ax, Ay, x0, y0, mdbg)

    return (x0, y0, Ua, Ub, Ax, Ay)


def mk_units(units_spec, utypes_name, ucounts_name, ehandler):
    '''Will create/sample units types, if they not given in units_spec,
    and units counts, for the same reason.
    If the "types" not given then also and "count" should not.
    - ``units_spec`` -- if have no "types" attribute, must contain
    the "possible_types" one, which will describe possible types
    (the sample space) for `utypes`.

    - ``utypes_name`` -- used for `pyro.sample` name of `utypes`.
    (ex: "aU").

    - ``ucounts_name`` -- used for `pyro.sample` name of `ucounts`.
    (ex: "x0").
    '''
    if "types" not in units_spec or units_spec["types"] is None:
        assert "possible_types" in units_spec
        assert units_spec["possible_types"] is not None
        assert "chosen_count" in units_spec
        assert units_spec["chosen_count"] is not None
        assert units_spec["chosen_count"] <= len(units_spec["possible_types"])

        ptypes = units_spec["possible_types"]
        n = units_spec["chosen_count"]

        sample_space = list(combinations(ptypes, n))

        '''
        if type(ptypes) is list:
            ptypes = torch.tensor(ptypes).type(torch.long)
        bptypes = torch.tensor([False]*n).type(torch.bool)
        bptypes[ptypes] = True
        utypes = torch.arange(0, n)[
            (bptypes == True).logical_and(chosen >= 0.5)]
        '''
        chosen = pyro.sample(
            utypes_name, pdist.Uniform(0, len(sample_space)))
        utypes = sample_space[chosen.type(torch.long)]

        if ehandler is not None:
            # another usage of observe effectful. No event, just context:
            ehandler.observe(
                "sample_space_of_"+utypes_name,
                state_context={"utypes": utypes},
                dbg=False)

            if len(utypes) == 0:
                # exit with no father work (by raising `NonlocalExit`)
                ehandler.trigger("force_exit", {"factor": -500}, True)
                # should not been here in the `with` context
                assert False
    else:
        utypes = units_spec["types"]
    
    if "counts" not in units_spec or units_spec["counts"] is None:
        # n = len(utypes)
        # print("--------counts n-------")
        # print(n)
        # with pyro.plate(ucounts_name+"_plate", size=n):
        assert "max_count" in units_spec
        assert units_spec["max_count"] is not None
        max_count = units_spec["max_count"]
        if type(max_count) == list:
            max_count = torch.tensor(max_count).type(torch.float)

        if "min_count" in units_spec and units_spec["min_count"] is not None:
            min_count = units_spec["min_count"]
            if type(min_count) == list:
                min_count = torch.tensor(min_count).type(torch.float)

            if type(max_count) == torch.Tensor and type(min_count) == torch.Tensor:
                assert (min_count < max_count).all()
            else:
                assert min_count < max_count
        else:
            min_count = (
                torch.zeros_like(max_count)
                if type(max_count) == torch.Tensor else 0.)
        if type(max_count) == torch.Tensor:
            ucounts = pyro.sample(
                ucounts_name, pdist.Uniform(
                    min_count, max_count))
        else:
            # if types given they size will be used for
            # counts vector size.
            # Otherwise chosen_count param will be used:
            if "types" in units_spec and units_spec["types"] is not None:
                n = len(units_spec["types"])
            else:
                assert units_spec["chosen_count"] is not None
                if ("possible_types" in units_spec
                    and units_spec["possible_types"] is not None):
                    assert (units_spec["chosen_count"]
                            <= len(units_spec["possible_types"]))
                n = units_spec["chosen_count"]

            ucounts = pyro.sample(
                ucounts_name, pdist.Uniform(
                    min_count*torch.ones(n), max_count*torch.ones(n)))

        if ehandler is not None:
            # another usage of observe effectful. No event, just context:
            ehandler.observe(
                "counts_of_"+ucounts_name,
                state_context={"ucounts": ucounts},
                dbg=False)

            # print(ucounts)
    else:
        ucounts = units_spec["counts"]
    
    return (utypes, ucounts)


def make_U(U, aUnits, bUnits, dbg):
    '''Return efficiency matrix only for
    given units A and B.'''
    
    if dbg:
        print("U: ")
        print(U)

    idxsA = list(set(aUnits))
    idxsB = list(set(bUnits))
    Ua = torch.cat(list(
        map(lambda x: x.unsqueeze(0),
            [U[a].index_select(0, torch.tensor(idxsB)) for a in idxsA])))
    Ub = torch.cat(list(
        map(lambda x: x.unsqueeze(0),
            [U[a].index_select(0, torch.tensor(idxsA)) for a in idxsB])))

    # U set by U[k][:] like k against [:] (all)
    # but we need it to pairwise mult with
    # A[:][k] decis. matrix for each k (how many of k attack [:] (all))
    # so We use U1.T here:
    Ua = Ua.T
    Ub = Ub.T
    '''
    idxs = list(set(list(aUnits)+list(bUnits)))
    U1 = torch.cat(list(
        map(lambda x: x.unsqueeze(0),
            [U[a].index_select(0, torch.tensor(idxs)) for a in idxs])))
    '''
    if dbg:
        print("Ua=U[aUnits][bUnits].T=U[%s][%s].T:"
              % (str(aUnits), str(bUnits)))
        print(Ua)
        print("Ub=U[bUnits][aUnits].T=U[%s, %s].T:"
              % (str(bUnits), str(aUnits)))
        print(Ub)
        
    return (Ua, Ub)


def mk_decisions(Ax, Ay, x0, y0, mdbg):
    '''To sample (Ax_T, Ay_T) then transpose them if they not given else just
    return what been given (i.e. Ax, Ay). Will return
    normalized (Ax, Ay) in that case. (if they is they should be alredy)'''

    if Ax is None:
        # why `_T` is used here see description in `elbo_guide`:
        Ax_T_shape = (len(x0), len(y0))  # should be (len(y0), len(x0)) for Ax
        Ax_T = pyro.sample("Ax_T", pdist.Uniform(
            torch.zeros(Ax_T_shape), torch.ones(Ax_T_shape)))
        Ax = Ax_T.T
    # since mcmc do not used guide and hence simplex constrain:
    # also used here since this not change Ax if it alredy normalized:
    Ax = F.normalize(Ax, p=1, dim=0)
    if mdbg:
        print("Ax:")
        print(Ax)

    if Ay is None:
        Ay_T_shape = (len(y0), len(x0))  # should be (len(x0), len(y0)) for Ay
        # We make use of minmax strategy:
        Ay_T = dist.Uniform(torch.zeros(Ay_T_shape),
                            torch.ones(Ay_T_shape)).sample()
        '''
        Ay_T = pyro.sample("Ay_T", pdist.Uniform(
            torch.zeros(Ay_T_shape), torch.ones(Ay_T_shape)))
        '''
        Ay = Ay_T.T
    # since mcmc do not used guide and hence simplex constrain:
    # also used here since this not change Ax if it alredy normalized:
    Ay = F.normalize(Ay, p=1, dim=0)
    if mdbg:
        print("Ay:")
        print(Ay)
    return (Ax, Ay)


def run_model(x0, y0, Ua, Ub, T, dt, Ax=None, Ay=None, ehandler=None,
              mdbg=False, edbg=False):
    if mdbg:
        print("x0:", x0)
        print("y0:", y0)

    x_prev = x0.detach().clone()
    y_prev = y0.detach().clone()
    x = x0.detach().clone()
    y = y0.detach().clone()
    for t in range(T):
        
        x = x_prev - dt*torch.matmul(torch.mul(Ub, Ay), y_prev)
        # x = torch.where(x < 0, 0.001*torch.ones_like(x), x)
        x = torch.masked_fill(x, x < 0, 0.)
        # x[x < 0] = 0.
        # x = torch.clamp(x, min=0, max=max(x0))
        # this should not happend if x,y>0 and Ax, Ay
        # have been normalized properly:
        # x = torch.where(x > x0, x0, x)
        # x[x > x0] = x0[x > x0][:]
        if mdbg:
            print("x:", x)
        
        y = y_prev - dt*torch.matmul(torch.mul(Ua, Ax), x_prev)
        # y = torch.where(y < 0, 0.001*torch.ones_like(y), y)
        y = torch.masked_fill(y, y < 0, 0.)
        # y[y < 0] = 0.
        # y = torch.clamp(y, min=0, max=max(y0))
        # y = torch.where(y > y0, y0, y)
        # y[y > y0] = y0[y > y0][:]

        # only now We can do that
        x_prev = x
        y_prev = y

        if mdbg:
            print("y:", y)
            print("-------")
        if ehandler is not None:
            ehandler.observe(
                "observation_%d" % t,
                events=["goalA", "goalB", "goalAoverB", "exit?"],
                state_context={
                    "x": x, "y": y,

                    # adding here since pyro.sample only `*_candidates`:
                    "Ua": Ua, "Ub": Ub,

                    "Ax": Ax,
                    # adding here since `torch.sample` (not `pyro` one)
                    "Ay": Ay},
                dbg=edbg)
    if mdbg:
        print("result of x:")
        print(x)
        print("result of y:")
        print(y)
    return (x, y, Ax, Ay)


def elbo_guide(x0, y0, goal, U, T, Ax=None, Ay=None):
    # because simplex used sum(-1) (sum in columns direction)
    # and we need to apply it as $\sum_{i} A_{i, k}=1$ (sum in row direction),
    # we ought to define Ax_T and use it as Ax = Ax_T.T:
    Ax_T_shape = (len(x0), len(y0))  # should be (len(y0), len(x0)) for Ax
    Ay_T_shape = (len(y0), len(x0))  # should be (len(x0), len(y0)) for Ay
    pyro.param("Ax_T", torch.ones(Ax_T_shape), constraint=constraints.simplex)
    pyro.param("Ay_T", torch.ones(Ay_T_shape), constraint=constraints.simplex)


# ########################### tests:

def test_mk_U(dbg=True):
    
    print("make_U([0, 1, 4], [3, 7]):")
    make_U([0, 1, 4], [3, 7], dbg)


def test_mk_model(model_spec, losses, mdbg, edbg):
    ehandler = model(model_spec, losses=losses, mdbg=mdbg, edbg=edbg)
    return ehandler


def test_ehandler(dbg=True):
    ehandler = EventsHandler(0.0)

    @ehandler.eregister("jump")
    def jump(econtext, msg, trace):
        ename = econtext["ename"]
        print("econtext of event %s is:" % (ename))
        print(econtext)

    print("FOR without context:")
    ehandler.trigger("jump", {"last_state": "on surface"}, dbg=dbg)
    print("END FOR without context")
    
    print("FOR with context:")
    pyro.clear_param_store()
    with ehandler:
        ehandler.trigger("jump", {"last_state": "on surface"}, dbg=dbg)
        # print("_PYRO_STACK:")
        # print(_PYRO_STACK)
    print("END FOR with context")


def test_inference(T, dt=0.01, mdbg=True, edbg=True):
    pyro.clear_param_store()
    init_factor = -110.0
    ehandler = mk_ehandler(init_factor)
    print("make_U([0, 1, 4], [2, 3]):")
    U1 = make_U([0, 1, 4], [2, 3])
    print(U1)

    '''
    print("FOR testing without the context:")
    # no exception ounside the context (exception raise in cont() methed!)
    model(torch.tensor([3, 3, 0, 0, 4]).type(torch.float),
          torch.tensor([0, 1, 4,  4, 0]).type(torch.float), U1.T, 3, 0.01,
          ehandler=ehandler, mdbg=mdbg, edbg=edbg)
    print("END FOR testing without the context")
    '''
    print("FOR testing with the context:")
    pyro.clear_param_store()
    ehandler = mk_ehandler(init_factor)
    with ehandler:
        # U set by U[k][:] like k against [:]
        # but we need it to pairwise mult with
        # A[:][k] decis. matrix for each k (how many of k attack [:])
        # so We use U1.T here:
        model(torch.tensor([3, 3, 0, 0, 4]).type(torch.float),
              torch.tensor([0, 1, 4,  4, 0]).type(torch.float), U1.T, T, dt,
              ehandler=ehandler, mdbg=mdbg, edbg=edbg)
    onames, observations = ehandler.get("observation")
    print("last observation name:", onames[-1])
    # print("last_observation msg:", observations[-1])
    last_observation_context = observations[-1]["value"]
    events_to_check = last_observation_context["events_to_check"]
    print("goals:")
    print(events_to_check)
    print("factor:", last_observation_context["factor"])
    print("goalAoverB:", events_to_check["goalAoverB"])
    
    print("END FOR testing with the context")


def test_mcmc(steps, sim_spec, mdbg=False, edbg=False):

    pyro.clear_param_store()
    
    # U1 = make_U([0, 1, 4], [2, 3])
    # x0 = torch.tensor([3, 3, 0, 0, 4]).type(torch.float)
    # y0 = torch.tensor([0, 1, 4,  4, 0]).type(torch.float)
    # ehandler = mk_ehandler(init_factor)
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
            nuts_kernel,
            num_samples=steps,
            # warmup_steps=1,
            num_chains=1,)
    losses = []
    mcmc.run(sim_spec, losses=losses, mdbg=mdbg, edbg=edbg)
    # mcmc.run(x0, y0, U1.T, T, dt, Ay=Ay,  # ehandler=ehandler,
    #          mdbg=mdbg, edbg=edbg)
    # mcmc.run(torch.ones(3))
    if mdbg:
        print(mcmc.summary())
    return(mcmc, losses)


if __name__ == "__main__":
    
    # efficiency matrix:
    # (will be transposed)
    U = 0.1*torch.ones((8, 8))
    U[0][:3] = torch.tensor([0.7, 0.8, 0.3])
    U[1][2:] = torch.tensor([0.3, 0.5, 0.9, 0.9, 0.9, 0.2])
    U[2][:4] = torch.tensor([0.9, 0.9, 0.5, 0.2])
    U[3][2:-1] = torch.tensor([0.9, 0.5, 0.7, 0.7, 0.7])
    U[4][:] = torch.tensor([0.6, 0.7, 0.9, 0.9, 0.5, 0.5, 0.5, 0.2])
    U[5][:] = torch.tensor([0.9, 0.9, 0.6, 0.4, 0.1, 0.1, 0.1, 0.1])
    U[6][4:] = torch.tensor([0.9, 0.9, 0.9, 0.5])
    U[7][4:] = torch.tensor([0.9, 0.9, 0.9, 0.7])

    A_spec = {
        "decision_matrix": None,
        "units": {

            # needed only if there is neither types nor counts given:
            "possible_types": [0, 1, 2, 3, 4],
            "chosen_count": 3,
            # "types": [0, 1, 4],

            # "counts": torch.tensor([3, 3, 4]).type(torch.float)
            # maximal amount of units to sample for any type
            # must be given if "counts" not:
            "max_count": [3, 3, 2],
            ## "min_count": 0,
            ## "max_count": 4,
        },
        "goal": lambda x, y: (y <= 0).all(),
        ## "goal": lambda x, y: y[1] <= 0 and y[2] <= 0
        # "goal": lambda x, y: y[0] <= 0 and y[1] <= 0 and x[-1] > 2
    }
    B_spec = {
        "decision_matrix": None,
        "units": {
            "types": [1, 2, 3],
            "counts": torch.tensor([1, 4,  4]).type(torch.float)
        },
        "goal": lambda x, y: (x <= 0).all() and y[2] >= 2
        # "goal": lambda x, y: x[1] <= 0 and x[-1] <= 0,  # and y[2] >= 2
    }

    sim_spec = {
        "agents": {"A": A_spec, "B": B_spec},

        "T": 30,
        "dt": 0.5,
        # "dt": 0.5,
        "init_factor": -110.0,

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
                "test": lambda goalA, goalB: not goalA and not goalB,
                # only once from all times will factor been used  
                "once": True,
                "factor": 10,
                "exit": False
            }
        ] 
    }
    
    # FOR testing mcmc:
    mcmc, losses = test_mcmc(10, sim_spec, mdbg=False, edbg=False)
    losses_len = len(losses)
    if losses_len > 0:
        losses = torch.cat([
            (torch.tensor(l).unsqueeze(0)
             if type(l) != torch.Tensor else l.unsqueeze(0))
            for l in losses]).numpy()
        print("losses:")
        print(losses_len)
        plt.hist(losses)
        plt.show()
    # print("mcmc.get_samples:")
    # print(mcmc.get_samples())

    '''
    print("\nFOR test original model:")
    ehandler = test_mk_model(sim_spec, losses, False, False)
    show_model_states_context(
        ehandler, ["x", "y", "Ua", "Ub", "Ax", "Ay", "events_to_check"],
        state_key=-1)
    show_model_states_context(
        ehandler, ["utypes"],
        # None here means mean of all states
        state_key=None,
        observation_name="sample_space_of_a_utypes")
    show_model_states_context(
        ehandler, ["ucounts"],
        # None here means mean of all states
        state_key=None,
        observation_name="counts_of_x0")
    print("END FOR test original model")
    '''

    print("\nFOR testing inferenced model:")
    # collect factors:
    sim_spec = update_spec(sim_spec, mcmc, idx=-1, side="A", dbg=False)

    losses = []
    ehandler = test_mk_model(sim_spec, losses, False, False)
    show_model_states_context(
        ehandler, ["x", "y", "Ua", "Ub", "Ax", "Ay", "events_to_check"],
        state_key=-1)
    show_model_states_context(
        ehandler, ["utypes"],
        state_key=-1,
        observation_name="sample_space_of_a_utypes")
    show_model_states_context(
        ehandler, ["ucounts"],
        state_key=-1,
        observation_name="counts_of_x0")

    show_model_trace(ehandler, ["x0", "Ax_T", "y0", "Ay_T"])
    print("\nehanler.final_factors:")
    print(ehandler.final_factors)
    print("ehandler.final_factors:")
    print(losses)
    print("updated sim_spec (solution):")
    print('sim_spec["agents"]["A"]')
    print(sim_spec["agents"]["A"])
    print("\nEND FOR testing inferenced model")

    print("\nFOR factors hist:")
    factors = []
    losses = []
    for i in range(10):
        ehandler = test_mk_model(sim_spec, losses, False, False)
        factor = ehandler.trace.nodes["event_error_factor"]
        factor_value = factor["value"]
        factor_log = factor["fn"].log_prob(factor_value)
        factors.append(factor_log)

    print("factors:")
    print(torch.cat([factor.unsqueeze(0) for factor in factors]))
    print("losses:")
    print(losses)
    
    plt.hist(factors, bins=15)
    plt.title("factors")
    plt.show()
    print("\nEND FOR factors hist")
    # END FOR
    
    # no need for that: get_samples just return what
    # already been done
    # ehandler1 = mk_ehandler()
    # with ehandler1:
    #    mcmc.get_samples()
    # print(ehandler1.get("sample"))
    # print("ehandler.trace.nodes:")
    # print(ehandler.get("sample"))
    # print(mcmc.get_samples()["Ax_T"].shape)
    # print("scores:")
    # print(scores)
    # test_inference(10, 0.5, mdbg=True, edbg=False)
    
    
    '''
    # FOR testing model:
    losses = []
    ehandler = test_mk_model(sim_spec, losses, False, False)
    show_model_last_state_context(
        ehandler, ["x", "y", "Ua", "Ub", "Ay", "events_to_check"])
    show_model_last_state_context(
        ehandler, ["utypes"],
        observation_name="sample_space_with_size_a_utype_sample_space_size")
    show_model_trace(ehandler, ["x0", "Ax_T", "y0", "Ay_T"])
    print("\nehanler.final_factors:")
    print(ehandler.final_factors)
    print("ehandler.final_factors:")
    print(losses)
    # END FOR
    '''
    
    # test_ehandler(dbg=True)
    # test_mk_U()
