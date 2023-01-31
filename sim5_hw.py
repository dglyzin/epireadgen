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

import torch.distributions.constraints as constraints
from torch.functional import F
from pyro.infer import MCMC, NUTS


class EventsHandler(TraceMessenger):
    '''In order to affect scores the `factor` attribute will be given
    to context of each event, which could be changed by event registred
    handler. This factor will affect pyro.factor call during exiting
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

        # python self will allow to do that:
        self.eregister("exit")(self._goal_exit)
        
        TraceMessenger.__init__(self, *args, **kwargs)

    def get(self, type):
        '''To get primitives of given type from the `self.trace`'''
        msgs = self.trace.nodes.copy()
        sites_names = list(filter(
            lambda msg_key: msgs[msg_key]["type"] == type, msgs))
        sites_msgs = [msgs[key] for key in sites_names]
        return (sites_names, sites_msgs)

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

    def _goal_exit(self, econtext, msg, trace):
        '''Will be called at the end of `with ehandler` statements.
        Do not use observe and trigger effectful computations here,
        only pyro.sample (factor) or pyro.param (they do not been watched
        by `ehandler._process_message`).
        '''
        if econtext["dbg"]:
            print("from goal_exit")
        pyro.factor("event_error_factor", econtext["factor"])
        
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
        
        '''
        observation_state_context = state_context
        events_ordered = OrderedDict()
        for ename in events:
            events_ordered[ename] = None
        observation_state_context["events_to_check"] = events_ordered
        observation_state_context["dbg"] = dbg
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
        (see it source in 'pyro/poutine/runtime.py').'''
        econtext["dbg"] = dbg
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


def mk_ehandler(init_factor):
    ehandler = EventsHandler(init_factor)

    @ehandler.eregister("goalA")
    def goalA(econtext, msg, trace):
        '''Supposing:
        x0 = torch.tensor([3, 3, 0, 0, 4]),
        y0 =   torch.tensor([0, 1, 4,  4, 0])
        '''
        x = econtext["x"]
        y = econtext["y"]
        goal = y[1] <= 0 and y[2] <= 0 and x[-1] > 2
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
        goal = x[1] <= 0 and x[-1] <= 0 and y[3] >= 2
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
        goalA = events_to_check["goalA"]
        goalB = events_to_check["goalB"]
        onames, omsgs = ehandler.get("observation")

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

        # TODO: collect factors and apply it only to the end
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


def model(x0, y0, U, T, dt, Ax=None, Ay=None, ehandler=None,
          mdbg=False, edbg=False):
    '''
    - ``Ax, Ay`` --  is optional for a test purpose.
    '''
    if Ax is None:
        # why `_T` is used here see description in `elbo_guide`:
        Ax_T_shape = (len(x0), len(y0))  # should be (len(y0), len(x0)) for Ax
        Ax_T = pyro.sample("Ax_T", pdist.Uniform(
            torch.zeros(Ax_T_shape), torch.ones(Ax_T_shape)))
        Ax = Ax_T.T
        # since mcmc do not used guide and hence simplex constrain:
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
        Ay = F.normalize(Ay, p=1, dim=0)
        if mdbg:
            print("Ay:")
            print(Ay)

    if mdbg:
        print("x0:", x0)
        print("y0:", y0)

    x = x0.detach().clone()
    y = y0.detach().clone()

    for t in range(T):
        
        x -= dt*torch.matmul(torch.mul(U, Ay), y)
        x[x < 0] = 0.
        x[x > x0] = x0[x > x0][:]
        if mdbg:
            print("x:", x)

        y -= dt*torch.matmul(torch.mul(U, Ax), x)
        y[y < 0] = 0.
        y[y > y0] = y0[y > y0][:]
        if mdbg:
            print("y:", y)
            print("-------")
        if ehandler is not None:
            ehandler.observe(
                "observation_%d" % t,
                events=["goalA", "goalB", "goalAoverB", "exit?"],
                state_context={"x": x, "y": y},
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


def make_U(aUnits, bUnits):
    '''Return efficiency matrix only for
    given units A and B.'''
    # efficiency matrix:
    U = 0.1*torch.ones((8, 8))
    U[0][:3] = torch.tensor([0.7, 0.8, 0.3])
    U[1][2:] = torch.tensor([0.3, 0.5, 0.9, 0.9, 0.9, 0.2])
    U[2][:4] = torch.tensor([0.9, 0.9, 0.5, 0.2])
    U[3][2:-1] = torch.tensor([0.9, 0.5, 0.7, 0.7, 0.7])
    U[4][:] = torch.tensor([0.6, 0.7, 0.9, 0.9, 0.5, 0.5, 0.5, 0.2])
    U[5][:] = torch.tensor([0.9, 0.9, 0.6, 0.4, 0.1, 0.1, 0.1, 0.1])
    U[6][4:] = torch.tensor([0.9, 0.9, 0.9, 0.5])
    U[7][4:] = torch.tensor([0.9, 0.9, 0.9, 0.7])
    
    print("U: ")
    print(U)
    
    idxs = list(set(list(aUnits)+list(bUnits)))
    return torch.cat(list(
        map(lambda x: x.unsqueeze(0),
            [U[a].index_select(0, torch.tensor(idxs)) for a in idxs])))


def test_mk_model():
    
    print("make_U([0, 1, 4], [2, 3]):")
    U1 = make_U([0, 1, 4], [2, 3])
    print(U1)
    model(torch.tensor([3, 3, 0, 0, 4]),
          torch.tensor([0, 1, 4,  4, 0]), U1.T, 1)


def test_ehandler(dbg=True):
    ehandler = EventsHandler(0.0)

    @ehandler.eregister("jump")
    def jump(econtext, msg, trace):
        ename = econtext["ename"]
        print("econtext of event %s is:" % (ename))
        print(econtext)

    ehandler.trigger("jump", {"last_state": "on surface"}, dbg=dbg)
    
    pyro.clear_param_store()
    with ehandler:
        ehandler.trigger("jump", {"last_state": "on surface"}, dbg=dbg)
        # print("_PYRO_STACK:")
        # print(_PYRO_STACK)


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


def test_mcmc(steps, T=10, dt=0.5, Ay=None, mdbg=False, edbg=False):

    pyro.clear_param_store()

    init_factor = -110.0
    
    U1 = make_U([0, 1, 4], [2, 3])
    x0 = torch.tensor([3, 3, 0, 0, 4]).type(torch.float)
    y0 = torch.tensor([0, 1, 4,  4, 0]).type(torch.float)
    ehandler = mk_ehandler(init_factor)

    def emodel(*args, **kwargs):
        with ehandler:
            model(*args, ehandler=ehandler, **kwargs)
            
    nuts_kernel = NUTS(emodel)
    mcmc = MCMC(
            nuts_kernel,
            num_samples=steps,
            # warmup_steps=1,
            num_chains=1,)

    mcmc.run(x0, y0, U1.T, T, dt, Ay=Ay,  # ehandler=ehandler,
             mdbg=mdbg, edbg=edbg)
    # mcmc.run(torch.ones(3))
    print(mcmc.summary())
    return(mcmc, ehandler)


if __name__ == "__main__":
    mcmc, ehandler = test_mcmc(10, T=10, dt=0.5, Ay=None, mdbg=False, edbg=False)
    # TODO:
    # ehandler1 = mk_ehandler()
    # with ehandler1:
    #    mcmc.get_samples()
    # print(ehandler1.get("sample"))
    print("ehandler.trace.nodes:")
    print(ehandler.get("sample"))
    # test_inference(10, 0.5, mdbg=True, edbg=False)
    # test_ehandler(dbg=True)
