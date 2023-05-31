import numpy as np
from functools import reduce
from collections import OrderedDict


def choice_optimal(policies):
    ordered_policies = OrderedDict()
    keys = list(policies.keys())

    def expectation(entry_key):
        return(sum(
            map(lambda x: x[0]*x[1], policies[entry_key])))
    keys.sort(key=expectation, reverse=False)

    for key in keys:
        ordered_policies[key] = (expectation(key), policies[key])

    return(ordered_policies)

    # policies[x] = [(-1.09, 3.2768000000000016e-05,...)]
    # argmax_{policy} sum([value * p(value) for all values])
    argmax_policy_E = max(policies, key=lambda entry: sum(
        map(lambda x: x[0]*x[1], policies[entry])))
    # maxim = max(policies, key=lambda x: max(
    #     policies[x], key=lambda y: y[0])[0])
    
    # return((argmax_policy_E, policies[argmax_policy_E]))


def get_policies(histories, policies, goals, N):
    '''
    Recursively collect paths to goals.
    histories list like
    [[(s0, p0, a0)], [(s0, p0, a0), (s1, p1, a1)], ...]
    i.e. like [history0, history1, ...]
    
    Usage:
    histories, policies = get_policies(
        [[("start_", 1.0, s0)]],
        {}, [(0, 3), (1, 3)], 900)
    '''
    if N < 0:
        return(histories, policies)
    N -= 1
    if len(histories) == 0:
        print("N:", N)
        return(histories, policies)

    history = histories.pop(0)
    rest_history = histories

    np_history = np.array(history)
    print("history:", history)
    print("np_history:", np_history)
    
    # policy is sequence of actions:
    policy = np_history[:, 0]
    history_states = np_history[:, -1]
    print("history_states:", history_states)
    history_probs = np_history[:, 1]
    print("history_probs:", history_probs)

    s = history_states[-1]
    for a in actions(s):
        for s1, prob in result(a, s):
            # print("s1:", s1)
            # if s1 not in list(history_states[:-1]):
            if s1 in goals:

                # name for dict key:
                policy_states = "".join(list(policy)+[a])

                # collect probabilities of path:
                probs = reduce(lambda acc, p: acc*p,
                               list(history_probs), prob)

                if policy_states not in policies:
                    # create new entry for policy:
                    policies[policy_states] = [
                        (reward(list(history_states)+[s1]),
                         probs, history+[(a, prob, s1)])]
                else:
                    # add state sequence to existing set of actions:
                    # (with is policy):
                    policies[policy_states].append(
                        (reward(list(history_states)+[s1]), probs,
                         history+[(a, prob, s1)]))
            else:
                rest_history.append(history+[(a, prob, s1)])

    return(get_policies(rest_history, policies, goals, N))


def explore_policy(aseq, states, probs):
    if len(aseq) == 0:
        return([(states, probs)])
    
    afirst, arest = aseq[0], aseq[1:]
    histories = []
    histories.extend(sum([
        explore_policy(arest, states+[s1], probs+[prob])
        for s1, prob in result(afirst, states[-1])], []))
    return(histories)

            
def reward(states, r=-0.04):
    
    rewards = np.array(
        [[0, 0, 0, 0, 0, 0],
         [0, r, r, r, 1., 0],
         [0, r, 0, r, -1., 0],
         [0, r, r, r, r, 0],
         [0, 0, 0, 0, 0, 0]])
    return(reduce(lambda acc, s: acc+rewards[s], states, 0))


def actions(s):
    '''Return allowable actions for each state `s`.'''
    # left, right, top, down:
    if s not in [(1, 4), (2, 4)]:
        return(["l", "r", "u", "d"])
    else:
        return ["exit"]

    env = np.array(
        [[[0., 1., 0., 1.],
          [1., 1., 0., 0.],
          [1., 1., 0., 1.],
          [0., 0., 0., 0.]],
         
         [[0., 0., 1., 1.],
          [0., 0., 0., 0.],
          [0., 0., 1., 1.],
          [0., 0., 0., 0.]],
         
         [[0., 1., 1., 0.],
          [1., 1., 0., 0.],
          [1., 1., 1., 0.],
          [1., 0., 1., 0.]]])

    # print("s:", s)
    return([["l", "r", "u", "d"][idx] for idx, ind in enumerate(env[s])
            if ind > 0.])


def result(action, state):
    '''return all states that lead from `state`
    with `action` applied. (i.e. return support of dist)'''

    # walls:
    env = np.ones((5, 6)).astype(np.bool)
    env[0, :] = False * env[0, :]
    env[-1, :] = False * env[-1, :]
    env[:, 0] = False * env[:, 0]
    env[:, -1] = False * env[:, -1]
    env[2, 2] = False
    # env[1, 4] = False
    # env[2, 4] = False

    # actions deltas:
    ij = {"l": (0, -1),
          "r": (0, 1),
          "u": (-1, 0),
          "d": (1, 0)}

    # each action has stochastic outcome:
    noises = {"l": ["u", "d"],
              "r": ["u", "d"],
              "u": ["l", "r"],
              "d": ["l", "r"],
              "exit": []}

    def get_support(action, probability):
        if action == "exit":
            return (state, 1.)
        else:
            di, dj = ij[action]
            if env[(state[0]+di, state[1]+dj)]:
                return(((state[0]+di, state[1]+dj), probability))
            else:
                return((state, probability))

    # main action:
    support = []
    support.append(get_support(action, 0.8))

    for noise_action in noises[action]:
        new_state, prob = get_support(noise_action, 0.1)

        # in case state unchenged several times
        # (ex: in case of two walls):
        # if (new_state, prob) not in support:
        support.append((new_state, prob))
    return(support)


def test(s0=(2, 3)):
    
    s0_actions = actions(s0)

    # start with s0 and its actions results:
    histories = [[("start_", 1.0, s0)]]
    policies = {}
    for k in range(24):
        histories, policies = get_policies(
            histories,
            policies, [(1, 4), (2, 4)], 900)
    
    print("policies:")
    print(policies)
    # print("histories:")
    # for history in histories:
    #     print(history)
    optimal_policy = choice_optimal(policies)
    # print("optimal_policy:", optimal_policy)
    print("optimal_policy:")
    for policy in optimal_policy:
        print()
        print(policy)
        print(optimal_policy[policy])
    print("len optimal_policy: ", len(optimal_policy))
    # print(optimal_policy["start_rruur"])


if __name__ == "__main__":
    # s = (2, 3)
    # print("actions %s:" % str(s), actions(s))
    # a = "l"
    '''
    for i in range(1, 4):
        for j in range(1, 5):
            s = (i, j)
            print()
            for a in ["l", "r", "u", "d"]:
                print('result "%s" %s:' % (a, str(s)), result(a, s))
    '''
    # print("reward:", reward([(1, 0), (0, 0)]))
    # test(s0=(3, 1))
    # history = explore_policy(list('u'), [(2, 0)], [1.0])
    a_seq = list('uurrr')
    history = explore_policy(a_seq, [(2, 0)], [1.0])
    print("actions:", a_seq)
    print("len history:", len(history))
    print("history[:10]:")
    print(history[:10])
    for entry in history:
        if entry[0] == [(3, 1)]*6:
            print(entry)
    
    print("history[-1]:", history[-1])

    calc_prob = lambda entry: reduce(lambda acc, x: acc*x, entry[1])
    print("max prob:", max(history, key=calc_prob))
    print("e:", reduce(lambda acc, entry: acc+reward(entry[0])*calc_prob(entry),
                       history, 0))
    '''
    policy = {
        'start_rrlr':
        [(-1.04, 0.051200000000000016,
          [('start_', 1.0, (3, 1)), ('r', 0.8, (3, 2)),
           ('r', 0.8, (3, 3)), ('l', 0.1, (2, 3)), ('r', 0.8, (2, 4))])],
        'start_rddu':
        [(-1.04, 0.006400000000000002,
          [('start_', 1.0, (3, 1)), ('r', 0.8, (3, 2)), ('d', 0.1, (3, 3)),
           ('d', 0.1, (3, 4)), ('u', 0.8, (2, 4))])]}
    print(policy)
    print(choice_optimal(policy))
    '''
