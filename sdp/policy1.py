import numpy as np
import matplotlib.pyplot as plt

import policy


class MDP():
    '''Markov Decision Problem'''
    def __init__(self, r=-0.04):
        self.shape = (5, 6)
        self.gamma = 1.

        # rewards:
        rewards = np.array(
            [[0, 0, 0, 0, 0, 0],
             [0, r, r, r, 1., 0],
             [0, r, 0, r, -1., 0],
             [0, r, r, r, r, 0],
             [0, 0, 0, 0, 0, 0]])
                
        self.rewards = rewards
        ix, iy = np.indices(self.shape)

        # states:
        states = np.concatenate([
            np.expand_dims(ix, 2), np.expand_dims(iy, 2)], 2)

        # all states will be flatten with tuple of indexes (as [(1, 2),...]):
        states = states.reshape((self.shape[0]*self.shape[1], 2)).tolist()
        
        # this state will be ignored:
        states = filter(lambda state: state[0] not in [0, 4] and state[1] not in [0, 5], states)
        states = filter(lambda state: state not in [[2, 2], [1, 4], [2, 4]], states)  
        
        states = map(tuple, states)
        self.states = list(states)
        print("states:", self.states)

    def fix_borders(self, U):
        U[2, 3] = 0.
        U[1, 4] = 1.
        U[2, 4] = -1.
        return U

    def reward(self, state):
        return self.rewards[state]


def value_iteration(mdp, steps):
 
    U1 = np.zeros(mdp.shape)
    
    policy1 = np.zeros(mdp.shape, dtype=np.int)
    deltas = []
    for step in range(steps):
        U = U1.copy()
        delta = 0.
        for state in mdp.states:
            actions = policy.actions(state)
            bU = np.array([q_value(mdp, state, action, U)
                           for action in actions])
            # print("bU")
            # print(bU)
            policy1[state] = np.argmax(bU)
            U1[state] = max(bU)

            # norm here is max . abs:
            delta1 = np.abs(U1[state]-U[state])
            if delta1 > delta:
                delta = delta1
                print("delta:", delta)
        # U1 = mdp.fix_borders(U1)
        deltas.append(delta)
    return(U1, list(map(
        lambda row: list(map(lambda elm: actions[elm], row)), policy1)),
           deltas)


def q_value(mdp, state, action, U):
    
    return sum([(mdp.reward(state1)+mdp.gamma*U[state1])*prob
                for (state1, prob) in policy.result(action, state)])


def explore_policy(bs):
    first_bs, rest_bs = bs[0], bs[1:]
    astack = first_bs[0]
    shist, ahist = first_bs[1]
    print("astack:", astack)
    print("shist", shist)
    print("rest_bs:", rest_bs)
    if len(astack) == 0:
        if len(rest_bs) == 0:
            print("should return now")
            return ahist
        else:
            yield ahist
            yield from explore_policy(rest_bs)
    else:
        a, rest_astack = astack[0], astack[1:]
        s, rest_shist = shist[0], shist[1:]
        # s = (state, prob)
        new_shist = policy.result(a, s[0])

        yield from explore_policy(
            # action in state s
            [(rest_astack, (new_shist, ahist+[(a, s)]))]
            +
            ([(astack, (rest_shist, ahist))]+rest_bs
             if len(rest_shist) != 0 else rest_bs))


def test_3x4(steps, r=-0.04):
    mdp = MDP(r=r)
    U, policy, deltas = value_iteration(mdp, steps)
    print("U:")
    print(U)

    print("policy:")
    for pentry in policy:
        print(pentry)
    # print("delta:", delta)
    plt.plot(deltas)
    plt.show()


def test_explore():
    actions = 'uru'
    init_state = ((3, 1), 1.0)
    print("explore_policy '%s' for the init_state %s"
          % (actions, str(init_state)))
    histories = explore_policy(
        [(list(actions), ([init_state], [("start", init_state)]))])
    for hist in list(histories):
        print("action in state s:")
        
        print(hist)
        # print(list(states))


if __name__ == "__main__":
    test_3x4(70, -0.04)
    
    # live is too plesant:
    test_3x4(70, -0.01)
