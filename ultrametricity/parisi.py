import numpy as np
import matplotlib.pyplot as plt
import itertools as it


def calc_P(Q, k):
    states = []
    partsps = []
    ps = []
    for a in range(k):
        for b in range(k):
            for c in range(k):
                if a != b and a != c and b != c:
                    state = [a, b, c]
                    if state not in states:
                        cstates = list(it.permutations(state))
                        states.append(cstates)
                        cps = [(Q[a1, b1], Q[a1, c1], Q[b1, c1])
                               for a1, b1, c1 in cstates]
                        ps.append(cps)
                        # ps.append(len(cps))
                        # print("state: ", str(state))
                        # states.append([a, b, c])
                        # q = Q[a, b], Q[a, c], Q[b, c]
                        # if q[0] != q[1] and q[0] != q[2] and q[1] != q[2]:
                        #     print("q: ", str(q))
    '''
    for idx, state in enumerate(states):
        print(state)
        print(partsps[idx])
        print(ps[idx])
    '''
    return(ps)
        

def gen_Q(n, p):
    q = np.zeros((n, n))
    for i, row in enumerate(q):
        for j, column in enumerate(row):
            if i == j:
                q[i, j] = 0
            elif(i < j):
                q[i, j] = find_min_div(i, j, p)
            elif(i > j):
                q[i, j] = q[j, i]

    return(q)


def find_min_div(a, b, p, i=0):
    if a // p**i == b // p**i:
        return(i)
    return(find_min_div(a, b, p, i=i+1))

            
def frustration_disconfort(J, s):
    triu = np.triu_indices(J.shape[0], m=J.shape[0])
    s1 = np.matrix(s)
    C = s1.T*s1
    C.T[triu] = 0
    J[triu] = 0
    return(- (J * C).trace())


def frustration_disconfort1(J, s):
    
    return(-sum([J[i, k] * s[0, i] * s[0, k]
                 for i in range(len(J))
                 for k in range(i)]))
    

def indexing(states, N, n):
    '''indexing([[]], 3, 0)'''
    if n > N-1:
        return(states)
    n += 1
    
    new_states = []
    for state in states:
        for idx in [-1, 1]:
            new_states.append(state[:]+[idx])
    return(indexing(new_states, N, n))


def test0_H():
    s = np.random.randint(0, 2, 3)
    s[s == 0] = -1
    J = np.ones((3, 3))
    print("s:")
    print(s)
    H = frustration_disconfort(np.matrix(J), np.matrix(s))
    print("H: ", H)
    H1 = frustration_disconfort1(np.matrix(J), np.matrix(s))
    print("H1: ", H1)


def test1_H(J, n=3):
    # s = np.zeros(n)
    # J = np.ones((n, n))
    # Hmin = frustration_disconfort(np.matrix(J), np.matrix(s))
    states = indexing([[]], n, 0)
    Hs = []
    for state in states:
        H = frustration_disconfort(np.matrix(J), np.matrix(state))
        Hs.append((H[0, 0], state))
    return(Hs)


def test2_H():
    Hs = test1_H(np.random.normal(0, 1/7, 7), 7)
    print("Hs:")
    for H, s in Hs:
        print(H, ": ", s)
    print("min H: ", min(Hs, key=lambda x: x[0]))
    

def test_div():
    qi = find_min_div(1, 4, 2)
    print("qi: ", qi)

def test_Q():
    q = gen_Q(24, 2)
    plt.imshow(q)
    plt.show()
    print(q)


if __name__ == "__main__":

    q = gen_Q(24, 2)
    ps = calc_P(q, 5)
    print(len(ps))
    # Hs = test1_H(np.ones((7, 7)), 7)
    # test2_H()
    # states = indexing([[]], 3, 0)
    # print(states)

