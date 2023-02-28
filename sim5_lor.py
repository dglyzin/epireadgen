import numpy as np
import torch
import pyro
import pyro.distributions as pdist
import characteristics as chars
# import sim5_hw as sm
import matplotlib.pyplot as plt


def model(N, b, sigma, epsilon=0.01):
    # ehandler = sm.EventsHandler(-np.inf)
    # with ehandler:
    with chars.Lyapunov(epsilon, epsilon, trace=None, step=1) as lpv:
        args = init_model(b, sigma, epsilon)
        r = run_model(N, *args)
    lapunov = lpv.results["Lambda"][-1]
    return r, lapunov
   

def init_model(b, sigma, epsilon):
    # param:
    r = pyro.sample("r", pdist.Uniform(25, 30))

    x0 = torch.sqrt(b*(r-1))
    y0 = torch.sqrt(b*(r-1))
    z0 = r-1
    x0e = x0+epsilon
    y0e = y0+epsilon
    z0e = z0+epsilon
    
    return (
        torch.tensor([x0, x0e]), torch.tensor([y0, y0e]),
        torch.tensor([z0, z0e]), sigma, b, r)


def run_model(N, x0, y0, z0, sigma, b, r):
    
    x = x0.detach().clone()
    y = y0.detach().clone()
    z = z0.detach().clone()

    for i in range(N):
        x_prev = x.detach().clone()
        y_prev = y.detach().clone()
        z_prev = z.detach().clone()
        x += - sigma*x_prev+sigma*y_prev
        y += - x_prev*z_prev + r*x_prev - y_prev
        z += x_prev*y_prev - b*z_prev

    pyro.param("x_%d" % N, torch.cat([
        x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], 0))

    print("x:", x)
    print("y:", y)
    print("z:", z)
    return r


def test_lapunov():
    b = 8/3
    sigma = 10.

    r, lapunov = model(9, b, sigma, epsilon=0.001)
    r = torch.tensor(1.)
    print("b, sigma, r:", b, sigma, r)

    print("lapunov direct:", lapunov)

    l0 = []
    l1 = []
    l2 = []
    rs = np.arange(-0.997701, -0.99760, 0.0000001)
    for r1 in rs:
        r = torch.tensor(r1)
        print("r: ", r)
        c = torch.sqrt(b*(r-1)) if r-1 > 0 else torch.sqrt(-b*(r-1))
        J = torch.tensor([
            [-sigma, sigma, 0],
            [1., -1., c],
            [c, c, -b]
        ])
        # print("J:", J)
        laccurate = torch.eig(J)
        print("laccurate:", laccurate[0][:, 0])
        l0.append(laccurate[0][0, 0])
        l1.append(laccurate[0][1, 0])
        l2.append(laccurate[0][2, 0])
    plt.plot(rs, l0)
    plt.plot(rs, l1)
    plt.plot(rs, l2)
    plt.plot(rs, np.zeros_like(rs))
    plt.show()


if __name__ == "__main__":
    test_lapunov()
