# $ python3 -m stoch_proc -model diff
# $ python3 -m stoch_proc -model sumOfSum
# $ python3 -m stoch_proc -model indInc -test_number 0
# $ python3 -m stoch_proc -model indInc -test_number 1
# $ python3 -m stoch_proc -model snm -test_number 0
# $ python3 -m stoch_proc -model snm -test_number 1 -dim 1
# $ python3 -m stoch_proc -model snm -test_number 1 -dim 2

import numpy as np
import itertools as it
from plotter import plot
import sys

        
class IndIncProcess:
    '''
    Independent Increments
    (which is markovian
     since $E(X_{j}(X_{j+1}-X_{j}))=0$
     lead to $\\rho_{j k}=\\rho_{j v}*\\rho_{v k}$ (8.3) in ch3 Feller vol 2)
    $Var(X_{k}) = \\sigma(\\xi)^2 t_{k} = t_{k}$
    depending only  on length $t_{k}$ between increases since:
    $X_{k} = \\sum_{i}\\xi_{i}$
    # REF: example (a) of the Feller. vol 2 ch3 subch 8
    '''
    def __init__(self, N=10, size=1, count=0):
        self.N = N
        self.size = size
        self.count = count
        
    def gen(self):
        if self.size == 1:
            eps = np.random.normal(0, 1, (self.N, 1))
            initial = np.zeros(1)
        elif self.size == 2:
            eps = np.random.normal(np.array([0, 0]), np.array([1, 1]),
                                   (self.N, 2))
            initial = np.zeros(2)
        else:
            raise(Exception("size '%d' is not supported" % self.size))

        xs = list(it.accumulate(
            eps, func=lambda acc, x: acc+x, initial=initial))
        self.res = (np.array(xs), eps)
        self.initial = initial
        
    def get_xs(self):
        # print(self.res[0])
        return self.res[0]
    
    def get_title(self, idx, xs):
        return("x%s" % idx)

    def get_filename(self):
        return("si_dim=%d_N=%d.jpg" % (self.size, self.N))

    
class SumOfSumProcess(IndIncProcess):
    '''Sum of sum of markovian is not markovian
    # REF: example (b) of the Feller. vol 1 ch15 subch 13
    '''
    def gen(self):
        IndIncProcess.gen(self)
        xs = IndIncProcess.get_xs(self) 
        ys = list(it.accumulate(
            xs, func=lambda acc, x: acc+x, initial=self.initial))
        self.res = (np.array(ys), xs)


class DiffProcess(IndIncProcess):
    '''
    $\\frac{dx}{dt} = y$
    $\\frac{dy}{dt} = -w^2 x-2\\alpha w y + w^2 \\sqrt(q) \\frac{d\\xi}{dt}$
    # REF: example (3.7) of the Афанасьев, Колмановский. vol 2 ch2 subch 3
    '''
    def __init__(
            self, N=10, dt=0.01,
            w=0.03, alpha=0.3, q=0.09,
            x0=[1., 1.]):
        self.dt = dt
        self.w = w
        self.alpha = alpha
        self.q = q
        IndIncProcess.__init__(self, N=N, size=1)
        self.x0 = np.array([x0])
        
    def gen(self):
        IndIncProcess.gen(self)
        deps = self.res[1]
        dt = self.dt
        w = self.w
        alpha = self.alpha
        q = self.q
        
        def succ(acc, de):
            # print("acc:", acc)
            # print("de:", de)
            res = [
                acc[0, 0]+dt*acc[0, 1],
                -w**2*dt*acc[0, 0]+(1-2*alpha*w*dt)*acc[0, 1]]
            # print("res:", res)
            return(np.array([[res[0], res[1]+w*np.sqrt(q)*de[0]]]))
        
        xs = np.concatenate(list(it.accumulate(
            deps, func=succ, initial=self.x0)), 0)
        # print("xs:", xs)
        
        # self.res = (xs[:, 0], xs[:, 1]) 
        self.res = xs

    
class SNMProcess:
    '''Stationary normal markovian
    # REF: example (c) of the Feller. vol 2 ch3 subch 8
    '''
    def __init__(self, N=10, count=10, ro=0.3, sigma=1, size=1):
        self.N = N
        self.count = count
        self.ro = ro
        self.sigma = sigma
        self.size = size
        
    def gen(self):
        res = gen_stationary_norm_markovian(
            N=self.N, ro=self.ro, sigma=self.sigma, size=self.size)
        self.res = res
        # print(self.res[0][0])
        # print(self.res[0][1])
        # print(self.res[0].shape)
        
    def get_xs(self):
        return self.res[0]
    
    def collect_params(self, xs):
        res = xs
        self.mean = np.mean(res)
        self.var = np.var(res)
        # self.var1 = np.mean(res**2)-self.mean**2
        self.lim = (self.sigma*np.sqrt((1+self.ro)/(1-self.ro))
                    * (1-self.ro**self.N))
        
    def get_title(self, idx, xs):
        self.collect_params(xs)
        return(
            "xs_%s: ro=%.3f, sigma=%.3f. lim=%.3f, mean=%.3f, var=%.3f"
            % (idx, self.ro, self.sigma, self.lim, self.mean, self.var))
    
    def get_filename(self):
        return("snm_Count_%d_N_%d_ro=%.3f .jpg"
               % (self.count, self.N, self.ro))
        
        
def gen_stationary_norm_markovian(N=10, ro=0.3, sigma=1, size=1):
    '''
    If $E(Z_{k})=0$ and $E(Z_{k}^{2})=1$ then the seq of $X_{k}$:

    $X_{1}=\\lambda_{1} Z_{1}$
    $X_{k}=a_{k}X_{k-1}+\\lambda_{k}Z_{k}$

    will be stationary markovian normal dist:
    since:
    $ E(X_{j}, X_{k}) = E(X_{j}(a_{k}X_{k-1}+\\lambda_{k}Z_{k}))=$
    $a_{k}E(X_{j}X_{k-1})+\\lambda_{k}E(X_{j}Z_{k})$
    since $E(X_{j}Z_{k}) = 0$ ($Z_{k}$ is independent of $(X_{1},...X_{k-1})$)
    then
    $a_{k}=\\frac{E(X_{j}X_{k})}{E(X_{j}X_{k-1})}$=
    since Xk is stationary = 
    =$\\frac{\\sigma^{2}\\rho^{k-j}}{\\sigma^{2}\\rho^{j-(k-1)}}=$
    (since markovian) 
    = $\\rho_{k-1,k}$=
    (since stationary)
    = $\\rho$
    # REF: example (c) of the Feller. vol 2 ch3 subch 8
    '''
    resX, resZ = _gen_snm(N, 0, ro, sigma, size, [], [])

    # print(sigma*np.sqrt((1+ro)/(1-ro))*(1-ro**N))
    return np.array(resX), np.array(resZ)


def _gen_snm(N, n, ro, sigma, size, resX, resZ):

    # finish
    if n > N:
        return resX, resZ
    
    zk = np.random.normal(0, 1, size)
    lk = sigma*np.sqrt(1-ro**2)

    # init
    if resX == []:
        return _gen_snm(N, n+1, ro, sigma, size, [lk*zk], [zk])

    # main
    xk1 = resX[-1]    
    xk = ro*xk1+lk*zk
    
    return _gen_snm(N, n+1, ro, sigma, size, resX+[xk], resZ+[zk])


# ========= tests =========:

def test_diff(show=False):
    process = DiffProcess(N=1000, alpha=.3, w=7.,  q=0.0001)
    process.gen()
    xs = process.res
    # print("xs:", xs)
    plot("diff", [xs], ["xs"], dim=2, show=show)

    
def test_X_Nth(process, idxs=[-1], show=False):
    count = process.count
    res = []

    # collection indexes:
    for _ in range(count):
        process.gen()
        xIdxs = process.get_xs()[idxs]
        # print(xIdxs)
        res.append(xIdxs)

    # print(res)
    res = np.concatenate(res, 1)
    # print(res)
    plot(process.get_filename(), res,
         [process.get_title(idxs[idx], r)
          for idx, r in enumerate(res)],
         show=show)


def test_gen_seq(process, show=False):
    process.gen()
    xs, zs = process.res
    plot("test0_"+process.get_filename(),
         [xs, zs], [process.get_title("", xs), "z"],
         dim=process.size, show=show)
        

if __name__ == "__main__":

    names = ["model", "test_number", "dim"]
    names_values = dict(zip(
        names,
        [["diff", "sumOfSum", "indInc", "snm"],
         ["0", "1"], ["1", "2"]]))
    
    parsed_args = {}
    for name in names:
        if "-"+name in sys.argv:
            name_val = sys.argv[sys.argv.index("-"+name)+1]
            # print("%s:" % name, name)
            if name_val not in names_values[name]:
                raise Exception(
                    "-%s values is: %s"
                    % (name, ",".join(names_values[name])))
            parsed_args[name] = name_val
    '''
        else:
            raise Exception(
                "'-%s' argument needed (%s)"
                % (name, ",".join(names_values[name])))
    '''
    if parsed_args["model"] == "diff":
        print("running %s" % parsed_args["model"])
        test_diff()
    elif parsed_args["model"] == "sumOfSum":
        print("running %s" % parsed_args["model"])
        test_gen_seq(SumOfSumProcess(N=983, size=1))
    elif parsed_args["model"] == "indInc":
        print("running %s" % parsed_args["model"])
        '''
        test_X_Nth(
            IndIncProcess(N=10, count=1000, size=1),
            idxs=[1, -1])
        '''
        if parsed_args["test_number"] == "0":
            test_X_Nth(
                IndIncProcess(N=1000, count=1000, size=1),
                idxs=[10, -1])
        elif parsed_args["test_number"] == "1":
            test_gen_seq(IndIncProcess(N=983, size=1))
    elif parsed_args["model"] == "snm":
        print("running %s" % parsed_args["model"])
        if parsed_args["test_number"] == "0":
        
            # test_X_Nth(
            #     SNMProcess(N=10, count=1000, ro=0.03, sigma=1, size=1),
            #     idxs=[1, -1])
            test_X_Nth(
                SNMProcess(N=100, count=1000, ro=0.7, sigma=1, size=1),
                idxs=[10, -1])
        elif parsed_args["test_number"] == "1":
            if parsed_args["dim"] == "1":
                test_gen_seq(SNMProcess(N=983, ro=0.93, sigma=1, size=1))
            elif parsed_args["dim"] == "2":
                test_gen_seq(SNMProcess(N=983, ro=0.93, sigma=1, size=2))
    print("done")
