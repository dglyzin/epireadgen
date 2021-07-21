import sim1
from scipy import stats
import matplotlib.pyplot as plt


def wrapper(dist):
    class SciDist():
        def __init__(self, n=1, **kwgs):
            # print(kwgs)
            self.dist_kwgs = kwgs
            self.dist = dist(**kwgs)
            self.__name__ = str(stats.uniform.__class__).split(".")[-1]
            self.n = n

        def __setattr__(self, var, val):
            '''
            for changing dist attributes in reset_params
            (like uniform.a, uniform.b)
            '''
            if hasattr(self, 'dist_kwgs'):
                if var in self.dist_kwgs:
                    if hasattr(self, 'dist'):
                        
                        self.dist.kwds[var] = val
                        # print("FROM setattr")
                        # print(self.dist.kwds)
                        # setattr(self.dist.dist, var, val)
            object.__setattr__(self, var, val)
            
        def sample(self):
            
            return(self.dist.rvs(self.n))

        def log_prob(self, value):
            return(self.dist.logpdf(value))

        # def enumerate_support(self, events):
        #     return(self.dist.support)
    return(SciDist)


def plot_results1(isamples):
    for var in isamples:
        print("var:")
        print(var)
        # var_data = np.array(isamples[var]).astype(np.float)
        # print(var_data)
        # print("probs of var (bin=2):")
        # var_data = torch.histc(torch.tensor(var_data), 2) 
        # print(var_data/var_data.sum())

        r = plt.hist(isamples[var], 30,  # density=True,
                    stacked=True
                    # , rwidth=0.1, label=label
        )
        
        plt.savefig("lsim1_test0_var_%s.jpg" % var)

        # plt.show()

    # plt.legend(label)


def test_set_attr():
    udist = wrapper(stats.uniform)
    p = sim1.make_dist(udist, "a", loc=0, scale=1)
    setattr(p, "loc", 3)
    print(p.dist.kwds)
    print(p.sample(None))
    

def test_make_dist(N=3, cond="$a>=0.7"):
    udist = wrapper(stats.uniform)
    # uniform = udist(1, loc=3, scale=1)
    
    p = sim1.make_dist(udist, "a", loc=0, scale=1)
    # p = make_dist(dist.Uniform(0, 1), "a")
    print(p.sample(None))
    pp = sim1.make_dist(udist, "b", loc=0, scale=1,
                        parents=["a"], params=[("loc", "a")])

    # P(Bag==1) = b (and P(Bag!=1) = 1-b):
    p0 = sim1.CondProb("Bag", parents=["b"], support=[[0, 1], ["b"]],
                       params=[("Bag==1", "b")])
    
    # P(Color|Bag):
    p1 = sim1.CondProb("Color", parents=["Bag"],
                       support=[["red", "green"], [0, 1]])
    p1.set({"Bag": 0}, [0.1, 0.9])
    p1.set({"Bag": 1}, [0.5, 0.5])

    net = sim1.BayesNet([p, pp])
    # net = sim1.BayesNet([p, p0, p1, pp])
    sample = net.prior_sample()
    print("sample:")
    print(sample)
    
    print('rejection_sampling(net, %s)' % cond)
    samples, indexes, labels = sim1.rejection_sampling(net, N,
                                                       cond=cond)
    print(len(samples['a']))
    # print(samples)
    plot_results1(samples)
        

def test_wrapper():
    udist = wrapper(stats.uniform)
    uniform = udist(1, loc=3, scale=1)
    r = uniform.sample()
    print("r = sample():")
    print(r)
    print("log_prob(r):")
    print(uniform.log_prob(r))


if __name__ == '__main__':
    # test_set_attr()
    test_make_dist(N=300, cond="float(np.array($a))>=0.7")
    # test_wrapper()
