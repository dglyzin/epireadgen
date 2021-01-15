import numpy as np
from random import random
import functools as fs
import inspect


class Dist():
    
    '''\sum_{x_{i}\leq x} x_{i} \leq u < \sum_{x_{i}>x} x_{i}
    where u from Uniform(0, 1)'''
    
    def __init__(self, dx=1):
        self.dx = 1

        # for error with override:
        self.number_of_failure = 0
        self.sample_number = 0

    def F(self, m):
        # if m > self.n:
        #     return(1)
        x = 0
        p = 0
        while(x <= m):
            p += self.f(x)*self.dx
            x += self.dx
            
        return(p)
        # return(sum([self.f(k) for k in range(m+1)]))
  
    def sample(self, dbg=False):
        u = random()
        s = 0
        x = 0
        i = 0
        dx = self.dx
        
        number_of_failure_tmp = 0
        self.sample_number += 1

        while(s < u):
            s += self.f(x)*dx
            
            if s < u:
                x += dx
            
            # in case of never rich the end:
            i += 1
            if i > 100000:
                dx = 0.1
                
                # print("u, s, x:")
                # print(u, s, x)
                if number_of_failure_tmp == 0:
                    number_of_failure_tmp = 1
                    self.number_of_failure += 1
                    # print(number_of_failure)
                # break 
        if dbg:
            print("u, s, x:")
            print(u, s, x)
        return(x)
    
    def sample_n(self, n, dbg=False):
        self.number_of_failure = 0
        self.sample_number = 0
        return(np.array([self.sample(dbg=dbg) for k in range(n)]))
 
    def print_failures(self):
        print("number_of_failure, sample_number:")
        print(self.number_of_failure, self.sample_number)
        

class Dist0(Dist):
    
    '''Same as previous but without end points
    (for continues)'''
    
    def __init__(self, dx=1):
        Dist.__init__(self, dx=dx)
        self.dx = dx
        
    def F(self, m):
        # if m > self.n:
        #     return(1)
        x = 0
        p = 0
        while(x < m):
            p += self.f(x)*self.dx
            x += self.dx
            
        return(p)
        # return(sum([self.f(k) for k in range(m+1)]))
        

class Dist1:
    
    '''X = F_{X}^{-1}(U) where U = Uniform(0, 1)'''
    
    def inv_F(self, y):
        pass
    
    def sample(self):
        u = random()
        # print(u)
        return(self.inv_F(u))
  
    def sample_n(self, n):
        return(np.array([self.sample() for k in range(n)]))
  

def cond_decorator(func, add_args={}):
    '''applay some cond (array(x==True)) to all func args which
    is parameters (like mu or sigma) and not applay to random variable itself
    (like x)'''

    def wd(*args, cond=None):
        func_args_names = inspect.getfullargspec(func).args
        
        # if no cond given, use all arrays:
        if cond is None:
            args += tuple(add_args[arg_name] for arg_name in func_args_names
                          if arg_name in add_args) 
            # print(args)
            return(func(*args))
        # print(args[0].size)
        # print(add_args)
        # if cond given cut it from parameters, not from random variable
        # itself (see ARDist. sample_n code).
        args += tuple(add_args[arg_name][cond]
                      if type(add_args[arg_name]) == np.ndarray else add_args[arg_name]
                      for arg_name in func_args_names
                      if arg_name in add_args) 
        return(func(*args))
    return(wd)


class ARDist():
    '''Acceptance-rejection alg (p.47 in [1.])
    Used arrays to generate values. Values in array
    generated at each place with use of numpy conditions
    (like a[a==True]).
    '''
    
    def __init__(self, a, b, f, fargs):

        self.a = self.check_type(a)
        self.b = self.check_type(b)
        self.f = cond_decorator(f, fargs)
        self.fargs = fargs

    def check_type(self, a):
        '''Convert all to arrays'''

        if type(a) == np.ndarray:
            return(a)
        elif(type(a) == list):
            return(np.array(a))
        else:
            return(np.array([a]))

    def preproc(self):

        a = self.a
        b = self.b
        f = self.f

        self.usampler = lambda n: np.random.random(n)
        # self.usampler = lambda n: np.random.random(np.size(a))
        # self.usampler = lambda n: np.random.random(np.size(a)*n).reshape(n, np.size(a))
        
        # hdist = UniformAB1(a, b)
        # self.hsampler = hdist.sample_n
        inv_H = lambda u, a, b: a+(b-a)*u
        self.hsampler = lambda n , a, b: inv_H(np.random.random(n), a, b)
        # self.hsampler = lambda n: inv_H(np.random.random(np.size(a)))
        # self.hsampler = lambda n: inv_H(np.random.random(np.size(a)*n).reshape(n, np.size(a)))
        
        # maximum of f:
        # self.M = np.max(f(np.random.uniform(np.min(a), np.max(b),
        #                                     10000)))
        self.M = np.max(f(np.random.uniform(np.min(a), np.max(b),
                                            np.size(a)*10000)
                          .reshape(10000, np.size(a))))
        self.C = self.M*(b-a)
        
    def sample_n(self, n):

        self.preproc()
        # if type(self.a) == "numpy.ndarray":
        #     n = np.size(self.a)
        # use arrays for iterators, rather then n:
        a = self.a * np.ones(n) if len(self.a) == 1 else self.a
        b = self.b * np.ones(n) if len(self.b) == 1 else self.b
        M = self.M
        f = self.f

        u1 = self.usampler(n)
        y = self.hsampler(n, a, b)
        icond = u1 <= f(y)/M
        assert (icond.size == a.size)

        # not icond:
        nicond = (icond == False)
        assert (nicond.size == icond.size)

        result = np.ones(n)
        result[icond] = y[icond]
        
        # result = [y.T[i][u1.T[i] <= f(y).T[i]/M]
        #           for i in range(np.size(self.a))]
        
        # print("u1.shape ", u1.shape)
        # print("y.shape ", y.shape)
        # print("f(y).shape ", f(y).shape)

        # print((u1 <= f(y)/M).any())
        # print("result.shape ", result.shape)
        # print(result)

        m = len(result[nicond])
        
        # adjustment since count of samples is random:
        # t = 0
        while(nicond.any()):
            # size of (not condition == True):
            m = nicond[nicond==True].size
            # print(m)
            
            u1 = self.usampler(m)
            
            assert m == a[nicond].size
            assert m == b[nicond].size
            y = self.hsampler(m, a[nicond], b[nicond])
            # print(nicond.size, u1.size, y.size)

            '''
            Algorithm:
            icond = [t, t, t, f, f, f, f]
            nicond = [f, f, f, t, t, t, t]
            icond[nicond] = [f, f, f, f]
            (*) icond[nicond] = u1 <= f(y, cond=nicond)/M
            Using (*) means for example
            icond[nicond] became [t, f, t, f]

            # since icond.size == nicond.size
            # after (*) should be:
            result[nicond][icond[nicond]] same as result[icond==nicond]
                [t, t, t, t] [t, f, t, f] -> result[t,,t,]
            
            # last step:
            nicond = (icond==False)
            # so size to sample again should decrease
            '''

            # not all done:
            assert not (icond[nicond]).all()
            
            # size decreased:
            # 
            icond[nicond] = u1 <= f(y, cond=nicond)/M
            # icond has same size as nicond
            assert (icond.size == nicond.size)
            # print("icond[nicond]", icond[nicond][icond[nicond]==True].size)
            # t += y[icond[nicond]].size
            # print("y.size", y[icond[nicond]].size)
            
            result[nicond == icond] = y[icond[nicond]]
            # print(result[nicond==icond])
            # print(y[icond[nicond]])
            # assert result[nicond==icond] == y[icond[nicond]]
            
            nicond = (icond == False)
            assert (icond.size == nicond.size)
            # result = np.concatenate([result, y[wcond]])
            # m = len(result)
        # print("t=", t)
        # print((result[result==1]).size)
        return(result)
        # return(result[:n])


class ARDistSimple():
    '''Acceptance-rejection alg (p.47 in [1.])
    No arrays'''
    
    def __init__(self, a, b, f, fargs={}):

        self.a = a
        self.b = b
        # fix parameters in f (like sigma or mu):
        self.f = fs.partial(f, **fargs)
        self.fargs = fargs
        self.max_t = 0

    def preproc(self):

        a = self.a
        b = self.b
        
        f = self.f
        
        self.usampler = lambda: np.random.random()
        
        inv_H = lambda u, a, b: a+(b-a)*u
        self.hsampler = lambda a, b: inv_H(np.random.random(), a, b)
        
        # maximum of f:
        self.M = np.max(f(np.random.uniform(np.min(a), np.max(b),
                                            10000)))
        # self.M = np.max(f(np.random.uniform(np.min(a), np.max(b),
        #                                     np.size(a)*10000)
        #                   .reshape(10000, np.size(a))))
        self.C = self.M*(b-a)

    def sample(self):
        '''Generate one value until it found
        self.max_t is count of attempts.'''

        self.preproc()
        a = self.a
        b = self.b
        M = self.M
        f = self.f

        u1 = self.usampler()
        y = self.hsampler(a, b)
        # print("y=", y)
        icond = u1 <= f(y)/M
        t = 0
        while(not icond):
            u1 = self.usampler()
            y = self.hsampler(a, b)
            
            icond = u1 <= f(y)/M
            t += 1
            
        if t > self.max_t:
            self.max_t = t
            print("t= ", t)
        
        return(y)

    def sample_n(self, n):
        '''Generate each value one by one without array usage.'''
        self.preproc()
                
        result = []

        for idx in range(n):
            
            y = self.sample()
            result.append(y)
        
        return(np.array(result))


class MH():
    
    def __init__(self, f):
        self.f = f
        
        mu = 0
        sigma = 0.1
        low_lim, up_lim = [-1, 1]
        
        # normal distribution here:
        norm_density = lambda x: (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-((x-mu)/sigma)**2/2)
    
        ar = ARDist(low_lim, up_lim, norm_density)
        self.eps_sampler = lambda n: ar.sample_n(n)
    
        # uniform distribution:
        self.usampler = lambda n: np.random.random(n)
    
    def sample_n(self, n, x0=7):
        
        f = self.f
        
        eps = self.eps_sampler(n)
        u = self.usampler(n)
        xs = np.zeros(n)
        xs[0] = x0
        self.err = np.zeros(n)
        
        for i in range(n-1):
            y = xs[i] + eps[i]
            
            rate = min(f(y)/f(xs[i]), 1)
            if u[i] <= rate:
                xs[i+1] = y
                self.err[i] = rate
            else:
                xs[i+1] = xs[i]
            # print("xs[i], y, xs[i+1], u[i], rate:")    
            # print(xs[i], y, xs[i+1], u[i], rate)
            
        return(xs[-2])


class Property1(object):

    def __init__(self, param_name):
        self.param_name = param_name
        # self.param_int = 3
        # self.param_dict = {}

    def __get__(self, instance, owner=None):
        
        # print(self.param_name)
        # print(self.param_int)
        # print(self.param_list)
        # print(self.param_dict)

        obj = instance.param[self.param_name]
        if hasattr(obj, "type") and getattr(obj, "type") == "Dist":
            # if object:
            return(getattr(obj, "value"))
        else:
            return(obj)
        
    def __set__(self, instance, value):

        if not hasattr(instance, "param"):
            setattr(instance, "param", {})
        instance.param[self.param_name] = value
        
        # self.param_int += 1
        # self.param_dict[self.param_name] = self.param_int
        # if self.param_int > 4:
        #     self.param_dict.clear()



class Property(object):
    storage = {}

    def __init__(self, attr):
        self.attr = attr

    def __get__(self, instance, owner=None):

        # in case we cleared context:
        if instance.name not in self.storage:
            self.storage[instance.name] = {"value": None,
                                           "params": {}}
        if self.attr not in self.storage[instance.name]["params"]:
            self.storage[instance.name]["params"][self.attr] = {"value": None,
                                                                "dist": None,
                                                                "done": False}
        '''
        param_dist = self.storage[instance.name][self.attr]['dist']
        if param_dist is not None:
            value = self.storage[param_dist.name]['value']
            if value is not None:
                return()
        '''
        # return as is:
        return(self.storage[instance.name][self.attr]['value'])
    
    def __set__(self, instance, value):

        # in case of initialization:
        if instance.name not in self.storage:
            self.storage[instance.name] = {}
        if self.attr not in self.storage[instance.name]:
            self.storage[instance.name][self.attr] = {"value": None,
                                                      "dist": None,
                                                      "done": False}
        
        if 'type' in dir(value) and value.type == 'Dist':
            # if dist:
            if value.name in self.storage:
                
                # if has value
                if self.storage[value.name]['value'] is not None:
                    self.storage[instance.name][self.attr]['value'] = self.storage[value.name]['value']
                else:
                    self.storage[instance.name][self.attr]['dist'] = value
        else:
            # if int or array
            self.storage[instance.name][self.attr]['value'] = value

        
class Normal():
    mu = Property1("mu")
    sigma = Property1("sigma")

    def __init__(self, name, mu=0, sigma=0.1, lims=[-1, 1]):

        self.type = "Dist"
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.lims = lims
        self.value = None

    def preproc(self):
        '''all access to properties here'''
        self.low_lim, self.up_lim = self.lims
        
        self.low_lim += self.mu
        self.up_lim += self.mu

        # normal distribution here:
        self.norm_density = lambda x: (1/(np.sqrt(2*np.pi)*self.sigma))*np.exp(-((x-self.mu)/self.sigma)**2/2)
    
        ar = ARDist(self.low_lim, self.up_lim, self.norm_density)
        self.sampler = lambda n: ar.sample_n(n)
        self.M = ar.M
        
    # def sampler(self, n):
    #     for mu1 in self.mu:
    #         pass
    #         # TODO

    def sample_n(self, n):
        self.preproc()
        return(self.sampler(n))


class SimpleRejection():
    '''
    - ``params`` -- dict with sampler (distribution) for each
    param.

    - ``conditions`` -- return bool array for
    all params.
    ex: conditions({'a': values, 'b': values, ...})
       :-> array[bool]
    '''
    def __init__(self, params, conditions):
        # conditions(params)-> array[bool]
        self.conditions = conditions
        self.params = params

        # for access storage:
        self.props = Property("_simple_rejection_")
        self.storage = self.props.storage

    def run(self, n):
        samples = {}
        result = np.array([])
        m = 0
        # TODO:
        # order conditioned
        # dist object like param for dist (hierarchy)

        # adjustment since count of samples is random:
        while(m < n):

            tmp = [dist_name for dist_name in self.storage]
            while(len(tmp) > 0):
                for dist_name in tmp:
                    dist = self.storage[dist_name]
                    if dist['value'] is None:
                        if all([param.value is not None for param in dist['params']]):
                            dist["value"] = dist.sample_n(n)
                            tmp.pop(tmp.index(dist_name))
            '''
            for idx, param in enumerate(self.params):
                
                samples[param.name] = self.params[idx]['sampler'].sample_n(n)
            '''
            cond = self.conditions(samples)

            for idx, param in enumerate(self.params):
                result = self.params[idx]['values']
                result = np.concatenate([result, samples[param.name][cond]])
                self.params[idx]['values'] = result

                # len of any sample:
                m = len(self.params[idx]['values'])
                print("m = ", m)

        return(result[:n])


def test_ardist_simple():
    import matplotlib.pyplot as plt
    
    N = 3000
    mu = 10
    sigma = 0.1
    f = lambda x: 7*np.sin(x)+np.sin(4*x)
    dist = ARDistSimple(0, np.pi, f)
    xs = dist.sample_n(N)
    m1 = max(xs)

    fig, ax = plt.subplots()

    v, b, p = ax.hist(xs, 70,  # normed=True, # density=True,
                      stacked=True)
    xs1 = np.linspace(dist.a, dist.b, N)
    ax.plot(xs1, (f(xs1)/dist.M)*max(v))
    plt.show()


def test_ardist_norm_simple():
    import matplotlib.pyplot as plt

    N = 3000
    
    mu = 10
    sigma = 0.1
    
    f = lambda x, mu, sigma: (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-((x-mu)/sigma)**2/2)
    
    dist = ARDistSimple(mu-sigma, mu+sigma, f, fargs={"mu": mu, "sigma": sigma})
    xs = dist.sample_n(N)
    m1 = max(xs)

    fig, ax = plt.subplots()

    v, b, p = ax.hist(xs, 70,  # normed=True, # density=True,
                      stacked=True)
    xs1 = np.linspace(dist.a, dist.b, N)
    ax.plot(xs1, (f(xs1, mu, sigma)/dist.M)*max(v))
    plt.show()


def test_ardist():
    import matplotlib.pyplot as plt
    
    N = 3000
    mu = 10
    sigma = 0.1
    f = lambda x: 7*np.sin(x)+np.sin(4*x)
    dist = ARDist(0, np.pi, f, {})
    xs = dist.sample_n(N)
    m1 = max(xs)

    fig, ax = plt.subplots()

    v, b, p = ax.hist(xs, 70,  # normed=True, # density=True,
                      stacked=True)
    xs1 = np.linspace(dist.a, dist.b, N)
    ax.plot(xs1, (f(xs1)/dist.M)*max(v))
    plt.show()


def test_ardist_norm():
    import matplotlib.pyplot as plt

    N = 3000
    
    mu = 10*np.zeros(N)
    sigma = 0.1*np.ones(N)
    
    f = lambda x, mu, sigma: (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-((x-mu)/sigma)**2/2)
        
    dist = ARDist(mu-sigma, mu+sigma, f, {"mu": mu, "sigma": sigma})
    xs = dist.sample_n(N)
    m1 = max(xs)

    fig, ax = plt.subplots()

    v, b, p = ax.hist(xs, 70,  # normed=True, # density=True,
                      stacked=True)
    xs1 = np.linspace(dist.a[0], dist.b[0], N)
    ax.plot(xs1, (f(xs1, mu, sigma)/dist.M)*max(v))
    plt.show()
    

if __name__ == "__main__":

    # test_ardist_norm_simple()
    # test_ardist_simple()
    # test_ardist()
    test_ardist_norm()
    '''
    mu = 10*np.zeros(3)
    sigma = 0.1*np.ones(3)
    # mu = 10
    # sigma = 0.1

    norm = Normal("x", mu, sigma)

    xs = norm.sample_n(N)
    m1 = max(xs)

    fig, ax = plt.subplots()
    print(xs.shape)
    for idx, xxs1 in enumerate(xs.T):
        print(idx)
        v, b, p = ax.hist(xxs1, 70,  # normed=True, # density=True,
                          stacked=True)
        xs1 = np.linspace(norm.low_lim, norm.up_lim, N)
        ax.plot(xs1.T[idx], (norm.norm_density(xs1).T[idx]/norm.M)*max(v))
    plt.show()
    '''
    
