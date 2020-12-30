import numpy as np
from random import random


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
  

class ARDist():
    '''Acceptance-rejection alg (p.47 in [1.])'''
    
    def __init__(self, a, b, f):
        self.a = a
        self.b = b
        self.f = f
        
        self.usampler = lambda n: np.random.random(n)
        
        # hdist = UniformAB1(a, b)
        # self.hsampler = hdist.sample_n
        inv_H = lambda u: a+(b-a)*u 
        self.hsampler = lambda n: inv_H(np.random.random(n))
        
        # maximum of f:
        self.M = np.max(f(np.random.uniform(np.min(a), np.max(b),
                                            np.size(a)*10000)
                          .reshape(10000, np.size(a))))
        self.C = self.M*(b-a)
        
    def sample_n(self, n):
        M = self.M
        f = self.f
        u1 = self.usampler(n)
        y = self.hsampler(n)
        result = y[u1 <= f(y)/M]
        m = len(result)

        # adjustment since count of samples is random:
        while(m < n):
            u1 = self.usampler(n)
            y = self.hsampler(n)
            
            result = np.concatenate([result, y[u1 <= f(y)/M]])
            m = len(result)
        
        return(result[:n])


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


class Property(object):
    storage = {}

    def __init__(self, attr):
        self.attr = attr

    def __get__(self, instance, owner=None):

        # in case we cleared context:
        if instance.name not in self.storage:
            self.storage[instance.name] = {}
        if self.attr not in self.storage[instance.name]:
            self.storage[instance.name][self.attr] = {"value": None,
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
    mu = Property("mu")
    sigma = Property("sigma")

    def __init__(self, name, mu=0, sigma=0.1, lims=[-1, 1]):
        self.type = "Dist"
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.lims = lims

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

            tmp = []
            for param_name, param in list(self.storage):
                if param["done"] == True:
                    pass
                    

            for idx, param in enumerate(self.params):
                
                samples[param.name] = self.params[idx]['sampler'].sample_n(n)
                
            cond = self.conditions(samples)

            for idx, param in enumerate(self.params):
                result = self.params[idx]['values']
                result = np.concatenate([result, samples[param.name][cond]])
                self.params[idx]['values'] = result

                # len of any sample:
                m = len(self.params[idx]['values'])
                print("m = ", m)

        return(result[:n])


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    N = 3000
    mu = 10*np.zeros(3)
    sigma = 0.1*np.zeros(3)
    # mu = 10
    # sigma = 0.1

    norm = Normal("x", mu, sigma)

    xs = norm.sample_n(N)
    m1 = max(xs)

    fig, ax = plt.subplots()

    for idx, xxs1 in enumerate(xs):
        v, b, p = ax.hist(xxs1, 70,  # normed=True, # density=True,
                          stacked=True)
        xs1 = np.linspace(norm.low_lim[idx], norm.up_lim[idx], N)
        ax.plot(xs1, (norm.norm_density(xs1)[idx]/norm.M)*max(v))
    plt.show()
    '''
    v, b, p = ax.hist(xs, 70,  # normed=True, # density=True,
                      stacked=True)
    xs1 = np.linspace(norm.low_lim, norm.up_lim, N)
    ax.plot(xs1, (norm.norm_density(xs1)/norm.M)*max(v))
    plt.show()
    '''
