import numpy as np
from random import random


class Dist():
    
    '''\sum_{x_{i}\leq x} x_{i} \leq u < \sum_{x_{i}>x} x_{i}
    where u from Uniform(0, 1)'''
    
    def __init__(self, dx=1):
        self.dx = 1
        
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
        while(s < u):
            s += self.f(x)*self.dx
            
            if s < u:
                x += self.dx
            
            # in case of never rich the end:
            i += 1
            if i > 100000:
                break 
        if dbg:
            print(u, s, x)
        return(x)
    
    def sample_n(self, n, dbg=False):
        return(np.array([self.sample(dbg=dbg) for k in range(n)]))
  

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
  
