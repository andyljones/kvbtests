import scipy as sp
import pandas as pd

def wiener(size, steps=1000):
    r = sp.arange(1, steps+1)/steps
    z = sp.random.normal(size=(size, steps))
    return r, z.cumsum(1)/sp.sqrt(steps), 
    
def sample_kvb(size):
    r, w = wiener(size)
    w1 = w[:, [-1]]
    bottom = 1./len(r)*((w - r[None, :]*w1)**2).sum(1, keepdims=True)
    return (w1/sp.sqrt(bottom))[:, 0]