import scipy as sp
import pandas as pd

# Kiefer, Vogelsang, Bunzel 2000 - Simple, Robust Testing of Regression Hypotheses

def wiener(size, steps=1000):
    r = sp.arange(1, steps+1)/steps
    z = sp.random.normal(size=(size, steps))
    return r, z.cumsum(1)/sp.sqrt(steps), 
    
def sample_kvb(size):
    r, w = wiener(size)
    w1 = w[:, [-1]]
    bottom = 1./len(r)*((w - r[None, :]*w1)**2).sum(1, keepdims=True)
    return (w1/sp.sqrt(bottom))[:, 0]

def critical_values(ps=[90, 95, 99], size=100000):
    chunk_size = min(size, 10000)
    n_chunks = int(size/chunk_size) + 1
    samples = sp.hstack([sample_kvb(chunk_size) for _ in range(n_chunks)])[:size]
    return pd.Series(sp.percentile(samples, ps), ps)
    
def t_stat(X, y):
    T = y.shape[0]
    beta = sp.linalg.solve(X.T.dot(X), X.T.dot(y))
    u = y - X.dot(beta)
    
    S = sp.cumsum(X*u[:, None], 0)
    C = 1./T**2 * (S[:, :, None]*S[:, None, :]).sum(0)
    C_chol = sp.linalg.cholesky(C)
    
    Q = 1./T * (X[:, None, :]*X[:, :, None]).sum(0)
    
#    B = sp.linalg.inv(Q).dot(C).dot(sp.linalg.inv(Q))
    B_chol = sp.linalg.solve(Q, C_chol)
    
    sigmas = sp.diag(B_chol)/sp.sqrt(T)
    return sigmas
