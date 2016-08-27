import scipy as sp
import pandas as pd
from mpmath import meijerg

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

def sample_critical_values(ps=[.9, .95, .99], size=100000):
    """Calculates critical values for the KVB distribution using `size` samples"""
    chunk_size = min(size, 10000)
    n_chunks = int(size/chunk_size) + 1
    samples = sp.hstack([sample_kvb(chunk_size) for _ in range(n_chunks)])[:size]
    return pd.Series(sp.percentile(samples, 100*ps), ps)
    
def t_stat(X, y):
    """Calculate the KVB t-stats for the regression coefficients of y on X"""
    T = y.shape[0]
    beta = sp.linalg.solve(X.T.dot(X), X.T.dot(y))
    u = y - X.dot(beta)
    
    S = sp.cumsum(X*u[:, None], 0)
    C = 1./T**2 * (S[:, :, None]*S[:, None, :]).sum(0)
#    C_chol = sp.linalg.cholesky(C)
    
    Q = 1./T * (X[:, None, :]*X[:, :, None]).sum(0)
    
    B = sp.linalg.inv(Q).dot(C).dot(sp.linalg.inv(Q))
#    B_chol = sp.linalg.solve(Q, C_chol)
    
#    sigmas = sp.diag(B_chol)/sp.sqrt(T)
    sigmas = sp.sqrt(sp.diag(B)/T)
    return beta/sigmas

def pdf(z, tol=1e-5):
    if z == 0:
        return 0.15085282
    
    first_terms = 1./sp.pi * sp.sqrt(2./sp.absolute(z))
    
    tol = tol/first_terms
    summands = []
    j = 0
    while True:
        binomial = sp.special.binom(-0.5, j)
        sign = (-1)**j

        g_as = [[], [-.25]]
        g_bs = [[.25, .5, 0], []]
        g_z = z**2 * (j + .25)**2
        g = float(meijerg(g_as, g_bs, g_z))
        
        summand = binomial*sign*g
        summands.append(summand)        
        j += 1

        if sp.absolute(summand) < tol:
            break
        
    return first_terms*sum(summands)
    
def cdf(lim=20, dx=0.1):
    xs = sp.arange(0, lim + dx, dx)
    fs = [pdf(x) for x in xs]
          
    Fs = 0.5 + sp.integrate.cumtrapz(fs, xs)
    
    return xs, Fs
    
def critical_vals(ps=[.9, .95, .99]):
    xs, Fs = cdf()
    
    return 
    
def f(y):
    g_as = [[], [-.25]]
    g_bs = [[.25, .5, 0], []]
    g = float(meijerg(g_as, g_bs, y))
    
    return y**(-.75)*g

def F(y):
    g_as = [[1], [0]]
    g_bs = [[0.5, .75, .25], [0]]
    g = float(meijerg(g_as, g_bs, y))
    
    return g