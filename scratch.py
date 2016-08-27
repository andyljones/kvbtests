import scipy as sp
import pandas as pd
from mpmath import meijerg

# Kiefer, Vogelsang, Bunzel 2000 - Simple, Robust Testing of Regression Hypotheses
# Abadir, Paruolo 2002 - Simple Robust Testing of Regression Hypotheses: A Comment
# Abadir, Paruolo 1997 - Two Mixed Normal Densities from Cointegration Analysis
    
def t_stat(X, y):
    """Calculate the KVB t-stats for the regression coefficients of y on X"""
    T = y.shape[0]
    beta = sp.linalg.solve(X.T.dot(X), X.T.dot(y))
    u = y - X.dot(beta)
    
    v = X*u[:, None]
    k = 1 - sp.arange(T)/T
    diag = (v[:, :, None]*v[:, None, :]).sum(0)
    offdiag = sum(k[i]*(v[i:, :, None]*v[:T-i, None, :]).sum(0) for i in range(1, T))
    C = (diag + offdiag + offdiag.T)/T
    
    Q = 1/T*(X[:, None, :]*X[:, :, None]).sum(0)
    
    B = sp.linalg.solve(Q, sp.linalg.solve(Q, C).T) # B = Q^-1 C Q^-1
    
    sigmas = sp.sqrt(sp.diag(B)/T)
    return beta/sigmas

def pdf(z, tol=1e-5):
    """Calculates the PDF of the KVB distribution"""
    if z == 0:
        return 0.15085282
    
    z = sp.absolute(z)
        
    first_terms = 1./sp.pi * 2**.25/sp.sqrt(z)
    
    tol = tol/first_terms
    summands = []
    j = 0
    while True:
        binomial = sp.special.binom(-0.5, j)
        sign = (-1)**j

        g_as = [[], [-.25]]
        g_bs = [[.25, .5, 0], []]
        g_z = (sp.sqrt(2)*z)**2 * (j + .25)**2
        g = float(meijerg(g_as, g_bs, g_z))
        
        summand = binomial*sign*g
        summands.append(summand)        
        j += 1

        if sp.absolute(summand) < tol:
            break
        
    return first_terms*sum(summands)
    
def cdf(z, tol=1e-5):
    """Calculates the CDF of the KVB distribution"""
    if z < 0:
        return 1 - cdf(-z, tol)
    elif z == 0:
        return 0.5
    
    first_terms = sp.sqrt(2)/sp.pi

    tol = tol/first_terms
    summands = []
    j = 0
    while True:
        binomial = sp.special.binom(-0.5, j)
        sign = (-1)**j

        g_as = [[1], [0]]
        g_bs = [[0.5, .75, .25], [0]]
        g_z = 2*z**2 * (j + .25)**2
        g = float(meijerg(g_as, g_bs, g_z))
        
        constant = 0.5/sp.sqrt(j + .25)
        summand = binomial*sign*constant*g
        summands.append(summand)
        j += 1

        if sp.absolute(summand) < tol:
            break
        
    return first_terms*sum(summands) + 1

def quantile(q, tol=1e-3):
    """Calculates the quantiles of the KVB distribution"""
    if q > .5:
        x0 = 2
    elif q < .5:
        x0 = -2
    elif q == 0.5:
        x0 = 0
    
    return sp.optimize.newton(lambda x: cdf(x) - q, x0, pdf, tol=tol)
    