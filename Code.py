import sys, math
from decimal import Decimal, getcontext
import numpy as np
#validation of pi
getcontext().prec = 60
PI60 = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
#to store into file
LINES= []
def say(text):
    print(text, flush=True)
    LINES.append(text)
    
#normal
def phi(x):
    #strd norm dens
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
def series(x,tol):
    # sum over k >= 0 of x^(2k+1) / (1*3*5*...*(2k+1))
    term = total = x
    k = 0
    while abs(term) > tol * abs(total):
        k+= 1
        term *= x * x / (2 * k + 1)
        total += term
    return total
def Phi(x):
    # strd norm dist func
    if abs(x) > 8:
        return 1.0 if x > 0 else 0.0
    return 0.5 + phi(x) * series(x, 1e-18)

#Black-Scholes
def bs(S, K, r, sig, T):
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    disc = math.exp(-r * T)
    return d1, d2, S * Phi(d1) - K * disc* Phi(d2), disc *  Phi(d2)

#Binomial 
def lattice(S, K, r, sig, T, n):
    h = T / n
    s = sig * math.sqrt(h)
    u, d = math.exp(s), math.exp(-s)
    p = (math.exp(r * h) - d) / (u - d)
    p_hat = p * u * math.exp(-r * h)
    alpha = 0.5 * n + math.log(K / S) / (2.0 * s)
    a = math.floor(alpha) + 1 
    return p, p_hat, a, 1.0 - 2.0 * (a - alpha)
def binom_tail(n, p, a):
    if a <= 0: return 1.0
    if a > n:  return 0.0
    q = 1.0 - p
    m = int((n + 1) * p)
    j = np.arange(m, n, dtype=float)
    above = np.cumprod((n - j) / (j + 1.0) * (p / q))
    j = np.arange(m, 0, -1, dtype=float)
    below = np.cumprod(j / (n - j + 1.0) * (q / p))
    w = np.concatenate((below[::-1], [1.0], above))
    return float(np.sum(w[a:]) / np.sum(w))