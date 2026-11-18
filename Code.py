import math
from decimal import Decimal, getcontext
import numpy as np
import matplotlib.pyplot as plt
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
def crr(S, K, r, sig, T, n):
    #use the CRR model to compute the call price and delta
    p, p_hat, a, Delta = lattice(S, K, r, sig, T, n)
    disc = math.exp(-r * T)
    tail, tail_hat = binom_tail(n, p, a), binom_tail(n, p_hat, a)
    return S * tail_hat - K * disc * tail, disc * tail, Delta

#predicted coefficients
def call_coefficient(S, K, r, sig, T, Delta):
    #coeff of 1/n for the call
    d1, d2, _, _ = bs(S, K, r, sig, T)
    A = -sig**2 * T * (6 + d1**2 + d2**2) + 4 * T * (d1**2 - d2**2) * r - 12 * T**2 * r**2
    return S * math.exp(-0.5 * d1 * d1) / (24 * sig * math.sqrt(2 * math.pi * T)) \
             * (A - 12 * sig**2 * T * (Delta**2 - 1))
def digital_coefficient(S, K, r, sig, T, Delta):
    #coeff of n^(-1/2) for the digital call
    d1, d2, _, _ = bs(S, K, r, sig, T)
    return math.exp(-r * T) * phi(d2) * Delta
def slope(xs, ys):
    #least squ of y/x
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

#Precision check of rounding error
def exact_bs(S, K, r, sig, T):
    #finding black-scholes price and delta in 60 digit decimal arithmetic
    S, K, r, sig, T = (Decimal(str(v)) for v in (S, K, r, sig, T))
    def Phi60(x):
        return Decimal("0.5") + (-x * x / 2).exp() / (2 * PI60).sqrt() * series(x, Decimal("1e-55"))
    d1 = ((S / K).ln() + (r + sig * sig / 2) * T) / (sig * T.sqrt())
    d2 = d1 - sig * T.sqrt()
    disc = (-r * T).exp()
    return S * Phi60(d1) - K * disc * Phi60(d2), disc * Phi60(d2)

def exact_crr(S, K, r, sig, T, n):
    #finding crr price and delta in 60 digit decimal arithmetic
    S, K, r, sig, T = (Decimal(str(v)) for v in (S, K, r, sig, T))
    h = T / n
    s = sig * h.sqrt()
    u, d = s.exp(), (-s).exp()
    p = ((r * h).exp() - d) / (u - d)
    p_hat = p * u * (-r * h).exp()
    alpha = Decimal(n) / 2 + (K / S).ln() / (2 * s)
    a = int(alpha.to_integral_value(rounding="ROUND_FLOOR")) + 1
    def tail(prob):
        coeff, total = math.comb(n, a), Decimal(0)
        for j in range(a, n + 1):
            total += Decimal(coeff) * prob**j * (1 - prob)**(n - j)
            coeff = coeff * (n - j) // (j + 1)        # exact: C(n,j+1) = C(n,j)(n-j)/(j+1)
        return total
    disc = (-r * T).exp()
    tail_p = tail(p)
    return S * tail(p_hat) - K * disc * tail_p, disc * tail_p

#uspensky and berry-esseen
def uspensky_check(S, K, r, sig, T, n):
    #remainder omega and bound
    p, _, a, _ = lattice(S, K, r, sig, T, n)
    q = 1.0 - p
    s = math.sqrt(n * p * q)
    def G(t):
        return Phi(t) + (q - p) / (6 * s) * (1 - t * t) * phi(t)
    omega = binom_tail(n, p, a) - (G((n + 0.5 - n * p) / s) - G((a - 0.5 - n * p) / s))
    bound = (0.13 + 0.18 * abs(p - q)) / s**2 + math.exp(-1.5 * s)
    return s * s, omega, bound

def berry_esseen(n, p):
    #bound
    return 3 * (p * p + (1 - p) ** 2) / math.sqrt(n * p * (1 - p))

#Experiment section
BASE = dict(S=100.0, K=110.0, r=0.05, sig=0.2, T=1.0)
SETS = [("base  K=110", BASE), ("K=90", dict(BASE, K=90.0)), ("K=100.5", dict(BASE, K=100.5)), ("sig=0.4", dict(BASE, sig=0.4)), ("r=0, T=0.5", dict(BASE, r=0.0, T=0.5))]
BLOCKS = [50, 100, 200, 400, 800, 1600, 3200]    

def scan(par, n_lo=50, n_hi=6400):
    #For every n in ...: n, C_n - C, D_n - D, Delta_n."""
    _, _, C, D = bs(**par)
    ns = np.arange(n_lo, n_hi)
    prices = np.array([crr(n=int(n), **par) for n in ns])
    return ns, prices[:, 0] - C, prices[:, 1] - D, prices[:, 2]

def block_stats(par, ns, eC, eD, De):
    #error stat for each block
    f = call_coefficient(Delta=De, **par)
    g = digital_coefficient(Delta=De, **par)
    rows = []
    for N in BLOCKS:
        m = (ns >= N) & (ns < 2 * N)
        n, c, d = ns[m], eC[m], eD[m]
        k = np.argmin(np.abs(De[m]))
        rows.append(dict(
            N=N,
            rmsC=math.sqrt(np.mean(c**2)), rmsD=math.sqrt(np.mean(d**2)),
            maxC=np.max(np.abs(c)), maxD=np.max(np.abs(d)),
            RC=np.max(np.sqrt(n) * np.abs(n * c - f[m])),
            RD=np.max(np.sqrt(n) * np.abs(np.sqrt(n) * d - g[m])),
            nstar=int(n[k]), Dstar=De[m][k], nstar_err=n[k] * abs(d[k]),
            scaledD=np.max(np.sqrt(n) * np.abs(d))))
    return rows

def rate(rows, key):
    #Slope of log(error)/log N
    return slope([math.log(row["N"]) for row in rows], [math.log(row[key]) for row in rows])
