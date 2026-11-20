import math
import os
from decimal import Decimal, getcontext
import numpy as np
import matplotlib.pyplot as plt
# 60-digit decimal arithmetic for the rounding-error check (M3), and pi to 60 digits
getcontext().prec = 60
PI60 = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
# every line printed by say() is also saved to Report/crr_results.txt at the end
LINES = []
def say(text):
    print(text, flush=True)
    LINES.append(text)

#normal
def phi(x):
    # standard normal density phi(x)
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
def series(x, tol):
    # sum over k >= 0 of x^(2k+1) / (1*3*5*...*(2k+1))
    term = total = x
    k = 0
    while abs(term) > tol * abs(total):
        k += 1
        term *= x * x / (2 * k + 1)
        total += term
    return total
def Phi(x):
    # standard normal distribution function Phi(x) = 1/2 + phi(x) * series(x)  (M1)
    if abs(x) > 8:
        return 1.0 if x > 0 else 0.0
    return 0.5 + phi(x) * series(x, 1e-18)

#Black-Scholes
def bs(S, K, r, sig, T):
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    disc = math.exp(-r * T)
    return d1, d2, S * Phi(d1) - K * disc * Phi(d2), disc * Phi(d2)

#Binomial
def lattice(S, K, r, sig, T, n):
    h = T / n
    s = sig * math.sqrt(h)
    u, d = math.exp(s), math.exp(-s)
    p = (math.exp(r * h) - d) / (u - d)
    alpha = 0.5 * n + math.log(K / S) / (2.0 * s)
    a = math.floor(alpha) + 1
    return p, p * u * math.exp(-r * h), a, 1.0 - 2.0 * (a - alpha)
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
    tail = binom_tail(n, p, a)
    return S * binom_tail(n, p_hat, a) - K * disc * tail, disc * tail, Delta

#predicted coefficients
def call_coefficient(S, K, r, sig, T, Delta):
    #coeff of 1/n for the call
    d1, d2, _, _ = bs(S, K, r, sig, T)
    A = -sig**2 * T * (6 + d1**2 + d2**2) + 4 * T * (d1**2 - d2**2) * r - 12 * T**2 * r**2
    return S * math.exp(-0.5 * d1 * d1) / (24 * sig * math.sqrt(2 * math.pi * T)) \
        * (A - 12 * sig**2 * T * (Delta**2 - 1))
def digital_coefficient(S, K, r, sig, T, Delta):
    #coeff of n^(-1/2) for the digital call
    return math.exp(-r * T) * phi(bs(S, K, r, sig, T)[1]) * Delta
def slope(xs, ys):
    # least-squares slope of the straight-line fit of ys against xs
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

#Precision check of rounding error
def exact_bs(S, K, r, sig, T):
    #finding black-scholes price and delta in 60 digit decimal arithmetic
    S, K, r, sig, T = (Decimal(str(v)) for v in (S, K, r, sig, T))
    Phi60 = lambda x: Decimal("0.5") + (-x * x / 2).exp() / (2 * PI60).sqrt() * series(x, Decimal("1e-55"))
    d1 = ((S / K).ln() + (r + sig * sig / 2) * T) / (sig * T.sqrt())
    d2 = d1 - sig * T.sqrt()
    disc = (-r * T).exp()
    return S * Phi60(d1) - K * disc * Phi60(d2), disc * Phi60(d2)

def exact_crr(S, K, r, sig, T, n):
    #finding crr price and delta in 60 digit decimal arithmetic
    S, K, r, sig, T = (Decimal(str(v)) for v in (S, K, r, sig, T))
    s = sig * (T / n).sqrt()
    u, d = s.exp(), (-s).exp()
    p = ((r * T / n).exp() - d) / (u - d)
    p_hat = p * u * (-r * T / n).exp()
    alpha = Decimal(n) / 2 + (K / S).ln() / (2 * s)
    a = int(alpha.to_integral_value(rounding="ROUND_FLOOR")) + 1
    tiny = Decimal("1e-80")
    def tail(prob):
        q = 1 - prob
        m = int((n + 1) * prob)
        total, upper, w = Decimal(1), Decimal(1 if m >= a else 0), Decimal(1)
        for j in range(m, n):
            w = w * (n - j) * prob / ((j + 1) * q)
            total += w
            if j + 1 >= a: upper += w
            if w < tiny: break
        w = Decimal(1)
        for j in range(m, 0, -1):
            w = w * j * q / ((n - j + 1) * prob)
            total += w
            if j - 1 >= a: upper += w
            if w < tiny: break
        return upper / total
    disc = (-r * T).exp()
    tail_p = tail(p)
    return S * tail(p_hat) - K * disc * tail_p, disc * tail_p, 1 - 2 * (a - alpha)

#uspensky and berry-esseen
def uspensky_check(S, K, r, sig, T, n):
    # Uspensky remainder Omega for the tail [a_n, n] (Fact 5), and its upper bound
    p, _, a, _ = lattice(S, K, r, sig, T, n)
    q = 1.0 - p
    s = math.sqrt(n * p * q)
    G = lambda t: Phi(t) + (q - p) / (6 * s) * (1 - t * t) * phi(t)
    omega = binom_tail(n, p, a) - (G((n + 0.5 - n * p) / s) - G((a - 0.5 - n * p) / s))
    return s * s, omega, (0.13 + 0.18 * abs(p - q)) / s**2 + math.exp(-1.5 * s)

def berry_esseen(n, p):
    # Berry-Esseen bound 3*rho/(sigma^3 sqrt n) for Bin(n, p), with rho/sigma^3 = (p^2+q^2)/sqrt(pq)
    return 3 * (p * p + (1 - p) ** 2) / math.sqrt(n * p * (1 - p))

#Experiment section
BASE = dict(S=100.0, K=110.0, r=0.05, sig=0.2, T=1.0)
SETS = [("base  K=110", BASE), ("K=90", dict(BASE, K=90.0)), ("K=100.5", dict(BASE, K=100.5)),
        ("sig=0.4", dict(BASE, sig=0.4)), ("r=0, T=0.5", dict(BASE, r=0.0, T=0.5))]
BLOCKS = [50, 100, 200, 400, 800, 1600, 3200]

def scan(par, n_lo=50, n_hi=6400):
    # for every n in [n_lo, n_hi): returns n, C_n - C, D_n - D, Delta_n
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
            N=N, rmsC=math.sqrt(np.mean(c**2)), rmsD=math.sqrt(np.mean(d**2)),
            maxC=np.max(np.abs(c)), maxD=np.max(np.abs(d)),
            RC=np.max(np.sqrt(n) * np.abs(n * c - f[m])),
            RD=np.max(np.sqrt(n) * np.abs(np.sqrt(n) * d - g[m])),
            nstar=int(n[k]), Dstar=De[m][k], nstar_err=n[k] * abs(d[k]),
            scaledD=np.max(np.sqrt(n) * np.abs(d))))
    return rows

def rate(rows, key):
    #Slope of log(error)/log N
    return slope([math.log(row["N"]) for row in rows], [math.log(row[key]) for row in rows])

def rate_range(rows, key):
    # smallest and largest slope over the refits that leave out one block (sensitivity check)
    fits = [rate(rows[:i] + rows[i + 1:], key) for i in range(len(rows))]
    return min(fits), max(fits)

if __name__ == "__main__":
    # outputs go into a Report folder next to where the script is run (created if missing)
    os.makedirs("Report", exist_ok=True)

    # [0] checks
    say("Checks")
    d1, d2, C, D = bs(**BASE)
    say("base case: d1 = %.6f  d2 = %.6f  C_BS = %.10f  D_BS = %.10f" % (d1, d2, C, D))
    Cx, Dx = exact_bs(**BASE)
    say("double precision minus 60-digit: C: %.1e, D: %.1e" % (C - float(Cx), D - float(Dx)))
    say("Uspensky: n, s^2, Omega, bound")
    for n in (101, 6400):
        say("%6d  %8.2f  %+.2e  %.2e" % ((n,) + uspensky_check(n=n, **BASE)))

    # [1] main scan
    say("\n Base case S=100, K=110, r=0.05, sigma=0.2, T=1; all n in [50, 6400)")
    scans = {name: scan(par) for name, par in SETS}
    ns, eC, eD, De = scans["base  K=110"]
    rows = block_stats(BASE, ns, eC, eD, De)
    say("block N | rms(C_n-C) | rms(D_n-D) | R_C | R_D | n* | Delta_n* | n*|D_n*-D| | "
        "max|C_n-C| | max|D_n-D| | max sqrt(n)|D_n-D|")
    for row in rows:
        say("%(N)5d  %(rmsC).3e  %(rmsD).3e  %(RC).3f  %(RD).4f   %(nstar)5d  %(Dstar)+.1e  "
            "%(nstar_err).4f  %(maxC).3e  %(maxD).3e  %(scaledD).4f" % row)
    say("fitted slopes of log(rms error) against log N:  call %.3f   digital %.3f" %
        (rate(rows, "rmsC"), rate(rows, "rmsD")))
    say("leave-one-block-out range:  call [%.3f, %.3f]   digital [%.3f, %.3f]" %
        (rate_range(rows, "rmsC") + rate_range(rows, "rmsD")))
    say("fitted slopes of log(max error) against log N:  call %.3f   digital %.3f" %
        (rate(rows, "maxC"), rate(rows, "maxD")))
    say("ratio rms(D_n-D) / rms(C_n-C) in the last block: %.1f" % (rows[-1]["rmsD"] / rows[-1]["rmsC"]))
    say("predicted: n(C_n-C) between %.4f (Delta=+-1) and %.4f (Delta=0);  "
        "sqrt(n)(D_n-D) = %.5f * Delta_n" %
        (call_coefficient(Delta=1.0, **BASE), call_coefficient(Delta=0.0, **BASE),
         digital_coefficient(Delta=1.0, **BASE)))
    last = ns >= 3200
    nC, rootnD = ns[last] * eC[last], np.sqrt(ns[last]) * eD[last]
    say("3200<=n<6400: least-squares slope of sqrt(n)(D_n-D) on Delta_n: %.5f" % slope(De[last], rootnD))
    say("3200<=n<6400: max sqrt(n)|D_n-D| = %.4f, min Delta_n = %+.5f" % (np.max(np.abs(rootnD)), De[last].min()))
    absC = dict(zip(ns.tolist(), np.abs(eC)))             # |C_n - C| by n
    say("doubling n: |C_250-C| = %.3e, |C_500-C| = %.3e;  |C_2n-C| > |C_n-C| for "
        "%d of the 3150 values 50 <= n < 3200" %
        (absC[250], absC[500], sum(absC[2 * n] > absC[n] for n in range(50, 3200))))

    # [2] other parameter sets
    say("\n Other parameters: rms slopes with leave-one-block-out ranges, and R_C, R_D in the first and last block")
    for name, par in SETS:
        rw = block_stats(par, *scans[name])
        say("%-12s call %.3f [%.3f, %.3f]  digital %.3f [%.3f, %.3f]  R_C %.3f -> %.3f  R_D %.4f -> %.4f" %
            ((name, rate(rw, "rmsC")) + rate_range(rw, "rmsC") + (rate(rw, "rmsD"),) + rate_range(rw, "rmsD")
             + (rw[0]["RC"], rw[-1]["RC"], rw[0]["RD"], rw[-1]["RD"])))

    # [3] large n
    say("\n n | |C_n(double)-C_n(60)| | |D_n(double)-D_n(60)| | sqrt(n)(n(C_n-C)-f(Delta_n)) in double | same in 60 digits")
    for n in (10, 100, 1000, 6399, 20000, 10**5, 10**6, 10**7):
        Cn, Dn, Delta = crr(n=n, **BASE)
        Cnx, Dnx, Deltax = exact_crr(n=n, **BASE)
        stat = math.sqrt(n) * (n * (Cn - C) - call_coefficient(Delta=Delta, **BASE))
        statx = math.sqrt(n) * (n * float(Cnx - Cx) - call_coefficient(Delta=float(Deltax), **BASE))
        say("%9d  %.1e  %.1e  %+.3f  %+.3f" % (n, abs(Cn - float(Cnx)), abs(Dn - float(Dnx)), stat, statx))

    # Berry-Esseen alternative
    say("\n Berry-Esseen bound 3*rho/(sigma^3 sqrt n) per tail, summed with weights S and K e^{-rT}")
    for n in (100, 1000, 6399):
        p, p_hat, _, _ = lattice(n=n, **BASE)
        bound = BASE["S"] * berry_esseen(n, p_hat) + BASE["K"] * math.exp(-BASE["r"] * BASE["T"]) * berry_esseen(n, p)
        say("%6d  bound %.3f   actual |C_n-C| %.3e" % (n, bound, absC[n]))

    # figure: (a) errors and envelopes; (b), (c) scaled errors against Delta_n
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.3, 1.8))
    dots = dict(ms=1.0, rasterized=True)
    g1 = digital_coefficient(Delta=1.0, **BASE)
    f1 = max(abs(call_coefficient(Delta=1.0, **BASE)), abs(call_coefficient(Delta=0.0, **BASE)))
    axa.loglog(ns, np.abs(eD), ".", color="tab:orange", **dots)
    axa.loglog(ns, np.abs(eC), ".", color="tab:blue", **dots)
    axa.loglog(ns, g1 / np.sqrt(ns), "k-", lw=0.8, label=r"digital: $%.3f\,n^{-1/2}$" % g1)
    axa.loglog(ns, f1 / ns, "k--", lw=0.8, label=r"call: $%.3f\,n^{-1}$" % f1)
    axa.set(xlabel=r"$n$", title=r"(a) $|C_n-C|$, $|D_n-D|$")
    grid = np.linspace(-1, 1, 201)
    axb.plot(grid, call_coefficient(Delta=grid, **BASE), color="grey", lw=3, label=r"$f(\Delta)$")
    axb.plot(De[last], nC, ".", color="tab:blue", **dots)
    axb.set(xlabel=r"$\Delta_n$", title=r"(b) call: $n\,(C_n-C)$, $n\geq 3200$")
    axc.plot(grid, digital_coefficient(Delta=grid, **BASE), color="grey", lw=3,
             label=r"$\mathrm{e}^{-rT}\varphi(d_2)\,\Delta$")
    axc.plot(De[last], rootnD, ".", color="tab:orange", **dots)
    axc.set(xlabel=r"$\Delta_n$", title=r"(c) digital: $\sqrt{n}\,(D_n-D)$, $n\geq 3200$")
    for ax in (axa, axb, axc):
        ax.grid(True, lw=0.3)
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig("Report/crr_figure.pdf", dpi=400)
    say("\nfigure written to Report/crr_figure.pdf")
    with open("Report/crr_results.txt", "w") as out:
        out.write("\n".join(LINES) + "\n")
