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
    LINES.append(Text)
#normal
def phi(x):
    #strd norm dens
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
def series(x,tol):
    #
    term = total = x
    k = 0
    while abs(term) > tol * abs(total):
        k+= 1
        term *= x * x / (2 * k + 1)
        total += term
    return total