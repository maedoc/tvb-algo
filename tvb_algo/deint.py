"""Differential equation integration.
"""

import numpy as np


def em_white(f, g, dt, x):
    """Euler-Maruyama for white noise.
    """
    i = 0
    nd = x.shape
    while True:
        yield x
        i += 1
        x += dt * f(i, x) + np.random.randn(*nd) * np.sqrt(g(i, x))


def em_color(f, g, dt, lam, x):
    """Euler-Maruyama for colored noise.
    """
    i = 0
    nd = x.shape
    e = np.sqrt(g(i, x) * lam) * np.random.randn(*nd)
    E = np.exp(-lam * dt)
    while True:
        yield x, e
        i += 1
        x += dt * (f(i, x) + e)
        h = np.sqrt(g(i, x) * lam * (1 - E ** 2)) * np.random.randn(*nd)
        e = e * E + h