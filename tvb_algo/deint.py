"""Differential equation integration.
"""

import numpy as np
import tensorflow as tf


def em_white(f, g, dt, x):
    """Euler-Maruyama for white noise.
    """
    i = 0
    nd = x.shape
    if tf.is_tensor(x):
        G = lambda i, x: tf.random.normal(nd) * tf.sqrt(g(i, x))
    else:
        G = lambda i, x: np.random.randn(*nd) * np.sqrt(g(i, x))
    while True:
        yield x
        i += 1
        x += dt * f(i, x) + G(i, x)


def em_color(f, g, dt, lam, x):
    """Euler-Maruyama for colored noise.
    """
    i = 0
    nd = x.shape
    if tf.is_tensor(x):
        N = lambda : tf.random.normal(nd)
        sqrt = tf.sqrt
        e = tf.sqrt(g(i, x) * lam) * N()
        E = tf.exp(-lam * dt)
    else:
        N = lambda : np.random.randn(*nd)
        sqrt = np.sqrt
        e = np.sqrt(g(i, x) * lam) * N()
        E = np.exp(-lam * dt)
    while True:
        yield x, e
        i += 1
        x += dt * (f(i, x) + e)
        h = sqrt(g(i, x) * lam * (1 - E ** 2)) * N()
        e = e * E + h