import tensorflow as tf
import numpy as np
from tvb_algo.network import wm_lin
from tvb_algo.deint import em_color
from tvb_algo.data import tvb76_weights_lengths

# run simulation w/ numpy and tf and compare speed

W, D = tvb76_weights_lengths(tensors=True)

dt=0.05
simlen=150.0
k=0.0
speed=1.0
freq=1.0

n = W.shape[0]
pre = lambda i, j: j - 1.0
post = lambda gx: k * gx
prop = wm_lin(W, D / speed, dt, pre, post, 1)
next(prop) # discard first
if tf.is_tensor(W):
    sqrt = tf.sqrt
    zeros = tf.zeros
    rand = tf.random.uniform
    icrange = tf.constant([1.0, -0.6])
    seq = lambda n: tf.range(0, n)
    def f(i, X):  # monostable
        x, y = tf.transpose(X)
        c, = tf.transpose(prop.send(tf.reshape(x, (-1, 1))))
        dx = freq * (x - x ** 3 / 3 + y) * 3.0
        dy = freq * (1.01 - x + c) / 3.0
        return tf.transpose(tf.stack([dx, dy]))
else:
    sqrt = np.sqrt
    zeros = lambda sh: np.zeros(sh, 'f')
    rand = np.random.uniform
    icrange = np.r_[1.0, -0.6]
    seq = lambda n: np.r_[:n]
    def f(i, X):  # monostable
        x, y = X.T
        c, = prop(i, x.reshape((-1, 1))).T
        dx = freq * (x - x ** 3 / 3 + y) * 3.0
        dy = freq * (1.01 - x + c) / 3.0
        return np.array([dx, dy]).T

def g(i, X):  # additive linear noise
    return sqrt(1e-9)

X = zeros((n, 2))
T = seq(int(simlen / dt))
Xs = []
for t, (x, _) in zip(T, em_color(f, g, dt, 1e-1, X)):
    if t == 0:
        x += -1.0
    if t == 1:
        x += rand((n, 2)) / 5 + icrange
    Xs.append(x)
