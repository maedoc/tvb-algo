import numpy as np
from .network import wm_ring
from .multiscale import vtx2roi, roi2vtx
from .deint import em_color
from .data import tvb76_weights_lengths


def test_wm_ring_exact():
    n = 4

    # build simple connectivity
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = 1
    D = np.r_[:n * n].reshape((n, n))

    # custom initial conditions function
    def icf(hist):
        hist[:] = 1.0
        return hist

    # make propagator
    prop = wm_ring(W, D, 1, lambda i, j: j, lambda g: g, 1, icf=icf)

    # run simulation
    x = np.ones((n, 1))
    xs = []
    for i in range(10):
        # equiv. to Sum(Model) w/ Identity(Integrator) in TVB test
        x += prop(i, x)
        xs.append(x.flat[:])

    # correct output
    xs_ = np.array([
        [2., 2., 2., 1.],
        [3., 3., 3., 1.],
        [5., 4., 4., 1.],
        [8., 5., 5., 1.],
        [12., 6., 6., 1.],
        [17., 7., 7., 1.],
        [23., 8., 8., 1.],
        [30., 10., 9., 1.],
        [38., 13., 10., 1.],
        [48., 17., 11., 1.]])
    assert np.allclose(np.array(xs), xs_)


def test_nd_em_color():
    f = lambda i, x: x - x ** 3 / 3 - sum(x)
    g = lambda i, x: np.exp(x) * 0.5
    X = np.zeros(3)
    Xs = np.zeros((10000, X.size))
    T = np.r_[:Xs.shape[0]]
    for t, (x, _) in zip(T, em_color(f, g, 0.01, 0.5, X)):
        Xs[t] = x
    # figure(figsize=(10, 5))
    # subplot(121), plot(Xs)
    # subplot(122), hist(Xs.flat[:], 100, color='k')


def test_rmap():
    def pre(xi, xj):
        return xj - xi

    def post(gx):
        return 0.1 * gx - 0.2

    A = np.random.randn(64, 64)
    prop = wm_ring(A, A, 0.1, pre, post, 1)
    nv = 5000
    vtx = np.r_[:nv][:, np.newaxis]
    rmap = np.random.randint(0, A.shape[0], nv)
    assert roi2vtx(rmap, prop(23, vtx2roi(rmap, vtx))).shape


def test_sim():
    W, D = tvb76_weights_lengths()

    def sim(dt=0.05, tf=150.0, k=0.0, speed=1.0, freq=1.0):
        n = W.shape[0]
        pre = lambda i, j: j - 1.0
        post = lambda gx: k * gx
        prop = wm_ring(W, D / speed, dt, pre, post, 1)

        def f(i, X):  # monostable
            x, y = X.T
            c, = prop(i, x.reshape((-1, 1))).T
            dx = freq * (x - x ** 3 / 3 + y) * 3.0
            dy = freq * (1.01 - x + c) / 3.0
            return np.array([dx, dy]).T

        def g(i, X):  # additive linear noise
            return np.sqrt(1e-9)

        X = np.zeros((n, 2))
        Xs = np.zeros((int(tf / dt),) + X.shape)
        T = np.r_[:Xs.shape[0]]
        for t, (x, _) in zip(T, em_color(f, g, dt, 1e-1, X)):
            if t == 0:
                x[:] = -1.0
            if t == 1:
                x[:] = np.random.rand(n, 2) / 5 + np.r_[1.0, -0.6]
            Xs[t] = x
        return T, Xs

    assert sim(tf=10.0)