from pylab import *
from tvb_algo import data, deint, network

print('downloading weights')
W, D = data.tvb76_weights_lengths()

def sim(dt=0.05, tf=150.0, k=0.0, speed=1.0, freq=1.0):
    n = W.shape[0]
    pre = lambda i, j: j - 1.0
    post = lambda gx: k * gx
    prop = network.wm_ring(W, D / speed, dt, pre, post, 1)

    def f(i, X):  # monostable
        x, y = X.T
        c, = prop(i, x.reshape((-1, 1))).T
        dx = freq * (x - x ** 3 / 3 + y) * 3.0
        dy = freq * (1.01 - x + c) / 3.0
        return array([dx, dy]).T

    def g(i, X):  # additive linear noise
        return sqrt(1e-9)

    X = zeros((n, 2))
    Xs = zeros((int(tf / dt),) + X.shape)
    T = r_[:Xs.shape[0]]
    for t, (x, _) in zip(T, deint.em_color(f, g, dt, 1e-1, X)):
        if t == 0:
            x[:] = -1.0
        if t == 1:
            x[:] = rand(n, 2) / 5 + r_[1.0, -0.6]
        Xs[t] = x
    return T, Xs

dt = 0.05
figure(figsize=(12, 6))
from time import time
from tqdm import tqdm
elapsed = 0.0
speeds = [1.0, 2.0, 10.0]
for i, speed in enumerate(tqdm(speeds)):
    tic = time()
    t, x = sim(dt, 150.0, 1e-3, speed)
    elapsed += time() - tic
    subplot(2, 3, i + 1)
    plot(t[::5] * dt, x[::5, :, 0] + 0 * r_[:W.shape[0]], 'k', alpha=0.3)
    grid(True, axis='x')
    xlim([0, t[-1] * dt])
    title('Speed = %g mm/ms' % (speed,))
    xlabel('time (ms)')
    ylabel('X(t)')
    subplot(2, 3, i + 4)
    hist((D[W != 0] / speed).flat[:], 100, color='k')
    xlim([0, t[-1] * dt])
    grid(True)
    xlabel('delay (ms)')
    ylabel('# delay')
tight_layout()
print('%.3fs elapsed' % (elapsed,))
#show()
