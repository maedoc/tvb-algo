"""Network implementation including weights & delays.
"""

import numpy as np


def wm_ring(W, D, dt, pre, post, ncv, cut=0, icf=lambda h: h):
    """Build white matter connectome model with sparse weights, ring buffer.
    """
    n = W.shape[0]
    m = W > cut  # non-zero mask
    w = W[m]  # non-zero weights
    d = D[m]  # non-zero delays
    di = (d / dt).astype('i')  # non-zero delays in time steps
    r, c = np.argwhere(m).T  # non-zero row & col indices
    lri, = np.argwhere(np.diff(np.r_[-1, r])).T  # local reduction indices
    nzr = np.unique(r)  # rows with non-zeros
    H = di.max() + 1
    hist = icf(np.zeros((H, n, ncv), 'f'))

    def step(i, xi):
        hist[i % H] = xi
        xj = hist[(i - di) % H, c]
        gx = np.add.reduceat((w * pre(xi[c], xj).T).T, lri)
        out = np.zeros_like(xi)
        out[nzr] = post(gx)
        return out

    return step


def wm_lin(W, D, dt, pre, post, ncv, icf=lambda h: h, buf_len=10):
    """White matter connectome model with linear delay buffer.
    """
    n = W.shape[0]
    di = (D / dt).astype('i')
    nid = np.tile(np.r_[:n], (n, 1))
    dm = di.max()
    H = dm * buf_len
    hist = np.zeros((H, n, ncv), 'f')
    hist[:dm] = icf(hist[:dm])
    i = dm
    while True:
        if i == H:
            hist[:dm] = hist[-dm:]
            i = dm
        hist[i + 1] = yield post((W[..., np.newaxis] * pre(hist[i], hist[i - di, nid])).sum(axis=-2))
        i += 1


def wm_no_delay(W, pre, post, ncv, icf=lambda h: h, buf_len=10):
    """White matter connectome model without delays.
    """
    W_ = W[..., np.newaxis]
    state = icf(np.zeros((len(W), ncv), 'f'))
    while True:
        state = yield post((W_ * pre(state)).sum(axis=-2))