"""Elastic response spectrum of a ground-motion record (pseudo-spectral acceleration).

Linear SDOF, Newmark average-acceleration integration (gamma = 1/2, beta = 1/4),
unconditionally stable; sub-steps the record so dt <= T/20 for accuracy.
Units: acceleration in g, period in s.
"""
import numpy as np


def sa(acc_g, dt, T, zeta=0.05, max_ratio=0.05):
    """Pseudo-spectral acceleration Sa(T, zeta) in g for record acc_g (g) at time step dt (s)."""
    acc = np.asarray(acc_g, float)
    n_sub = max(1, int(np.ceil(dt / (max_ratio * T))))
    if n_sub > 1:  # linear interpolation of the record onto a finer grid
        t = np.arange(len(acc)) * dt
        tf = np.arange(0, t[-1] + 1e-12, dt / n_sub)
        acc = np.interp(tf, t, acc)
        dt = dt / n_sub
    w = 2 * np.pi / T
    m, c, k = 1.0, 2 * zeta * w, w * w
    g, b = 0.5, 0.25
    a1 = m / (b * dt * dt) + g * c / (b * dt)
    a2 = m / (b * dt) + (g / b - 1) * c
    a3 = (1 / (2 * b) - 1) * m + dt * (g / (2 * b) - 1) * c
    kh = k + a1
    u = v = 0.0
    a = -acc[0]
    umax = 0.0
    for p in -acc[1:]:
        ph = p + a1 * u + a2 * v + a3 * a
        un = ph / kh
        vn = g / (b * dt) * (un - u) + (1 - g / b) * v + dt * (1 - g / (2 * b)) * a
        an = (un - u) / (b * dt * dt) - v / (b * dt) - (1 / (2 * b) - 1) * a
        u, v, a = un, vn, an
        umax = max(umax, abs(u))
    return w * w * umax  # in g, because the input was in g
