"""
A synthetic coordination substrate with a reflexivity knob, alpha in [0, 1].

This is the ground-truth harness: we build a system where we know, by
construction, whether the coordination signal tracks reality (exogenous) or
just recirculates the system's own past (reflexive). Then we check that the
diagnostic reads it correctly.

At each step t:

    w_t      ~ N(0, 1)                          the world (i.i.d. reality)
    m_t      = lam * m_{t-1} + (1 - lam) * x_t  reflexive recirculation: a moving
                                                average of PAST outcomes
    c_t      = (1 - alpha) * w_t + alpha * m_t  the coordination signal
    a_t      = delta * c_t + eta_t              agents act on the signal
    x_{t+1}  = beta * w_t + gamma * a_t + eps_t the next outcome

At alpha = 0 the signal is the current state of the world. At alpha = 1 it is
pure recirculation: it still varies (so it still correlates with outcomes),
but it carries nothing about the current w_t that drives the next outcome.
The reflexive component has to keep varying while staying uninformative about
the present; a frozen focal point, or one built from variables already in the
conditioning set, gives degenerate results instead.
"""

import numpy as np


def generate(alpha, n=20000, delta=1.0, beta=1.0, gamma=1.0,
             sigma_eta=1.2, sigma_eps=0.3, lam=0.8, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, n)
    eta = rng.normal(0, sigma_eta, n)
    eps = rng.normal(0, sigma_eps, n)
    x = np.zeros(n); c = np.zeros(n); a = np.zeros(n)
    x[0] = rng.normal()
    m = 0.0
    for t in range(n - 1):
        m = lam * m + (1 - lam) * x[t]
        c[t] = (1 - alpha) * w[t] + alpha * m
        a[t] = delta * c[t] + eta[t]
        x[t + 1] = beta * w[t] + gamma * a[t] + eps[t]
    return dict(x_t=x[:-1], a_t=a[:-1], c_t=c[:-1], x_next=x[1:])
