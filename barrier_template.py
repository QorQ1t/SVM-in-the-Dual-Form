import time

import numpy as np

def barrier_method(f, grad, hess, G, h, x0, update_t, A=None, b=None,
                   t0=1.0, eps=1e-8,newton_tol=1e-9, max_newton=100,#они тут сука
                   max_outer=200, alpha=0.01, beta=0.5, verbose=False):
                   
    x = np.array(x0, dtype=float)
    G = np.asarray(G, dtype=float)
    h = np.asarray(h, dtype=float)
    n = x.size
    m = G.shape[0]
    has_eq = A is not None and np.size(A) > 0
    if has_eq:
        A = np.atleast_2d(np.asarray(A, dtype=float))
        b = np.atleast_1d(np.asarray(b, dtype=float))
        p = A.shape[0]
        resid = np.max(np.abs(A @ x - b))
        if resid > 1e-8 * (1.0 + np.max(np.abs(b))):
            raise ValueError("x0 violates the equality constraints "
                             "(max |Ax - b| = %.3g)" % resid)
    slack = h - G @ x
    if np.any(slack <= 0):
        raise ValueError("x0 must be STRICTLY feasible: G x0 < h "
                         "(smallest slack = %.3g)" % slack.min())

    def phi(z, t):
        """Barrier objective t*f(z) - sum log(h - Gz); +inf outside the domain."""
        s = h - G @ z
        if np.any(s <= 0):
            return np.inf
        return t * f(z) - np.sum(np.log(s))

    def center(x, t):
        """Newton's method with equality constraints for the centering problem."""
        steps = 0
        for _ in range(max_newton):
            s = h - G @ x
            d = 1.0 / s
            g = t * grad(x) + G.T @ d
            H = t * hess(x) + (G.T * d ** 2) @ G
            if has_eq:
                K = np.block([[H, A.T], [A, np.zeros((p, p))]])
                rhs = np.concatenate([-g, np.zeros(p)])
                dx = np.linalg.solve(K, rhs)[:n]
            else:
                dx = np.linalg.solve(H, -g)
            lam2 = -g @ dx                 
            if lam2 / 2.0 <= newton_tol or lam2 < 0.0:
                break
            phi0 = phi(x, t)
            step = 1.0
            while phi(x + step * dx, t) > phi0 - alpha * step * lam2:
                step *= beta
                if step < 1e-14:
                    break
            if step < 1e-14:                    # line search stalled
                break
            x = x + step * dx
            steps += 1
        return x, steps

    hist = {"t": [], "gap": [], "f": [], "newton": [], "newton_total": [],
            "time": []}
    t = float(t0)
    t_start = time.perf_counter()
    newton_total = 0
    for k in range(max_outer):
        x, steps = center(x, t)
        newton_total += steps
        hist["t"].append(t)
        hist["gap"].append(m / t)
        hist["f"].append(float(f(x)))
        hist["newton"].append(steps)
        hist["newton_total"].append(newton_total)
        hist["time"].append(time.perf_counter() - t_start)
        if verbose:
            print("outer %2d  t = %.3e  gap <= %.2e  f = %.8f  newton = %d"
                  % (k, t, m / t, hist["f"][-1], steps))
        if m / t <= eps:
            break
        t = float(update_t(t, k))
    return x, hist

def objective(x, data):
    Q = data["Q"]
    return (1/2)*x @ Q @ x - x.sum()
def gradient(x, data):
    Q = data["Q"]
    return Q@x - np.ones_like(x)
def hessian(x, data):
    Q = data["Q"]
    return Q
def constraints(data):
    n = data["n"]
    C = data["C"]
    y = data["y"]
    G = np.vstack((-np.eye(n), np.eye(n)))
    h = np.hstack((np.zeros(n), C * np.ones(n)))
    A = y.reshape(1, -1)
    b = np.array([0.0])
    return G, h, A, b
def initial_point(data):
    n = data["n"]
    C = data["C"]
    return np.ones(n) * C / 2
def update_t(t, k):
    mu = 10
    return mu * t


def solve(data, **kwargs):
    """Solve your problem with the provided barrier_method."""
    G, h, A, b = constraints(data)
    return barrier_method(lambda x: objective(x, data),
                          lambda x: gradient(x, data),
                          lambda x: hessian(x, data),
                          G, h, initial_point(data), update_t, A=A, b=b,
                          **kwargs)


