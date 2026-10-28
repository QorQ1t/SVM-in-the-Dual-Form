import numpy as np

def projection(v, y, C):
    left = -50.0
    right = 50.0
    for _ in range(50):
        nu = (left + right) / 2
        alpha = np.clip(v - nu*y,0,C)
        phi = np.dot(y, alpha)
        if phi>0:
            left = nu
        else:
            right = nu
    nu = (left + right) / 2
    alpha = np.clip(v - nu * y, 0, C)
    return alpha

def projected_gradient(data, max_iter=20000):
    Q = data["Q"]
    y = data["y"]
    C = data["C"]
    n = data["n"]
    x = np.zeros(n)
    history = []
    L = np.linalg.eigvalsh(Q)[-1]
    t = 1 / L

    for k in range(max_iter):
        v = Q @ x - np.ones_like(x)
        x_new = x - t * v
        x_new = projection(x_new, y, C)
        residual = np.linalg.norm(x - x_new) / t
        objective_value = 0.5 * x_new @ Q @ x_new - x_new.sum()
        history.append(objective_value)
        x = x_new
        if residual < 1e-6:
            break

    return x, history


