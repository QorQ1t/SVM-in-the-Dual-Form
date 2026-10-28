from numpy.testing import verbose
import numpy as np 
from barrier_template import solve
from projection import projected_gradient
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import rbf_kernel

X, y = make_moons(n_samples=200, noise=0.25 , random_state=0)
y = 2*y -1 

C = 1.0
gamma = 1.0


K = rbf_kernel(X, X, gamma=gamma)
Q = np.outer(y,y) * K
data = {
    "Q":Q,
    "y":y,
    "C":C,
    "n":len(y)
}

import time
x_bar, hist_bar = solve(data)

start = time.perf_counter()
x_pg, hist_pg = projected_gradient(data)
pg_time = time.perf_counter() - start

def objective(x):
    return 0.5 * x @ Q @ x - x.sum()

#тут просто сравнение результатов
print("Barrier objective:", objective(x_bar))
print("PG objective:", objective(x_pg))
print("Barrier equality:", abs(y @ x_bar))
print("PG equality:", abs(y @ x_pg))

#Тут высчитывается эталонная альфа, на которую мы дрочим и сравниваем значения
from sklearn.svm import SVC
clf = SVC(kernel='rbf', tol=1e-12, gamma=gamma, C=C)
clf.fit(X,y)
alpha_svc = np.zeros(len(y))
alpha_svc[clf.support_] = np.abs(clf.dual_coef_.ravel())


print("SVC objective:", objective(alpha_svc))
print("SVC equality:", abs(y @ alpha_svc))

def check_kkt(alpha, y, K, C, tol=1e-5):
    lower = alpha <= tol
    upper = alpha >= C - tol
    free = ~(lower | upper)
    scores = K @ (alpha * y)
    if not np.any(free):
        raise ValueError("No free support vectors")
    b = np.mean(y[free] - scores[free])
    margin = y * (scores + b)

    v_lower = np.max(np.maximum(1 - margin[lower], 0)) if np.any(lower) else 0.0
    v_free = np.max(np.abs(margin[free] - 1)) if np.any(free) else 0.0
    v_upper = np.max(np.maximum(margin[upper] - 1, 0)) if np.any(upper) else 0.0

    equality = abs(y @ alpha)
    box = max(0, -alpha.min(), alpha.max() - C)
    return max(v_lower, v_free, v_upper, equality, box)


print("Barrier KKT:", check_kkt(x_bar, y, K, C))
print("PG KKT:", check_kkt(x_pg, y, K, C))
print("SVC KKT:", check_kkt(alpha_svc, y, K, C))


print("Barrier SV:", np.sum(x_bar > 1e-6))
print("PG SV:", np.sum(x_pg > 1e-6))
print("SVC SV:", np.sum(alpha_svc > 1e-6))


print("Barrier outer iterations:", len(hist_bar["t"]))
print("Barrier Newton steps:", hist_bar["newton_total"][-1])
print("PG iterations:", len(hist_pg))
print("Barrier time:", hist_bar["time"][-1])
print("PG time:", pg_time)

print("Newton steps per outer iteration:", hist_bar["newton"])



import matplotlib.pyplot as plt
import numpy as np

f_opt = objective(alpha_svc)
gap_bar = np.abs(np.array(hist_bar["f"]) - f_opt)
gap_pg = np.abs(np.array(hist_pg) - f_opt)
iter_bar = np.array(hist_bar["newton_total"])
iter_pg = np.arange(1, len(gap_pg) + 1)
plt.figure(figsize=(10, 6))
plt.plot(
    iter_bar,
    np.maximum(gap_bar, 1e-15),
    marker="o",
    markersize=5,
    linewidth=2,
    label="Barrier Method"
)
plt.plot(
    iter_pg,
    np.maximum(gap_pg, 1e-15),
    linewidth=2,
    label="Projected Gradient"
)

# Logarithmic scales
plt.xscale("log")
plt.yscale("log")

# Labels
plt.xlabel("Newton Steps / PG Iterations")
plt.ylabel("Objective Gap")
plt.title("SVM Dual: Convergence Comparison")

plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
