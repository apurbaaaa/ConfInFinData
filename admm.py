"""
admm.py

Minimal, practical ADMM implementations:
- admm_lasso: solves 0.5*||Ax - b||_2^2 + lambda*||x||_1
- consensus_admm_ls: consensus ADMM for distributed least-squares:
      minimize sum_i 0.5 ||A_i x_i - b_i||_2^2  subject to x_i = z  for all i

Author: generated for user
"""

from typing import Tuple, Optional, List
import numpy as np


# -------------------------------
# Utilities
# -------------------------------
def soft_threshold(v: np.ndarray, kappa: float) -> np.ndarray:
    """Element-wise soft thresholding."""
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)


# -------------------------------
# ADMM for LASSO
# -------------------------------
def admm_lasso(
    A: np.ndarray,
    b: np.ndarray,
    lmbda: float,
    rho: float = 1.0,
    alpha: float = 1.0,
    max_iter: int = 1000,
    abstol: float = 1e-4,
    reltol: float = 1e-3,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Solve LASSO via ADMM:
        minimize 0.5 * ||A x - b||_2^2 + lambda * ||x||_1

    Variables:
        x - primal variable
        z - auxiliary variable for L1
        u - scaled dual variable

    Returns (x, info)
    info contains objective, residuals, iterations.
    """
    m, n = A.shape
    AT = A.T
    ATb = AT.dot(b)

    # Pre-factorization: solve (A^T A + rho I) x = ATb + rho (z - u)
    # Use Cholesky or direct solve depending on dimension
    P = AT.dot(A) + rho * np.eye(n)
    try:
        L = np.linalg.cholesky(P)
        use_cholesky = True
    except np.linalg.LinAlgError:
        use_cholesky = False

    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    history = {"objval": [], "r_norm": [], "s_norm": [], "eps_pri": [], "eps_dual": []}

    for k in range(max_iter):
        # x-update (quadratic)
        q = ATb + rho * (z - u)  # right-hand side
        if use_cholesky:
            # solve P x = q via cholesky: L L^T x = q
            y = np.linalg.solve(L, q)
            x = np.linalg.solve(L.T, y)
        else:
            x = np.linalg.solve(P, q)

        # z-update with relaxation
        x_hat = alpha * x + (1 - alpha) * z
        z_old = z.copy()
        z = soft_threshold(x_hat + u, lmbda / rho)

        # u-update (scaled dual)
        u = u + (x_hat - z)

        # diagnostics, termination checks
        r_norm = np.linalg.norm(x - z)
        s_norm = np.linalg.norm(-rho * (z - z_old))

        obj = 0.5 * np.linalg.norm(A.dot(x) - b) ** 2 + lmbda * np.linalg.norm(z, 1)
        history["objval"].append(obj)
        history["r_norm"].append(r_norm)
        history["s_norm"].append(s_norm)

        eps_pri = np.sqrt(n) * abstol + reltol * max(np.linalg.norm(x), np.linalg.norm(-z))
        eps_dual = np.sqrt(n) * abstol + reltol * np.linalg.norm(rho * u)
        history["eps_pri"].append(eps_pri)
        history["eps_dual"].append(eps_dual)

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            print(
                f"iter {k:4d}: obj {obj:.5e}, r_norm {r_norm:.3e}, s_norm {s_norm:.3e}, eps_pri {eps_pri:.3e}"
            )

        if r_norm <= eps_pri and s_norm <= eps_dual:
            break

    return x, {"history": history, "iters": k + 1}


# -------------------------------
# Consensus ADMM for least-squares
# -------------------------------
def consensus_admm_ls(
    A_list: List[np.ndarray],
    b_list: List[np.ndarray],
    rho: float = 1.0,
    max_iter: int = 500,
    abstol: float = 1e-4,
    reltol: float = 1e-3,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Consensus ADMM:
        min sum_i 0.5 ||A_i x_i - b_i||_2^2  s.t. x_i = z for all i

    Returns consensus z and info dict.
    """
    num_workers = len(A_list)
    if num_workers == 0:
        raise ValueError("A_list must be non-empty")

    n = A_list[0].shape[1]
    x_list = [np.zeros(n) for _ in range(num_workers)]
    u_list = [np.zeros(n) for _ in range(num_workers)]
    z = np.zeros(n)

    # Precompute factorizations for each worker: solve (A_i^T A_i + rho I) x = A_i^T b_i + rho (z - u_i)
    P_list = []
    L_list = []
    ATb_list = []
    for A, b in zip(A_list, b_list):
        AT = A.T
        P = AT.dot(A) + rho * np.eye(n)
        ATb = AT.dot(b)
        ATb_list.append(ATb)
        try:
            L = np.linalg.cholesky(P)
            L_list.append(L)
            P_list.append(None)
        except np.linalg.LinAlgError:
            P_list.append(P)
            L_list.append(None)

    history = {"r_norm": [], "s_norm": [], "objval": []}

    for k in range(max_iter):
        # x-update: each worker solves its local problem
        for i in range(num_workers):
            ATb = ATb_list[i]
            rhs = ATb + rho * (z - u_list[i])
            if L_list[i] is not None:
                y = np.linalg.solve(L_list[i], rhs)
                x_list[i] = np.linalg.solve(L_list[i].T, y)
            else:
                x_list[i] = np.linalg.solve(P_list[i], rhs)

        # z-update: average of (x_i + u_i)
        z_old = z.copy()
        z = np.mean([x_list[i] + u_list[i] for i in range(num_workers)], axis=0)

        # u-update
        for i in range(num_workers):
            u_list[i] = u_list[i] + (x_list[i] - z)

        # diagnostics
        r_norm = np.sqrt(sum(np.linalg.norm(x_list[i] - z) ** 2 for i in range(num_workers)))
        s_norm = np.sqrt(sum(np.linalg.norm(-rho * (z - z_old)) ** 2 for _ in range(num_workers)))
        obj = sum(0.5 * np.linalg.norm(A_list[i].dot(x_list[i]) - b_list[i]) ** 2 for i in range(num_workers))

        history["r_norm"].append(r_norm)
        history["s_norm"].append(s_norm)
        history["objval"].append(obj)

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            print(f"iter {k:4d}: obj {obj:.5e}, r {r_norm:.3e}, s {s_norm:.3e}")

        eps_pri = np.sqrt(num_workers * n) * abstol + reltol * max(
            np.sqrt(sum(np.linalg.norm(x_list[i]) ** 2 for i in range(num_workers))),
            np.sqrt(num_workers) * np.linalg.norm(z),
        )
        eps_dual = np.sqrt(num_workers * n) * abstol + reltol * np.sqrt(sum(np.linalg.norm(rho * u_list[i]) ** 2 for i in range(num_workers)))

        if r_norm <= eps_pri and s_norm <= eps_dual:
            break

    return z, {"history": history, "iters": k + 1}


# -------------------------------
# Example usage / quick test
# -------------------------------
if __name__ == "__main__":
    # Quick LASSO test
    np.random.seed(0)
    m, n = 80, 30
    A = np.random.randn(m, n)
    x_true = np.zeros(n)
    x_true[:5] = np.array([1.5, -2.0, 0.5, 0.0, 3.0])
    b = A.dot(x_true) + 0.5 * np.random.randn(m)

    x_hat, info = admm_lasso(A, b, lmbda=0.5, rho=1.0, max_iter=200, verbose=True)
    print("LASSO recovered top coeffs:", np.round(x_hat[:10], 3))
    print("ADMM iters:", info["iters"])

    # Quick consensus test: split rows of A and b into 3 workers
    A_list = np.array_split(A, 3, axis=0)
    b_list = np.array_split(b, 3, axis=0)
    z, cinfo = consensus_admm_ls(A_list, b_list, rho=1.0, max_iter=200, verbose=True)
    print("Consensus z top:", np.round(z[:10], 3))
    print("Consensus iters:", cinfo["iters"])
