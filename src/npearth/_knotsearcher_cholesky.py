import numpy as np
from copy import copy

from npearth._cholesky_update import cholupdate, choldowndate, cholsolve
from npearth._knotsearcher_base import KnotSearcherBase
import logging

logger = logging.getLogger(__name__)

class KnotSearcherCholesky(KnotSearcherBase):

    def search_over_knots(self, ts: np.ndarray, lof_star: float) -> None:
        ssr_min = lof_star  # Initialize LOF to LOF^*
        sort_idx = np.argsort(self.xv)[
            ::-1
        ]  # Indices for sorting xv in descending order. Need u>=t for update to work

        # We should search from largest knot so:
        gn = copy(self.g)
        gn.add_split_end(self.m, self.v, max(self.xv))

        # W = np.diag(
        #     self.weights
        # )  # W_ij = \delta_ij w_i (for normal distribution assumption w_i = 1/\sigma_i)
        w = np.sqrt(self.sample_weight.astype(np.float64))

        # Sort all relevant matrices/vectors accordingly
        bx_sorted = np.ascontiguousarray(
            (gn.bx * w[:, None])[sort_idx, :].astype(np.float64)
        )  # Sort basis matrix accordingly
        y_sorted = np.ascontiguousarray(
            (self.y * w)[sort_idx].astype(np.float64)
        )  # Sort response vector accordingly
        # y_square = y_sorted @ y_sorted  # Precompute y^T y as is O(N)
        xv_sorted = np.ascontiguousarray(
            self.xv[sort_idx].astype(np.float64)
        )  # Sort predictor vector accordingly
        y_square = y_sorted @ y_sorted  # Precompute y^T y as is O(N)
        active_basis_mask = bx_sorted[:, self.m] > 0

        # rank = np.linalg.matrix_rank(bx_sorted)
        N, M = bx_sorted.shape

        V = (
            bx_sorted.transpose() @ bx_sorted
        )  # V made from sorted bx (shouldnt matter though) is MxM
        c = (
            bx_sorted.transpose() @ y_sorted
        )  # Also should matter if sorted as dot product is independent of order
        L = np.linalg.cholesky(V + self.ridge * np.eye(M))
        V_m_old = copy(V[:, M - 1])
        x_u = np.zeros(M)
        x_d = np.zeros(M)

        u = xv_sorted[0]
        last_i = 0
        last_last_i = 0
        a_best = np.full(M, np.nan)
        t_star = None

        # Initialize constants for updates
        C_u = 0
        V_u = np.zeros(M)
        D_u = 0
        F_u = 0

        skip_counter = 0

        for i, t in enumerate(xv_sorted):
            if i == N - 1 or i == 0:
                continue  # Skip first and last index
            if t == u:
                continue  # Dont recompute on same knot value
            if not active_basis_mask[i]:
                continue  # Only consider knots where basis m is active, meaning B_im > 0
            k_2 = range(last_i, i)  # k2 = {k|t<x_vk<=u}
            kc = range(
                last_last_i, last_i
            )  # kc = {k|u<x_vk<=last_u} for C,V,D,F updates
            # Update c:
            C_t = C_u + bx_sorted[kc, self.m] @ y_sorted[kc]
            C_u = C_t
            c[M - 1] = (
                c[M - 1]
                + bx_sorted[k_2, self.m] * y_sorted[k_2] @ (xv_sorted[k_2] - t)
                + (u - t) * C_t
            )
            # Update V:
            V_m_old[:] = V[:, M - 1]
            for j in range(M - 1):
                V_tj = bx_sorted[kc, self.m] @ bx_sorted[kc, j] + V_u[j]
                V_u[j] = V_tj
                V[j, M - 1] = (
                    V[j, M - 1]
                    + bx_sorted[k_2, self.m] * bx_sorted[k_2, j] @ (xv_sorted[k_2] - t)
                    + (u - t) * V_tj
                )
                V[M - 1, j] = V[j, M - 1]  # SPD
            # Update V[M,M]
            D_t = bx_sorted[kc, self.m] @ bx_sorted[kc, self.m] + D_u
            D_u = D_t
            F_t = (bx_sorted[kc, self.m] ** 2) @ xv_sorted[kc] + F_u
            F_u = F_t
            V[M - 1, M - 1] = (
                V[M - 1, M - 1]
                + (bx_sorted[k_2, self.m] ** 2) @ ((xv_sorted[k_2] - t) ** 2)
                + (t**2 - u**2) * D_t
                + 2 * (u - t) * F_t
            )

            # Now we have update c and V to knot t in O(1) time we solve the system.
            # We update the L matrix using cholupdate and choldowndate
            dVM = V[:, M - 1] - V_m_old
            if np.all(dVM == 0):
                u = t
                last_last_i = last_i
                last_i = i
                continue 
            denom = dVM[-1]
            if denom <= 0:
                u = t
                last_last_i = last_i
                last_i = i
                skip_counter += 1
                continue
            x_u = dVM / np.sqrt(denom)
            x_d = np.copy(x_u)
            x_d[M - 1] = 0
            L = choldowndate(cholupdate(L, x_u), x_d)

            # L = np.linalg.cholesky(V + self.ridge * np.eye(M))
            a = cholsolve(L, c)
            SSR = y_square - a @ c  # only O(M) now so in total (O(NM)) for all
            if SSR < 0:
                SSR = np.inf

            if SSR is not None and SSR < ssr_min:
                ssr_min = SSR
                a_best = a
                t_star = t
            u = t
            last_last_i = last_i
            last_i = i
        if skip_counter > 0:
            logger.info(
                f"Skipped {skip_counter} knots due to denom<=0. "
                f"(ridge={self.ridge:.1e}, m={self.m}, basis_size={M})"
            )
        return ssr_min, t_star, a_best


if __name__ == "__main__":
    # bx = np.genfromtxt("C:/Users/Bruger/Code/EARTH/src/earth/data/bx.csv")
    # y = np.genfromtxt("C:/Users/Bruger/Code/EARTH/src/earth/data/y.csv")

    from npearth._basis_function import BasisMatrix

    X = np.reshape([5, 4, 3, 2, 1, 1], (-1, 1))
    v = 0
    xv = X[:, v]

    print(f"X: {X}")

    noise = 0 * np.random.normal(0, 0.1, size=xv.shape)
    y = 2 + 3 * np.maximum(0, xv - 3) + noise
    print(f"y: {y}")

    bm = BasisMatrix(X)

    knts2 = KnotSearcherCholesky(bm, y, xv, 0, 0, np.ones_like(y), 1e-10)
    ssr, t, a = knts2.search_over_knots(xv, np.inf)
    print(f"t: {t}, ssr: {ssr}, a: {a}")

    for i, b in enumerate(bm.basis):
        print(b.svt)
