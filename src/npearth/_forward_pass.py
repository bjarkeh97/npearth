import numpy as np
from npearth._basis_function import BasisFunction, BasisMatrix
from npearth._knotsearcher_base import KnotSearcherBase
from npearth._knotsearcher_svd import KnotSearcherSVD
from npearth._knotsearcher_cholesky import KnotSearcherCholesky
from npearth._knotsearcher_cholesky_numba import KnotSearcherCholeskyNumba
from typing import Optional, Type
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class ForwardPasser:
    def __init__(self) -> None:
        self.coefs = []
        self.bx = None

    def forward_pass(
        self,
        X: np.ndarray,
        y: np.ndarray,
        M_max: float,
        sample_weights: np.ndarray,
        KnotSearcher: Type[KnotSearcherBase] = KnotSearcherCholeskyNumba,
        ridge: float = 1e-8,
    ) -> tuple[list, list[BasisFunction]]:
        bx = BasisMatrix(X=X)
        M = 1
        coeffs_star = [np.sum(y * sample_weights) / np.sum(sample_weights)]
        lof_star = sum(sample_weights * (y - coeffs_star[0])**2)
        while M <= M_max:
            last_lof = lof_star
            m_star, v_star, t_star = None, None, None
            for m in range(M):
                vs_in_m = bx.basis[m].return_variables_used()
                vs_not_in_m = [v for v in range(bx.n) if v not in vs_in_m]
                for v in vs_not_in_m:
                    active_basis_mask = bx.bx[:, m] > 0
                    xv = X[:, v]
                    ts = X[active_basis_mask, v]
                    knot_searcher = KnotSearcher(
                        deepcopy(bx), y, xv, m, v, sample_weights, ridge
                    )
                    lof, t, coeffs = knot_searcher.search_over_knots(ts, lof_star)
                    if lof < lof_star:
                        lof_star = lof
                        m_star = m
                        v_star = v
                        t_star = t
                        coeffs_star = coeffs
            if lof_star == last_lof:
                logger.info("Forward pass terminated at %d basis functions (LOF=%.6g)", M, lof_star)
                # self.coefs = coeffs_star
                # self.bx = bx
                # return self.coefs, bx.basis
                break
            bx.add_split_end(m_star, v_star, t_star)
            M += 2

        self.coefs = coeffs_star
        self.bx = bx
        return self.coefs, bx.basis
