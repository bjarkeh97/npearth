from __future__ import annotations

import numpy as np
from typing import Optional, Type
from npearth._forward_pass import ForwardPasser
from npearth._knotsearcher_base import KnotSearcherBase
from npearth._knotsearcher_cholesky import KnotSearcherCholesky
from npearth._knotsearcher_cholesky_numba import KnotSearcherCholeskyNumba
from npearth._backward_pass import BackwardsStepwise
from npearth._basis_function import BasisFunction


class EARTH:
    def __init__(
        self,
        M_max: int = 15,
        ridge: float = 1e-6,
        d: float = 3,
        knot_searcher: Type[KnotSearcherBase] = KnotSearcherCholeskyNumba,
        prune_model: bool = True,
    ) -> None:
        """
        Multivariate Adaptive Regression Splines (EARTH) regressor [1].

        A lightweight sklearn-compatible implementation of the EARTH algorithm.
        The model fits a series of basis functions (hinge functions) using a forward–pass and optionally applies pruning.

        Parameters
        ----------
        M_max : int, default=15
            Maximum number of basis functions to generate during the forward pass.

        ridge : float, default=1e-6
            Ridge regularization strength used when fitting coefficients as basis functions are not nessecarily orthogonal duing forward-pass

        d : float, default = 3
            Smoothing parameter for backward pass penalizing model complexity in LOF

        knot_searcher : KnotSearcherBase, default=KnotSearcherCholeskyNumba
            Class used for knot searching during the forward pass. Other classes are KnotSearcherSVD using SVD for parameter fitting but much slower (at least 100X)

        prune_model : bool, default=True
            If True, run the backward‐pass pruning procedure after the forward step. Prunes using the GCV score in [Friedman, J. (1991)]

        Attributes
        ----------
        coef_ : ndarray of shape (n_basis,)
            Coefficients of the fitted basis functions.

        basis_ : list
            List of fitted basis function objects.

        n_features_in_ : int
            Number of features in the input data seen during fit.

        References
        ----------

        [1] Friedman, Jerome. Multivariate Adaptive Regression Splines.
            Annals of Statistics. Volume 19, Number 1 (1991), 1-67.
        """
        self.M_max = M_max
        self.ridge = ridge
        self.knot_searcher = knot_searcher
        self.prune_model = prune_model
        self.d = d
        self.coef_: Optional[list] = None
        self.basis_: Optional[list[BasisFunction]] = None

    def _get_sample_weight(
        self, sample_weight: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Validate and broadcast sample weights.

        Parameters
        ----------
        sample_weight : array-like or None
            Sample weights provided by user. Should theoretically be 1/sigma_i^2

        y : array-like
            Target vector.

        Returns
        -------
        sample_weight : ndarray
            Validated weights, with ones if None was provided.

        Raises
        ------
        ValueError
            If weight array has wrong size or contains negative values.
        """
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        else:
            sample_weight = np.asarray(sample_weight)
            if len(sample_weight) != len(y):
                raise ValueError("Sample weights should have same dimensions as y")
            if np.any(sample_weight < 0):
                raise ValueError("Sample weights must be positive")
        return sample_weight

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> EARTH:
        """
        Fit the EARTH model.
        Employs algorithm 2 (ForwardPasser) and 3 (BackwardsStepwise) in [1]. Pruning is not needded for fitting model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Optional weights for each sample. Theoretically `1 / sigma_i^2` for
            heteroskedastic normal regression.

        Returns
        -------
        self : EARTH
            Fitted estimator.

        References
        ----------

        [1] Friedman, Jerome. Multivariate Adaptive Regression Splines.
            Annals of Statistics. Volume 19, Number 1 (1991), 1-67.
        """
        X = np.asarray(X)  # Validate input
        y = np.asarray(y)  # Validate input
        sample_weight = self._get_sample_weight(sample_weight, y)
        self.coef_, self.basis_ = ForwardPasser().forward_pass(
            X,
            y,
            self.M_max,
            sample_weight,
            self.knot_searcher,
            self.ridge,
        )
        if self.prune_model:
            pruner = BackwardsStepwise(ridge=self.ridge, d=self.d)
            self.basis_, self.coef_ = pruner.backward_pass(
                self.basis_, X, y, sample_weight
            )
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted EARTH model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.

        Raises
        ------
        RuntimeError
            If model is used before calling `fit`.
        """
        if self.basis_ is None:
            raise RuntimeError("Model not fitted yet")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        bx = np.column_stack([b.evaluate(X) for b in self.basis_])
        y_hat = bx @ self.coef_
        return y_hat

    def score(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute R² score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values.

        sample_weight : array-like, default=None
            Optional sample weights.

        Returns
        -------
        score : float
            Weighted coefficient of determination (R²).
        """
        y_pred = self.predict(X)
        sample_weight = self._get_sample_weight(sample_weight, y)
        w_sum = np.sum(sample_weight)
        if w_sum == 0:
            return np.nan
        y_mean_weighted = np.sum(y * sample_weight) / w_sum
        u = (((y - y_pred) ** 2) * sample_weight).sum()
        v = (((y - y_mean_weighted) ** 2) * sample_weight).sum()
        if v > 0:
            return 1 - u / v
        else:
            return np.nan

    def get_params(self, deep=True):
        return {
            "M_max": self.M_max,
            "ridge": self.ridge,
            "knot_searcher": self.knot_searcher,
            "prune_model": self.prune_model,
            "d": self.d,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
