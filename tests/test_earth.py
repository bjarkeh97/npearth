# tests/test_earth.py
import numpy as np
import pytest
from npearth.earth import EARTH

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def gen_linear(n=200, noise=0.0):
    """Produces a simple linear relation"""
    X = np.random.randn(n, 1)
    y = 3.0 * X[:, 0] + 2.0 + noise * np.random.randn(n)
    return X, y


def gen_hinge(n=300, noise=0.0):
    """Produces on noisy dimension and a single hinge"""
    X = np.linspace(-2, 2, n).reshape(-1, 1)
    y = np.maximum(0, X[:, 0] - 0.3) * 4 + noise * np.random.randn(n)
    return X, y


# ---------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------


def test_fit_and_predict_linear():
    X, y = gen_linear(noise=0.0, n=100)
    model = EARTH(M_max=8, prune_model=True).fit(X, y)

    y_pred = model.predict(X)
    r2 = model.score(X, y)

    assert r2 > 0.99
    assert y_pred.shape == y.shape
    assert len(model.coef_) == len(model.basis_)


def test_fit_hinge_function():
    X, y = gen_hinge(noise=0.01)
    model = EARTH(M_max=10).fit(X, y)

    y_pred = model.predict(X)
    r2 = model.score(X, y)

    assert r2 > 0.95


def test_pruning_reduces_complexity():
    X, y = gen_hinge(noise=0.05)

    m_raw = EARTH(M_max=12, prune_model=False).fit(X, y)
    m_pruned = EARTH(M_max=12, prune_model=True).fit(X, y)

    assert len(m_pruned.basis_) <= len(m_raw.basis_)
    assert len(m_pruned.coef_) == len(m_pruned.basis_)


def test_sample_weights():
    X, y = gen_linear(noise=0.0, n=150)
    huge_noise = np.random.normal(0, 2, 50)
    y[:50] += huge_noise

    # High variance points downweighted
    sample_weight = np.ones_like(y)
    sample_weight[:50] = (
        0  # very noisy region gets tiny weight essentially removing the "bad" fit
    )

    m = EARTH(M_max=5).fit(X, y, sample_weight=sample_weight)
    r2 = m.score(X, y, sample_weight=sample_weight)

    assert r2 > 0.9

def test_sklearn_api_params():
    m = EARTH(M_max=5, ridge=1e-5, prune_model=False)

    params = m.get_params()
    assert params["M_max"] == 5
    assert params["ridge"] == 1e-5
    assert params["prune_model"] is False

    m.set_params(M_max=10)
    assert m.M_max == 10


def test_predict_before_fit_raises():
    m = EARTH()
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((10, 2)))


def test_score_behaviour():
    X, y = gen_linear()
    m = EARTH().fit(X, y)
    assert m.score(X, y) > 0.99


def test_coefficient_dimensions():
    X, y = gen_hinge()
    m = EARTH(M_max=10).fit(X, y)
    assert m.coef_.shape == (len(m.basis_),)
