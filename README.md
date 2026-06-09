# npearth

A NumPy-based implementation of the Multivariate Adaptive Regression Splines (EARTH) algorithm, following Friedman’s original paper, with support for weighted regression. The goal of this project is to provide a lightweight, efficient, and transparent implementation suitable for research, numerical experiments, and scientific computing. For an introduction to the EARTH model please read https://w.wiki/GVPL and Chapter 9.4 in *The Elements of Statistical Learning* by Friedman, Tibshirani and Hastie

## Overview

`npearth` implements the main components of the EARTH algorithm:

* Forward pass using hinge functions and interaction terms with RMSE as loss function
* Backward pruning based on Generalized Cross Validation (GCV)
* Efficient regression fitting using Cholesky updates
* Weighted regression support

The package is implemented using NumPy and Numba, with a focus on clarity, performance, and maintainability.

## Installation

The package is currently distributed directly from GitHub. Install using:

```
pip install git+https://github.com/bjarkeh97/npearth  
```

### Platform notes

`npearth` depends on `numba` and `llvmlite`. Pre-built wheels exist for Linux (x86_64, aarch64), Windows (amd64), and Apple Silicon macOS (arm64). For these platforms, `pip install` works directly.

**Intel macOS (x86_64):** Numba's official `llvmlite` wheels are not published for Intel Macs from `llvmlite 0.46` onwards. There are two options:

- Stay on `numba <= 0.62.x` (the last version with Intel Mac wheels). This is the easiest path and what this project currently uses for local development on Intel Macs.
- Use [conda-forge](https://conda-forge.org/), which provides pre-built Intel Mac binaries:
  ```
  conda install -c conda-forge numba
  ```

## Quick start

```
from npearth.earth import EARTH
import numpy as np
from time import time as timer

np.random.seed(42)  # For reproducibility
N = 1000
X = np.random.rand(N, 4)  # N samples, 3 features, one noisy feature
y = (
    4 + 2 * X[:, 0] + 3 * X[:, 1] - 1 * X[:, 2] + 0.0 * np.random.randn(N)
)  # Linear combination with noise

# Step 2: Create an instance of EARTH and fit the model
earth_model = EARTH(M_max=8)
t0 = timer()
earth_model.fit(X, y)
print("Took time ", round(timer() - t0, 3), "seconds")
print("earth coefs ", earth_model.coef_)
# Step 3: Make predictions on the same input
y_pred = earth_model.predict(X)

# Step 4: Print the results (printing just the first 5 for brevity)
print("Original y values (first 5):", y[:5])
print("Predicted y values (first 5):", y_pred[:5])
print("SSR", ((y - y_pred) ** 2).sum())
```


## Project Structure

```
npearth/
  earth.py                     High-level EARTH model class
  _forward_pass.py             Forward pass implementation
  _backward_pass.py            Backward pruning and model simplification
  _basis_function.py           Basis function representation
  _knotsearcher_*              Knot search strategies and optimizations
  _cholesky_update.py          Rank-1 Cholesky update utilities
  data/                        Sample datasets for testing and examples
examples/                      Example scripts
tests/                         Unit tests
```

## Examples

Example scripts demonstrating typical use cases are available in the `examples/` directory. Examples might need you to install eg scikit-learn, matplotlib for running the notebook but should be straight forward:

* `example_linear.py`   # Very simple example also given in quick start
* `example_sine.py`     # Fit a sine function and compare to Random Forest and MLP
* `example_complex_multidim.py`   # More complex fit and show the power of EARTH
* `example_weights.py`      # Weighted regression on heteroscedastic data
* `generating_confidence_bands.ipynb`   # Simple example of how to generate confidence bands from epistemic uncertainty in model parameters

## Contributing

Contributions are welcome. Areas where contributions are particularly helpful include improvements to numerical performance, additional diagnostics, enhanced loss functions, and expanded test coverage.

To contribute:

1. Fork the repository on GitHub
2. Create a branch for your feature or fix
3. Commit and push your changes
4. Submit a pull request

## Roadmap

The following items could be interesting for future releases:

* Packaging and release on PyPI
* Model diagnostics, including ANOVA decompositions as in Friedman (1991) 
  ** Plot per variable and two-interactions
* Feature importance
* Additional statistical summaries and tools like confidence intervals (Not true as this is not BMARS)
* Bagging
* Handle nan features -  how?

## License

This project is licensed under the MIT License.

## References
This implementation is inspired by Chapter 9.4 in *The Elements of Statistical Learning* by Friedman, Tibshirani and Hastie.

Some references for the implementation:

1) Jerome H. Friedman (1991): Multivariate Adaptive Regression Splines, The Annals of Statistics, 19(1), 1–67.
2) The py-earth project: https://github.com/scikit-learn-contrib/py-earth
3) Stephen Milborrow’s notes (http://www.milbo.org/doc/earth-notes.pdf) from the R package (http://CRAN.R-project.org/package=earth)

(1) was the primary source for implementing the model. (2) provided valuable insights with its sources and project structure. Given that this project was also an EARTH implementation but is no longer actively maintained (it seems) and is written in Cython, it inspired me to do this project primarily using NumPy/Numba for better maintainability. (3) explained some ideas about the implementation.