"""
Gaussian Basis Function (GBF) Smoothing for Local Projections
Implements smoothing methodology from Jordà & Taylor JEL
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
from scipy import stats
from typing import Tuple, Optional, Dict
import warnings


class GaussianBasisFunction:
    """
    Gaussian Basis Function (GBF) for IRF Smoothing

    The GBF parametric form:
        β(h) = a * exp(-(h - b)² / c²)

    Parameters
    ----------
    a : float, optional
        Amplitude (maximum response)
    b : float, optional
        Peak horizon (when response is largest)
    c : float, optional
        Width/persistence parameter

    Examples
    --------
    >>> gbf = GaussianBasisFunction()
    >>> gbf.fit(horizons, betas, ses)
    >>> smoothed = gbf.predict(horizons)
    """

    def __init__(self, a: float = None, b: float = None, c: float = None):
        self.a = a
        self.b = b
        self.c = c
        self.se_a = None
        self.se_b = None
        self.se_c = None
        self.cov_matrix = None

    @staticmethod
    def gbf_func(h: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Evaluate GBF at horizons h"""
        return a * np.exp(-((h - b) / c) ** 2)

    def fit(
        self,
        horizons: np.ndarray,
        betas: np.ndarray,
        ses: Optional[np.ndarray] = None,
        bounds: Optional[Tuple] = None,
        initial_guess: Optional[Tuple] = None
    ) -> 'GaussianBasisFunction':
        """
        Fit GBF parameters to estimated IRF

        Parameters
        ----------
        horizons : np.ndarray
            Horizon values
        betas : np.ndarray
            Point estimates at each horizon
        ses : np.ndarray, optional
            Standard errors (used as weights if provided)
        bounds : tuple, optional
            Parameter bounds ((a_min, b_min, c_min), (a_max, b_max, c_max))
        initial_guess : tuple, optional
            Initial parameter values (a0, b0, c0)

        Returns
        -------
        self : GaussianBasisFunction
            Fitted model
        """
        # Remove NaN values
        mask = ~(np.isnan(betas) | np.isnan(horizons))
        h = horizons[mask]
        b = betas[mask]

        if ses is not None:
            s = ses[mask]
            # Use inverse variance weighting
            sigma = s
        else:
            sigma = None

        # Default bounds
        if bounds is None:
            # a can be positive or negative, b should be within horizon range, c > 0
            bounds = (
                [-np.inf, 0, 0.1],
                [np.inf, horizons.max() * 1.5, horizons.max() * 2]
            )

        # Default initial guess
        if initial_guess is None:
            peak_idx = np.argmax(np.abs(b))
            initial_guess = (b[peak_idx], h[peak_idx], len(h) / 3)

        try:
            popt, pcov = curve_fit(
                self.gbf_func,
                h,
                b,
                p0=initial_guess,
                sigma=sigma,
                absolute_sigma=True if sigma is not None else False,
                bounds=bounds,
                maxfev=10000
            )

            self.a, self.b, self.c = popt
            self.cov_matrix = pcov

            # Standard errors from covariance matrix diagonal
            if pcov is not None:
                self.se_a = np.sqrt(pcov[0, 0])
                self.se_b = np.sqrt(pcov[1, 1])
                self.se_c = np.sqrt(pcov[2, 2])

        except Exception as e:
            warnings.warn(f"GBF fitting failed: {str(e)}")
            # Fall back to simple least squares
            self._fit_simple(h, b)

        return self

    def _fit_simple(self, horizons: np.ndarray, betas: np.ndarray):
        """Simple fitting without bounds"""
        def objective(params):
            a, b, c = params
            pred = self.gbf_func(horizons, a, b, c)
            return np.sum((betas - pred) ** 2)

        peak_idx = np.argmax(np.abs(betas))
        x0 = [betas[peak_idx], horizons[peak_idx], len(horizons) / 3]

        result = minimize(objective, x0, method='Nelder-Mead')
        self.a, self.b, self.c = result.x

    def predict(
        self,
        horizons: np.ndarray,
        return_ci: bool = False,
        alpha: float = 0.05
    ) -> np.ndarray:
        """
        Predict smoothed IRF values

        Parameters
        ----------
        horizons : np.ndarray
            Horizons to predict
        return_ci : bool, default=False
            If True, return confidence intervals
        alpha : float, default=0.05
            Significance level for CIs

        Returns
        -------
        np.ndarray or tuple : Predictions (and CIs if requested)
        """
        if self.a is None:
            raise ValueError("Model not fitted. Call fit() first.")

        pred = self.gbf_func(horizons, self.a, self.b, self.c)

        if return_ci and self.cov_matrix is not None:
            # Delta method for confidence intervals
            se_pred = self._compute_prediction_se(horizons)
            z = stats.norm.ppf(1 - alpha / 2)
            ci_lower = pred - z * se_pred
            ci_upper = pred + z * se_pred
            return pred, ci_lower, ci_upper

        return pred

    def _compute_prediction_se(self, horizons: np.ndarray) -> np.ndarray:
        """
        Compute standard errors for predictions using delta method

        ∂β/∂a = exp(-(h-b)²/c²)
        ∂β/∂b = 2a(h-b)/c² * exp(-(h-b)²/c²)
        ∂β/∂c = 2a(h-b)²/c³ * exp(-(h-b)²/c²)
        """
        se_pred = np.zeros(len(horizons))

        for i, h in enumerate(horizons):
            exp_term = np.exp(-((h - self.b) / self.c) ** 2)

            # Gradient vector
            grad = np.array([
                exp_term,  # ∂β/∂a
                2 * self.a * (h - self.b) / (self.c ** 2) * exp_term,  # ∂β/∂b
                2 * self.a * ((h - self.b) ** 2) / (self.c ** 3) * exp_term  # ∂β/∂c
            ])

            # Variance via delta method: Var(β(h)) = grad' * Cov * grad
            var_pred = grad @ self.cov_matrix @ grad
            se_pred[i] = np.sqrt(max(0, var_pred))

        return se_pred

    def summary(self) -> str:
        """Return summary of fitted parameters"""
        if self.a is None:
            return "Model not fitted"

        s = "GBF Parameter Estimates\n"
        s += "=" * 40 + "\n"
        s += f"  a (amplitude):     {self.a:.4f}"
        if self.se_a:
            s += f" ({self.se_a:.4f})"
        s += "\n"
        s += f"  b (peak horizon):  {self.b:.4f}"
        if self.se_b:
            s += f" ({self.se_b:.4f})"
        s += "\n"
        s += f"  c (width):         {self.c:.4f}"
        if self.se_c:
            s += f" ({self.se_c:.4f})"
        s += "\n"

        return s


def smooth_irf(
    horizons: np.ndarray,
    betas: np.ndarray,
    ses: Optional[np.ndarray] = None,
    method: str = 'gbf',
    **kwargs
) -> Dict:
    """
    Smooth IRF estimates using various methods

    Parameters
    ----------
    horizons : np.ndarray
        Horizon values
    betas : np.ndarray
        Point estimates
    ses : np.ndarray, optional
        Standard errors
    method : str, default='gbf'
        Smoothing method: 'gbf', 'lowess', 'polynomial'
    **kwargs : dict
        Additional arguments passed to smoothing method

    Returns
    -------
    dict : Dictionary with smoothed values and parameters
    """
    if method == 'gbf':
        gbf = GaussianBasisFunction()
        gbf.fit(horizons, betas, ses)

        pred, ci_lower, ci_upper = gbf.predict(
            horizons,
            return_ci=True,
            alpha=kwargs.get('alpha', 0.05)
        )

        return {
            'smoothed': pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'params': {'a': gbf.a, 'b': gbf.b, 'c': gbf.c},
            'se_params': {'a': gbf.se_a, 'b': gbf.se_b, 'c': gbf.se_c},
            'model': gbf
        }

    elif method == 'lowess':
        from statsmodels.nonparametric.smoothers_lowess import lowess

        frac = kwargs.get('frac', 0.3)
        smoothed = lowess(betas, horizons, frac=frac, return_sorted=False)

        return {
            'smoothed': smoothed,
            'ci_lower': None,
            'ci_upper': None,
            'params': {'frac': frac},
            'model': None
        }

    elif method == 'polynomial':
        degree = kwargs.get('degree', 3)
        coeffs = np.polyfit(horizons, betas, degree)
        smoothed = np.polyval(coeffs, horizons)

        return {
            'smoothed': smoothed,
            'ci_lower': None,
            'ci_upper': None,
            'params': {'degree': degree, 'coeffs': coeffs},
            'model': None
        }

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def plot_gbf_illustration(
    a: float = 1.0,
    b: float = 8.0,
    c: float = 8.0,
    horizon: int = 30,
    ax=None
):
    """
    Plot illustration of GBF parameters (replicates Example 3)

    Parameters
    ----------
    a : float
        Amplitude
    b : float
        Peak horizon
    c : float
        Width
    horizon : int
        Maximum horizon to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    matplotlib.axes.Axes : Plot axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    h = np.arange(horizon + 1)
    response = a * np.exp(-((h - b) / c) ** 2)

    ax.plot(h, response, 'b-', linewidth=2, label='GBF Response')
    ax.axhline(y=a, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=a / 2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=b, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Add parameter annotations
    ax.text(
        0.75, 0.85,
        f'GBF parameters:\na = {a}\nb = {b}\nc = {c}',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_xlabel('Horizon, h', fontsize=12)
    ax.set_ylabel('Response, φ(h; Θ)', fontsize=12)
    ax.set_ylim(0, max(1.3, a * 1.3))
    ax.set_xlim(0, horizon)
    ax.grid(True, alpha=0.3)

    return ax
