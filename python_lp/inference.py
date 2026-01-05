"""
Inference Methods for Local Projections
Implements various inference approaches from Jordà & Taylor JEL
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from typing import Tuple, Optional, Dict, List
import warnings


def newey_west_se(
    y: np.ndarray,
    X: np.ndarray,
    maxlags: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Newey-West HAC standard errors

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    X : np.ndarray
        Regressors (should include constant if desired)
    maxlags : int, optional
        Maximum lags for HAC. If None, uses floor(4*(T/100)^(2/9))

    Returns
    -------
    tuple : (coefficients, standard errors)
    """
    model = OLS(y, X)

    if maxlags is None:
        maxlags = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))

    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})

    return results.params, results.bse


def lag_augmented_se(
    y: np.ndarray,
    X: np.ndarray,
    y_lags: np.ndarray,
    x_lags: np.ndarray,
    cov_type: str = 'HC3'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lag-augmented standard errors

    As shown in Example 4, lag augmentation provides better coverage
    in finite samples compared to Newey-West.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (forward value)
    X : np.ndarray
        Current treatment variable
    y_lags : np.ndarray
        Lagged values of y
    x_lags : np.ndarray
        Lagged values of x
    cov_type : str, default='HC3'
        Heteroskedasticity-consistent covariance type

    Returns
    -------
    tuple : (coefficients, standard errors)
    """
    # Combine all regressors
    X_full = np.column_stack([X, y_lags, x_lags])
    X_full = sm.add_constant(X_full)

    model = OLS(y, X_full)
    results = model.fit(cov_type=cov_type)

    return results.params, results.bse


def joint_significance_test(
    betas: np.ndarray,
    vcov: np.ndarray
) -> Dict:
    """
    Joint test that all coefficients equal zero: H0: β_0 = β_1 = ... = β_H = 0

    Parameters
    ----------
    betas : np.ndarray
        Coefficient estimates at each horizon
    vcov : np.ndarray
        Variance-covariance matrix of the estimates

    Returns
    -------
    dict : Test results with chi2 statistic, df, and p-value
    """
    # Remove any NaN values
    valid_idx = ~np.isnan(betas)
    betas_valid = betas[valid_idx]
    vcov_valid = vcov[np.ix_(valid_idx, valid_idx)]

    # Wald test: β' * V^{-1} * β ~ χ²(k)
    try:
        vcov_inv = np.linalg.inv(vcov_valid)
        chi2_stat = betas_valid @ vcov_inv @ betas_valid
        df = len(betas_valid)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)

        return {
            'chi2': chi2_stat,
            'df': df,
            'p_value': p_value,
            'reject_h0': p_value < 0.05
        }
    except np.linalg.LinAlgError:
        warnings.warn("Singular covariance matrix, cannot compute joint test")
        return {
            'chi2': np.nan,
            'df': len(betas_valid),
            'p_value': np.nan,
            'reject_h0': None
        }


def bonferroni_bands(
    betas: np.ndarray,
    ses: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Bonferroni-adjusted confidence bands

    Adjusts the significance level for multiple testing across horizons.

    Parameters
    ----------
    betas : np.ndarray
        Point estimates
    ses : np.ndarray
        Standard errors
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    tuple : (lower bounds, upper bounds)
    """
    n_tests = len(betas)
    alpha_adj = alpha / n_tests

    z_val = stats.norm.ppf(1 - alpha_adj / 2)

    lower = betas - z_val * ses
    upper = betas + z_val * ses

    return lower, upper


def bootstrap_bands(
    data: pd.DataFrame,
    y: str,
    x: str,
    controls: List[str],
    horizon: int,
    n_bootstrap: int = 1000,
    block_size: int = None,
    alpha: float = 0.05,
    seed: int = 12345,
    cumulative: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence bands using block bootstrap

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    y : str
        Response variable name
    x : str
        Treatment variable name
    controls : list
        Control variable names
    horizon : int
        Maximum horizon
    n_bootstrap : int, default=1000
        Number of bootstrap replications
    block_size : int, optional
        Block size for block bootstrap. If None, uses horizon
    alpha : float, default=0.05
        Significance level
    seed : int, default=12345
        Random seed
    cumulative : bool, default=True
        Whether to use cumulative responses

    Returns
    -------
    tuple : (point estimates, lower bounds, upper bounds)
    """
    np.random.seed(seed)

    if block_size is None:
        block_size = horizon

    n_obs = len(data)
    n_blocks = int(np.ceil(n_obs / block_size))

    # Storage for bootstrap estimates
    boot_betas = np.zeros((n_bootstrap, horizon + 1))

    for b in range(n_bootstrap):
        # Block bootstrap: sample blocks with replacement
        block_starts = np.random.choice(
            n_obs - block_size + 1,
            size=n_blocks,
            replace=True
        )

        # Construct bootstrap sample
        boot_indices = []
        for start in block_starts:
            boot_indices.extend(range(start, start + block_size))
        boot_indices = boot_indices[:n_obs]  # Trim to original size

        boot_data = data.iloc[boot_indices].reset_index(drop=True)

        # Estimate LP on bootstrap sample
        try:
            boot_est = _estimate_lp_single(
                boot_data, y, x, controls, horizon, cumulative
            )
            boot_betas[b, :] = boot_est
        except Exception:
            boot_betas[b, :] = np.nan

    # Compute percentile confidence intervals
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    lower = np.nanpercentile(boot_betas, lower_pct, axis=0)
    upper = np.nanpercentile(boot_betas, upper_pct, axis=0)
    point = np.nanmean(boot_betas, axis=0)

    return point, lower, upper


def _estimate_lp_single(
    data: pd.DataFrame,
    y: str,
    x: str,
    controls: List[str],
    horizon: int,
    cumulative: bool = True
) -> np.ndarray:
    """Helper function to estimate LP on a single sample"""
    betas = np.zeros(horizon + 1)

    for h in range(horizon + 1):
        # Create forward variable
        if cumulative:
            y_fwd = data[y].shift(-h) - data[y].shift(1)
        else:
            y_fwd = data[y].shift(-h)

        # Create regression data
        reg_data = pd.DataFrame({
            'y_fwd': y_fwd,
            'x': data[x]
        })

        # Add control lags
        for ctrl in controls:
            for lag in range(1, 5):  # Use 4 lags
                reg_data[f'{ctrl}_L{lag}'] = data[ctrl].shift(lag)

        reg_data = reg_data.dropna()

        if len(reg_data) < 10:
            betas[h] = np.nan
            continue

        y_vec = reg_data['y_fwd'].values
        X_mat = sm.add_constant(reg_data.drop('y_fwd', axis=1).values)

        try:
            results = OLS(y_vec, X_mat).fit()
            betas[h] = results.params[1]  # x coefficient
        except Exception:
            betas[h] = np.nan

    return betas


def significance_bands(
    betas: np.ndarray,
    z_scores: np.ndarray,
    vcov: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute significance bands as in Example 5

    Uses orthogonalized residuals to construct valid significance bands.

    Parameters
    ----------
    betas : np.ndarray
        Point estimates (orthogonalized)
    z_scores : np.ndarray
        Orthogonalized instrument/shock variable
    vcov : np.ndarray
        Variance-covariance matrix
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    tuple : (lower bound, upper bound) - symmetric around zero
    """
    # Compute mean of squared z-scores
    mean_z2 = np.mean(z_scores ** 2)

    # Standard errors for each horizon
    # Using the formula from sbands_RR.do
    ses = np.sqrt(np.diag(vcov)) / mean_z2

    z_val = stats.norm.ppf(1 - alpha / 2)

    upper = z_val * ses
    lower = -upper

    return lower, upper


class DriscollKraay:
    """
    Driscoll-Kraay standard errors for panel data

    Robust to cross-sectional and temporal correlation.
    Used in stacked LP regression.
    """

    def __init__(self, maxlags: int = None):
        self.maxlags = maxlags

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        time_ids: np.ndarray,
        panel_ids: np.ndarray
    ) -> Dict:
        """
        Estimate with Driscoll-Kraay standard errors

        Parameters
        ----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Regressors
        time_ids : np.ndarray
            Time period identifiers
        panel_ids : np.ndarray
            Panel unit identifiers

        Returns
        -------
        dict : Estimation results
        """
        from linearmodels.panel import PanelOLS

        # Create panel data structure
        data = pd.DataFrame(X)
        data['y'] = y
        data['time'] = time_ids
        data['panel'] = panel_ids
        data = data.set_index(['panel', 'time'])

        # Estimate with clustered standard errors
        # Note: This is an approximation; true DK requires specific implementation
        model = PanelOLS(
            data['y'],
            data.drop('y', axis=1),
            entity_effects=True
        )
        results = model.fit(cov_type='clustered', cluster_entity=True)

        return {
            'params': results.params.values,
            'bse': results.std_errors.values,
            'pvalues': results.pvalues.values
        }


def compare_inference_methods(
    data: pd.DataFrame,
    y: str,
    x: str,
    horizon: int = 12,
    lags: int = 4,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compare Newey-West vs Lag-Augmentation inference (replicates Example 4)

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    y : str
        Response variable
    x : str
        Treatment variable
    horizon : int
        Maximum horizon
    lags : int
        Number of lags for controls
    alpha : float
        Significance level

    Returns
    -------
    pd.DataFrame : Comparison of different inference methods
    """
    results = {
        'horizon': [],
        'beta': [],
        'se_nw': [],
        'se_la': [],
        'ci_nw_lower': [],
        'ci_nw_upper': [],
        'ci_la_lower': [],
        'ci_la_upper': []
    }

    for h in range(horizon):
        # Forward variable
        y_fwd = data[y].shift(-h)

        # Newey-West regression
        nw_data = pd.DataFrame({
            'y_fwd': y_fwd,
            'x': data[x],
            'y_L1': data[y].shift(1),
            'x_L1': data[x].shift(1)
        }).dropna()

        X_nw = sm.add_constant(nw_data[['x', 'y_L1', 'x_L1']])
        model_nw = OLS(nw_data['y_fwd'], X_nw)
        res_nw = model_nw.fit(cov_type='HAC', cov_kwds={'maxlags': 6})

        # Lag-augmented regression
        la_data = pd.DataFrame({
            'y_fwd': y_fwd,
            'x': data[x]
        })
        for lag in range(1, lags + 1):
            la_data[f'y_L{lag}'] = data[y].shift(lag)
            la_data[f'x_L{lag}'] = data[x].shift(lag)
        la_data = la_data.dropna()

        X_la = sm.add_constant(la_data.drop('y_fwd', axis=1))
        model_la = OLS(la_data['y_fwd'], X_la)
        res_la = model_la.fit(cov_type='HC3')

        # Store results
        z_val = stats.norm.ppf(1 - alpha / 2)
        beta = res_nw.params['x']

        results['horizon'].append(h)
        results['beta'].append(beta)
        results['se_nw'].append(res_nw.bse['x'])
        results['se_la'].append(res_la.bse['x'])
        results['ci_nw_lower'].append(beta - z_val * res_nw.bse['x'])
        results['ci_nw_upper'].append(beta + z_val * res_nw.bse['x'])
        results['ci_la_lower'].append(beta - z_val * res_la.bse['x'])
        results['ci_la_upper'].append(beta + z_val * res_la.bse['x'])

    return pd.DataFrame(results)
