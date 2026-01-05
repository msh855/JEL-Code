"""
Core Local Projections Implementation
Replicates STATA functionality for LP estimation
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.sandbox.regression.gmm import IV2SLS
from typing import Optional, List, Tuple, Dict, Union
import warnings


class LocalProjections:
    """
    Local Projections (LP) Estimation Class

    Implements the LP methodology from Jordà (2005) with extensions from
    Jordà & Taylor (JEL) including:
    - Level responses
    - Long-difference (cumulative) responses
    - LP-IV estimation
    - Multiple inference methods

    Parameters
    ----------
    data : pd.DataFrame
        Panel or time series data
    y : str
        Name of the response variable
    x : str
        Name of the treatment/shock variable
    controls : list of str, optional
        Names of control variables
    lags : int, default=4
        Number of lags to include for controls
    horizon : int, default=12
        Maximum forecast horizon
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        x: str,
        controls: Optional[List[str]] = None,
        lags: int = 4,
        horizon: int = 12
    ):
        self.data = data.copy()
        self.y = y
        self.x = x
        self.controls = controls if controls else []
        self.lags = lags
        self.horizon = horizon

        # Results storage
        self.betas = None
        self.ses = None
        self.ci_lower = None
        self.ci_upper = None
        self.pvalues = None

    def _create_forward_variables(self, cumulative: bool = True) -> pd.DataFrame:
        """Create forward-looking dependent variables for LP regression"""
        df = self.data.copy()

        for h in range(self.horizon + 1):
            if cumulative:
                # Long-difference: y_{t+h} - y_{t-1}
                df[f'{self.y}_f{h}'] = df[self.y].shift(-h) - df[self.y].shift(1)
            else:
                # Level: y_{t+h}
                df[f'{self.y}_f{h}'] = df[self.y].shift(-h)

        return df

    def _create_lag_controls(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create lagged control variables"""
        control_vars = []

        # Add lags of all control variables
        for var in self.controls:
            for lag in range(1, self.lags + 1):
                col_name = f'L{lag}_{var}'
                df[col_name] = df[var].shift(lag)
                control_vars.append(col_name)

        # Add lags of y and x
        for lag in range(1, self.lags + 1):
            col_name = f'L{lag}_{self.y}'
            df[col_name] = df[self.y].shift(lag)
            control_vars.append(col_name)

            col_name = f'L{lag}_{self.x}'
            df[col_name] = df[self.x].shift(lag)
            control_vars.append(col_name)

        return df, control_vars

    def estimate(
        self,
        cumulative: bool = True,
        cov_type: str = 'HAC',
        cov_kwds: Optional[dict] = None,
        alpha: float = 0.05
    ) -> Dict:
        """
        Estimate Local Projections

        Parameters
        ----------
        cumulative : bool, default=True
            If True, estimate cumulative (long-difference) responses
        cov_type : str, default='HAC'
            Covariance type: 'HAC' (Newey-West), 'HC3', 'cluster'
        cov_kwds : dict, optional
            Keyword arguments for covariance estimation
        alpha : float, default=0.05
            Significance level for confidence intervals

        Returns
        -------
        dict : Dictionary with estimation results
        """
        # Create forward variables
        df = self._create_forward_variables(cumulative=cumulative)
        df, control_vars = self._create_lag_controls(df)

        # Initialize results
        n_horizons = self.horizon + 1
        self.betas = np.zeros(n_horizons)
        self.ses = np.zeros(n_horizons)
        self.pvalues = np.zeros(n_horizons)

        # Default HAC options
        if cov_kwds is None:
            cov_kwds = {'maxlags': self.horizon}

        # Estimate for each horizon
        for h in range(n_horizons):
            y_var = f'{self.y}_f{h}'

            # Create regression data
            X_vars = [self.x] + control_vars
            reg_data = df[[y_var] + X_vars].dropna()

            if len(reg_data) < len(X_vars) + 2:
                warnings.warn(f"Insufficient observations at horizon {h}")
                continue

            y = reg_data[y_var]
            X = sm.add_constant(reg_data[X_vars])

            # Estimate
            model = OLS(y, X)

            if cov_type == 'HAC':
                results = model.fit(cov_type='HAC', cov_kwds=cov_kwds)
            elif cov_type == 'HC3':
                results = model.fit(cov_type='HC3')
            else:
                results = model.fit()

            # Store results for treatment variable
            self.betas[h] = results.params[self.x]
            self.ses[h] = results.bse[self.x]
            self.pvalues[h] = results.pvalues[self.x]

        # Compute confidence intervals
        z_val = stats.norm.ppf(1 - alpha / 2)
        self.ci_lower = self.betas - z_val * self.ses
        self.ci_upper = self.betas + z_val * self.ses

        return {
            'betas': self.betas,
            'ses': self.ses,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'pvalues': self.pvalues,
            'horizons': np.arange(n_horizons)
        }

    def estimate_iv(
        self,
        instrument: str,
        cumulative: bool = True,
        cov_type: str = 'HAC',
        cov_kwds: Optional[dict] = None,
        alpha: float = 0.05
    ) -> Dict:
        """
        Estimate Local Projections with Instrumental Variables (LP-IV)

        Parameters
        ----------
        instrument : str
            Name of the instrument variable
        cumulative : bool, default=True
            If True, estimate cumulative responses
        cov_type : str, default='HAC'
            Covariance type
        cov_kwds : dict, optional
            Keyword arguments for covariance estimation
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        dict : Dictionary with estimation results
        """
        from linearmodels.iv import IV2SLS as LM_IV2SLS

        # Create forward variables
        df = self._create_forward_variables(cumulative=cumulative)
        df, control_vars = self._create_lag_controls(df)

        # Initialize results
        n_horizons = self.horizon + 1
        self.betas = np.zeros(n_horizons)
        self.ses = np.zeros(n_horizons)
        self.pvalues = np.zeros(n_horizons)

        for h in range(n_horizons):
            y_var = f'{self.y}_f{h}'

            # Prepare data
            all_vars = [y_var, self.x, instrument] + control_vars
            reg_data = df[all_vars].dropna()

            if len(reg_data) < len(control_vars) + 5:
                warnings.warn(f"Insufficient observations at horizon {h}")
                continue

            # IV regression
            dependent = reg_data[y_var]
            exog = sm.add_constant(reg_data[control_vars]) if control_vars else None
            endog = reg_data[[self.x]]
            instruments = reg_data[[instrument]]

            try:
                model = LM_IV2SLS(dependent, exog, endog, instruments)
                results = model.fit(cov_type='robust')

                self.betas[h] = results.params[self.x]
                self.ses[h] = results.std_errors[self.x]
                self.pvalues[h] = results.pvalues[self.x]
            except Exception as e:
                warnings.warn(f"IV estimation failed at horizon {h}: {str(e)}")
                continue

        # Confidence intervals
        z_val = stats.norm.ppf(1 - alpha / 2)
        self.ci_lower = self.betas - z_val * self.ses
        self.ci_upper = self.betas + z_val * self.ses

        return {
            'betas': self.betas,
            'ses': self.ses,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'pvalues': self.pvalues,
            'horizons': np.arange(n_horizons)
        }


def orthogonalize(y: pd.Series, controls: pd.DataFrame) -> pd.Series:
    """
    Orthogonalize variable with respect to controls (Frisch-Waugh-Lovell)

    Parameters
    ----------
    y : pd.Series
        Variable to orthogonalize
    controls : pd.DataFrame
        Control variables

    Returns
    -------
    pd.Series : Residuals from regressing y on controls
    """
    data = pd.concat([y, controls], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = sm.add_constant(data.iloc[:, 1:])

    model = OLS(y_clean, X_clean)
    results = model.fit()

    residuals = pd.Series(index=y.index, dtype=float)
    residuals.loc[data.index] = results.resid

    return residuals


def create_cumulative_treatment(treatment: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Create cumulative treatment variables: sum from t to t+h

    Parameters
    ----------
    treatment : pd.Series
        Treatment variable
    horizon : int
        Maximum horizon

    Returns
    -------
    pd.DataFrame : DataFrame with cumulative treatments at each horizon
    """
    result = pd.DataFrame(index=treatment.index)

    for h in range(horizon + 1):
        # Cumulative sum from t to t+h
        cum_sum = treatment.copy()
        for j in range(1, h + 1):
            cum_sum = cum_sum + treatment.shift(-j)
        result[f'S{h}'] = cum_sum

    return result
