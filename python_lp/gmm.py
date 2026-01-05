"""
GMM Estimation for Local Projections
Implements joint inference and GBF estimation from Example 6
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Optional, List, Dict, Tuple
import warnings


class GMM_LP:
    """
    GMM Estimation for Local Projections

    Implements joint LP estimation across all horizons using GMM,
    allowing for proper joint inference.

    Moment conditions:
        E[Z_t * (y_{t+h} - β_h * x_t - c_h)] = 0 for all h

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    y : str
        Response variable name
    x : str
        Treatment/shock variable name
    z : str
        Instrument variable name
    controls : list, optional
        Control variable names
    horizon : int, default=48
        Maximum horizon
    lags : int, default=6
        Number of lags for controls and instruments
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        x: str,
        z: str,
        controls: Optional[List[str]] = None,
        horizon: int = 48,
        lags: int = 6
    ):
        self.data = data.copy()
        self.y = y
        self.x = x
        self.z = z
        self.controls = controls if controls else []
        self.horizon = horizon
        self.lags = lags

        # Results
        self.betas = None
        self.constants = None
        self.vcov = None
        self.ses = None

    def _orthogonalize(self) -> pd.DataFrame:
        """
        Orthogonalize variables with respect to controls (FWL)

        This mirrors the STATA code:
        - reg y_f{h} controls
        - predict ry_f{h}, resid
        """
        import statsmodels.api as sm

        df = self.data.copy()

        # Create control matrix
        control_cols = []
        for var in [self.y] + self.controls:
            for lag in range(1, self.lags + 1):
                col_name = f'{var}_L{lag}'
                df[col_name] = df[var].shift(lag)
                control_cols.append(col_name)

        # Orthogonalize forward y variables
        for h in range(self.horizon + 1):
            # Create forward variable: y_{t+h} - y_{t-1}
            df[f'{self.y}_f{h}'] = df[self.y].shift(-h) - df[self.y].shift(1)

            # Regress on controls
            reg_data = df[[f'{self.y}_f{h}'] + control_cols].dropna()
            if len(reg_data) > len(control_cols) + 2:
                X = sm.add_constant(reg_data[control_cols])
                model = sm.OLS(reg_data[f'{self.y}_f{h}'], X)
                results = model.fit()
                df[f'r{self.y}_f{h}'] = np.nan
                df.loc[reg_data.index, f'r{self.y}_f{h}'] = results.resid

        # Orthogonalize treatment
        reg_data = df[[self.x] + control_cols].dropna()
        if len(reg_data) > len(control_cols) + 2:
            X = sm.add_constant(reg_data[control_cols])
            model = sm.OLS(reg_data[self.x], X)
            results = model.fit()
            df['r_x'] = np.nan
            df.loc[reg_data.index, 'r_x'] = results.resid

        # Orthogonalize instrument
        reg_data = df[[self.z] + control_cols].dropna()
        if len(reg_data) > len(control_cols) + 2:
            X = sm.add_constant(reg_data[control_cols])
            model = sm.OLS(reg_data[self.z], X)
            results = model.fit()
            df['r_z'] = np.nan
            df.loc[reg_data.index, 'r_z'] = results.resid

        return df

    def estimate(
        self,
        weight_matrix: str = 'unadjusted',
        cov_type: str = 'HAC',
        maxlags: Optional[int] = None
    ) -> Dict:
        """
        Estimate LP-IV using GMM

        Parameters
        ----------
        weight_matrix : str, default='unadjusted'
            Initial weight matrix type
        cov_type : str, default='HAC'
            Covariance type for standard errors
        maxlags : int, optional
            Maximum lags for HAC

        Returns
        -------
        dict : Estimation results
        """
        # Orthogonalize variables
        df = self._orthogonalize()

        # Prepare data for GMM
        y_cols = [f'r{self.y}_f{h}' for h in range(self.horizon + 1)]
        all_cols = y_cols + ['r_x', 'r_z']

        # Add lagged instruments
        for lag in range(1, self.lags + 1):
            df[f'r_z_L{lag}'] = df['r_z'].shift(lag)
            all_cols.append(f'r_z_L{lag}')

        gmm_data = df[all_cols].dropna()
        T = len(gmm_data)

        if maxlags is None:
            maxlags = self.lags

        # Extract data matrices
        Y = gmm_data[y_cols].values  # T x (H+1)
        x = gmm_data['r_x'].values   # T
        z = gmm_data['r_z'].values   # T

        # Instruments: z and its lags
        z_cols = ['r_z'] + [f'r_z_L{lag}' for lag in range(1, self.lags + 1)]
        Z = gmm_data[z_cols].values  # T x (lags+1)

        # Two-step GMM
        # Step 1: Initial estimates with identity weight matrix
        betas_init = np.zeros(self.horizon + 1)
        for h in range(self.horizon + 1):
            # Simple IV: β = (Z'X)^{-1} Z'Y
            ZX = Z.T @ x
            ZY = Z.T @ Y[:, h]
            if np.abs(ZX.sum()) > 1e-10:
                betas_init[h] = (ZX @ ZY) / (ZX @ ZX)

        # Step 2: Compute optimal weight matrix with HAC
        # Residuals: e_h = y_h - β_h * x
        E = Y - np.outer(x, betas_init)  # T x (H+1)

        # Moment conditions: g_t = Z_t * e_{t,h}
        n_moments = Z.shape[1] * (self.horizon + 1)

        # Compute HAC covariance of moments
        S = self._compute_hac_cov(E, Z, maxlags)

        # Step 2 estimates with optimal weight
        betas = self._gmm_step(Y, x, Z, S)

        # Standard errors
        # Jacobian: ∂g/∂β = -E[Z * x]
        G = np.zeros((n_moments, self.horizon + 1))
        for h in range(self.horizon + 1):
            for j in range(Z.shape[1]):
                G[h * Z.shape[1] + j, h] = -np.mean(Z[:, j] * x)

        # Variance: (G' S^{-1} G)^{-1}
        try:
            S_inv = np.linalg.inv(S)
            vcov = np.linalg.inv(G.T @ S_inv @ G) / T
            ses = np.sqrt(np.diag(vcov))
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in variance computation")
            vcov = np.eye(self.horizon + 1) * np.nan
            ses = np.ones(self.horizon + 1) * np.nan

        self.betas = betas
        self.vcov = vcov
        self.ses = ses

        # Joint test: H0: all β = 0
        try:
            vcov_inv = np.linalg.inv(vcov)
            chi2_stat = betas @ vcov_inv @ betas
            df = self.horizon + 1
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        except:
            chi2_stat = np.nan
            p_value = np.nan
            df = self.horizon + 1

        # Confidence intervals
        z_val = 1.96
        ci_lower = betas - z_val * ses
        ci_upper = betas + z_val * ses

        return {
            'betas': betas,
            'ses': ses,
            'vcov': vcov,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_1se_lower': betas - ses,
            'ci_1se_upper': betas + ses,
            'horizons': np.arange(self.horizon + 1),
            'joint_test': {
                'chi2': chi2_stat,
                'df': df,
                'p_value': p_value
            }
        }

    def _compute_hac_cov(
        self,
        E: np.ndarray,
        Z: np.ndarray,
        maxlags: int
    ) -> np.ndarray:
        """Compute HAC covariance matrix of moment conditions"""
        T = E.shape[0]
        H = E.shape[1]
        n_z = Z.shape[1]
        n_moments = n_z * H

        # Moment conditions at each t
        G = np.zeros((T, n_moments))
        for h in range(H):
            for j in range(n_z):
                G[:, h * n_z + j] = Z[:, j] * E[:, h]

        # Newey-West HAC
        S = G.T @ G / T

        for lag in range(1, maxlags + 1):
            weight = 1 - lag / (maxlags + 1)
            Gamma = G[lag:].T @ G[:-lag] / T
            S += weight * (Gamma + Gamma.T)

        return S

    def _gmm_step(
        self,
        Y: np.ndarray,
        x: np.ndarray,
        Z: np.ndarray,
        S: np.ndarray
    ) -> np.ndarray:
        """Single GMM estimation step"""
        H = Y.shape[1]
        n_z = Z.shape[1]

        betas = np.zeros(H)

        try:
            S_inv = np.linalg.inv(S)
        except:
            S_inv = np.eye(S.shape[0])

        # Solve for each horizon (block diagonal structure)
        for h in range(H):
            idx = slice(h * n_z, (h + 1) * n_z)

            # Normal equations for horizon h
            ZX = Z.T @ x
            ZY = Z.T @ Y[:, h]

            # Weighted least squares
            W = S_inv[idx, idx]
            num = ZX @ W @ ZY
            den = ZX @ W @ ZX

            if np.abs(den) > 1e-10:
                betas[h] = num / den

        return betas


class GMM_GBF:
    """
    GMM Estimation with Gaussian Basis Function Smoothing

    Directly estimates GBF parameters (a, b, c) in the GMM framework,
    providing correct standard errors for the smoothed IRF.

    IRF form: β(h) = a * exp(-(h - b)² / c²)

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    y : str
        Response variable name
    x : str
        Treatment variable name
    z : str
        Instrument variable name
    controls : list, optional
        Control variable names
    horizon : int, default=48
        Maximum horizon
    lags : int, default=6
        Number of lags
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        x: str,
        z: str,
        controls: Optional[List[str]] = None,
        horizon: int = 48,
        lags: int = 6
    ):
        self.data = data.copy()
        self.y = y
        self.x = x
        self.z = z
        self.controls = controls if controls else []
        self.horizon = horizon
        self.lags = lags

        # GBF parameters
        self.a = None
        self.b = None
        self.c = None
        self.se_a = None
        self.se_b = None
        self.se_c = None

    def _orthogonalize(self) -> pd.DataFrame:
        """Orthogonalize variables (same as GMM_LP)"""
        import statsmodels.api as sm

        df = self.data.copy()

        # Create control matrix
        control_cols = []
        for var in [self.y] + self.controls:
            for lag in range(1, self.lags + 1):
                col_name = f'{var}_L{lag}'
                df[col_name] = df[var].shift(lag)
                control_cols.append(col_name)

        # Orthogonalize forward y variables
        for h in range(self.horizon + 1):
            df[f'{self.y}_f{h}'] = df[self.y].shift(-h) - df[self.y].shift(1)

            reg_data = df[[f'{self.y}_f{h}'] + control_cols].dropna()
            if len(reg_data) > len(control_cols) + 2:
                X = sm.add_constant(reg_data[control_cols])
                model = sm.OLS(reg_data[f'{self.y}_f{h}'], X)
                results = model.fit()
                df[f'r{self.y}_f{h}'] = np.nan
                df.loc[reg_data.index, f'r{self.y}_f{h}'] = results.resid

        # Orthogonalize treatment
        reg_data = df[[self.x] + control_cols].dropna()
        if len(reg_data) > len(control_cols) + 2:
            X = sm.add_constant(reg_data[control_cols])
            model = sm.OLS(reg_data[self.x], X)
            results = model.fit()
            df['r_x'] = np.nan
            df.loc[reg_data.index, 'r_x'] = results.resid

        # Orthogonalize instrument
        reg_data = df[[self.z] + control_cols].dropna()
        if len(reg_data) > len(control_cols) + 2:
            X = sm.add_constant(reg_data[control_cols])
            model = sm.OLS(reg_data[self.z], X)
            results = model.fit()
            df['r_z'] = np.nan
            df.loc[reg_data.index, 'r_z'] = results.resid

        return df

    def _gbf(self, h: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Gaussian Basis Function"""
        return a * np.exp(-((h - b) / c) ** 2)

    def _moment_conditions(
        self,
        params: np.ndarray,
        Y: np.ndarray,
        x: np.ndarray,
        Z: np.ndarray,
        horizons: np.ndarray
    ) -> np.ndarray:
        """
        Compute moment conditions for GBF-GMM

        g_h = Z * (y_h - a * exp(-(h-b)²/c²) * x)
        """
        a, b, c = params
        H = len(horizons)
        T = Y.shape[0]

        # Predicted IRF at each horizon
        beta_h = self._gbf(horizons, a, b, c)

        # Residuals
        E = Y - np.outer(x, beta_h)

        # Moment conditions: mean(Z * E)
        moments = np.zeros(H)
        for h in range(H):
            moments[h] = np.mean(Z[:, 0] * E[:, h])

        return moments

    def _objective(
        self,
        params: np.ndarray,
        Y: np.ndarray,
        x: np.ndarray,
        Z: np.ndarray,
        horizons: np.ndarray,
        W: np.ndarray
    ) -> float:
        """GMM objective function: g' W g"""
        g = self._moment_conditions(params, Y, x, Z, horizons)
        return g @ W @ g

    def estimate(
        self,
        initial_params: Tuple[float, float, float] = (1.0, 24.0, 10.0)
    ) -> Dict:
        """
        Estimate GBF parameters using GMM

        Parameters
        ----------
        initial_params : tuple, default=(1.0, 24.0, 10.0)
            Initial values for (a, b, c)

        Returns
        -------
        dict : Estimation results
        """
        # Orthogonalize variables
        df = self._orthogonalize()

        # Prepare data
        y_cols = [f'r{self.y}_f{h}' for h in range(self.horizon + 1)]
        all_cols = y_cols + ['r_x', 'r_z']
        gmm_data = df[all_cols].dropna()

        Y = gmm_data[y_cols].values
        x = gmm_data['r_x'].values
        Z = gmm_data[['r_z']].values
        horizons = np.arange(self.horizon + 1)
        T = len(Y)

        # Two-step GMM
        # Step 1: Identity weight matrix
        W1 = np.eye(self.horizon + 1)

        result1 = minimize(
            self._objective,
            initial_params,
            args=(Y, x, Z, horizons, W1),
            method='Nelder-Mead',
            options={'maxiter': 10000}
        )

        # Step 2: Optimal weight matrix
        params1 = result1.x
        beta_h = self._gbf(horizons, *params1)
        E = Y - np.outer(x, beta_h)

        # HAC covariance of moments
        G = np.zeros((T, self.horizon + 1))
        for h in range(self.horizon + 1):
            G[:, h] = Z[:, 0] * E[:, h]

        # Simple covariance (can extend to HAC)
        S = G.T @ G / T

        try:
            W2 = np.linalg.inv(S)
        except:
            W2 = np.eye(self.horizon + 1)

        result2 = minimize(
            self._objective,
            params1,
            args=(Y, x, Z, horizons, W2),
            method='Nelder-Mead',
            options={'maxiter': 10000}
        )

        self.a, self.b, self.c = result2.x

        # Standard errors via numerical gradient
        self._compute_standard_errors(Y, x, Z, horizons, W2, T)

        # Compute smoothed IRF and confidence bands
        horizons_fine = np.arange(self.horizon + 1)
        beta_smooth = self._gbf(horizons_fine, self.a, self.b, self.c)

        # Delta method for each horizon
        se_beta = self._compute_irf_se(horizons_fine)

        ci_lower = beta_smooth - 1.96 * se_beta
        ci_upper = beta_smooth + 1.96 * se_beta
        ci_1se_lower = beta_smooth - se_beta
        ci_1se_upper = beta_smooth + se_beta

        return {
            'params': {'a': self.a, 'b': self.b, 'c': self.c},
            'se_params': {'a': self.se_a, 'b': self.se_b, 'c': self.se_c},
            'smoothed': beta_smooth,
            'se_smoothed': se_beta,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_1se_lower': ci_1se_lower,
            'ci_1se_upper': ci_1se_upper,
            'horizons': horizons_fine
        }

    def _compute_standard_errors(
        self,
        Y: np.ndarray,
        x: np.ndarray,
        Z: np.ndarray,
        horizons: np.ndarray,
        W: np.ndarray,
        T: int
    ):
        """Compute standard errors using numerical gradient"""
        eps = 1e-6
        params = np.array([self.a, self.b, self.c])

        # Numerical Jacobian
        J = np.zeros((self.horizon + 1, 3))
        for i in range(3):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps

            g_plus = self._moment_conditions(params_plus, Y, x, Z, horizons)
            g_minus = self._moment_conditions(params_minus, Y, x, Z, horizons)

            J[:, i] = (g_plus - g_minus) / (2 * eps)

        # Covariance: (J' W J)^{-1} / T
        try:
            vcov = np.linalg.inv(J.T @ W @ J) / T
            self.se_a = np.sqrt(vcov[0, 0])
            self.se_b = np.sqrt(vcov[1, 1])
            self.se_c = np.sqrt(vcov[2, 2])
            self.vcov_params = vcov
        except:
            self.se_a = np.nan
            self.se_b = np.nan
            self.se_c = np.nan
            self.vcov_params = np.eye(3) * np.nan

    def _compute_irf_se(self, horizons: np.ndarray) -> np.ndarray:
        """
        Compute IRF standard errors using delta method

        ∂β/∂a = exp(-(h-b)²/c²)
        ∂β/∂b = 2a(h-b)/c² * exp(-(h-b)²/c²)
        ∂β/∂c = 2a(h-b)²/c³ * exp(-(h-b)²/c²)
        """
        se = np.zeros(len(horizons))

        for i, h in enumerate(horizons):
            exp_term = np.exp(-((h - self.b) / self.c) ** 2)

            grad = np.array([
                exp_term,
                2 * self.a * (h - self.b) / (self.c ** 2) * exp_term,
                2 * self.a * ((h - self.b) ** 2) / (self.c ** 3) * exp_term
            ])

            var = grad @ self.vcov_params @ grad
            se[i] = np.sqrt(max(0, var))

        return se

    def summary(self) -> str:
        """Return parameter summary"""
        s = "GBF-GMM Parameter Estimates\n"
        s += "=" * 40 + "\n"
        s += f"  a (amplitude):     {self.a:.4f} ({self.se_a:.4f})\n"
        s += f"  b (peak horizon):  {self.b:.4f} ({self.se_b:.4f})\n"
        s += f"  c (width):         {self.c:.4f} ({self.se_c:.4f})\n"
        return s
