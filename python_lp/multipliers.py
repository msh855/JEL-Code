"""
Fiscal Multiplier Estimation using Local Projections
Replicates Example 2 from Jordà & Taylor JEL
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Optional, List, Dict, Tuple
import warnings


class FiscalMultiplier:
    """
    Fiscal Multiplier Estimation using LP-IV

    Implements the methodology from Example 2 of the JEL paper:
    - Cumulative output responses to fiscal consolidation
    - LP-IV with government size as instrument
    - State-dependent analysis (boom/slump)
    - Joint hypothesis testing

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with country-year observations
    panel_id : str
        Name of panel identifier variable (e.g., country code)
    time_id : str
        Name of time variable
    """

    def __init__(
        self,
        data: pd.DataFrame,
        panel_id: str = 'iso',
        time_id: str = 'year'
    ):
        self.data = data.copy()
        self.panel_id = panel_id
        self.time_id = time_id
        self.results = None

    def prepare_data(
        self,
        y: str,
        treatment: str,
        instrument: str,
        controls: Optional[List[str]] = None,
        horizon: int = 4,
        hp_smooth: int = 400
    ) -> pd.DataFrame:
        """
        Prepare data for multiplier estimation

        Creates:
        - Long-difference outcomes: D_h y = y_{t+h} - y_{t-1}
        - Cumulative treatments: S_h dCAPB
        - HP-filtered output gap for state stratification

        Parameters
        ----------
        y : str
            Outcome variable (typically log real GDP)
        treatment : str
            Treatment variable (e.g., dCAPB)
        instrument : str
            Instrument variable (e.g., government size)
        controls : list, optional
            Additional control variables
        horizon : int, default=4
            Maximum horizon
        hp_smooth : int, default=400
            HP filter smoothing parameter

        Returns
        -------
        pd.DataFrame : Prepared data
        """
        df = self.data.copy()

        # Set panel structure
        df = df.sort_values([self.panel_id, self.time_id])

        # Create long-differences for outcome: y_{t+h} - y_{t-1}
        for h in range(horizon + 1):
            df[f'D{h}y'] = (
                df.groupby(self.panel_id)[y].shift(-h) -
                df.groupby(self.panel_id)[y].shift(1)
            )

        # Create cumulative sum of long-differences
        for h in range(horizon + 1):
            cols = [f'D{j}y' for j in range(h + 1)]
            df[f'S{h}y'] = df[cols].sum(axis=1)

        # Create cumulative treatment: sum from t to t+h
        for h in range(horizon + 1):
            if h == 0:
                df[f'{treatment}{h}'] = df[treatment]
            else:
                df[f'{treatment}{h}'] = (
                    df[f'{treatment}{h-1}'] +
                    df.groupby(self.panel_id)[treatment].shift(-h)
                )

        # Create cumulative sum of treatments
        for h in range(horizon + 1):
            cols = [f'{treatment}{j}' for j in range(h + 1)]
            df[f'S{treatment}{h}'] = df[cols].sum(axis=1)

        # HP filter for output gap (state stratification)
        df['y_hpcyc'] = np.nan
        df['y_hptrend'] = np.nan

        for panel in df[self.panel_id].unique():
            mask = df[self.panel_id] == panel
            y_panel = df.loc[mask, y].dropna()

            if len(y_panel) > 10:
                cycle, trend = self._hp_filter(y_panel.values, hp_smooth)
                df.loc[y_panel.index, 'y_hpcyc'] = cycle
                df.loc[y_panel.index, 'y_hptrend'] = trend

        # Create state indicators
        df['boom'] = (df.groupby(self.panel_id)['y_hpcyc'].shift(1) > 0).astype(int)
        df['slump'] = (df.groupby(self.panel_id)['y_hpcyc'].shift(1) <= 0).astype(int)

        # Create lagged controls
        control_vars = []
        for lag in range(1, 3):
            # Lagged changes in y
            df[f'Ldy{lag}'] = df.groupby(self.panel_id)[y].diff().shift(lag)
            control_vars.append(f'Ldy{lag}')

            # Lagged treatment
            df[f'L{treatment}{lag}'] = df.groupby(self.panel_id)[treatment].shift(lag)
            control_vars.append(f'L{treatment}{lag}')

        # Lagged output gap
        df['L_y_hpcyc'] = df.groupby(self.panel_id)['y_hpcyc'].shift(1)
        control_vars.append('L_y_hpcyc')

        # Store instrument
        df['instrument'] = df[instrument]

        self.prepared_data = df
        self.horizon = horizon
        self.treatment = treatment
        self.control_vars = control_vars

        return df

    def _hp_filter(
        self,
        y: np.ndarray,
        lamb: int = 400
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hodrick-Prescott filter

        Parameters
        ----------
        y : np.ndarray
            Time series
        lamb : int
            Smoothing parameter (400 for annual data)

        Returns
        -------
        tuple : (cycle, trend)
        """
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        n = len(y)

        # Construct the penalty matrix
        e = np.ones(n)
        D = sparse.diags([e, -2*e, e], [0, 1, 2], shape=(n-2, n))

        # Solve for trend
        I = sparse.eye(n)
        trend = spsolve(I + lamb * D.T @ D, y)
        cycle = y - trend

        return cycle, trend

    def estimate(
        self,
        state: str = 'full',
        alpha: float = 0.05
    ) -> Dict:
        """
        Estimate fiscal multipliers using LP-IV

        Parameters
        ----------
        state : str, default='full'
            Sample restriction: 'full', 'boom', or 'slump'
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        dict : Estimation results
        """
        from linearmodels.iv import IV2SLS

        df = self.prepared_data.copy()

        # Apply state filter
        if state == 'boom':
            df = df[df['boom'] == 1]
        elif state == 'slump':
            df = df[df['slump'] == 1]

        # Initialize results storage
        betas = np.zeros(self.horizon + 1)
        ses = np.zeros(self.horizon + 1)

        # Estimate for each horizon
        for h in range(self.horizon + 1):
            y_var = f'S{h}y'
            t_var = f'S{self.treatment}{h}'

            # Prepare regression data
            reg_vars = [y_var, t_var, 'instrument'] + self.control_vars
            reg_data = df[reg_vars + [self.panel_id]].dropna()

            if len(reg_data) < 20:
                warnings.warn(f"Insufficient observations at horizon {h}")
                continue

            # IV regression with panel FE
            try:
                # Create fixed effects
                dummies = pd.get_dummies(reg_data[self.panel_id], prefix='fe', drop_first=True)

                dependent = reg_data[y_var]
                exog_vars = reg_data[self.control_vars].join(dummies)
                exog = sm.add_constant(exog_vars)
                endog = reg_data[[t_var]]
                instruments = reg_data[['instrument']]

                model = IV2SLS(dependent, exog, endog, instruments)
                results = model.fit(cov_type='clustered', clusters=reg_data[self.panel_id])

                betas[h] = results.params[t_var]
                ses[h] = results.std_errors[t_var]

            except Exception as e:
                warnings.warn(f"Estimation failed at horizon {h}: {str(e)}")
                betas[h] = np.nan
                ses[h] = np.nan

        # Confidence intervals
        z_val = stats.norm.ppf(1 - alpha / 2)
        ci_lower = betas - z_val * ses
        ci_upper = betas + z_val * ses

        self.results = {
            'state': state,
            'betas': betas,
            'ses': ses,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'horizons': np.arange(self.horizon + 1)
        }

        return self.results

    def estimate_stacked(
        self,
        state: str = 'full',
        alpha: float = 0.05
    ) -> Dict:
        """
        Estimate using stacked regression (joint estimation across horizons)

        This enables joint hypothesis testing across all horizons.

        Parameters
        ----------
        state : str, default='full'
            Sample restriction
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        dict : Results including joint test
        """
        from linearmodels.iv import IV2SLS

        df = self.prepared_data.copy()

        # Apply state filter
        if state == 'boom':
            df = df[df['boom'] == 1]
        elif state == 'slump':
            df = df[df['slump'] == 1]

        # Create stacked dataset
        stacked_dfs = []
        for h in range(self.horizon + 1):
            temp = df[[self.panel_id, self.time_id,
                      f'S{h}y', f'S{self.treatment}{h}', 'instrument'] + self.control_vars].copy()
            temp.columns = [self.panel_id, self.time_id,
                          'Y', 'T', 'Z'] + self.control_vars
            temp['H'] = h
            temp = temp.dropna()
            stacked_dfs.append(temp)

        stacked = pd.concat(stacked_dfs, ignore_index=True)

        # Create horizon-specific treatment and instrument interactions
        for h in range(self.horizon + 1):
            stacked[f'T_h{h}'] = stacked['T'] * (stacked['H'] == h)
            stacked[f'Z_h{h}'] = stacked['Z'] * (stacked['H'] == h)

            # Zero out controls for other horizons
            for ctrl in self.control_vars:
                stacked[f'{ctrl}_h{h}'] = stacked[ctrl] * (stacked['H'] == h)

        # Create panel-horizon fixed effects
        stacked['FE'] = stacked[self.panel_id].astype(str) + '_' + stacked['H'].astype(str)

        # IV regression
        T_vars = [f'T_h{h}' for h in range(self.horizon + 1)]
        Z_vars = [f'Z_h{h}' for h in range(self.horizon + 1)]
        ctrl_vars = [f'{c}_h{h}' for h in range(self.horizon + 1) for c in self.control_vars]

        # Create fixed effects dummies
        fe_dummies = pd.get_dummies(stacked['FE'], prefix='fe', drop_first=True)

        dependent = stacked['Y']
        exog = sm.add_constant(stacked[ctrl_vars].join(fe_dummies))
        endog = stacked[T_vars]
        instruments = stacked[Z_vars]

        try:
            model = IV2SLS(dependent, exog, endog, instruments)
            results = model.fit(cov_type='clustered', clusters=stacked['FE'])

            # Extract coefficients
            betas = np.array([results.params[f'T_h{h}'] for h in range(self.horizon + 1)])
            ses = np.array([results.std_errors[f'T_h{h}'] for h in range(self.horizon + 1)])

            # Joint test: all coefficients equal zero
            vcov = results.cov.loc[T_vars, T_vars].values
            chi2_stat = betas @ np.linalg.inv(vcov) @ betas
            df_joint = len(betas)
            p_joint = 1 - stats.chi2.cdf(chi2_stat, df_joint)

            # Average multiplier
            avg_beta = np.mean(betas)
            avg_se = np.sqrt(np.sum(vcov)) / (self.horizon + 1)

        except Exception as e:
            warnings.warn(f"Stacked estimation failed: {str(e)}")
            return None

        z_val = stats.norm.ppf(1 - alpha / 2)

        self.stacked_results = {
            'state': state,
            'betas': betas,
            'ses': ses,
            'ci_lower': betas - z_val * ses,
            'ci_upper': betas + z_val * ses,
            'horizons': np.arange(self.horizon + 1),
            'average_multiplier': avg_beta,
            'average_se': avg_se,
            'joint_test': {
                'chi2': chi2_stat,
                'df': df_joint,
                'p_value': p_joint
            }
        }

        return self.stacked_results

    def plot(
        self,
        results: Optional[Dict] = None,
        title: str = None,
        ax=None
    ):
        """
        Plot multiplier estimates with confidence bands

        Parameters
        ----------
        results : dict, optional
            Results to plot. If None, uses last estimation
        title : str, optional
            Plot title
        ax : matplotlib.axes.Axes, optional
            Axes to plot on

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if results is None:
            results = self.results if self.results else self.stacked_results

        if results is None:
            raise ValueError("No results to plot. Run estimate() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        h = results['horizons']

        # Plot confidence bands
        ax.fill_between(h, results['ci_lower'], results['ci_upper'],
                       alpha=0.2, color='blue', label='95% CI')

        # Plot point estimates
        ax.plot(h, results['betas'], 'b-', linewidth=2, label='Multiplier')

        # Zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add average multiplier if available
        if 'average_multiplier' in results:
            avg = results['average_multiplier']
            ax.axhline(y=avg, color='red', linestyle='--',
                      linewidth=1, alpha=0.7, label=f'Average: {avg:.2f}')

        ax.set_xlabel('Horizon (years)', fontsize=12)
        ax.set_ylabel('Multiplier, m(h)', fontsize=12)

        if title is None:
            title = f"Fiscal Multiplier ({results['state']} sample)"
        ax.set_title(title, fontsize=14)

        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add joint test annotation if available
        if 'joint_test' in results:
            jt = results['joint_test']
            ax.text(0.95, 0.95,
                   f"Joint test: χ²({jt['df']})={jt['chi2']:.1f}\n(p={jt['p_value']:.3f})",
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        return ax


def compute_cumulative_multiplier(
    betas_y: np.ndarray,
    betas_x: np.ndarray
) -> np.ndarray:
    """
    Compute cumulative multiplier as ratio of cumulative responses

    m(h) = Σ_{j=0}^{h} β_y(j) / Σ_{j=0}^{h} β_x(j)

    Parameters
    ----------
    betas_y : np.ndarray
        Response coefficients for output
    betas_x : np.ndarray
        Response coefficients for fiscal variable

    Returns
    -------
    np.ndarray : Cumulative multipliers
    """
    cum_y = np.cumsum(betas_y)
    cum_x = np.cumsum(betas_x)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        multiplier = np.where(cum_x != 0, cum_y / cum_x, np.nan)

    return multiplier
