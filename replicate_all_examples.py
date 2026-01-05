"""
Comprehensive Replication: STATA to Python
Jordà & Taylor JEL Local Projections Paper

This script replicates Examples 2-9 from the paper and creates
side-by-side comparisons between STATA and Python outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# For PDF to image conversion
from pdf2image import convert_from_path
from PIL import Image
import os

# Paths
STATA_PATH = '/home/user/JEL-Code/LP_JEL_Replication'
OUTPUT_PATH = '/home/user/JEL-Code/output/side_by_side'

def load_stata_figure(example_folder, figure_name):
    """Load STATA PDF figure and convert to image"""
    pdf_path = os.path.join(STATA_PATH, example_folder, figure_name)
    if os.path.exists(pdf_path):
        images = convert_from_path(pdf_path, dpi=150)
        return images[0] if images else None
    return None


# =============================================================================
# EXAMPLE 3: GBF Smoothing (Simulation)
# =============================================================================
def example3_gbf():
    """
    Replicate Example 3: GBF Illustration

    STATA: GBF.do
    Output: Figure3.pdf (GBF_plot.pdf)
    Data: None (pure simulation)
    """
    print("=" * 70)
    print("EXAMPLE 3: GBF Smoothing Illustration")
    print("=" * 70)

    # Parameters from STATA code (GBF.do lines 12-14)
    a = 1
    b = 8
    c = 8

    # Generate GBF response
    horizon = np.arange(31)
    response = a * np.exp(-((horizon - b) / c) ** 2)

    # Create side-by-side figure
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 1])

    # Left: STATA figure
    ax1 = fig.add_subplot(gs[0])
    stata_img = load_stata_figure('Example3_Smoothing', 'GBF_plot.pdf')
    if stata_img:
        ax1.imshow(stata_img)
        ax1.set_title('STATA Output (GBF_plot.pdf)', fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'STATA PDF not found', ha='center', va='center', fontsize=14)
        ax1.set_title('STATA Output', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Right: Python figure
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(horizon, response, 'b-', linewidth=2.5)
    ax2.axhline(y=a, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=a/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=b, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax2.text(20, 1.1, f'GBF parameters:\na = {a}\nb = {b}\nc = {c}',
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax2.set_xlabel('Horizon, h', fontsize=11)
    ax2.set_ylabel('Response, φ(h; Θ)', fontsize=11)
    ax2.set_ylim(0, 1.3)
    ax2.set_xlim(0, 30)
    ax2.set_title('Python Output', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Example 3: Gaussian Basis Function (Figure 3)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/example3_side_by_side.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Example 3 completed: output/side_by_side/example3_side_by_side.png")
    return True


# =============================================================================
# EXAMPLE 4: Newey-West vs Lag Augmentation (Simulation)
# =============================================================================
def example4_inference():
    """
    Replicate Example 4: NW vs LA Inference

    STATA: nw_v_la.do
    Output: Figure4.pdf (fig_nw_v_la.pdf)
    Data: None (simulation with specific parameters)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Newey-West vs Lag Augmentation")
    print("=" * 70)

    # Parameters from STATA code (nw_v_la.do)
    nobs = 300
    burn = 1000
    tobs = nobs + burn

    # VAR parameters
    ayy, ayx = 0.85, 0.2
    axy, axx = 0.2, 0.85
    byx = 1
    p = 0.05
    horizon = 13

    # Simulate VAR(1) data
    np.random.seed(12345)
    uy = np.random.normal(size=tobs)
    ux = np.random.normal(size=tobs)

    y = np.zeros(tobs)
    x = np.zeros(tobs)

    for t in range(1, tobs):
        y[t] = ayy * y[t-1] + ayx * x[t-1] + byx * ux[t] + uy[t]
        x[t] = axy * y[t-1] + axx * x[t-1] + ux[t]

    # Remove burn-in
    y = y[burn:]
    x = x[burn:]

    df = pd.DataFrame({'y': y, 'x': x})

    # Estimate LP: response of forward x to shock in y
    b = np.zeros(horizon)
    snw = np.zeros(horizon)
    sla = np.zeros(horizon)

    for h in range(horizon):
        # Forward x
        df[f'x_f{h}'] = df['x'].shift(-h)

        # Newey-West: newey f{h}.x y l.y l.x, lag(6)
        reg_data = df[['y', f'x_f{h}']].copy()
        reg_data['y_L1'] = df['y'].shift(1)
        reg_data['x_L1'] = df['x'].shift(1)
        reg_data = reg_data.dropna()

        X_nw = sm.add_constant(reg_data[['y', 'y_L1', 'x_L1']])
        model_nw = OLS(reg_data[f'x_f{h}'], X_nw)
        results_nw = model_nw.fit(cov_type='HAC', cov_kwds={'maxlags': 6})

        b[h] = results_nw.params['y']
        snw[h] = results_nw.bse['y']

        # Lag-augmented: reg f{h}.x y l(1/2).y l(1/2).x, vce(hc3)
        reg_data2 = df[['y', f'x_f{h}']].copy()
        for lag in range(1, 3):
            reg_data2[f'y_L{lag}'] = df['y'].shift(lag)
            reg_data2[f'x_L{lag}'] = df['x'].shift(lag)
        reg_data2 = reg_data2.dropna()

        lag_cols = ['y'] + [f'y_L{l}' for l in range(1, 3)] + [f'x_L{l}' for l in range(1, 3)]
        X_la = sm.add_constant(reg_data2[lag_cols])
        model_la = OLS(reg_data2[f'x_f{h}'], X_la)
        results_la = model_la.fit(cov_type='HC3')

        sla[h] = results_la.bse['y']

    # Confidence bands
    z_val = stats.norm.ppf(1 - p/2)
    unw = b + z_val * snw
    dnw = b - z_val * snw
    ula = b + z_val * sla
    dla = b - z_val * sla

    # Create side-by-side figure
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 1])

    # Left: STATA figure
    ax1 = fig.add_subplot(gs[0])
    stata_img = load_stata_figure('Example4_LagAugmentation', 'fig_nw_v_la.pdf')
    if stata_img:
        ax1.imshow(stata_img)
        ax1.set_title('STATA Output (fig_nw_v_la.pdf)', fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'STATA PDF not found', ha='center', va='center', fontsize=14)
        ax1.set_title('STATA Output', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Right: Python figure
    ax2 = fig.add_subplot(gs[1])
    h_vals = np.arange(horizon)

    ax2.fill_between(h_vals, dnw, unw, alpha=0.3, color='blue', label='Newey-West CI')
    ax2.plot(h_vals, ula, 'r--', linewidth=1.5, label='Lag-augmented CI')
    ax2.plot(h_vals, dla, 'r--', linewidth=1.5)
    ax2.plot(h_vals, b, 'b-', linewidth=2.5, label='Response βₕ')
    ax2.axhline(y=0, color='black', linewidth=0.5)

    ax2.set_xlabel('Horizon, h', fontsize=11)
    ax2.set_ylabel('Response, βₕ', fontsize=11)
    ax2.set_xlim(0, horizon - 1)
    ax2.set_xticks(np.arange(0, horizon, 2))
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_title('Python Output', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Example 4: Newey-West vs Lag-Augmented Inference (Figure 4)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/example4_side_by_side.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Example 4 completed: output/side_by_side/example4_side_by_side.png")
    return True


# =============================================================================
# EXAMPLE 9: State-Dependence KOB Simulation
# =============================================================================
def example9_state_dependence():
    """
    Replicate Example 9: Kitagawa-Oaxaca-Blinder Decomposition

    STATA: kob_sim.do
    Output: Figure10a.pdf (fig_kob_1nt.pdf), Figure10b.pdf (fig_kob_2nt.pdf)
    Data: None (simulation)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 9: State-Dependence (KOB Decomposition)")
    print("=" * 70)

    # Parameters from STATA code (kob_sim.do)
    nobs = 500
    burn = 500
    tobs = nobs + burn

    rhox = 0.75
    rhos = 0.75
    rhoy = 0.75
    mu0 = 0
    beta = 0.5
    theta = 0.5
    gamma0 = 0.75

    horizon = 12

    # Simulate data
    np.random.seed(12345)
    vy = np.random.normal(size=tobs)
    vx = np.random.normal(size=tobs)
    vs = np.random.normal(size=tobs)

    y = np.zeros(tobs)
    x = np.zeros(tobs)
    s = np.zeros(tobs)

    for t in range(1, tobs):
        x[t] = rhox * x[t-1] + vx[t]
        s[t] = rhos * s[t-1] + vs[t]

    # State indicator
    I = (np.abs(s) > 1).astype(float)

    for t in range(1, tobs):
        y[t] = mu0 + rhoy * y[t-1] + gamma0 * x[t] + I[t] * s[t] * (beta + theta * x[t]) + vy[t]

    # Remove burn-in
    y = y[burn:]
    x = x[burn:]
    s = s[burn:]
    I = I[burn:]

    df = pd.DataFrame({'y': y, 'x': x, 's': s, 'I': I})
    df['Is'] = df['I'] * df['s']
    df['Isx'] = df['I'] * df['s'] * df['x']

    # Estimate LP
    b_est = np.zeros(horizon + 1)
    gam_est = np.zeros(horizon + 1)
    thet_est = np.zeros(horizon + 1)
    sb_est = np.zeros(horizon + 1)

    for h in range(horizon + 1):
        df[f'y_f{h}'] = df['y'].shift(-h)

        reg_data = df[['x', f'y_f{h}', 'Is', 'Isx']].copy()
        reg_data['y_L1'] = df['y'].shift(1)
        reg_data = reg_data.dropna()

        X = sm.add_constant(reg_data[['x', 'y_L1', 'Is', 'Isx']])
        model = OLS(reg_data[f'y_f{h}'], X)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': horizon})

        b_est[h] = results.params['Is']
        gam_est[h] = results.params['x']
        thet_est[h] = results.params['Isx']
        sb_est[h] = results.bse['Is']

    # Generate time-varying responses
    betx = {}
    for h in [1, 3, 5, 7]:
        betx[h] = b_est[h] + thet_est[h] * x[:100]

    # Create side-by-side figure
    fig = plt.figure(figsize=(14, 10))

    # Top row: Panel (a) - Time series response variation
    ax1 = fig.add_subplot(2, 2, 1)
    stata_img1 = load_stata_figure('Example9_StateDependence', 'fig_kob_1nt.pdf')
    if stata_img1:
        ax1.imshow(stata_img1)
        ax1.set_title('STATA: fig_kob_1nt.pdf', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    t_vals = np.arange(100)
    ax2.plot(t_vals, betx[1], 'b-', alpha=0.7, linewidth=1.5, label='R(1)')
    ax2.plot(t_vals, betx[3], 'g-', alpha=0.7, linewidth=1.5, label='R(3)')
    ax2.plot(t_vals, betx[5], 'purple', linestyle='--', alpha=0.7, linewidth=1.5, label='R(5)')
    ax2.plot(t_vals, betx[7], 'orange', linestyle='--', alpha=0.7, linewidth=1.5, label='R(7)')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Observation, t', fontsize=10)
    ax2.set_ylabel('Response, state-varying, R(h)', fontsize=10)
    ax2.legend(loc='best', fontsize=9)
    ax2.set_title('Python: Time series response variation', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Bottom row: Panel (b) - Variation due to secondary treatment
    ax3 = fig.add_subplot(2, 2, 3)
    stata_img2 = load_stata_figure('Example9_StateDependence', 'fig_kob_2nt.pdf')
    if stata_img2:
        ax3.imshow(stata_img2)
        ax3.set_title('STATA: fig_kob_2nt.pdf', fontsize=11, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 2, 4)
    h_vals = np.arange(horizon + 1)

    bu1 = b_est + thet_est
    bu2 = b_est + 2 * thet_est
    bd1 = b_est - thet_est
    bd2 = b_est - 2 * thet_est
    sbu = b_est + 1.96 * sb_est
    sbd = b_est - 1.96 * sb_est

    ax4.fill_between(h_vals, sbd, sbu, alpha=0.2, color='blue')
    ax4.plot(h_vals, b_est, 'gray', linewidth=2, label='Base')
    ax4.plot(h_vals, bu1, 'b--', linewidth=1.5, alpha=0.75, label='2nd Treat = 1')
    ax4.plot(h_vals, bd1, 'purple', linestyle='--', linewidth=1.5, alpha=0.75, label='2nd Treat = -1')
    ax4.plot(h_vals, bu2, 'b--', linewidth=1, alpha=0.5, label='2nd Treat = 2')
    ax4.plot(h_vals, bd2, 'purple', linestyle='--', linewidth=1, alpha=0.5, label='2nd Treat = -2')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_xlabel('Horizon, h', fontsize=10)
    ax4.set_ylabel('Response, state-varying, R(h)', fontsize=10)
    ax4.legend(loc='best', fontsize=8)
    ax4.set_title('Python: Variation due to secondary treatment', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Example 9: State-Dependence / KOB Decomposition (Figure 10)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/example9_side_by_side.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Example 9 completed: output/side_by_side/example9_side_by_side.png")
    return True


# =============================================================================
# EXAMPLE 6: Joint GMM Inference with GBF
# =============================================================================
def example6_gmm_gbf():
    """
    Replicate Example 6: Joint LP-IV and GBF Estimation

    STATA: GMM_LPIV_GBF_estimate_part1.do, GMM_LPIV_GBF_output_part2.do
    Output: Figure6a.pdf (fig_urate_LPIV.pdf), Figure6b.pdf (fig_urate_gbf.pdf)
    Data: data_fred.dta
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Joint GMM Inference with GBF")
    print("=" * 70)

    # Load data
    data = pd.read_stata(f'{STATA_PATH}/Example6_JointInference/data_fred.dta')

    # Sample restriction (1985m1 to 2000m1)
    data['date'] = pd.to_datetime(data['date'])
    data = data[(data['date'] >= '1985-01-01') & (data['date'] <= '2000-01-01')].copy()
    data = data.sort_values('date').reset_index(drop=True)

    print(f"  Sample: {data['date'].min()} to {data['date'].max()}, N={len(data)}")

    # Variables
    y = 'urate'
    horizon = 48
    lags = 6

    # Create forward variables (long-differences)
    for h in range(horizon + 1):
        data[f'{y}_f{h}'] = data[y].shift(-h) - data[y].shift(1)

    # Create lagged controls
    control_cols = []
    for var in ['urate', 'infl', 'ffr']:
        for lag in range(1, lags + 1):
            col = f'{var}_L{lag}'
            data[col] = data[var].shift(lag)
            control_cols.append(col)

    # Orthogonalize LHS, treatment, and instrument
    def orthogonalize(y_var, controls, df):
        reg_data = df[[y_var] + controls].dropna()
        X = sm.add_constant(reg_data[controls])
        model = OLS(reg_data[y_var], X)
        results = model.fit()
        resid = pd.Series(index=df.index, dtype=float)
        resid.loc[reg_data.index] = results.resid
        return resid

    # Orthogonalize treatment (ffr)
    data['rffr'] = orthogonalize('ffr', control_cols, data)

    # Orthogonalize instrument (RRCGShock)
    data['rz'] = orthogonalize('RRCGShock', control_cols, data)

    # Orthogonalize forward y variables
    for h in range(horizon + 1):
        data[f'r{y}_f{h}'] = orthogonalize(f'{y}_f{h}', control_cols, data)

    # Simple LP-IV estimation (equation by equation)
    betas = np.zeros(horizon + 1)
    ses = np.zeros(horizon + 1)

    for h in range(horizon + 1):
        # IV estimation using 2SLS
        y_col = f'r{y}_f{h}'
        reg_data = data[[y_col, 'rffr', 'rz']].dropna()

        if len(reg_data) < 50:
            continue

        # First stage: rffr on rz
        X_fs = sm.add_constant(reg_data['rz'])
        fs_model = OLS(reg_data['rffr'], X_fs)
        fs_results = fs_model.fit()
        rffr_hat = fs_results.fittedvalues

        # Second stage: y on rffr_hat
        X_ss = sm.add_constant(rffr_hat)
        ss_model = OLS(reg_data[y_col], X_ss)
        ss_results = ss_model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        betas[h] = ss_results.params.iloc[1]
        ses[h] = ss_results.bse.iloc[1]

    # GBF fitting
    def gbf_func(h, a, b, c):
        return a * np.exp(-((h - b) / c) ** 2)

    def gbf_objective(params, h, y):
        a, b, c = params
        pred = gbf_func(h, a, b, c)
        return np.sum((y - pred) ** 2)

    h_arr = np.arange(horizon + 1)
    valid_mask = ~np.isnan(betas)

    result = minimize(gbf_objective, [1.5, 24, 10],
                     args=(h_arr[valid_mask], betas[valid_mask]),
                     method='Nelder-Mead')
    a_est, b_est, c_est = result.x

    gbf_smooth = gbf_func(h_arr, a_est, b_est, c_est)

    # Joint test (simplified)
    valid_betas = betas[valid_mask]
    valid_ses = ses[valid_mask]
    chi2_stat = np.sum((valid_betas / valid_ses) ** 2)
    df_test = len(valid_betas)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df_test)

    # Confidence bands
    ci_upper = betas + 1.96 * ses
    ci_lower = betas - 1.96 * ses
    ci_1se_upper = betas + ses
    ci_1se_lower = betas - ses

    # Create side-by-side figure
    fig = plt.figure(figsize=(14, 10))

    # Top row: LPIV figure
    ax1 = fig.add_subplot(2, 2, 1)
    stata_img1 = load_stata_figure('Example6_JointInference', 'fig_urate_LPIV.pdf')
    if stata_img1:
        ax1.imshow(stata_img1)
        ax1.set_title('STATA: fig_urate_LPIV.pdf', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.fill_between(h_arr, ci_1se_lower, ci_1se_upper, alpha=0.1, color='blue')
    ax2.fill_between(h_arr, ci_lower, ci_upper, alpha=0.2, color='blue')
    ax2.plot(h_arr, betas, 'b--', linewidth=2, label='LPIV estimate')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Horizon, months, h', fontsize=10)
    ax2.set_ylabel('Response, percentage points', fontsize=10)
    ax2.set_xlim(0, horizon)
    ax2.set_xticks(np.arange(0, horizon + 1, 12))
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_title('Python: LPIV Estimates', fontsize=11, fontweight='bold')
    ax2.text(0.02, 0.98, f'Joint test: χ²({df_test})={chi2_stat:.1f}\n(p={p_value:.3f})',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.grid(True, alpha=0.3)

    # Bottom row: GBF figure
    ax3 = fig.add_subplot(2, 2, 3)
    stata_img2 = load_stata_figure('Example6_JointInference', 'fig_urate_gbf.pdf')
    if stata_img2:
        ax3.imshow(stata_img2)
        ax3.set_title('STATA: fig_urate_gbf.pdf', fontsize=11, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.fill_between(h_arr, ci_1se_lower, ci_1se_upper, alpha=0.1, color='purple')
    ax4.fill_between(h_arr, ci_lower, ci_upper, alpha=0.2, color='purple')
    ax4.plot(h_arr, betas, 'b--', linewidth=2, alpha=0.7, label='LPIV')
    ax4.plot(h_arr, gbf_smooth, 'purple', linewidth=2.5, label='GBF smoothed')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_xlabel('Horizon, months, h', fontsize=10)
    ax4.set_ylabel('Response, percentage points', fontsize=10)
    ax4.set_xlim(0, horizon)
    ax4.set_xticks(np.arange(0, horizon + 1, 12))
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_title('Python: GBF Approximation', fontsize=11, fontweight='bold')
    ax4.text(0.65, 0.98, f'GBF parameters:\na={a_est:.3f}\nb={b_est:.3f}\nc={c_est:.3f}',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Example 6: Joint GMM Inference with GBF (Figure 6)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/example6_side_by_side.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  GBF parameters: a={a_est:.3f}, b={b_est:.3f}, c={c_est:.3f}")
    print("✓ Example 6 completed: output/side_by_side/example6_side_by_side.png")
    return True


# =============================================================================
# EXAMPLE 5: Significance Bands (Romer-Romer)
# =============================================================================
def example5_significance_bands():
    """
    Replicate Example 5: Significance Bands

    STATA: sbands_RR.do
    Output: Figure5a.pdf (gc_lcpi_17.pdf), Figure5b.pdf (gs_lcpi_17.pdf)
    Data: aggregatedata_final.dta
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Significance Bands")
    print("=" * 70)

    # Load data
    data = pd.read_stata(f'{STATA_PATH}/Example5_SignificanceBands/aggregatedata_final.dta')

    # Scale variables
    data['lcpi'] = 100 * data['lcpi']
    data['lrgdp'] = 100 * data['lrgdp']

    # Sample restriction (1985q1 to 2007q4)
    data['qdate'] = pd.to_datetime(data['qdate'])
    data = data[(data['qdate'] >= '1985-01-01') & (data['qdate'] <= '2007-12-31')].copy()
    data = data.sort_values('qdate').reset_index(drop=True)

    print(f"  Sample size: {len(data)}")

    # Parameters
    horizon = 17
    lags = 4
    nwlag = horizon
    p = 0.05

    y = 'lcpi'
    x = 'stir'
    z = 'rr_shock'

    # Create forward variables (long-differences)
    for h in range(horizon + 1):
        data[f'{y}_f{h}'] = data[y].shift(-h) - data[y].shift(1)

    # Create lagged controls
    for var in ['dlrgdp', 'dlcpi', 'dstir']:
        for lag in range(1, lags + 1):
            data[f'{var}_L{lag}'] = data[var].shift(lag)

    control_cols = [f'{v}_L{l}' for v in ['dlrgdp', 'dlcpi', 'dstir'] for l in range(1, lags + 1)]

    # LP estimation with Newey-West
    betas = np.zeros(horizon + 1)
    ses = np.zeros(horizon + 1)

    for h in range(horizon + 1):
        y_var = f'{y}_f{h}'
        reg_data = data[[y_var, z] + control_cols].dropna()

        if len(reg_data) < len(control_cols) + 5:
            continue

        X = sm.add_constant(reg_data[[z] + control_cols])
        model = OLS(reg_data[y_var], X)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': nwlag})

        betas[h] = results.params[z]
        ses[h] = results.bse[z]

    # Confidence bands
    u1 = betas + ses
    d1 = betas - ses
    u2 = betas + 1.96 * ses
    d2 = betas - 1.96 * ses

    # Significance bands (symmetric around zero)
    sig_upper = 1.96 * ses
    sig_lower = -sig_upper

    # Bonferroni bands
    z_bonf = stats.norm.ppf(1 - p / (2 * (horizon + 1)))
    bonf_upper = z_bonf * ses
    bonf_lower = -bonf_upper

    # Joint test
    t_stats = betas / ses
    chi2_stat = np.nansum(t_stats ** 2)
    p_joint = 1 - stats.chi2.cdf(chi2_stat, horizon + 1)

    # Create side-by-side figure
    fig = plt.figure(figsize=(14, 10))

    h_arr = np.arange(horizon + 1)

    # Top row: Confidence bands only
    ax1 = fig.add_subplot(2, 2, 1)
    stata_img1 = load_stata_figure('Example5_SignificanceBands', 'gc_lcpi_17.pdf')
    if stata_img1:
        ax1.imshow(stata_img1)
        ax1.set_title('STATA: gc_lcpi_17.pdf', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.fill_between(h_arr, d2, u2, alpha=0.15, color='green')
    ax2.fill_between(h_arr, d1, u1, alpha=0.25, color='green')
    ax2.plot(h_arr, betas, 'g-', linewidth=2.5)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Horizon, quarters, h', fontsize=10)
    ax2.set_ylabel('Response, log x100', fontsize=10)
    ax2.set_xlim(0, horizon)
    ax2.set_xticks(np.arange(0, horizon + 1, 5))
    ax2.set_title('Python: Confidence Bands', fontsize=11, fontweight='bold')
    ax2.text(0.02, 0.02, f'p-value joint test: {p_joint:.3f}',
             transform=ax2.transAxes, fontsize=9, verticalalignment='bottom')
    ax2.grid(True, alpha=0.3)

    # Bottom row: With significance bands
    ax3 = fig.add_subplot(2, 2, 3)
    stata_img2 = load_stata_figure('Example5_SignificanceBands', 'gs_lcpi_17.pdf')
    if stata_img2:
        ax3.imshow(stata_img2)
        ax3.set_title('STATA: gs_lcpi_17.pdf', fontsize=11, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.fill_between(h_arr, d2, u2, alpha=0.15, color='green')
    ax4.fill_between(h_arr, d1, u1, alpha=0.25, color='green')
    ax4.plot(h_arr, betas, 'g-', linewidth=2.5, label='Response')
    ax4.plot(h_arr, sig_upper, 'b--', linewidth=1.5, label='Significance band')
    ax4.plot(h_arr, sig_lower, 'b--', linewidth=1.5)
    ax4.plot(h_arr, bonf_upper, 'r:', linewidth=1.5, label='Bonferroni band')
    ax4.plot(h_arr, bonf_lower, 'r:', linewidth=1.5)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_xlabel('Horizon, quarters, h', fontsize=10)
    ax4.set_ylabel('Response, log CPI x100', fontsize=10)
    ax4.set_xlim(0, horizon)
    ax4.set_xticks(np.arange(0, horizon + 1, 5))
    ax4.legend(loc='lower left', fontsize=8)
    ax4.set_title('Python: With Significance Bands', fontsize=11, fontweight='bold')
    ax4.text(0.98, 0.02, f'p-value joint test: {p_joint:.3f}',
             transform=ax4.transAxes, fontsize=9, verticalalignment='bottom', ha='right')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Example 5: Significance Bands (Figure 5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/example5_side_by_side.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Joint test p-value: {p_joint:.4f}")
    print("✓ Example 5 completed: output/side_by_side/example5_side_by_side.png")
    return True


# =============================================================================
# EXAMPLE 8: Counterfactuals
# =============================================================================
def example8_counterfactuals():
    """
    Replicate Example 8: Counterfactual Analysis

    STATA: GMM_GBF_counterfactual_clean.do
    Output: Figure8a.pdf (counterfig_a.pdf), Figure8b.pdf (counterfig_b.pdf)
    Data: data_fred.dta (same as Example 6)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Counterfactual Analysis")
    print("=" * 70)

    # Load data
    data = pd.read_stata(f'{STATA_PATH}/Example8_Counterfactuals/data_fred.dta')

    # Sample restriction
    data['date'] = pd.to_datetime(data['date'])
    data = data[(data['date'] >= '1985-01-01') & (data['date'] <= '2000-12-01')].copy()
    data = data.sort_values('date').reset_index(drop=True)

    print(f"  Sample: {data['date'].min()} to {data['date'].max()}, N={len(data)}")

    horizon = 48
    lags = 6

    # Create forward variables for both urate and ffr
    for h in range(horizon + 1):
        data[f'urate_f{h}'] = data['urate'].shift(-h) - data['urate'].shift(1)
        data[f'ffr_f{h}'] = data['ffr'].shift(-h) - data['ffr'].shift(1)

    # Create controls
    control_cols = []
    for var in ['urate', 'infl', 'ffr']:
        for lag in range(1, lags + 1):
            col = f'{var}_L{lag}'
            data[col] = data[var].shift(lag)
            control_cols.append(col)

    # Orthogonalize
    def orthogonalize(y_var, controls, df):
        reg_data = df[[y_var] + controls].dropna()
        if len(reg_data) < len(controls) + 5:
            return pd.Series(index=df.index, dtype=float)
        X = sm.add_constant(reg_data[controls])
        model = OLS(reg_data[y_var], X)
        results = model.fit()
        resid = pd.Series(index=df.index, dtype=float)
        resid.loc[reg_data.index] = results.resid
        return resid

    data['rffr'] = orthogonalize('ffr', control_cols, data)
    data['rz'] = orthogonalize('RRCGShock', control_cols, data)

    for h in range(horizon + 1):
        data[f'rurate_f{h}'] = orthogonalize(f'urate_f{h}', control_cols, data)
        data[f'rffr_f{h}'] = orthogonalize(f'ffr_f{h}', control_cols, data)

    # Estimate LP-IV for urate
    betas_u = np.zeros(horizon + 1)
    ses_u = np.zeros(horizon + 1)

    for h in range(horizon + 1):
        y_col = f'rurate_f{h}'
        reg_data = data[[y_col, 'rffr', 'rz']].dropna()

        if len(reg_data) < 50:
            continue

        # 2SLS
        X_fs = sm.add_constant(reg_data['rz'])
        fs_model = OLS(reg_data['rffr'], X_fs)
        fs_results = fs_model.fit()
        rffr_hat = fs_results.fittedvalues

        X_ss = sm.add_constant(rffr_hat)
        ss_model = OLS(reg_data[y_col], X_ss)
        ss_results = ss_model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        betas_u[h] = ss_results.params.iloc[1]
        ses_u[h] = ss_results.bse.iloc[1]

    # GBF for unemployment
    def gbf_func(h, a, b, c):
        return a * np.exp(-((h - b) / c) ** 2)

    def gbf_objective(params, h, y):
        a, b, c = params
        pred = gbf_func(h, a, b, c)
        return np.sum((y - pred) ** 2)

    h_arr = np.arange(horizon + 1)
    valid_mask = ~np.isnan(betas_u)

    result_u = minimize(gbf_objective, [1.5, 24, 10],
                       args=(h_arr[valid_mask], betas_u[valid_mask]),
                       method='Nelder-Mead')
    a_u, b_u, c_u = result_u.x
    gbf_u = gbf_func(h_arr, a_u, b_u, c_u)

    # Counterfactual: shift peak earlier
    b_counter = b_u - 5  # Earlier peak
    gbf_counter = gbf_func(h_arr, a_u, b_counter, c_u)

    # Confidence bands
    ci_upper = gbf_u + 1.96 * ses_u
    ci_lower = gbf_u - 1.96 * ses_u
    ci_1se_upper = gbf_u + ses_u
    ci_1se_lower = gbf_u - ses_u

    # Create side-by-side figure
    fig = plt.figure(figsize=(14, 10))

    # Top row: LP vs GBF
    ax1 = fig.add_subplot(2, 2, 1)
    stata_img1 = load_stata_figure('Example8_Counterfactuals', 'counterfig_a.pdf')
    if stata_img1:
        ax1.imshow(stata_img1)
        ax1.set_title('STATA: counterfig_a.pdf', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.fill_between(h_arr, ci_1se_lower, ci_1se_upper, alpha=0.2, color='purple')
    ax2.fill_between(h_arr, ci_lower, ci_upper, alpha=0.3, color='purple')
    ax2.plot(h_arr, betas_u, 'b--', linewidth=2, label='LP')
    ax2.plot(h_arr, gbf_u, 'purple', linewidth=2.5, label='GBF')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Horizon, months, h', fontsize=10)
    ax2.set_ylabel('Response, percentage points', fontsize=10)
    ax2.set_xlim(0, horizon)
    ax2.set_xticks(np.arange(0, horizon + 1, 8))
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_title('Python: Raw LP vs GBF', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Bottom row: Counterfactual
    ax3 = fig.add_subplot(2, 2, 3)
    stata_img2 = load_stata_figure('Example8_Counterfactuals', 'counterfig_b.pdf')
    if stata_img2:
        ax3.imshow(stata_img2)
        ax3.set_title('STATA: counterfig_b.pdf', fontsize=11, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.fill_between(h_arr, ci_1se_lower, ci_1se_upper, alpha=0.2, color='purple')
    ax4.fill_between(h_arr, ci_lower, ci_upper, alpha=0.3, color='purple')
    ax4.plot(h_arr, gbf_u, 'purple', linewidth=2.5, label='GBF')
    ax4.plot(h_arr, gbf_counter, 'g--', linewidth=2.5, label='Counterfactual')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_xlabel('Horizon, months, h', fontsize=10)
    ax4.set_ylabel('Response, percentage points', fontsize=10)
    ax4.set_xlim(0, horizon)
    ax4.set_xticks(np.arange(0, horizon + 1, 8))
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_title('Python: GBF vs Counterfactual', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Example 8: Counterfactual Analysis (Figure 8)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/example8_side_by_side.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  GBF parameters: a={a_u:.3f}, b={b_u:.3f}, c={c_u:.3f}")
    print("✓ Example 8 completed: output/side_by_side/example8_side_by_side.png")
    return True


# =============================================================================
# EXAMPLE 2: Fiscal Multipliers
# =============================================================================
def example2_multipliers():
    """
    Replicate Example 2: Fiscal Multipliers

    STATA: GLP2_responses_Mfull.do, etc.
    Output: Figure2a.pdf (Rfull.pdf), Figure2b.pdf (Mfull.pdf)
    Data: fiscal_consolidation_v032023.dta, dcapb.dta, JSTdatasetR6.dta
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Fiscal Multipliers")
    print("=" * 70)

    # Load and merge data
    fiscal = pd.read_stata(f'{STATA_PATH}/Example2_Multipliers/fiscal_consolidation_v032023.dta')
    jst = pd.read_stata(f'{STATA_PATH}/Example2_Multipliers/JSTdatasetR6.dta')
    dcapb = pd.read_stata(f'{STATA_PATH}/Example2_Multipliers/dcapb.dta')

    # Merge datasets
    data = fiscal.merge(jst, on=['ifs', 'year'], how='inner')
    data = data.merge(dcapb, on=['ifs', 'year'], how='inner')

    print(f"  Merged data: {len(data)} observations")

    # Create treatment variable
    data['dCAPB'] = data['dnlgxqa']

    # Outcome variable
    data['y'] = np.log(data['rgdpbarro']) * 100

    # Sort by panel and time
    data = data.sort_values(['ifs', 'year']).reset_index(drop=True)

    horizon = 4

    # Create long-differences and cumulative sums
    for h in range(horizon + 1):
        # D_h y = y_{t+h} - y_{t-1}
        data[f'D{h}y'] = data.groupby('ifs')['y'].shift(-h) - data.groupby('ifs')['y'].shift(1)

    # Cumulative outcomes
    for h in range(horizon + 1):
        cols = [f'D{j}y' for j in range(h + 1)]
        data[f'S{h}y'] = data[cols].sum(axis=1)

    # Cumulative treatment
    for h in range(horizon + 1):
        if h == 0:
            data[f'dCAPB{h}'] = data['dCAPB']
        else:
            data[f'dCAPB{h}'] = data[f'dCAPB{h-1}'] + data.groupby('ifs')['dCAPB'].shift(-h)

    for h in range(horizon + 1):
        cols = [f'dCAPB{j}' for j in range(h + 1)]
        data[f'SdCAPB{h}'] = data[cols].sum(axis=1)

    # Controls
    data['Ldy1'] = data.groupby('ifs')['y'].diff().shift(1)
    data['Ldy2'] = data.groupby('ifs')['y'].diff().shift(2)
    data['LdCAPB1'] = data.groupby('ifs')['dCAPB'].shift(1)
    data['LdCAPB2'] = data.groupby('ifs')['dCAPB'].shift(2)

    control_cols = ['Ldy1', 'Ldy2', 'LdCAPB1', 'LdCAPB2']

    # LP-IV estimation (simplified without full panel FE)
    betas = np.zeros(horizon + 1)
    ses = np.zeros(horizon + 1)

    for h in range(horizon + 1):
        y_var = f'S{h}y'
        t_var = f'SdCAPB{h}'
        z_var = 'size'  # Government size as instrument

        reg_data = data[[y_var, t_var, z_var] + control_cols + ['ifs']].dropna()

        if len(reg_data) < 50:
            continue

        # Convert all to float to avoid dtype issues
        for col in [y_var, t_var, z_var] + control_cols:
            reg_data[col] = reg_data[col].astype(float)

        # Create country dummies (use ifs as int for cleaner dummy names)
        reg_data['ifs_int'] = reg_data['ifs'].astype(int)
        dummies = pd.get_dummies(reg_data['ifs_int'], prefix='fe', drop_first=True)
        dummies = dummies.astype(float)

        # First stage
        X_fs = pd.concat([reg_data[[z_var] + control_cols].reset_index(drop=True),
                         dummies.reset_index(drop=True)], axis=1)
        X_fs = sm.add_constant(X_fs)
        fs_model = OLS(reg_data[t_var].values, X_fs.values)
        fs_results = fs_model.fit()
        t_hat = fs_results.fittedvalues

        # Second stage
        X_ss = np.column_stack([t_hat, reg_data[control_cols].values, dummies.values])
        X_ss = sm.add_constant(X_ss)
        ss_model = OLS(reg_data[y_var].values, X_ss)
        ss_results = ss_model.fit(cov_type='cluster', cov_kwds={'groups': reg_data['ifs'].values})

        betas[h] = ss_results.params[1]  # coefficient on t_hat (first non-constant)
        ses[h] = ss_results.bse[1]

    # Confidence bands
    ci_upper = betas + 1.96 * ses
    ci_lower = betas - 1.96 * ses

    # Average multiplier
    avg_mult = np.mean(betas)

    # Create side-by-side figure
    fig = plt.figure(figsize=(14, 5))

    h_arr = np.arange(horizon + 1)

    ax1 = fig.add_subplot(1, 2, 1)
    stata_img = load_stata_figure('Example2_Multipliers', 'Mfull.pdf')
    if stata_img:
        ax1.imshow(stata_img)
        ax1.set_title('STATA: Mfull.pdf (Figure 2b)', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.fill_between(h_arr, ci_lower, ci_upper, alpha=0.2, color='blue')
    ax2.plot(h_arr, betas, 'b-', linewidth=2.5, marker='o', label='Multiplier m(h)')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=avg_mult, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Average: {avg_mult:.2f}')
    ax2.set_xlabel('Horizon, years, h', fontsize=11)
    ax2.set_ylabel('Multiplier, m(h)', fontsize=11)
    ax2.set_xlim(-0.2, horizon + 0.5)
    ax2.set_xticks(h_arr)
    ax2.legend(loc='best', fontsize=9)
    ax2.set_title('Python: Fiscal Multiplier', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Example 2: Fiscal Multipliers (Figure 2b)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/example2_side_by_side.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Multipliers at each horizon: {betas}")
    print(f"  Average multiplier: {avg_mult:.3f}")
    print("✓ Example 2 completed: output/side_by_side/example2_side_by_side.png")
    return True


# =============================================================================
# EXAMPLE 7: UK Phillips Curve - Minimum Distance
# =============================================================================
def example7_minimum_distance():
    """
    Replicate Example 7: UK Phillips Curve with Minimum Distance

    STATA: UK_Phillips_Curve.do
    Output: Figure7a-d
    Data: monthlyData.dta, monthlyShocks.dta, logrpoiluk.dta, logOilNEER.dta
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: UK Phillips Curve - Minimum Distance")
    print("=" * 70)

    # Load data
    monthly = pd.read_stata(f'{STATA_PATH}/Example7_MinimumDistance/monthlyData.dta')
    shocks = pd.read_stata(f'{STATA_PATH}/Example7_MinimumDistance/monthlyShocks.dta')
    oil = pd.read_stata(f'{STATA_PATH}/Example7_MinimumDistance/logrpoiluk.dta')
    neer = pd.read_stata(f'{STATA_PATH}/Example7_MinimumDistance/logOilNEER.dta')

    # Merge
    data = monthly.merge(shocks, on='month', how='left')
    data = data.merge(oil, on='month', how='left')
    data = data.merge(neer, on='month', how='left')

    data = data.sort_values('month').reset_index(drop=True)

    print(f"  Merged data: {len(data)} observations")

    # Create variables
    data['urate'] = data['UnempRate']
    data['policyrate'] = data['BankRate']

    # Inflation
    data['infla'] = 100 * (np.log(data['CPIindex']) - np.log(data['CPIindex'].shift(12)))
    data['infle'] = data['infla'].shift(-1)  # Expected inflation (RE)

    # Shock
    data['Shock'] = data['shock_m']

    horizon = 17
    lags = 4

    # Create forward variables
    for y_var in ['infla', 'infle', 'urate']:
        for h in range(horizon + 1):
            data[f'{y_var}_f{h}'] = data[y_var].shift(-h)

    # Create controls
    control_cols = []
    for var in ['infla', 'infle', 'urate']:
        for lag in range(1, lags + 1):
            col = f'{var}_L{lag}'
            data[col] = data[var].shift(lag)
            control_cols.append(col)

    # Orthogonalize and estimate
    def orthog_and_estimate(y_var, data, control_cols, horizon, lags):
        betas = np.zeros(horizon + 1)

        # Orthogonalize
        for h in range(horizon + 1):
            y_col = f'{y_var}_f{h}'
            reg_data = data[[y_col] + control_cols].dropna()
            if len(reg_data) < len(control_cols) + 5:
                continue
            X = sm.add_constant(reg_data[control_cols])
            model = OLS(reg_data[y_col], X)
            results = model.fit()
            data[f'r{y_var}_f{h}'] = np.nan
            data.loc[reg_data.index, f'r{y_var}_f{h}'] = results.resid

        # Orthogonalize policy rate
        reg_data = data[['policyrate'] + control_cols].dropna()
        X = sm.add_constant(reg_data[control_cols])
        model = OLS(reg_data['policyrate'], X)
        results = model.fit()
        data['rpolicyrate'] = np.nan
        data.loc[reg_data.index, 'rpolicyrate'] = results.resid

        # Orthogonalize shock
        reg_data = data[['Shock'] + control_cols].dropna()
        X = sm.add_constant(reg_data[control_cols])
        model = OLS(reg_data['Shock'], X)
        results = model.fit()
        data['rz'] = np.nan
        data.loc[reg_data.index, 'rz'] = results.resid

        # LP-IV estimation
        for h in range(horizon + 1):
            y_col = f'r{y_var}_f{h}'
            reg_data = data[[y_col, 'rpolicyrate', 'rz']].dropna()

            if len(reg_data) < 50:
                continue

            # 2SLS
            X_fs = sm.add_constant(reg_data['rz'])
            fs_model = OLS(reg_data['rpolicyrate'], X_fs)
            fs_results = fs_model.fit()
            t_hat = fs_results.fittedvalues

            X_ss = sm.add_constant(t_hat)
            ss_model = OLS(reg_data[y_col], X_ss)
            ss_results = ss_model.fit()

            betas[h] = ss_results.params.iloc[1]

        return betas

    b_infla = orthog_and_estimate('infla', data.copy(), control_cols, horizon, lags)
    b_infle = orthog_and_estimate('infle', data.copy(), control_cols, horizon, lags)
    b_ugap = orthog_and_estimate('urate', data.copy(), control_cols, horizon, lags)

    # Create side-by-side figure
    fig = plt.figure(figsize=(14, 10))

    h_arr = np.arange(horizon + 1)

    # Inflation response
    ax1 = fig.add_subplot(2, 2, 1)
    stata_img1 = load_stata_figure('Example7_MinimumDistance', 'Rinfl.pdf')
    if stata_img1:
        ax1.imshow(stata_img1)
        ax1.set_title('STATA: Rinfl.pdf', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(h_arr, b_infla, 'b-', linewidth=2)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Horizon, h', fontsize=10)
    ax2.set_ylabel('Response, inflation π', fontsize=10)
    ax2.set_title('Python: Inflation Response', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Unemployment gap response
    ax3 = fig.add_subplot(2, 2, 3)
    stata_img2 = load_stata_figure('Example7_MinimumDistance', 'Rugap.pdf')
    if stata_img2:
        ax3.imshow(stata_img2)
        ax3.set_title('STATA: Rugap.pdf', fontsize=11, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(h_arr, b_ugap, 'b-', linewidth=2)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_xlabel('Horizon, h', fontsize=10)
    ax4.set_ylabel('Response, unemployment gap x', fontsize=10)
    ax4.set_title('Python: Unemployment Response', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Example 7: UK Phillips Curve (Figure 7)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/example7_side_by_side.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Example 7 completed: output/side_by_side/example7_side_by_side.png")
    return True


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run all replications"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE REPLICATION: STATA to Python")
    print("Jordà & Taylor JEL Local Projections Paper")
    print("Examples 2-9")
    print("=" * 70)

    results = {}

    # Simulations first
    results['example3'] = example3_gbf()
    results['example4'] = example4_inference()
    results['example9'] = example9_state_dependence()

    # Data-based examples
    results['example6'] = example6_gmm_gbf()
    results['example8'] = example8_counterfactuals()
    results['example5'] = example5_significance_bands()
    results['example2'] = example2_multipliers()
    results['example7'] = example7_minimum_distance()

    # Summary
    print("\n" + "=" * 70)
    print("REPLICATION SUMMARY")
    print("=" * 70)

    for ex, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {ex}")

    print(f"\nAll outputs saved to: {OUTPUT_PATH}/")
    print("\nFiles created:")
    for f in sorted(os.listdir(OUTPUT_PATH)):
        print(f"  - {f}")

    return results


if __name__ == '__main__':
    main()
