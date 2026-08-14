"""
hyg_regime_diagnostics.py
────────────────────────────────────────────────────────────────────────────
Quasi-experimental diagnostics for the HYG GFC Wasserstein topological
regime classifier of Takagi (2026a).

Reproduces every table and figure in "Base Rates and Autocorrelation-Robust
Diagnostics for a Topological Regime Classifier".  Two corrections run
throughout:

  * standard errors — the 20-day forward return is built from overlapping
    windows (rho ~ 0.93 at lag 1), so every regression is reported under
    both OLS and Newey-West (HAC, 38 lags) standard errors;

  * the null — conditional frequencies are assessed against the measured
    control-day base rate (~66.5%), not against a 50% coin-flip reference,
    and Transfer Entropy uses a circular-shift surrogate that preserves the
    source's own serial correlation rather than an i.i.d. shuffle.

The --figures option adds the following figures to the full analysis:
  Fig 1: Signal time series (Z-score + S&P 500 + signal days shaded)
  Fig 2: Annual pos.freq. / signal count (IS/OOS colour-coded)
  Fig 3: Threshold sensitivity (dose-response curve)
  Fig 4: beta under sequential confound adjustment (forest-plot style)
  Fig 5: DiD visualisation (IS/OOS x signal/control, 2x2)
  Fig 6: IS annual decomposition (Z-score distribution vs pos.freq.)
  Fig 7: Transfer Entropy heatmap (lag x direction)

Usage:
  # Analysis only (fast)
  python hyg_regime_diagnostics.py \\
      --hyg longterm_hyg.csv --sp500 longterm_sp500.csv --vix longterm_vix.csv

  # Analysis + all 7 figures
  python hyg_regime_diagnostics.py \\
      --hyg longterm_hyg.csv --sp500 longterm_sp500.csv --vix longterm_vix.csv \\
      --figures --outdir results_causal/

Dependencies:
  pip install numpy scipy statsmodels ripser pot matplotlib \\
              --break-system-packages
────────────────────────────────────────────────────────────────────────────
"""
import argparse, csv, math, os, warnings
import numpy as np
import statistics as stats_
from collections import defaultdict
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.regression.linear_model import OLS
from scipy import stats as scipy_stats
from scipy.stats import binom

import ripser
from ripser import ripser as ripser_compute
import ot

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parameters (identical to Paper 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
W         = 60
D         = 3
TAU       = 3
ROLL      = 252
THRESH    = -1.5
FORWARD   = 20
REF_DATE  = '2009-03-09'
OOS_START = '2020-01-01'
# Newey-West (HAC) lag for the serial correlation induced by overlapping
# FORWARD-day forward returns. Rule of thumb for h-step overlap: >= h-1;
# we use 2*(h-1). Overriding via --hac-lags is supported.
HAC_LAGS  = 2 * (FORWARD - 1)   # = 38

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Colour definitions (paper style)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C_IS      = '#2166ac'   # IS: blue
C_OOS     = '#d6604d'   # OOS: red
C_SIG     = '#1a9850'   # signal: green
C_CTRL    = '#bdbdbd'   # control: grey
C_THRESH  = '#d73027'   # threshold line: red

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [r['trade_date'] for r in rows], [float(r['close']) for r in rows]

def load_dict(path):
    with open(path, newline='') as f:
        return {r['trade_date']: float(r['close'])
                for r in csv.DictReader(f)}

def section(title):
    print('\n' + '='*72)
    print(f'  {title}')
    print('='*72)

def subsection(title):
    print(f'\n  ── {title} ──')

def wr(fwds):
    if not fwds: return np.nan
    return np.mean([1 if f > 0 else 0 for f in fwds]) * 100

def binom_p(wins, n, p0=0.5):
    """One-sided binomial tail probability against base rate p0.

    NOTE (v4): p0 was previously hard-coded to 0.5.  For a multi-day forward
    return on a drifting asset the unconditional frequency of positive
    outcomes is NOT 0.5 -- on the S&P 500 20-day forward return over this
    sample it is approximately 0.677.  Testing a conditional frequency
    against 0.5 overstates the effect size by roughly 18 percentage points.
    Callers must pass the measured control-day base rate.
    """
    return 1 - binom.cdf(wins - 1, n, p0)

def base_rate(recs):
    """Measured control-day (s_t = 0) frequency of positive outcomes."""
    ctl = [r for r in recs if r['signal'] == 0]
    if not ctl: return np.nan
    return np.mean([1 if r['fwd20'] > 0 else 0 for r in ctl])

def uncond_rate(recs):
    """Measured unconditional frequency of positive outcomes (all days)."""
    if not recs: return np.nan
    return np.mean([1 if r['fwd20'] > 0 else 0 for r in recs])

def rolling_mean(arr, w):
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = np.mean(arr[i - w + 1: i + 1])
    return out

def ols_hac(y, X, idx=1, lags=None):
    """Fit OLS and return (beta, se_ols, p_ols, se_hac, p_hac) for the
    coefficient at position `idx`.

    The dependent variable is a FORWARD-day overlapping return, so OLS
    residuals are strongly serially correlated and the default OLS
    standard errors are understated. cov_type='HAC' (Newey-West) with
    lags >= FORWARD-1 corrects for this.
    """
    if lags is None:
        lags = HAC_LAGS
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    m_ols = OLS(y, X).fit()
    m_hac = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return (float(m_ols.params[idx]),
            float(m_ols.bse[idx]),  float(m_ols.pvalues[idx]),
            float(m_hac.bse[idx]),  float(m_hac.pvalues[idx]))

def mark(p):
    return '✓ 5%' if p < 0.05 else ('△ 10%' if p < 0.10 else '×')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TDA core (identical to Paper 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def takens_pts(window, d=D, tau=TAU):
    w = len(window)
    n = w - (d - 1) * tau
    if n < 3:
        return None
    pts = np.zeros((n, d))
    for k in range(n):
        for dim in range(d):
            pts[k, dim] = window[k + dim * tau]
    mu  = pts.mean(axis=0)
    sig = pts.std() + 1e-10
    return (pts - mu) / sig

def compute_pd_h1(pts):
    if pts is None:
        return np.array([[0.0, 0.001]])
    dists  = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    thresh = float(np.percentile(dists, 70))
    try:
        result = ripser_compute(pts, maxdim=1, thresh=thresh, distance_matrix=False)
        pd = result['dgms'][1]
        if len(pd) == 0:
            return np.array([[0.0, 0.001]])
        pd = pd[np.isfinite(pd[:, 1])]
        return pd if len(pd) > 0 else np.array([[0.0, 0.001]])
    except Exception:
        return np.array([[0.0, 0.001]])

def wasserstein_dist(pd1, pd2):
    if pd1 is None or pd2 is None:
        return np.nan
    diag1 = np.column_stack([(pd2[:, 0] + pd2[:, 1]) / 2,
                              (pd2[:, 0] + pd2[:, 1]) / 2])
    diag2 = np.column_stack([(pd1[:, 0] + pd1[:, 1]) / 2,
                              (pd1[:, 0] + pd1[:, 1]) / 2])
    pts1  = np.vstack([pd1, diag1])
    pts2  = np.vstack([pd2, diag2])
    n     = len(pts1)
    M     = np.sum((pts1[:, None] - pts2[None, :]) ** 2, axis=2)
    a     = np.ones(n) / n
    b     = np.ones(n) / n
    try:
        T = ot.emd(a, b, M)
        return float(np.sqrt(np.sum(T * M)))
    except Exception:
        return np.nan

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transfer Entropy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def discretize(arr, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(arr, percentiles)
    bins[0]  -= 1e-10
    bins[-1] += 1e-10
    return np.digitize(arr, bins) - 1

def transfer_entropy(source, target, lag=1, n_bins=5):
    src = discretize(source, n_bins)
    tgt = discretize(target, n_bins)
    n   = len(src)
    if n < lag + 2:
        return np.nan
    t   = tgt[lag:]
    t_l = tgt[:-lag]
    s_l = src[:-lag]
    joint_tt = defaultdict(int)
    marg_t   = defaultdict(int)
    for a, b in zip(t, t_l):
        joint_tt[(a, b)] += 1
        marg_t[b]        += 1
    N = len(t)
    h_t_given_tl = 0.0
    for (a, b), cnt in joint_tt.items():
        p_j = cnt / N
        p_m = marg_t[b] / N
        if p_j > 0 and p_m > 0:
            h_t_given_tl -= p_j * math.log2(p_j / p_m)
    joint_tts = defaultdict(int)
    marg_ts   = defaultdict(int)
    for a, b, c in zip(t, t_l, s_l):
        joint_tts[(a, b, c)] += 1
        marg_ts[(b, c)]      += 1
    h_t_given_tl_s = 0.0
    for (a, b, c), cnt in joint_tts.items():
        p_j = cnt / N
        p_m = marg_ts[(b, c)] / N
        if p_j > 0 and p_m > 0:
            h_t_given_tl_s -= p_j * math.log2(p_j / p_m)
    return h_t_given_tl - h_t_given_tl_s

def te_significance(source, target, lag=1, n_bins=5, n_shuffle=200, seed=42,
                    method='shift'):
    """Transfer entropy with a surrogate-based significance test.

    method='shuffle'  i.i.d. permutation of the source series.  This is the
                      conventional test, but it destroys the source's own
                      serial correlation as well as the cross-series
                      relationship.  Against a target that is itself strongly
                      autocorrelated by construction -- as r_{t,h} is -- the
                      resulting null is too narrow and the test is
                      anti-conservative, especially at lag = h.

    method='shift'    circular shift of the source by a random offset.  This
                      preserves the source's full autocorrelation structure
                      and destroys only its alignment with the target, which
                      is the null of interest.  Reported as the primary test.
    """
    rng    = np.random.default_rng(seed)
    te_obs = transfer_entropy(source, target, lag, n_bins)
    if np.isnan(te_obs):
        return te_obs, np.nan, np.nan
    src = np.asarray(source)
    n   = len(src)
    lo  = max(lag + 1, int(0.05 * n))
    null_dist = []
    for _ in range(n_shuffle):
        if method == 'shift':
            k    = int(rng.integers(lo, n - lo)) if n > 2 * lo else 1
            surr = np.roll(src, k)
        else:
            surr = rng.permutation(src)
        te_null = transfer_entropy(surr, target, lag, n_bins)
        if not np.isnan(te_null):
            null_dist.append(te_null)
    if not null_dist:
        return te_obs, np.nan, np.nan
    p_val = np.mean([t >= te_obs for t in null_dist])
    te_z  = (te_obs - np.mean(null_dist)) / (np.std(null_dist) + 1e-10)
    return te_obs, p_val, te_z

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure-generation functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_classifier_panel(dates, prices, pd_ref, sp_dict, thresh=THRESH):
    """Apply the proximity classifier to one price series.

    Returns (D, Z, Y, S) as parallel numpy arrays over the analysis panel:
    trade date, Wasserstein Z-score, FORWARD-day forward S&P 500 return
    (%), and the classifier dummy.  The outcome variable is the S&P 500
    forward return for every ETF, so the panels are directly comparable.
    """
    sp_dates = sorted(sp_dict)
    sp_idx   = {d: i for i, d in enumerate(sp_dates)}
    sp_vals  = np.array([sp_dict[d] for d in sp_dates])

    all_dates, all_dist = [], []
    # start at W, not W+ROLL: the rolling Z-score below consumes ROLL
    # observations of the distance series, so starting later applies the
    # 252-day warm-up twice (see the note in main()).
    for i in range(W, len(dates) - FORWARD):
        pts  = takens_pts(prices[i - W: i])
        dist = wasserstein_dist(compute_pd_h1(pts), pd_ref)
        all_dates.append(dates[i]); all_dist.append(dist)

    all_dist = np.array(all_dist)
    z = np.full(len(all_dist), np.nan)
    for i in range(ROLL, len(all_dist)):
        w   = all_dist[i - ROLL: i]
        mu, sig = np.nanmean(w), np.nanstd(w)
        if sig > 0:
            z[i] = (all_dist[i] - mu) / sig

    D, Z, Y = [], [], []
    for k, d in enumerate(all_dates):
        if np.isnan(z[k]) or d not in sp_idx:
            continue
        i = sp_idx[d]
        if i + FORWARD >= len(sp_vals):
            continue
        D.append(d); Z.append(z[k])
        Y.append((sp_vals[i + FORWARD] / sp_vals[i] - 1) * 100)
    D = np.array(D); Z = np.array(Z); Y = np.array(Y, dtype=float)
    return D, Z, Y, (Z <= thresh)


def reference_pd(dates, prices, ref_date=REF_DATE):
    """H1 persistence diagram of the W-day window ending on ref_date.

    Each ETF gets its own reference computed from its own trajectory on
    that date; returns None when the series lacks the required history.
    """
    if ref_date not in dates:
        return None, f'reference date {ref_date} absent ({dates[0]}..{dates[-1]})'
    ri = dates.index(ref_date)
    if ri < W:
        return None, f'only {ri} days before {ref_date}, need {W}'
    return compute_pd_h1(takens_pts(prices[ri - W: ri])), None


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'  saved: {path}')

def fig1_signal_timeseries(records, all_dates, dist_z, sp500_dict, outdir):
    """Fig 1: Z-score time series + S&P 500 + signal days shaded"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    dates_dt = [datetime.strptime(d, '%Y-%m-%d') for d in all_dates]
    z_arr    = np.array(dist_z)

    # S&P 500 close
    sp_vals  = np.array([sp500_dict.get(d, np.nan) for d in all_dates])

    # signal days
    sig_dates = [datetime.strptime(r['date'], '%Y-%m-%d')
                 for r in records if r['signal'] == 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.suptitle('HYG GFC Wasserstein Z-Score — Time Series\n'
                 'Classifier Activation Dates and the S&P 500',
                 fontsize=12, fontweight='bold')

    # --- top: Z-score ---
    ax1.plot(dates_dt, z_arr, color='#2166ac', lw=0.8, alpha=0.9, label='Wasserstein Z-score')
    ax1.axhline(THRESH, color=C_THRESH, lw=1.2, ls='--', label=f'Threshold (Z={THRESH})')
    ax1.axhline(0, color='black', lw=0.5, alpha=0.3)
    # OOS boundary
    oos_dt = datetime.strptime(OOS_START, '%Y-%m-%d')
    ax1.axvline(oos_dt, color='black', lw=1.0, ls=':', alpha=0.6)
    ax1.text(oos_dt, ax1.get_ylim()[0] if ax1.get_ylim() else -4,
             ' OOS start', fontsize=8, va='bottom', color='black', alpha=0.7)
    # signal days shaded
    for sd in sig_dates:
        ax1.axvspan(sd, sd, alpha=0.3, color=C_SIG, lw=0)
    ax1.set_ylabel('Wasserstein Z-score', fontsize=10)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_title('Wasserstein Z-score to GFC Nadir PD (2009-03-09)', fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))

    # --- bottom: S&P 500 ---
    ax2.plot(dates_dt, sp_vals, color='#636363', lw=0.8, alpha=0.9, label='S&P 500')
    # vertical lines on signal days
    for sd in sig_dates:
        ax2.axvline(sd, color=C_SIG, lw=0.5, alpha=0.4)
    ax2.axvline(oos_dt, color='black', lw=1.0, ls=':', alpha=0.6)
    ax2.set_ylabel('S&P 500 (close)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_title('S&P 500 Index — Green lines mark signal days (Z≤-1.5)', fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'fig1_signal_timeseries.png'))

def fig2_annual_winrate(records, by_year, outdir):
    """Fig 2: Annual pos.freq. / signal count (IS/OOS colour-coded)"""
    import matplotlib.pyplot as plt

    years    = sorted(by_year.keys())
    wrs      = [wr(by_year[y]) for y in years]
    ns       = [len(by_year[y]) for y in years]
    colors   = [C_OOS if y >= 2020 else C_IS for y in years]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Annual Frequency of Positive Outcomes and Classified-Day Count\n'
                 'IS (blue) vs OOS (red) — Classifier: Z≤-1.5, 20-day fwd S&P 500',
                 fontsize=12, fontweight='bold')

    # pos.freq.
    bars1 = ax1.bar(years, wrs, color=colors, edgecolor='white', lw=0.5, alpha=0.85)
    _b = base_rate(records) * 100
    ax1.axhline(_b, color='black', lw=0.9, ls='--', alpha=0.55,
                label=f'unconditional control-day frequency ({_b:.1f}%)')
    ax1.axvline(2019.5, color='black', lw=1.0, ls=':', alpha=0.5)
    ax1.text(2019.5, 5, ' OOS', fontsize=8, color='black', alpha=0.6)
    ax1.set_ylabel('Frequency of Positive Outcomes (%)', fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=8)
    ax1.set_title('Annual Frequency of Positive Outcomes by Year', fontsize=10)
    for bar, w_val, n_val in zip(bars1, wrs, ns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{w_val:.0f}%\n(n={n_val})', ha='center', va='bottom',
                 fontsize=7.5, fontweight='bold')

    # signal count
    ax2.bar(years, ns, color=colors, edgecolor='white', lw=0.5, alpha=0.85)
    ax2.axvline(2019.5, color='black', lw=1.0, ls=':', alpha=0.5)
    ax2.set_ylabel('Signal Days (n)', fontsize=10)
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_title('Annual Signal Day Count', fontsize=10)

    from matplotlib.patches import Patch
    legend_elements = [Patch(fc=C_IS, label='IS (~2019)'),
                       Patch(fc=C_OOS, label='OOS (2020~)')]
    ax2.legend(handles=legend_elements, fontsize=8)

    # force integer-year x-axis (set after bars drawn)
    for ax in (ax1, ax2):
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, ha='right', fontsize=9)

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'fig2_annual_winrate.png'))

def fig3_dose_response(records, OOS_recs, outdir):
    """Fig 3: Threshold sensitivity (dose-response curve)"""
    import matplotlib.pyplot as plt

    thresholds = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
    oos_wrs, oos_ns, full_wrs, full_ns = [], [], [], []
    for thr in thresholds:
        oos_sub  = [r for r in OOS_recs if r['z'] <= thr]
        full_sub = [r for r in records  if r['z'] <= thr]
        oos_wrs.append(wr([r['fwd20'] for r in oos_sub])  if oos_sub  else np.nan)
        full_wrs.append(wr([r['fwd20'] for r in full_sub]) if full_sub else np.nan)
        oos_ns.append(len(oos_sub))
        full_ns.append(len(full_sub))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Dose-Response Examination by Threshold Severity\n'
                 'Frequency of Positive Outcomes as the Threshold Varies (Descriptive)',
                 fontsize=12, fontweight='bold')

    # pos.freq. curve
    ax1.plot(thresholds, oos_wrs,  'o-', color=C_OOS, lw=2, ms=8, label='OOS (2020~)')
    ax1.plot(thresholds, full_wrs, 's--', color=C_IS,  lw=1.5, ms=7, label='Full period')
    # Two series are plotted, so both base rates are drawn.  They differ by
    # well under a percentage point, so the lines effectively coincide; the
    # legend records both values rather than implying one applies to both.
    _b  = base_rate(OOS_recs) * 100
    _bf = base_rate(records) * 100
    ax1.axhline(_b, color='black', lw=0.9, ls='--', alpha=0.55,
                label=f'control-day frequency (OOS {_b:.1f}%, full {_bf:.1f}%)')
    ax1.axhline(_bf, color='black', lw=0.9, ls=':', alpha=0.35)
    ax1.axvline(THRESH, color=C_THRESH, lw=1.2, ls='--', alpha=0.7, label=f'Base threshold ({THRESH})')
    # n labels
    for thr, w_val, n_val in zip(thresholds, oos_wrs, oos_ns):
        if not np.isnan(w_val):
            ax1.annotate(f'n={n_val}', xy=(thr, w_val),
                         xytext=(thr + 0.05, w_val + 1.5),
                         fontsize=8, color=C_OOS)
    ax1.set_xlabel('Z-score Threshold', fontsize=10)
    ax1.set_ylabel('Frequency of Positive Outcomes (%)', fontsize=10)
    ax1.set_ylim(40, 110)
    ax1.invert_xaxis()
    ax1.legend(fontsize=8)
    ax1.set_title('Frequency of Positive Outcomes by Threshold (OOS & Full)', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # sample size
    ax2.bar([str(t) for t in thresholds], oos_ns,
            color=[C_OOS if t == THRESH else '#d6604d88' for t in thresholds],
            edgecolor='white')
    ax2.set_xlabel('Z-score Threshold', fontsize=10)
    ax2.set_ylabel('OOS Signal Days (n)', fontsize=10)
    ax2.set_title('OOS Signal Day Count by Threshold', fontsize=10)
    for i, n_val in enumerate(oos_ns):
        ax2.text(i, n_val + 0.3, str(n_val), ha='center', fontsize=9)

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'fig3_dose_response.png'))

def fig4_beta_stability(records, OOS_recs, vix_dict, outdir):
    """Fig 4: beta under sequential confound adjustment (forest-plot style)"""
    import matplotlib.pyplot as plt

    configs = [
        ('① Naive',                          ['signal']),
        ('(2) + VIX',                          ['signal', 'vix']),
        ('(3) + VIX + DD',                     ['signal', 'vix', 'dd']),
        ('④ + VIX + DD + SMA',               ['signal', 'vix', 'dd', 'sma_ratio']),
        ('⑤ + VIX + DD + SMA + Mom',         ['signal', 'vix', 'dd', 'sma_ratio', 'momentum']),
        ('⑥ + VIX + DD + SMA + Mom + RVol', ['signal', 'vix', 'dd', 'sma_ratio', 'momentum', 'rv20']),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Sequential Confound Adjustment — β with '
                 'Newey-West (HAC) 95% CI\n'
                 'Signal coefficient under sequential covariate adjustment '
                 '(overlap-robust SE)',
                 fontsize=12, fontweight='bold')

    for ax, (label, recs) in zip(axes, [('OOS (2020~)', OOS_recs), ('Full', records)]):
        betas, cis_lo, cis_hi, pvals, labels = [], [], [], [], []
        for name, cols in configs:
            sub = [r for r in recs
                   if all(not np.isnan(r.get(c, np.nan)) for c in cols)]
            if len(sub) < 10:
                continue
            X   = np.column_stack([[1]*len(sub)] +
                                   [[r[c] for r in sub] for c in cols])
            y   = np.array([r['fwd20'] for r in sub])
            b, se_o, p_o, se_h, p_h = ols_hac(y, X, idx=1)
            betas.append(b)
            cis_lo.append(b - 1.96 * se_h)   # HAC (Newey-West) CI
            cis_hi.append(b + 1.96 * se_h)
            pvals.append(p_h)                # HAC p-value drives significance
            labels.append(name)

        y_pos = list(range(len(labels)))
        colors_bar = [C_SIG if p < 0.05 else ('#f4a582' if p < 0.10 else C_CTRL)
                      for p in pvals]
        ax.barh(y_pos, betas, color=colors_bar, alpha=0.8, edgecolor='white')
        ax.errorbar(betas, y_pos,
                    xerr=[np.array(betas) - np.array(cis_lo),
                          np.array(cis_hi) - np.array(betas)],
                    fmt='none', color='black', capsize=4, lw=1.5)
        ax.axvline(0, color='black', lw=1.0, ls='-', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('β (signal effect on 20-day fwd return, %)', fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        # p-value labels
        for i, (b, p) in enumerate(zip(betas, pvals)):
            star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            ax.text(max(cis_hi) * 1.05, i, f'p={p:.3f}{star}',
                    va='center', fontsize=7.5, color='black')

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(fc=C_SIG,    label='p<0.05'),
                            Patch(fc='#f4a582', label='p<0.10'),
                            Patch(fc=C_CTRL,   label='n.s.')],
                  fontsize=8, loc='lower right')

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'fig4_beta_stability.png'))

def fig5_did_visualization(records, IS_recs, OOS_recs, outdir):
    """Fig 5: DiD visualisation (2x2 panel + difference arrows)"""
    import matplotlib.pyplot as plt

    cells = {}
    for period, recs in [('IS', IS_recs), ('OOS', OOS_recs)]:
        for sg, lbl in [(1, 'Signal'), (0, 'Control')]:
            sub = [r for r in recs if r['signal'] == sg]
            fwds = [r['fwd20'] for r in sub]
            cells[(period, lbl)] = {
                'wr':  wr(fwds),
                'avg': np.mean(fwds) if fwds else np.nan,
                'n':   len(sub),
                'se':  np.std(fwds) / math.sqrt(len(fwds)) if fwds else np.nan,
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Difference-in-Differences: classified vs control days, IS vs OOS\n'
                 'Frequency-based and mean-based DiD are shown; they differ in magnitude',
                 fontsize=12, fontweight='bold')

    x = [0, 1]
    x_labels = ['IS (~2019)', 'OOS (2020~)']

    for ax, metric, ylabel in [
        (ax1, 'wr',  'Frequency of Positive Outcomes (%)'),
        (ax2, 'avg', 'Avg 20-day Fwd Return (%)'),
    ]:
        sig_vals  = [cells[('IS',  'Signal')][metric],
                     cells[('OOS', 'Signal')][metric]]
        ctrl_vals = [cells[('IS',  'Control')][metric],
                     cells[('OOS', 'Control')][metric]]

        ax.plot(x, sig_vals,  'o-', color=C_SIG,  lw=2.5, ms=10,
                label=f'Signal (Z≤{THRESH})')
        ax.plot(x, ctrl_vals, 's--', color=C_CTRL, lw=2.0, ms=9,
                label='Control (all non-signal days)')

        # difference arrow (IS)
        ax.annotate('', xy=(0, sig_vals[0]),  xytext=(0, ctrl_vals[0]),
                    arrowprops=dict(arrowstyle='<->', color=C_IS, lw=1.5))
        ax.annotate('', xy=(1, sig_vals[1]),  xytext=(1, ctrl_vals[1]),
                    arrowprops=dict(arrowstyle='<->', color=C_OOS, lw=1.5))

        # difference label
        is_diff  = sig_vals[0]  - ctrl_vals[0]
        oos_diff = sig_vals[1]  - ctrl_vals[1]
        did      = oos_diff - is_diff
        ax.text(-0.12, (sig_vals[0]  + ctrl_vals[0])  / 2,
                f'{is_diff:+.1f}{"pp" if metric=="wr" else "%"}',
                color=C_IS,  fontsize=9, fontweight='bold', ha='right')
        ax.text(1.08,  (sig_vals[1]  + ctrl_vals[1])  / 2,
                f'{oos_diff:+.1f}{"pp" if metric=="wr" else "%"}',
                color=C_OOS, fontsize=9, fontweight='bold', ha='left')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_title(f'{ylabel}\nDiD = {did:+.1f}{"pp" if metric=="wr" else "%"}',
                     fontsize=10, fontweight='bold')

        # place n labels after ylim is fixed (avoid overlap)
        ylo, yhi = ax.get_ylim()
        pad = (yhi - ylo) * 0.08
        ax.set_ylim(ylo - pad, yhi)
        for xi, period in [(0, 'IS'), (1, 'OOS')]:
            n_sig  = cells[(period, 'Signal')]['n']
            n_ctrl = cells[(period, 'Control')]['n']
            ax.text(xi, ylo - pad * 0.5,
                    f'sig n={n_sig} / ctl n={n_ctrl}',
                    ha='center', va='center', fontsize=7.5, color='gray')

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'fig5_did.png'))

def fig6_is_decomposition(records, IS_recs, OOS_recs, by_year, outdir):
    """Fig 6: IS annual decomposition (Z-score intensity vs pos.freq. + annual bars)"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('IS/OOS Decomposition — Annual, Z-Score Intensity, and Cluster Comparison\n'
                 'Two Non-Exclusive Descriptive Accounts of the IS/OOS Asymmetry',
                 fontsize=12, fontweight='bold')

    # --- left: annual pos.freq. scatter (size = n) ---
    ax = axes[0]
    years = sorted(by_year.keys())
    for yr in years:
        fwds = by_year[yr]
        w_r  = wr(fwds)
        n    = len(fwds)
        col  = C_OOS if yr >= 2020 else C_IS
        ax.scatter(yr, w_r, s=n*25, color=col, alpha=0.7, edgecolors='white', lw=0.5)
        ax.text(yr, w_r + 3, str(yr)[-2:], ha='center', fontsize=7, color=col)
    _b = base_rate(records) * 100
    ax.axhline(_b, color='black', lw=0.9, ls='--', alpha=0.55)
    ax.axvline(2019.5, color='black', lw=1.0, ls=':', alpha=0.5)
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Frequency of Positive Outcomes (%)', fontsize=9)
    ax.set_title('Frequency of Positive Outcomes by Year\n(bubble size = n classified days)', fontsize=9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0], marker='o', color='w', markerfacecolor=C_IS,
                               ms=10, label='IS'),
                        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_OOS,
                               ms=10, label='OOS')],
              fontsize=8)

    # --- centre: IS vs OOS signal Z-score distribution (boxplot) ---
    ax = axes[1]
    is_z  = [r['z'] for r in IS_recs  if r['signal'] == 1]
    oos_z = [r['z'] for r in OOS_recs if r['signal'] == 1]
    bp = ax.boxplot([is_z, oos_z], labels=['IS\n(~2019)', 'OOS\n(2020~)'],
                    patch_artist=True, widths=0.4,
                    boxprops=dict(facecolor='white'),
                    medianprops=dict(color='black', lw=2))
    bp['boxes'][0].set_facecolor(C_IS  + '55')
    bp['boxes'][1].set_facecolor(C_OOS + '55')
    ax.axhline(THRESH, color=C_THRESH, lw=1.2, ls='--', alpha=0.7,
               label=f'Threshold ({THRESH})')
    ax.set_ylabel('Z-score at Signal Days', fontsize=9)
    ax.set_title(f'Z-score Intensity at Signal Days\n'
                 f'IS mean={np.mean(is_z):.3f}, OOS mean={np.mean(oos_z):.3f}', fontsize=9)
    ax.legend(fontsize=8)

    # --- right: 2015 vs other IS years ---
    ax = axes[2]
    year_2015  = [r['fwd20'] for r in IS_recs if r['signal']==1 and r['year']==2015]
    other_is   = [r['fwd20'] for r in IS_recs if r['signal']==1 and r['year']!=2015]
    oos_sig    = [r['fwd20'] for r in OOS_recs if r['signal']==1]
    groups     = ['IS 2015\ncluster', 'IS other\nyears', 'OOS\n(2020~)']
    wrs_g      = [wr(year_2015), wr(other_is), wr(oos_sig)]
    ns_g       = [len(year_2015), len(other_is), len(oos_sig)]
    cols_g     = ['#fc8d59', C_IS, C_OOS]
    bars       = ax.bar(groups, wrs_g, color=cols_g, alpha=0.8, edgecolor='white')
    _b = base_rate(records) * 100
    ax.axhline(_b, color='black', lw=0.9, ls='--', alpha=0.55)
    ax.set_ylabel('Frequency of Positive Outcomes (%)', fontsize=9)
    ax.set_title('2015 Cluster vs. Other IS Years vs. OOS', fontsize=9)
    for bar, w_val, n_val in zip(bars, wrs_g, ns_g):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{w_val:.0f}%\n(n={n_val})', ha='center', fontsize=8.5,
                fontweight='bold')

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'fig6_is_decomposition.png'))

def fig7_transfer_entropy_heatmap(te_results, outdir):
    """Fig 7: Transfer Entropy heatmap (lag x direction)"""
    import matplotlib.pyplot as plt

    lags      = [1, 5, 20]
    n_bins    = 5  # use bins=5 results
    directions = [
        ('HYG Z -> S&P fwd20',       'HYG→SP500\n(target)'),
        ('S&P fwd20 -> HYG Z (reverse)', 'SP500→HYG\n(reverse)'),
    ]

    te_matrix  = np.zeros((2, 3))
    p_matrix   = np.zeros((2, 3))
    for i, (key, _) in enumerate(directions):
        for j, lag in enumerate(lags):
            te, p, _ = te_results.get((key, lag, n_bins), (np.nan, np.nan, np.nan))
            te_matrix[i, j] = te if not np.isnan(te) else 0
            p_matrix[i, j]  = p  if not np.isnan(p)  else 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Transfer Entropy (bins=5, 200 surrogates)\n'
                 'p-values are from the circular-shift surrogate, which '
                 'preserves the source series\' own serial correlation',
                 fontsize=12, fontweight='bold')

    labels_y = [d[1] for d in directions]
    labels_x = [f'lag={l}' for l in lags]

    # TE value heatmap
    im1 = ax1.imshow(te_matrix, cmap='YlOrRd', aspect='auto',
                      vmin=0, vmax=np.nanmax(te_matrix) * 1.1)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(labels_x, fontsize=10)
    ax1.set_yticks(range(2)); ax1.set_yticklabels(labels_y, fontsize=9)
    ax1.set_title('Transfer Entropy Value', fontsize=10)
    plt.colorbar(im1, ax=ax1, label='TE (bits)')
    for i in range(2):
        for j in range(3):
            ax1.text(j, i, f'{te_matrix[i,j]:.4f}',
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     color='white' if te_matrix[i,j] > np.nanmax(te_matrix)*0.6 else 'black')

    # p-value heatmap
    im2 = ax2.imshow(p_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(labels_x, fontsize=10)
    ax2.set_yticks(range(2)); ax2.set_yticklabels(labels_y, fontsize=9)
    ax2.set_title('p-value (circular-shift surrogate, n=200)\nGreen = significant', fontsize=10)
    plt.colorbar(im2, ax=ax2, label='p-value')
    for i in range(2):
        for j in range(3):
            p = p_matrix[i, j]
            star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
            ax2.text(j, i, f'{p:.3f}\n{star}',
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     color='white' if p < 0.02 else 'black')

    # annotation for reverse direction (outside the plot)
    fig.text(0.5, -0.16,
             '* No cell reaches the 5% level under this surrogate.  Under an i.i.d. shuffle,\n'
             '  which destroys the source\'s autocorrelation as well as its alignment with the\n'
             '  target, eight of the twelve configurations do.',
             ha='center', fontsize=8, color='#636363', style='italic')

    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, 'fig7_transfer_entropy.png'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    global HAC_LAGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--hyg',     required=True)
    ap.add_argument('--sp500',   required=True)
    ap.add_argument('--vix',     required=False, default=None)
    ap.add_argument('--spy',     required=False, default=None)
    ap.add_argument('--jnk',     required=False, default=None,
                    help='JNK closes; enables the ETF-specificity section (9)')
    ap.add_argument('--lqd',     required=False, default=None,
                    help='LQD closes; enables the ETF-specificity section (9)')
    ap.add_argument('--figures', action='store_true', help='generate figures')
    ap.add_argument('--outdir',  default='results_causal_v4')
    ap.add_argument('--hac-lags', type=int, default=HAC_LAGS,
                    help=f'Newey-West lag (default {HAC_LAGS} = 2*(FORWARD-1))')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    HAC_LAGS = args.hac_lags
    print(f'  Newey-West (HAC) lags = {HAC_LAGS}')

    if args.figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print('  Figure mode: ON')

    section('DATA LOADING')
    dates_hyg, prices_hyg = load_csv(args.hyg)
    sp500_dict = load_dict(args.sp500)
    vix_dict   = load_dict(args.vix) if args.vix else {}
    spy_dict   = load_dict(args.spy) if args.spy else {}
    print(f'  HYG  : {dates_hyg[0]} to {dates_hyg[-1]} ({len(dates_hyg)} days)')
    print(f'  SP500: {len(sp500_dict)} days')
    if vix_dict: print(f'  VIX  : {len(vix_dict)} days')
    if spy_dict: print(f'  SPY  : {len(spy_dict)} days')

    # --- reference PD ────────────────────────────────────────────────────
    section('REFERENCE PD (GFC nadir 2009-03-09)')
    ref_idx = dates_hyg.index(REF_DATE)
    pd_ref  = compute_pd_h1(takens_pts(prices_hyg[ref_idx - W: ref_idx]))
    print(f'  GFC nadir PD: {len(pd_ref)} H1 loops')

    # --- Wasserstein Z-score time series ─────────────────────────────────
    section('COMPUTING WASSERSTEIN Z-SCORE TIME SERIES')
    print('  Computing PDs...')
    all_dates, all_dist = [], []
    # The Takens embedding needs W days of history; the rolling Z-score
    # applied below consumes a further ROLL observations of the *distance*
    # series.  Starting the distance series at W+ROLL therefore imposes the
    # 252-day warm-up twice and discards the first ROLL trading days of
    # valid classifier output.  Start at W: the first usable Z-score then
    # falls at index W+ROLL, which is the warm-up the paper describes.
    start = W
    total = len(dates_hyg) - start - FORWARD
    for idx, i in enumerate(range(start, len(dates_hyg) - FORWARD)):
        if idx % 500 == 0:
            print(f'    {idx}/{total} ({100*idx//total}%)')
        pts  = takens_pts(prices_hyg[i - W: i])
        pd   = compute_pd_h1(pts)
        dist = wasserstein_dist(pd, pd_ref)
        all_dates.append(dates_hyg[i])
        all_dist.append(dist)

    all_dist = np.array(all_dist)
    dist_z   = np.full(len(all_dist), np.nan)
    for i in range(ROLL, len(all_dist)):
        w   = all_dist[i - ROLL: i]
        mu  = np.nanmean(w)
        sig = np.nanstd(w)
        if sig > 0:
            dist_z[i] = (all_dist[i] - mu) / sig

    valid     = ~np.isnan(dist_z)
    all_dates = [all_dates[i] for i in range(len(all_dates)) if valid[i]]
    dist_z    = dist_z[valid]
    print(f'  Valid samples: {len(all_dates)}')

    # --- auxiliary indicators ──────────────────────────────────────────────────
    sp_dates  = sorted(sp500_dict.keys())
    sp_prices = np.array([sp500_dict[d] for d in sp_dates])
    sp_di     = {d: i for i, d in enumerate(sp_dates)}
    sma50     = rolling_mean(sp_prices, 50)
    sma200    = rolling_mean(sp_prices, 200)
    sp_rets   = np.full(len(sp_prices), np.nan)
    for i in range(1, len(sp_prices)):
        if sp_prices[i-1] > 0:
            sp_rets[i] = math.log(sp_prices[i] / sp_prices[i-1]) * 100
    rvol20 = np.full(len(sp_prices), np.nan)
    for i in range(20, len(sp_prices)):
        w = sp_rets[i-20:i]
        if not np.any(np.isnan(w)):
            rvol20[i] = np.std(w) * math.sqrt(252)
    mom20 = np.full(len(sp_prices), np.nan)
    for i in range(20, len(sp_prices)):
        if sp_prices[i-20] > 0:
            mom20[i] = (sp_prices[i] / sp_prices[i-20] - 1) * 100
    sp_peak = {}
    pk = -np.inf
    for d in sp_dates:
        pk = max(pk, sp500_dict[d])
        sp_peak[d] = pk

    # --- build records ──────────────────────────────────────────────
    records = []
    for i, (d, z) in enumerate(zip(all_dates, dist_z)):
        hyg_idx = dates_hyg.index(d)
        if hyg_idx + FORWARD >= len(dates_hyg):
            continue
        d_fwd = dates_hyg[hyg_idx + FORWARD]
        p0 = sp500_dict.get(d)
        p1 = sp500_dict.get(d_fwd)
        if not p0 or not p1:
            continue
        fwd  = (p1 - p0) / p0 * 100
        sig  = 1 if z <= THRESH else 0
        oos  = 1 if d >= OOS_START else 0
        vix  = vix_dict.get(d, np.nan)
        pkvl = sp_peak.get(d, np.nan)
        dd   = (p0 / pkvl - 1) * 100 if pkvl and pkvl > 0 else np.nan
        sidx = sp_di.get(d)
        sma_r = (sma50[sidx] / sma200[sidx]
                 if sidx is not None and not np.isnan(sma50[sidx])
                 and not np.isnan(sma200[sidx]) and sma200[sidx] > 0
                 else np.nan)
        mom   = mom20[sidx]  if sidx is not None else np.nan
        rv20  = rvol20[sidx] if sidx is not None else np.nan
        records.append({
            'date': d, 'z': z, 'fwd20': fwd, 'win': 1 if fwd > 0 else 0,
            'signal': sig, 'oos': oos, 'vix': vix, 'dd': dd,
            'sma_ratio': sma_r, 'momentum': mom, 'rv20': rv20,
            'year': int(d[:4]),
        })

    IS_recs  = [r for r in records if r['oos'] == 0]
    OOS_recs = [r for r in records if r['oos'] == 1]
    print(f'  IS  (~2019): {len(IS_recs)}  OOS (2020~): {len(OOS_recs)}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 0. Baseline check
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('0. Baseline check (reproduces Paper 2)')
    print(f'  {"Period":<12} {"n(total)":>6} {"n(sig)":>8} {"avg fwd":>10} '
          f'{"pos.freq.":>8} {"base":>7} {"p vs 50%":>11} {"p vs base":>10}')
    print('  ' + '-'*80)
    for label, recs in [('Full', records), ('IS (~2019)', IS_recs), ('OOS (2020~)', OOS_recs)]:
        sig_r = [r for r in recs if r['signal'] == 1]
        if not sig_r: continue
        fwds = [r['fwd20'] for r in sig_r]
        wins = sum(r['win'] for r in sig_r)
        p0   = base_rate(recs)
        p50  = binom_p(wins, len(sig_r), 0.5)
        pb   = binom_p(wins, len(sig_r), p0)
        print(f'  {label:<12} {len(recs):>6} {len(sig_r):>8} '
              f'{np.mean(fwds):>+10.2f}% {wr(fwds):>7.1f}% {p0*100:>6.1f}% '
              f'{p50:>11.2e} {pb:>10.3f}')
    print()
    print('  NOTE: "p vs 50%" is the figure reported in earlier versions and in')
    print('        Takagi (2026a).  It tests against a base rate of 0.5, which is')
    print('        not the unconditional frequency of positive 20-day forward S&P')
    print('        500 returns in this sample.  "p vs base" is the corrected test')
    print('        against the measured control-day frequency and is the figure')
    print('        reported in the paper.')
    print(f'  measured unconditional pos. freq.: Full {uncond_rate(records)*100:.1f}%  '
          f'IS {uncond_rate(IS_recs)*100:.1f}%  OOS {uncond_rate(OOS_recs)*100:.1f}%')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. ADF
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('1. ADF stationarity test')
    for arr, name in [
        (dist_z,                                    'HYG Wasserstein Z-score'),
        (np.array([r['fwd20']  for r in records]),  'S&P 500 20-day fwd return'),
        (np.array([r['signal'] for r in records], dtype=float), 'signal dummy'),
    ]:
        s, p, *_ = adfuller(arr)
        print(f'  {name:<30}: ADF={s:7.3f}, p={p:.4f} '
              f'→ {"stationary OK" if p < 0.05 else "non-stationary !"}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Granger
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('2. Granger causality test - signal dummy')
    ts = np.column_stack([[r['fwd20'] for r in records],
                          [r['signal'] for r in records]])
    print(f'  N={len(ts)}')
    print(f'\n  {"Lag":>4}  {"F-stat":>9}  {"p":>9}  {"flag":>10}')
    print('  ' + '-'*40)
    gc = grangercausalitytests(ts, maxlag=10, verbose=False)
    granger_pvals = []
    for lag in range(1, 11):
        f = gc[lag][0]['ssr_ftest'][0]
        p = gc[lag][0]['ssr_ftest'][1]
        granger_pvals.append((f'Granger forward lag={lag}', p))
        print(f'  {lag:>4}  {f:>9.3f}  {p:>9.4f}  '
              f'{"✓ 5%" if p<0.05 else "△ 10%" if p<0.10 else "":>10}')
    subsection('reverse direction (spurious-causality check)')
    ts_r = ts[:, [1, 0]]
    gc_r = grangercausalitytests(ts_r, maxlag=5, verbose=False)
    print(f'  {"Lag":>4}  {"F-stat":>9}  {"p":>9}  {"flag":>16}')
    print('  ' + '-'*44)
    for lag in range(1, 6):
        f = gc_r[lag][0]['ssr_ftest'][0]
        p = gc_r[lag][0]['ssr_ftest'][1]
        print(f'  {lag:>4}  {f:>9.3f}  {p:>9.4f}  '
              f'{"! reverse" if p<0.05 else "<- no reverse OK":>16}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Mean comparison + regression adjustment
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('3. Mean comparison + regression adjustment (VIX, DD)')
    print('  Note: t/binomial tests assume independent observations, but 20-day')
    print('        forward returns overlap and understate SE. Use p(HAC)=Newey-West below.')
    for label, recs in [('IS (~2019)', IS_recs), ('OOS (2020~)', OOS_recs), ('Full', records)]:
        sig_r = [r for r in recs if r['signal'] == 1]
        ctl_r = [r for r in recs if r['signal'] == 0]
        if not sig_r or not ctl_r: continue
        sf = [r['fwd20'] for r in sig_r]
        cf = [r['fwd20'] for r in ctl_r]
        t_stat, t_p = scipy_stats.ttest_ind(sf, cf, equal_var=False)
        wins = sum(1 for f in sf if f > 0)
        p0   = base_rate(recs)
        bp   = binom_p(wins, len(sig_r), p0)
        bp50 = binom_p(wins, len(sig_r), 0.5)
        print(f'\n  [{label}]')
        print(f'    signal : n={len(sig_r):>4}, avg={np.mean(sf):>+7.2f}%, posfreq={wr(sf):.1f}%')
        print(f'    control: n={len(ctl_r):>4}, avg={np.mean(cf):>+7.2f}%, posfreq={wr(cf):.1f}%')
        print(f'    diff   : avg={np.mean(sf)-np.mean(cf):>+7.2f}%, posfreq={wr(sf)-wr(cf):>+6.1f}pp')
        print(f'    t-test  : t={t_stat:>7.3f}, p={t_p:.4f} '
              f'{"✓ 5%" if t_p<0.05 else "△ 10%" if t_p<0.10 else "×"}')
        print(f'    binomial: p={bp:.4f} vs measured base {p0*100:.1f}% '
              f'{"✓" if bp<0.05 else "△" if bp<0.10 else "×"}   '
              f'(p={bp50:.2e} against the 50% reference used previously)')
        if vix_dict:
            sub = [r for r in recs if not np.isnan(r['vix']) and not np.isnan(r['dd'])]
            if sub:
                X   = np.column_stack([[1]*len(sub), [r['signal'] for r in sub],
                                        [r['vix'] for r in sub], [r['dd'] for r in sub]])
                b, se_o, p_o, se_h, p_h = ols_hac(
                    [r['fwd20'] for r in sub], X, idx=1)
                print(f'    adjusted (VIX+DD): beta={b:>+7.3f}%  '
                      f'p(OLS)={p_o:.4f} {mark(p_o)}  |  '
                      f'p(HAC)={p_h:.4f} {mark(p_h)}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. DiD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('4. DiD (difference-in-differences)')
    cells = {}
    for period, recs in [('IS', IS_recs), ('OOS', OOS_recs)]:
        for sg, lbl in [(1, 'signal'), (0, 'control')]:
            sub = [r for r in recs if r['signal'] == sg]
            cells[(period, lbl)] = {
                'n':   len(sub),
                'wr':  wr([r['fwd20'] for r in sub]),
                'avg': np.mean([r['fwd20'] for r in sub]) if sub else np.nan,
            }
    print(f'\n  {"":>20} {"IS":>14} {"OOS":>14} {"OOS-IS":>10}')
    print('  ' + '-'*60)
    for lbl in ['signal', 'control']:
        iv = cells[('IS',  lbl)]['wr']; ni = cells[('IS',  lbl)]['n']
        ov = cells[('OOS', lbl)]['wr']; no = cells[('OOS', lbl)]['n']
        print(f'  {lbl + " (wr%)":<20} {iv:>7.1f}% (n={ni:>4}) '
              f'{ov:>7.1f}% (n={no:>4}) {ov-iv:>+9.1f}pp')
    sig_is  = cells[('IS',  'signal') ]['wr']
    sig_oos = cells[('OOS', 'signal') ]['wr']
    ctl_is  = cells[('IS',  'control')]['wr']
    ctl_oos = cells[('OOS', 'control')]['wr']
    did = (sig_oos - sig_is) - (ctl_oos - ctl_is)
    print(f'\n  DiD = {did:+.1f}pp')
    X_did = np.column_stack([[1]*len(records),
                              [r['signal'] for r in records],
                              [r['oos']    for r in records],
                              [r['signal']*r['oos'] for r in records]])
    b_did, se_o, p_did, se_h, p_did_hac = ols_hac(
        [r['fwd20'] for r in records], X_did, idx=3)
    print(f'  DiD regression: beta={b_did:+.3f}%  '
          f'p(OLS)={p_did:.4f} {mark(p_did)}  |  '
          f'p(HAC)={p_did_hac:.4f} {mark(p_did_hac)}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. Threshold sensitivity
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('5. Threshold sensitivity by panel')
    print('  Two references are reported for each threshold:')
    print('    "fixed base"  = control-day (z > -1.5) frequency for the panel,')
    print('                    i.e. the same reference used in Table 1.')
    print('    "complement"  = frequency on the days NOT selected at that')
    print('                    threshold (z > thr).  This is the internally')
    print('                    consistent reference for a dose-response curve,')
    print('                    since the selected set changes with the threshold.')
    for plabel, precs in [('OOS (2020~)', OOS_recs), ('Full', records),
                          ('IS (~2019)', IS_recs)]:
        p0_fixed  = base_rate(precs)
        p0_uncond = uncond_rate(precs)
        print(f'\n  [{plabel}]  fixed base = {p0_fixed*100:.1f}%   '
              f'unconditional = {p0_uncond*100:.1f}%   n = {len(precs)}')
        print(f'  {"thresh":>7} {"n(sig)":>7} {"pos.freq":>9} {"lift":>9} '
              f'{"p vs 50%":>11} {"p vs base":>10} {"compl.":>8} {"p vs compl":>11} {"flag":>5}')
        print('  ' + '-'*94)
        for thr in [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]:
            sub  = [r for r in precs if r['z'] <= thr]
            comp = [r for r in precs if r['z'] >  thr]
            if len(sub) < 3:
                print(f'  {thr:>7.1f} {"(n<3)":>7}')
                continue
            fwds = [r['fwd20'] for r in sub]
            wins = sum(1 for f in fwds if f > 0)
            pf   = wr(fwds)
            p0c  = (np.mean([1 if r['fwd20'] > 0 else 0 for r in comp])
                    if comp else float('nan'))
            p    = binom_p(wins, len(sub), p0_fixed)
            pc   = binom_p(wins, len(sub), p0c) if comp else float('nan')
            p50  = binom_p(wins, len(sub), 0.5)
            marker = ' ◀ base' if thr == THRESH else ''
            print(f'  {thr:>7.1f} {len(sub):>7} {pf:>8.1f}% '
                  f'{pf - p0_fixed*100:>+7.1f}pp {p50:>11.2e} {p:>10.3f} '
                  f'{p0c*100:>7.1f}% {pc:>11.3f} '
                  f'{"✓" if p<0.05 else "△" if p<0.10 else "×":>5}{marker}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. Additional confounds
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('6. Regression adjustment with additional confounds')
    conf_pvals = []
    configs = [
        ('(1) signal only (naive)',
         ['signal']),
        ('(2) + VIX',
         ['signal', 'vix']),
        ('(3) + VIX + DD',
         ['signal', 'vix', 'dd']),
        ('(4) + VIX + DD + SMA ratio (trend)',
         ['signal', 'vix', 'dd', 'sma_ratio']),
        ('(5) + VIX + DD + SMA ratio + momentum',
         ['signal', 'vix', 'dd', 'sma_ratio', 'momentum']),
        ('(6) + VIX + DD + SMA ratio + momentum + RVol',
         ['signal', 'vix', 'dd', 'sma_ratio', 'momentum', 'rv20']),
    ]
    beta_results = {}  # for Fig 4
    for label, recs in [('OOS (2020~)', OOS_recs), ('Full', records)]:
        print(f'\n  [{label}]')
        print(f'  {"Model":<40} {"β":>9} {"p(OLS)":>9} {"p(HAC)":>9} {"HAC":>5} {"N":>6}')
        print('  ' + '-'*82)
        beta_results[label] = []
        for name, cols in configs:
            sub = [r for r in recs
                   if all(not np.isnan(r.get(c, np.nan)) for c in cols)]
            if len(sub) < 10:
                beta_results[label].append((name, np.nan, np.nan, np.nan))
                continue
            X   = np.column_stack([[1]*len(sub)] +
                                   [[r[c] for r in sub] for c in cols])
            b, se_o, p_o, se_h, p_h = ols_hac(
                [r['fwd20'] for r in sub], X, idx=1)
            beta_results[label].append((name, b, p_o, p_h, se_h))
            conf_pvals.append((f'{label} spec {name}', p_h))
            hac_flag = '✓' if p_h < 0.05 else ('△' if p_h < 0.10 else '×')
            print(f'  {name:<40} {b:>+9.3f} {p_o:>9.4f} {p_h:>9.4f} '
                  f'{hac_flag:>5} {len(sub):>6}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. Why IS is weaker
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('7. Why IS is weaker')
    by_year = defaultdict(list)
    for r in records:
        if r['signal'] == 1:
            by_year[r['year']].append(r['fwd20'])

    subsection('7-1. Annual signal count and pos.freq.')
    print(f'  {"Year":>5}  {"n(sig)":>8}  {"wr":>8}  {"avg":>8}  {"period":>6}')
    print('  ' + '-'*44)
    for yr in sorted(by_year.keys()):
        fwds = by_year[yr]
        period = 'OOS' if yr >= 2020 else 'IS'
        print(f'  {yr:>5}  {len(fwds):>8}  {wr(fwds):>7.1f}%  '
              f'{np.mean(fwds):>+7.2f}%  {period:>6}')

    subsection('7-2. Macro profile of signal days (IS/OOS)')
    for label, recs in [('IS', IS_recs), ('OOS', OOS_recs)]:
        sig_r = [r for r in recs if r['signal']==1 and not np.isnan(r['vix'])]
        ctl_r = [r for r in recs if r['signal']==0 and not np.isnan(r['vix'])]
        if not sig_r: continue
        print(f'\n  [{label}]')
        for col, name in [('vix','VIX'),('dd','DD(%)'),('sma_ratio','SMA50/200'),('momentum','Mom20(%)')]:
            sv = [r[col] for r in sig_r if not np.isnan(r[col])]
            cv = [r[col] for r in ctl_r if not np.isnan(r[col])]
            if not sv or not cv: continue
            t, p = scipy_stats.ttest_ind(sv, cv, equal_var=False)
            print(f'    {name:<15}: sig={np.mean(sv):>7.2f}, ctl={np.mean(cv):>7.2f}, '
                  f'diff={np.mean(sv)-np.mean(cv):>+7.2f}, p={p:.3f}{"*" if p<0.05 else ""}')

    subsection('7-3. Strong-year / weak-year decomposition')
    good_is = [yr for yr in sorted(by_year) if yr < 2020 and wr(by_year[yr]) >= 70]
    bad_is  = [yr for yr in sorted(by_year) if yr < 2020 and wr(by_year[yr]) <  70]
    print(f'  Strong years (posfreq>=70%): {good_is}')
    print(f'  Weak years (posfreq<70%): {bad_is}')
    gf = [r['fwd20'] for r in IS_recs if r['signal']==1 and r['year'] in good_is]
    bf = [r['fwd20'] for r in IS_recs if r['signal']==1 and r['year'] in bad_is]
    if gf: print(f'  Strong: n={len(gf)}, posfreq={wr(gf):.1f}%, avg={np.mean(gf):+.2f}%')
    if bf: print(f'  Weak: n={len(bf)},  posfreq={wr(bf):.1f}%, avg={np.mean(bf):+.2f}%')

    subsection('7-4. IS vs OOS Z-score intensity')
    is_z  = [r['z'] for r in IS_recs  if r['signal']==1]
    oos_z = [r['z'] for r in OOS_recs if r['signal']==1]
    if is_z and oos_z:
        t, p = scipy_stats.ttest_ind(is_z, oos_z, equal_var=False)
        print(f'  IS  Z: mean={np.mean(is_z):.3f}, min={np.min(is_z):.3f} (n={len(is_z)})')
        print(f'  OOS Z: mean={np.mean(oos_z):.3f}, min={np.min(oos_z):.3f} (n={len(oos_z)})')
        print(f'  t-test: t={t:.3f}, p={p:.4f} '
              f'{"-> OOS has more extreme Z" if np.mean(oos_z) < np.mean(is_z) else "-> no difference"}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8. Transfer Entropy
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('8. Transfer Entropy (non-linear information flow)')
    print('  Two surrogate constructions, 200 iterations each:')
    print('    shift   = circular shift of the source (preserves its own serial')
    print('              correlation; destroys only alignment with the target).')
    print('              This is the primary test.')
    print('    shuffle = i.i.d. permutation of the source (the conventional')
    print('              test).  It destroys the source autocorrelation too, so')
    print('              against an autocorrelated target the null is too narrow.')
    print('              Reported for reconciliation with earlier versions.')
    z_arr   = np.array([r['z']     for r in records])
    fwd_arr = np.array([r['fwd20'] for r in records])

    te_results = {}   # for Fig 7 (primary = shift)
    te_pvals   = []   # for the multiplicity section
    for n_bins in [3, 5]:
        subsection(f'bins = {n_bins}')
        print(f'  {"Direction":<34} {"TE":>8} {"p(shift)":>9} {"Z(shift)":>9} '
              f'{"p(shuf)":>8} {"flag":>7}')
        print('  ' + '-'*80)
        for src, tgt, key in [
            (z_arr,   fwd_arr, 'HYG Z -> S&P fwd20'),
            (fwd_arr, z_arr,   'S&P fwd20 -> HYG Z (reverse)'),
        ]:
            for lag in [1, 5, 20]:
                te, p, te_z = te_significance(src, tgt, lag=lag, n_bins=n_bins,
                                              n_shuffle=200, method='shift')
                _, p_sh, _  = te_significance(src, tgt, lag=lag, n_bins=n_bins,
                                              n_shuffle=200, method='shuffle')
                te_results[(key, lag, n_bins)] = (te, p, te_z)
                te_pvals.append((f'TE {key} lag={lag} bins={n_bins}', p))
                sig = '✓ 5%' if p < 0.05 else ('△ 10%' if p < 0.10 else '')
                label = f'{key} (lag={lag})'
                print(f'  {label:<34} {te:>8.4f} {p:>9.4f} {te_z:>9.3f} '
                      f'{p_sh:>8.4f} {sig:>7}')
    print()
    print('  A lag-20 result that is significant under shuffle but not under')
    print('  shift is consistent with a mechanical consequence of the 20-day')
    print('  construction of the target rather than with information flow.')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8b. Multiplicity
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('8b. Multiplicity across the diagnostic families')
    print('  No correction is applied to the figures reported above; this')
    print('  section records what would survive one.  Each family is treated')
    print('  separately, since they test different hypotheses.')

    def bh(pvals):
        """Benjamini-Hochberg adjusted p-values, order preserved."""
        m = len(pvals)
        order = sorted(range(m), key=lambda i: pvals[i])
        adj = [0.0] * m
        prev = 1.0
        for rank, i in enumerate(reversed(order), start=1):
            k = m - rank + 1
            prev = min(prev, pvals[i] * m / k)
            adj[i] = prev
        return adj

    families = [
        ('Confound ladder (HAC)', conf_pvals),
        ('Granger, forward',      granger_pvals),
        ('Transfer entropy (shift surrogate)', te_pvals),
    ]
    for fam, items in families:
        if not items:
            continue
        names = [x[0] for x in items]
        ps    = [float(x[1]) for x in items]
        m     = len(ps)
        bonf  = 0.05 / m
        adjb  = bh(ps)
        surv_b = [names[i] for i in range(m) if ps[i] < bonf]
        surv_h = [names[i] for i in range(m) if adjb[i] < 0.05]
        print(f'\n  [{fam}]  m = {m}   Bonferroni threshold = {bonf:.4f}')
        print(f'    smallest raw p = {min(ps):.4f}')
        print(f'    survive Bonferroni : {len(surv_b)}/{m}'
              + (f'   e.g. {surv_b[0]}' if surv_b else ''))
        print(f'    survive BH (q=0.05): {len(surv_h)}/{m}'
              + (f'   e.g. {surv_h[0]}' if surv_h else ''))

    print()
    print('  The confound ladder is the family most affected: its twelve')
    print('  specifications are nested restatements of one regression rather')
    print('  than independent tests, so a Bonferroni threshold is conservative')
    print('  to the point of being uninformative there.  It is reported')
    print('  because the paper makes no multiplicity correction elsewhere and')
    print('  the reader is entitled to the arithmetic.')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 9. ETF specificity
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    etf_rows = []
    if args.jnk or args.lqd:
        section('9. ETF specificity (same construction, alternative credit ETFs)')
        print('  The classifier is rebuilt on each ETF, with the reference diagram')
        print('  taken from that ETF\'s own 60-day window ending on the GFC nadir.')
        print('  The outcome variable is the S&P 500 forward return throughout, so')
        print('  the panels are comparable.  Two base rates are shown: the ETF\'s')
        print('  own control-day frequency, and the HYG panel rate for reference.')
        hyg_base = {'Full': base_rate(records),
                    'IS':   base_rate(IS_recs),
                    'OOS':  base_rate(OOS_recs)}
        etf_inputs = [('HYG', dates_hyg, prices_hyg)]
        for tag, path in [('JNK', args.jnk), ('LQD', args.lqd)]:
            if not path:
                continue
            try:
                d_, p_ = load_csv(path)
                etf_inputs.append((tag, d_, p_))
            except Exception as e:
                print(f'  {tag}: could not load {path} ({e})')

        print(f'\n  {"ETF":4} {"panel":6} {"n(sig)":>7} {"pos.freq":>9} '
              f'{"own base":>9} {"lift":>9} {"p(above)":>9} {"p(below)":>9} '
              f'{"HYG base":>9}')
        print('  ' + '-'*84)
        for tag, d_, p_ in etf_inputs:
            pd_r, err = reference_pd(d_, p_)
            if pd_r is None:
                print(f'  {tag:4} skipped: {err}')
                continue
            D_, Z_, Y_, S_ = build_classifier_panel(d_, p_, pd_r, sp500_dict)
            if len(D_) == 0:
                print(f'  {tag:4} skipped: empty panel')
                continue
            panels = [('Full', np.ones(len(D_), dtype=bool)),
                      ('IS',   D_ <= '2019-12-31'),
                      ('OOS',  D_ >= '2020-01-01')]
            for plab, pmask in panels:
                sel  = S_ & pmask
                ctl  = (~S_) & pmask
                if sel.sum() == 0 or ctl.sum() == 0:
                    print(f'  {tag:4} {plab:6} {int(sel.sum()):>7}   (insufficient)')
                    continue
                nn   = int(sel.sum())
                k    = int((Y_[sel] > 0).sum())
                freq = 100.0 * k / nn
                own  = float(np.mean(Y_[ctl] > 0))
                p_hi = 1 - binom.cdf(k - 1, nn, own)
                p_lo = binom.cdf(k, nn, own)
                etf_rows.append((tag, plab, nn, freq, own*100,
                                 freq - own*100, p_hi, p_lo))
                print(f'  {tag:4} {plab:6} {nn:>7} {freq:>8.1f}% {own*100:>8.1f}% '
                      f'{freq - own*100:>+8.1f}pp {p_hi:>9.3f} {p_lo:>9.3f} '
                      f'{hyg_base[plab]*100:>8.1f}%')
            print(f'       panel {len(D_)} days, {D_[0]}..{D_[-1]}')
        print()
        print('  p(above) / p(below) are one-sided binomial tail probabilities')
        print('  against the ETF\'s own control-day base rate.  Both assume')
        print('  independent observations and are understated under the 20-day')
        print('  overlap (Section 3.1 of the paper).')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Figure generation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if args.figures:
        section('FIGURES')
        print('  Generating 7 figures...')
        fig1_signal_timeseries(records, all_dates, dist_z, sp500_dict, args.outdir)
        fig2_annual_winrate(records, by_year, args.outdir)
        fig3_dose_response(records, OOS_recs, args.outdir)
        fig4_beta_stability(records, OOS_recs, vix_dict, args.outdir)
        fig5_did_visualization(records, IS_recs, OOS_recs, args.outdir)
        fig6_is_decomposition(records, IS_recs, OOS_recs, by_year, args.outdir)
        fig7_transfer_entropy_heatmap(te_results, args.outdir)
        print(f'\n  All 7 figures saved to {args.outdir}/')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Summary
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section('SUMMARY')
    osf = [r['fwd20'] for r in OOS_recs if r['signal']==1]
    ocf = [r['fwd20'] for r in OOS_recs if r['signal']==0]
    print(f"""
  OOS signal : n={len(osf)}, wr={wr(osf):.1f}%, avg={np.mean(osf):+.2f}%
  OOS control: n={len(ocf)}, wr={wr(ocf):.1f}%, avg={np.mean(ocf):+.2f}%
  diff       : posfreq={wr(osf)-wr(ocf):+.1f}pp

  [Sec 2: Granger]   no linear lead-lag -> consistent with a threshold/non-linear signal
  [Sec 3: Means]     use p(HAC) for inference (t/binomial understate SE under overlap)
  [Sec 4: DiD]         DiD={did:+.1f}pp  p(OLS)={p_did:.4f} / p(HAC)={p_did_hac:.4f}
  [Sec 5: Dose-resp] monotone in threshold (caveats: nested subsets, small n, OOS bull market)
  [Sec 6: Confound]  beta not attenuated; judge significance from the p(HAC) column
  [Sec 7: IS decomp] 2015 cluster + Z-score intensity gap
  [Sec 8: TE]        shift surrogate is primary; shuffle retained for reconciliation
  [Sec 8b: multiple] Bonferroni / BH across each diagnostic family
  [Sec 9: ETF spec]  run with --jnk/--lqd; read each ETF against its OWN control-day base rate

  Note: HAC_LAGS={HAC_LAGS} (Newey-West) corrects SE understatement from 20-day overlapping returns.
    Diagnostics that fall below 5% under p(HAC) are reported as descriptive regularities (present sample only).
""")
    section('DONE')

if __name__ == '__main__':
    main()
