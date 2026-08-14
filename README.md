# Base Rates and Autocorrelation-Robust Diagnostics for a Topological Regime Classifier — Reproduction Code

**Base Rates and Autocorrelation-Robust Diagnostics for a Topological Regime Classifier:**
*Reassessing Quasi-Experimental Designs under Overlapping Return Horizons and a Drifting Outcome*

Keishi Takagi, Independent Researcher

> *Working Paper — Empirical Evidence from HYG Daily Data, 2007–2026*

---

## Overview

This repository contains the full reproduction script for all tables
and figures in the paper.

The paper subjects the HYG GFC Wasserstein topological regime
classifier introduced in Takagi (2026a) — *Topological Memory of the
GFC Nadir* — to five quasi-experimental diagnostic designs from the
empirical econometrics literature: sequential confound adjustment,
difference-in-differences, dose-response examination, Granger
causality testing, and Transfer Entropy.

Throughout, the paper treats two additional methodological questions
as first-order concerns.

**(1) Standard errors.** The 20-day forward return used as the outcome
variable is constructed from overlapping daily windows, which induces
first-order autocorrelation of approximately **0.93** in the residuals
of any regression that uses it, decaying to near zero only close to
the 20-day construction horizon. Ordinary least squares standard
errors do not account for this structure. The script reports **both**
the conventional and the Newey–West (HAC, 38-lag)
autocorrelation-robust version of every regression-based diagnostic.

**(2) The reference value — new in v4.** The frequency of positive
outcomes has conventionally been assessed against a **50%** reference,
including in Takagi (2026a). That is not the unconditional frequency
of the outcome variable. Over this panel the unconditional frequency
of positive 20-day forward S&P 500 returns is **66.7%**, and the
frequency on control days (`z > -1.5`) is **66.5%** overall, 66.1%
in-sample and 67.1% out-of-sample. Equities drift upward over a 20-day
horizon, so a coin-flip reference is not the applicable null. The
script now reports every conditional frequency against the **measured
control-day base rate**, alongside the 50%-referenced figure for
reconciliation with earlier versions.

The two corrections act on different things and are not substitutes.
A HAC estimator widens the interval around a fixed point estimate.
Correcting the reference **moves the point estimate**. Statistics fall
into two classes:

| Class | Examples | Affected by |
|---|---|---|
| **Contrast** | regression coefficient on the classifier dummy, DiD interaction, classified-vs-control comparison | standard error only (base rate cancels) |
| **Level** | conditional frequency of positive outcomes, conditional mean | **both** |

**Headline result — contrast statistics.** Point estimates of the
classifier's association with the 20-day forward S&P 500 return are
stable in sign and magnitude across sequential covariate adjustments.
Under conventional standard errors, several coefficients are
significant at the 5% level. Under Newey–West standard errors, the
out-of-sample confound-adjusted coefficient falls to the 10% level
under all six specifications considered, and the full-sample and
difference-in-differences coefficients are no longer distinguishable
from zero at conventional levels. For these statistics the precision
of the estimate, not its central tendency, is what is sensitive to the
treatment of serial correlation.

**Headline result — level statistics.** Stability does **not** carry
over. Against the measured control-day base rate rather than 50%, the
out-of-sample conditional frequency remains significantly elevated
(90.0%, a lift of +22.9pp, *p* = 0.004), the full-sample figure is
significant (*p* = 0.020), and the **in-sample figure — 70.4% against
a base rate of 66.1% — is not distinguishable from the unconditional
distribution at conventional levels** (*p* = 0.307). The two cells are
separated less by the raw frequency than by whether the departure from
the base rate is detectable in 54 and 30 classified days.

**Correction to Takagi (2026a).** That paper describes 50% as "the
unconditional reference" for its binomial test, and tests its
conditional mean against zero. Neither is the unconditional value.
Against the applicable references the reported 77.4% frequency is a
lift of **+10.9pp** (one-sided binomial *p* = **0.020**) rather than
+27.4pp (*p* = 2.37e-7), and the +2.98% conditional mean is an excess
of **+2.12pp** over the control-day mean of +0.86% rather than +2.98pp
over zero. These are measured on the panel that reproduces Takagi
(2026a) cell for cell, so the correction uses that paper's own
sample. Direction and sign are unchanged; magnitudes and nominal
significance are substantially smaller. See Section 1.1 of the paper.

**The exercise is descriptive throughout.** It characterises the
autocorrelation-robust properties of a topological classifier's
conditional return distribution under five quasi-experimental designs;
it does not constitute a forecast, a causal claim, an investment
recommendation, or a trading rule. See the Limitations section and
Section 11 of the paper.

---

## Repository Structure

```
.
├── hyg_regime_diagnostics.py  # Main reproduction script
├── requirements.txt                 # Python dependencies
├── README.md
├── README_data.md                   # Data sourcing instructions
├── LICENSE
├── .gitignore
└── results/                         # Output directory (created on run)
    ├── fig1_signal_timeseries.png   # supplementary (not embedded in the paper)
    ├── fig2_annual_winrate.png      # → Figure 2 (annual decomposition)
    ├── fig3_dose_response.png       # → Figure 2 (dose-response, Section 8)
    ├── fig4_beta_stability.png      # → Figure 1 (confound adjustment, HAC CIs)
    ├── fig5_did.png                  # → Figure 2 (difference-in-differences)
    ├── fig6_is_decomposition.png    # → Figure 3 (IS/OOS decomposition)
    └── fig7_transfer_entropy.png    # → Figure 4 (Transfer Entropy heatmap)
```

> **Note on figure numbering.** The output filenames retain the
> script's internal generation order (`fig1`…`fig7`); the paper embeds
> six of the seven (all but `fig1_signal_timeseries.png`, which is a
> supplementary overview figure). See the Outputs section below for
> the precise mapping to in-paper figure numbers and sections.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The script requires `statsmodels` (OLS, Newey–West/HAC covariance,
Granger causality, ADF), `ripser` (Vietoris–Rips persistent homology),
and `pot` (Python Optimal Transport, for the Wasserstein-2 distance).
All are listed in `requirements.txt`.

### 2. Prepare data

Place the three required CSV files in `data/` — see `README_data.md`
for sourcing (Yahoo Finance) and the exact expected format.

### 3. Run — analysis only (no figures, faster)

```bash
python hyg_regime_diagnostics.py \
    --hyg   data/longterm_hyg.csv \
    --sp500 data/longterm_sp500.csv \
    --vix   data/longterm_vix.csv
```

### 4. Run — full set (all tables + all 7 figures)

```bash
python hyg_regime_diagnostics.py \
    --hyg     data/longterm_hyg.csv \
    --sp500   data/longterm_sp500.csv \
    --vix     data/longterm_vix.csv \
    --figures \
    --outdir  results/
```

The full run takes on the order of a minute: persistent homology is
computed for each daily 60-day window (4,474 windows), followed by the
regression-based diagnostics, Granger tests, and a 200-iteration
shuffle test for Transfer Entropy at each of 12 lag/direction/bin
configurations.

### Optional: ETF specificity (Section 9)

Pass `--jnk` and/or `--lqd` to rebuild the classifier on the
alternative credit ETFs. The reference diagram is taken from each
ETF's *own* 60-day window ending 2009-03-09, and the outcome variable
remains the S&P 500 forward return, so the panels are comparable. The
section is skipped silently when neither flag is given.

```bash
python hyg_regime_diagnostics.py \\
    --hyg   data/longterm_hyg.csv \\
    --sp500 data/longterm_sp500.csv \\
    --vix   data/longterm_vix.csv \\
    --jnk   data/longterm_jnk.csv \\
    --lqd   data/longterm_lqd.csv
```

Each ETF is reported against **its own** control-day base rate, since
the panels differ (JNK starts 2010-03, LQD 2004-10); the HYG panel
rate is shown alongside for reference. Both one-sided tails are
printed, because the notable JNK result is a departure *below* the
base rate: the JNK out-of-sample frequency of 22.7% sits 45.5 points
under it (p < 0.001), while the JNK in-sample frequency of 88.2% sits
19.9 points above it (p = 0.007) and HYG in-sample sits only 4.3
points above its own (p = 0.307). The ETF-specificity check does not
point the same way in all three panels and should be read panel by
panel. On the corrected panel this section reproduces Takagi (2026a,
Table 12) in all nine cells.

### Optional: sensitivity to the HAC lag choice

```bash
python hyg_regime_diagnostics.py \
    --hyg data/longterm_hyg.csv --sp500 data/longterm_sp500.csv --vix data/longterm_vix.csv \
    --hac-lags 20
```

`--hac-lags` defaults to 38 (`= 2 * (20 - 1)`, the rule-of-thumb
truncation for a 20-day-overlapping outcome variable). Override it to
check the sensitivity of the reported `p(HAC)` values to the lag
choice; this does not change the point estimates, only their standard
errors.

---

## Outputs

### Tables (printed to stdout)

| # | Content |
|---|---------|
| 0 | Baseline conditional distribution: Full / IS (≤2019) / OOS (2020–) — reproduces the source classifier's headline numbers |
| 1 | Augmented Dickey–Fuller stationarity tests (Z-score, forward return, classifier dummy) |
| 2 | Granger causality on the classifier dummy, forward and reverse, lags 1–10 |
| 3 | Mean comparison + VIX/drawdown-adjusted regression, by IS / OOS / Full, with **p(OLS)** and **p(HAC)** |
| 4 | Difference-in-differences: 2×2 table and interaction regression, **p(OLS)** and **p(HAC)** |
| 5 | Threshold (dose-response) sensitivity, six thresholds, reported for **three panels** (OOS / Full / IS), each against its own control-day base rate and against the threshold-specific complement (`z > thr`) |
| 6 | Sequential confound-adjustment ladder (6 specifications), OOS and Full, **p(OLS)** and **p(HAC)** |
| 7 | IS/OOS decomposition: annual breakdown, macro profile, classifier-intensity (Z-score) comparison |
| 8 | Transfer Entropy: HYG Z-score ↔ 20-day forward return, lags 1/5/20, bins 3 and 5, 200-iteration surrogate test — **circular-shift surrogate is primary**, i.i.d. shuffle reported for reconciliation |
| 8b | Multiplicity: Bonferroni and Benjamini–Hochberg across each diagnostic family |
| 9 | ETF specificity: the classifier rebuilt on JNK and LQD, each against its **own** control-day base rate (requires `--jnk` / `--lqd`) |

**Table 0 (baseline; reproduces the source classifier).**

| Period | n (total) | n (classified) | Mean | Pos. freq. | Base rate | Lift | *p* vs base | *p* vs 50% |
|--------|-----------|-----------------|------|-----------|-----------|------|-------------|------------|
| Full           | 4,474 | 84 | +2.98% | 77.4% | 66.5% | +10.9pp | **0.020** | 2.37e-7 |
| IS (≤2019)     | 2,893 | 54 | +2.76% | 70.4% | 66.1% | +4.3pp  | **0.307** | 1.92e-3 |
| OOS (2020–)    | 1,581 | 30 | +3.38% | 90.0% | 67.1% | +22.9pp | **0.004** | 4.22e-6 |

*Base rate* is the measured control-day (`z > -1.5`) frequency of
positive outcomes for that sub-period; the full-sample figure pools
the two. The `p vs 50%` column is what earlier versions reported and
is retained only for reconciliation. Control-day **means** over the
same panel are +0.86% (Full), +0.73% (IS), +1.10% (OOS) — the Mean
column should be read against these, not against zero.

**Table 6 (headline; sequential confound adjustment, OLS vs HAC).**

| Spec. | OOS β | *p*(OLS) | *p*(HAC) | Full β | *p*(OLS) | *p*(HAC) |
|-------|------:|---------:|---------:|-------:|---------:|---------:|
| (1) Signal only            | +2.278% | 0.0154 | 0.0831 | +1.003% | 0.0358 | 0.1320 |
| (2) + VIX                  | +2.237% | 0.0132 | 0.0771 | +0.913% | 0.0493 | 0.1534 |
| (3) + VIX + DD              | +2.230% | 0.0134 | 0.0752 | +0.916% | 0.0483 | 0.1529 |
| (4) + SMA                  | +2.479% | 0.0060 | 0.0534 | +0.923% | 0.0467 | 0.1450 |
| (5) + Momentum             | +2.474% | 0.0060 | 0.0582 | +0.952% | 0.0398 | 0.1361 |
| (6) + Realised volatility  | +2.586% | 0.0040 | 0.0581 | +0.963% | 0.0377 | 0.1364 |

Every OOS specification is significant at the 10% but not the 5%
level under HAC; no Full specification is distinguishable from zero
under HAC. The point estimate does not attenuate as controls are
added under either sample.

### Figures (saved to `--outdir`, default: `results/`)

| Output file | In-paper Figure | Section | Content |
|-------------|-----------------|---------|---------|
| `fig4_beta_stability.png`     | Figure 1 | 5 (Confound Adjustment) | β with Newey–West 95% CIs, sequential specifications, OOS & Full |
| `fig5_did.png`                 | Figure 2 | 6 (DiD) | 2×2 IS/OOS × classified/control panel |
| `fig3_dose_response.png`       | Figure 3 | 7 (Dose-Response) | Frequency of positive outcomes by threshold severity |
| `fig7_transfer_entropy.png`    | Figure 4 | 8 (Transfer Entropy) | TE values and shuffle-test *p*-values, by lag and direction |
| `fig2_annual_winrate.png`      | Figure 5 | 9.1 (Annual Decomposition) | Annual frequency of positive outcomes and classified-day count |
| `fig6_is_decomposition.png`    | Figure 6 | 9.3 (IS/OOS Summary) | Annual scatter, Z-score boxplots, 2015-cluster comparison |
| `fig1_signal_timeseries.png`   | — (supplementary) | — | Full Z-score time series with classified days marked |

---

## Methodology

```
Persistence Diagram (PD) construction — identical to the source paper:
  1. For each trading day t, take the 60-day closing-price window
     p[t-59 .. t]  (W = 60).
  2. Apply Takens delay embedding (D = 3, τ = 3), yielding
     W - (D-1)τ = 54 points in R³.
  3. Centre the point cloud coordinate-wise and scale by the
     full-cloud scalar standard deviation.
  4. Compute Vietoris–Rips persistent homology via Ripser, with the
     filtration threshold at the 70th percentile of pairwise distances.
  5. Retain the H1 (loop) persistence diagram.

Reference PD (fixed): the H1 diagram from the 60-day window ending on
2009-03-09 (S&P 500 GFC nadir), 5 H1 loops.

Classifier:
  1. w_t = W2(PD_t, PD_ref) via Python Optimal Transport (POT).
  2. z_t = (w_t - mean_252(w)) / std_252(w)  (252-day rolling Z-score).
  3. Classified day: s_t = 1 iff z_t <= -1.5.

Outcome variable:
  r_{t,20} = p^SPX_{t+20} / p^SPX_t - 1  (20-day forward S&P 500 return).

Autocorrelation-robust correction (the central addition of this paper):
  r_{t,20} and r_{t+1,20} share 19 of 20 constituent daily returns.
  Empirically (this sample): first-order autocorrelation ≈ 0.93,
  decaying to ≈0 near lag 19-20. Newey-West (HAC) standard errors,
  lag = 2*(20-1) = 38, are reported alongside conventional OLS
  standard errors for every regression coefficient in Tables 3, 4, 6.

Diagnostics:
  - Granger causality on the classifier dummy, lags 1-10, both directions
  - OLS regression of r_{t,20} on s_t with sequential covariate
    adjustment (VIX, drawdown, SMA50/200, 20-day momentum,
    20-day realised volatility), OLS and HAC standard errors
  - Difference-in-differences: s_t x 1{OOS} interaction, OLS and HAC
  - Dose-response: pos. freq. at 6 thresholds (nested, non-independent
    subsets -- see Limitations)
  - Transfer Entropy (Schreiber 2000): HYG Z-score <-> r_{t,20},
    lags 1/5/20, bins 3 and 5, 200-iteration shuffle test
```

---

## Diagnostics Summary

Five quasi-experimental designs are applied, with an
autocorrelation-robust reassessment of the two regression-based
designs among them (paper Sections 4-9):

1. **Granger causality** — Not significant at any lag from 1 to 10 in
   either direction. Consistent with — not evidence against — a
   threshold-based regime indicator rather than a linear predictor.

2. **Sequential confound adjustment** — The signal coefficient does
   not attenuate as VIX, drawdown, SMA50/200, momentum, and realised
   volatility are added sequentially, in either the OOS or Full sample.
   Under HAC, the OOS coefficient is significant at the 10% but not
   the 5% level under all six specifications; the Full-sample
   coefficient is not distinguishable from zero under HAC; the
   IS-sample coefficient (+0.118%) is small and non-significant under
   either standard error.

3. **Difference-in-differences** — DiD = +22.2pp descriptively.
   Regression interaction: p(OLS) = 0.0346, **p(HAC) = 0.147** — not
   significant once the overlap correction is applied.

4. **Dose-response** — OOS frequency of positive outcomes rises
   monotonically from 71.9% (z ≤ −0.5, n=594) to 100.0% (z ≤ −2.0,
   n=7). Against the OOS control-day base rate of 67.1% rather than
   50%, the corresponding tail probabilities are 0.007, 0.0001, 0.004,
   **0.061, 0.136, 0.136** — the three strictest OOS thresholds, which
   produce the visually striking 100.0% figures, no longer reach the
   5% level, and the strongest evidence sits at z ≤ −1.0 (n=204)
   rather than at the base threshold. The monotonicity of the
   frequency is unaffected by the choice of reference; the
   distribution of statistical support across thresholds is.
   Reported with three explicit caveats: the six thresholds are
   nested (non-independent) subsets; the three strictest thresholds
   have n = 5–7; and the OOS control-day frequency is itself elevated
   (67.1%) over the 2020–2026 bull market.

   **Monotonicity is a property of the OOS panel only.** v4 reports
   the same ladder for the Full and IS panels; neither is monotone,
   and in both the loosest threshold lies *below* the applicable base
   rate:

   | Panel | z≤−0.5 | z≤−1.0 | z≤−1.5 | z≤−2.0 | base | monotone |
   |---|---|---|---|---|---|---|
   | OOS  | 71.9% | 79.4% | 90.0% | 100.0% | 67.1% | ✅ |
   | Full | **64.8%** | 72.0% | 77.4% | 77.3% | 66.5% | ❌ |
   | IS   | **61.0%** | 68.3% | 70.4% | **66.7%** | 66.1% | ❌ |

   Comparing selected against non-selected days directly: at z ≤ −0.5
   the classified days return a positive outcome 64.8% of the time
   against 67.9% on unclassified days (Full), and 61.0% against 69.5%
   (IS). None of this is visible under a 50% reference, against which
   every cell in all three panels is significant. See paper
   Table 6 (Panels A–C) and Section 7.

5. **Transfer Entropy** — under the conventional i.i.d. shuffle test,
   HYG Z-score → 20-day forward return is significant at lag 5 and
   lag 20 in both bin configurations. **Under the circular-shift
   surrogate, which preserves the source's own serial correlation, none
   of the twelve tests is significant.** The shuffle result is therefore
   read as a mechanical consequence of the overlapping-window
   construction of the target rather than as evidence of information
   flow; see "The Transfer Entropy surrogate" above.

6. **IS/OOS decomposition** (descriptive, not a hypothesis test) — The
   2015 cluster (18 of 54 IS classified days, 55.6% pos. freq.) and a
   difference in mean classifier intensity at activation (IS z̄ =
   −1.834 vs OOS z̄ = −2.003, t = 1.392, p = 0.171 — not significant on
   the corrected panel) are reported as two
   non-exclusive descriptive accounts of the IS/OOS asymmetry.

No formal multiple-comparison correction is applied across the five
diagnostics, the six thresholds, the ten Granger lags, or the twelve
Transfer Entropy configurations in the reported figures; Section 8b of
the script records what would survive Bonferroni and Benjamini–Hochberg
correction within each family, so the arithmetic is available to the
reader. The exercise is exploratory and
descriptive (see Limitations).

---

## The Transfer Entropy surrogate

Section 8 reports two surrogate constructions:

| Surrogate | What it destroys | What it preserves |
|---|---|---|
| `shuffle` (conventional) | cross-series alignment **and** the source's own serial correlation | nothing about the source |
| `shift` (primary) | cross-series alignment only | the source's full autocorrelation |

The target `r_{t,20}` is autocorrelated by construction (≈0.93 at lag 1).
Against such a target, an i.i.d.-shuffled source is a much weaker
competitor than the real one, so the shuffle null is too narrow and the
test is anti-conservative — most acutely at lag = 20, where the lag
coincides with the construction horizon of the target.

A result significant under `shuffle` but not under `shift` should be
read as a mechanical consequence of the overlapping-window construction
rather than as information flow. **On the present panel this describes
every Transfer Entropy result in the table**, so the section is reported
as a null finding under the primary surrogate.

## Notes on v5

1. **Warm-up correction (v5) — this changes every number.** Versions up
   to v4 started the Wasserstein distance series at `W + ROLL = 312`
   trading days and then applied a further 252-day rolling window to
   compute the Z-score, imposing the 252-day warm-up twice. The Takens
   embedding needs only `W = 60` days; the correct start is `W`, after
   which the first usable Z-score falls at day 312. The defect
   discarded the first 252 trading days of valid classifier output
   (2008-07-07 to 2009-07-07) — a window containing the reference date
   2009-03-09 itself. The panel goes from 4,222 to **4,474** and the
   classified-day count from 78 to **84**. The corrected run reproduces
   Takagi (2026a) exactly, including all nine cells of its ETF table;
   the earlier version did not, and that mismatch is what surfaced the
   defect. If you are comparing against older output, expect every
   figure to move.

2. **Choice of reference does not drive the results.** Section 5
   reports each threshold against two references: the fixed
   control-day rate (`z > -1.5`, the same reference as Table 0) and
   the threshold-specific complement (`z > thr`, which moves with the
   threshold and is the internally consistent choice for a
   dose-response curve). No tail probability in the table differs by
   more than 0.005 between the two.

3. **Figure 3 reference lines.** The dose-response figure plots both
   the OOS and the Full-period series, and draws the control-day base
   rate for each. They differ by well under a percentage point, so the
   two lines effectively coincide; the legend records both values.

---

## Limitations

The framework is descriptive throughout. Key limitations (paper
Section 11):

- **Overlapping-return autocorrelation** — Addressed directly for the
  regression-based diagnostics via Newey–West standard errors. The
  *t*-test, binomial-test, and per-year descriptive statistics
  elsewhere in the paper are computed against the measured base rate
  (v4) but are **not** corrected for autocorrelation, and are reported
  as descriptive summaries rather than confirmatory tests. As an
  order-of-magnitude check on what survives when the 78 classified
  days are not treated as 78 independent draws: 9 of the 12 activation
  years exceed the pooled base rate, a sign test *p* of 0.073.

- **Base rate of the outcome variable** — Conditional frequencies are
  assessed against the control-day frequency (`z > -1.5`) rather than
  the fully unconditional frequency; the two differ by ≤0.2pp in every
  sub-period. The base rate is itself estimated and its sampling
  uncertainty is not propagated into the reported tail probabilities
  (control-day n = 2,593 and 1,551, against classified-day n = 48 and
  30, so the omission is small but real).

- **Transfer Entropy at lag=20** — Coincides with the 20-day
  construction of the outcome variable; cannot, on the shuffle test
  alone, be distinguished from a mechanical consequence of that
  construction. Reported as an open question.

- **Sample size** — 30 classified days in the OOS subsample; as few as
  5–7 in the strictest dose-response thresholds.

- **Single asset** — Applied to HYG only. An ETF-specificity
  comparison against JNK and LQD is reported in the source classifier
  paper (Takagi 2026a, Section 5.5); a fuller mechanism investigation
  of the cross-ETF differences is left to future work.

- **Quasi-experimental, not causal** — DiD and OLS-with-controls adjust
  for observable covariates only; unobserved confounders cannot be
  excluded.

- **No multiplicity correction** — See Diagnostics Summary above.

- **No causal claim / no tradability claim** — Temporal precedence is
  not causation. The paper does not propose, simulate, or recommend any
  exposure rule conditioned on the classifier, and makes no claim about
  realised performance after implementation frictions.

---

## Requirements

- Python ≥ 3.8
- numpy, scipy (required)
- statsmodels (required — OLS, Newey–West/HAC, Granger, ADF)
- ripser, pot (required — persistent homology and optimal transport)
- matplotlib (optional — only needed for `--figures`)

See `requirements.txt`.

---

## Citation

```bibtex
@misc{takagi2026diagnostics,
  author       = {Takagi, Keishi},
  title        = {Autocorrelation-Robust Diagnostics for a Topological
                  Regime Classifier: Reassessing Quasi-Experimental
                  Designs under Overlapping Return Horizons},
  year         = {2026},
  howpublished = {Working Paper}
}
```

---

## Related Work

- **Takagi (2026a)** (SSRN 7057698): *Topological Memory of the GFC
  Nadir: A Regime Classifier for HYG Based on Wasserstein Proximity to
  Crisis Persistence Diagrams* — introduces the classifier
  reassessed in this paper.

- **Takagi (2026b)** (SSRN 6883958): *A Sequential Regime Framework for
  U.S. Equity Drawdown Periods: Integration of Two Empirical
  Classifications and Distribution-Free Validation via Conformal
  Prediction* — a related regime-classification study using split
  conformal prediction as a distribution-free alternative to the
  parametric (Newey–West/HAC) approach to inference used here; see
  Section 3.1 of the paper for the contrast between the two routes.

---

## License

MIT License — see `LICENSE`.

---

## Disclaimer

This code and paper are provided for research purposes only. The
findings are descriptive and quasi-experimental characterisations of a
topological classifier's historical conditional return distribution,
not forecasts of future returns or recommendations. Nothing in this
repository constitutes financial advice. Past empirical regularities
described herein do not guarantee, and should not be construed as
implying, future results.
