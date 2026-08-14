# Data Sourcing Instructions

The reproduction script requires three CSV files.
**Data files are not included in this repository** due to licensing
restrictions. Download them yourself using the instructions below.

---

## Required Files

| Filename | Ticker | Source | Start date used |
|----------|--------|--------|------------------|
| `longterm_hyg.csv`   | HYG   | Yahoo Finance | 2007-04-11 |
| `longterm_sp500.csv` | ^GSPC | Yahoo Finance | 1990-01-02 |
| `longterm_vix.csv`   | ^VIX  | Yahoo Finance | 2007-01-03 |
| `longterm_jnk.csv`   | JNK   | Yahoo Finance | 2007-12-04 | *(optional — Section 9 only)* |
| `longterm_lqd.csv`   | LQD   | Yahoo Finance | 2002-07-30 | *(optional — Section 9 only)* |

All three files are required — none of the diagnostics in this paper
can be computed without VIX (used as a control in the confound
adjustment of Section 5 and in the macroeconomic profile of
Section 9.2). This differs from the companion *Memory of Crisis*
reproduction bundle, which allows a VIX-free core run; the diagnostics
here do not.

All files must be placed in the `data/` directory.

---

## Download with yfinance (Python)

```python
import yfinance as yf

tickers = {
    'longterm_hyg.csv':   ('HYG',   '2007-01-01'),
    'longterm_sp500.csv': ('^GSPC', '1990-01-01'),
    'longterm_vix.csv':   ('^VIX',  '2007-01-01'),
    # optional, for the ETF-specificity section (9):
    'longterm_jnk.csv':   ('JNK',   '2007-01-01'),
    'longterm_lqd.csv':   ('LQD',   '2002-01-01'),
}

for filename, (ticker, start) in tickers.items():
    df = yf.download(ticker, start=start, auto_adjust=True)
    df = df[['Close']].dropna()
    df.index.name = 'trade_date'
    df.index = df.index.strftime('%Y-%m-%d')
    df.columns = ['close']
    df.to_csv(f'data/{filename}')
    print(f'Saved {filename}: {len(df)} rows')
```

---

## Expected CSV Format

```
trade_date,close
2007-04-11,31.8551
2007-04-12,31.8765
2007-04-13,31.9135
...
```

- Header row required: `trade_date,close`
- Date format: `YYYY-MM-DD`
- No missing dates acceptable (weekends/holidays are naturally absent)

---

## Coverage Requirements

| File | Minimum start | Reason |
|------|---------------|--------|
| `longterm_hyg.csv`   | 2007-04-11 | HYG inception (60-day Takens warm-up + 252-day Z-score warm-up) |
| `longterm_sp500.csv` | 1990-01-02 or earlier | Outcome-variable series; longer history improves the reliability of the overlap-autocorrelation estimate in Section 3.1 of the paper |
| `longterm_vix.csv`   | 2007-01-03 or earlier | Control variable in the confound-adjustment diagnostics (Section 5) |

The effective analysis panel is **April 2007 – May 2026, 4,474 trading
days** with a complete 20-day forward return after the 60-day Takens
warm-up and the 252-day rolling Z-score warm-up (see Section 2.2 /
Table 1 of the paper for the exact sample-frame accounting).

---

## A Note on the Overlap-Robust Diagnostics

Sections 5 and 6 of the paper (sequential confound adjustment and
difference-in-differences) report both conventional (OLS) and
Newey–West (HAC) standard errors, with a lag of `2*(20-1) = 38` trading
days — informed by the 20-day construction of the forward-return
outcome variable (see Section 3.1 of the paper). No additional data
beyond the three files above is required to reproduce this; the HAC
correction is applied entirely within `hyg_regime_diagnostics.py`
via `statsmodels`.

---

## A Note on the Base Rate (v4)

The conditional frequencies in the paper are assessed against the
**measured** frequency of positive 20-day forward S&P 500 returns on
control days, not against 50%. That base rate is computed from the
same three files — no additional data is needed — and the script
prints it explicitly:

```
measured unconditional pos. freq.: Full 66.7%  IS 66.2%  OOS 67.6%
```

with the control-day rates (66.5% / 66.1% / 67.1%) shown in the
`base` column of the baseline table. If your download differs from
the panel described above, these figures will shift slightly and the
reported tail probabilities will shift with them. Check the printed
base rate against the paper before comparing *p*-values: a base rate
that differs by more than a few tenths of a percentage point means
your panel is not the paper's panel.
