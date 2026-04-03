"""
Empirical Distributed Lag Model: Crude Oil → Retail Gasoline
=============================================================
Downloads 2 years of weekly data from EIA public XLS endpoints (no API key):
  - US retail gasoline price: EMM_EPMR_PTE_NUS_DPG ($/gal, weekly)
  - WTI spot crude:           RWTC ($/bbl, daily → resampled weekly)

Fits:  pump_t = α + Σ_{k=0}^{8} β_k × (crude_{t-k}/42) + ε_t

No AR term: the model must be applicable to forward-looking forecast dates
where future pump prices are unknown (pure distributed lag on crude).

Collinearity check: if max off-diagonal correlation in lag matrix > 0.95,
switches to Polynomial Distributed Lag (Almon, degree 2) to stabilise
individual β_k while preserving the total pass-through estimate.

Asymmetry test: runs a single-indicator "rockets-and-feathers" test;
records the result but keeps the symmetric spec unless clearly significant
(p < 0.05) — too few observations to run full asymmetric DL reliably.

Outputs data/lag-model.json consumed by oil.html.
"""

import json
import math
import time
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUT_FILE = Path(__file__).parents[2] / "data" / "lag-model.json"
HEADERS  = {"User-Agent": "oil-lag-model/1.0 (academic/personal use)"}
N_LAGS   = 8          # weeks: captures ~85-90% of pass-through per literature
PDL_THRESH = 0.95     # max off-diagonal correlation before switching to PDL
PDL_DEGREE = 2        # Almon polynomial degree


# ---------------------------------------------------------------------------
# Data fetching (with retry + extended timeout for flaky external endpoints)
# ---------------------------------------------------------------------------

def fetch_with_retry(url: str, retries: int = 4, base_timeout: int = 60) -> bytes:
    """GET with exponential backoff. Raises on final failure."""
    for attempt in range(retries):
        timeout = base_timeout * (2 ** attempt)   # 60 → 120 → 240 → 480 s
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (2 ** attempt)          # 5 → 10 → 20 s
                print(f"  Attempt {attempt + 1} failed ({e}), retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise


def fetch_eia_gasoline(lookback_years: int = 3) -> pd.Series:
    """
    Fetch US weekly retail regular gasoline price from EIA's CSV LeafHandler.
    Falls back to the XLS endpoint if the CSV fails.
    Returns a pd.Series indexed by date.
    """
    # Primary: EIA CSV (no Excel library needed)
    csv_url = (
        "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx"
        "?n=PET&s=EMM_EPMR_PTE_NUS_DPG&f=W"
    )
    try:
        content = fetch_with_retry(csv_url)
        # EIA CSV: first 2 rows are metadata, row 3 is header, data follows
        from io import StringIO
        text = content.decode("utf-8", errors="replace")
        df = pd.read_csv(StringIO(text), skiprows=2, header=0)
        df.columns = ["date", "price"] + list(df.columns[2:])
        df = df[["date", "price"]].dropna()
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["price"].astype(float)
        s.index.name = "date"
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=lookback_years)
        s = s[s.index >= cutoff]
        if len(s) > 10:
            return s
        raise ValueError("Too few rows from CSV")
    except Exception as e:
        print(f"  CSV attempt failed ({e}), trying XLS...")

    # Fallback: XLS endpoint
    xls_url = "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGw.xls"
    content = fetch_with_retry(xls_url)
    for sheet in ("Data 1", 0):
        try:
            df = pd.read_excel(BytesIO(content), sheet_name=sheet, skiprows=2, header=0)
            df.columns = ["date", "price"] + list(df.columns[2:])
            df = df[["date", "price"]].dropna()
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["price"].astype(float)
            s.index.name = "date"
            cutoff = pd.Timestamp.today() - pd.DateOffset(years=lookback_years)
            return s[s.index >= cutoff]
        except Exception:
            continue
    raise ValueError("Could not fetch EIA gasoline price data via CSV or XLS")


def fetch_crude_weekly(lookback_years: int = 3) -> pd.Series:
    """
    Fetch weekly WTI crude price via yfinance (CL=F front-month continuous).
    Falls back to EIA RWTC XLS if yfinance fails.
    Returns a pd.Series indexed by date, resampled to weekly (Monday).
    """
    import yfinance as yf
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=lookback_years)
    try:
        raw = yf.download("CL=F", start=cutoff.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("yfinance returned empty data")
        s = raw["Close"].squeeze()
        if hasattr(s.index, "tz") and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index.name = "date"
        s = s.dropna()
        weekly = s.resample("W-MON").last().dropna()
        print(f"  yfinance CL=F: {len(weekly)} weekly obs")
        return weekly
    except Exception as e:
        print(f"  yfinance failed ({e}), falling back to EIA RWTC XLS...")

    # Fallback: EIA RWTC XLS (daily → weekly)
    content = fetch_with_retry("https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls")
    for sheet in ("Data 1", 0):
        try:
            df = pd.read_excel(BytesIO(content), sheet_name=sheet, skiprows=2, header=0)
            df.columns = ["date", "price"] + list(df.columns[2:])
            df = df[["date", "price"]].dropna()
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["price"].astype(float)
            s.index.name = "date"
            s = s[s.index >= cutoff].dropna()
            return s.resample("W-MON").last().dropna()
        except Exception:
            continue
    raise ValueError("Could not fetch crude price data via yfinance or EIA")


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_lag_matrix(crude_gal: pd.Series, gas: pd.Series, n_lags: int):
    """
    Align series, build lag matrix, return (X, y, dates).
    X columns: [const, crude_lag_0, crude_lag_1, ..., crude_lag_n_lags]
    """
    df = pd.DataFrame({"gas": gas, "crude": crude_gal}).dropna()

    for k in range(n_lags + 1):
        df[f"lag_{k}"] = df["crude"].shift(k)

    df = df.dropna()
    lag_cols = [f"lag_{k}" for k in range(n_lags + 1)]
    X_raw = df[lag_cols].values
    y     = df["gas"].values
    const = np.ones((len(y), 1))
    X     = np.hstack([const, X_raw])
    return X, y, df.index, X_raw


# ---------------------------------------------------------------------------
# VIF / collinearity check
# ---------------------------------------------------------------------------

def max_off_diagonal_corr(X_raw: np.ndarray) -> float:
    """Return the maximum absolute off-diagonal correlation in X_raw."""
    corr = np.corrcoef(X_raw.T)
    np.fill_diagonal(corr, 0.0)
    return float(np.max(np.abs(corr)))


# ---------------------------------------------------------------------------
# Polynomial Distributed Lag (Almon, degree 2)
# ---------------------------------------------------------------------------

def fit_pdl(X_raw: np.ndarray, y: np.ndarray, degree: int = 2):
    """
    Fit a Polynomial Distributed Lag model.
    Constrains β_k = a_0 + a_1*k + a_2*k^2 (for degree 2).
    Returns (betas, alpha, residuals, r2).
    """
    n_lags_plus1 = X_raw.shape[1]
    k_vals = np.arange(n_lags_plus1)

    # Build Z: Z[:, d] = sum_k (k^d * lag_k) for d = 0..degree
    Z = np.column_stack([X_raw @ (k_vals ** d) for d in range(degree + 1)])
    X_pdl = np.hstack([np.ones((len(y), 1)), Z])

    # OLS on reduced parameter set [alpha, a0, a1, a2]
    coeffs, *_ = np.linalg.lstsq(X_pdl, y, rcond=None)
    alpha = coeffs[0]
    a     = coeffs[1:]  # polynomial coefficients

    # Recover β_k
    betas = np.array([sum(a[d] * (k ** d) for d in range(degree + 1))
                      for k in k_vals])

    y_hat    = X_pdl @ coeffs
    residuals = y - y_hat
    ss_res   = float(np.sum(residuals ** 2))
    ss_tot   = float(np.sum((y - y.mean()) ** 2))
    r2       = 1.0 - ss_res / ss_tot

    return betas, float(alpha), residuals, float(r2)


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------

def fit_ols(X: np.ndarray, y: np.ndarray):
    """Plain OLS. Returns (coeffs, residuals, r2)."""
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat      = X @ coeffs
    residuals  = y - y_hat
    ss_res     = float(np.sum(residuals ** 2))
    ss_tot     = float(np.sum((y - y.mean()) ** 2))
    r2         = 1.0 - ss_res / ss_tot
    return coeffs, residuals, float(r2)


# ---------------------------------------------------------------------------
# Asymmetry (rockets-and-feathers) test
# ---------------------------------------------------------------------------

def rockets_feathers_test(X_raw: np.ndarray, y: np.ndarray,
                           crude_gal: pd.Series) -> dict:
    """
    Add a single indicator term D_t × Δcrude_t where D_t = 1 if crude rose.
    Tests whether β_asymmetry is significantly positive (rockets > feathers).
    Returns dict with coefficient and p-value (approximate, based on t-ratio).
    """
    delta   = crude_gal.diff().dropna()
    aligned = delta.reindex(pd.RangeIndex(len(y)))  # rough alignment
    # If alignment fails, skip
    if aligned.isna().all():
        return {"coeff": None, "p_approx": None, "significant": False}

    indicator = (aligned > 0).astype(float).fillna(0).values
    asym_term = (indicator * aligned.fillna(0)).values

    X_aug = np.hstack([np.ones((len(y), 1)), X_raw, asym_term.reshape(-1, 1)])
    coeffs, residuals, r2_aug = fit_ols(X_aug, y)
    asym_coeff = float(coeffs[-1])

    # Approx std error from residual variance
    n, p  = X_aug.shape
    mse   = float(np.sum(residuals ** 2)) / max(n - p, 1)
    xtx_inv = np.linalg.pinv(X_aug.T @ X_aug)
    se    = math.sqrt(max(mse * xtx_inv[-1, -1], 0))
    t_stat = asym_coeff / se if se > 0 else 0.0

    # Rough two-tailed p-value from t-distribution approximation
    from math import lgamma, exp, log
    def t_cdf_approx(t, df):
        # Approximation good enough for our purpose
        x = df / (df + t * t)
        # Incomplete beta approximation
        if df < 1:
            return 0.5
        # Simple normal approximation for large df
        if df > 30:
            import math
            return math.erfc(abs(t) / math.sqrt(2))
        # For small df, use rough approximation
        return min(1.0, 2 * exp(-0.717 * abs(t) - 0.416 * t * t))

    p_approx = float(t_cdf_approx(abs(t_stat), n - p))
    significant = p_approx < 0.05 and asym_coeff > 0

    return {
        "coeff":       round(asym_coeff, 6),
        "t_stat":      round(t_stat, 3),
        "p_approx":    round(p_approx, 4),
        "significant": significant,
    }


# ---------------------------------------------------------------------------
# Seasonal residual
# ---------------------------------------------------------------------------

def compute_seasonal_premium(residuals: np.ndarray, dates: pd.DatetimeIndex) -> float:
    """
    Average residual in Apr-Sep minus average residual in Oct-Mar.
    Represents the pump price premium in summer NOT explained by crude alone.
    """
    summer_mask = dates.month.isin([4, 5, 6, 7, 8, 9])
    summer_resid = residuals[summer_mask]
    winter_resid = residuals[~summer_mask]
    if len(summer_resid) == 0 or len(winter_resid) == 0:
        return 0.0
    return float(np.mean(summer_resid) - np.mean(winter_resid))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fetching US retail gasoline price (weekly)...")
    gas_raw = fetch_eia_gasoline(lookback_years=3)
    print(f"  {len(gas_raw)} weekly obs, {gas_raw.index.min().date()} – {gas_raw.index.max().date()}")

    print("Fetching WTI crude price (weekly)...")
    crude_weekly = fetch_crude_weekly(lookback_years=3)
    crude_gal    = crude_weekly / 42.0
    print(f"  {len(crude_weekly)} weekly obs, {crude_weekly.index.min().date()} – {crude_weekly.index.max().date()}")

    # Trim to last 2 years for modelling (keep 3-year download for lag build-up)
    cutoff_2y = pd.Timestamp.today() - pd.DateOffset(years=2)
    gas_2y    = gas_raw[gas_raw.index   >= cutoff_2y]
    crude_2y  = crude_gal[crude_gal.index >= cutoff_2y]

    print(f"\nModelling on 2-year window: "
          f"{max(gas_2y.index.min(), crude_2y.index.min()).date()} – "
          f"{min(gas_2y.index.max(), crude_2y.index.max()).date()}")

    X, y, dates, X_raw = build_lag_matrix(crude_2y, gas_2y, N_LAGS)
    print(f"  {len(y)} observations after lagging ({N_LAGS} lags × weekly = {N_LAGS}-week lag window)")

    # Collinearity check
    max_corr = max_off_diagonal_corr(X_raw)
    use_pdl  = max_corr > PDL_THRESH
    print(f"\nMax off-diagonal lag correlation: {max_corr:.3f}  →  "
          f"{'PDL (Almon)' if use_pdl else 'OLS'} selected")

    if use_pdl:
        betas, alpha, residuals, r2 = fit_pdl(X_raw, y, degree=PDL_DEGREE)
        method = f"PDL(degree={PDL_DEGREE})"
    else:
        coeffs, residuals, r2 = fit_ols(X, y)
        alpha = float(coeffs[0])
        betas = coeffs[1:]
        method = "OLS"

    total_passthrough = float(np.sum(betas))
    cumulative        = np.cumsum(betas).tolist()

    print(f"\nFit: {method}, R² = {r2:.4f}")
    print(f"  α = {alpha:.4f}")
    for k, b in enumerate(betas):
        print(f"  β_{k} = {b:.5f}  (cumul {cumulative[k]:.3f})")
    print(f"  Total pass-through: {total_passthrough:.3f}")

    # Asymmetry test (informational; symmetric model retained regardless)
    asym = rockets_feathers_test(X_raw, y, crude_2y)
    print(f"\nRockets-and-feathers test: coeff={asym['coeff']}, "
          f"t={asym['t_stat']}, p≈{asym['p_approx']} "
          f"({'significant' if asym['significant'] else 'not significant'})")

    # Seasonal premium from residuals
    seasonal_premium = compute_seasonal_premium(residuals, dates)
    print(f"\nResidual seasonal premium (summer vs winter): "
          f"${seasonal_premium:+.4f}/gal")

    # Diagnostics
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))
    print(f"\nIn-sample diagnostics: RMSE={rmse:.4f}, MAE={mae:.4f}")

    output = {
        "generated":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_start":         str(dates.min().date()),
        "data_end":           str(dates.max().date()),
        "n_obs":              int(len(y)),
        "method":             method,
        "n_lags":             N_LAGS,
        "r2":                 round(r2, 5),
        "rmse":               round(rmse, 5),
        "mae":                round(mae, 5),
        "alpha":              round(alpha, 6),
        "betas":              [round(float(b), 6) for b in betas],
        "cumulative_passthrough": [round(float(c), 4) for c in cumulative],
        "total_passthrough":  round(total_passthrough, 4),
        "seasonal_premium":   round(seasonal_premium, 4),
        "asymmetry_test":     asym,
        "note": (
            "alpha captures national-average non-crude costs (taxes, distribution, "
            "avg crack spread). Add regional_offset for non-average regions. "
            "seasonal_premium is added Apr-Sep on top of model output."
        ),
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
