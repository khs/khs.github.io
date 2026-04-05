"""
Empirical Distributed Lag Model: Crude Oil → Retail Gasoline
=============================================================
Downloads up to 14 years of weekly data using EIA Open Data API (with key)
or public EIA/FRED XLS endpoints (no key required):
  - US retail gasoline price: EMM_EPMR_PTE_NUS_DPG ($/gal, weekly)
  - WTI spot crude:           RWTC ($/bbl, daily → resampled weekly)

Fits two models and outputs both to data/lag-model.json:

1. SYMMETRIC model (backward-compatible):
   pump_t = α + Σ_{k=0}^{8} β_k × (crude_{t-k}/42) + ε_t
   Method: PDL (Almon degree-2) if max lag correlation > 0.80, else OLS.

2. NARDL model (Shin, Yu & Greenwood-Nimmo 2014):
   pump_t = α + Σ β⁺_k × crude⁺_{t-k} + Σ β⁻_k × crude⁻_{t-k} + ε_t
   where crude⁺/crude⁻ are positive/negative partial sums of Δcrude.
   Captures "rockets and feathers": crude rises pass through faster than falls.

EIA_API_KEY environment variable: if set (from GitHub Actions secret), the EIA
Open Data API is used as the primary data source for full history and reliability.
Falls back to FRED streaming and EIA/XLS public endpoints if key is absent.

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
import os

OUT_FILE = Path(__file__).parents[2] / "data" / "lag-model.json"
HEADERS  = {"User-Agent": "oil-lag-model/1.0 (academic/personal use)"}
N_LAGS   = 8          # weeks: captures ~85-90% of pass-through per literature
PDL_THRESH = 0.80     # max off-diagonal correlation before switching to PDL
                      # 0.80 catches collinearity that produces physically impossible
                      # negative betas (e.g. β4=-0.21 under OLS) while staying well
                      # below the 0.95 threshold where OLS standard errors explode.
PDL_DEGREE = 2        # Almon polynomial degree
NARDL_LOOKBACK_YEARS = 14  # years of history for NARDL partial sums (back to 2012)
NARDL_ORIGIN = "2012-01-01"  # partial sums computed from this date


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


def fetch_gasoline_fred_streaming() -> pd.Series:
    """FRED GASREGCOVW via streaming GET — avoids the read-timeout that
    kills a blocking request on slow FRED responses."""
    import requests as _req
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GASREGCOVW"
    r = _req.get(url, headers=HEADERS, timeout=(10, 120), stream=True)
    r.raise_for_status()
    chunks = []
    for chunk in r.iter_content(chunk_size=8192):
        chunks.append(chunk)
    content = b"".join(chunks)
    from io import StringIO
    df = pd.read_csv(StringIO(content.decode("utf-8", errors="replace")),
                     parse_dates=["DATE"], index_col="DATE", na_values=[".", ""])
    s = df.squeeze().dropna().astype(float)
    s.index.name = "date"
    return s


def fetch_eia_gasoline_csv() -> pd.Series:
    """EIA LeafHandler CSV for US regular gasoline retail price (weekly)."""
    url = ("https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx"
           "?n=PET&s=EMM_EPMR_PTE_NUS_DPG&f=W")
    content = fetch_with_retry(url, retries=2, base_timeout=30)
    from io import StringIO
    text = content.decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(text), skiprows=2, header=0)
    df.columns = ["date", "price"] + list(df.columns[2:])
    df = df[["date", "price"]].dropna()
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["price"].astype(float)
    s.index.name = "date"
    if len(s) < 10:
        raise ValueError("Too few rows")
    return s


def fetch_gasoline_rbob_proxy() -> pd.Series:
    """Last resort: RBOB futures from yfinance as a proxy for retail gasoline.
    RBOB ≈ retail minus taxes/distribution (~$1.05/gal markup) but the
    lag-model coefficients are still meaningful since RBOB tracks retail closely."""
    import yfinance as yf
    raw = yf.download("RB=F", start="2020-01-01", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("yfinance RB=F returned empty")
    s = raw["Close"].squeeze()
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_convert(None)
    s.index.name = "date"
    # RBOB is in $/gallon already; add avg fixed retail markup to approximate retail
    RBOB_TO_RETAIL = 1.05   # taxes + distribution + retail margin
    return (s.dropna() + RBOB_TO_RETAIL).resample("W-MON").last().dropna()


def fetch_eia_gasoline_xls() -> pd.Series:
    """EIA hist_xls endpoint — same pattern as the working RWTC crude fetch."""
    url = "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGw.xls"
    content = fetch_with_retry(url, retries=2, base_timeout=30)
    for sheet in ("Data 1", 0):
        try:
            df = pd.read_excel(BytesIO(content), sheet_name=sheet, skiprows=2, header=0)
            df.columns = ["date", "price"] + list(df.columns[2:])
            df = df[["date", "price"]].dropna()
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["price"].astype(float)
            s.index.name = "date"
            if len(s) < 10:
                raise ValueError("Too few rows")
            return s
        except Exception:
            continue
    raise ValueError("Could not parse EIA gasoline XLS")


def fetch_eia_api_gasoline(api_key: str) -> pd.Series:
    """EIA Open Data API v2 for weekly retail gasoline — faster and more reliable
    than the XLS endpoint. Requires free EIA_API_KEY from api.eia.gov."""
    if not api_key:
        raise ValueError("EIA_API_KEY not set")
    url = (
        "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
        f"?api_key={api_key}&frequency=weekly&data[0]=value"
        "&facets[product][]=EPMR&facets[duoarea][]=NUS"
        "&sort[0][column]=period&sort[0][direction]=asc&length=5000"
    )
    content = fetch_with_retry(url, retries=3, base_timeout=30)
    data = json.loads(content)
    records = data["response"]["data"]
    df = pd.DataFrame(records)[["period", "value"]].dropna()
    df["date"] = pd.to_datetime(df["period"])
    s = df.set_index("date")["value"].astype(float)
    s.index.name = "date"
    if len(s) < 10:
        raise ValueError("Too few rows from EIA API")
    return s


def fetch_eia_api_crude(api_key: str) -> pd.Series:
    """EIA Open Data API v2 for WTI Cushing daily spot price."""
    if not api_key:
        raise ValueError("EIA_API_KEY not set")
    url = (
        "https://api.eia.gov/v2/petroleum/pri/spt/data/"
        f"?api_key={api_key}&frequency=daily&data[0]=value"
        "&facets[series][]=RWTC"
        "&sort[0][column]=period&sort[0][direction]=asc&length=20000"
    )
    content = fetch_with_retry(url, retries=3, base_timeout=30)
    data = json.loads(content)
    records = data["response"]["data"]
    df = pd.DataFrame(records)[["period", "value"]].dropna()
    df["date"] = pd.to_datetime(df["period"])
    s = df.set_index("date")["value"].astype(float)
    s.index.name = "date"
    if len(s) < 10:
        raise ValueError("Too few rows from EIA API")
    return s


def fetch_gasoline_weekly(lookback_years: int = 14) -> pd.Series:
    """Try multiple sources for US weekly retail gasoline price.
    Uses EIA Open Data API (requires EIA_API_KEY env var) as first choice;
    falls back to public XLS/CSV endpoints that return full history.
    """
    api_key = os.environ.get("EIA_API_KEY", "")
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=lookback_years)
    errors = []

    sources = []
    if api_key:
        sources.append(("EIA API", lambda: fetch_eia_api_gasoline(api_key)))
    sources += [
        ("FRED streaming",  fetch_gasoline_fred_streaming),
        ("EIA XLS",         fetch_eia_gasoline_xls),
        ("EIA LeafHandler", fetch_eia_gasoline_csv),
        ("RBOB proxy",      fetch_gasoline_rbob_proxy),
    ]

    for name, fn in sources:
        try:
            print(f"  Trying {name}...")
            s = fn()
            s = s[s.index >= cutoff]
            if len(s) < 10:
                raise ValueError("Too few observations")
            print(f"  {name} OK: {len(s)} obs, {s.index.min().date()} – {s.index.max().date()}")
            return s
        except Exception as e:
            msg = f"{name}: {e}"
            print(f"  {name} failed: {e}")
            errors.append(msg)

    raise RuntimeError("All gasoline data sources failed:\n" + "\n".join(errors))


def fetch_crude_weekly(lookback_years: int = 14) -> pd.Series:
    """Fetch weekly WTI crude price.
    Tries EIA Open Data API first (requires EIA_API_KEY), then yfinance,
    then EIA RWTC XLS. Returns weekly (Monday) resampled series.
    """
    api_key = os.environ.get("EIA_API_KEY", "")
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=lookback_years)

    # Try EIA API first (most reliable, full history)
    if api_key:
        try:
            print("  Trying EIA API (RWTC daily)...")
            s = fetch_eia_api_crude(api_key)
            s = s[s.index >= cutoff].dropna()
            weekly = s.resample("W-MON").last().dropna()
            if len(weekly) >= 10:
                print(f"  EIA API OK: {len(weekly)} weekly obs")
                return weekly
        except Exception as e:
            print(f"  EIA API failed ({e}), trying yfinance...")

    import yfinance as yf
    try:
        raw = yf.download("CL=F", start=cutoff.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("yfinance returned empty data")
        s = raw["Close"].squeeze()
        if hasattr(s.index, "tz") and s.index.tz is not None:
            s.index = s.index.tz_convert(None)
        s.index.name = "date"
        s = s.dropna()
        weekly = s.resample("W-MON").last().dropna()
        print(f"  yfinance CL=F: {len(weekly)} weekly obs")
        return weekly
    except Exception as e:
        print(f"  yfinance failed ({e}), falling back to EIA RWTC XLS...")

    # Fallback: EIA RWTC XLS (daily → weekly) — returns full history
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
    raise ValueError("Could not fetch crude price data via EIA API, yfinance, or EIA XLS")


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


def build_nardl_matrix(crude_gal: pd.Series, gas: pd.Series, n_lags: int,
                        origin: str = NARDL_ORIGIN):
    """
    Build the NARDL regressor matrix using Shin-Yu-Greenwood-Nimmo (2014) partial sums.

    Decomposes Δcrude into positive and negative components:
        crude⁺_t = Σ_{j≤t} max(Δcrude_j, 0)   (cumulative upward moves)
        crude⁻_t = Σ_{j≤t} min(Δcrude_j, 0)   (cumulative downward moves)

    Then builds lag matrices for both partial sums and aligns with gas prices.

    Returns (X, y, dates, X_pos_raw, X_neg_raw) where:
        X = [const | pos_lags | neg_lags]  shape (T, 1 + 2*(n_lags+1))
        X_pos_raw = lagged positive partial sums  shape (T, n_lags+1)
        X_neg_raw = lagged negative partial sums  shape (T, n_lags+1)
    """
    df = pd.DataFrame({"gas": gas, "crude": crude_gal}).dropna()

    # Compute partial sums from the origin date
    origin_ts = pd.Timestamp(origin)
    crude_full = df["crude"]
    delta = crude_full.diff()

    # Positive and negative increments
    pos_inc = delta.clip(lower=0).fillna(0)
    neg_inc = delta.clip(upper=0).fillna(0)

    # Cumulative partial sums; reset before origin to 0
    crude_pos = pos_inc.cumsum()
    crude_neg = neg_inc.cumsum()

    # Anchor partial sums to zero at origin
    if origin_ts in crude_pos.index:
        offset_pos = crude_pos.loc[origin_ts]
        offset_neg = crude_neg.loc[origin_ts]
    else:
        # Use nearest available date before origin
        before = crude_pos.index[crude_pos.index <= origin_ts]
        offset_pos = crude_pos.loc[before[-1]] if len(before) > 0 else 0.0
        offset_neg = crude_neg.loc[before[-1]] if len(before) > 0 else 0.0
    crude_pos = crude_pos - offset_pos
    crude_neg = crude_neg - offset_neg

    df["crude_pos"] = crude_pos
    df["crude_neg"] = crude_neg

    # Build lag matrices for both partial sums
    for k in range(n_lags + 1):
        df[f"pos_{k}"] = df["crude_pos"].shift(k)
        df[f"neg_{k}"] = df["crude_neg"].shift(k)

    df = df.dropna()

    pos_cols = [f"pos_{k}" for k in range(n_lags + 1)]
    neg_cols = [f"neg_{k}" for k in range(n_lags + 1)]
    X_pos_raw = df[pos_cols].values
    X_neg_raw = df[neg_cols].values
    y         = df["gas"].values
    const     = np.ones((len(y), 1))
    X         = np.hstack([const, X_pos_raw, X_neg_raw])
    return X, y, df.index, X_pos_raw, X_neg_raw


# ---------------------------------------------------------------------------
# VIF / collinearity check
# ---------------------------------------------------------------------------

def max_off_diagonal_corr(X_raw: np.ndarray) -> float:
    """Return the maximum absolute off-diagonal correlation in X_raw."""
    if X_raw.shape[1] < 2:
        return 0.0   # single column has no off-diagonal entries
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


def fit_nardl_ols(X_pos_raw: np.ndarray, X_neg_raw: np.ndarray,
                  y: np.ndarray):
    """
    Fit NARDL via OLS.  Combines positive and negative partial-sum lag matrices,
    runs fit_ols, then splits the coefficients back out.

    Returns (betas_pos, betas_neg, alpha, residuals, r2).
    """
    K     = X_pos_raw.shape[1]  # n_lags + 1
    const = np.ones((len(y), 1))
    X     = np.hstack([const, X_pos_raw, X_neg_raw])
    coeffs, residuals, r2 = fit_ols(X, y)
    alpha     = float(coeffs[0])
    betas_pos = coeffs[1 : K + 1]
    betas_neg = coeffs[K + 1:]
    return betas_pos, betas_neg, alpha, residuals, r2


def wald_asymmetry_test(betas_pos: np.ndarray, betas_neg: np.ndarray,
                         X: np.ndarray, residuals: np.ndarray) -> dict:
    """
    Wald test for long-run asymmetry: H0: Σβ⁺ = |Σβ⁻|.

    Computes W = (Σβ⁺ − |Σβ⁻|)² / Var(contrast), asymptotically χ²(1).
    p-value via normal approximation (erfc).

    Returns dict with sum_pos, sum_neg, W_stat, p_approx, significant.
    """
    n, p   = X.shape
    K      = len(betas_pos)
    mse    = float(np.sum(residuals ** 2)) / max(n - p, 1)
    xtx_inv = np.linalg.pinv(X.T @ X)

    # Contrast vector: sum of pos-beta columns minus sum of neg-beta columns
    # Coefficients: [alpha, pos_0..pos_K-1, neg_0..neg_K-1]
    c = np.zeros(p)
    c[1 : K + 1]  =  1.0   # sum β⁺
    c[K + 1:]     = -1.0   # minus sum β⁻  (but β⁻ are negative, so |Σβ⁻| = -Σβ⁻)

    var_contrast = mse * float(c @ xtx_inv @ c)
    sum_pos = float(np.sum(betas_pos))
    sum_neg = float(np.sum(betas_neg))
    contrast_val = sum_pos - sum_neg   # β⁻ ≤ 0, so this is sum_pos + |sum_neg|

    if var_contrast <= 0:
        return {"sum_pos": round(sum_pos, 4), "sum_neg": round(sum_neg, 4),
                "W_stat": None, "p_approx": None, "significant": False}

    W = contrast_val ** 2 / var_contrast
    # p-value: χ²(1) tail — use sqrt(W) ~ N(0,1)
    p_approx = float(math.erfc(math.sqrt(W / 2) / math.sqrt(2)))
    significant = p_approx < 0.05 and sum_pos > abs(sum_neg)

    return {
        "sum_pos":    round(sum_pos, 4),
        "sum_neg":    round(sum_neg, 4),
        "W_stat":     round(W, 3),
        "p_approx":   round(p_approx, 4),
        "significant": significant,
    }


# ---------------------------------------------------------------------------
# Asymmetry (rockets-and-feathers) test
# ---------------------------------------------------------------------------

def rockets_feathers_test(X_raw: np.ndarray, y: np.ndarray,
                           crude_gal: pd.Series, dates=None) -> dict:
    """
    Add a single indicator term D_t × Δcrude_t where D_t = 1 if crude rose.
    Tests whether β_asymmetry is significantly positive (rockets > feathers).
    Returns dict with coefficient and p-value (approximate, based on t-ratio).
    """
    delta = crude_gal.diff().dropna()
    if dates is not None:
        aligned = delta.reindex(dates)
    else:
        # Fallback: align by tail length
        tail = delta.iloc[-len(y):] if len(delta) >= len(y) else delta
        pad  = len(y) - len(tail)
        aligned = pd.Series([np.nan] * pad + list(tail.values))
    # If alignment fails, skip
    if aligned.isna().all():
        return {"coeff": None, "t_stat": None, "p_approx": None, "significant": False}

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
    def t_cdf_approx(t, df):
        # Approximation good enough for our purpose
        if df < 1:
            return 0.5
        # Simple normal approximation for large df
        if df > 30:
            return math.erfc(abs(t) / math.sqrt(2))
        # For small df, use rough approximation
        return min(1.0, 2 * math.exp(-0.717 * abs(t) - 0.416 * t * t))

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
    print("Fetching US retail gasoline price (weekly, 14-year history)...")
    gas_raw = fetch_gasoline_weekly(lookback_years=NARDL_LOOKBACK_YEARS)
    print(f"  {len(gas_raw)} weekly obs, {gas_raw.index.min().date()} – {gas_raw.index.max().date()}")

    print("Fetching WTI crude price (weekly, 14-year history)...")
    crude_weekly = fetch_crude_weekly(lookback_years=NARDL_LOOKBACK_YEARS)
    crude_gal    = crude_weekly / 42.0
    print(f"  {len(crude_weekly)} weekly obs, {crude_weekly.index.min().date()} – {crude_weekly.index.max().date()}")

    # Trim both series to the NARDL origin (2012-01-01) for modelling
    cutoff = pd.Timestamp(NARDL_ORIGIN)
    gas_fit   = gas_raw[gas_raw.index     >= cutoff]
    crude_fit = crude_gal[crude_gal.index >= cutoff]

    print(f"\nModelling window: {cutoff.date()} – "
          f"{min(gas_fit.index.max(), crude_fit.index.max()).date()}")

    # ------------------------------------------------------------------ #
    # 1. Symmetric model (PDL/OLS) — retained for backward compat         #
    # ------------------------------------------------------------------ #
    X, y, dates, X_raw = build_lag_matrix(crude_fit, gas_fit, N_LAGS)
    print(f"  {len(y)} observations after lagging")

    max_corr = max_off_diagonal_corr(X_raw)
    use_pdl  = max_corr > PDL_THRESH
    print(f"\nSymmetric model — max lag corr: {max_corr:.3f}  →  "
          f"{'PDL(Almon)' if use_pdl else 'OLS'} selected")

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

    print(f"  {method}, R²={r2:.4f}, total_passthrough={total_passthrough:.4f}")
    for k, b in enumerate(betas):
        print(f"  β_{k} = {b:+.5f}  (cumul {cumulative[k]:.3f})")

    # Symmetric diagnostics
    rmse_sym = float(np.sqrt(np.mean(residuals ** 2)))
    mae_sym  = float(np.mean(np.abs(residuals)))
    seasonal_premium = compute_seasonal_premium(residuals, dates)
    asym = rockets_feathers_test(X_raw, y, crude_fit, dates=dates)
    print(f"\nRockets-and-feathers (single indicator): p≈{asym['p_approx']} "
          f"({'sig' if asym['significant'] else 'not sig'})")
    print(f"Seasonal premium: ${seasonal_premium:+.4f}/gal")

    # ------------------------------------------------------------------ #
    # 2. NARDL model                                                       #
    # ------------------------------------------------------------------ #
    print(f"\nFitting NARDL (Shin-Yu-Greenwood-Nimmo 2014) on {NARDL_ORIGIN}+ data...")
    X_n, y_n, dates_n, X_pos, X_neg = build_nardl_matrix(
        crude_fit, gas_fit, N_LAGS, origin=NARDL_ORIGIN
    )
    print(f"  {len(y_n)} observations, {X_n.shape[1]} regressors")

    betas_pos, betas_neg, alpha_n, resid_n, r2_n = fit_nardl_ols(X_pos, X_neg, y_n)

    sum_pos = float(np.sum(betas_pos))
    sum_neg = float(np.sum(betas_neg))
    rmse_n  = float(np.sqrt(np.mean(resid_n ** 2)))
    print(f"  NARDL R²={r2_n:.4f}, RMSE={rmse_n:.5f}")
    print(f"  Σβ⁺ (upward passthrough)  = {sum_pos:+.4f}")
    print(f"  Σβ⁻ (downward passthrough) = {sum_neg:+.4f}")

    # Wald asymmetry test
    K     = X_pos.shape[1]
    const = np.ones((len(y_n), 1))
    X_n_full = np.hstack([const, X_pos, X_neg])
    wald = wald_asymmetry_test(betas_pos, betas_neg, X_n_full, resid_n)
    print(f"  Wald test: W={wald['W_stat']}, p≈{wald['p_approx']} "
          f"({'sig' if wald['significant'] else 'not sig'})")

    # band_passthrough: worst-case per-unit amplification used for band width.
    # Take the larger of upward/downward long-run multipliers so bands are
    # never narrower than either direction warrants.
    band_passthrough = round(max(sum_pos, abs(sum_neg)), 4)

    seasonal_n = compute_seasonal_premium(resid_n, dates_n)

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #
    output = {
        "generated":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_start":         str(dates.min().date()),
        "data_end":           str(dates.max().date()),
        "n_obs":              int(len(y)),
        "method":             method,
        "n_lags":             N_LAGS,
        "r2":                 round(r2, 5),
        "rmse":               round(rmse_sym, 5),
        "mae":                round(mae_sym, 5),
        "alpha":              round(alpha, 6),
        "betas":              [round(float(b), 6) for b in betas],
        "cumulative_passthrough": [round(float(c), 4) for c in cumulative],
        "total_passthrough":  round(total_passthrough, 4),
        "seasonal_premium":   round(seasonal_premium, 4),
        "asymmetry_test":     asym,
        # NARDL additions
        "nardl": True,
        "nardl_r2":               round(r2_n, 5),
        "nardl_rmse":             round(rmse_n, 5),
        "nardl_data_start":       str(dates_n.min().date()),
        "nardl_data_end":         str(dates_n.max().date()),
        "nardl_n_obs":            int(len(y_n)),
        "alpha_nardl":            round(float(alpha_n), 6),
        "betas_pos":              [round(float(b), 6) for b in betas_pos],
        "betas_neg":              [round(float(b), 6) for b in betas_neg],
        "total_passthrough_pos":  round(sum_pos, 4),
        "total_passthrough_neg":  round(sum_neg, 4),
        "band_passthrough":       band_passthrough,
        "seasonal_premium_nardl": round(seasonal_n, 4),
        "wald_test":              wald,
        "note": (
            "alpha/betas: symmetric PDL/OLS model (backward compat). "
            "alpha_nardl/betas_pos/betas_neg: NARDL asymmetric model "
            "(Shin-Yu-Greenwood-Nimmo 2014). "
            "band_passthrough = max(total_passthrough_pos, |total_passthrough_neg|) "
            "used by JS for pump band width. "
            "seasonal_premium added Apr-Sep; seasonal_premium_nardl from NARDL residuals."
        ),
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    import traceback
    DEBUG_FILE = Path(__file__).parents[2] / "data" / "lag-model-debug.json"
    try:
        main()
        # Clear any previous debug file on success
        if DEBUG_FILE.exists():
            DEBUG_FILE.unlink()
    except Exception:
        tb = traceback.format_exc()
        traceback.print_exc()
        with open(DEBUG_FILE, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "error": tb,
            }, f, indent=2)
        print(f"Wrote error details to {DEBUG_FILE}")
        raise
