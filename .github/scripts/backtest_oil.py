"""
Oil Futures Backtest
====================
Downloads 20 years of WTI data and evaluates how well futures prices
predicted realized spot prices at 1-, 2-, 3-, and 4-month horizons.

Prediction series  : EIA RCLC1-4 (1st through 4th nearby NYMEX WTI futures)
Realized series    : EIA RWTC (WTI Cushing spot price, daily)
Naive benchmark    : spot_t used as forecast for spot_{t+h}

Also computes pump-price backtest:
Prediction: WTI futures → simple crude-to-pump passthrough (Brent adj + fixed costs)
Realized  : EIA US retail gasoline price (weekly, EMM_EPMR_PTE_NUS_DPG)

Also computes band coverage: what % of realized prices fell inside the
symmetric lognormal bands used by the widget (VOL_UP=0.33, VOL_DOWN=0.33).
Pump bands use PUMP_PASSTHROUGH and PUMP_RESIDUAL_GAL to match oil.html
pumpBandEndpoint exactly.

Outputs data/oil-backtest.json consumed by the oil.html widget.
"""

import json
import math
import os
import time
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import date, timezone, datetime
from pathlib import Path

OUT_FILE = Path(__file__).parents[2] / "data" / "oil-backtest.json"

# Approximate trading-day offsets for each "nearby" contract horizon
# RCLC1 ≈ 30 cal days, RCLC2 ≈ 60, RCLC3 ≈ 90, RCLC4 ≈ 120
HORIZONS = {
    "1m":  {"series": "RCLC1", "biz_days": 21,  "label": "1 month"},
    "2m":  {"series": "RCLC2", "biz_days": 42,  "label": "2 months"},
    "3m":  {"series": "RCLC3", "biz_days": 63,  "label": "3 months"},
    "4m":  {"series": "RCLC4", "biz_days": 84,  "label": "4 months"},
}

REGIMES = [
    {"name": "Pre-crisis bull run",  "start": "2004-01-01", "end": "2007-12-31"},
    {"name": "Financial crisis",     "start": "2008-01-01", "end": "2009-06-30"},
    {"name": "Recovery & plateau",   "start": "2009-07-01", "end": "2014-06-30"},
    {"name": "Shale/OPEC crash",     "start": "2014-07-01", "end": "2016-12-31"},
    {"name": "Moderate period",      "start": "2017-01-01", "end": "2019-12-31"},
    {"name": "COVID crash",          "start": "2020-01-01", "end": "2020-12-31"},
    {"name": "Recovery & Ukraine",   "start": "2021-01-01", "end": "2022-12-31"},
    {"name": "Normalization",        "start": "2023-01-01", "end": "2025-12-31"},
]

HEADERS = {"User-Agent": "oil-backtest-script/1.0 (academic/personal use)"}

# Band parameters — must match oil.html constants exactly
VOL_UP   = 0.33   # annualised upside vol
VOL_DOWN = 0.33   # annualised downside vol
MR_CAP   = 2.5    # mean-reversion cap in years

# Crude → pump passthrough constants — must match oil.html exactly
WTI_TO_BRENT_ADJ  = 4.0    # $/bbl premium Brent over WTI
PUMP_FIXED_TOTAL   = 1.399  # $/gal: federal+state tax + refining + distribution + retail
# Lag-model helpers — loaded at startup so band constants stay in sync after
# a model refit without requiring a manual edit here.
def _load_lag_model() -> dict:
    """Load the lag model JSON; returns empty dict on failure."""
    try:
        _lm = Path(__file__).parents[2] / "data" / "lag-model.json"
        with open(_lm) as _f:
            return json.load(_f)
    except Exception:
        return {}

def _load_pump_passthrough(fallback: float = 0.954) -> float:
    """band_passthrough = max(|pos|,|neg|) worst-case amplification for band width."""
    lm = _load_lag_model()
    return float(lm.get("band_passthrough", lm.get("total_passthrough", fallback)))

PUMP_PASSTHROUGH   = _load_pump_passthrough()
# Non-crude pump price residual — calibrated so 1m pump band achieves ~68% coverage.
# Represents crack-spread, tax, and distribution variance not captured by crude alone.
# Must match PUMP_RESIDUAL_GAL in oil.html.
PUMP_RESIDUAL_GAL  = 0.37


# ---------------------------------------------------------------------------
# Data fetching (with retry + extended timeout)
# ---------------------------------------------------------------------------

def fetch_with_retry(url: str, retries: int = 4, base_timeout: int = 60) -> bytes:
    for attempt in range(retries):
        timeout = base_timeout * (2 ** attempt)
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"  Attempt {attempt + 1} failed ({e}), retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise


def fetch_eia_spot() -> pd.Series:
    """Download WTI Cushing spot price from EIA public XLS (no API key required)."""
    print("Fetching EIA RWTC (WTI spot)...")
    content = fetch_with_retry(
        "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
    )
    for sheet in ("Data 1", 0):
        try:
            df = pd.read_excel(BytesIO(content), sheet_name=sheet, skiprows=2, header=0)
            df.columns = ["date", "price"] + list(df.columns[2:])
            df = df[["date", "price"]].dropna()
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["price"].astype(float)
            s.index.name = "date"
            print(f"  {len(s)} observations, {s.index.min().date()} – {s.index.max().date()}")
            return s
        except Exception:
            continue
    raise ValueError("Could not parse EIA RWTC XLS")


def fetch_eia_api_futures(series: str, api_key: str) -> pd.Series:
    """Fetch nth-nearby WTI futures from EIA API v2 (extends XLS beyond Apr 2024)."""
    url = "https://api.eia.gov/v2/petroleum/pri/fut/data/"
    params = {
        "api_key":              api_key,
        "frequency":            "daily",
        "data[0]":              "value",
        "facets[series][]":     series,
        "sort[0][column]":      "period",
        "sort[0][direction]":   "asc",
        "length":               5000,
        "offset":               0,
    }
    all_rows: list = []
    while True:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        rows  = payload.get("response", {}).get("data", [])
        total = payload.get("response", {}).get("total", 0)
        all_rows.extend(rows)
        if len(all_rows) >= total or not rows:
            break
        params["offset"] += params["length"]
    if not all_rows:
        raise ValueError(f"No data returned from EIA API for {series}")
    df = pd.DataFrame(all_rows)
    df["date"]  = pd.to_datetime(df["period"])
    df["price"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["price"].dropna().sort_index()
    s.index.name = "date"
    return s


def fetch_eia_futures(series: str) -> pd.Series:
    """Download an EIA nth-nearby futures series.

    Tries EIA API v2 first (when EIA_API_KEY is set) to get post-Apr 2024
    data, then falls back to the public XLS endpoint.
    """
    api_key = os.environ.get("EIA_API_KEY", "")
    if api_key:
        try:
            print(f"Fetching EIA {series} via API v2...")
            s = fetch_eia_api_futures(series, api_key)
            print(f"  {len(s)} observations, {s.index.min().date()} – {s.index.max().date()}")
            return s
        except Exception as e:
            print(f"  API fetch failed ({e}), falling back to XLS...")

    print(f"Fetching EIA {series} via XLS...")
    content = fetch_with_retry(
        f"https://www.eia.gov/dnav/pet/hist_xls/{series}d.xls"
    )
    # EIA XLS: rows 0-1 are metadata, row 2 is header, data from row 3.
    for sheet in ("Data 1", 0):
        try:
            df = pd.read_excel(BytesIO(content), sheet_name=sheet, skiprows=2, header=0)
            df.columns = ["date", "price"] + list(df.columns[2:])
            df = df[["date", "price"]].dropna()
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["price"].astype(float)
            s.index.name = "date"
            print(f"  {len(s)} observations, {s.index.min().date()} – {s.index.max().date()}")
            return s
        except Exception:
            continue
    raise ValueError(f"Could not parse EIA XLS for {series}")


def fetch_eia_gasoline() -> pd.Series:
    """Download EIA US National Average retail gasoline price (weekly)."""
    print("Fetching EIA retail gasoline (EMM_EPMR_PTE_NUS_DPG)...")
    content = fetch_with_retry(
        "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGw.xls"
    )
    for sheet in ("Data 1", 0):
        try:
            df = pd.read_excel(BytesIO(content), sheet_name=sheet, skiprows=2, header=0)
            df.columns = ["date", "price"] + list(df.columns[2:])
            df = df[["date", "price"]].dropna()
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["price"].astype(float)
            s.index.name = "date"
            print(f"  {len(s)} observations, {s.index.min().date()} – {s.index.max().date()}")
            return s
        except Exception:
            continue
    raise ValueError("Could not parse EIA retail gasoline XLS")


def crude_to_pump_simple(wti_bbl: float) -> float:
    """WTI $/bbl → US national avg pump price $/gal (simple passthrough, no lag)."""
    brent_bbl = wti_bbl + WTI_TO_BRENT_ADJ
    return brent_bbl / 42.0 + PUMP_FIXED_TOTAL


def crude_to_pump_lag_model_steady_state(
    wti_bbl: float, lm: dict, delivery_month: int
) -> float:
    """WTI $/bbl → US national avg pump price $/gal using the lag model.

    Uses the model's alpha intercept plus total passthrough applied to the Brent
    spot equivalent, plus the seasonal premium (Apr–Sep) when applicable.
    This is the steady-state prediction: same crude price held constant for all
    lag windows, which is a reasonable approximation for the backtest where we
    don't have the full week-by-week history at each prediction date.

    Using the lag model eliminates the ~$0.50–0.73/gal over-prediction bias
    seen in 2004–2011 when the simple formula is applied to those years.
    """
    brent_gal = (wti_bbl + WTI_TO_BRENT_ADJ) / 42.0
    alpha     = lm.get("alpha", 1.594)
    # Prefer NARDL positive pass-through (rising crude → pump response dominates
    # the average consumer experience); fall back to symmetric total.
    tp        = lm.get("total_passthrough_pos", lm.get("total_passthrough", 0.954))
    seasonal  = lm.get("seasonal_premium", 0.0) if 4 <= delivery_month <= 9 else 0.0
    return alpha + tp * brent_gal + seasonal


# ---------------------------------------------------------------------------
# Band computation (mirrors oil.html computeBands exactly)
# ---------------------------------------------------------------------------

def compute_bands(price: float, T_years: float) -> dict:
    """Return 1σ/2σ asymmetric lognormal band endpoints for a given price and horizon."""
    Teff     = min(T_years, MR_CAP)
    sig_up   = VOL_UP   * math.sqrt(Teff)
    sig_down = VOL_DOWN * math.sqrt(Teff)
    return {
        "upper1": price * math.exp(    sig_up),
        "lower1": price * math.exp(-   sig_down),
        "upper2": price * math.exp(2 * sig_up),
        "lower2": price * math.exp(-2 * sig_down),
    }


def compute_pump_bands(crude_bbl: float, pump_price: float, T_years: float) -> dict:
    """Band endpoints for pump price predictions ($/gal).

    Mirrors oil.html pumpBandEndpoint: applies lognormal vol to the crude price
    ($/bbl), converts the resulting crude delta to a pump delta via passthrough
    and /42, then adds PUMP_RESIDUAL_GAL in quadrature to account for non-crude
    variance (crack spreads, taxes, distribution costs).

    Args:
        crude_bbl:  WTI futures price at time of prediction ($/bbl).
        pump_price: Predicted pump price at time of prediction ($/gal).
        T_years:    Forecast horizon in years.
    """
    crude_bands = compute_bands(crude_bbl, T_years)
    def combine(crude_band_bbl: float, n: int) -> float:
        crude_delta_gal = (crude_band_bbl - crude_bbl) / 42.0 * PUMP_PASSTHROUGH
        # Mirror JS Math.sign behaviour: zero crude delta → zero total delta
        # (matches oil.html pumpBandEndpoint: totalDelta = Math.sign(crudeDelta) * sqrt(...))
        if crude_delta_gal == 0.0:
            return pump_price
        sign = math.copysign(1.0, crude_delta_gal)
        return pump_price + sign * math.sqrt(crude_delta_gal ** 2 + (n * PUMP_RESIDUAL_GAL) ** 2)
    return {
        "upper1": combine(crude_bands["upper1"], 1),
        "lower1": combine(crude_bands["lower1"], 1),
        "upper2": combine(crude_bands["upper2"], 2),
        "lower2": combine(crude_bands["lower2"], 2),
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))

def mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err)))

def bias(err: np.ndarray) -> float:
    """Mean error (positive = futures under-predicted realized price)."""
    return float(np.mean(err))

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)

def hit_rate(spot_t: np.ndarray, futures_t: np.ndarray, spot_realized: np.ndarray) -> float:
    """Fraction of times futures correctly called direction of price change."""
    futures_direction = np.sign(futures_t - spot_t)
    actual_direction  = np.sign(spot_realized - spot_t)
    # Exclude flat cases
    mask = (futures_direction != 0) & (actual_direction != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((futures_direction[mask] == actual_direction[mask]).mean())

def within_pct(err: np.ndarray, pct: float, mid_price: float) -> float:
    threshold = mid_price * pct
    return float((np.abs(err) <= threshold).mean())

def round2(x) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), 2)


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def run_backtest(spot: pd.Series, futures_map: dict[str, pd.Series]) -> dict:
    results = {}

    for key, cfg in HORIZONS.items():
        series_name = cfg["series"]
        biz_days    = cfg["biz_days"]

        if series_name not in futures_map:
            continue

        fut = futures_map[series_name]

        # Align: inner join on dates both series have data
        joined = pd.concat({"fut": fut, "spot": spot}, axis=1).dropna()
        joined = joined[joined.index >= "2004-01-01"]

        # Shift spot forward by biz_days to get "realized price at horizon"
        # Use positional shift on the sorted, unique-index series
        spot_sorted = joined["spot"]
        fut_vals    = joined["fut"]

        # Build a shifted realized series: for each date at position i,
        # realized = spot at position i + biz_days
        idx = spot_sorted.index
        realized_prices = spot_sorted.shift(-biz_days)
        spot_at_t       = spot_sorted

        # Drop rows where we don't have a realized price yet
        mask = ~realized_prices.isna()
        fut_pred   = fut_vals[mask].values
        spot_0     = spot_at_t[mask].values
        spot_real  = realized_prices[mask].values
        dates      = idx[mask]

        err      = spot_real - fut_pred
        naive_err = spot_real - spot_0   # random-walk benchmark

        avg_price = float(np.median(spot_real))

        # Band coverage: apply asymmetric lognormal bands to each historical prediction.
        # Spot series is daily (EIA RWTC), so biz_days rows ≈ biz_days trading days;
        # divide by 252 trading-days/year for the horizon in years.
        # (Contrast: pump backtest uses weeks_ahead/52 because gasoline data is weekly.)
        T_years = biz_days / 252.0
        band_arr = np.array([list(compute_bands(p, T_years).values()) for p in fut_pred])
        upper1, lower1, upper2, lower2 = (
            band_arr[:, 0], band_arr[:, 1], band_arr[:, 2], band_arr[:, 3]
        )
        inside_68 = (spot_real >= lower1) & (spot_real <= upper1)
        inside_95 = (spot_real >= lower2) & (spot_real <= upper2)

        stats = {
            "label":           cfg["label"],
            "n":               int(len(err)),
            "date_start":      str(dates.min().date()),
            "date_end":        str(dates.max().date()),
            "bias":            round2(bias(err)),
            "mae":             round2(mae(err)),
            "rmse":            round2(rmse(err)),
            "naive_rmse":      round2(rmse(naive_err)),
            "r2":              round2(r_squared(spot_real, fut_pred)),
            "hit_rate":        round2(hit_rate(spot_0, fut_pred, spot_real)),
            "within_10pct":    round2(within_pct(err, 0.10, avg_price)),
            "within_20pct":    round2(within_pct(err, 0.20, avg_price)),
            # Band calibration
            "coverage_68":          round2(float(inside_68.mean())),
            "coverage_95":          round2(float(inside_95.mean())),
            "pct_above_upper1":     round2(float((spot_real > upper1).mean())),
            "pct_below_lower1":     round2(float((spot_real < lower1).mean())),
            "pct_above_upper2":     round2(float((spot_real > upper2).mean())),
            "pct_below_lower2":     round2(float((spot_real < lower2).mean())),
        }
        results[key] = stats

        # Monthly summary for time-series chart
        df_monthly = pd.DataFrame({
            "date":      dates,
            "error":     err,
            "pct_error": err / spot_0 * 100,
            "pred":      fut_pred,
            "realized":  spot_real,
            "inside_68": inside_68,
            "inside_95": inside_95,
        })
        df_monthly["month"] = df_monthly["date"].dt.to_period("M").astype(str)
        monthly_agg = (
            df_monthly
            .groupby("month")
            .agg(
                error     = ("error",     "mean"),
                pct_error = ("pct_error", "mean"),
                pred      = ("pred",      "mean"),
                realized  = ("realized",  "mean"),
                inside_68 = ("inside_68", "mean"),
                inside_95 = ("inside_95", "mean"),
            )
            .reset_index()
        )
        results[key]["monthly"] = [
            {
                "month":     r["month"],
                "error":     round2(r["error"]),
                "pct_error": round2(r["pct_error"]),
                "pred":      round2(r["pred"]),
                "realized":  round2(r["realized"]),
                "inside_68": round2(r["inside_68"]),
                "inside_95": round2(r["inside_95"]),
            }
            for _, r in monthly_agg.iterrows()
        ]

        # Per-year breakdown
        df_monthly["year"] = pd.to_datetime(df_monthly["month"]).dt.year
        by_year = (
            df_monthly
            .assign(abs_err=lambda d: d["error"].abs())
            .groupby("year")
            .agg(
                mae          = ("abs_err",   "mean"),
                rmse_val     = ("error",     lambda x: math.sqrt((x**2).mean())),
                bias_val     = ("error",     "mean"),
                pct_error    = ("pct_error", lambda x: x.abs().mean()),
                n            = ("error",     "count"),
            )
            .reset_index()
        )
        results[key]["by_year"] = [
            {
                "year":      int(r["year"]),
                "mae":       round2(r["mae"]),
                "rmse":      round2(r["rmse_val"]),
                "bias":      round2(r["bias_val"]),
                "pct_error": round2(r["pct_error"]),
                "n":         int(r["n"]),
            }
            for _, r in by_year.iterrows()
        ]

    # Regime breakdown (1m only for brevity)
    if "1m" in results:
        key = "1m"
        fut  = futures_map["RCLC1"]
        joined = pd.concat({"fut": fut, "spot": spot}, axis=1).dropna()
        joined = joined[joined.index >= "2004-01-01"]
        spot_sorted = joined["spot"]
        fut_vals    = joined["fut"]
        realized_prices = spot_sorted.shift(-21)
        mask = ~realized_prices.isna()
        df_full = pd.DataFrame({
            "date":     spot_sorted.index[mask],
            "error":    (realized_prices[mask] - fut_vals[mask]).values,
            "pct_err":  ((realized_prices[mask] - fut_vals[mask]) / spot_sorted[mask] * 100).values,
        })

        regime_stats = []
        for reg in REGIMES:
            sub = df_full[
                (df_full["date"] >= reg["start"]) &
                (df_full["date"] <= reg["end"])
            ]
            if len(sub) < 5:
                continue
            e = sub["error"].values
            regime_stats.append({
                "name":      reg["name"],
                "start":     reg["start"][:7],
                "end":       reg["end"][:7],
                "n":         int(len(e)),
                "bias":      round2(bias(e)),
                "mae":       round2(mae(e)),
                "rmse":      round2(rmse(e)),
                "pct_error": round2(float(np.mean(np.abs(sub["pct_err"].values)))),
            })
        results["regimes"] = regime_stats

    return results


# ---------------------------------------------------------------------------
# Pump price backtest
# ---------------------------------------------------------------------------

def run_pump_backtest(gasoline: pd.Series, futures_map: dict[str, pd.Series]) -> dict:
    """
    Backtest pump price predictions against realized EIA retail gasoline.
    Prediction = lag model steady-state(WTI RCLC{n} futures at date t).
    Realized   = EIA weekly retail gasoline price at t + h weeks.

    Cutoff is 2012-01-01: applying current tax/cost structure to 2004-2011
    data causes a ~$0.50-0.73/gal over-prediction bias that distorts coverage
    statistics.  The lag model was trained on 2012-onwards data anyway.
    """
    lm = _load_lag_model()
    results = {}

    for key, cfg in HORIZONS.items():
        series_name = cfg["series"]
        biz_days    = cfg["biz_days"]

        if series_name not in futures_map:
            continue

        fut = futures_map[series_name]

        # Align daily crude futures to weekly gasoline dates via forward-fill
        combined = pd.concat([fut.rename("fut"), gasoline.rename("gas")], axis=1)
        combined["fut"] = combined["fut"].ffill(limit=7)
        joined = combined.dropna()
        # 2012-01-01 avoids the pre-shale-era tax-structure bias
        joined = joined[joined.index >= "2012-01-01"]

        if len(joined) < 50:
            continue

        gas_sorted = joined["gas"]
        if lm:
            pump_pred_s = joined.apply(
                lambda row: crude_to_pump_lag_model_steady_state(
                    row["fut"], lm, row.name.month
                ),
                axis=1,
            )
        else:
            pump_pred_s = joined["fut"].apply(crude_to_pump_simple)

        # Shift gasoline forward to get realized price at horizon
        # Gasoline is weekly, so approximate biz_days as calendar weeks
        weeks_ahead = max(1, round(biz_days / 5))
        realized_gas = gas_sorted.shift(-weeks_ahead)

        mask = ~realized_gas.isna()
        crude_pred = joined["fut"][mask].values   # WTI $/bbl at prediction time
        pump_pred  = pump_pred_s[mask].values
        gas_real   = realized_gas[mask].values
        gas_t      = gas_sorted[mask].values
        dates      = joined.index[mask]

        err       = gas_real - pump_pred
        naive_err = gas_real - gas_t

        avg_price = float(np.median(gas_real))

        # Use weeks_ahead (the actual shift applied) not biz_days, since gasoline
        # data is weekly and the realized horizon is weeks_ahead weeks exactly.
        T_years  = weeks_ahead / 52.0
        band_arr = np.array([
            list(compute_pump_bands(c, p, T_years).values())
            for c, p in zip(crude_pred, pump_pred)
        ])
        upper1, lower1, upper2, lower2 = (
            band_arr[:, 0], band_arr[:, 1], band_arr[:, 2], band_arr[:, 3]
        )
        inside_68 = (gas_real >= lower1) & (gas_real <= upper1)
        inside_95 = (gas_real >= lower2) & (gas_real <= upper2)

        stats = {
            "label":            cfg["label"],
            "n":                int(len(err)),
            "date_start":       str(dates.min().date()),
            "date_end":         str(dates.max().date()),
            "bias":             round2(bias(err)),
            "mae":              round2(mae(err)),
            "rmse":             round2(rmse(err)),
            "naive_rmse":       round2(rmse(naive_err)),
            "r2":               round2(r_squared(gas_real, pump_pred)),
            "hit_rate":         round2(hit_rate(gas_t, pump_pred, gas_real)),
            "within_10pct":     round2(within_pct(err, 0.10, avg_price)),
            "within_20pct":     round2(within_pct(err, 0.20, avg_price)),
            "coverage_68":      round2(float(inside_68.mean())),
            "coverage_95":      round2(float(inside_95.mean())),
            "pct_above_upper1": round2(float((gas_real > upper1).mean())),
            "pct_below_lower1": round2(float((gas_real < lower1).mean())),
            "pct_above_upper2": round2(float((gas_real > upper2).mean())),
            "pct_below_lower2": round2(float((gas_real < lower2).mean())),
        }

        # Monthly summary
        df_m = pd.DataFrame({
            "date":      dates,
            "error":     err,
            "pct_error": err / gas_t * 100,
            "pred":      pump_pred,
            "realized":  gas_real,
            "inside_68": inside_68,
            "inside_95": inside_95,
        })
        df_m["month"] = df_m["date"].dt.to_period("M").astype(str)
        monthly_agg = (
            df_m.groupby("month")
            .agg(
                error     = ("error",     "mean"),
                pct_error = ("pct_error", "mean"),
                pred      = ("pred",      "mean"),
                realized  = ("realized",  "mean"),
                inside_68 = ("inside_68", "mean"),
                inside_95 = ("inside_95", "mean"),
            )
            .reset_index()
        )
        stats["monthly"] = [
            {
                "month":     r["month"],
                "error":     round2(r["error"]),
                "pct_error": round2(r["pct_error"]),
                "pred":      round2(r["pred"]),
                "realized":  round2(r["realized"]),
                "inside_68": round2(r["inside_68"]),
                "inside_95": round2(r["inside_95"]),
            }
            for _, r in monthly_agg.iterrows()
        ]

        # Per-year breakdown
        df_m["year"] = pd.to_datetime(df_m["month"]).dt.year
        by_year = (
            df_m.assign(abs_err=lambda d: d["error"].abs())
            .groupby("year")
            .agg(
                mae       = ("abs_err",   "mean"),
                rmse_val  = ("error",     lambda x: math.sqrt((x**2).mean())),
                bias_val  = ("error",     "mean"),
                pct_error = ("pct_error", lambda x: x.abs().mean()),
                n         = ("error",     "count"),
            )
            .reset_index()
        )
        stats["by_year"] = [
            {
                "year":      int(r["year"]),
                "mae":       round2(r["mae"]),
                "rmse":      round2(r["rmse_val"]),
                "bias":      round2(r["bias_val"]),
                "pct_error": round2(r["pct_error"]),
                "n":         int(r["n"]),
            }
            for _, r in by_year.iterrows()
        ]

        results[key] = stats

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Fetch data
    spot = fetch_eia_spot()

    futures_map = {}
    for cfg in HORIZONS.values():
        series = cfg["series"]
        if series not in futures_map:
            try:
                futures_map[series] = fetch_eia_futures(series)
            except Exception as e:
                print(f"  WARNING: could not fetch {series}: {e}")

    if not futures_map:
        print("ERROR: No futures data fetched. Aborting.")
        return

    print("\nRunning WTI crude backtest...")
    backtest = run_backtest(spot, futures_map)

    output = {
        "generated":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": (
            "How well did WTI futures predict realized spot prices, and how well "
            "were the uncertainty bands calibrated? "
            "Prediction = EIA nth-nearby futures (RCLC1-4). "
            "Realized = EIA RWTC WTI Cushing spot price. "
            "Naive benchmark = today's spot price used as forecast. "
            f"Bands: VOL_UP={VOL_UP}, VOL_DOWN={VOL_DOWN}, MR_CAP={MR_CAP}y. "
            "pump_results: same horizons but prediction = WTI futures converted to "
            "pump price via lag-model steady state (alpha + total_passthrough_pos * brent_gal + seasonal); "
            "realized = EIA retail gasoline weekly. Cutoff 2012-01-01 avoids pre-shale-era tax-structure bias. "
            f"Pump bands: crude vol applied to crude $/bbl, converted via "
            f"PUMP_PASSTHROUGH={PUMP_PASSTHROUGH}/42, then PUMP_RESIDUAL_GAL={PUMP_RESIDUAL_GAL} $/gal "
            "added in quadrature for non-crude pump price variance (crack spreads, taxes, distribution)."
        ),
        "results": backtest,
    }

    try:
        gasoline = fetch_eia_gasoline()
        print("\nRunning pump price backtest...")
        pump_backtest = run_pump_backtest(gasoline, futures_map)
        output["pump_results"] = pump_backtest
        for key, stats in pump_backtest.items():
            print(
                f"  pump {key}: n={stats['n']}, bias={stats['bias']:+.4f}, "
                f"RMSE={stats['rmse']:.4f} (naive={stats['naive_rmse']:.4f})"
            )
    except Exception as e:
        print(f"  WARNING: Pump backtest skipped: {e}")

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {OUT_FILE}")
    for key, stats in backtest.items():
        if key == "regimes":
            continue
        print(
            f"  {key}: n={stats['n']}, bias={stats['bias']:+.2f}, "
            f"RMSE={stats['rmse']:.2f} (naive={stats['naive_rmse']:.2f}), "
            f"hit={stats['hit_rate']:.1%}"
        )


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
