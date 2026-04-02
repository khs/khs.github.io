"""
Oil Futures Backtest
====================
Downloads 20 years of WTI data and evaluates how well futures prices
predicted realized spot prices at 1-, 2-, 3-, and 4-month horizons.

Prediction series  : EIA RCLC1-4 (1st through 4th nearby NYMEX WTI futures)
Realized series    : FRED DCOILWTICO (WTI spot price)
Naive benchmark    : spot_t used as forecast for spot_{t+h}

Outputs data/oil-backtest.json consumed by the oil.html widget.
"""

import json
import math
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


def fetch_fred_spot() -> pd.Series:
    """Download WTI spot price from FRED (no API key required)."""
    print("Fetching FRED DCOILWTICO (WTI spot)...")
    content = fetch_with_retry(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"
    )
    df = pd.read_csv(
        BytesIO(content),
        parse_dates=["DATE"],
        index_col="DATE",
        na_values=[".", ""],
    )
    s = df.squeeze().dropna()
    s.index.name = "date"
    print(f"  {len(s)} observations, {s.index.min().date()} – {s.index.max().date()}")
    return s.astype(float)


def fetch_eia_futures(series: str) -> pd.Series:
    """Download an EIA nth-nearby futures series as an XLS file (no API key)."""
    print(f"Fetching EIA {series}...")
    content = fetch_with_retry(
        f"https://www.eia.gov/dnav/pet/hist_xls/{series}d.xls"
    )
    # EIA XLS: rows 0-1 are metadata, row 2 is header, data from row 3.
    # Try the known sheet name first; fall back to first sheet if it differs.
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

        stats = {
            "label":        cfg["label"],
            "n":            int(len(err)),
            "date_start":   str(dates.min().date()),
            "date_end":     str(dates.max().date()),
            "bias":         round2(bias(err)),
            "mae":          round2(mae(err)),
            "rmse":         round2(rmse(err)),
            "naive_rmse":   round2(rmse(naive_err)),
            "r2":           round2(r_squared(spot_real, fut_pred)),
            "hit_rate":     round2(hit_rate(spot_0, fut_pred, spot_real)),
            "within_10pct": round2(within_pct(err, 0.10, avg_price)),
            "within_20pct": round2(within_pct(err, 0.20, avg_price)),
        }
        results[key] = stats

        # Monthly summary for time-series chart
        df_monthly = pd.DataFrame({
            "date":      dates,
            "error":     err,
            "pct_error": err / spot_0 * 100,
            "pred":      fut_pred,
            "realized":  spot_real,
        })
        df_monthly["month"] = df_monthly["date"].dt.to_period("M").astype(str)
        monthly_agg = (
            df_monthly
            .groupby("month")
            .agg(
                error    = ("error",     "mean"),
                pct_error= ("pct_error", "mean"),
                pred     = ("pred",      "mean"),
                realized = ("realized",  "mean"),
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
# Main
# ---------------------------------------------------------------------------

def main():
    # Fetch data
    spot = fetch_fred_spot()

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

    print("\nRunning backtest...")
    backtest = run_backtest(spot, futures_map)

    output = {
        "generated":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": (
            "How well did WTI futures predict realized spot prices? "
            "Prediction = EIA nth-nearby futures (RCLC1-4). "
            "Realized = FRED DCOILWTICO spot price. "
            "Naive benchmark = today's spot price used as forecast."
        ),
        "results": backtest,
    }

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
    main()
