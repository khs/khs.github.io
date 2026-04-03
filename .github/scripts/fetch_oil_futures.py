"""
Fetches WTI, Brent, and RBOB gasoline futures from Yahoo Finance and updates
data/oil-futures.json with:
  - Current forward curve (WTI + Brent)
  - Crack spreads by month (RBOB gasoline futures minus WTI/42)
  - Rolling historical snapshots (one per week, kept for 3 years)
"""

import json
import calendar
from datetime import datetime, date, timezone
from pathlib import Path

import yfinance as yf

# ---------------------------------------------------------------------------
# Futures contract month codes (CME / NYMEX convention)
# ---------------------------------------------------------------------------
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z',
}

DATA_FILE = Path(__file__).parents[2] / 'data' / 'oil-futures.json'
MONTHS_AHEAD = 18          # how many future months to fetch
MAX_SNAPSHOTS = 156        # ~3 years of weekly snapshots


def build_ticker(base: str, exchange: str, year: int, month: int) -> str:
    code = MONTH_CODES[month]
    yr2 = str(year)[-2:]
    return f"{base}{code}{yr2}.{exchange}"


def fetch_curve(base: str, exchange: str) -> list[dict]:
    """Return a list of {expiry, label, price, ticker} for the forward curve."""
    today = date.today()
    contracts = []

    # Always include the front-month rolling contract as a reference point
    front_ticker = f"{base}=F"
    front_data = yf.Ticker(front_ticker)
    try:
        front_price = front_data.fast_info.last_price
    except Exception:
        front_price = None

    for offset in range(1, MONTHS_AHEAD + 1):
        total_months = today.month - 1 + offset
        year = today.year + total_months // 12
        month = total_months % 12 + 1

        ticker = build_ticker(base, exchange, year, month)
        label = f"{calendar.month_abbr[month]} {year}"
        expiry = f"{year}-{month:02d}"

        price = None
        try:
            info = yf.Ticker(ticker).fast_info
            p = info.last_price
            if p and p > 0:
                price = round(float(p), 2)
        except Exception:
            pass

        # If a specific contract returns nothing, skip it (exchange may not
        # list that expiry yet, or the ticker format differs)
        if price is not None:
            contracts.append({
                "ticker": ticker,
                "expiry": expiry,
                "label": label,
                "price": price,
            })

    # Prepend front-month if we got it and have no near-term contracts
    if not contracts and front_price:
        contracts.insert(0, {
            "ticker": front_ticker,
            "expiry": today.strftime("%Y-%m"),
            "label": "Front Month",
            "price": round(float(front_price), 2),
        })

    return contracts


def fetch_historical_monthly(years_back: int = 3) -> list[dict]:
    """
    Fetch monthly average WTI spot prices from EIA RWTC daily XLS.
    Returns [{month: "YYYY-MM", price: float}, ...] sorted oldest-first.
    Used by oil.html to draw the historical price line on the forward-curve chart.
    """
    import requests
    import pandas as pd
    from io import BytesIO
    from datetime import date

    cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=years_back)
    headers = {"User-Agent": "oil-futures-fetch/1.0 (personal use)"}

    try:
        r = requests.get(
            "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls",
            headers=headers, timeout=60,
        )
        r.raise_for_status()
        for sheet in ("Data 1", 0):
            try:
                df = pd.read_excel(BytesIO(r.content), sheet_name=sheet,
                                   skiprows=2, header=0)
                df.columns = ["date", "price"] + list(df.columns[2:])
                df = df[["date", "price"]].dropna()
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] >= cutoff]
                df["month"] = df["date"].dt.to_period("M").astype(str)
                monthly = (
                    df.groupby("month")["price"]
                    .mean()
                    .round(2)
                    .reset_index()
                )
                result = [
                    {"month": row["month"], "price": float(row["price"])}
                    for _, row in monthly.iterrows()
                ]
                print(f"  Historical WTI: {len(result)} monthly averages, "
                      f"{result[0]['month']} – {result[-1]['month']}")
                return result
            except Exception:
                continue
    except Exception as e:
        print(f"  WARNING: could not fetch historical WTI: {e}")
    return []


def fill_gaps(contracts: list[dict]) -> list[dict]:
    """
    Linearly interpolate any months that are missing between the first and
    last fetched contract.  Yahoo Finance occasionally omits a low-liquidity
    month; leaving it out creates a visible jump in the forward curve chart.
    Interpolated entries are flagged with "interpolated": true.
    """
    if len(contracts) < 2:
        return contracts

    # Build a price map keyed by (year, month)
    price_map = {}
    for c in contracts:
        y, m = int(c["expiry"][:4]), int(c["expiry"][5:7])
        price_map[(y, m)] = c["price"]

    first = contracts[0]
    last  = contracts[-1]
    y0, m0 = int(first["expiry"][:4]), int(first["expiry"][5:7])
    y1, m1 = int(last["expiry"][:4]),  int(last["expiry"][5:7])

    filled = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        if (y, m) in price_map:
            # Real contract — find original entry to preserve ticker etc.
            entry = next(c for c in contracts
                         if int(c["expiry"][:4]) == y and int(c["expiry"][5:7]) == m)
            filled.append(entry)
        else:
            # Find the nearest real contracts on either side for interpolation
            prev_ym = max((k for k in price_map if k < (y, m)), default=None)
            next_ym = min((k for k in price_map if k > (y, m)), default=None)
            if prev_ym and next_ym:
                py, pm = prev_ym
                ny, nm = next_ym
                # Ordinal month distances
                prev_ord = py * 12 + pm
                next_ord = ny * 12 + nm
                cur_ord  = y  * 12 + m
                frac  = (cur_ord - prev_ord) / (next_ord - prev_ord)
                price = round(price_map[prev_ym] + frac * (price_map[next_ym] - price_map[prev_ym]), 2)
                expiry = f"{y}-{m:02d}"
                filled.append({
                    "ticker":       f"(interpolated)",
                    "expiry":       expiry,
                    "label":        f"{calendar.month_abbr[m]} {y}",
                    "price":        price,
                    "interpolated": True,
                })
                print(f"  Interpolated missing contract {expiry} at ${price}")

        # Advance one month
        m += 1
        if m > 12:
            m = 1
            y += 1

    return filled


def fetch_crack_spreads(wti_contracts: list[dict]) -> dict:
    """
    Fetch RBOB gasoline futures for each WTI contract month and compute the
    crack spread (refining margin) = RBOB $/gal - WTI $/bbl / 42.

    RBOB tickers follow the same NYMEX month-code convention as WTI (RB).
    Returns:
      {
        "by_month": [{expiry, rbob_price, crack_spread}, ...],
        "average":  float,   # avg over available months
        "fallback": float,   # used when no RBOB data available
      }
    """
    FALLBACK_CRACK = 0.43  # $/gal historical average (EIA 2024)

    results = []
    for c in wti_contracts:
        year  = int(c["expiry"][:4])
        month = int(c["expiry"][5:7])
        rbob_ticker = build_ticker("RB", "NYM", year, month)

        try:
            info = yf.Ticker(rbob_ticker).fast_info
            rbob = info.last_price
            if rbob and rbob > 0:
                spread = round(float(rbob) - c["price"] / 42, 4)
                results.append({
                    "expiry":      c["expiry"],
                    "rbob_price":  round(float(rbob), 4),
                    "crack_spread": spread,
                })
        except Exception:
            pass

    if results:
        avg = round(sum(r["crack_spread"] for r in results) / len(results), 4)
    else:
        avg = FALLBACK_CRACK

    print(f"  RBOB crack spreads: {len(results)} months fetched, avg ${avg:.3f}/gal")
    return {
        "by_month": results,
        "average":  avg,
        "fallback": FALLBACK_CRACK,
    }


def main():
    # Load existing data
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            data = json.load(f)
    else:
        data = {"snapshots": []}

    # Fetch fresh curves
    print("Fetching WTI futures (CL, NYM)...")
    wti = fill_gaps(fetch_curve("CL", "NYM"))
    print(f"  Got {len(wti)} contracts")

    print("Fetching Brent futures (BZ, NYM)...")
    brent = fill_gaps(fetch_curve("BZ", "NYM"))
    print(f"  Got {len(brent)} contracts")

    print("Fetching historical WTI monthly prices (CL=F, 3 years)...")
    historical_wti = fetch_historical_monthly(years_back=3)

    # Crack spreads: RBOB gasoline futures vs WTI (use WTI as the refinery input)
    print("Fetching RBOB gasoline futures (RB, NYM) for crack spreads...")
    crack_spreads = fetch_crack_spreads(wti)

    now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
    today_str = date.today().isoformat()

    # Build snapshot (compact: just expiry + price)
    snapshot = {
        "date": today_str,
        "wti":  [{"expiry": c["expiry"], "price": c["price"]} for c in wti],
        "brent":[{"expiry": c["expiry"], "price": c["price"]} for c in brent],
    }

    # Append snapshot, dedup by date (keep latest), trim to MAX_SNAPSHOTS
    existing = [s for s in data.get("snapshots", []) if s["date"] != today_str]
    existing.append(snapshot)
    existing.sort(key=lambda s: s["date"])
    snapshots = existing[-MAX_SNAPSHOTS:]

    # Write back
    output = {
        "last_updated":   now_iso,
        "data_available": bool(wti or brent),
        "wti":            wti,
        "brent":          brent,
        "crack_spreads":  crack_spreads,
        "historical_wti": historical_wti,
        "snapshots":      snapshots,
    }

    with open(DATA_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {DATA_FILE}")
    print(f"  WTI: {len(wti)}, Brent: {len(brent)}, "
          f"Crack spreads: {len(crack_spreads['by_month'])}, "
          f"Snapshots: {len(snapshots)}")


if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Write whatever partial data we have so the file timestamp updates
        # even on failure, giving the commit step something to commit.
        err_path = Path(__file__).parents[2] / "data" / "futures-error.json"
        with open(err_path, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "error": traceback.format_exc(),
            }, f, indent=2)
        sys.exit(1)
