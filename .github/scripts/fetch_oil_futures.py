"""
Fetches WTI and Brent crude oil futures prices from Yahoo Finance
and updates data/oil-futures.json with:
  - Current forward curve
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
        label = f"Front Month"
        contracts.insert(0, {
            "ticker": front_ticker,
            "expiry": today.strftime("%Y-%m"),
            "label": label,
            "price": round(float(front_price), 2),
        })

    return contracts


def main():
    # Load existing data
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            data = json.load(f)
    else:
        data = {"snapshots": []}

    # Fetch fresh curves
    print("Fetching WTI futures (CL, NYM)...")
    wti = fetch_curve("CL", "NYM")
    print(f"  Got {len(wti)} contracts")

    print("Fetching Brent futures (BZ, NYM)...")
    brent = fetch_curve("BZ", "NYM")
    print(f"  Got {len(brent)} contracts")

    now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
    today_str = date.today().isoformat()

    # Build snapshot (compact: just expiry + price)
    snapshot = {
        "date": today_str,
        "wti": [{"expiry": c["expiry"], "price": c["price"]} for c in wti],
        "brent": [{"expiry": c["expiry"], "price": c["price"]} for c in brent],
    }

    # Append snapshot, dedup by date (keep latest), trim to MAX_SNAPSHOTS
    existing = [s for s in data.get("snapshots", []) if s["date"] != today_str]
    existing.append(snapshot)
    existing.sort(key=lambda s: s["date"])
    snapshots = existing[-MAX_SNAPSHOTS:]

    # Write back
    output = {
        "last_updated": now_iso,
        "data_available": bool(wti or brent),
        "wti": wti,
        "brent": brent,
        "snapshots": snapshots,
    }

    with open(DATA_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {DATA_FILE}")
    print(f"  WTI contracts: {len(wti)}, Brent contracts: {len(brent)}, Snapshots: {len(snapshots)}")


if __name__ == "__main__":
    main()
