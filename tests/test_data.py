"""
Data integrity and schema tests for the project's JSON data files.
Run with:  pytest tests/test_data.py -v

These tests validate the *currently committed data*, not the live fetching.
They catch schema drift, corrupted output, obviously wrong values, and
internal consistency failures (e.g. sum(betas) != total_passthrough).
"""
import json
import math
import pytest
from pathlib import Path

ROOT = Path(__file__).parents[1]
FUTURES_FILE  = ROOT / "data" / "oil-futures.json"
LAG_MODEL_FILE = ROOT / "data" / "lag-model.json"
BACKTEST_FILE  = ROOT / "data" / "oil-backtest.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def futures():
    with open(FUTURES_FILE) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def lag_model():
    with open(LAG_MODEL_FILE) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def backtest():
    with open(BACKTEST_FILE) as f:
        return json.load(f)


# ===========================================================================
# oil-futures.json
# ===========================================================================

class TestFuturesSchema:
    def test_top_level_keys(self, futures):
        for key in ("last_updated", "data_available", "wti", "brent",
                    "crack_spreads", "snapshots"):
            assert key in futures, f"Missing key: {key}"

    def test_data_available_true(self, futures):
        assert futures["data_available"] is True

    def test_last_updated_is_string(self, futures):
        assert isinstance(futures["last_updated"], str)
        # Should look like an ISO timestamp
        assert "T" in futures["last_updated"]

    def test_wti_and_brent_non_empty(self, futures):
        assert len(futures["wti"]) >= 6,   "Expected at least 6 WTI contracts"
        assert len(futures["brent"]) >= 6, "Expected at least 6 Brent contracts"

    def test_contract_required_fields(self, futures):
        for benchmark in ("wti", "brent"):
            for c in futures[benchmark]:
                for field in ("ticker", "expiry", "label", "price"):
                    assert field in c, f"{benchmark} contract missing field {field}: {c}"

    def test_contract_prices_positive(self, futures):
        for benchmark in ("wti", "brent"):
            for c in futures[benchmark]:
                assert c["price"] > 0, f"{benchmark} {c['expiry']}: price {c['price']} not positive"

    def test_contract_prices_plausible(self, futures):
        # WTI and Brent crude: reasonably expect $20–$200/bbl range
        for benchmark in ("wti", "brent"):
            for c in futures[benchmark]:
                assert 20 <= c["price"] <= 300, \
                    f"{benchmark} {c['expiry']}: price ${c['price']}/bbl outside plausible range"

    def test_contracts_ordered_by_expiry(self, futures):
        for benchmark in ("wti", "brent"):
            expiries = [c["expiry"] for c in futures[benchmark]]
            assert expiries == sorted(expiries), \
                f"{benchmark} contracts not in chronological order: {expiries}"

    def test_expiry_format(self, futures):
        import re
        pattern = re.compile(r"^\d{4}-\d{2}$")
        for benchmark in ("wti", "brent"):
            for c in futures[benchmark]:
                assert pattern.match(c["expiry"]), \
                    f"Bad expiry format: {c['expiry']}"

    def test_no_duplicate_expiries(self, futures):
        for benchmark in ("wti", "brent"):
            expiries = [c["expiry"] for c in futures[benchmark]]
            assert len(expiries) == len(set(expiries)), \
                f"{benchmark} has duplicate expiry months"

    def test_brent_higher_than_wti(self, futures):
        # Brent historically trades at a premium to WTI; check front months
        wti_front   = futures["wti"][0]["price"]
        brent_front = futures["brent"][0]["price"]
        assert brent_front > wti_front - 10, \
            f"Brent ${brent_front} unexpectedly much lower than WTI ${wti_front}"

    def test_forward_curve_direction_reasonable(self, futures):
        # Extreme backwardation/contango: no single-month jump > $40/bbl
        for benchmark in ("wti", "brent"):
            prices = [c["price"] for c in futures[benchmark]]
            for i in range(1, len(prices)):
                delta = abs(prices[i] - prices[i-1])
                assert delta < 40, \
                    f"{benchmark}: month-to-month jump of ${delta:.2f}/bbl between " \
                    f"{futures[benchmark][i-1]['expiry']} and {futures[benchmark][i]['expiry']}"


class TestCrackSpreads:
    def test_crack_spreads_schema(self, futures):
        cs = futures["crack_spreads"]
        assert "by_month" in cs
        assert "average" in cs
        assert "fallback" in cs

    def test_crack_spreads_positive(self, futures):
        # Refining margin should be positive (it costs to refine crude)
        cs = futures["crack_spreads"]
        assert cs["average"] > 0, "Average crack spread should be positive"
        for m in cs["by_month"]:
            assert m["crack_spread"] > -0.5, \
                f"Crack spread {m['crack_spread']} for {m['expiry']} suspiciously negative"

    def test_crack_spreads_plausible(self, futures):
        # Crack spreads: ~$0.20/gal (low) to ~$1.20/gal (high)
        cs = futures["crack_spreads"]
        for m in cs["by_month"]:
            assert -0.5 <= m["crack_spread"] <= 2.0, \
                f"Crack spread {m['crack_spread']} for {m['expiry']} outside plausible range"

    def test_average_close_to_by_month_mean(self, futures):
        cs = futures["crack_spreads"]
        if cs["by_month"]:
            computed_avg = sum(m["crack_spread"] for m in cs["by_month"]) / len(cs["by_month"])
            assert abs(cs["average"] - computed_avg) < 0.01, \
                f"Stored average {cs['average']} differs from computed {computed_avg:.4f}"

    def test_rbob_prices_in_correct_units(self, futures):
        # RBOB should be in $/gallon (~$2–$5/gal range)
        for m in futures["crack_spreads"]["by_month"]:
            if "rbob_price" in m:
                assert 1.0 <= m["rbob_price"] <= 8.0, \
                    f"RBOB price {m['rbob_price']} for {m['expiry']} seems wrong (should be $/gal)"

    def test_crack_equals_rbob_minus_wti_over_42(self, futures):
        # Verify crack_spread = rbob_price - wti_price/42 for each month
        wti_map = {c["expiry"]: c["price"] for c in futures["wti"]}
        for m in futures["crack_spreads"]["by_month"]:
            if "rbob_price" not in m:
                continue
            wti = wti_map.get(m["expiry"])
            if wti is None:
                continue
            expected_crack = m["rbob_price"] - wti / 42.0
            assert abs(m["crack_spread"] - expected_crack) < 0.001, \
                f"crack_spread mismatch for {m['expiry']}: " \
                f"stored {m['crack_spread']:.4f}, computed {expected_crack:.4f}"


class TestSnapshots:
    def test_snapshots_is_list(self, futures):
        assert isinstance(futures["snapshots"], list)

    def test_snapshot_schema(self, futures):
        for s in futures["snapshots"]:
            assert "date" in s, "Snapshot missing 'date'"
            for bm in ("wti", "brent"):
                assert bm in s, f"Snapshot missing '{bm}'"
                assert isinstance(s[bm], list)

    def test_snapshot_dates_ordered(self, futures):
        dates = [s["date"] for s in futures["snapshots"]]
        assert dates == sorted(dates), "Snapshots not in chronological order"

    def test_snapshot_no_duplicate_dates(self, futures):
        dates = [s["date"] for s in futures["snapshots"]]
        assert len(dates) == len(set(dates)), "Duplicate snapshot dates"

    def test_snapshot_prices_positive(self, futures):
        for s in futures["snapshots"]:
            for bm in ("wti", "brent"):
                for c in s[bm]:
                    assert c["price"] > 0


class TestHistoricalWTI:
    def test_historical_wti_present_if_key_exists(self, futures):
        if "historical_wti" in futures:
            hist = futures["historical_wti"]
            assert len(hist) >= 12, "historical_wti should have at least 1 year of data"

    def test_historical_wti_prices_positive(self, futures):
        for h in futures.get("historical_wti", []):
            assert h["price"] > 0

    def test_historical_wti_ordered(self, futures):
        months = [h["month"] for h in futures.get("historical_wti", [])]
        assert months == sorted(months), "historical_wti not in chronological order"

    def test_historical_wti_no_duplicates(self, futures):
        months = [h["month"] for h in futures.get("historical_wti", [])]
        assert len(months) == len(set(months))

    def test_historical_wti_month_format(self, futures):
        import re
        pat = re.compile(r"^\d{4}-\d{2}$")
        for h in futures.get("historical_wti", []):
            assert pat.match(h["month"]), f"Bad month format: {h['month']}"

    def test_historical_wti_prices_plausible(self, futures):
        # WTI has ranged from ~$10 (1998/2020 crash) to ~$147 (2008 peak)
        for h in futures.get("historical_wti", []):
            assert 5 <= h["price"] <= 160, \
                f"Historical WTI price ${h['price']} for {h['month']} outside plausible range"


# ===========================================================================
# lag-model.json
# ===========================================================================

class TestLagModelSchema:
    def test_required_keys(self, lag_model):
        for key in ("alpha", "betas", "total_passthrough",
                    "seasonal_premium", "n_obs", "r2"):
            assert key in lag_model, f"lag-model.json missing key: {key}"

    def test_values_not_null(self, lag_model):
        for key in ("alpha", "betas", "total_passthrough"):
            assert lag_model[key] is not None, f"{key} is null — model not built?"

    def test_betas_sum_equals_total_passthrough(self, lag_model):
        if lag_model["betas"] is None:
            pytest.skip("Lag model not built")
        computed = sum(lag_model["betas"])
        stored   = lag_model["total_passthrough"]
        assert abs(computed - stored) < 0.001, \
            f"sum(betas)={computed:.4f} != total_passthrough={stored:.4f}"

    def test_cumulative_passthrough_consistent(self, lag_model):
        if lag_model.get("cumulative_passthrough") is None or lag_model["betas"] is None:
            pytest.skip("cumulative_passthrough not present")
        betas  = lag_model["betas"]
        cumul  = lag_model["cumulative_passthrough"]
        assert len(cumul) == len(betas), "cumulative_passthrough length != betas length"
        running = 0.0
        for i, (b, c) in enumerate(zip(betas, cumul)):
            running += b
            assert abs(running - c) < 0.002, \
                f"cumulative_passthrough[{i}]={c} but sum(betas[:i+1])={running:.4f}"

    def test_r2_in_range(self, lag_model):
        if lag_model.get("r2") is None:
            pytest.skip("R² not present")
        r2 = lag_model["r2"]
        assert 0.0 < r2 < 1.0, f"R²={r2} outside (0,1)"
        assert r2 > 0.5, f"R²={r2} unusually low — model may be broken"

    def test_alpha_plausible(self, lag_model):
        # Alpha captures taxes + distribution + avg refining (~$1.0–$2.0/gal)
        if lag_model["alpha"] is None:
            pytest.skip("Lag model not built")
        alpha = lag_model["alpha"]
        assert 0.5 <= alpha <= 3.0, \
            f"alpha={alpha} outside expected range $0.50–$3.00/gal"

    def test_total_passthrough_plausible(self, lag_model):
        if lag_model["total_passthrough"] is None:
            pytest.skip("Lag model not built")
        tp = lag_model["total_passthrough"]
        # Literature: 50%–130% pass-through; extremes suggest misspecification
        assert 0.3 <= tp <= 1.5, \
            f"total_passthrough={tp} outside expected range 0.30–1.50"

    def test_betas_length(self, lag_model):
        if lag_model["betas"] is None:
            pytest.skip("Lag model not built")
        n_lags = lag_model.get("n_lags", 8)
        assert len(lag_model["betas"]) == n_lags + 1, \
            f"Expected {n_lags+1} betas, got {len(lag_model['betas'])}"

    def test_n_obs_reasonable(self, lag_model):
        if lag_model.get("n_obs") is None:
            pytest.skip("n_obs not present")
        # 3-year window, weekly = ~156 obs; 2-year window = ~96 obs
        assert lag_model["n_obs"] >= 50, \
            f"n_obs={lag_model['n_obs']} suspiciously low"

    def test_rmse_plausible(self, lag_model):
        if lag_model.get("rmse") is None:
            pytest.skip("rmse not present")
        rmse = lag_model["rmse"]
        # In-sample RMSE for weekly pump price: expect $0.03–$0.50/gal
        assert 0.0 < rmse < 0.5, \
            f"in-sample RMSE={rmse} outside plausible range"

    def test_backtest_pump_passthrough_matches_lag_model(self, lag_model):
        """backtest_oil.PUMP_PASSTHROUGH must stay in sync with lag-model.json.

        If build_lag_model.py is re-run and produces a different total_passthrough,
        backtest_oil._load_pump_passthrough() should pick it up automatically.
        This test catches any regression where the loading mechanism breaks.
        """
        import sys
        sys.path.insert(0, str(ROOT / ".github" / "scripts"))
        from backtest_oil import PUMP_PASSTHROUGH
        expected = lag_model["total_passthrough"]
        assert abs(PUMP_PASSTHROUGH - expected) < 1e-4, (
            f"backtest_oil.PUMP_PASSTHROUGH={PUMP_PASSTHROUGH:.6f} diverged from "
            f"lag-model.json total_passthrough={expected:.6f}"
        )

    def test_pump_price_sanity_at_current_brent(self, futures, lag_model):
        """Compute pump for the front Brent contract and check it's in a sane range."""
        if lag_model["alpha"] is None or lag_model["betas"] is None:
            pytest.skip("Lag model not built")
        brent_front = futures["brent"][0]["price"]
        alpha = lag_model["alpha"]
        betas = lag_model["betas"]
        # Worst-case: all lags at same price (front-month clamped)
        crude_gal = brent_front / 42.0
        weighted  = sum(b * crude_gal for b in betas)
        pump_approx = alpha + weighted  # no seasonal, no crack dev
        # US national avg pump price: expect $2.50–$8.00/gal
        assert 1.50 <= pump_approx <= 10.0, \
            f"Estimated pump price ${pump_approx:.2f}/gal for Brent ${brent_front}/bbl " \
            f"is outside plausible range"


# ===========================================================================
# oil-backtest.json
# ===========================================================================

class TestBacktestSchema:
    def test_top_level_keys(self, backtest):
        for key in ("generated", "description", "results"):
            assert key in backtest, f"oil-backtest.json missing key: {key}"

    def test_all_horizons_present(self, backtest):
        for h in ("1m", "2m", "3m", "4m"):
            assert h in backtest["results"], f"Missing horizon {h}"

    def _horizon_items(self, backtest):
        """Iterate only over the 4 forecast horizons, skipping 'regimes'."""
        for h, stats in backtest["results"].items():
            if h in ("1m", "2m", "3m", "4m"):
                yield h, stats

    def test_horizon_required_fields(self, backtest):
        required = ("label", "n", "bias", "mae", "rmse",
                    "r2", "hit_rate", "coverage_68", "coverage_95")
        for h, stats in self._horizon_items(backtest):
            for field in required:
                assert field in stats, f"Horizon {h} missing field {field}"

    def test_n_observations_large_enough(self, backtest):
        for h, stats in self._horizon_items(backtest):
            assert stats["n"] >= 100, \
                f"Horizon {h}: only {stats['n']} obs — backtest likely failed"

    def test_mae_positive(self, backtest):
        for h, stats in self._horizon_items(backtest):
            assert stats["mae"] > 0, f"Horizon {h}: MAE should be positive"

    def test_rmse_ge_mae(self, backtest):
        for h, stats in self._horizon_items(backtest):
            assert stats["rmse"] >= stats["mae"], \
                f"Horizon {h}: RMSE {stats['rmse']} < MAE {stats['mae']} (impossible)"

    def test_coverage_in_range(self, backtest):
        for h, stats in self._horizon_items(backtest):
            assert 0.0 <= stats["coverage_68"] <= 1.0, \
                f"Horizon {h}: coverage_68={stats['coverage_68']} out of [0,1]"
            assert 0.0 <= stats["coverage_95"] <= 1.0, \
                f"Horizon {h}: coverage_95={stats['coverage_95']} out of [0,1]"

    def test_95_coverage_exceeds_68(self, backtest):
        for h, stats in self._horizon_items(backtest):
            assert stats["coverage_95"] >= stats["coverage_68"], \
                f"Horizon {h}: 95% band covers less than 68% band (impossible)"

    def test_coverage_68_roughly_right(self, backtest):
        for h, stats in self._horizon_items(backtest):
            cov = stats["coverage_68"]
            assert 0.50 <= cov <= 0.85, \
                f"Horizon {h}: 68% band coverage={cov:.2%} far from 68% target"

    def test_coverage_95_roughly_right(self, backtest):
        for h, stats in self._horizon_items(backtest):
            cov = stats["coverage_95"]
            assert 0.80 <= cov <= 1.00, \
                f"Horizon {h}: 95% band coverage={cov:.2%} far from 95% target"

    def test_hit_rate_near_50pct(self, backtest):
        for h, stats in self._horizon_items(backtest):
            if stats["hit_rate"] is not None:
                assert 0.35 <= stats["hit_rate"] <= 0.75, \
                    f"Horizon {h}: hit_rate={stats['hit_rate']:.2%} outside expected range"

    def test_horizons_ordered_by_mae(self, backtest):
        results = backtest["results"]
        maes = [results[h]["mae"] for h in ("1m", "2m", "3m", "4m") if h in results]
        for i in range(len(maes) - 1):
            assert maes[i] <= maes[i+1] * 1.5, \
                f"MAE ordering suspicious: {maes}"

    def test_above_plus_below_sum_to_one_minus_inside(self, backtest):
        for h, stats in self._horizon_items(backtest):
            inside_68  = stats.get("coverage_68", 0)
            above_68   = stats.get("pct_above_upper1", 0)
            below_68   = stats.get("pct_below_lower1", 0)
            if all(v is not None for v in [inside_68, above_68, below_68]):
                total = inside_68 + above_68 + below_68
                assert abs(total - 1.0) < 0.02, \
                    f"Horizon {h}: inside({inside_68})+above({above_68})+below({below_68})={total:.3f} != 1"

    def test_pump_results_present(self, backtest):
        assert "pump_results" in backtest, "pump_results missing from backtest"

    def test_pump_mae_in_cents_range(self, backtest):
        if "pump_results" not in backtest:
            pytest.skip("No pump_results")
        for h, stats in backtest["pump_results"].items():
            # Pump price MAE should be in $/gal; 1m ≈ $0.10-0.30/gal
            if stats.get("mae") is not None:
                assert 0.01 <= stats["mae"] <= 1.50, \
                    f"Pump horizon {h}: MAE=${stats['mae']}/gal outside expected range"

    def test_monthly_data_present(self, backtest):
        for h, stats in backtest["results"].items():
            if "monthly" in stats:
                assert len(stats["monthly"]) >= 100, \
                    f"Horizon {h}: only {len(stats['monthly'])} monthly records"
                for m in stats["monthly"][:5]:
                    for field in ("month", "error", "pred", "realized"):
                        assert field in m, f"Horizon {h} monthly missing field {field}"

    def test_regimes_present_and_valid(self, backtest):
        if "regimes" not in backtest["results"]:
            pytest.skip("No regime breakdown")
        regimes = backtest["results"]["regimes"]
        assert len(regimes) >= 4, "Expected at least 4 market regimes"
        for r in regimes:
            assert "name" in r
            assert "mae"  in r
            assert r["mae"] > 0
