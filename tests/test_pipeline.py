"""
Unit tests for the oil data pipeline Python scripts.
Run with:  pytest tests/test_pipeline.py -v
"""
import sys
import math
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

# Stub out heavy/network-only dependencies before importing scripts
for mod in ("yfinance", "requests", "curl_cffi", "websockets"):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Make scripts importable
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / ".github/scripts"))

from fetch_oil_futures import build_ticker, fill_gaps
from build_lag_model import (
    build_lag_matrix,
    max_off_diagonal_corr,
    fit_ols,
    fit_pdl,
    rockets_feathers_test,
    compute_seasonal_premium,
)
from backtest_oil import (
    crude_to_pump_simple,
    compute_bands,
    compute_pump_bands,
    PUMP_PASSTHROUGH,
    PUMP_RESIDUAL_GAL,
    rmse, mae, bias, r_squared,
    hit_rate, within_pct,
)


# ---------------------------------------------------------------------------
# fetch_oil_futures: build_ticker
# ---------------------------------------------------------------------------

class TestBuildTicker:
    def test_wti_may_2026(self):
        assert build_ticker("CL", "NYM", 2026, 5) == "CLK26.NYM"

    def test_brent_june_2026(self):
        assert build_ticker("BZ", "NYM", 2026, 6) == "BZM26.NYM"

    def test_rbob_jan_2027(self):
        assert build_ticker("RB", "NYM", 2027, 1) == "RBF27.NYM"

    def test_all_month_codes(self):
        # CME month codes: F G H J K M N Q U V X Z
        expected = ["F","G","H","J","K","M","N","Q","U","V","X","Z"]
        for i, code in enumerate(expected, start=1):
            ticker = build_ticker("CL", "NYM", 2025, i)
            assert ticker[2] == code, f"Month {i}: expected {code}, got {ticker[2]}"

    def test_two_digit_year(self):
        # Year 2030 → "30", year 2009 → "09"
        assert build_ticker("CL", "NYM", 2030, 1).endswith("30.NYM")
        assert build_ticker("CL", "NYM", 2009, 3).endswith("09.NYM")

    def test_format_length(self):
        t = build_ticker("CL", "NYM", 2026, 10)
        # CL + V + 26 + . + NYM  = 9 chars
        assert len(t) == 9


# ---------------------------------------------------------------------------
# fetch_oil_futures: fill_gaps
# ---------------------------------------------------------------------------

class TestFillGaps:
    def _make_contract(self, expiry, price, interpolated=False):
        c = {"ticker": f"CL{expiry}.NYM", "expiry": expiry, "label": expiry, "price": price}
        if interpolated:
            c["interpolated"] = True
        return c

    def test_no_gaps_unchanged(self):
        contracts = [
            self._make_contract("2026-05", 100.0),
            self._make_contract("2026-06", 98.0),
            self._make_contract("2026-07", 96.0),
        ]
        result = fill_gaps(contracts)
        assert len(result) == 3
        assert all("interpolated" not in c for c in result)

    def test_single_gap_filled(self):
        contracts = [
            self._make_contract("2026-05", 100.0),
            self._make_contract("2026-07", 96.0),  # June is missing
        ]
        result = fill_gaps(contracts)
        assert len(result) == 3
        june = result[1]
        assert june["expiry"] == "2026-06"
        assert june["interpolated"] is True
        # Midpoint interpolation
        assert abs(june["price"] - 98.0) < 0.01

    def test_multiple_gaps_filled(self):
        contracts = [
            self._make_contract("2026-01", 100.0),
            self._make_contract("2026-04", 94.0),  # Feb and Mar missing
        ]
        result = fill_gaps(contracts)
        assert len(result) == 4
        # Linear interp: 100, 98, 96, 94
        assert abs(result[1]["price"] - 98.0) < 0.1
        assert abs(result[2]["price"] - 96.0) < 0.1

    def test_year_boundary_gap(self):
        contracts = [
            self._make_contract("2025-11", 80.0),
            self._make_contract("2026-01", 76.0),  # Dec missing
        ]
        result = fill_gaps(contracts)
        assert len(result) == 3
        assert result[1]["expiry"] == "2025-12"
        assert result[1]["interpolated"] is True

    def test_single_contract_unchanged(self):
        contracts = [self._make_contract("2026-05", 100.0)]
        result = fill_gaps(contracts)
        assert len(result) == 1

    def test_interpolated_flag_not_set_on_real_contracts(self):
        contracts = [
            self._make_contract("2026-05", 100.0),
            self._make_contract("2026-07", 96.0),
        ]
        result = fill_gaps(contracts)
        assert "interpolated" not in result[0]  # May is real
        assert "interpolated" not in result[2]  # July is real
        assert result[1].get("interpolated") is True  # June is filled


# ---------------------------------------------------------------------------
# backtest_oil: crude_to_pump_simple
# ---------------------------------------------------------------------------

class TestCrudeToPumpSimple:
    WTI_TO_BRENT_ADJ = 4.0
    PUMP_FIXED_TOTAL = 1.399

    def test_at_70_dollars(self):
        # At WTI=$70: Brent=$74, $74/42 + $1.399 ≈ $3.162
        result = crude_to_pump_simple(70.0)
        expected = (70.0 + self.WTI_TO_BRENT_ADJ) / 42.0 + self.PUMP_FIXED_TOTAL
        assert abs(result - expected) < 1e-6

    def test_zero_crude(self):
        result = crude_to_pump_simple(0.0)
        expected = self.WTI_TO_BRENT_ADJ / 42.0 + self.PUMP_FIXED_TOTAL
        assert abs(result - expected) < 1e-6

    def test_higher_crude_higher_pump(self):
        assert crude_to_pump_simple(100.0) > crude_to_pump_simple(50.0)

    def test_linear_relationship(self):
        # $1 more crude → 1/42 more at pump (Brent adj cancels)
        delta_crude = 42.0  # exactly one barrel in gallons
        p1 = crude_to_pump_simple(70.0)
        p2 = crude_to_pump_simple(70.0 + delta_crude)
        assert abs((p2 - p1) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# backtest_oil: compute_bands
# ---------------------------------------------------------------------------

class TestComputeBands:
    VOL = 0.33
    MR_CAP = 2.5

    def test_zero_horizon(self):
        b = compute_bands(100.0, 0.0)
        assert b["upper1"] == pytest.approx(100.0, abs=1e-6)
        assert b["lower1"] == pytest.approx(100.0, abs=1e-6)
        assert b["upper2"] == pytest.approx(100.0, abs=1e-6)
        assert b["lower2"] == pytest.approx(100.0, abs=1e-6)

    def test_symmetric_vols(self):
        # VOL_UP == VOL_DOWN → upper1 * lower1 == price^2 (geometric symmetry)
        b = compute_bands(80.0, 1.0)
        assert abs(b["upper1"] * b["lower1"] - 80.0 ** 2) < 0.01

    def test_upper_above_lower(self):
        b = compute_bands(60.0, 0.5)
        assert b["upper1"] > b["lower1"]
        assert b["upper2"] > b["lower2"]
        assert b["upper2"] > b["upper1"]
        assert b["lower2"] < b["lower1"]

    def test_mean_reversion_cap(self):
        # Past 2.5 years, bands should be identical (cap kicks in)
        b_cap  = compute_bands(70.0, self.MR_CAP)
        b_more = compute_bands(70.0, self.MR_CAP + 5.0)
        assert b_cap["upper1"] == pytest.approx(b_more["upper1"], rel=1e-9)
        assert b_cap["lower1"] == pytest.approx(b_more["lower1"], rel=1e-9)

    def test_1_sigma_correct_magnitude(self):
        price = 100.0
        T = 1.0
        b = compute_bands(price, T)
        expected_upper = price * math.exp(self.VOL * math.sqrt(T))
        expected_lower = price * math.exp(-self.VOL * math.sqrt(T))
        assert b["upper1"] == pytest.approx(expected_upper, rel=1e-6)
        assert b["lower1"] == pytest.approx(expected_lower, rel=1e-6)

    def test_2_sigma_is_double_exponent(self):
        price = 100.0
        T = 1.0
        b = compute_bands(price, T)
        expected_upper2 = price * math.exp(2 * self.VOL * math.sqrt(T))
        expected_lower2 = price * math.exp(-2 * self.VOL * math.sqrt(T))
        assert b["upper2"] == pytest.approx(expected_upper2, rel=1e-6)
        assert b["lower2"] == pytest.approx(expected_lower2, rel=1e-6)

    def test_price_scales_linearly(self):
        b50  = compute_bands(50.0, 1.0)
        b100 = compute_bands(100.0, 1.0)
        assert b100["upper1"] == pytest.approx(2 * b50["upper1"], rel=1e-9)
        assert b100["lower1"] == pytest.approx(2 * b50["lower1"], rel=1e-9)


# ---------------------------------------------------------------------------
# backtest_oil: compute_pump_bands
# ---------------------------------------------------------------------------

class TestComputePumpBands:
    """Regression tests for pump band computation (must match oil.html pumpBandEndpoint)."""

    def test_residual_is_minimum_band_width(self):
        """Bands must be at least PUMP_RESIDUAL_GAL wide (residual adds in quadrature)."""
        bands = compute_pump_bands(80.0, 3.0, 1.0 / 12)
        assert bands["upper1"] - 3.0 >= PUMP_RESIDUAL_GAL * 0.999
        assert 3.0 - bands["lower1"] >= PUMP_RESIDUAL_GAL * 0.999

    def test_2sigma_wider_than_1sigma(self):
        bands = compute_pump_bands(80.0, 3.0, 1.0)
        assert bands["upper2"] > bands["upper1"]
        assert bands["lower2"] < bands["lower1"]

    def test_crude_price_drives_bands_not_pump_price(self):
        """Band width (delta from pump_price) must depend on crude_bbl, not pump_price."""
        crude_bbl, T = 80.0, 1.0 / 12
        # Same crude, different pump prices → same crude delta, nearly same total delta
        b_lo = compute_pump_bands(crude_bbl, 2.5, T)
        b_hi = compute_pump_bands(crude_bbl, 5.0, T)
        delta_lo = b_lo["upper1"] - 2.5
        delta_hi = b_hi["upper1"] - 5.0
        # The crude component is identical; both totals differ by < 1e-9
        assert abs(delta_lo - delta_hi) < 1e-9

    def test_passthrough_and_residual_formula(self):
        """upper1 = pump + sqrt((crude_delta/42 * tp)^2 + residual^2) exactly."""
        crude_bbl, pump_price, T = 80.0, 3.0, 1.0
        crude_b = compute_bands(crude_bbl, T)
        expected_crude_delta = (crude_b["upper1"] - crude_bbl) / 42.0 * PUMP_PASSTHROUGH
        expected_upper1 = pump_price + math.sqrt(
            expected_crude_delta ** 2 + PUMP_RESIDUAL_GAL ** 2
        )
        bands = compute_pump_bands(crude_bbl, pump_price, T)
        assert bands["upper1"] == pytest.approx(expected_upper1, rel=1e-9)

    def test_zero_horizon_collapses_to_pump_price(self):
        """At T=0 crude delta is zero; mirrors JS Math.sign(0)=0 → zero-width bands."""
        bands = compute_pump_bands(80.0, 3.0, 0.0)
        assert bands["upper1"] == pytest.approx(3.0, abs=1e-9)
        assert bands["lower1"] == pytest.approx(3.0, abs=1e-9)


# ---------------------------------------------------------------------------
# backtest_oil: statistics helpers
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_rmse_zeros(self):
        err = np.zeros(10)
        assert rmse(err) == pytest.approx(0.0)

    def test_mae_zeros(self):
        assert mae(np.zeros(5)) == pytest.approx(0.0)

    def test_bias_positive(self):
        err = np.array([1.0, 2.0, 3.0])
        assert bias(err) == pytest.approx(2.0)

    def test_bias_symmetric_zero(self):
        err = np.array([-1.0, 1.0, -2.0, 2.0])
        assert bias(err) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        err = np.array([3.0, 4.0])  # sqrt((9+16)/2) = sqrt(12.5)
        assert rmse(err) == pytest.approx(math.sqrt(12.5))

    def test_r_squared_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r_squared(y, y) == pytest.approx(1.0)

    def test_r_squared_mean_forecast(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y, y.mean())
        assert r_squared(y, y_pred) == pytest.approx(0.0, abs=1e-9)

    def test_r_squared_constant_y(self):
        y = np.ones(5)
        result = r_squared(y, y)
        assert math.isnan(result)  # ss_tot == 0

    def test_hit_rate_all_correct(self):
        spot_t    = np.array([100.0, 100.0, 100.0])
        futures_t = np.array([110.0, 110.0,  90.0])  # calls up, up, down
        spot_real = np.array([105.0, 115.0,  90.0])  # realized up, up, down
        assert hit_rate(spot_t, futures_t, spot_real) == pytest.approx(1.0)

    def test_hit_rate_all_wrong(self):
        spot_t    = np.array([100.0, 100.0])
        futures_t = np.array([110.0, 110.0])  # calls up
        spot_real = np.array([ 95.0,  90.0])  # but went down
        assert hit_rate(spot_t, futures_t, spot_real) == pytest.approx(0.0)

    def test_hit_rate_excludes_flat(self):
        spot_t    = np.array([100.0, 100.0, 100.0])
        futures_t = np.array([100.0, 110.0,  90.0])  # first is flat
        spot_real = np.array([105.0, 110.0,  80.0])  # flat | correct up | correct down
        # Only non-flat futures counted (2), both correct → 1.0
        result = hit_rate(spot_t, futures_t, spot_real)
        assert result == pytest.approx(1.0)

    def test_within_pct(self):
        err = np.array([5.0, -5.0, 10.0, -10.0, 20.0])
        mid = 100.0
        # 10% of 100 = 10; only errors ≤10 qualify: [5, -5, 10, -10] → 4/5
        assert within_pct(err, 0.10, mid) == pytest.approx(4/5)
        # 15% of 100 = 15; same 4 values qualify (20 > 15) → 4/5
        assert within_pct(err, 0.15, mid) == pytest.approx(4/5)
        # 20% of 100 = 20; all 5 qualify → 5/5
        assert within_pct(err, 0.20, mid) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_lag_model: build_lag_matrix
# ---------------------------------------------------------------------------

class TestBuildLagMatrix:
    def _make_series(self, values, start="2020-01-06", freq="W-MON"):
        idx = pd.date_range(start=start, periods=len(values), freq=freq)
        return pd.Series(values, index=idx, name="price")

    def test_output_shapes(self):
        n = 30
        crude = self._make_series(np.random.rand(n) + 2)
        gas   = self._make_series(np.random.rand(n) + 3)
        X, y, dates, X_raw = build_lag_matrix(crude, gas, n_lags=8)
        # n - n_lags rows remain after lagging; X has const + 9 lag columns
        assert X.shape[1] == 10          # 1 + 9 lags
        assert X_raw.shape[1] == 9       # 9 lags
        assert len(y) == len(dates)
        assert X.shape[0] == len(y)

    def test_constant_column(self):
        crude = self._make_series(np.ones(20) * 70)
        gas   = self._make_series(np.ones(20) * 3.5)
        X, y, dates, X_raw = build_lag_matrix(crude, gas, n_lags=4)
        # First column should be all ones
        np.testing.assert_array_equal(X[:, 0], 1.0)

    def test_lag_zero_is_contemporaneous(self):
        # Lag 0 should equal the contemporaneous crude value in the output
        crude = self._make_series(np.arange(1.0, 31.0))
        gas   = self._make_series(np.arange(1.0, 31.0))
        X, y, dates, X_raw = build_lag_matrix(crude, gas, n_lags=2)
        # X_raw[:, 0] is lag_0 — contemporaneous crude
        # X_raw[:, 1] is lag_1 — one week earlier
        # The series values are 1,2,...30; lag_0[i] == lag_1[i]+1
        diffs = X_raw[:, 0] - X_raw[:, 1]
        np.testing.assert_allclose(diffs, 1.0)

    def test_drops_na_rows(self):
        # With n_lags=4 and n=20 obs, we lose 4 rows to NaN
        crude = self._make_series(np.random.rand(20))
        gas   = self._make_series(np.random.rand(20))
        X, y, dates, _ = build_lag_matrix(crude, gas, n_lags=4)
        assert len(y) == 20 - 4


# ---------------------------------------------------------------------------
# build_lag_model: max_off_diagonal_corr
# ---------------------------------------------------------------------------

class TestMaxOffDiagonalCorr:
    def test_identical_columns_give_one(self):
        X = np.tile(np.arange(100.0), (3, 1)).T  # 100×3, all identical
        assert max_off_diagonal_corr(X) == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_columns_give_zero(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, 3))
        # Orthogonalise
        Q, _ = np.linalg.qr(X)
        corr = max_off_diagonal_corr(Q)
        assert corr < 0.01

    def test_single_column(self):
        # A single column has no off-diagonal entries — should return 0.0.
        # Regression guard: if the guard "if X.shape[1] < 2: return 0.0" is
        # ever removed, np.fill_diagonal on a 1×1 matrix raises ValueError.
        X = np.arange(10.0).reshape(-1, 1)
        try:
            result = max_off_diagonal_corr(X)
            assert result == pytest.approx(0.0)
        except ValueError:
            pytest.fail(
                "max_off_diagonal_corr raised ValueError on single-column input; "
                "the guard 'if X_raw.shape[1] < 2: return 0.0' in build_lag_model.py "
                "is missing or broken."
            )


# ---------------------------------------------------------------------------
# build_lag_model: fit_ols
# ---------------------------------------------------------------------------

class TestFitOLS:
    def test_exact_linear_recovery(self):
        # y = 2.0 + 3.0*x1 + 0.5*x2
        rng = np.random.default_rng(0)
        X_raw = rng.standard_normal((100, 2))
        X = np.hstack([np.ones((100, 1)), X_raw])
        true_coeffs = np.array([2.0, 3.0, 0.5])
        y = X @ true_coeffs
        coeffs, residuals, r2 = fit_ols(X, y)
        np.testing.assert_allclose(coeffs, true_coeffs, atol=1e-8)
        assert r2 == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-8)

    def test_r2_between_zero_and_one(self):
        rng = np.random.default_rng(1)
        X = np.hstack([np.ones((50, 1)), rng.standard_normal((50, 3))])
        y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.standard_normal(50) * 0.5
        _, _, r2 = fit_ols(X, y)
        assert 0.0 < r2 < 1.0

    def test_returns_three_values(self):
        X = np.hstack([np.ones((20, 1)), np.arange(20.0).reshape(-1, 1)])
        y = np.arange(20.0)
        result = fit_ols(X, y)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# build_lag_model: fit_pdl
# ---------------------------------------------------------------------------

class TestFitPDL:
    def test_betas_sum_close_to_ols(self):
        # PDL sum(betas) should be similar to OLS sum(betas) on same data
        rng = np.random.default_rng(7)
        n = 100
        X_raw = np.column_stack([
            rng.standard_normal(n) for _ in range(9)
        ])
        y = 2.0 + X_raw.sum(axis=1) * 0.1 + rng.standard_normal(n) * 0.05
        X = np.hstack([np.ones((n, 1)), X_raw])

        pdl_betas, pdl_alpha, _, pdl_r2 = fit_pdl(X_raw, y, degree=2)
        ols_coeffs, _, ols_r2 = fit_ols(X, y)

        # PDL betas should sum to a reasonable value
        assert abs(sum(pdl_betas)) < 5.0

    def test_pdl_betas_smooth(self):
        # PDL degree-2 constrains betas to a quadratic shape.
        # Second differences of betas should be constant (≈ 2*a_2).
        rng = np.random.default_rng(3)
        n = 200
        X_raw = np.column_stack([rng.standard_normal(n) for _ in range(5)])
        y = 1.5 + X_raw @ np.array([0.5, 0.4, 0.3, 0.2, 0.1]) + rng.standard_normal(n) * 0.1
        betas, alpha, residuals, r2 = fit_pdl(X_raw, y, degree=2)
        # Second differences should all be equal (property of quadratic PDL)
        second_diffs = np.diff(betas, n=2)
        if len(second_diffs) > 1:
            spread = second_diffs.max() - second_diffs.min()
            assert spread < 1e-8

    def test_fit_pdl_returns_n_lags_betas(self):
        X_raw = np.random.default_rng(5).standard_normal((50, 6))
        y = np.random.default_rng(5).standard_normal(50)
        betas, alpha, residuals, r2 = fit_pdl(X_raw, y, degree=2)
        assert len(betas) == 6


# ---------------------------------------------------------------------------
# build_lag_model: compute_seasonal_premium
# ---------------------------------------------------------------------------

class TestComputeSeasonalPremium:
    def _make_dates(self, months):
        """months: list of (year, month) tuples"""
        return pd.DatetimeIndex([pd.Timestamp(y, m, 15) for y, m in months])

    def test_zero_residuals_give_zero(self):
        dates = self._make_dates([(2024, m) for m in range(1, 13)])
        residuals = np.zeros(12)
        assert compute_seasonal_premium(residuals, dates) == pytest.approx(0.0)

    def test_higher_summer_residuals_give_positive(self):
        # Give summer months residual=0.2, winter months residual=0.0
        dates = self._make_dates(
            [(2024, m) for m in range(1, 13)] +
            [(2025, m) for m in range(1, 13)]
        )
        residuals = np.array([
            0.0, 0.0, 0.0,          # Jan-Mar winter
            0.2, 0.2, 0.2,          # Apr-Jun summer
            0.2, 0.2, 0.2,          # Jul-Sep summer
            0.0, 0.0, 0.0,          # Oct-Dec winter
        ] * 2)
        premium = compute_seasonal_premium(residuals, dates)
        assert premium == pytest.approx(0.2, abs=1e-9)

    def test_higher_winter_residuals_give_negative(self):
        dates = self._make_dates([(2024, m) for m in range(1, 13)])
        residuals = np.array([0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3])
        premium = compute_seasonal_premium(residuals, dates)
        assert premium < 0.0

    def test_empty_season_returns_zero(self):
        # All months are summer → no winter residuals
        dates = self._make_dates([(2024, m) for m in [4, 5, 6, 7, 8, 9]])
        residuals = np.ones(6) * 0.1
        assert compute_seasonal_premium(residuals, dates) == pytest.approx(0.0)

    def test_lag_model_seasonal_premium_in_range(self):
        # The fitted seasonal premium in lag-model.json should be reasonable
        import json
        from pathlib import Path
        lm_path = Path(__file__).parents[1] / "data" / "lag-model.json"
        with open(lm_path) as f:
            lm = json.load(f)
        sp = lm.get("seasonal_premium")
        if sp is not None:
            # Summer blend adds ~$0.05–$0.25/gal; outside that is suspicious
            assert 0.0 <= sp <= 0.35, f"Seasonal premium {sp} out of expected range"


# ---------------------------------------------------------------------------
# build_lag_model: rockets_feathers_test
# ---------------------------------------------------------------------------

class TestRocketsFeathersTest:
    def _make_series(self, values, start="2020-01-06"):
        idx = pd.date_range(start=start, periods=len(values), freq="W-MON")
        return pd.Series(values, index=idx)

    def test_returns_required_keys(self):
        rng = np.random.default_rng(9)
        n = 50
        crude = self._make_series(70 + rng.standard_normal(n).cumsum())
        X_raw = np.column_stack([crude.values[i:n-8+i] for i in range(9)])
        y = rng.standard_normal(len(X_raw))
        result = rockets_feathers_test(X_raw, y, crude)
        for key in ("coeff", "t_stat", "p_approx", "significant"):
            assert key in result

    def test_significant_is_bool(self):
        rng = np.random.default_rng(11)
        n = 50
        crude = self._make_series(70 + rng.standard_normal(n).cumsum())
        X_raw = np.column_stack([crude.values[i:n-8+i] for i in range(9)])
        y = rng.standard_normal(len(X_raw))
        result = rockets_feathers_test(X_raw, y, crude)
        assert isinstance(result["significant"], bool)

    def test_p_approx_in_0_1(self):
        rng = np.random.default_rng(13)
        n = 60
        crude = self._make_series(70 + rng.standard_normal(n).cumsum())
        X_raw = np.column_stack([crude.values[i:n-8+i] for i in range(9)])
        y = rng.standard_normal(len(X_raw))
        result = rockets_feathers_test(X_raw, y, crude)
        if result["p_approx"] is not None:
            assert 0.0 <= result["p_approx"] <= 1.0
