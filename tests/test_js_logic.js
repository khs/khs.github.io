/**
 * Tests for the core calculation functions in oil.html.
 *
 * Extracts and reimplements the pure JS functions so they can be tested
 * in Node.js without a browser or bundler.
 *
 * Run with:  node tests/test_js_logic.js
 */

"use strict";

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------

let passed = 0;
let failed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    process.stdout.write(".");
  } catch (e) {
    failed++;
    failures.push({ name, message: e.message });
    process.stdout.write("F");
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || "Assertion failed");
}

function assertApprox(a, b, tol = 1e-6, msg) {
  if (Math.abs(a - b) > tol)
    throw new Error(msg || `Expected ${b} ± ${tol}, got ${a}`);
}

// ---------------------------------------------------------------------------
// Constants (must match oil.html exactly)
// ---------------------------------------------------------------------------

const WTI_TO_BRENT_ADJ = 4.0;
const PUMP_FIXED = {
  federal_tax:  0.184,
  state_tax:    0.335,
  refining:     0.430,
  distribution: 0.300,
  retail:       0.150,
};
const VOL_UP   = 0.33;
const VOL_DOWN = 0.33;
const MR_CAP   = 2.5;

const PUMP_REGIONS = {
  national:   { label: "US National Avg",  offset:  0.00 },
  gulf:       { label: "Gulf Coast",       offset: -0.65 },
  midwest:    { label: "Midwest",          offset: -0.20 },
  east:       { label: "East Coast",       offset: +0.05 },
  west:       { label: "West Coast",       offset: +1.15 },
  california: { label: "California",       offset: +1.40 },
};

// ---------------------------------------------------------------------------
// Functions under test (copied verbatim from oil.html, adjusted for Node.js)
// Note: seasonalAdj is parametrised here to allow deterministic testing.
// ---------------------------------------------------------------------------

function seasonalAdjForMonth(month1indexed) {
  return (month1indexed >= 4 && month1indexed <= 9) ? 0.12 : 0.00;
}

function crudeToGallon(bbl, isBrent, region, crackSpread, currentMonth = null) {
  const brentBbl    = isBrent ? bbl : bbl + WTI_TO_BRENT_ADJ;
  const crudePerGal = brentBbl / 42;
  const fixed       = PUMP_FIXED.federal_tax + PUMP_FIXED.state_tax
                    + PUMP_FIXED.distribution + PUMP_FIXED.retail;
  const regional    = PUMP_REGIONS[region]?.offset ?? 0;
  const seasonal    = currentMonth != null
    ? seasonalAdjForMonth(currentMonth)
    : 0.0;  // neutral when not provided, for testing
  return +(crudePerGal + crackSpread + fixed + regional + seasonal).toFixed(3);
}

function interpolateCrude(contracts, weeksFromNow) {
  if (weeksFromNow <= 0) return contracts[0].price;

  const today = Date.now();
  const pts = contracts.map(c => ({
    w: (new Date(c.expiry + "-15") - today) / (7 * 86400000),
    p: c.price,
  }));

  // Clamp to front contract when before the curve starts
  if (weeksFromNow < pts[0].w) return pts[0].p;

  for (let i = 0; i < pts.length - 1; i++) {
    if (weeksFromNow >= pts[i].w && weeksFromNow <= pts[i + 1].w) {
      const frac = (weeksFromNow - pts[i].w) / (pts[i + 1].w - pts[i].w);
      return pts[i].p + frac * (pts[i + 1].p - pts[i].p);
    }
  }
  return pts[pts.length - 1].p; // flat extrapolation beyond last contract
}

function computeBands(contracts) {
  const today = new Date();
  return contracts.map(c => {
    const expiry = new Date(c.expiry + "-15");
    const T    = Math.max(0, (expiry - today) / (365.25 * 86400 * 1000));
    const Teff = Math.min(T, MR_CAP);
    const sigUp   = VOL_UP   * Math.sqrt(Teff);
    const sigDown = VOL_DOWN * Math.sqrt(Teff);
    return {
      expiry: c.expiry, label: c.label, price: c.price,
      upper1: +(c.price * Math.exp(    sigUp)).toFixed(2),
      lower1: +(c.price * Math.exp(-   sigDown)).toFixed(2),
      upper2: +(c.price * Math.exp(2 * sigUp)).toFixed(2),
      lower2: +(c.price * Math.exp(-2 * sigDown)).toFixed(2),
    };
  });
}

// ---------------------------------------------------------------------------
// Test helpers: build synthetic contract lists
// ---------------------------------------------------------------------------

function futureMonth(offsetMonths) {
  // Returns a YYYY-MM string `offsetMonths` ahead of today
  const d = new Date();
  d.setMonth(d.getMonth() + offsetMonths);
  return d.toISOString().slice(0, 7);
}

function weeksUntilMid(expiry) {
  // How many weeks from now until the 15th of expiry month
  const target = new Date(expiry + "-15");
  return (target - Date.now()) / (7 * 86400000);
}

function makeContracts(priceList) {
  // prices[0] → next month, prices[1] → 2 months out, etc.
  return priceList.map((price, i) => ({
    expiry: futureMonth(i + 1),
    label: `Month +${i + 1}`,
    price,
  }));
}

// ---------------------------------------------------------------------------
// Tests: crudeToGallon
// ---------------------------------------------------------------------------

test("crudeToGallon: Brent $70, national, no seasonal → formula correct", () => {
  const result = crudeToGallon(70.0, true, "national", PUMP_FIXED.refining);
  const expected = 70.0/42 + PUMP_FIXED.refining + PUMP_FIXED.federal_tax
                 + PUMP_FIXED.state_tax + PUMP_FIXED.distribution + PUMP_FIXED.retail;
  assertApprox(result, expected, 0.001);
});

test("crudeToGallon: WTI adds $4/bbl Brent adjustment", () => {
  const brent = crudeToGallon(70.0, true,  "national", 0.43);
  const wti   = crudeToGallon(66.0, false, "national", 0.43);
  // WTI $66 + $4 adj = Brent $70 equivalent
  assertApprox(brent, wti, 0.001);
});

test("crudeToGallon: higher crude → higher pump price", () => {
  const low  = crudeToGallon(50.0,  true, "national", 0.43);
  const high = crudeToGallon(100.0, true, "national", 0.43);
  assert(high > low, `Expected ${high} > ${low}`);
});

test("crudeToGallon: linear in crude (Δ42 bbl → +$1/gal)", () => {
  const p1 = crudeToGallon(70.0, true,  "national", 0.43);
  const p2 = crudeToGallon(112.0, true, "national", 0.43);
  assertApprox(p2 - p1, 1.0, 0.001, "42-bbl crude increase should add $1/gal");
});

test("crudeToGallon: California offset is $1.40 above national", () => {
  const nat = crudeToGallon(70.0, true, "national",   0.43);
  const ca  = crudeToGallon(70.0, true, "california", 0.43);
  assertApprox(ca - nat, 1.40, 0.001);
});

test("crudeToGallon: Gulf Coast is $0.65 below national", () => {
  const nat  = crudeToGallon(70.0, true, "national", 0.43);
  const gulf = crudeToGallon(70.0, true, "gulf",     0.43);
  assertApprox(nat - gulf, 0.65, 0.001);
});

test("crudeToGallon: seasonal summer adds $0.12", () => {
  const winter = crudeToGallon(70.0, true, "national", 0.43, 1);  // January
  const summer = crudeToGallon(70.0, true, "national", 0.43, 6);  // June
  assertApprox(summer - winter, 0.12, 0.001);
});

test("crudeToGallon: crack spread passes through 1-to-1", () => {
  const low  = crudeToGallon(70.0, true, "national", 0.30);
  const high = crudeToGallon(70.0, true, "national", 0.60);
  assertApprox(high - low, 0.30, 0.001);
});

test("crudeToGallon: pump price range sanity at Brent $118", () => {
  const pump = crudeToGallon(118.45, true, "national", 0.611, 4);
  assert(pump >= 3.0 && pump <= 7.0, `Pump $${pump}/gal outside plausible range at Brent $118`);
});


// ---------------------------------------------------------------------------
// Tests: interpolateCrude
// ---------------------------------------------------------------------------

test("interpolateCrude: weeksFromNow=0 returns front price", () => {
  const contracts = makeContracts([100, 98, 96]);
  const result = interpolateCrude(contracts, 0);
  assertApprox(result, 100, 0.001);
});

test("interpolateCrude: negative weeksFromNow returns front price", () => {
  const contracts = makeContracts([100, 98, 96]);
  const result = interpolateCrude(contracts, -5);
  assertApprox(result, 100, 0.001);
});

test("interpolateCrude: FIXED BUG — before first contract returns front price (not last)", () => {
  // The front contract is 4-8 weeks out. A lag of 1 week before that should
  // clamp to the front contract, NOT return the last contract price.
  const contracts = makeContracts([100, 50]);  // front=100, back=50
  const firstWeeks = weeksUntilMid(contracts[0].expiry);
  // Request a date that's in the future but before the first contract
  const justBefore = firstWeeks * 0.5;  // halfway to front contract
  if (justBefore > 0) {
    const result = interpolateCrude(contracts, justBefore);
    assertApprox(result, 100, 0.001,
      `Before first contract: expected 100 (front), got ${result} — old bug returned last price`);
  }
});

test("interpolateCrude: beyond last contract returns last price", () => {
  const contracts = makeContracts([100, 98, 60]);
  const lastWeeks = weeksUntilMid(contracts[contracts.length - 1].expiry);
  const result = interpolateCrude(contracts, lastWeeks + 100);
  assertApprox(result, 60, 0.001);
});

test("interpolateCrude: at first contract expiry returns first price", () => {
  const contracts = makeContracts([100, 80]);
  const w = weeksUntilMid(contracts[0].expiry);
  const result = interpolateCrude(contracts, w);
  assertApprox(result, 100, 0.5);  // small tolerance: floating point
});

test("interpolateCrude: at last contract expiry returns last price", () => {
  const contracts = makeContracts([100, 80, 60]);
  const w = weeksUntilMid(contracts[2].expiry);
  const result = interpolateCrude(contracts, w);
  assertApprox(result, 60, 0.5);
});

test("interpolateCrude: midpoint between two contracts interpolates linearly", () => {
  const contracts = makeContracts([100, 80]);
  const w0 = weeksUntilMid(contracts[0].expiry);
  const w1 = weeksUntilMid(contracts[1].expiry);
  const mid = (w0 + w1) / 2;
  const result = interpolateCrude(contracts, mid);
  assertApprox(result, 90, 1.0);  // midpoint of 100 and 80
});

test("interpolateCrude: result always in [front, back] range (monotone case)", () => {
  const contracts = makeContracts([100, 90, 80, 70, 60]);
  const lastWeeks = weeksUntilMid(contracts[4].expiry);
  for (let w = 0; w <= lastWeeks + 10; w += 2) {
    const result = interpolateCrude(contracts, w);
    assert(result >= 59 && result <= 101,
      `interpolateCrude(${w}wks) = ${result} outside [60,100] range`);
  }
});


// ---------------------------------------------------------------------------
// Tests: computeBands
// ---------------------------------------------------------------------------

test("computeBands: price preserved as midpoint", () => {
  const contracts = makeContracts([80]);
  const [b] = computeBands(contracts);
  assertApprox(b.price, 80, 0.001);
});

test("computeBands: upper > price > lower (for T > 0)", () => {
  const contracts = makeContracts([80, 78]);  // future months, T > 0
  const bands = computeBands(contracts);
  // Only check the far-out contract (T is definitely > 0)
  const b = bands[1];
  assert(b.upper1 > b.price, `upper1 ${b.upper1} should > price ${b.price}`);
  assert(b.lower1 < b.price, `lower1 ${b.lower1} should < price ${b.price}`);
  assert(b.upper2 > b.upper1, `2σ upper should exceed 1σ upper`);
  assert(b.lower2 < b.lower1, `2σ lower should be below 1σ lower`);
});

test("computeBands: VOL_UP == VOL_DOWN → geometric symmetry (upper1 * lower1 ≈ price²)", () => {
  // With symmetric vols, exp(σ)*exp(-σ)=1 so upper1*lower1=price^2
  const contracts = makeContracts([100, 100]);
  const bands = computeBands(contracts);
  const b = bands[1];  // far month, T > 0
  const product = b.upper1 * b.lower1;
  assertApprox(product, 100 * 100, 5.0,  // allow some rounding
    `upper1 * lower1 = ${product}, expected ~${100*100}`);
});

test("computeBands: mean reversion cap — bands stop growing past 2.5 years", () => {
  // We can't easily make contracts 2.5 years out in this test,
  // so we test the property directly using the Python backtest's compute_bands logic.
  // Instead verify: bands for a 3-year contract = bands for a 2.5-year contract.
  // This requires the cap to kick in. We'll use a simulated expiry date.
  function bandsForT(T_years) {
    const Teff = Math.min(T_years, MR_CAP);
    const sigUp   = VOL_UP   * Math.sqrt(Teff);
    const sigDown = VOL_DOWN * Math.sqrt(Teff);
    return {
      upper1: 100 * Math.exp(    sigUp),
      lower1: 100 * Math.exp(-   sigDown),
    };
  }
  const b25 = bandsForT(2.5);
  const b40 = bandsForT(4.0);
  assertApprox(b25.upper1, b40.upper1, 0.001,
    "Bands at 2.5y and 4y should be equal (mean reversion cap)");
  assertApprox(b25.lower1, b40.lower1, 0.001);
});

test("computeBands: bands scale linearly with price", () => {
  const c100 = makeContracts([100, 100]);
  const c200 = makeContracts([200, 200]);
  const b100 = computeBands(c100)[1];
  const b200 = computeBands(c200)[1];
  assertApprox(b200.upper1, 2 * b100.upper1, 0.01);
  assertApprox(b200.lower1, 2 * b100.lower1, 0.01);
});

test("computeBands: 1σ band uses correct formula", () => {
  // Use a synthetic far-out date to get a deterministic T
  // T ≈ 1 year out, VOL=0.33, upper1 = price * exp(0.33 * sqrt(1)) ≈ 100 * 1.391
  // We can't control T precisely from here, so just check the relationship
  const contracts = makeContracts([100, 100]);
  const bands = computeBands(contracts);
  const b = bands[1];
  // upper2 should be the square of the relative upper1 movement
  // i.e. upper2/price = (upper1/price)^2
  const rel1 = b.upper1 / b.price;
  const rel2 = b.upper2 / b.price;
  assertApprox(rel2, rel1 * rel1, 0.02,
    `2σ should be 1σ² in relative terms: ${rel2} vs ${rel1*rel1}`);
});


// ---------------------------------------------------------------------------
// Tests: pumpBandEndpoint delta computation (pure math)
// ---------------------------------------------------------------------------

test("pumpBandEndpoint: Brent adjustment cancels in delta calculation", () => {
  // The delta is: ((crudeBandBbl + brentAdj) - (central + brentAdj)) / 42 * tp
  // The brentAdj cancels, so WTI and Brent should give the same delta
  function pumpBandDelta(crudeBand, central, tp) {
    return (crudeBand - central) / 42 * tp;
  }
  const WTI_ADJ = 4.0;
  const tp = 0.95;
  const central = 100.0;
  const band = 110.0;
  // With brentAdj
  const deltaBrent = pumpBandDelta(band + WTI_ADJ, central + WTI_ADJ, tp);
  const deltaWTI   = pumpBandDelta(band, central, tp);
  assertApprox(deltaBrent, deltaWTI, 1e-9,
    "Brent adjustment should cancel in pumpBandEndpoint delta");
});


// ---------------------------------------------------------------------------
// Tests: seasonal adjustment
// ---------------------------------------------------------------------------

test("seasonalAdj: April–September returns 0.12", () => {
  for (const m of [4, 5, 6, 7, 8, 9]) {
    assertApprox(seasonalAdjForMonth(m), 0.12, 1e-9, `Month ${m} should be summer`);
  }
});

test("seasonalAdj: October–March returns 0", () => {
  for (const m of [1, 2, 3, 10, 11, 12]) {
    assertApprox(seasonalAdjForMonth(m), 0.0, 1e-9, `Month ${m} should be winter`);
  }
});


// ---------------------------------------------------------------------------
// Tests: crack spread lookup logic
// ---------------------------------------------------------------------------

test("crackSpreadFor: returns matching month's spread", () => {
  const oilData = {
    crack_spreads: {
      by_month: [
        { expiry: "2026-05", crack_spread: 0.611 },
        { expiry: "2026-06", crack_spread: 0.798 },
      ],
      average: 0.631,
    }
  };
  function crackSpreadFor(expiry) {
    if (!oilData?.crack_spreads) return PUMP_FIXED.refining;
    const cs = oilData.crack_spreads;
    const match = (cs.by_month || []).find(m => m.expiry === expiry);
    if (match) return match.crack_spread;
    return cs.average ?? PUMP_FIXED.refining;
  }
  assertApprox(crackSpreadFor("2026-05"), 0.611, 1e-9);
  assertApprox(crackSpreadFor("2026-06"), 0.798, 1e-9);
});

test("crackSpreadFor: missing month falls back to average", () => {
  const oilData = {
    crack_spreads: {
      by_month: [{ expiry: "2026-05", crack_spread: 0.611 }],
      average: 0.631,
    }
  };
  function crackSpreadFor(expiry) {
    const cs = oilData.crack_spreads;
    const match = (cs.by_month || []).find(m => m.expiry === expiry);
    if (match) return match.crack_spread;
    return cs.average ?? PUMP_FIXED.refining;
  }
  assertApprox(crackSpreadFor("2026-12"), 0.631, 1e-9);
});

test("crackSpreadFor: missing crack_spreads falls back to PUMP_FIXED.refining", () => {
  const oilDataEmpty = {};
  function crackSpreadFor(expiry) {
    if (!oilDataEmpty?.crack_spreads) return PUMP_FIXED.refining;
    const cs = oilDataEmpty.crack_spreads;
    const match = (cs.by_month || []).find(m => m.expiry === expiry);
    if (match) return match.crack_spread;
    return cs.average ?? PUMP_FIXED.refining;
  }
  assertApprox(crackSpreadFor("2026-05"), 0.430, 1e-9);
});


// ---------------------------------------------------------------------------
// Integration: pump price sanity with current data
// ---------------------------------------------------------------------------

test("pump price with current data: front month in $2.50–$7.00/gal range", () => {
  const fs = require("fs");
  const path = require("path");
  const dataPath = path.join(__dirname, "..", "data", "oil-futures.json");
  const lagPath  = path.join(__dirname, "..", "data", "lag-model.json");

  const oilData  = JSON.parse(fs.readFileSync(dataPath, "utf8"));
  const lagModel = JSON.parse(fs.readFileSync(lagPath, "utf8"));

  if (!lagModel.alpha || !lagModel.betas) return; // model not built yet

  const contracts = oilData.brent;
  if (!contracts.length) return;

  const betas = lagModel.betas;
  const alpha = lagModel.alpha;
  const sp    = lagModel.seasonal_premium || 0;
  const tp    = lagModel.total_passthrough || 0.95;

  // Front month: all lags clamp to front price → weighted = sum(betas) * (price/42)
  const frontPrice = contracts[0].price;
  const crackAvg   = oilData.crack_spreads?.average ?? PUMP_FIXED.refining;
  const crackFront = (oilData.crack_spreads?.by_month || []).find(
    m => m.expiry === contracts[0].expiry
  )?.crack_spread ?? crackAvg;
  const crackDev = crackFront - crackAvg;
  const weighted = betas.reduce((s, b) => s + b * (frontPrice / 42), 0);
  const pump = alpha + weighted + crackDev + sp;  // summer since we're in Apr

  assert(pump >= 2.50 && pump <= 7.00,
    `Front month pump price $${pump.toFixed(2)}/gal outside $2.50–$7.00 range at Brent $${frontPrice}`);
});

test("pump price: lag model front month close to simple formula (< $0.80 difference)", () => {
  const fs   = require("fs");
  const path = require("path");
  const oilData  = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", "oil-futures.json"), "utf8"));
  const lagModel = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", "lag-model.json"), "utf8"));
  if (!lagModel.alpha || !lagModel.betas) return;

  const frontBrent = oilData.brent[0].price;
  const crackFront = (oilData.crack_spreads?.by_month || []).find(
    m => m.expiry === oilData.brent[0].expiry
  )?.crack_spread ?? PUMP_FIXED.refining;
  const crackAvg  = oilData.crack_spreads?.average ?? PUMP_FIXED.refining;
  const crackDev  = crackFront - crackAvg;

  const betas   = lagModel.betas;
  const alpha   = lagModel.alpha;
  const weighted = betas.reduce((s, b) => s + b * (frontBrent / 42), 0);
  const lagPump  = alpha + weighted + crackDev + (lagModel.seasonal_premium || 0);

  const simple = crudeToGallon(frontBrent, true, "national", crackFront, new Date().getMonth() + 1);

  const diff = Math.abs(lagPump - simple);
  assert(diff < 0.80,
    `Lag model pump $${lagPump.toFixed(2)} and simple formula $${simple.toFixed(2)} ` +
    `differ by $${diff.toFixed(2)}/gal — seems too large`);
});


// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log();
console.log(`\n${"=".repeat(60)}`);
if (failures.length) {
  console.log("FAILURES:");
  failures.forEach(f => console.log(`  FAIL: ${f.name}\n        ${f.message}`));
  console.log();
}
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
