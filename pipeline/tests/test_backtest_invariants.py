"""Invariants for the walk-forward test.

SYNTHETIC INPUT, unlike the rest of this suite -- and for the same reason
`test_universe_invariants.py` writes a broken universe to `tmp_path`. Every other test file
here reads the shipped artifacts because that makes them fail when the DATA is wrong. This
file cannot: the properties that matter are things the shipped panel does not exercise. It
holds 116 assets that all trade on the same 3,941 days with no gaps, so it cannot distinguish
a monthly rebalance from a buy-and-hold at the level of "did the code reset anything", and it
certainly cannot answer the only question that decides whether `backtest.py` means anything
at all -- whether the weights depend on data from after the cutoff.

THE LOOKAHEAD TEST IS THE POINT OF THE FILE. A backtest with a leak produces numbers that are
plausible, ordered the way you expect, and completely worthless, and no assertion about a
return, a Sharpe ratio or a rank correlation can see it. The only test that can is to change
the future and require the decision not to move.
"""
import json

import numpy as np
import pandas as pd
import pytest

import backtest as bt
import fetch


# ------------------------------------------------------------------------------- fixtures

def _panel(n_assets: int = 6, n_bars: int = 1300, seed: int = 7) -> pd.DataFrame:
    """A total-return panel: business days, positive prices, a spread of drifts and vols so
    the assets are actually rankable and the optimiser has something to choose between."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n_bars, name="date")
    drift = np.linspace(0.00002, 0.00060, n_assets)
    vol = np.linspace(0.004, 0.020, n_assets)
    steps = rng.standard_normal((n_bars, n_assets)) * vol + drift
    prices = 100.0 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def _irx(index: pd.DatetimeIndex) -> pd.Series:
    """A risk-free series that MOVES, so `rf_asof` and `rf_mean_over_test` cannot coincide by
    accident and a rate taken from the wrong end of a window is a different number."""
    return pd.Series(0.02 + 0.02 * np.sin(np.linspace(0, 3.0, len(index))), index=index, name="rf")


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def irx(panel):
    return _irx(panel.index)


# The rolling tests need a longer panel than the rest: a 3-year span in 1-year steps with a
# 2-year training window reaches 5 years back from the end, and 1,300 business days is 5.2
# years in total -- the first training window would start before the panel does. 2,000 bars is
# ~7.9 years, which leaves room for the windows to be wrong in a way the assertion can see.
@pytest.fixture(scope="module")
def long_panel():
    return _panel(n_bars=2000, seed=11)


@pytest.fixture(scope="module")
def long_irx(long_panel):
    return _irx(long_panel.index)


# ------------------------------------------------------------------ the no-lookahead guard

@pytest.mark.parametrize("model", [bt.MuModel(), bt.MuModel(kind="momentum")],
                         ids=["history", "momentum"])
def test_the_solved_weights_do_not_depend_on_a_single_bar_after_the_cutoff(panel, irx, model):
    """Change the future; the decision must not move.

    The perturbation is applied STRICTLY AFTER the cutoff, leaving the cutoff bar itself
    alone -- it is the last training bar and the entry price, so moving it is a legitimate
    change of input. Everything after it is information the investor did not have.

    BOTH INPUTS ARE PERTURBED, in two separate runs, because the tangency portfolio has two
    of them and only one is prices. `rf` moves the point where the capital market line touches
    the frontier, so a rate read from the wrong end of the window is a leak that changes the
    weights while every price in the training half stays exactly as it was. Perturbing the
    panel alone left `backtest-lookahead-rf-from-the-end-of-the-window` alive in the mutation
    harness: `panel.index.max()` is not moved by rescaling prices, so the mutant returned
    identical weights and this test passed.

    Two halves of each assertion are load-bearing. The weights must be IDENTICAL (bit for
    bit, not approximately -- the same inputs through the same solver give the same answer,
    and a tolerance here would hide a small leak). Something downstream must DIFFER, or the
    test would also pass on a `run_cutoff` that ignored the test window entirely, which is
    the one other way to make the weights invariant.

    RUN UNDER BOTH EXPECTED-RETURN MODELS, because they read the training window differently and
    a leak-free one says nothing about the other. `"history"` averages every bar up to the
    cutoff; momentum reads exactly two bars and locates them by date arithmetic from the cutoff,
    which is a second, independent way to end up one month on the wrong side of it.
    """
    def decisions(p, r):
        # WITH a benchmark, so the matched-risk and levered rows are in the comparison. Their
        # risk target and leverage factor are decisions too, and a target read off the test
        # window is a leak that leaves every weight vector untouched -- so `leverage` and
        # `risk_target` have to be in the tuple or those two rows are guarded by nothing.
        out = bt.run_cutoff(p, 1, r, [1.0], "A3", None, 8, model)
        return out, [(s["strategy"], s["top_holdings"], s["predicted"],
                      s.get("leverage"), s.get("risk_target")) for s in out["strategies"]]

    base, base_w = decisions(panel, irx)
    cutoff = pd.Timestamp(base["cutoff"])
    future = panel.index > cutoff
    assert future.sum() > 60

    # (1) the future PRICES change, including their order -- A0 goes from worst to best.
    shifted = panel.copy()
    shifted.loc[future] *= 1.5
    shifted.loc[future, "A0"] *= 4.0
    priced, priced_w = decisions(shifted, irx)
    assert priced["cutoff"] == base["cutoff"]
    assert priced_w == base_w
    assert base["mu_rank_correlation"] != priced["mu_rank_correlation"]
    assert all(a["realized_monthly"]["growth"] != b["realized_monthly"]["growth"]
               for a, b in zip(base["strategies"], priced["strategies"])), \
        "the test window changed and nothing moved"

    # (2) the future RISK-FREE RATE changes. Doubled, so the tangency portfolio would move a
    # long way if the solve could see it, and the cutoff's own quote is left untouched.
    later = irx.copy()
    later.loc[future] *= 2.0
    rated, rated_w = decisions(panel, later)
    assert rated["rf_at_cutoff"] == base["rf_at_cutoff"]
    assert rated_w == base_w
    assert rated["rf_mean_over_test"] != base["rf_mean_over_test"], "the rf series did not move"


def test_the_cutoff_bar_belongs_to_both_halves_and_is_counted_once_each(panel, irx):
    """`train` ends on the cutoff and `test` starts on it, so the bar counts overlap by
    exactly one. That shared bar is the entry price: it is the last return in training and
    the zeroth (returnless) row of the test path, and it must not be double-counted as a
    return or dropped from either side."""
    r = bt.run_cutoff(panel, 1, irx, [0.2], None, None, 8)
    assert r["train"]["bars"] + r["test"]["bars"] == len(panel) + 1
    assert r["train"]["years"] == pytest.approx((r["train"]["bars"] - 1) / 252.0, abs=1e-3)
    assert r["test"]["years"] == pytest.approx((r["test"]["bars"] - 1) / 252.0, abs=1e-3)


def test_the_cutoff_is_a_trading_day_at_or_before_the_requested_date(panel, irx):
    for years_back in (1, 2, 3):
        r = bt.run_cutoff(panel, years_back, irx, [0.2], None, None, 8)
        cutoff = pd.Timestamp(r["cutoff"])
        assert cutoff in panel.index
        assert cutoff <= panel.index.max() - pd.DateOffset(years=years_back)


def test_evaluate_refuses_a_test_window_that_does_not_begin_at_the_cutoff(panel, irx):
    """The one structural assumption `evaluate` makes about its two arguments, checked rather
    than documented: a test window starting a bar late loses the entry price and the value path
    is then indexed to the wrong day, which shifts every return by one bar and nothing else."""
    cutoff = panel.index[800]
    train = panel.loc[:cutoff]
    with pytest.raises(ValueError, match="not at the cutoff"):
        bt.evaluate(train, panel.loc[panel.index > cutoff], irx, [0.2], None, None, 8)


# ------------------------------------------------------------------- the rolling walk-forward

@pytest.mark.parametrize("hold_years,span_years,n", [(1.0, 3.0, 3), (2.0, 4.0, 2)])
def test_rolling_holding_periods_are_laid_end_to_end_and_never_overlap(
    long_panel, long_irx, hold_years, span_years, n
):
    """The property that makes N periods N observations instead of one. Consecutive periods
    touch at a single shared bar -- one period's last day is the next one's entry -- and never
    share a return. Overlapping them is the defect the nested `--years 1,2,3` mode has by
    construction, and the reason this mode exists.

    TESTED AT TWO HOLDING LENGTHS, and the second one is not thoroughness. The cutoffs step by
    `hold_years * k`; at a one-year hold that expression equals `k`, so a version that stepped
    by `k` regardless of the holding period would be identical here and produce overlapping
    two-year periods in production. `backtest-rolling-periods-overlap` survived the one-year
    case for exactly that reason.
    """
    periods = bt.run_rolling(long_panel, 2.0, hold_years, span_years, long_irx,
                             [0.2], None, None, 8)
    assert len(periods) == n
    ends = []
    for p in periods:
        start, end = (pd.Timestamp(x) for x in p["period"].split(" to "))
        assert start < end
        if ends:
            assert start >= ends[-1], f"{p['period']} overlaps the period before it"
        ends.append(end)
        assert pd.Timestamp(p["cutoff"]) == start


def test_a_rolling_training_window_is_fixed_length_and_ends_at_the_cutoff(long_panel, long_irx):
    """FIXED length, not everything-so-far -- the difference from `run_cutoff`, whose training
    window grows with every step. A one-character edit here turns one into the other.

    Asserted on DATES and on consistency between periods, not against `2.0` in the `years`
    field. That field divides bars by 252, and `bdate_range` has no holidays in it, so a
    two-calendar-year window of synthetic business days measures 2.07 by that convention.
    Which is a fact about the fixture, not about the window -- the window's actual claim is
    "it starts two calendar years before the cutoff", and that is exact.
    """
    periods = bt.run_rolling(long_panel, 2.0, 1.0, 3.0, long_irx, [0.2], None, None, 8)
    bars = []
    for p in periods:
        cutoff = pd.Timestamp(p["cutoff"])
        target = cutoff - pd.DateOffset(years=2)
        start = pd.Timestamp(p["train"]["start"])
        # `_bar_at_or_before`, so at or just before the target and never after it.
        assert target - pd.Timedelta(days=7) < start <= target, p["period"]
        bars.append(p["train"]["bars"])
    # A window that grew instead of rolling would gain a year of bars per step.
    assert max(bars) - min(bars) < 10, f"training windows are not the same length: {bars}"


def test_a_rolling_period_cannot_see_past_its_own_cutoff(long_panel, long_irx):
    """The no-lookahead guard again, on the other driver, and it has to be its own test: the
    rolling path chooses its training window differently, so one leak-free driver says nothing
    about the other.

    Perturbed after the LAST cutoff, which is the only span no period is allowed to train on.
    Every earlier cutoff has legitimate training data after it -- that is what walking forward
    means -- so a perturbation there would move later weights for the right reason and the test
    would be asserting the opposite of the property."""
    base = bt.run_rolling(long_panel, 2.0, 1.0, 3.0, long_irx, [0.2], None, None, 8)
    last_cutoff = pd.Timestamp(base[-1]["cutoff"])

    shifted = long_panel.copy()
    shifted.loc[long_panel.index > last_cutoff] *= 0.5
    shifted.loc[long_panel.index > last_cutoff, "A5"] *= 6.0
    lied = bt.run_rolling(shifted, 2.0, 1.0, 3.0, long_irx, [0.2], None, None, 8)

    for a, b in zip(base, lied):
        assert a["period"] == b["period"]
        for sa, sb in zip(a["strategies"], b["strategies"]):
            assert sa["top_holdings"] == sb["top_holdings"], (a["period"], sa["strategy"])
    assert (base[-1]["strategies"][0]["realized_monthly"]["growth"]
            != lied[-1]["strategies"][0]["realized_monthly"]["growth"])


def test_a_rolling_span_that_is_not_at_least_two_periods_is_refused(long_panel, long_irx):
    """One period is the nested mode with extra steps. The whole claim of this mode is that the
    periods can disagree with each other, and one cannot."""
    with pytest.raises(ValueError, match="period"):
        bt.run_rolling(long_panel, 2.0, 3.0, 3.0, long_irx, [0.2], None, None, 8)


def test_a_split_too_thin_to_measure_is_refused(panel, irx):
    """Both refusals, because they are different failures with different causes. A cutoff at
    the panel's own end leaves a test window of one bar (nothing to measure); a cutoff before
    the panel starts leaves no cutoff at all. Either one silently returning a result would be
    a table row computed from a handful of days."""
    with pytest.raises(ValueError, match="too thin"):
        bt.run_cutoff(panel, 0, irx, [0.2], None, None, 8)
    with pytest.raises(ValueError, match="before the panel starts"):
        bt.run_cutoff(panel, 20, irx, [0.2], None, None, 8)


# ------------------------------------------------------------------------------ rf_asof

def test_rf_asof_never_returns_a_rate_from_the_future(irx):
    """The last quote AT OR BEFORE the date. Tested against a rate that moves and on a date
    with no quote of its own, because `asof` and "nearest" agree whenever the date is a
    quote date -- which is every date the shipped series would be asked about."""
    dates = irx.index
    assert bt.rf_asof(irx, dates[100]) == irx.iloc[100]

    gap = dates[100] + pd.Timedelta(days=1)          # a Saturday or a missing bar
    while gap in dates:
        gap += pd.Timedelta(days=1)
    got = bt.rf_asof(irx, gap)
    assert got == float(irx[dates <= gap].iloc[-1])
    assert got != float(irx[dates > gap].iloc[0])

    with pytest.raises(ValueError, match="no .* quote at or before"):
        bt.rf_asof(irx, dates[0] - pd.Timedelta(days=1))


def test_the_realized_sharpe_uses_the_test_window_rate_not_the_cutoff_rate(panel, irx):
    """Two different rates in the output, and they must not be the same number by accident:
    charging the realized return the cutoff's hurdle would be a real error whenever rates
    move, which is whenever the window is long enough to be interesting."""
    r = bt.run_cutoff(panel, 2, irx, [0.2], None, None, 8)
    assert r["rf_at_cutoff"] != r["rf_mean_over_test"]
    assert r["rf_at_cutoff"] == pytest.approx(bt.rf_asof(irx, pd.Timestamp(r["cutoff"])), abs=5e-7)

    s = r["strategies"][0]
    m = s["realized_monthly"]
    assert m["sharpe"] == pytest.approx((m["ret"] - r["rf_mean_over_test"]) / m["vol"], abs=1e-3)


def test_an_rf_override_pins_both_columns_and_ignores_the_series(panel, irx):
    r = bt.run_cutoff(panel, 1, irx, [0.2], None, 0.037, 8)
    assert r["rf_at_cutoff"] == 0.037
    assert r["rf_mean_over_test"] == 0.037


# --------------------------------------------------------------------------- value_path

def test_holding_one_asset_reproduces_that_assets_own_total_return(panel):
    w = np.array([1.0, 0, 0, 0, 0, 0])
    p = panel["A0"]
    for rule in ("hold", "monthly"):
        v = bt.value_path(panel, w, rule)
        # A one-asset portfolio has nothing to rebalance, so BOTH rules must be the price
        # relative exactly. This is also the arithmetic check on the monthly branch's
        # base_row/base_val bookkeeping: any error there shows up as drift here.
        assert np.allclose(v.to_numpy(), (p / p.iloc[0]).to_numpy(), rtol=0, atol=1e-12)


def test_monthly_rebalancing_actually_resets_the_weights_at_the_month_boundary():
    """A hand-computed case, because "monthly" is otherwise indistinguishable from "hold"
    on any panel where the assets move together.

    Two assets, 50/50, A = 1 -> 2 -> 3 -> 3 and B = 1 -> 1 -> 1 -> 2, with the month boundary
    between bars 1 and 2. Buy-and-hold ends at 0.5*3 + 0.5*2 = 2.5. Rebalanced at the CLOSE of
    January the portfolio is worth 1.5, splits it evenly, and ends at
    1.5 * (0.5*(3/2) + 0.5*(2/1)) = 2.625.

    A MOVES ON THE FIRST BAR OF FEBRUARY (A goes 2 -> 3 there) and that is the reason the
    fixture has four bars rather than three. Rebalancing on the first bar of the new month
    instead of the last bar of the old one -- the drift leak `value_path`'s comment warns
    about -- gives 2.25, but only if some asset actually moves on that bar; with a flat
    February open the two conventions coincide and the test would pass on either.
    """
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-31", "2020-02-03", "2020-02-28"], name="date")
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0, 3.0], "B": [1.0, 1.0, 1.0, 2.0]}, index=idx)
    w = np.array([0.5, 0.5])

    assert bt.value_path(prices, w, "hold").iloc[-1] == pytest.approx(2.5, abs=1e-12)
    assert bt.value_path(prices, w, "monthly").iloc[-1] == pytest.approx(2.625, abs=1e-12)


def test_the_monthly_path_is_not_the_daily_rebalanced_weighted_sum_of_returns():
    """The trap `value_path`'s docstring names. `sum_i w_i * r_i` compounded daily is the
    CONTINUOUSLY rebalanced portfolio, a different number from either rule here, and it is
    the shape a "simplification" of this function would take.

    Hand-computed on purpose, and NOT on the random panel: measured there the gap is only
    0.1% of terminal value, which is a difference a broken implementation could hide inside.
    Both moves sit inside one month, so no boundary is crossed and hold and monthly must
    agree at 0.5*2 + 0.5*2 = 2.0, while daily rebalancing captures A's jump, splits the
    proceeds, and rides B's jump with 1.5 times the money: 1.5 * 1.5 = 2.25.
    """
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-15", "2020-01-31"], name="date")
    prices = pd.DataFrame({"A": [1.0, 2.0, 2.0], "B": [1.0, 1.0, 2.0]}, index=idx)
    w = np.array([0.5, 0.5])

    daily = float((1.0 + prices.pct_change().dropna() @ w).prod())
    assert daily == pytest.approx(2.25, abs=1e-12)
    assert float(bt.value_path(prices, w, "hold").iloc[-1]) == pytest.approx(2.0, abs=1e-12)
    assert float(bt.value_path(prices, w, "monthly").iloc[-1]) == pytest.approx(2.0, abs=1e-12)


def test_value_path_starts_at_one_and_refuses_a_rule_it_does_not_implement(panel):
    w = np.full(panel.shape[1], 1.0 / panel.shape[1])
    for rule in ("hold", "monthly"):
        # `approx` and not `== 1.0`: the hold branch computes the first row as `(P[0]/P[0]) @ w`,
        # which is six copies of 1/6 summed in float and lands one ulp low. The property is
        # "the path is indexed to 1 at the entry bar", not bit equality with the literal.
        assert bt.value_path(panel, w, rule).iloc[0] == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="rebalance must be"):
        bt.value_path(panel, w, "quarterly")


# ----------------------------------------------------------------------------- realized

def test_annualisation_round_trips_at_252_over_returns_not_bars():
    """A path growing at exactly 10% a year for two years must report 0.10 -- not 0.0999
    (annualising over `len(v)` bars instead of `len(v) - 1` returns) and not 0.0999-ish from
    365. Both are one-character edits away and both look right at three decimals over a
    short window, which is why the window here is long enough for them to separate."""
    n = 504
    v = pd.Series(1.10 ** (np.arange(n + 1) / fetch.TRADING_DAYS_PER_YEAR))
    r = bt.realized(v, rf=0.0)
    assert r["ret"] == pytest.approx(0.10, abs=1e-9)
    assert r["growth"] == pytest.approx(1.21, abs=1e-6)
    # Monotone up, so there is no drawdown at all and no vol beyond float noise.
    assert r["max_drawdown"] == 0.0


def test_max_drawdown_is_a_positive_fraction_of_the_peak(panel):
    """Positive, as `export.ts` ships it: a bare drawdown is ambiguous in sign once it
    leaves the page. Measured from the running PEAK, not from the start -- a path that rises
    then falls back to its starting value has drawn down, and a naive `min(v) - 1` says it
    has not."""
    v = pd.Series([1.0, 1.25, 1.0, 1.10, 2.0])
    r = bt.realized(v, rf=0.0)
    assert r["max_drawdown"] == pytest.approx(0.20, abs=1e-9)  # 1.25 -> 1.00
    assert r["max_drawdown"] > 0

    real = bt.realized(bt.value_path(panel, np.full(panel.shape[1], 1.0 / panel.shape[1]),
                                    "monthly"), rf=0.02)
    assert 0.0 < real["max_drawdown"] < 1.0


def test_a_window_of_one_return_is_refused_rather_than_annualised():
    with pytest.raises(ValueError, match="measures nothing"):
        bt.realized(pd.Series([1.0, 1.01]), rf=0.0)


# --------------------------------------------------------------------------- strategies

def test_every_strategy_is_a_feasible_long_only_portfolio_respecting_its_cap(panel):
    est = bt.fr.estimate(panel)
    caps = [1.0, 0.34, 0.2]
    for label, cap, w in bt.strategies(est, caps, rf=0.03, benchmark="A0"):
        assert w.shape == (panel.shape[1],)
        assert w.sum() == pytest.approx(1.0, abs=1e-6), label
        assert (w >= -1e-9).all(), label
        if cap is not None:
            assert w.max() <= cap + 1e-6, label
    labels = [s[0] for s in bt.strategies(est, caps, rf=0.03, benchmark="A0")]
    # The comparators must be there. An optimiser reported without 1/N and the benchmark
    # beside it is a number with nothing to be better than.
    assert "equal_weight" in labels and "benchmark:A0" in labels
    for cap in caps:
        slug = bt.build.cap_slug(cap)
        assert f"tangency@{slug}" in labels and f"min_variance@{slug}" in labels
    # `matched_risk` is per-cap AND optional -- a tight cap can put the benchmark's volatility
    # out of reach, and omitting the row is the honest answer. So the count is bounded rather
    # than fixed, and the bounds are asserted so an extra unnamed row still fails.
    assert 2 * len(caps) + 2 <= len(labels) <= 3 * len(caps) + 2
    assert set(labels) - {"equal_weight", "benchmark:A0"} <= {
        f"{k}@{bt.build.cap_slug(c)}" for c in caps
        for k in ("tangency", "min_variance", "matched_risk")
    }

    # WITH the momentum comparator, which is the only row whose presence depends on an argument
    # rather than on the data. It must be exactly one more label, and it must be a portfolio on
    # the same terms as the rest -- long-only and summing to one.
    with_mom = bt.strategies(est, caps, rf=0.03, benchmark="A0",
                             momentum=bt.estimates(panel, bt.MuModel())[1], mom_top=3)
    mom_labels = [s[0] for s in with_mom]
    assert set(mom_labels) - set(labels) == {"momentum_top3"}
    for label, cap, w in with_mom:
        assert w.sum() == pytest.approx(1.0, abs=1e-6), label
        assert (w >= -1e-9).all(), label


def test_a_benchmark_outside_the_panel_is_dropped_rather_than_faked(panel):
    est = bt.fr.estimate(panel)
    labels = [s[0] for s in bt.strategies(est, [0.2], rf=0.03, benchmark="SPY")]
    assert not any(l.startswith("benchmark:") for l in labels)


def test_the_predicted_columns_are_the_training_estimate_and_nothing_else(panel, irx):
    """`predicted` must be `frontier.performance` on the TRAINING estimates -- the same
    function the shipped frontier reports with, so "predicted Sharpe" cannot mean two
    things in one repo."""
    r = bt.run_cutoff(panel, 1, irx, [0.2], None, None, 8)
    train = panel.loc[:pd.Timestamp(r["cutoff"])]
    est = bt.fr.estimate(train)
    w = bt.fr._solve(est.mu, est.cov, 0.2, "max_sharpe", risk_free_rate=r["rf_at_cutoff"])
    assert r["strategies"][0]["strategy"] == "tangency@cap20"
    assert r["strategies"][0]["predicted"] == bt.predicted(w, est, r["rf_at_cutoff"])


def test_the_risk_target_is_the_benchmarks_forecast_volatility_not_its_realized_one(panel, irx):
    """The whole point of the matched-risk comparison, and the one place it could cheat.

    `matched_risk` and `tangency_levered` both aim at a volatility. If that number came from the
    TEST window -- the benchmark's realized volatility, which is the obviously fairer target --
    then the size of the position would be chosen with knowledge of the period it is measured
    over, and the comparison would be rigged in a way that reads as scrupulous fairness. So the
    target has to be the training window's forecast, and this pins it to that exact number: the
    diagonal of the shrunk covariance the optimiser itself was handed.
    """
    r = bt.run_cutoff(panel, 1, irx, [1.0], "A3", None, 8)
    train = panel.loc[:pd.Timestamp(r["cutoff"])]
    test = panel.loc[pd.Timestamp(r["cutoff"]):]
    est = bt.fr.estimate(train)
    target = bt.predicted_vol(est, "A3")

    bench = next(s for s in r["strategies"] if s["strategy"] == "benchmark:A3")
    assert bench["predicted"]["vol"] == pytest.approx(target, abs=5e-7)

    realized_bench_vol = test["A3"].pct_change().dropna().std(ddof=1) * np.sqrt(
        fetch.TRADING_DAYS_PER_YEAR)
    assert abs(realized_bench_vol - target) > 1e-3, \
        "the fixture cannot separate the forecast target from the realized one"

    mr = next(s for s in r["strategies"] if s["strategy"] == "matched_risk@cap100")
    lv = next(s for s in r["strategies"] if s["strategy"] == "tangency_levered@cap100")
    for row in (mr, lv):
        assert row["predicted"]["vol"] == pytest.approx(target, abs=1e-5), row["strategy"]
        assert row["predicted"]["vol"] != pytest.approx(realized_bench_vol, abs=1e-4), row["strategy"]
    assert lv["risk_target"] == pytest.approx(target, abs=5e-7)


def test_leverage_scales_the_excess_return_and_the_risk_by_the_same_factor(panel, irx):
    """The identity that makes the comparison fair: `k` moves a portfolio ALONG the capital
    market line, so the predicted Sharpe ratio is unchanged. If the levered row's predicted
    Sharpe ever differs from the tangency row's, the leverage is not being applied to the
    excess return -- charging the borrowing to the total return instead of the excess is the
    natural way to get that wrong, and it makes leverage look free."""
    r = bt.run_cutoff(panel, 1, irx, [1.0], "A3", None, 8)
    tan = next(s for s in r["strategies"] if s["strategy"] == "tangency@cap100")
    lv = next(s for s in r["strategies"] if s["strategy"] == "tangency_levered@cap100")
    rf = r["rf_at_cutoff"]

    # UNROUNDED, recomputed. `leverage` ships at 4dp and `predicted.vol` at 6, so reconstructing
    # the identity from the shipped fields fails on correct code by ~5e-6 -- the third of this
    # repo's rounding traps. The rounding is checked below, at its own precision.
    est = bt.fr.estimate(panel.loc[:pd.Timestamp(r["cutoff"])])
    w_tan = bt.fr._solve(est.mu, est.cov, 1.0, "max_sharpe", risk_free_rate=rf)
    tan_ret, tan_vol, _ = bt.fr.performance(w_tan, est.mu.to_numpy(), est.cov.to_numpy(), rf)
    k = bt.predicted_vol(est, "A3") / tan_vol

    assert k != pytest.approx(1.0, abs=0.05), "the fixture levers by ~1x and tests nothing"
    assert lv["leverage"] == pytest.approx(k, abs=5e-5), "leverage is not the shipped 4dp of k"
    assert lv["predicted"]["sharpe"] == tan["predicted"]["sharpe"]
    assert lv["predicted"]["vol"] == pytest.approx(k * tan_vol, abs=5e-7)
    assert lv["predicted"]["ret"] == pytest.approx(rf + k * (tan_ret - rf), abs=5e-7)
    # The weights are the tangency portfolio's, UNCHANGED. Leverage is financing, not a
    # portfolio, and a row reporting a rescaled weight vector would break every consumer that
    # reads these as convex combinations.
    assert lv["top_holdings"] == tan["top_holdings"]
    assert lv["n_holdings"] == tan["n_holdings"]


def test_levering_by_one_is_the_unlevered_path_and_by_zero_is_cash():
    """Both ends pin `lever` to something independently known. At `k = 1` the financing term
    has coefficient zero, so the rate must not touch the answer -- which is what fails if the
    cash leg is added rather than mixed in. At `k = 0` nothing is held but cash, so the path is
    the risk-free compounding and the portfolio must not touch it."""
    idx = pd.bdate_range("2020-01-02", periods=300)
    v = pd.Series(np.linspace(1.0, 1.4, len(idx)), index=idx)
    rf = pd.Series(0.05, index=idx)

    assert bt.lever(v, 1.0, rf).to_numpy() == pytest.approx(v.to_numpy(), abs=1e-12)
    assert bt.lever(v, 1.0, pd.Series(0.20, index=idx)).to_numpy() == pytest.approx(
        v.to_numpy(), abs=1e-12), "the financing rate leaked into an unlevered position"

    cash = bt.lever(v, 0.0, rf)
    daily = 1.05 ** (1.0 / fetch.TRADING_DAYS_PER_YEAR) - 1.0
    assert cash.iloc[-1] == pytest.approx((1.0 + daily) ** (len(idx) - 1), rel=1e-12)


def test_levering_charges_the_rate_that_moved_not_the_one_at_the_cutoff():
    """A year of borrowing is rolled, so it is paid at the rates that happened. Charging the
    cutoff's quote for the whole window is a subsidy when rates rise, and the error is
    proportional to `k - 1` -- invisible at the tangency portfolio, worth several percent at
    the 3x leverage matching an index's risk needs."""
    idx = pd.bdate_range("2020-01-02", periods=300)
    v = pd.Series(np.linspace(1.0, 1.4, len(idx)), index=idx)
    flat = pd.Series(0.01, index=idx)
    rising = pd.Series(np.linspace(0.01, 0.09, len(idx)), index=idx)

    assert bt.lever(v, 3.0, rising).iloc[-1] < bt.lever(v, 3.0, flat).iloc[-1]
    # And it must be the whole series, not just its first value: a `.iloc[0]` would reproduce
    # the flat answer exactly, which is the mistake this is here to catch.
    assert bt.lever(v, 3.0, rising).iloc[-1] != pytest.approx(
        bt.lever(v, 3.0, pd.Series(rising.iloc[0], index=idx)).iloc[-1], rel=1e-6)


def test_levering_refuses_a_financing_rate_not_quoted_on_the_paths_own_bars():
    """Silent misalignment is the failure this prevents: pandas would broadcast a shorter or
    differently-indexed series into NaN or into the wrong days, and a financing cost applied on
    the wrong dates is wrong by an amount nothing else in the file could reveal."""
    idx = pd.bdate_range("2020-01-02", periods=100)
    v = pd.Series(np.linspace(1.0, 1.2, len(idx)), index=idx)
    with pytest.raises(ValueError, match="quoted on exactly"):
        bt.lever(v, 2.0, pd.Series(0.03, index=idx[:-1]))
    with pytest.raises(ValueError, match="quoted on exactly"):
        bt.lever(v, 2.0, pd.Series(0.03, index=pd.bdate_range("2021-01-04", periods=len(idx))))


def test_the_reported_average_compounds_and_is_not_the_arithmetic_mean():
    """These tables compare strategies at DIFFERENT volatilities, and the arithmetic mean of
    per-period returns exceeds the compounded rate by roughly half the variance -- so it pays a
    bonus for being volatile, which is precisely the confound the matched-risk table exists to
    remove. A hand case, because the two are close enough on real data to look like rounding."""
    assert bt._compound([0.45, -0.25]) == pytest.approx((1.45 * 0.75) ** 0.5 - 1.0, abs=1e-15)
    assert bt._compound([0.45, -0.25]) == pytest.approx(0.042833, abs=1e-6)
    assert np.mean([0.45, -0.25]) == pytest.approx(0.10)
    assert bt._compound([0.1, 0.1, 0.1]) == pytest.approx(0.1, abs=1e-15)


def test_a_cap_that_cannot_reach_the_benchmarks_risk_omits_the_row_rather_than_missing_it():
    """`efficient_risk` at an unreachable target must produce NO row, not a row that quietly
    sits at a lower volatility -- the label says "matched risk" and a row carrying 12% where the
    target was 22% is the one outcome worse than a gap. The panel is built so the benchmark is
    the most volatile asset, which no capped combination of the others can match."""
    p = _panel(n_assets=6, n_bars=1300, seed=7)
    est = bt.fr.estimate(p)
    bench = est.mu.index[-1]  # `_panel`'s vol is increasing, so the last column is the riskiest
    target = bt.predicted_vol(est, bench)
    assert target == pytest.approx(max(np.sqrt(np.diag(est.cov.to_numpy()))))

    labels = [s[0] for s in bt.strategies(est, [0.2], rf=0.03, benchmark=bench)]
    assert "matched_risk@cap20" not in labels

    loose = bt.strategies(est, [1.0], rf=0.03, benchmark=bench)
    w = next(w for label, _, w in loose if label == "matched_risk@cap100")
    assert bt.predicted(w, est, 0.03)["vol"] == pytest.approx(target, abs=1e-5)


# ---------------------------------------------------------------- the momentum forecast
#
# THE WINDOW IS THE WHOLE CONTENT of `momentum_mu`: it reads exactly two bars, and which two is
# the entire specification. Both edges are located by date arithmetic from the cutoff, so both
# can be off by a month while the function keeps returning plausible annualised returns in the
# right units for the right assets -- nothing about the output's SHAPE reveals it.
#
# Momentum is path-independent by construction (only the endpoints are read), so a perturbation
# applied to the middle of the window changes nothing and a test built that way would pass on
# any window at all. Every test below therefore perturbs a SINGLE BAR or a half-open span
# hanging off one edge, and asserts the exact factor the score must move by.

def _mom_edges(train: pd.DataFrame, lookback: int = 12, skip: int = 1):
    """Where the window's two ends must be, derived independently of `momentum_mu`."""
    cutoff = train.index.max()
    idx = train.index
    end = idx[idx <= cutoff - pd.DateOffset(months=skip)].max()
    start = idx[idx <= cutoff - pd.DateOffset(months=lookback)].max()
    return start, end, len(train.loc[start:end]) - 1


def test_momentum_stops_short_of_the_cutoff_by_the_skip(panel):
    """The skip is the difference between momentum and "what went up last month". A one-month
    winner is disproportionately a gap on news that gives itself back, so buying it earns the
    reversal -- and a backtest built without the skip fails by an amount that looks like the
    signal not existing.

    Everything strictly after the window's right edge is inside the TRAINING window and must
    still be invisible: this is the one property that distinguishes a 12/1 window from a 12/0
    one, and no assertion about the score's magnitude or units can see it.
    """
    train = panel.loc[:panel.index[1000]]
    _, end, _ = _mom_edges(train)
    assert (train.index > end).sum() > 10, "the fixture has no bars in the skipped month"

    skipped = train.copy()
    skipped.loc[skipped.index > end] *= 4.0
    assert bt.momentum_mu(skipped).to_numpy() == pytest.approx(
        bt.momentum_mu(train).to_numpy(), abs=1e-15), "the skipped month moved the score"

    # And the edge bar itself is IN the window, by exactly the factor it was moved by. Without
    # this, a window ending a month earlier still than it should would pass the assertion above.
    moved = train.copy()
    moved.loc[end] *= 2.0
    _, _, n = _mom_edges(train)
    grown = (1.0 + bt.momentum_mu(moved)) / (1.0 + bt.momentum_mu(train))
    assert grown.to_numpy() == pytest.approx(2.0 ** (fetch.TRADING_DAYS_PER_YEAR / n), rel=1e-12)


def test_momentum_ignores_everything_before_its_lookback_window(panel):
    """The other edge. A window that ran back to the start of the training panel would be the
    five-year mean again -- i.e. `"history"` under a different name, which is the one result
    that would make the whole comparison vacuous while every number stayed plausible."""
    train = panel.loc[:panel.index[1000]]
    start, _, n = _mom_edges(train)
    assert (train.index < start).sum() > 100, "the fixture has no history before the window"

    older = train.copy()
    older.loc[older.index < start] *= 0.25
    assert bt.momentum_mu(older).to_numpy() == pytest.approx(
        bt.momentum_mu(train).to_numpy(), abs=1e-15), "pre-window history moved the score"

    moved = train.copy()
    moved.loc[start] *= 2.0
    shrunk = (1.0 + bt.momentum_mu(moved)) / (1.0 + bt.momentum_mu(train))
    assert shrunk.to_numpy() == pytest.approx(0.5 ** (fetch.TRADING_DAYS_PER_YEAR / n), rel=1e-12)


def test_momentum_is_annualised_on_the_same_252_day_convention_as_everything_else():
    """It sits in the same `predicted.ret` column as `frontier.estimate`'s output, so it has to
    be the same kind of number. An un-annualised 11-month total return is ~8% low on a 10%/yr
    asset -- close enough to look like a different estimator rather than a missing exponent, and
    it would shift every rank in the same direction, so no ranking test would notice."""
    idx = pd.bdate_range("2019-01-01", periods=800, name="date")
    grow = pd.DataFrame({
        "steady": 1.10 ** (np.arange(len(idx)) / fetch.TRADING_DAYS_PER_YEAR),
        "flat": np.ones(len(idx)),
    }, index=idx)
    mom = bt.momentum_mu(grow)
    assert mom["steady"] == pytest.approx(0.10, abs=1e-9)
    assert mom["flat"] == pytest.approx(0.0, abs=1e-12)


def test_the_momentum_model_replaces_the_return_forecast_and_nothing_else(panel):
    """One input changed, so a difference in the result is attributable to it.

    The covariance is the half that measured +0.91, and re-estimating it over the momentum
    window as well would make the two runs differ in two inputs at once -- after which neither
    result means anything. `min_variance` is the check with teeth: it does not read `mu` at all,
    so its weights MUST be bit-identical across the two models, while its reported forecast
    return moves because that is computed from `mu`.
    """
    train = panel.loc[:panel.index[1000]]
    hist, mom_h = bt.estimates(train, bt.MuModel())
    mom, mom_m = bt.estimates(train, bt.MuModel(kind="momentum"))

    assert np.array_equal(hist.cov.to_numpy(), mom.cov.to_numpy())
    assert hist.shrinkage == mom.shrinkage
    assert list(hist.mu.index) == list(mom.mu.index)
    assert not np.allclose(hist.mu.to_numpy(), mom.mu.to_numpy()), \
        "the momentum forecast is the historical one, so the flag does nothing"
    # The momentum score is returned under BOTH models and is the same number either way -- it
    # is what `momentum_top` and `momentum_rank_correlation` are built from, so a "history" run
    # still measures the signal.
    assert mom_h.to_numpy() == pytest.approx(mom_m.to_numpy(), abs=1e-15)
    assert mom.mu.to_numpy() == pytest.approx(mom_m.to_numpy(), abs=1e-15)

    w_hist = bt.fr._solve(hist.mu, hist.cov, 0.34, "min_volatility")
    w_mom = bt.fr._solve(mom.mu, mom.cov, 0.34, "min_volatility")
    assert np.array_equal(w_hist, w_mom), "min_variance moved, so the covariance was touched"


def test_the_momentum_basket_is_equal_weight_in_the_highest_scoring_assets():
    """The comparator that makes a momentum result attributable. It is the signal held with no
    optimiser at all, so if the optimised version loses to it the covariance matrix spent the
    signal rather than the signal being absent.

    Twelve assets against a basket of four, because with `top_n >= n_assets` the basket collapses
    to `equal_weight` and every assertion here would hold trivially.
    """
    p = _panel(n_assets=12, n_bars=1300, seed=3)
    est, mom = bt.estimates(p, bt.MuModel(kind="momentum", top_n=4))
    symbols = list(est.mu.index)
    labels = bt.strategies(est, [0.34], rf=0.02, benchmark=None, momentum=mom, mom_top=4)

    label, cap, w = next(t for t in labels if t[0].startswith("momentum_top"))
    assert label == "momentum_top4"
    assert cap is None
    assert (w > 0).sum() == 4
    assert w[w > 0] == pytest.approx(0.25, abs=1e-15)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)

    picked = {symbols[i] for i in np.flatnonzero(w)}
    assert picked == set(mom.nlargest(4).index)
    # The HIGHEST, not the lowest: with a spread of drifts the two sets are disjoint, so an
    # `nsmallest` would still produce a legal equal-weight basket of four.
    assert picked.isdisjoint(set(mom.nsmallest(4).index))
    # And it is not just `equal_weight` under another name, which is what a `top_n >= n` run
    # would silently be.
    ew = next(w2 for lbl, _, w2 in labels if lbl == "equal_weight")
    assert not np.array_equal(w, ew)


def test_the_momentum_comparator_and_diagnostic_are_present_under_both_forecasts(panel, irx):
    """`momentum_top` and `momentum_rank_correlation` are the fixed points the two runs share:
    the same portfolio and the same score, so the two printouts can be laid side by side. Gating
    either on `kind == "momentum"` would leave the history run with nothing to attribute a
    difference to, and it is the natural way to write it."""
    out = {kind: bt.run_cutoff(panel, 1, irx, [0.34], "A3", None, 8,
                               bt.MuModel(kind=kind, top_n=3))
           for kind in ("history", "momentum")}
    for kind, r in out.items():
        assert any(s["strategy"] == "momentum_top3" for s in r["strategies"]), kind
        assert r["mu_model"]["kind"] == kind

    a, b = out["history"], out["momentum"]
    assert a["momentum_rank_correlation"] == b["momentum_rank_correlation"]
    assert a["vol_rank_correlation"] == b["vol_rank_correlation"]
    # The forecast in use differs, and under momentum the two diagnostics are the same number
    # because `est.mu` IS the momentum score there.
    assert a["mu_rank_correlation"] != b["mu_rank_correlation"]
    assert b["mu_rank_correlation"] == b["momentum_rank_correlation"]

    mom_a = next(s for s in a["strategies"] if s["strategy"] == "momentum_top3")
    mom_b = next(s for s in b["strategies"] if s["strategy"] == "momentum_top3")
    assert mom_a["top_holdings"] == mom_b["top_holdings"]
    assert mom_a["realized_monthly"] == mom_b["realized_monthly"]


def test_a_training_window_too_short_for_the_momentum_lookback_is_refused(panel):
    """TWO DISTINCT REFUSALS, asserted separately and matched on their own messages.

    A single `pytest.raises` with an alternation across both would be satisfied by either one,
    and the two have different causes: the panel not reaching back far enough is caught by
    `_bar_at_or_before`, while a window that exists but holds too few bars to rank on is caught
    by the length check. Written as one alternation, the length check's own mutant SURVIVED --
    the harness reported it caught and the test guarded only the first failure.
    """
    with pytest.raises(ValueError, match="before the panel starts"):
        bt.momentum_mu(panel.iloc[:30])

    # A window that EXISTS at both ends and still measures almost nothing: month-end bars, so
    # eleven months of them is eleven returns. `--train-years 0.5` on daily data and a coarse
    # panel are the same failure, and neither may return a score labelled as a twelve-month one.
    sparse = panel.resample("ME").last().dropna()
    assert len(sparse) > 20, "the fixture is not long enough to place both window edges"
    with pytest.raises(ValueError, match="too few to rank on"):
        bt.momentum_mu(sparse)


def test_the_momentum_model_refuses_a_configuration_that_cannot_mean_anything():
    """A validator, so it needs input the real runs do not contain. A skip at or past the
    lookback inverts the window; an unknown `kind` must not fall through to the historical mean
    while the output records the name that was asked for."""
    with pytest.raises(ValueError, match="need 0 <= skip < lookback"):
        bt.MuModel(kind="momentum", lookback_months=12, skip_months=12)
    with pytest.raises(ValueError, match="unknown expected-return model"):
        bt.MuModel(kind="reversal")
    with pytest.raises(ValueError, match="top_n"):
        bt.MuModel(top_n=0)
    assert bt.MuModel(kind="momentum", lookback_months=6, skip_months=0).skip_months == 0


# ------------------------------------------------------------------------------ the printers
#
# A HAND-BUILT BLOCK, not a run. `matched_risk` is dropped in any period whose cap could not
# reach the benchmark's risk, so the printers have to cope with a strategy that is present in
# some periods and absent in others -- and the shipped ETF universe reaches the target at cap
# 100% in all ten periods, so it cannot produce the hole. Momentum did, on the first run, and
# the printer crashed on it.

def _fake_row(label: str, ret: float, vol: float = 0.15, **extra) -> dict:
    return {"strategy": label, "cap": 1.0, "n_holdings": 3, "top_holdings": {},
            "predicted": {"ret": 0.1, "vol": vol, "sharpe": 0.5},
            "realized_linear_ret": ret,
            "realized_monthly": {"growth": 1.0 + ret, "ret": ret, "vol": vol,
                                 "sharpe": 0.4, "max_drawdown": 0.2},
            "realized_hold": {"growth": 1.0 + ret, "ret": ret, "vol": vol,
                              "sharpe": 0.4, "max_drawdown": 0.2},
            **extra}


def test_a_strategy_missing_from_some_periods_is_compared_against_the_right_years():
    """The failure this prevents is a WRONG NUMBER, not a crash.

    Compacting a strategy's returns and the benchmark's independently, then zipping them, pairs
    period 3 of one against period 4 of the other the moment either has a hole -- and the win
    count that comes out is plausible, in range, and about the wrong years. The returns here are
    chosen so the two readings disagree: aligned, the long-only row wins both of the periods it
    ran in; compacted, it wins one.
    """
    mr, bm = "matched_risk@cap100", "benchmark:SPY"
    rets = {0: (0.10, 0.05), 1: (None, 0.50), 2: (0.30, 0.20)}
    periods = []
    for i, (m, b) in rets.items():
        rows = [_fake_row(bm, b)]
        if m is not None:
            rows.insert(0, _fake_row(mr, m, risk_target=0.15))
        periods.append({"period": f"20{20 + i}-01-01 to 20{21 + i}-01-01",
                        "strategies": rows,
                        "mu_rank_correlation": 0.1, "vol_rank_correlation": 0.9,
                        "momentum_rank_correlation": 0.2})
    block = {"train_years": 5.0, "hold_years": 1, "span_years": 3.0, "caps": [1.0],
             "benchmark": "SPY", "mu_model": bt.MuModel().describe(), "periods": periods}

    out = bt.matched_risk_table(block)
    assert "long-only beat it in 2 of 2 periods" in out, out
    # And the reader is told the average is over fewer periods than the table has rows, or two
    # rows compounded over different numbers of years sit side by side looking comparable.
    assert "(2 of 3 periods)" in out, out

    # The whole-block printer has the same hole to cope with and its own alignment.
    summary = bt.summary_table(block)
    assert f"{mr:<26}" in summary
    assert "2 / 2" in summary, summary


def test_the_rank_correlations_are_computed_over_the_whole_cross_section(panel, irx):
    """Spearman over all assets, and both correlations are real numbers in [-1, 1] rather
    than a NaN that would print as a plausible-looking blank. The `mu` one is allowed to be
    anything -- that it can be near zero is the finding, not a bug."""
    r = bt.run_cutoff(panel, 1, irx, [0.2], None, None, 8)
    for key in ("mu_rank_correlation", "vol_rank_correlation"):
        assert -1.0 <= r[key] <= 1.0
        assert not np.isnan(r[key])


# ------------------------------------------- a benchmark that is priced but not investable

def test_holding_out_the_benchmark_prices_it_without_letting_anything_buy_it(panel, irx):
    """The whole point of `investable`: a universe of single stocks measured against an index
    fund, where the index fund must be in the panel and must not be in the candidate set.

    Without the hold-out, "did mean-variance beat the index" is asked of a portfolio that is
    allowed to BE the index -- and it would be, because an index fund's variance is lower than
    almost any single constituent's, so the min-variance end of the frontier buys it on sight.

    Three separate claims, and the middle one is the one that silently broke first: the benchmark
    is not held by anything, it is still PRICED (its own row, its forecast volatility, and
    therefore a target for the risk-matched rows), and the risk-matched rows still exist. An
    earlier version gated the target on `benchmark in symbols`, so holding the benchmark out
    removed `matched_risk` -- the one comparison that settles the question -- and nothing failed.
    """
    bm = "A3"
    keep = [c for c in panel.columns if c != bm]
    r = bt.run_cutoff(panel, 1, irx, [1.0], bm, None, 8, bt.MuModel(), keep)

    assert r["n_investable"] == len(keep)
    labels = [s["strategy"] for s in r["strategies"]]
    assert f"benchmark:{bm}" in labels
    assert "matched_risk@cap100" in labels
    assert "tangency_levered@cap100" in labels

    for s in r["strategies"]:
        if s["strategy"] == f"benchmark:{bm}":
            assert s["top_holdings"] == {bm: 1.0}
            continue
        assert bm not in s["top_holdings"], f"{s['strategy']} holds the benchmark"

    # And it is priced: the target the risk-matched rows aim at is the held-out benchmark's own
    # forecast volatility, computed from the FULL estimate rather than the restricted one.
    train = panel.loc[:pd.Timestamp(r["cutoff"])]
    target = bt.predicted_vol(bt.fr.estimate(train), bm)
    lv = next(s for s in r["strategies"] if s["strategy"] == "tangency_levered@cap100")
    assert lv["risk_target"] == pytest.approx(target, abs=5e-7)


def test_holding_nothing_out_is_the_same_run_as_not_holding_out_at_all(panel, irx):
    """`investable=None` and `investable=every column` must be the SAME numbers, bit for bit.

    This is what makes the feature safe to add to a file whose other results are already
    published: the ETF universe passes `None` and must be unaffected. Bit-identical rather than
    close, because `restrict` reindexes `mu` and `cov` and a reindex that reordered them would
    give an answer that is right to five decimals and not the same answer.
    """
    a = bt.run_cutoff(panel, 1, irx, [1.0, 0.2], "A3", None, 8)
    b = bt.run_cutoff(panel, 1, irx, [1.0, 0.2], "A3", None, 8, bt.MuModel(), list(panel.columns))
    assert a == b


def test_restrict_takes_the_submatrix_and_does_not_re_estimate_or_re_shrink(panel):
    """A subset of the estimates, never a fresh estimate over the subset.

    Re-estimating would change the shrinkage intensity and every covariance entry, so the
    held-out run would differ from the full run in ways that have nothing to do with the
    hold-out -- and the two runs are supposed to differ in exactly one thing. It also has to
    RAISE on an asset that was never estimated rather than produce a NaN row, which is how a
    typo'd symbol would otherwise reach the solver as an asset with no risk and no return.
    """
    est = bt.fr.estimate(panel)
    keep = ["A4", "A1", "A2"]
    sub = bt.restrict(est, keep)

    assert list(sub.mu.index) == keep
    assert list(sub.cov.index) == keep and list(sub.cov.columns) == keep
    assert sub.shrinkage == est.shrinkage
    assert sub.n_obs == est.n_obs
    for s in keep:
        assert sub.mu[s] == est.mu[s]
        for t in keep:
            assert sub.cov.loc[s, t] == est.cov.loc[s, t]

    with pytest.raises(ValueError, match="were not estimated"):
        bt.restrict(est, ["A1", "NOPE"])


# --------------------------------------------------- one window, several nested hold lengths

def test_a_window_hold_ends_at_the_requested_length_not_at_the_panel_end(long_panel, long_irx):
    """The difference between `run_window` and `run_cutoff`, asserted rather than described.

    `run_cutoff` holds to the last bar of the panel, so its hold length is whatever the panel
    happens to give and grows every week the data is refreshed. Here both ends are stated, and
    the failure this guards is the easy one: a 1-year row that quietly ran for four years
    because the test window was taken as `panel.loc[cutoff:]`.
    """
    rows = bt.run_window(long_panel, 2.0, 3.0, [1.0, 2.0, 3.0], long_irx, [1.0], "A3", None, 8)
    assert [r["hold_years"] for r in rows] == [1.0, 2.0, 3.0]

    cutoffs = {r["cutoff"] for r in rows}
    assert len(cutoffs) == 1, "the rows are supposed to share one cutoff"
    cutoff = pd.Timestamp(rows[0]["cutoff"])

    # One fixed-length training window behind the cutoff, not everything before it. Measured in
    # CALENDAR terms against the requested offset, not against `train["years"]`: that field is
    # bars/252, and this fixture is business days with no holidays, so two calendar years of it
    # is 522 bars and reports 2.07 years. The window is right; 252 is a convention about a real
    # exchange calendar. Asserting the field here would be asserting a property of the fixture.
    assert pd.Timestamp(rows[0]["train"]["start"]) > long_panel.index.min()
    train_from = pd.Timestamp(rows[0]["train"]["start"])
    assert train_from <= cutoff - pd.DateOffset(years=2)
    assert train_from > cutoff - pd.DateOffset(years=2) - pd.Timedelta(days=7)

    for r in rows:
        end = pd.Timestamp(r["test"]["end"])
        wanted = cutoff + pd.DateOffset(years=r["hold_years"])
        assert end <= wanted
        assert end > wanted - pd.Timedelta(days=7), r["hold_years"]


def test_the_nested_window_rows_are_one_weight_vector_measured_at_several_dates(long_panel,
                                                                               long_irx):
    """The claim the docstring makes, so that quoting three rows as three wins is provably wrong.

    All three rows share one training window and therefore one solve per strategy. If the weights
    ever differed between hold lengths, something in the chain would be reading the test window --
    and the rows would look like three trials, which is exactly the misreading the mode invites.
    """
    rows = bt.run_window(long_panel, 2.0, 3.0, [1.0, 2.0, 3.0], long_irx, [1.0], "A3", None, 8)
    by_label = [{s["strategy"]: s["top_holdings"] for s in r["strategies"]} for r in rows]
    assert by_label[0] == by_label[1] == by_label[2]
    # ... and the returns are nested, not independent: the longer hold contains the shorter one.
    growth = [next(s for s in r["strategies"] if s["strategy"] == "tangency@cap100")
              ["realized_hold"]["growth"] for r in rows]
    assert len({round(g, 9) for g in growth}) == 3, "three end dates gave the same growth"


def test_a_window_hold_that_runs_past_the_panel_is_refused_rather_than_truncated(long_panel,
                                                                                long_irx):
    """A 5-year hold from 3 years back cannot exist, and the wrong answer is a 3-year hold
    labelled 5. Silently short windows are how an annualised return gets computed over the wrong
    denominator, and every row in the table would still look plausible."""
    with pytest.raises(ValueError, match="past the panel's last bar"):
        bt.run_window(long_panel, 2.0, 3.0, [1.0, 5.0], long_irx, [1.0], "A3", None, 8)


def test_a_window_row_cannot_see_past_its_own_cutoff(long_panel, long_irx):
    """The lookahead guard for this mode. Same shape as the `run_cutoff` one and not redundant
    with it: `run_window` does its own date arithmetic for both ends of both windows, which is a
    second place the training window could pick up a bar on the wrong side of the cutoff.

    Perturbed AFTER the cutoff, which is inside every hold -- so this also checks that a longer
    hold's returns cannot reach back and change a shorter hold's weights."""
    def decide(p, r):
        rows = bt.run_window(p, 2.0, 3.0, [1.0, 2.0], r, [1.0], "A3", None, 8)
        return rows, [[(s["strategy"], s["top_holdings"], s.get("leverage"),
                        s.get("risk_target")) for s in row["strategies"]] for row in rows]

    base, base_w = decide(long_panel, long_irx)
    cutoff = pd.Timestamp(base[0]["cutoff"])
    future = long_panel.index > cutoff
    assert future.sum() > 250

    shifted = long_panel.copy()
    shifted.loc[future] *= 1.7
    shifted.loc[future, "A0"] *= 3.0
    moved, moved_w = decide(shifted, long_irx)
    assert [r["cutoff"] for r in moved] == [r["cutoff"] for r in base]
    assert moved_w == base_w
    assert all(a["realized_monthly"]["growth"] != b["realized_monthly"]["growth"]
               for ra, rb in zip(base, moved)
               for a, b in zip(ra["strategies"], rb["strategies"])), \
        "the hold windows changed and nothing moved"

    later = long_irx.copy()
    later.loc[future] *= 2.0
    rated, rated_w = decide(long_panel, later)
    assert rated_w == base_w
    assert rated[0]["rf_mean_over_test"] != base[0]["rf_mean_over_test"]


# -------------------------------------------------------------------------- the cost model
#
# Synthetic for the same reason as the rest of this file: a shipped result is one parameter
# setting, and the load-bearing claim is about the RELATION between two settings -- that
# `Costs()` reproduces a pre-cost run bit for bit, and that the turnover convention is the one
# the file says it is. Neither is visible in a single artifact.

def test_a_zero_cost_run_is_bit_identical_to_one_made_before_costs_existed(panel, irx):
    """The reason `zero()` is checked before any net column is emitted rather than trusting the
    arithmetic to come out at 1.0.

    Every already-published result in `pipeline/results/` was generated without this parameter,
    and a default that changed them by 1e-16 would make every one of them 'stale' by the standard
    `test_a_stress_bound_names_a_result_that_exists_and_quotes_its_benchmark_correctly` applies.
    Compared as JSON text, not field by field: a new key with a zero in it is exactly the kind of
    difference a field-by-field loop is written not to notice.
    """
    cutoff = panel.index[900]
    train, test = panel.loc[:cutoff], panel.loc[cutoff:]
    args = (train, test, irx, [1.0, 0.2], "A3", None, 5, bt.MuModel(), None)
    default = json.dumps(bt.evaluate(*args), sort_keys=True, default=str)
    explicit = json.dumps(bt.evaluate(*args, bt.Costs()), sort_keys=True, default=str)
    assert default == explicit
    assert "net_monthly" not in default, (
        "a zero-cost run must not emit net columns at all -- a column of zeros invites being read "
        "as a measurement of frictions rather than as their absence"
    )


def test_costs_charge_the_optimiser_and_the_benchmark_or_neither(panel, irx):
    """The bias this whole parameter exists to remove, asserted as an inequality on the OUTPUT.

    An index fund is bought once and charges a published fee; an annually-reoptimised portfolio
    pays no fee and trades. Charging only the second is the comparison that flatters the index and
    charging only the first is the comparison that flatters the optimiser -- so the test is that
    BOTH sides lose return, and that the benchmark's loss is its expense ratio and not a trading
    cost it never paid.
    """
    cutoff = panel.index[900]
    train, test = panel.loc[:cutoff], panel.loc[cutoff:]
    costs = bt.Costs(trade_bps=25.0, expense={"A3": 0.001})
    row = bt.evaluate(train, test, irx, [1.0], "A3", None, 5, bt.MuModel(), None, costs)
    by = {s["strategy"]: s for s in row["strategies"]}
    bm = by["benchmark:A3"]
    assert bm["net_monthly"]["expense_ratio"] == pytest.approx(0.001)
    assert bm["net_monthly"]["turnover_per_year"] == pytest.approx(
        bm["net_hold"]["turnover_per_year"]), (
        "the benchmark is bought once and held, so its turnover cannot depend on the rebalance rule"
    )
    for label, s in by.items():
        assert s["net_monthly"]["ret"] < s["realized_monthly"]["ret"], f"{label} paid nothing"
        assert s["net_monthly"]["cost_drag_per_year"] > 0
    opt = by["tangency@cap100"]
    assert opt["net_monthly"]["turnover_per_year"] > bm["net_monthly"]["turnover_per_year"], (
        "the optimised portfolio must trade more than a held index, or the fixture cannot show "
        "which side the no-cost run was subsidising"
    )
    assert opt["net_monthly"]["expense_ratio"] == 0.0, (
        "a single stock charges no management fee; only symbols named in `expense` pay one"
    )


def test_turnover_is_buys_plus_sells_and_bar_zero_is_the_purchase(panel):
    """The factor of 2 is the size of the entire effect, so it is asserted rather than described.

    Bar 0 is `sum |w| == 1.0` -- the position has to be bought -- and under "hold" that is the
    only trade there ever is. A one-way convention would make every cost in every table exactly
    half of what it should be, which is a difference no shape check can see.
    """
    w = np.zeros(panel.shape[1])
    w[0] = 1.0
    hold = bt.turnover_path(panel, w, "hold")
    assert hold.iloc[0] == pytest.approx(1.0)
    assert hold.iloc[1:].sum() == 0.0

    even = np.full(panel.shape[1], 1.0 / panel.shape[1])
    monthly = bt.turnover_path(panel, even, "monthly")
    assert monthly.iloc[0] == pytest.approx(1.0)
    assert monthly.iloc[1:].sum() > 0.0, "a drifting equal-weight portfolio has to be rebalanced"
    # Two independent readings of the same month boundary: the drifted weights `turnover_path`
    # computes, and the same weights rebuilt from `value_path`'s own base row. They have to agree
    # about WHEN a trade happened, or the cost is charged on a bar the portfolio did not trade on.
    period = panel.index.to_period("M")
    traded = {i - 1 for i in range(2, len(panel)) if period[i] != period[i - 1]}
    nonzero = {i for i in range(1, len(panel)) if monthly.iloc[i] > 0}
    assert nonzero == traded

    with pytest.raises(ValueError, match="rebalance must be"):
        bt.turnover_path(panel, even, "daily")


def test_the_cost_haircut_is_multiplicative_so_net_weights_equal_gross_weights(panel):
    """Why the net path ships from the same solve rather than from a second one.

    `turnover_path`'s docstring claims a proportional charge scales the portfolio's value and
    leaves every subsequent return AND every subsequent WEIGHT untouched -- which is what lets
    the net reading be the gross path times a cumulative product. If it were false the net path
    would need its own weight trajectory and one solve could not produce both.

    Checked against an independent simulator that holds SHARE COUNTS and pays the bill out of the
    portfolio at each trade, so `turnover_path` and `cost_factor` are both verified against
    something that does not use either of them. `pd.Series * pd.Series` against itself would
    prove only that multiplication is associative.
    """
    w = np.full(panel.shape[1], 1.0 / panel.shape[1])
    bps = 30.0

    P = panel.to_numpy(dtype=float)
    period = panel.index.to_period("M")
    value = 1.0 - bps / 1e4 * float(np.abs(w).sum())   # the opening purchase
    shares = value * w / P[0]
    sim = [value]
    for i in range(1, len(P)):
        value = float(shares @ P[i])
        if i + 1 < len(P) and period[i + 1] != period[i]:
            drifted = shares * P[i] / value
            value *= 1.0 - bps / 1e4 * float(np.abs(w - drifted).sum())
            shares = value * w / P[i]
        sim.append(value)

    costs = bt.Costs(trade_bps=bps)
    net = bt.value_path(panel, w, "monthly") * bt.cost_factor(
        bt.turnover_path(panel, w, "monthly"), 0.0, costs)
    assert np.allclose(net.to_numpy(), np.array(sim), rtol=1e-12, atol=0.0)
    assert net.iloc[0] < 1.0, "the opening purchase has to be charged"
    assert float(bt.cost_factor(bt.turnover_path(panel, w, "monthly"), 0.0, bt.Costs()).min()) == 1.0, (
        "a zero-cost factor must be exactly 1.0 at every bar, not approximately -- `zero()` gates "
        "the net columns on that being true"
    )

    # The fee accrues on every bar and the trading charge only on bars that traded, so a position
    # bought once and never traded still loses ground: the index fund's own case.
    fee_only = bt.cost_factor(bt.turnover_path(panel, w, "hold"), 0.01, bt.Costs())
    assert fee_only.iloc[0] == 1.0
    assert fee_only.iloc[-1] == pytest.approx(
        (1.0 - 0.01) ** ((len(panel) - 1) / fetch.TRADING_DAYS_PER_YEAR), rel=1e-12)


def test_a_negative_trading_cost_or_an_impossible_fee_is_refused():
    """A validator, so it needs input the CLI cannot produce: `--trade-bps -5` is a subsidy that
    would make every net column better than its gross one, and an expense ratio of 1.0 wipes the
    position out in a year. Both are typos, and both would produce a complete plausible file."""
    with pytest.raises(ValueError, match="must not be negative"):
        bt.Costs(trade_bps=-5.0)
    for bad in (1.0, -0.01):
        with pytest.raises(ValueError, match="expense ratios must be"):
            bt.Costs(expense={"SPY": bad})
    assert bt.Costs(trade_bps=0.0, expense={"SPY": 0.0}).zero(), (
        "a fee of zero is not a cost, and `zero()` gating the net columns on it is what keeps an "
        "explicit `--trade-bps 0` reproducing a run made before the flag existed"
    )


def test_levering_charges_the_daily_reset_it_documents_as_free(panel, irx):
    """`lever`'s docstring says the daily mix reset "is free here, and it is free nowhere", and
    `mix_turnover` is what stops that being true. The convention has to match `turnover_path`'s
    buys-plus-sells, or the levered rows are charged half of what the unlevered ones are.

    Checked at k = 1 and k = 0, where the answer is known: an unlevered position resets nothing
    and pure cash resets nothing, so any turnover at either end is the formula charging drift that
    did not happen.
    """
    v = bt.value_path(panel, np.full(panel.shape[1], 1.0 / panel.shape[1]), "hold")
    rf = irx.reindex(v.index)
    assert float(bt.mix_turnover(v, 1.0, rf).abs().max()) == pytest.approx(0.0, abs=1e-12)
    assert float(bt.mix_turnover(v, 0.0, rf).abs().max()) == pytest.approx(0.0, abs=1e-12)
    t3 = bt.mix_turnover(v, 3.0, rf)
    assert t3.sum() > 0.0
    # 2x the one-way drift, matching `turnover_path`. Recomputed here from the definition rather
    # than from the function, so agreeing with itself is not what is being checked.
    r = v.pct_change()
    cash = (1.0 + rf) ** (1.0 / 252) - 1.0
    mixed = (3.0 * r + (1.0 - 3.0) * cash).fillna(0.0)
    one_way = (3.0 * (1.0 + r) / (1.0 + mixed) - 3.0).abs().fillna(0.0)
    assert np.allclose(t3.to_numpy(), 2.0 * one_way.to_numpy())
