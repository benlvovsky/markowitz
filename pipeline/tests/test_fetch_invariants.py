"""Invariants for the collector: what a price IS, and what the panel is allowed to contain.

No network. `parse` is tested against hand-built payloads, and the panel logic against the
parquet already on disk -- so the suite runs offline and a Yahoo outage cannot turn into a
red build that says nothing about this code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fetch
import store
import universe as uni
from conftest import PRICES

SYMBOLS = uni.load().symbols


def payload(timestamps, closes, adjcloses):
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [{"close": closes}],
                        "adjclose": [{"adjclose": adjcloses}],
                    },
                }
            ]
        }
    }


# --------------------------------------------------------------------------------- parse


def test_parse_indexes_by_the_new_york_trading_date_not_the_raw_timestamp():
    """Yahoo returns the session OPEN in exchange-local time, so the intraday component
    shifts by an hour across daylight saving. Indexing on the raw timestamp puts the same
    trading day in two different buckets either side of a DST change, and the panel
    intersection then silently loses those days for every asset."""
    # 2021-01-04 14:30 UTC (09:30 EST) and 2021-07-01 13:30 UTC (09:30 EDT).
    ts = [1609770600, 1625139000]
    df = fetch.parse(payload(ts, [100.0, 110.0], [99.0, 109.0]), "X")
    assert [str(d.date()) for d in df.index] == ["2021-01-04", "2021-07-01"]
    assert df.index.tz is None
    assert (df.index == df.index.normalize()).all(), "a time-of-day survived into the index"


def test_parse_keeps_close_and_adjclose_separately():
    """Both are stored so the adjustment can be audited rather than trusted; collapsing them
    to one column removes the only evidence that it happened."""
    df = fetch.parse(payload([1609770600], [100.0], [88.0]), "X")
    assert list(df.columns) == ["close", "adjclose"]
    assert df["close"].iloc[0] == 100.0 and df["adjclose"].iloc[0] == 88.0


def test_parse_drops_null_bars_rather_than_carrying_them_forward():
    ts = [1609770600, 1609857000, 1609943400]
    df = fetch.parse(payload(ts, [100.0, None, 102.0], [100.0, None, 102.0]), "X")
    assert len(df) == 2, "a null bar survived"
    assert df["adjclose"].notna().all()


def test_parse_deduplicates_timestamps_keeping_the_last():
    """The chart endpoint occasionally repeats the current session as both an intraday and a
    settled bar; keeping both would make one trading day contribute two returns."""
    ts = [1609770600, 1609770600 + 60]  # same NY date
    df = fetch.parse(payload(ts, [100.0, 101.0], [100.0, 101.0]), "X")
    assert len(df) == 1
    assert df["adjclose"].iloc[0] == 101.0


def test_parse_raises_rather_than_returning_a_partial_or_close_only_series():
    """Every one of these must RAISE. A collector that returns something usable-looking on a
    malformed response gets its symbol dropped by the window filter instead, i.e. the price
    error becomes a silent exclusion."""
    with pytest.raises(ValueError):  # adjclose absent: includeAdjustedClose was ignored
        fetch.parse(
            {"chart": {"result": [{"timestamp": [1609770600], "indicators": {"quote": [{"close": [1.0]}]}}]}},
            "X",
        )
    with pytest.raises(ValueError):  # no bars at all
        fetch.parse(payload([], [], []), "X")
    with pytest.raises(ValueError):  # an error response
        fetch.parse({"chart": {"result": None, "error": {"code": "Not Found"}}}, "X")
    with pytest.raises(ValueError):  # a zero or negative price is not a price
        fetch.parse(payload([1609770600], [1.0], [0.0]), "X")


# ------------------------------------------------------------- the stored series on disk


@pytest.fixture(scope="module")
def stored():
    have = [s for s in SYMBOLS if s in store.stored_symbols(PRICES)]
    if len(have) < 20:
        pytest.skip("no local price cache -- run `python -u pipeline/build.py` first")
    return have


@pytest.fixture(scope="module")
def closes(stored):
    """The wide `close` panel, read once. Every stored-series test below slices this."""
    return store.read_close(PRICES, stored)


@pytest.fixture(scope="module")
def adjusted(stored):
    """{symbol: reconstructed total-return series}, over each symbol's own full history."""
    close = store.read_close(PRICES, stored)
    divs = store.dividends_by_symbol(PRICES)
    return {s: store.total_return(close[s].dropna(), divs.get(s)) for s in stored}


def test_no_stored_series_contains_a_weekend_bar(closes, stored):
    """A weekend bar means the index is a calendar, not a trading calendar, and 40% of the
    rows in the panel would be fabricated."""
    for sym in stored:
        idx = closes[sym].dropna().index
        weekend = idx[idx.dayofweek >= 5]
        assert len(weekend) == 0, f"{sym} has {len(weekend)} weekend bars, first {weekend[:1]}"


def test_no_stored_series_has_a_duplicate_date(closes, stored):
    for sym in stored:
        idx = closes[sym].dropna().index
        assert not idx.has_duplicates, f"{sym} has duplicate dates"
        assert idx.is_monotonic_increasing, f"{sym} is not sorted"


def _cagr(series):
    return float((series.iloc[-1] / series.iloc[0]) ** (252 / (len(series) - 1)) - 1)


def test_the_adjustment_is_visible_at_the_start_of_a_dividend_payer(adjusted, closes):
    """If the adjusted series equalled `close` the pipeline would be running on price return,
    which understates the income assets here by up to 10%/yr each -- and understates them
    UNEQUALLY, so it reorders the cross-section rather than shifting it.

    The ratio is checked at the FIRST bar, not the last. The adjustment is applied backward, so
    the most recent adjusted price equals the most recent `close` for every symbol including
    SPY; comparing at the end tests nothing and passes on a pipeline that never adjusted at all.
    """
    for sym in ("SPY", "HYG", "VYM"):
        if sym not in adjusted:
            continue
        c, a = closes[sym].dropna(), adjusted[sym]
        start_ratio = float(c.iloc[0] / a.iloc[0])
        end_ratio = float(c.iloc[-1] / a.iloc[-1])
        assert start_ratio > 1.2, f"{sym}: close/adjusted at the first bar is only {start_ratio:.3f}"
        assert end_ratio == pytest.approx(1.0, abs=1e-3), (
            f"{sym}: end ratio {end_ratio:.4f} -- the adjustment is not anchored at the last bar, "
            "so the direction of the whole series is in question"
        )


def test_total_return_compounds_at_least_as_fast_as_price_return_for_every_asset(adjusted, closes, stored):
    """Dividends can only add, so the adjusted series must compound >= `close`. The reverse
    ordering is a sign error in the adjustment, and it is otherwise undetectable: both series
    are plausible price histories.

    The gap is also bounded, because it IS the realised distribution yield: 12%/yr is above
    every payer in this universe (REM, mortgage REITs, is the highest at 9.7%/yr) and far
    below what a mishandled split would produce.

    THE DIRECTION OF THE INEQUALITY IS NOT ENOUGH, and this test was worthless until the
    mutation harness said so: `gap >= 0` is satisfied by `gap == 0`, i.e. by an adjustment that
    never happens at all. Deleting the entire dividend adjustment left this test green. So the
    two populations are asserted separately, from the event log rather than from a guess about
    which tickers pay:

      a symbol with dividends in the log     must compound STRICTLY faster, by >= 1bp/yr
      a symbol with none                     must be bit-identical to its close

    The 1e-4 floor is measured, not chosen: the thinnest payer in this store is FXF at
    1.4e-3/yr, fourteen times the floor, and the heaviest is REM at 9.7e-2/yr. Ten symbols
    (GLD, IAU, SLV, PPLT, PALL, USO, UNG, GSG, DJP, FXY) have no distribution on record and are
    the ones the equality applies to -- IAU's and SLV's events are splits, not dividends, which
    is why the split is read from the log rather than inferred from the ticker.
    """
    divs = store.dividends_by_symbol(PRICES)
    payers = 0
    for sym in stored:
        gap = _cagr(adjusted[sym]) - _cagr(closes[sym].dropna())
        assert gap >= -1e-9, f"{sym}: price return beats total return by {-gap:.4f}/yr"
        assert gap < 0.12, f"{sym}: implied yield {gap:.4f}/yr is not a distribution yield"
        d = divs.get(sym)
        if d is not None and float(d.sum()) > 0:
            payers += 1
            assert gap > 1e-4, (
                f"{sym}: {len(d)} dividends in the log totalling {d.sum():.2f} but the total "
                f"return only compounds {gap:.2e}/yr faster than the price -- the adjustment "
                "is not being applied"
            )
        else:
            assert gap == 0.0, f"{sym}: no dividend on record yet the total return differs"
    assert payers > 100, f"only {payers} payers found; the event log is not what this test reads"


def test_no_panel_asset_has_a_split_sized_daily_move(panel_and_drops):
    """A move at or beyond 50% in a diversified ETF is a split or a bad print, not a market
    day -- and one such row dominates that asset's whole covariance row.

    Scoped to the PANEL, i.e. to what actually reaches the optimiser, and the bound sits in a
    measured gap: the largest real daily move among the 116 panel assets is XOP at 36.9%
    (March 2020), while CPER's fabricated 2014 round trip is 49.9%. The bound is not a round
    number chosen for comfort; if it ever fires, the ticker needs looking at rather than the
    bound needs raising. CPER is excluded by name for exactly this, with the evidence written
    down in `pipeline/universes/etf_global.toml` under [excluded].
    """
    panel, _ = panel_and_drops
    worst = panel.pct_change().dropna().abs().max()
    for sym, m in worst.items():
        assert m < 0.50, f"{sym}: {m:.1%} in one day"


# ---------------------------------------------------------------------------- load_panel


@pytest.fixture(scope="module")
def panel_and_drops(stored):
    return fetch.load_panel(stored, PRICES, start="2011-01-03")


def test_panel_has_no_holes_and_fabricates_no_dates(panel_and_drops, stored, closes):
    """Every date in the panel must be a real trading day for every asset in it. A forward
    fill would insert a zero return, and zero returns are the one contamination a covariance
    estimator cannot see through: they deflate variances AND pull correlations toward zero.

    This CANNOT catch a forward fill on the shipped store and it is worth knowing why: measured,
    all 116 surviving assets trade on exactly the same 3,941 days, so within the window there is
    not one hole for a fill to fill and `dropna` and `ffill().dropna()` return the identical
    panel. The property still has to hold for a universe where that is false -- an Asian or
    European book will not share a calendar -- so the guard is
    `test_load_panel_intersects_trading_days_rather_than_forward_filling`, below, on a store
    built with a hole in it on purpose. What THIS test does is assert the premise: the shipped
    panel really is hole-free.
    """
    panel, _ = panel_and_drops
    assert not panel.isna().to_numpy().any(), "the panel has NaN"
    dates = set(panel.index)
    for sym in panel.columns:
        own = set(closes[sym].dropna().index)
        missing = dates - own
        assert not missing, f"{sym}: panel contains {len(missing)} dates it never traded on"


def _tiny_store(price_dir, symbol, dates, closes):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
    px = np.asarray(closes, dtype=float)
    store.write(price_dir, {symbol: pd.DataFrame({"close": px, "adjclose": px}, index=idx)}, {})


def test_load_panel_intersects_trading_days_rather_than_forward_filling(tmp_path):
    """A hole is dropped for everyone, never filled for the one asset that has it.

    Built on a synthetic store because the shipped one cannot exercise this at all (see above),
    which means the loudest rule in `fetch`'s docstring rested on nothing until this test
    existed. The failure it guards is not a missing value -- it is a FABRICATED one: a filled
    hole repeats yesterday's price, which is a zero return the estimator reads as a real
    observation, deflating that asset's variance and pulling every correlation it appears in
    toward zero.

    BBB misses 2020-01-06, so the panel must be three rows and BBB's next return must be
    measured across the gap (21 -> 23), not out of a fabricated flat day.
    """
    _tiny_store(tmp_path, "AAA", ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"], [10, 11, 12, 13])
    _tiny_store(tmp_path, "BBB", ["2020-01-02", "2020-01-03", "2020-01-07"], [20, 21, 23])

    panel, dropped = fetch.load_panel(["AAA", "BBB"], tmp_path, start="2020-01-02", min_coverage=0.5)

    assert dropped == {}
    assert sorted(panel.columns) == ["AAA", "BBB"]
    assert [str(d.date()) for d in panel.index] == ["2020-01-02", "2020-01-03", "2020-01-07"], (
        "the hole was filled instead of intersected: the panel kept a day BBB did not trade"
    )
    assert panel.loc[pd.Timestamp("2020-01-07"), "BBB"] == 23.0
    assert (panel["BBB"].pct_change().dropna() != 0).all(), "a fabricated zero return survived"


def test_every_kept_asset_predates_the_requested_start(panel_and_drops, closes):
    panel, _ = panel_and_drops
    for sym in panel.columns:
        first = closes[sym].dropna().index.min()
        assert first <= pd.Timestamp("2011-01-03"), f"{sym} starts {first.date()}"


def test_every_drop_carries_a_reason(panel_and_drops, stored):
    panel, dropped = panel_and_drops
    assert set(panel.columns) | set(dropped) == set(stored)
    assert set(panel.columns).isdisjoint(dropped)
    for sym, why in dropped.items():
        assert why and isinstance(why, str), f"{sym} dropped with no reason"


def test_the_panel_is_a_plausible_trading_year(panel_and_drops):
    """252 bars/year measured, not assumed. The annualisation constant is used everywhere
    downstream, and a panel at 365 bars/year would mean the index is a calendar."""
    panel, _ = panel_and_drops
    span_years = (panel.index.max() - panel.index.min()).days / 365.25
    bars_per_year = len(panel) / span_years
    assert 248 < bars_per_year < 254, f"{bars_per_year:.1f} bars/year"


# ----------------------------------------------------------------------- monthly_returns


def test_monthly_returns_telescope_to_the_daily_total_growth(panel_and_drops):
    """The identity the SPA's growth curve rests on, and the reason the series is anchored at
    the panel's first bar rather than at the first month end.

    prod(1 + monthly return) must equal P_last / P_first EXACTLY, because chained month-end
    price ratios cancel. Without the anchor the product starts at the first month end and the
    error is proportional to the asset's first-month move -- 1.0 percentage point of
    annualised return on GDXJ, and near zero on a T-bill fund, which is the worst possible
    shape for a bug: invisible where it is checked casually.
    """
    panel, _ = panel_and_drops
    monthly = fetch.monthly_returns(panel)
    for sym in panel.columns:
        chained = float(np.prod(1.0 + monthly[sym].to_numpy()))
        direct = float(panel[sym].iloc[-1] / panel[sym].iloc[0])
        assert chained == pytest.approx(direct, rel=1e-12), f"{sym}: {chained} vs {direct}"


def test_monthly_returns_have_one_row_per_month_plus_the_anchor(panel_and_drops):
    panel, _ = panel_and_drops
    monthly = fetch.monthly_returns(panel)
    months = pd.period_range(panel.index.min(), panel.index.max(), freq="M")
    assert len(monthly) == len(months), f"{len(monthly)} returns for {len(months)} months"
    assert monthly.index.is_monotonic_increasing
    assert not monthly.isna().to_numpy().any()


def test_monthly_returns_are_bounded_like_returns_not_levels(panel_and_drops):
    panel, _ = panel_and_drops
    monthly = fetch.monthly_returns(panel)
    v = monthly.to_numpy()
    assert (v > -1.0).all(), "a monthly return at or below -100%"
    assert np.abs(v).max() < 2.0, f"largest monthly move {np.abs(v).max():.2f}: these look like levels"
