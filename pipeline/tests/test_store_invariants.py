"""Invariants for the price store: the partition layout, and the dividend reconstruction.

Two kinds of test here, and they answer different questions.

The LAYOUT tests are unit tests on a temporary store, because the properties being checked are
about the writer: that a year partition contains only that year, that re-writing a symbol
replaces it instead of duplicating it, and that a symbol paying nothing still clears its old
event rows. Those are all "what happens on the second write", which the shipped store cannot
show -- it only ever shows the result.

The RECONSTRUCTION test reads the real store, and it is the one that makes this layout
trustworthy: the store deliberately does NOT keep the vendor's `adjclose` column, so the claim
"our total returns are the vendor's total returns" has to be checked against something. That
something is `adjclose_sample.parquet`, 24 vendor bars per symbol recorded at fetch time. If
this test fails, the number the optimiser is estimated from is wrong -- not the storage.
"""
from __future__ import annotations

import tomllib

import numpy as np
import pandas as pd
import pytest

import pit
import store
import universe as uni
from conftest import PRICES


# ------------------------------------------------------------------ the adjustment, in the small


def _series(prices, start="2020-01-02"):
    idx = pd.bdate_range(start, periods=len(prices), name="date")
    return pd.Series(np.asarray(prices, dtype=float), index=idx)


def test_a_series_with_no_dividends_is_returned_unchanged():
    c = _series([100.0, 101.0, 102.0])
    for divs in (None, pd.Series(dtype=float)):
        out = store.total_return(c, divs)
        assert np.array_equal(out.to_numpy(), c.to_numpy())


def test_the_adjustment_scales_bars_before_the_ex_date_and_leaves_the_rest():
    """The definition, at its boundary. A dividend with ex-date on bar 2 scales bars 0 and 1 by
    (1 - D/P_1) and leaves bars 2 onward alone -- the ex-date bar is NOT adjusted, because its
    own price already reflects the distribution. Off-by-one here is invisible in aggregate
    (one bar in 4,000) and shows up as a spurious return on exactly the ex-date."""
    c = _series([100.0, 200.0, 300.0, 400.0])
    divs = pd.Series([10.0], index=[c.index[2]])
    out = store.total_return(c, divs)
    f = 1.0 - 10.0 / 200.0  # prior close is bar 1, not bar 2
    assert out.iloc[0] == pytest.approx(100.0 * f)
    assert out.iloc[1] == pytest.approx(200.0 * f)
    assert out.iloc[2] == pytest.approx(300.0)
    assert out.iloc[3] == pytest.approx(400.0)


def test_the_last_bar_is_never_adjusted_whatever_the_dividend_history():
    """Backward adjustment anchors at the most recent bar by construction. This is the property
    that makes `close/adjusted` at the END useless as a check and the ratio at the START the
    only informative one -- so it is asserted rather than assumed.

    THE DIVIDEND ON THE LAST BAR IS THE POINT, and this test did not have one until the mutation
    harness said so: with ex-dates at bars 2, 4, 6 and 8 of ten, `factor[:pos + 1]` -- the
    off-by-one at the ex-date boundary -- still leaves bar 9 alone, so "whatever the dividend
    history" was a claim about four histories that all excluded the only one that could fail.
    An ex-date ON the final bar is exactly the case the anchor has to survive, and it is not
    hypothetical: it happens for every payer in the store whose last bar IS an ex-date.
    """
    c = _series([100.0] * 10)
    divs = pd.Series([1.0] * 5, index=[c.index[i] for i in (2, 4, 6, 8, 9)])
    out = store.total_return(c, divs)
    assert out.iloc[-1] == c.iloc[-1], "the final bar was adjusted; the series has no anchor"
    assert out.iloc[-2] < c.iloc[-2], "the last dividend was not applied to the bar before it"
    assert out.iloc[0] < c.iloc[0]


def test_a_dividend_at_or_before_the_first_bar_is_skipped_not_applied():
    """There is no prior close to scale by and no earlier bar to scale. Reaching for
    `close.iloc[-1]` via a negative index -- which is what `pos - 1` does at pos == 0 -- would
    silently apply the factor computed from the LAST price to the whole series."""
    c = _series([100.0, 110.0, 120.0])
    divs = pd.Series([5.0, 5.0], index=[c.index[0] - pd.Timedelta(days=7), c.index[0]])
    out = store.total_return(c, divs)
    assert np.array_equal(out.to_numpy(), c.to_numpy())


def test_a_dividend_at_or_above_the_price_raises_rather_than_flipping_the_series():
    """(1 - D/P) at D >= P is zero or negative, which turns a price history into zeros or into
    a sign-flipped one. Both are catastrophic and neither looks like a missing value, so a bad
    record has to stop the build."""
    c = _series([100.0, 100.0, 100.0])
    for amount in (100.0, 250.0):
        with pytest.raises(ValueError, match="prior close"):
            store.total_return(c, pd.Series([amount], index=[c.index[2]]))


def test_dividends_compound_across_events_rather_than_summing():
    """Two dividends before a bar apply as a PRODUCT of factors, not a sum of amounts. Over 135
    SPY distributions the two differ by several percent of terminal wealth, in the direction
    that overstates the total return."""
    c = _series([100.0] * 5)
    divs = pd.Series([1.0, 2.0], index=[c.index[2], c.index[4]])
    out = store.total_return(c, divs)
    assert out.iloc[0] == pytest.approx(100.0 * (1 - 0.01) * (1 - 0.02))
    assert out.iloc[0] != pytest.approx(100.0 * (1 - 0.03))


# --------------------------------------------------------------------------- the layout, in the small


def _write(tmp, symbol, dates, closes, adj=None, divs=()):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
    df = pd.DataFrame(
        {"close": np.asarray(closes, dtype=float),
         "adjclose": np.asarray(adj if adj is not None else closes, dtype=float)},
        index=idx,
    )
    ev = pd.DataFrame(
        [{"symbol": symbol, "date": pd.Timestamp(d), "kind": "dividend", "amount": a} for d, a in divs],
        columns=["symbol", "date", "kind", "amount"],
    )
    store.write(tmp, {symbol: df}, {symbol: ev})


def test_a_year_partition_contains_only_that_year(tmp_path):
    """The one property the whole layout rests on: if a bar can land in the wrong partition, a
    frozen historical year is not frozen and the churn saving is imaginary."""
    _write(tmp_path, "AAA", ["2023-12-28", "2023-12-29", "2024-01-02", "2025-06-02"], [1.0, 2.0, 3.0, 4.0])
    years = sorted(int(p.stem) for p in store.close_dir(tmp_path).glob("*.parquet"))
    assert years == [2023, 2024, 2025]
    for y in years:
        df = pd.read_parquet(store.close_dir(tmp_path) / f"{y}.parquet")
        assert (df["date"].dt.year == y).all(), f"{y}.parquet holds bars from another year"


def test_rewriting_unchanged_history_touches_no_file(tmp_path):
    """The property the churn saving actually rests on, and the one most easily lost.

    The collector has no incremental endpoint -- it refetches each symbol's FULL history every
    run -- so a weekly refresh hands every year partition a chunk identical to what it holds.
    If those get written back, 34 files change every week, git stores 34 new compressed blobs,
    and the partitioned layout costs exactly what the per-symbol layout did.

    Checked on mtime AND on the reported year list, because either alone is weak: mtime alone
    passes on a writer that reports honestly and writes anyway on a filesystem with coarse
    timestamps, and the year list alone passes on a writer that reports nothing and writes
    everything.
    """
    dates = ["2023-06-01", "2024-06-03", "2025-06-02"]
    _write(tmp_path, "AAA", dates, [1.0, 2.0, 3.0], divs=[("2024-06-03", 0.1)])
    files = sorted(tmp_path.rglob("*.parquet"))
    assert len(files) == 5, [p.name for p in files]  # three years + events + sample
    before = {p: p.stat().st_mtime_ns for p in files}

    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "adjclose": [1.0, 2.0, 3.0]}, index=idx)
    ev = pd.DataFrame(
        [{"symbol": "AAA", "date": pd.Timestamp("2024-06-03"), "kind": "dividend", "amount": 0.1}]
    )
    assert store.write(tmp_path, {"AAA": df}, {"AAA": ev}) == [], "an unchanged year was reported as changed"
    assert {p: p.stat().st_mtime_ns for p in sorted(tmp_path.rglob("*.parquet"))} == before, (
        "a file was rewritten with identical content -- git cannot delta parquet, so this is a "
        "whole new blob in the history for no new data"
    )

    # And it still writes when something DID change, or the check above is satisfied by a writer
    # that never writes at all.
    _write(tmp_path, "AAA", [*dates, "2026-06-01"], [1.0, 2.0, 3.0, 4.0], divs=[("2024-06-03", 0.1)])
    assert store.read_close(tmp_path)["AAA"].dropna().tolist() == [1.0, 2.0, 3.0, 4.0]


def test_rewriting_a_symbol_replaces_its_rows_instead_of_duplicating_them(tmp_path):
    """A refetch has to be idempotent, and it has to CORRECT. Appending blindly would give the
    symbol two closes on the same date, which `read_close` would silently resolve to one of
    them -- so a bad print would become permanent and invisible."""
    _write(tmp_path, "AAA", ["2024-01-02", "2024-01-03"], [10.0, 999.0])
    _write(tmp_path, "AAA", ["2024-01-02", "2024-01-03", "2024-01-04"], [10.0, 11.0, 12.0])
    wide = store.read_close(tmp_path)
    assert list(wide.index.astype(str).str[:10]) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert wide["AAA"].tolist() == [10.0, 11.0, 12.0], "the corrected print did not win"
    raw = pd.read_parquet(store.close_dir(tmp_path) / "2024.parquet")
    assert not raw.duplicated(["symbol", "date"]).any()


def test_writing_one_symbol_leaves_the_others_alone(tmp_path):
    """The upsert filters by symbol, so a partial run must not delete the symbols it did not
    fetch. This is the failure that turns a rate-limited refresh into a smaller universe."""
    _write(tmp_path, "AAA", ["2024-01-02"], [1.0], divs=[("2024-01-02", 0.1)])
    _write(tmp_path, "BBB", ["2024-01-02"], [2.0])
    _write(tmp_path, "AAA", ["2024-01-02", "2024-01-03"], [1.0, 1.5])
    wide = store.read_close(tmp_path)
    assert sorted(wide.columns) == ["AAA", "BBB"]
    assert wide["BBB"].dropna().tolist() == [2.0]


def test_a_symbol_that_stops_paying_has_its_old_events_cleared(tmp_path):
    """`write_events` is passed the symbol set, not just the non-empty frames. Keying off the
    frames alone would leave a withdrawn or mis-parsed distribution in the log forever, still
    being applied to every earlier bar by `total_return`."""
    _write(tmp_path, "AAA", ["2024-01-02", "2024-01-03"], [1.0, 1.0], divs=[("2024-01-03", 0.1)])
    assert len(store.dividends_by_symbol(tmp_path)["AAA"]) == 1
    _write(tmp_path, "AAA", ["2024-01-02", "2024-01-03"], [1.0, 1.0], divs=[])
    assert "AAA" not in store.dividends_by_symbol(tmp_path)


def test_last_dates_reports_the_newest_bar_per_symbol(tmp_path):
    _write(tmp_path, "AAA", ["2025-12-30", "2026-01-05"], [1.0, 1.0])
    _write(tmp_path, "BBB", ["2025-12-30"], [1.0])
    last = store.last_dates(tmp_path)
    assert last["AAA"] == pd.Timestamp("2026-01-05")
    assert last["BBB"] == pd.Timestamp("2025-12-30"), (
        "only the newest partition was read, so a symbol that stopped printing last year "
        "reports nothing and the freshness check refetches the whole universe every January"
    )


def test_the_adjclose_sample_always_includes_the_first_and_last_bar(tmp_path):
    """The first bar is where the accumulated adjustment is largest, so it is where the
    reconstruction check has power. A sample that skipped it would pass on a series adjusted by
    a factor of one."""
    n = 500
    dates = pd.bdate_range("2020-01-02", periods=n).strftime("%Y-%m-%d").tolist()
    _write(tmp_path, "AAA", dates, np.arange(1.0, n + 1.0), adj=np.arange(0.5, n * 1.0, 1.0))
    s = store.read_adjclose_sample(tmp_path)
    assert len(s) == store.SAMPLE_BARS
    assert s["date"].min() == pd.Timestamp(dates[0])
    assert s["date"].max() == pd.Timestamp(dates[-1])


# -------------------------------------------------- the reconstruction, against the real store


@pytest.fixture(scope="module")
def sampled():
    s = store.read_adjclose_sample(PRICES)
    if s.empty:
        pytest.skip("no local price store -- run `python -u pipeline/build.py` first")
    return s


def test_reconstruction_matches_the_vendors_adjclose(sampled):
    """THE test for this layout. The store keeps `close` plus a dividend log and rebuilds the
    total-return series; the vendor ships the answer directly. They must agree, or the price
    the optimiser sees is not a total return and every number downstream is wrong in a way
    nothing else here would notice.

    The tolerance is 1e-5 relative and it is not slack for the model -- it is the vendor's own
    transport precision. Yahoo's `adjclose` arrives as float32 (761.78 comes back as
    761.7800292969, exactly representable at 24 bits and at no decimal count), which is ~1e-7
    relative per value, and its published adjustment factors carry their own rounding. Measured
    across this universe the residual runs ~1e-7 median and ~1e-6 worst case, so 1e-5 is two
    orders of margin over the noise and still an order tighter than any error that would matter
    -- a single missed dividend on a 2%-yield payer is a 2e-2 gap, not a 1e-5 one.

    Checked at the SAMPLED bars, which include each symbol's first, because a reconstruction
    error is systematic: a wrong or missing factor propagates to every bar before it.

    THE DECLARED-TRUNCATED SYMBOLS ARE EXCLUDED, AND THE EXCLUSION IS NOT A TOLERANCE. LEG is the
    case: the vendor serves five bars for it, 2026-07-17 and then a jump to 2026-08-24, and the
    0.05 dividend of 2026-08-10 falls in that hole. The store adjusts backwards across the gap --
    which is what its contract says it does -- and the vendor's own `adjclose` for the 2026-07-17
    bar equals its raw close, so the vendor contradicts its own event log on a series it truncated.
    Nothing here can reconcile that and no threshold should absorb it: 4.5e-3 is the size of a real
    missed dividend. The names come off `sp500_pit.toml`'s `[truncated]` table, so deleting a
    declaration brings the failure straight back, and each one is asserted to actually BE truncated
    below -- an entry cannot be used to excuse a full series.
    """
    close = store.read_close(PRICES)
    divs = store.dividends_by_symbol(PRICES)
    truncated = set(pit.load_membership()["truncated"])
    worst = (0.0, None)
    for sym, chunk in sampled.groupby("symbol"):
        sym = str(sym)
        if sym in truncated:
            continue
        assert sym in close.columns, f"{sym} has an adjclose sample but no stored closes"
        recon = store.total_return(close[sym].dropna(), divs.get(sym))
        got = recon.reindex(pd.DatetimeIndex(chunk["date"]))
        assert got.notna().all(), f"{sym}: sampled dates missing from the stored closes"
        err = np.abs(got.to_numpy() / chunk["adjclose"].to_numpy() - 1.0)
        if err.max() > worst[0]:
            worst = (float(err.max()), sym)
        assert err.max() < 1e-5, (
            f"{sym}: reconstructed total return differs from the vendor's adjclose by "
            f"{err.max():.2e} relative -- that is a dividend the log is missing or applying "
            "wrongly, not transport rounding"
        )
    print(f"\nworst reconstruction residual: {worst[0]:.2e} ({worst[1]})")


def test_every_point_in_time_member_is_stored_back_to_the_snapshot_that_holds_it():
    """THE SCREEN WHOSE ABSENCE LET FOUR LARGE CAPS OUT OF EVERY CANDIDATE SET WITHOUT A SOUND.

    A membership snapshot dated 2016-09-01 is a claim that those companies could have been bought
    that day, and the store has to be able to price them then. The primary vendor served AVB, EA,
    LEG and BBBY under the RIGHT company names with 5 to 36 bars, all beginning on the same day,
    2026-07-17 -- so it is one vendor defect and not four coincidences, and none of the four had
    stopped trading. Nothing failed: `load_panel`'s coverage filter drops a 27-bar series from
    every window and reports it under `dropped_history_starts_late`, in the same count as the
    genuine post-window IPOs, so three S&P 100 members were simply absent from all ten windows and
    the run looked healthy.

    So the rule is stated the other way round, as a property of the store rather than of a run: a
    symbol some snapshot holds must be stored back to the FIRST snapshot that holds it, or be
    named in `[truncated]` with the measured reason. Not in the store at all is a different and
    already-reported thing -- `vendor_two` and `unpriceable` cover those, and this test says so
    rather than conflating "the vendor serves nothing" with "the vendor serves a stub".

    IT THEN FOUND SIX MORE, of two kinds the four had not shown, and both are worse than a stub:
    APC serves 142 bars from 2026-02-12 as ARKO Petroleum, which is the PARA case again (a
    different company under a recycled ticker, every automatic check passing); DOW, FOX, FOXA and
    Q serve the right company from a date years after the first snapshot that holds them, because
    each ticker changed hands -- Dow Chemical to Dow Inc, Twenty-First Century Fox to Fox
    Corporation, QuintilesIMS to Qnity. The remedy differs by kind, so the file has three tables
    and this test names all three: `[recycled]` strips a ticker from every snapshot, `[reissued]`
    from the ones before its date, `[truncated]` from none. The four reissued names then pass here
    ON MERIT rather than by exemption, because the strip moves the first snapshot that holds them
    to one their stored history reaches -- which is the point of separating the tables.
    """
    mem = pit.load_membership()
    held = pit.held_between(mem)
    close = store.read_close(PRICES)
    declared = set(mem["truncated"]) | set(mem["vendor_two"]) | set(mem["unpriceable"])
    stubs = {}
    for sym, (first_snapshot, _) in sorted(held.items()):
        if sym not in close.columns or sym in declared:
            continue
        s = close[sym].dropna()
        if len(s) and s.index.min() > pd.Timestamp(first_snapshot):
            stubs[sym] = (str(s.index.min().date()), len(s), first_snapshot)
    assert not stubs, (
        "stored history begins after the first snapshot that holds these -- symbol: "
        f"(first stored bar, bars, snapshot needing it) {stubs}. Three remedies, and the right one "
        "is a question about the company, not about the length of the series: the vendor truncated "
        "it -> sp500_pit.toml's [truncated]; the ticker now belongs to someone else -> [recycled]; "
        "it belonged to someone else BEFORE a date and to a current member after -> [reissued]. "
        "Each with the measured reason."
    )
    raw = tomllib.load(open(pit.MEMBERSHIP, "rb"))["snapshots"]
    for sym, spec in sorted(mem["reissued"].items()):
        before = [d for d, syms in raw.items() if sym in syms and d < spec["from"]]
        assert before, (
            f"{sym} is declared reissued from {spec['from']} but no snapshot before that date "
            "holds it, so the declaration strips nothing. An entry that does no work is one that "
            "will not be noticed when a later snapshot needs it."
        )
    for sym in sorted(mem["truncated"]):
        if sym not in close.columns:
            continue
        n = int(close[sym].dropna().shape[0])
        assert n < 250, (
            f"{sym} is declared truncated but has {n} stored bars. A declaration excuses a symbol "
            "from the reconstruction check, so one covering a full series would hide a real "
            "missed dividend."
        )


def test_every_dividend_payer_in_the_store_actually_moves_the_series(sampled):
    """A dividend log that parsed into the right shape but the wrong values -- zeros, say --
    would sail through the reconstruction check for the non-payers and be caught only in
    aggregate. So: for every symbol the log says pays, the adjustment must be visible."""
    close = store.read_close(PRICES)
    divs = store.dividends_by_symbol(PRICES)
    assert len(divs) > 50, f"only {len(divs)} symbols have any dividend record"
    for sym, d in divs.items():
        if sym not in close.columns:
            continue
        c = close[sym].dropna()
        if len(c) < 250 or d.sum() <= 0:
            continue
        recon = store.total_return(c, d)
        assert recon.iloc[0] < c.iloc[0], f"{sym} has {len(d)} dividends but no adjustment at the first bar"


# A SPLIT BAR THAT ALSO CARRIES A DISTRIBUTION IS A CORPORATE ACTION, NOT A SPLIT, and no
# threshold on a price move can be applied to it. Yahoo records a spinoff as both: a split ratio
# AND a large cash-equivalent dividend on the same date. Three in the store, all real --
#
#   JCI  2007-07-02  split 0.25   $10.69 on a $70 close   (the Tyco/Covidien separation)
#   EXPE 2011-12-21  split 0.5    $15.13 on a $57 close   (the TripAdvisor spinoff)
#   DHR  2016-07-05  split 1.319  $24.56 on a $71 close   (Fortive)
#
# -- and each one moves the close by 50-61% in a single bar, legitimately, because shareholders
# received shares in another company. `store.total_return` reproduces the vendor's own adjusted
# close across all three to better than 5e-7, which is the test that they were handled correctly
# (`test_total_return_matches_the_vendors_adjusted_close`). It is a stronger check than any
# threshold, so these bars are handed to it rather than waved through: excluded HERE, asserted
# exactly THERE. Adding a same-day distribution to a bar to silence a failure below is therefore
# not available -- the distribution is in the event log or it is not.
def _held_symbols() -> set[str]:
    """Every symbol some universe file lists in [assets] -- i.e. every symbol that can reach an
    estimate. NOT every symbol in the store.

    The store is keyed by symbol and append-only, so it accumulates series no universe holds:
    an old candidate, a benchmark fetched for a comparison, and -- the case this exists for -- a
    RECYCLED TICKER. `PARA` was Paramount Global, a 2023 S&P 500 constituent; the vendor now
    serves Banzai International under it, a micro-cap that has done three reverse splits since,
    and its stored closes really are discontinuous across them. That is a broken series and the
    two tests below are right to say so, but the fix is not a threshold or an exception entry --
    it is that `sp500_2023.toml` excludes the ticker, because the series is two companies and no
    weight vector may contain it.

    Scoping to held symbols is therefore narrowing the tests to exactly the claim they make. Their
    rationale is that one unadjusted bar dominates an asset's covariance row, and a symbol no
    universe holds has no covariance row. It is also self-maintaining rather than a list: add
    `PARA` to any universe's [assets] and the failure comes straight back.

    THE POINT-IN-TIME MEMBERSHIP IS A SECOND SOURCE OF HELD SYMBOLS AND IS READ THROUGH ITS OWN
    LOADER. `uni.available()` globs `pipeline/universes/*.toml` and `sp500_pit.toml` lives there
    while having no `[assets]` at all -- it is a set of symbols PER DATE -- so `uni.load` raises on
    it, correctly, and `universe.py` must not learn to read it (CLAUDE.md). Its snapshots are
    unioned in here instead, with the recycled tickers already stripped by `pit.load_membership`,
    because every one of those symbols can reach an estimate in some window and the claim these
    tests make is about exactly that set.
    """
    keys = [k for k in uni.available() if k != pit.MEMBERSHIP.stem]
    held = {s for key in keys for s in uni.load(key).symbols}
    mem = pit.load_membership()
    return held | {s for syms in mem["snapshots"].values() for s in syms}


def _split_bars_with_no_distribution():
    """Every stored split bar that is a plain split on a HELD symbol: (symbol, pos, series, amount)."""
    splits = store.read_events(PRICES, kind="split")
    if splits.empty:
        return []
    divs = store.read_events(PRICES, kind="dividend")
    paid = {(r.symbol, r.date) for r in divs.itertuples()}
    held = _held_symbols()
    close = store.read_close(PRICES)
    out = []
    for row in splits.itertuples():
        if row.symbol not in held or row.symbol not in close.columns:
            continue
        if (row.symbol, row.date) in paid:
            continue
        c = close[row.symbol].dropna()
        pos = int(c.index.searchsorted(row.date, side="left"))
        if pos == 0 or pos >= len(c):
            continue
        out.append((row.symbol, pos, c, row.amount, row.date))
    return out


# One vendor bar, listed rather than tolerated by a looser threshold. BRO's close is split-
# adjusted either side of 1983-03-24 (the level across it moves 1.3%) and it pays no distribution
# there, but the split-day bar ITSELF is unadjusted: 0.222222 where its neighbours print 0.148148
# and 0.150000, exactly 1.5x, reverting the next day. That is a defect in one bar, not a failure of
# the split adjustment, and the two assertions below are separated so it cannot hide one behind the
# other. Listed with its date so a SECOND one fails loudly -- a threshold raised to 0.51 would have
# absorbed it silently, and this whole file exists because silent absorption is how a covariance
# row gets ruined. It is 1983, which is before the start date of every universe here; a bar inside
# a window a build uses would have to be fixed rather than listed.
KNOWN_UNADJUSTED_SPLIT_BARS = {("BRO", "1983-03-24")}


def test_close_is_split_adjusted_across_every_stored_split(sampled):
    """The claim `close` arrives already split-adjusted, checked against the split log rather
    than asserted. This is what makes storing splits worth it: an unadjusted 1:8 reverse split
    is a +700% single-day return, and one such row dominates that asset's whole covariance row
    -- but the same jump with no event log to point at is just an outlier of unknown cause.

    Measured across the split rather than INTO it -- the bar before against the bar after. Those
    are two adjusted bars whatever the vendor did with the split-day bar between them, so this
    assertion is about the adjustment and nothing else. The split-day bar is the next test's.
    """
    bars = _split_bars_with_no_distribution()
    if not bars:
        pytest.skip("no plain splits in the stored universe")
    checked = 0
    for symbol, pos, c, amount, date in bars:
        if pos + 1 >= len(c):
            continue
        move = abs(float(c.iloc[pos + 1] / c.iloc[pos - 1] - 1.0))
        assert move < 0.50, (
            f"{symbol}: {move:.1%} across its {amount} split on {date.date()} (the bar before "
            f"against the bar after, and no distribution on that bar to explain it) -- `close` "
            f"is NOT split-adjusted after all"
        )
        checked += 1
    assert checked > 0, "no plain split fell inside a stored history, so this checked nothing"


def test_no_split_day_bar_is_a_split_sized_spike(sampled):
    """The other half: a single bar the vendor forgot to adjust, which reverts the next day.

    Harmless to the *level* and just as damaging to an estimate -- one unadjusted bar is a pair of
    huge equal-and-opposite daily returns, and a variance does not care that they cancel. Checked
    separately from the adjustment itself because the failure, the cause and the fix are all
    different: that one is the vendor's split handling, this one is a bad print.
    """
    bars = _split_bars_with_no_distribution()
    if not bars:
        pytest.skip("no plain splits in the stored universe")
    spikes = []
    for symbol, pos, c, amount, date in bars:
        move = abs(float(c.iloc[pos] / c.iloc[pos - 1] - 1.0))
        if move >= 0.50 and (symbol, str(date.date())) not in KNOWN_UNADJUSTED_SPLIT_BARS:
            spikes.append(f"{symbol} {date.date()} ({amount} split): {move:.1%}")
    assert not spikes, (
        "split-day bars the vendor did not adjust, reverting the next day:\n  "
        + "\n  ".join(spikes)
        + "\nEach is two equal-and-opposite daily returns in one asset's estimate. Add it to "
          "KNOWN_UNADJUSTED_SPLIT_BARS only after checking the level across it is continuous, "
          "and only if it falls outside every window a build uses."
    )
