"""Invariants of the point-in-time membership walk-forward: `pit.py`.

DELIBERATELY SYNTHETIC, for the reason the repo's rule allows: every property here is one the
shipped files cannot express. `sp500_pit.toml` is valid, so no artifact built from it can show what
a malformed `[reissued]` entry does; its eleven snapshots share ~95% of their names, so no run over
it can distinguish "each window used its own membership" from "every window used the 2016 list"
where they overlap; and the two window-length assertions in `run` fire on a panel that does not
reach its requested ends, which is exactly the panel the shipped store never produces.

The one thing NOT tested here is estimation, solving or measurement. `pit.run` calls
`backtest.evaluate` unchanged, and `test_backtest_invariants.py` is where that is interrogated --
a second copy of those assertions here would be a second thing to keep in step with the first.
What is tested is the seam: the membership arithmetic, the per-window dates, and the bookkeeping
the stress bound reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pit  # noqa: E402
import store  # noqa: E402

CUTOFFS = ["2017-01-02", "2018-01-01"]
SYMBOLS = [f"S{i}" for i in range(8)]


# ------------------------------------------------------------------ the membership arithmetic

def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "mem.toml"
    path.write_text('name = "test"\nretrieved = "2026-01-01"\n' + body)
    return path


SNAPS = """
[snapshots]
"2016-09-01" = ["AAA", "OLD", "TURN"]
"2020-09-01" = ["AAA", "OLD", "TURN"]
"""


def test_a_recycled_ticker_leaves_every_snapshot_and_a_reissued_one_only_the_early_ones(tmp_path):
    """The two tables exist because they have different remedies, and this is that difference.

    A ticker whose current occupant is a different company must be gone from every snapshot. A
    ticker that meant one company before a date and a CURRENT MEMBER after it must be gone from the
    snapshots before that date only -- applying the recycled rule to it deletes Dow Inc from seven
    snapshots of an index it was in, which is a survivorship error in the other direction and just
    as invisible.
    """
    path = _write(tmp_path, SNAPS + """
[recycled]
OLD = "someone else entirely"

[reissued]
TURN = { from = "2019-03-20", why = "a different company before this date" }
""")
    mem = pit.load_membership(path)
    assert mem["snapshots"]["2016-09-01"] == ["AAA"]
    assert mem["snapshots"]["2020-09-01"] == ["AAA", "TURN"]


def test_every_removal_is_reported_per_snapshot_with_the_table_that_made_it(tmp_path):
    """`removed` is not diagnostics. A stripped name is a CONSTITUENT NOBODY PRICES and it never
    reaches `load_panel`, so it appears in no `dropped` reason -- the stress bound would leave it
    out of the set it bounds while being precisely about it. On the shipped file that is 12 names
    on the 2016 snapshot, against a `dropped_not_in_store` of 72 -- so the window is missing 84
    constituents and the membership file's own `unpriceable` list names 67."""
    path = _write(tmp_path, SNAPS + """
[recycled]
OLD = "someone else entirely"

[reissued]
TURN = { from = "2019-03-20", why = "a different company before this date" }
""")
    mem = pit.load_membership(path)
    assert set(mem["removed"]["2016-09-01"]) == {"OLD", "TURN"}
    assert set(mem["removed"]["2020-09-01"]) == {"OLD"}
    assert mem["removed"]["2016-09-01"]["OLD"] == "recycled"
    assert "2019-03-20" in mem["removed"]["2016-09-01"]["TURN"]


def test_a_reissue_declaration_without_a_date_or_a_reason_raises(tmp_path):
    """A validator can only be tested with input the shipped file does not contain. `from` missing
    strips nothing and the ticker stays in every snapshot as the wrong company; `why` missing is a
    declaration with no measurement behind it, which is the thing every table in that file is
    written to avoid."""
    for body in ['TURN = { why = "no date" }', 'TURN = { from = "2019-03-20" }']:
        path = _write(tmp_path, SNAPS + "\n[reissued]\n" + body + "\n")
        with pytest.raises(ValueError, match="needs both"):
            pit.load_membership(path)


def test_a_symbol_declared_both_recycled_and_reissued_raises(tmp_path):
    """The two remedies contradict each other, and the order the loader happens to apply them in
    would decide which one wins -- silently, and differently if the loop were ever rewritten."""
    path = _write(tmp_path, SNAPS + """
[recycled]
TURN = "gone for good"

[reissued]
TURN = { from = "2019-03-20", why = "back later" }
""")
    with pytest.raises(ValueError, match="both"):
        pit.load_membership(path)


def test_held_between_reports_the_first_and_last_snapshot_that_holds_a_symbol(tmp_path):
    path = _write(tmp_path, """
[snapshots]
"2016-09-01" = ["AAA", "BBB"]
"2017-09-01" = ["AAA"]
"2018-09-01" = ["AAA", "BBB"]
""")
    held = pit.held_between(pit.load_membership(path))
    assert held["AAA"] == ("2016-09-01", "2018-09-01")
    assert held["BBB"] == ("2016-09-01", "2018-09-01"), (
        "held_between reports the OUTER dates; a symbol absent from a middle snapshot is still a "
        "symbol whose history has to overlap the interval"
    )


@pytest.mark.parametrize("first,last,expect", [
    ("2019-01-02", "2026-01-02", "entirely after"),      # reissued to someone else
    ("2010-01-04", "2015-01-02", "entirely before"),     # a predecessor's series
    ("2013-01-02", "2019-01-02", None),                   # overlaps: plausibly the constituent
    ("2016-09-01", "2016-09-01", None),                   # one bar, inside the interval
])
def test_a_series_is_rejected_exactly_when_it_cannot_be_the_constituents_history(first, last, expect):
    """The screen is a NECESSARY condition, and both directions have to be checked because each
    catches a different real case: `entirely after` is a reissued ticker (CA, DNB on the shipped
    membership), `entirely before` is a series that stops before the index ever held the symbol.
    A one-sided version passes one of them, and the failure is a different company's returns in a
    candidate set."""
    held = {"XXX": ("2016-09-01", "2018-09-01")}
    why = pit.overlaps_its_membership("XXX", pd.Timestamp(first), pd.Timestamp(last), held)
    if expect is None:
        assert why is None
    else:
        assert why is not None and expect in why


def test_a_symbol_no_snapshot_holds_is_rejected_rather_than_admitted():
    """`held` is built from the snapshots AFTER the recycled and reissued strips, so a symbol the
    file removed is a symbol not in `held` -- and returning None for an unknown key would let the
    strip be undone by the very screen that exists to catch what it removes."""
    assert pit.overlaps_its_membership(
        "GHOST", pd.Timestamp("2016-01-04"), pd.Timestamp("2020-01-02"), {}) is not None


def test_the_screen_rejects_a_cached_series_that_cannot_be_the_constituent(tmp_path):
    """`screen` is the join between the second vendor's cache and the membership, and it is a
    separate function because the stress bound reads the same cache: an unscreened series there
    supplies a different company's growth multiple as the upper bound on a constituent nobody
    prices."""
    path = _write(tmp_path, """
[snapshots]
"2016-09-01" = ["GOOD", "LATE"]
"2018-09-01" = ["GOOD"]
""")
    mem = pit.load_membership(path)
    mem["vendor_two"] = ["GOOD", "LATE", "ABSENT"]
    raw = tmp_path / "raw"
    raw.mkdir()
    for sym, span in {"GOOD": ("2012-01-02", "2017-06-01"),
                      "LATE": ("2020-01-02", "2026-01-02")}.items():
        idx = pd.bdate_range(*span)
        pd.DataFrame({"close": 10.0, "adjusted": 10.0, "volume": 1e6},
                     index=pd.Index(idx, name="date")).to_csv(raw / f"{sym}.csv")
    kept, rejected = pit.screen(mem, tmp_path)
    assert kept == ["GOOD"]
    assert "entirely after" in rejected["LATE"]
    assert "nothing usable" in rejected["ABSENT"], (
        "a symbol with no cached file has to be reported as a rejection, not skipped -- otherwise "
        "the count of names the second vendor delivered is the count of files that happen to exist"
    )


# --------------------------------------------------------------------------- the windows

def test_the_training_window_is_fixed_length_and_measured_back_from_the_cutoff():
    """Fixed-length, so it does not lengthen every week the store refreshes -- the property that
    makes two runs of this file a month apart comparable."""
    train_from, test_to = pit.window_dates("2020-09-01", 5.0, 1.0)
    assert str(train_from.date()) == "2015-09-01"
    assert str(test_to.date()) == "2021-09-01"


def _store(price_dir: Path, symbols: list[str], end: str = "2019-12-31",
           after: str | None = None, factor: float = 1.0) -> None:
    """A store of independent random walks. `after`/`factor` scale every bar strictly after a date,
    which is how the no-lookahead test moves the future without touching the past."""
    idx = pd.bdate_range("2014-06-02", end)
    rng = np.random.default_rng(20260906)
    for i, sym in enumerate(symbols):
        step = rng.normal(3e-4 + 2e-4 * i, 0.008 + 0.002 * i, len(idx))
        px = 50.0 * np.exp(np.cumsum(step))
        if after is not None:
            px = np.where(idx > pd.Timestamp(after), px * factor, px)
        df = pd.DataFrame({"close": px, "adjclose": px}, index=pd.Index(idx, name="date"))
        store.write(price_dir, {sym: df}, {})


def _membership(tmp_path: Path, snaps: dict[str, list[str]] | None = None) -> dict:
    """Two snapshots that DIFFER: S7 is in the 2018 list only, S0 in the 2017 list only. The shipped
    file's snapshots share 95% of their names, so a run over it cannot show that each window used
    its own list -- this one can."""
    lines = ["[snapshots]"]
    snaps = snaps or {CUTOFFS[0]: SYMBOLS[:-1], CUTOFFS[1]: SYMBOLS[1:]}
    for d, syms in snaps.items():
        lines.append(f'"{d}" = {list(syms)!r}'.replace("'", '"'))
    return pit.load_membership(_write(tmp_path, "\n".join(lines) + "\n"))


def _irx(end: str = "2019-12-31") -> pd.Series:
    idx = pd.bdate_range("2014-06-02", end)
    return pd.Series(0.02, index=idx, name="rf")


def _run(tmp_path: Path, price_dir: Path, **kw) -> list[dict]:
    cutoffs = kw.pop("cutoffs", CUTOFFS)
    mem = kw.pop("membership", None) or _membership(
        tmp_path, None if cutoffs == CUTOFFS else {d: SYMBOLS for d in cutoffs})
    return pit.run(mem, cutoffs, kw.pop("train_years", 2.0), 1.0, [1.0],
                   kw.pop("benchmark", "SPY"), _irx(kw.pop("rf_end", "2019-12-31")),
                   price_dir=price_dir, top_n=3, **kw)


@pytest.fixture(scope="module")
def prices(tmp_path_factory):
    d = tmp_path_factory.mktemp("prices")
    _store(d, SYMBOLS + ["SPY"])
    return d


def test_each_window_chooses_from_the_membership_of_its_own_cutoff(prices, tmp_path):
    """THE POINT OF THE MODULE, asserted. The 2017 window may not see S7 and the 2018 window may
    not see S0 -- and both symbols are in the store for the whole span, so nothing but the
    membership can keep them out. A run that loaded one list for every window passes every other
    test in this file."""
    periods = _run(tmp_path, prices)
    assert [p["cutoff"] for p in periods] == CUTOFFS
    first, second = (set(p["strategies"][0]["top_holdings"]) for p in periods)
    assert periods[0]["membership"]["in_the_candidate_pool"] == 7
    assert periods[1]["membership"]["in_the_candidate_pool"] == 7
    for p, forbidden in zip(periods, ["S7", "S0"]):
        for s in p["strategies"]:
            assert forbidden not in (s["top_holdings"] or {}), (
                f"{p['cutoff']}: {s['strategy']} holds {forbidden}, which that snapshot does not "
                "list -- the window used another cutoff's membership"
            )
    assert first or second  # the fixture solved something, so the loop above examined weights


def test_the_benchmark_is_priced_but_never_investable_even_when_a_snapshot_lists_it(
        tmp_path_factory, tmp_path):
    """SPY is in the S&P 500's own membership file, so this is not a hypothetical.

    `run` builds `investable` as every panel column except the benchmark, unconditionally, and the
    document says so under `benchmark_investable: false`. That field is a CLAIM about behaviour, and
    `test_a_held_out_benchmark_is_priced_and_not_investable` reads the claim -- so flipping it to
    `true` makes that test SKIP, and a run that let the optimiser hold the index would pass
    everything. This is the behavioural half: the assertion is on the weights, and the flag is not
    consulted.

    The store here is built so the mutant would be caught rather than merely permitted. SPY is given
    the highest drift AND the lowest variance of any column, so it is the asset the minimum-variance
    solve and the tangency solve both want most -- which is the real situation an index fund is in
    among its own constituents, and the reason holding it out is what makes the comparison a
    comparison.
    """
    d = tmp_path_factory.mktemp("benchmark-in-the-snapshot")
    idx = pd.bdate_range("2014-06-02", "2019-12-31")
    rng = np.random.default_rng(4)
    for i, sym in enumerate(SYMBOLS):
        px = 50.0 * np.exp(np.cumsum(rng.normal(2e-4, 0.010 + 0.002 * i, len(idx))))
        store.write(d, {sym: pd.DataFrame({"close": px, "adjclose": px},
                                          index=pd.Index(idx, name="date"))}, {})
    px = 50.0 * np.exp(np.cumsum(rng.normal(6e-4, 0.003, len(idx))))
    store.write(d, {"SPY": pd.DataFrame({"close": px, "adjclose": px},
                                        index=pd.Index(idx, name="date"))}, {})
    mem = _membership(tmp_path, {CUTOFFS[0]: SYMBOLS + ["SPY"]})
    periods = pit.run(mem, [CUTOFFS[0]], 2.0, 1.0, [1.0], "SPY", _irx(), price_dir=d, top_n=3)
    rows = {s["strategy"]: s for s in periods[0]["strategies"]}
    assert "benchmark:SPY" in rows, (
        "the benchmark has to be PRICED -- holding it out of the candidate set must not remove the "
        "row every printer compares against"
    )
    assert periods[0]["membership"]["investable"] == len(SYMBOLS)
    held = 0
    for label, s in rows.items():
        if label == "benchmark:SPY":
            continue
        assert "SPY" not in (s["top_holdings"] or {}), (
            f"{label} holds SPY, which this run prices as the benchmark -- an index fund among its "
            "own constituents has lower variance than almost any one of them, so the question "
            "'did mean-variance beat the index' becomes a question about a portfolio allowed to BE "
            "the index"
        )
        held += len(s["top_holdings"] or {})
    assert held > 0, "no strategy held anything, so the loop above examined no weight vector"


def test_a_hold_that_would_run_past_the_last_bar_raises_rather_than_being_reported_short(prices,
                                                                                        tmp_path):
    """A hold is requested by length, and the panel simply stops -- so without this assertion a
    one-year hold whose year is not there yet is measured over whatever bars exist and labelled
    `1.0y`. It is the same class of error as the nested-window trap: a number that reads as an
    answer to the question asked."""
    with pytest.raises(ValueError, match="before the .* hold closes"):
        _run(tmp_path, prices, cutoffs=["2019-06-03"], train_years=2.0)


def test_a_training_window_the_panel_starts_inside_raises_rather_than_being_reported_at_length(
        tmp_path_factory, tmp_path):
    """The other end, and it needs a store the shipped one cannot imitate.

    `load_panel` DROPS a symbol whose own history starts after the window opens, so the survivors
    all predate it -- and the panel still starts late, because the panel's index is the
    INTERSECTION of the survivors' trading days. Two symbols with a decade of history each and no
    common calendar until 2016 produce a panel that begins in 2016 with nothing dropped and nothing
    said. That is a 1-year training window reported as 2, from inputs where every individual filter
    passed, and it is why the assertion is on the panel's own ends rather than on the drops.

    Reachable only from a calendar mismatch: a stray European listing, a store half-refreshed, a
    vendor holiday one symbol observes. None of those exist in `pipeline/data/prices/`.
    """
    d = tmp_path_factory.mktemp("split-calendar")
    idx = pd.bdate_range("2014-01-01", "2019-12-31")
    split = pd.Timestamp("2016-01-01")
    rng = np.random.default_rng(11)
    for i, sym in enumerate(SYMBOLS + ["SPY"]):
        px = 40.0 * np.exp(np.cumsum(rng.normal(3e-4, 0.009, len(idx))))
        s = pd.Series(px, index=idx)
        if sym == "S0":       # Mon/Wed/Fri before 2016
            s = s[(s.index >= split) | (s.index.dayofweek.isin([0, 2, 4]))]
        elif sym == "S1":     # Tue/Thu before 2016 -- no day in common with S0
            s = s[(s.index >= split) | (s.index.dayofweek.isin([1, 3]))]
        store.write(d, {sym: pd.DataFrame({"close": s.to_numpy(), "adjclose": s.to_numpy()},
                                          index=pd.Index(s.index, name="date"))}, {})
    with pytest.raises(ValueError, match="after the .* training window should open"):
        _run(tmp_path, d, cutoffs=[CUTOFFS[0]], train_years=2.0, min_coverage=0.0)


def test_the_weights_of_every_window_ignore_the_prices_after_its_own_cutoff(tmp_path_factory,
                                                                           tmp_path):
    """`run` does its own date arithmetic for both ends of both windows, so it is a second place a
    bar can land on the wrong side of a cutoff -- `test_backtest_invariants.py` guarding `evaluate`
    does not cover it.

    Perturbed strictly after the LAST cutoff, so no training window contains the change: every
    weight vector in the run must be identical, while the last hold's realized return must not be
    (or the perturbation missed the panel and the test proves nothing).
    """
    a, b = tmp_path_factory.mktemp("a"), tmp_path_factory.mktemp("b")
    _store(a, SYMBOLS + ["SPY"])
    _store(b, SYMBOLS + ["SPY"], after=CUTOFFS[-1], factor=1.35)
    base, moved = _run(tmp_path, a), _run(tmp_path, b)
    for p, q in zip(base, moved):
        for s, t in zip(p["strategies"], q["strategies"]):
            assert s["strategy"] == t["strategy"]
            assert s["top_holdings"] == t["top_holdings"], (
                f"{p['cutoff']} {s['strategy']}: the weights moved when prices AFTER the last "
                "cutoff changed"
            )
            assert s["predicted"] == t["predicted"]
    assert (base[-1]["strategies"][0]["realized_monthly"]["ret"]
            != moved[-1]["strategies"][0]["realized_monthly"]["ret"]), (
        "the perturbation changed no realized return, so it never reached the panel"
    )


def test_a_constituent_whose_ticker_now_means_another_company_is_counted_among_the_unpriced(
        prices, tmp_path):
    """The accounting the stress bound reads. A name the membership STRIPS is missing from the
    candidate set for the same reason as a name the store has no prices for, and it is missing
    from `dropped` -- it was never asked for. Counting only `dropped_not_in_store` bounds a set
    that leaves out exactly the names the strips removed."""
    mem = _membership(tmp_path)
    mem["removed"]["2017-01-02"] = {"GONE": "recycled"}
    periods = pit.run(mem, ["2017-01-02"], 2.0, 1.0, [1.0], "SPY", _irx(),
                      price_dir=prices, top_n=3)
    m = periods[0]["membership"]
    assert "GONE" in m["unpriced_constituents"]
    assert m["n_unpriced_constituents"] == len(m["unpriced_constituents"])
    assert m["members"] == m["in_the_candidate_pool"] + 1, (
        "`members` is the snapshot as the index stood, so a stripped name still counts in it"
    )


def test_a_restored_name_is_no_longer_an_unpriced_constituent(prices, tmp_path):
    """`--extra-panel` is the treatment, so a name the second vendor supplies has to leave the set
    the stress bound covers -- otherwise the restored run is bounded as if the restoration had not
    happened, which understates it in the direction that flatters the method."""
    idx = pd.bdate_range("2014-06-02", "2019-12-31")
    extra = tmp_path / "extra.parquet"
    pd.DataFrame({"EXTRA": 30.0 * np.exp(np.cumsum(
        np.random.default_rng(7).normal(3e-4, 0.01, len(idx))))}, index=idx).to_parquet(extra)
    mem = _membership(tmp_path, {"2017-01-02": SYMBOLS + ["EXTRA"]})
    periods = pit.run(mem, ["2017-01-02"], 2.0, 1.0, [1.0], "SPY", _irx(),
                      price_dir=prices, top_n=3, extra_panel=extra)
    m = periods[0]["membership"]
    assert periods[0]["extra_panel"]["added"] == ["EXTRA"]
    assert m["dropped_not_in_store"] == 1, "EXTRA is not in the store, so it must be counted there"
    assert "EXTRA" not in m["unpriced_constituents"], (
        "a name the extra panel restored is still counted as unpriced"
    )
