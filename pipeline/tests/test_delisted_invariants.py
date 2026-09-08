"""Invariants of the second-vendor path: `delisted.py` and `backtest.join_extra`.

DELIBERATELY SYNTHETIC, and it passes the repo's test for when that is allowed: the property is
one no file on disk can express. The twelve cached series all have a bad-looking final bar and a
clean penultimate one, so nothing among them distinguishes "drops the last bar" from "keeps it";
no panel in `pipeline/data/` has a column that clashes with the store, starts late, or has a hole
in the middle; and the load-bearing claim about `join_extra` is that it CANNOT MOVE THE PANEL'S
WINDOW, which is a statement about what does not happen and needs a panel whose window is known.

The one thing here that is not synthetic is `test_the_cached_series_all_reach_the_date_the_shipped_run_needs`,
which reads the real cache -- because "the vendor served enough history" is a fact about the files
and not about the code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest as bt  # noqa: E402
import delisted as dl  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "data" / "delisted"
BARS = pd.bdate_range("2020-01-01", "2020-12-31")


def _series(n: int, start: float = 100.0, step: float = 0.1, vol: float = 1e6) -> pd.DataFrame:
    """A boring rising series with real volume, so any anomaly in a test is the one it put there."""
    idx = BARS[:n]
    lvl = start + step * np.arange(n)
    return pd.DataFrame({"close": lvl, "adjusted": lvl, "volume": np.full(n, vol)},
                        index=pd.Index(idx, name="date"))


def _cache(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> Path:
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    for sym, df in frames.items():
        df.to_csv(tmp_path / "raw" / f"{sym}.csv")
    return tmp_path


# ------------------------------------------------------------------- the terminal bar

def test_the_last_traded_bar_is_dropped_and_the_one_before_it_is_the_terminal_level():
    """The rule the module argues for at length, asserted rather than trusted.

    Constructed so the two candidate answers differ by a lot and in a direction a bug would
    produce: the final bar is a 9% collapse on 20x volume, which is what CTRA's real last bar
    looks like. Keeping it understates that name's outcome by 9% and nothing else changes.
    """
    df = _series(50)
    df.loc[df.index[-1], ["close", "adjusted", "volume"]] = [90.0, 90.0, 2e7]
    when, level, ev = dl.terminal(df)
    assert when == df.index[-2]
    assert level == pytest.approx(float(df["adjusted"].iloc[-2]))
    assert ev["dropped_date"] == str(df.index[-1].date())
    assert ev["dropped_bar_return"] == pytest.approx(90.0 / df["adjusted"].iloc[-2] - 1.0, abs=1e-6)
    assert ev["dropped_bar_return"] < -0.08, "the fixture's point is a large dropped-bar move"


def test_a_stub_penultimate_bar_is_refused_rather_than_used():
    """Dropping one bar is only safe while the bar behind it is a real close. Three of the twelve
    print volume 0 or 1 on their last bar, so a series with TWO of those is not hypothetical -- and
    the uniform rule would silently return a level nobody traded at. It raises instead, which is
    the signal to look at that symbol on its own."""
    df = _series(50)
    df.loc[df.index[-2:], "volume"] = 0.0
    with pytest.raises(RuntimeError, match="stub"):
        dl.terminal(df)


# ------------------------------------------------------------------------- the stitch

def test_the_continuation_never_overwrites_a_real_bar_and_is_continuous_at_the_splice(tmp_path):
    """Two claims in one fixture because they fail together.

    The real half must survive untouched -- a continuation that starts one bar early replaces a
    traded close with a T-bill accrual and nothing downstream can tell. And the splice must be
    continuous in LEVEL: the first continued bar's return has to be the continuation asset's own
    return that day, not a jump from the vendor's price scale to the calendar's. `stitch` divides
    by `calendar.asof(end)` for exactly that reason, and dropping the division leaves a series
    that is still monotone, still NaN-free, and wrong by a factor of a hundred at one bar.
    """
    df = _series(40)
    cache = _cache(tmp_path, {"AAA": df})
    calendar = pd.Series(np.linspace(4.0, 8.0, len(BARS)), index=BARS)
    panel, facts = dl.stitch(["AAA"], "cash", calendar, cache)

    end = pd.Timestamp(facts["AAA"]["terminal_date"])
    assert end == df.index[-2]
    real = panel["AAA"].loc[:end]
    pd.testing.assert_series_equal(real, df["adjusted"].loc[:end], check_names=False)

    after = panel["AAA"].loc[panel.index > end]
    assert len(after) == len(calendar.loc[calendar.index > end])
    first = after.index[0]
    assert after.iloc[0] / panel["AAA"].loc[end] == pytest.approx(
        calendar.loc[first] / calendar.loc[end]), "the splice introduced a level jump"
    assert panel["AAA"].pct_change().loc[panel.index > end].iloc[1:].values == pytest.approx(
        calendar.pct_change().loc[calendar.index > end].iloc[1:].values)


def test_the_two_continuations_differ_only_after_the_terminal_bar(tmp_path):
    """The whole argument for running the backtest twice is that the treatments differ ONLY in the
    continuation. If they differed in the real half as well, "the conclusion holds under both"
    would be a statement about two different histories."""
    cache = _cache(tmp_path, {"AAA": _series(40)})
    idx = BARS
    cash = pd.Series(np.linspace(1.0, 1.05, len(idx)), index=idx)
    spy = pd.Series(np.linspace(1.0, 1.60, len(idx)), index=idx)
    a, fa = dl.stitch(["AAA"], "cash", cash, cache)
    b, fb = dl.stitch(["AAA"], "spy", spy, cache)
    end = pd.Timestamp(fa["AAA"]["terminal_date"])
    assert fa["AAA"]["terminal_date"] == fb["AAA"]["terminal_date"]
    pd.testing.assert_series_equal(a["AAA"].loc[:end], b["AAA"].loc[:end])
    assert a["AAA"].iloc[-1] != pytest.approx(b["AAA"].iloc[-1]), (
        "the two treatments produced the same terminal value, so the fixture is not testing them"
    )


def test_a_vendor_holiday_the_calendar_does_not_have_is_dropped_rather_than_carried(tmp_path):
    """The primary vendor's calendar is authoritative, because it is the one the panel is built
    on. A bar the second vendor serves on a day the store has no trading on is not a return the
    portfolio could have had, and forward-filling around it -- or keeping it -- fabricates one."""
    df = _series(40)
    extra_day = pd.Timestamp("2020-01-01")  # a holiday `BARS` happens to include; see below
    calendar = pd.Series(1.0 + 0.001 * np.arange(len(BARS) - 1), index=BARS[1:])
    panel, _ = dl.stitch(["AAA"], "cash", calendar, _cache(tmp_path, {"AAA": df}))
    assert extra_day in df.index and extra_day not in calendar.index
    assert extra_day not in panel.index, "a date absent from the calendar survived the stitch"


def test_the_cached_series_all_reach_the_date_the_shipped_run_needs():
    """The one non-synthetic test here. `range=MAX` returns ONE YEAR with a 200 response, so the
    served span is a fact to check and not a parameter to set -- and the check has to run against
    the committed files, because a short cache is a property of the fetch and not of the code."""
    if not (CACHE / "raw").exists():
        pytest.skip("no delisted cache -- run `python -u pipeline/delisted.py --refresh`")
    needs = pd.Timestamp("2019-08-01")
    for sym in dl.SERVED_AS:
        df = dl.load(sym, CACHE)
        assert df.index.min() <= needs, (
            f"{sym} starts {df.index.min().date()}, after {needs.date()}, so it cannot cover the "
            "training window and would be dropped by the panel's own start filter"
        )
        assert len(df) > 500 and (df["adjusted"] > 0).all()


def test_a_range_the_vendor_does_not_recognise_reads_as_a_short_history_and_raises(monkeypatch):
    """The trap named in the docstring, exercised. A 200 with 252 rows is indistinguishable from a
    young company, so `fetch_series` compares the served span against what the caller asked for."""
    rows = [{"t": str(d.date()), "c": 1.0, "a": 1.0, "v": 1e6}
            for d in pd.bdate_range("2025-01-01", "2025-12-31")]
    monkeypatch.setattr(dl, "_get", lambda *a, **k: rows)
    with pytest.raises(RuntimeError, match="does not reach"):
        dl.fetch_series("AAA", needs="2019-08-01")
    assert len(dl.fetch_series("AAA", needs="2025-06-01")) == len(rows)


def _rows(start: str, end: str) -> list[dict]:
    return [{"t": str(d.date()), "c": 1.0, "a": 1.0, "v": 1e6}
            for d in pd.bdate_range(start, end)]


def test_a_call_that_checks_neither_the_span_nor_the_date_is_refused(monkeypatch):
    """`_check_served` takes exactly one of `needs` and `min_span_years`, and the reason it raises
    on NEITHER rather than defaulting is that a one-year fallback arrives with a 200 and plausible
    data. A caller who passes nothing has written a fetch with no check in it, and the failure
    would surface as a panel quietly starting in 2025."""
    monkeypatch.setattr(dl, "_get", lambda *a, **k: _rows("2015-01-02", "2025-12-31"))
    with pytest.raises(ValueError, match="exactly one"):
        dl.fetch_series("AAA")


def test_the_longest_range_wins_even_when_the_range_that_serves_it_comes_last(monkeypatch):
    """The measured behaviour `fetch_longest` exists for: `MAX` served AVB 8,159 bars three times
    and then 252 -- one year -- while `10Y` served 2,513 throughout. So the ranges are not tried
    until one clears the floor; all of them are tried and the longest span wins.

    Two assertions in one test because they are two halves of the same rule. EVERY range is
    requested (a first-that-clears loop would stop at `MAX`, which cleared nothing here), and the
    floor is checked ONCE against the winner -- `MAX` falling back to one year against a 2-year
    floor is the expected outcome for most symbols, not an error.
    """
    served = {"MAX": _rows("2025-01-01", "2025-12-31"),
              "10Y": _rows("2015-01-02", "2025-12-31")}
    asked: list[str] = []

    def get(symbol, rng="10Y", **kw):
        asked.append(rng)
        return served[rng]

    monkeypatch.setattr(dl, "_get", get)
    df, rng = dl.fetch_longest("AVB", 2.0, pace=0.0)
    assert asked == list(dl.RANGES)
    assert rng == "10Y" and df.index.min().year == 2015


def test_a_one_year_fallback_on_every_range_is_refused_rather_than_returned(monkeypatch):
    """The floor still fires when there is no good range to fall back on. Returning the longest of
    two one-year responses would be a panel that covers 2025 for a name whose training window
    opens in 2011, and nothing downstream would say so -- `load_panel`'s start filter would drop
    the column and the run would look like one where the vendor simply had no history."""
    monkeypatch.setattr(dl, "_get", lambda *a, **k: _rows("2025-01-01", "2025-12-31"))
    with pytest.raises(RuntimeError, match="a span of"):
        dl.fetch_longest("AVB", 2.0, pace=0.0)



# --------------------------------------------------------------------- joining the panel

def _panel(cols: list[str], n: int = 60) -> pd.DataFrame:
    idx = BARS[:n]
    return pd.DataFrame(
        {c: 100.0 + np.arange(n) * (0.05 + 0.01 * i) for i, c in enumerate(cols)}, index=idx)


def test_joining_extra_columns_cannot_move_the_panels_window(tmp_path):
    """THE LOAD-BEARING PROPERTY. The restored run is only comparable with the baseline bar for bar
    if the join adds columns to a fixed set of rows. An extra column with more history than the
    panel, or less, must change the column count and nothing else -- an outer join would lengthen
    the window and make every number in the restored run answer a slightly different question.
    """
    panel = _panel(["AAA", "BBB"])
    longer = pd.DataFrame({"CCC": 50.0 + np.arange(len(BARS)) * 0.2}, index=BARS)
    path = tmp_path / "extra.parquet"
    longer.to_parquet(path)
    joined, info = bt.join_extra(panel, path, start="2020-01-06", min_coverage=0.98)
    assert list(joined.index) == list(panel.index)
    assert list(joined.columns) == ["AAA", "BBB", "CCC"]
    pd.testing.assert_frame_equal(joined[["AAA", "BBB"]], panel)
    assert info["added"] == ["CCC"] and info["dropped"] == {}


def test_an_extra_column_that_shadows_a_stored_one_is_refused(tmp_path):
    """A silent vendor swap for one asset. The panel would still be the right shape and the right
    length, and one column would come from a source with no event log behind it."""
    panel = _panel(["AAA", "BBB"])
    path = tmp_path / "clash.parquet"
    pd.DataFrame({"BBB": 1.0 + np.arange(len(BARS)) * 0.01}, index=BARS).to_parquet(path)
    with pytest.raises(SystemExit, match="already in the panel"):
        bt.join_extra(panel, path, start="2020-01-06", min_coverage=0.98)


def test_an_extra_column_is_held_to_the_panels_own_start_and_coverage_filters(tmp_path):
    """`load_panel` drops a late arrival and a thin series with a reason. An extra column admitted
    on looser terms would be an asset in the candidate set because of which FILE it came from,
    which is the one thing this whole comparison is built to avoid."""
    panel = _panel(["AAA", "BBB"])
    late = pd.Series(100.0 + np.arange(20) * 0.2, index=panel.index[-20:])
    thin = pd.Series(100.0 + np.arange(len(panel)) * 0.2, index=panel.index)
    thin.iloc[5:25] = np.nan
    good = pd.Series(100.0 + np.arange(len(BARS)) * 0.2, index=BARS)
    path = tmp_path / "mixed.parquet"
    pd.DataFrame({"LATE": late, "THIN": thin, "GOOD": good}).to_parquet(path)
    joined, info = bt.join_extra(panel, path, start="2020-01-06", min_coverage=0.98)
    assert info["added"] == ["GOOD"]
    assert set(info["dropped"]) == {"LATE", "THIN"}
    assert "after 2020-01-06" in info["dropped"]["LATE"]
    assert "coverage" in info["dropped"]["THIN"]
    assert list(joined.index) == list(panel.index)


def test_a_hole_inside_a_tolerated_column_refuses_the_join_rather_than_deleting_a_bar(tmp_path):
    """`min_coverage` tolerates a small gap by design, and `dropna(how="any")` would then delete
    that bar from every one of the 478 stored assets to accommodate one restored column. Two bars
    missing from one extra column is not a reason to shorten the panel."""
    panel = _panel(["AAA", "BBB"])
    s = pd.Series(100.0 + np.arange(len(panel)) * 0.2, index=panel.index)
    s.iloc[30] = np.nan
    path = tmp_path / "hole.parquet"
    pd.DataFrame({"CCC": s}).to_parquet(path)
    with pytest.raises(SystemExit, match="would delete"):
        bt.join_extra(panel, path, start="2020-01-06", min_coverage=0.9)


def test_restoring_nothing_raises_unless_the_caller_joins_once_per_window(tmp_path):
    """`allow_empty` both ways, because each way is wrong for the other caller.

    `backtest.py --extra-panel` joins ONCE: restoring nothing means the flag did nothing, and the
    run must not go on to be reported as a restored one -- there would be no difference from the
    baseline anywhere in the file except the presence of the flag in its own provenance. `pit.py`
    joins the same file into ten windows, and the earliest legitimately has no restorable name:
    the second vendor serves ten years back from each ticker's LAST bar, which does not reach a
    training window opening in 2011. Refusing there would make the honest outcome unrunnable.

    The permissive branch has to return the panel UNCHANGED, not merely a panel of the same shape:
    `pd.concat` with an empty frame is where a column order or an index dtype would quietly move,
    and every window after the first one joins something.
    """
    panel = _panel(["AAA", "BBB"])
    late = pd.Series(100.0 + np.arange(20) * 0.2, index=panel.index[-20:])
    path = tmp_path / "nothing-survives.parquet"
    pd.DataFrame({"LATE": late}).to_parquet(path)
    with pytest.raises(SystemExit, match="no column survived"):
        bt.join_extra(panel, path, start="2020-01-06", min_coverage=0.98)
    joined, info = bt.join_extra(panel, path, start="2020-01-06", min_coverage=0.98,
                                allow_empty=True)
    pd.testing.assert_frame_equal(joined, panel)
    assert info["added"] == []
    assert set(info["dropped"]) == {"LATE"}, (
        "a window that restored nothing still has to say which names it considered, or "
        "'restored nothing' is indistinguishable from 'was never asked'"
    )


# ------------------------------------------------------------------- the reverse bound

def _result(cutoff: str, end: str, bm_ret: float, bm_vol: float, ew_ret: float) -> dict:
    return {
        "universe": {"extra_panel": None},
        "rolling": [{
            "benchmark": "SPY",
            "periods": [{
                "period": f"{cutoff} to {end}", "cutoff": cutoff,
                "train": {"start": "2019-08-30", "bars": 1009, "years": 4.0},
                "test": {"end": end, "bars": 251, "years": 1.0},
                "mu_rank_correlation": 0.1, "momentum_rank_correlation": 0.1,
                "vol_rank_correlation": 0.7, "n_investable": 100,
                "strategies": [
                    {"strategy": "matched_risk@cap10", "cap": 0.1, "n_holdings": 20,
                     "top_holdings": {"AAA": 0.1},
                     "predicted": {"ret": 0.2, "vol": 0.16, "sharpe": 1.0},
                     "realized_monthly": {"growth": 1.3, "ret": 0.30, "vol": 0.16,
                                          "sharpe": 1.8, "max_drawdown": 0.1}},
                    {"strategy": "equal_weight", "cap": 1.0, "n_holdings": 100,
                     "top_holdings": {"AAA": 0.01},
                     "predicted": {"ret": 0.1, "vol": 0.14, "sharpe": 0.7},
                     "realized_monthly": {"growth": 1.0 + ew_ret, "ret": ew_ret, "vol": 0.14,
                                          "sharpe": 1.1, "max_drawdown": 0.1}},
                    {"strategy": "benchmark:SPY", "cap": 1.0, "n_holdings": 1,
                     "top_holdings": {"SPY": 1.0},
                     "predicted": {"ret": 0.1, "vol": bm_vol, "sharpe": 0.7},
                     "realized_monthly": {"growth": 1.0 + bm_ret, "ret": bm_ret, "vol": bm_vol,
                                          "sharpe": 1.0, "max_drawdown": 0.15}},
                ],
            }],
        }],
    }


def _block(result: dict) -> dict:
    """`stress_bound` takes the block, not the document. Kept as a helper over the same fixture so
    that the fixture still has the document's shape -- the `universe`/`rolling` nesting is what
    `backtest.py` writes, and a fixture reduced to just the block would stop resembling the file
    the two call sites actually read."""
    return result["rolling"][0]


def test_the_bounds_span_comes_from_the_cutoff_and_not_a_key_the_period_lacks():
    """A test block carries `end`, `bars` and `years` -- NOT `start`, because the hold begins at
    the cutoff. `pd.Timestamp(None)` is NaT and every number derived from it is NaN, so reading
    the wrong key produces a complete file full of nulls rather than an error. This is the same
    mistake `test_results_invariants._pid` exists because of, one file over."""
    res = _result("2023-09-01", "2024-08-30", 0.20, 0.15, 0.16)
    idx = pd.bdate_range("2023-08-01", "2024-09-30")
    panel = pd.DataFrame({"AAA": 100.0 + np.arange(len(idx)) * 0.1}, index=idx)
    spy = pd.Series(400.0 + np.arange(len(idx)) * 0.4, index=idx)
    out = dl.stress_bound(_block(res), panel, spy, unpriceable=["X", "Y"])
    assert out["span"] == {"start": "2023-09-01", "end": "2024-08-30",
                           "years": pytest.approx(0.997, abs=5e-4)}
    assert all(np.isfinite(v) for v in out["equal_weight"]["annualised"].values())


def test_the_equal_weight_bound_brackets_the_measured_value_from_both_sides():
    """Wiping the missing names out has to be worse than what was measured and matching the best
    observed outcome has to be better, or the arithmetic has the extra names entering with the
    wrong sign. Checked as an ordering rather than against a number, because the numbers depend on
    the fixture and the ordering is the claim."""
    res = _result("2023-09-01", "2024-08-30", 0.20, 0.15, 0.16)
    idx = pd.bdate_range("2023-08-01", "2024-09-30")
    panel = pd.DataFrame({"AAA": 100.0 * 1.002 ** np.arange(len(idx))}, index=idx)
    spy = pd.Series(400.0 * 1.0003 ** np.arange(len(idx)), index=idx)
    out = dl.stress_bound(_block(res), panel, spy, unpriceable=["X", "Y", "Z"])
    # The fixture's premise, asserted: "best observed" is only an upper bound if the panel
    # actually contains an outcome better than the basket's own. The first version of this test
    # had a column growing more slowly than the equal-weight row and the ordering failed on
    # correct code.
    assert out["best_observed_growth"]["multiple"] > 1.0 + 0.16
    a = out["equal_weight"]["annualised"]
    assert a["all_wiped_out"] < a["as_measured"] < a["all_at_the_best_observed"]
    assert a["as_measured"] == pytest.approx(0.16, abs=2e-3)


def test_the_count_of_restored_names_held_is_null_on_a_run_that_had_none():
    """Counted against a baseline the twelve are not in, "held none of them" is a tautology that
    reads like evidence. The field is None there and the file says which kind of run it came
    from -- and it is a real count on a restored run."""
    idx = pd.bdate_range("2023-08-01", "2024-09-30")
    panel = pd.DataFrame({"AAA": 100.0 + np.arange(len(idx)) * 0.1}, index=idx)
    spy = pd.Series(400.0 + np.arange(len(idx)) * 0.4, index=idx)

    base = dl.stress_bound(_block(_result("2023-09-01", "2024-08-30", 0.2, 0.15, 0.16)),
                           panel, spy)
    assert base["optimised"]["matched_risk@cap10"]["times_a_restored_name_was_held"] is None
    assert "baseline run" in base["restored_names_in_this_result"]

    res = _result("2023-09-01", "2024-08-30", 0.2, 0.15, 0.16)
    res["universe"]["extra_panel"] = {"file": "panel_cash.parquet", "added": ["AAA"]}
    got = dl.stress_bound(_block(res), panel, spy,
                          extra_panel=res["universe"]["extra_panel"])
    assert got["optimised"]["matched_risk@cap10"]["times_a_restored_name_was_held"] == 1
    assert got["restored_names_in_this_result"] == "panel_cash.parquet"


def test_the_provenance_files_state_the_size_of_the_dropped_bar_for_every_symbol():
    """The dropped-bar rule is an assumption, and the file is where its size is stated rather than
    asserted to be small. A provenance file that omits it -- or reports it for a subset -- turns a
    stated assumption back into an invisible one."""
    for treatment in dl.CONTINUATIONS:
        path = CACHE / f"provenance_{treatment}.json"
        if not path.exists():
            pytest.skip(f"{path.name} not committed -- run `python -u pipeline/delisted.py`")
        prov = json.loads(path.read_text())
        assert prov["continuation"] == treatment
        assert set(prov["symbols"]) == set(dl.SERVED_AS)
        for sym, f in prov["symbols"].items():
            assert {"dropped_date", "dropped_level", "dropped_volume",
                    "dropped_bar_return"} <= set(f), sym
            assert f["real_bars"] > 500 and f["continued_bars"] >= 0
        assert prov["unpriceable"] == dl.UNPRICEABLE
