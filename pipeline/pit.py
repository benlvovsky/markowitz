"""The walk-forward test with a DIFFERENT INDEX MEMBERSHIP AT EVERY REBALANCE DATE.

`backtest.py --universe sp500_2023` answers the question honestly for three years, and cannot
answer it for more: one membership list, dated 2023-08-30, is only free of hindsight FORWARD of
that date, and forward of it there are three years. This module removes that ceiling by giving
each rebalance date its own list -- the 2016 weights are chosen from the companies that were in
the index in 2016 -- which buys ten non-overlapping one-year holds instead of three.

WHAT IS REUSED, AND WHY IT MATTERS THAT IT IS: `backtest.evaluate` is called unchanged, once per
window. It is the single place a lookahead could enter, it is the thing `test_backtest_invariants`
perturbs prices and rates after the cutoff to interrogate, and a second copy of it here would be a
second place for a bar to land on the wrong side of a cutoff. This module owns the membership and
the per-window panel; it owns no estimation, no solve and no measurement.

FOUR DECISIONS THAT ARE NOT OBVIOUS:

1. **One panel per window, not one panel for the run.** `fetch.load_panel` intersects trading days
   and drops thin series, so a single panel over 2011-2026 would keep only the companies that
   traded on every day of fifteen years -- which is the survivorship filter this module exists to
   remove, reintroduced one layer down. Each window gets its own panel over its own
   [train_from, test_to], and the count that survived is reported per window.

2. **A name acquired DURING a hold still biases that window, and the bias is reported rather than
   corrected here.** Coverage inside the window is what drops it, so the surviving candidate set
   for each window is "members that traded through the whole window". `--extra-panel` is the
   treatment (`delisted.py` stitches a continuation onto the dead ones); this module states
   `n_dropped_midwindow` so that the size of the untreated bias is on the record either way.

3. **The recycled tickers are removed before anything else.** A ticker whose current occupant is a
   different company fetches a clean full-length series that passes every filter -- see
   `[recycled]` in the membership file. Nine of them, one of which is a leveraged Netflix ETF.

4. **The rebalance dates come off the file, not off the panel's last bar.** `run_rolling` counts
   backwards from the end of the data, which is right when the universe is fixed and wrong here:
   the membership snapshots are dated, so the windows have to line up with them or a 2016 weight
   vector is chosen from the 2017 list.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest as bt  # noqa: E402
import fetch  # noqa: E402
import frontier as fr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = ROOT / "pipeline" / "universes" / "sp500_pit.toml"
PRICE_DIR = ROOT / "pipeline" / "data" / "prices"
# Deliberately NOT `data/delisted/`. That cache is the 2023 exclusion set, its raw CSVs are
# committed because they are unrefetchable, and its provenance file names twelve symbols. Mixing
# a second membership's names into it would make both provenance files wrong about their own set.
PIT_CACHE = ROOT / "pipeline" / "data" / "delisted_pit"

# The floor `fetch_longest` checks the winning range against. Two years, not ten: what this guard
# has to catch is a silent fallback to ONE year, and a floor set to the length the caller would
# like rejects the names that are legitimately short -- CDAY/DAY is served 7.8 years because
# Ceridian floated in 2018, and TWTR misses a 9.0 floor by 12 days. Whether a series is long
# enough for a given WINDOW is not this function's question; `load_panel` and `join_extra` answer
# that one per window, with the reason attached.
MIN_SPAN_YEARS = 2.0

# How far a window's first or last bar may sit from the date it was asked for. A rebalance date is
# a calendar date and the panel has bars, so some slack is required; more than a long weekend plus
# a holiday means the window is not the window that was requested.
SLACK_DAYS = 6


def load_membership(path: Path = MEMBERSHIP) -> dict:
    """The membership file, with the recycled and reissued tickers already taken out.

    Removing them here rather than at the call site is deliberate: every consumer of this file
    wants them gone, and a consumer that forgot would get a plausible answer about a portfolio
    holding a leveraged Netflix ETF.

    The two removals are NOT the same removal, which is why the file has two tables. `[recycled]`
    is unconditional -- the ticker's series belongs to another company and no snapshot may hold it.
    `[reissued]` carries a date, and only snapshots BEFORE it lose the symbol, because the series
    after that date is the history of a constituent this file is right to hold. Applying the
    recycled rule to a reissued ticker deletes Dow Inc and Fox Corporation from seven snapshots
    each; applying the reissued rule to a recycled one keeps the wrong company forever.
    """
    doc = tomllib.load(open(path, "rb"))
    recycled = set(doc.get("recycled", {}))
    reissued = doc.get("reissued", {})
    for sym, spec in reissued.items():
        if not isinstance(spec, dict) or "from" not in spec or "why" not in spec:
            raise ValueError(f"{path.name}: [reissued].{sym} needs both `from` and `why`")
        if sym in recycled:
            raise ValueError(f"{path.name}: {sym} is in both [recycled] and [reissued]")
    cuts = {sym: pd.Timestamp(spec["from"]) for sym, spec in reissued.items()}
    snaps, removed = {}, {}
    for d, syms in doc["snapshots"].items():
        keep, gone = [], {}
        for s in syms:
            if s in recycled:
                gone[s] = "recycled"
            elif s in cuts and pd.Timestamp(d) < cuts[s]:
                gone[s] = f"reissued to another company on {cuts[s].date()}"
            else:
                keep.append(s)
        snaps[d], removed[d] = keep, gone
    # `removed` is returned rather than discarded because a stripped name is a CONSTITUENT NOBODY
    # PRICES, and it never reaches `load_panel` -- so it appears in no `dropped` reason and would
    # be missing from the stress bound's count of missing names while being exactly what that
    # bound is about. On the 2016 snapshot it is 12 names: nine of the ten recycled tickers (INFO's
    # company was not a member yet) and three of the four reissued (Q's was not either).
    return {"snapshots": snaps, "removed": removed,
            "recycled": sorted(recycled), "reissued": reissued,
            "vendor_two": doc.get("vendor_two", []), "unpriceable": doc.get("unpriceable", []),
            "truncated": doc.get("truncated", {}),
            "retrieved": doc.get("retrieved"), "name": doc.get("name")}


def held_between(membership: dict) -> dict[str, tuple[str, str]]:
    """symbol -> (first snapshot holding it, last snapshot holding it)."""
    out: dict[str, tuple[str, str]] = {}
    for d in sorted(membership["snapshots"]):
        for s in membership["snapshots"][d]:
            first = out.get(s, (d, d))[0]
            out[s] = (first, d)
    return out


def overlaps_its_membership(sym: str, first_bar: pd.Timestamp, last_bar: pd.Timestamp,
                            held: dict[str, tuple[str, str]]) -> str | None:
    """None if a fetched series plausibly IS that constituent's history; else why it is not.

    THE GENERAL FORM OF THE PARA CHECK, and the reason it is worth having as a rule rather than a
    list. A ticker whose company left the index and was later reissued to something else fetches
    a clean, full-length, filter-passing series belonging to a different company -- the manual
    screen for that was the vendor's `longName` against the constituent's, which needs a name to
    compare and a human to compare it. This needs neither: the index held the ticker between two
    known dates, so a series that does not reach into that interval cannot be the history of what
    the index held, whatever it is called.

    It caught three on the point-in-time set, all invisible to a span check and all of which the
    optimiser would otherwise have held as 2016 constituents: MON (Monsanto, gone by 2018, served
    2021-2022), CA (acquired by Broadcom in 2018, served 2023-2026) and DNB (taken private in
    2019, served from 2020 -- the relisted company, not the constituent). TWTR passes: served
    2013-2022 against membership 2018-2022.

    It is a NECESSARY and not a sufficient condition. A ticker reissued quickly enough to overlap
    its predecessor's membership would pass, and nothing automatic here would catch it.
    """
    if sym not in held:
        return "no snapshot holds this symbol"
    lo, hi = pd.Timestamp(held[sym][0]), pd.Timestamp(held[sym][1])
    if first_bar > hi:
        return (f"served {first_bar.date()} -> {last_bar.date()}, entirely after the index last "
                f"held it ({hi.date()}) -- a reissued ticker, not this constituent's history")
    if last_bar < lo:
        return (f"served {first_bar.date()} -> {last_bar.date()}, entirely before the index first "
                f"held it ({lo.date()})")
    return None


def window_dates(cutoff: str, train_years: float, hold_years: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    c = pd.Timestamp(cutoff)
    return c - pd.DateOffset(years=train_years), c + pd.DateOffset(years=hold_years)


def screen(membership: dict, cache: Path = PIT_CACHE) -> tuple[list[str], dict[str, str]]:
    """The cached second-vendor symbols this membership will accept, and why it rejects the rest.

    Its own function rather than a loop inside `build_extra` because the stress bound reads the
    same cache to find the best outcome any restored name achieved, and a rejected series in that
    calculation is a different company's growth multiple used as the upper bound on a constituent
    nobody prices -- which would widen the bound with a number that means nothing.
    """
    import delisted as dl

    ask = list(membership["vendor_two"]) + sorted(membership["truncated"])
    held = held_between(membership)
    kept, rejected = [], {}
    for sym in ask:
        if not (cache / "raw" / f"{sym}.csv").exists():
            rejected[sym] = f"{dl.SOURCE} serves nothing usable for it"
            continue
        df = dl.load(sym, cache)
        why = overlaps_its_membership(sym, df.index.min(), df.index.max(), held)
        if why is None:
            kept.append(sym)
        else:
            rejected[sym] = why
    return kept, rejected


def build_extra(membership: dict, cache: Path = PIT_CACHE, price_dir: Path = PRICE_DIR,
                refresh: bool = False, min_span_years: float = MIN_SPAN_YEARS,
                continuations: list[str] | None = None) -> dict[str, str]:
    """Fetch, SCREEN and stitch the second vendor's series for this membership. Returns the
    rejections: symbol -> why it is not in the panel.

    Two asks, and they are different failure modes with the same treatment: `vendor_two` is the
    names the primary vendor serves nothing for, `[truncated]` is the names it serves under the
    right company with a few weeks of history. Both reach the optimiser only through
    `--extra-panel`, and neither enters the store -- see `delisted.py`'s docstring for why that
    line is absolute.

    THE SCREEN IS WHY THIS LIVES HERE AND NOT IN `delisted.py`. `overlaps_its_membership` needs
    the snapshots, and `delisted.py` is the second vendor and knows nothing about index
    membership. A rejected series is dropped BEFORE the stitch rather than after: a stitch is a
    continuation of a real history onto a synthetic tail, and continuing the wrong company's
    history produces a column that passes every downstream check.
    """
    import delisted as dl

    ask = {s: s for s in list(membership["vendor_two"]) + sorted(membership["truncated"])}
    if refresh:
        dl.refresh(ask, cache=cache, min_span_years=min_span_years, skip_failures=True,
                   longest=True)

    kept, rejected = screen(membership, cache)
    for sym, why in sorted(rejected.items()):
        print(f"  rejected {sym:6s} {why}", flush=True)
    dl.write_panels(kept, continuations or list(dl.CONTINUATIONS), cache, price_dir,
                    unpriceable=list(membership["unpriceable"]),
                    caveats=[
                        "the last traded bar of every series is dropped, uniformly; the cost is "
                        "stated per symbol in this file rather than asserted to be small",
                        "K carries a spinoff the two vendors credit differently and there is no "
                        "second series to resolve it against",
                        f"{len(membership['unpriceable'])} constituents are served by neither "
                        "vendor at any date and are not in this panel at all -- they are bounded "
                        "in a *_stress.json artifact instead",
                    ],
                    extra={"membership_file": MEMBERSHIP.name,
                           "screen": "a series whose bars do not overlap the dates the index "
                                     "actually held the ticker is rejected: see "
                                     "pit.overlaps_its_membership",
                           "rejected": rejected,
                           "truncated_by_primary_vendor": membership["truncated"]})
    return rejected


def run(
    membership: dict,
    cutoffs: list[str],
    train_years: float,
    hold_years: float,
    caps: list[float],
    benchmark: str | None,
    irx: pd.Series,
    price_dir: Path = PRICE_DIR,
    top_n: int = 10,
    min_coverage: float = 0.98,
    extra_panel: Path | None = None,
    model: bt.MuModel = bt.MuModel(),
    costs: bt.Costs = bt.Costs(),
) -> list[dict]:
    """One window per cutoff, each with its own membership, panel and candidate set."""
    out = []
    snaps = membership["snapshots"]
    for cutoff in cutoffs:
        members = list(snaps[cutoff])
        train_from, test_to = window_dates(cutoff, train_years, hold_years)
        wanted = members + ([benchmark] if benchmark and benchmark not in members else [])
        panel, dropped = fetch.load_panel(wanted, price_dir, start=str(train_from.date()),
                                          end=str(test_to.date()), min_coverage=min_coverage)
        extra_info = None
        if extra_panel is not None:
            panel, extra_info = bt.join_extra(panel, extra_panel, start=str(train_from.date()),
                                              min_coverage=min_coverage, allow_empty=True)

        cut_bar = bt._bar_at_or_before(panel, pd.Timestamp(cutoff), f"the {cutoff} rebalance")
        # The panel was built with `start`/`end` already applied, so its own ends ARE the window's
        # ends -- but both are asserted rather than assumed. `load_panel` drops a symbol whose
        # history starts late; if every symbol starts late the window silently shortens instead,
        # and a 4-year training window reported as 5 is the kind of error that reads as correct.
        from_bar, to_bar = panel.index.min(), panel.index.max()
        if from_bar > train_from + pd.Timedelta(days=SLACK_DAYS):
            raise ValueError(f"{cutoff}: the panel starts {from_bar.date()}, "
                             f"{(from_bar - train_from).days} days after the {train_years}y "
                             f"training window should open ({train_from.date()})")
        if to_bar < test_to - pd.Timedelta(days=SLACK_DAYS):
            raise ValueError(f"{cutoff}: the panel ends {to_bar.date()}, before the {hold_years}y "
                             f"hold closes ({test_to.date()}) -- a hold reported at its requested "
                             "length must not be shorter than it says")
        train, test = panel.loc[from_bar:cut_bar], panel.loc[cut_bar:to_bar]

        investable = [c for c in panel.columns if c != benchmark]
        late = sum(1 for r in dropped.values() if "history starts" in r)
        thin = sum(1 for r in dropped.values() if "coverage" in r)
        # The two ways a constituent can be unpriced at this cutoff, unioned and NAMED. Counting
        # them was not enough: the stress bound needs the set, and the two halves come from
        # different places -- `dropped` for the ones asked for and not served, `removed` for the
        # ones never asked for because the ticker means someone else. `added` comes back off the
        # second vendor, so a restored name is no longer unpriced and must leave the set.
        gone = membership.get("removed", {}).get(cutoff, {})
        absent = {s for s, r in dropped.items() if "not in the price store" in r}
        restored_here = set(extra_info["added"]) if extra_info else set()
        unpriced = sorted((absent - restored_here) | set(gone))
        row = bt.evaluate(train, test, irx, caps, benchmark, None, top_n, model, investable, costs)
        out.append({
            "period": f"{cut_bar.date()} to {to_bar.date()}",
            "cutoff": str(cut_bar.date()),
            "membership": {
                "as_of": cutoff,
                "members": len(members) + len(gone),
                "in_the_candidate_pool": len(members),
                "investable": len(investable),
                "removed_as_a_different_company": gone,
                "dropped_not_in_store": len(absent),
                "dropped_history_starts_late": late,
                "dropped_midwindow": thin,
                "n_unpriced_constituents": len(unpriced),
                "unpriced_constituents": unpriced,
            },
            "extra_panel": extra_info,
            **row,
        })
        m = out[-1]["membership"]
        restored = "" if extra_info is None else f"  restored {len(extra_info['added']):>2}"
        print(f"  {cutoff}  members {m['members']:>4}  investable {m['investable']:>4}  "
              f"(no data {m['dropped_not_in_store']:>3}, another company {len(gone):>3}, "
              f"starts late {late:>3}, died mid-window {thin:>3})  "
              f"unpriced {len(unpriced):>3}  panel {len(panel)} bars{restored}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-years", type=float, default=5.0)
    ap.add_argument("--hold-years", type=float, default=1.0)
    ap.add_argument("--from-year", type=int, default=2016)
    ap.add_argument("--to-year", type=int, default=2025, help="the LAST rebalance year (its hold "
                    "ends a year later, so this is the last year with a full hold behind it)")
    ap.add_argument("--caps", default="1.0,0.2,0.1")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    ap.add_argument("--membership", type=Path, default=MEMBERSHIP)
    ap.add_argument("--extra-panel", type=Path, default=None)
    ap.add_argument("--build-extra", action="store_true",
                    help="stitch the second vendor's panels for this membership from the cache "
                         "and exit; add --refresh-extra to refetch them first")
    ap.add_argument("--refresh-extra", action="store_true",
                    help="with --build-extra: refetch from the second vendor before stitching")
    ap.add_argument("--extra-cache", type=Path, default=PIT_CACHE)
    ap.add_argument("--stress", type=Path, default=None,
                    help="a pit result JSON: bound what the constituents neither vendor prices "
                         "could have done to it, and write it beside the result")
    ap.add_argument("--min-coverage", type=float, default=0.98)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--refresh-rf", action="store_true")
    ap.add_argument("--trade-bps", type=float, default=0.0,
                    help="all-in one-way cost per dollar traded, in basis points. Non-zero turns "
                         "on the net-of-cost columns AND charges the benchmark its published fund "
                         "fee, because charging one side and not the other is the bias this flag "
                         "exists to remove. 0 reproduces a run made before it existed.")
    ap.add_argument("--out", type=Path, default=ROOT / "pipeline" / "results" / "backtest_pit.json")
    args = ap.parse_args(argv)

    mem = load_membership(args.membership)
    if args.stress is not None:
        import delisted as dl

        result = json.loads(args.stress.read_text())
        # `periods` is top level in this document and under `rolling[0]` in `backtest.py`'s, which
        # is why `stress_bound` takes a block.
        #
        # THE SET BOUNDED COMES OFF THE RUN, NOT OFF THE MEMBERSHIP FILE, and the two differ by
        # more than a rounding: the file's `unpriceable` is 67 names, and the 2016 window's own
        # count of constituents nothing priced is 84 -- the difference is the 23 the second vendor
        # serves (absent from a baseline run, present in a restored one, so the right number
        # depends on which run is being bounded) and the 14 whose ticker now means another company
        # (in no `dropped` reason at all, because they never reach `load_panel`). Reading the
        # literal would bound the wrong run with a number that looks deliberate.
        #
        # Unioned across periods, because the bound is one span-wide calculation: a name that was
        # a constituent at some cutoff and priced by nobody there belongs in it.
        block = {"benchmark": result["benchmark"], "periods": result["periods"]}
        unpriced = sorted({s for p in result["periods"]
                           for s in p["membership"]["unpriced_constituents"]})
        irx = pd.read_parquet(ROOT / "pipeline" / "data" / "irx.parquet")["rf"]
        import store  # noqa: E402

        spy = store.read_total_return(args.price_dir, ["SPY"])["SPY"].dropna()
        kept, _ = screen(mem, args.extra_cache)
        panel, _ = dl.stitch(kept, "cash",
                             dl._cash_path(irx.reindex(spy.index).ffill()), args.extra_cache)
        bound = dl.stress_bound(block, panel, spy, unpriceable=unpriced,
                                extra_panel=(result["periods"][-1].get("extra_panel")))
        bound["result"] = args.stress.name
        bound["membership_file"] = args.membership.name
        bound["unpriced_constituents"] = {
            "n": len(unpriced), "symbols": unpriced,
            "source": "the union of every period's own unpriced_constituents in " + args.stress.name,
            "note": "not sp500_pit.toml's `unpriceable` list, which is 67 names and describes the "
                    "membership file rather than this run",
        }
        out = args.stress.with_name(args.stress.stem + "_stress.json")
        out.write_text(json.dumps(bound, indent=2) + "\n")
        print(json.dumps(bound, indent=2), flush=True)
        print(f"wrote {out}", flush=True)
        return 0

    if args.build_extra:
        rejected = build_extra(mem, cache=args.extra_cache, price_dir=args.price_dir,
                               refresh=args.refresh_extra)
        asked = len(mem["vendor_two"]) + len(mem["truncated"])
        print(f"\n{asked - len(rejected)} of {asked} names stitched; {len(rejected)} rejected",
              flush=True)
        return 0

    cutoffs = [d for d in mem["snapshots"]
               if args.from_year <= int(d[:4]) <= args.to_year]
    caps = [float(c) for c in args.caps.split(",")]
    irx = bt.load_irx(refresh=args.refresh_rf)
    costs = bt.Costs(trade_bps=args.trade_bps,
                     expense=bt.EXPENSE_RATIOS if args.trade_bps else {})

    print(f"{mem['name']} -- {len(cutoffs)} rebalance dates, {args.train_years}y training window, "
          f"{args.hold_years}y holds, caps {caps}", flush=True)
    periods = run(mem, cutoffs, args.train_years, args.hold_years, caps, args.benchmark, irx,
                  price_dir=args.price_dir, top_n=args.top, min_coverage=args.min_coverage,
                  extra_panel=args.extra_panel, costs=costs)

    doc = {
        "kind": "pit_backtest",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "membership": {
            "file": args.membership.name,
            "retrieved": mem["retrieved"],
            "recycled_removed": mem["recycled"],
            "reissued_removed_before_their_dates": {s: spec["from"]
                                                    for s, spec in mem["reissued"].items()},
            "unpriceable": mem["unpriceable"],
            "priced_by_vendor_two": mem["vendor_two"],
            "note": "these four lists describe the FILE. What was missing from each window's "
                    "candidate set is per period, under membership.unpriced_constituents, and it "
                    "is the set a stress bound over this run has to cover.",
        },
        "method": {
            "train_years": args.train_years,
            "hold_years": args.hold_years,
            "caps": caps,
            "benchmark": args.benchmark,
            "rf": "^IRX as of each cutoff",
            "estimator": "geometric mean + Ledoit-Wolf shrunk covariance, 252-day annualised",
            "solver": fr.SOLVER,
            "costs": None if costs.zero() else costs.describe(),
            "caveats": [
                "the holds are non-overlapping, but consecutive TRAINING windows overlap by "
                f"{args.train_years - args.hold_years}y, so a persistent regime is estimated the "
                "same way several times running",
                "a member acquired during a hold fails the window's coverage filter and is "
                "dropped from that window's candidate set unless --extra-panel restores it: see "
                "dropped_midwindow per period",
                "the membership is reconstructed from a change log, not from the index provider; "
                "it reproduces 498 of the 503 hand-verified 2023 names and the 5 misses are "
                "ticker spelling",
                "no transaction costs, no taxes, no slippage; annual rebalancing only"
                if costs.zero() else
                f"trading charged at {args.trade_bps}bp per dollar traded, the benchmark charged "
                "its published fund fee; realized_* is gross and net_* is after costs, both ship. "
                "No taxes. Weights are reset monthly inside each hold and re-solved annually.",
            ],
        },
        "benchmark": args.benchmark,
        # Stated rather than left to be inferred: `run` builds `investable` as every panel column
        # except the benchmark, unconditionally, so SPY is priced here and never held. It is the
        # same claim `backtest.py` records under `universe.benchmark_investable`, and
        # `test_a_held_out_benchmark_is_priced_and_not_investable` reads whichever of the two the
        # document has.
        "benchmark_investable": False,
        "periods": periods,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"\nwrote {args.out}", flush=True)
    block = {"benchmark": args.benchmark, "periods": periods, "mu_model": bt.MuModel().describe(),
             "train_years": args.train_years, "hold_years": args.hold_years,
             "span_years": args.hold_years * len(periods), "caps": caps}
    print(bt.rolling_table(block))
    print(bt.matched_risk_table(block))
    print(bt.summary_table(block))
    if not costs.zero():
        print(bt.cost_table(block))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
