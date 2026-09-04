"""Mutation harness: break the pipeline on purpose, one line at a time, and check the suite notices.

    python -u pipeline/tests/_mutate.py            # all mutants
    python -u pipeline/tests/_mutate.py --only cap  # substring filter on the mutant name
    python -u pipeline/tests/_mutate.py --keep      # leave the mutant trees on disk to inspect

WHY THIS EXISTS
---------------
A green suite is evidence of nothing until you have seen it go red for the right reason.
"72 passed" is compatible with 72 tests that assert a file parses. Every invariant in
`test_*.py` claims to guard a specific way the numbers can be wrong; this file is where
that claim is checked, by reintroducing the wrong version and requiring the named test to
fail.

Leading underscore so pytest does not collect it: it SHELLS OUT to pytest, and a harness
collected by the run it launches is a fork bomb.

HOW A MUTANT IS BUILT
---------------------
The pipeline package is copied to a temp directory, one exact string is replaced in the
copy, and the copy is rebuilt and re-tested with the ORIGINAL tests. Text substitution
rather than monkeypatching, because three of the mutations below live in `build.py`'s
straight-line body where there is no function to patch -- and because a patch that has to
reach inside a function is a patch that tests a different thing than the source says.

A substitution that no longer matches is a HARD ERROR, never a skip. The failure mode of a
stale mutation harness is that it reports "caught" for a mutation it never applied, which
is worse than not having one: it manufactures confidence. Every `old` string below is
asserted to appear exactly once.

Each mutant declares the tests it must kill. "Some test failed" is too weak a bar -- a
mutation that makes the build crash at import would satisfy it while proving nothing about
the invariants. So the expected test names must appear in the failure list, and anything
ELSE that failed is reported as collateral rather than silently accepted.

WHY A 26-SYMBOL SUBSET AND A CACHED PRICE DIRECTORY
---------------------------------------------------
Full runs are 116 assets x 3 caps x 62 quadratic programs per rebuild, and there are 14
mutants. The subset keeps every structural feature the suite needs -- a dividend payer, a
reverse-splitter, two near-duplicate pairs (GLD/IAU, UUP/FXE), a T-bill fund, two funds
that start after the window and must be dropped -- while making the whole harness a
one-minute job. `MARKOWITZ_PRICE_DIR` points the copy at the real price store so no mutant
touches the network, and `MARKOWITZ_DATA_DIR` points the suite at the mutant's own output so
the harness cannot overwrite the committed artifacts it is meant to be testing.

WHY SOME MUTANTS ARE NEVER EXERCISED BY THE REBUILD
---------------------------------------------------
The rebuild runs `--skip-fetch`, so it only ever READS the price store. The mutations to
`store.py`'s writer therefore cannot be caught by anything the rebuild produces -- they are
caught by `test_store_invariants.py`, whose layout tests write a throwaway store with the
mutant's own code. That is the reason those tests are unit tests on `tmp_path` rather than
assertions about the shipped store: a store can only be wrong on the SECOND write, and the
shipped one only shows the result of the last.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
ROOT = PIPELINE.parent
PRICES = PIPELINE / "data" / "prices"

# Chosen for coverage of the STRUCTURE the suite tests, not for portfolio realism:
#   SPY HYG VYM  -- dividend payers, for the close-vs-adjclose adjustment checks
#   USO UNG      -- reverse-splitters
#   GLD IAU      -- the same bullion twice: a near-exact linear dependency
#   UUP FXE FXY  -- a dollar basket against two of its own legs: another one
#   SHV SHY IEF TLT LQD -- the low-vol end, so the minimum-variance solve is not degenerate
#   XLK SMH QQQ  -- the high-return end, so the frontier has somewhere to go
#   XLRE MTUM    -- launched after 2011-01-03, so the window filter has something to drop
SUBSET = (
    "SPY QQQ XLK XLV XLP SMH XLE XLF EEM EFA EWJ VNQ HYG VYM "
    "SHV SHY IEF TLT LQD AGG GLD IAU UUP FXE FXY USO UNG XLRE MTUM"
).split()

BUILD_ARGS = [
    "--skip-fetch",
    "--price-dir", str(PRICES),
    "--symbols", ",".join(SUBSET),
    "--caps", "1.0,0.2",
    "--points", "20",
    "--rf", "0.03",  # never 'auto': a mutation harness must not depend on Yahoo being up
    "--start", "2011-01-03",
]


@dataclass
class Mutation:
    name: str
    why: str
    file: str
    edits: list[tuple[str, str]]
    kills: list[str] = field(default_factory=list)


MUTATIONS: list[Mutation] = [
    Mutation(
        name="monthly-anchor-dropped",
        why="resample().pct_change() without the anchor: the growth curve silently covers a "
            "shorter window than the panel, by an amount proportional to the asset's vol",
        file="fetch.py",
        edits=[
            (
                '    m = panel.resample("ME").last()\n'
                "    anchor = panel.iloc[[0]]\n"
                "    m = pd.concat([anchor, m.loc[m.index > anchor.index[0]]])\n"
                '    return m.pct_change().dropna(how="all")',
                '    m = panel.resample("ME").last()\n'
                '    return m.pct_change().dropna(how="all")',
            )
        ],
        kills=[
            "test_monthly_returns_telescope_to_the_daily_total_growth",
            "test_monthly_compounding_reproduces_the_annualised_mean_return",
        ],
    ),
    Mutation(
        name="annualise-over-bars-not-returns",
        why="the original off-by-one: 3,939 price bars carry 3,938 returns, and `years` is "
            "the exponent that has to match `mean_historical_return`'s",
        file="build.py",
        edits=[("years = (len(panel) - 1) / fetch.TRADING_DAYS_PER_YEAR",
                "years = len(panel) / fetch.TRADING_DAYS_PER_YEAR")],
        kills=["test_monthly_compounding_reproduces_the_annualised_mean_return"],
    ),
    Mutation(
        name="cap-enforced-only-to-solver-tolerance",
        why="clip at cap+1e-4 instead of cap: exactly what the conic solver hands back, and "
            "the file would then state a weight_cap it contains a weight above",
        file="frontier.py",
        edits=[
            ("    w = np.minimum(w / total, cap)", "    w = np.minimum(w / total, cap + 1e-4)"),
            ("    w = np.round(np.minimum(w, cap), 6)", "    w = np.round(np.minimum(w, cap + 1e-4), 6)"),
        ],
        kills=["test_clean_ships_a_vector_that_sums_to_one_and_respects_the_cap"],
    ),
    Mutation(
        name="rounding-residual-ignored",
        why="round for transport but skip the repair: the shipped weights no longer sum to "
            "one, which is the one property everything downstream assumes",
        file="frontier.py",
        edits=[("    w = np.round(np.minimum(w, cap), 6)", "    w = np.minimum(w, cap)")],
        kills=[
            "test_every_frontier_point_is_a_feasible_long_only_portfolio",
            "test_clean_ships_a_vector_that_sums_to_one_and_respects_the_cap",
        ],
    ),
    Mutation(
        name="extrema-picked-from-the-rounded-fields",
        why="locate the tangency by argmax over the shipped `sharpe`, which is rounded to 6 "
            "decimals: the frontier is stationary there, so its neighbour ships the same number "
            "and the tie goes to whichever came first. This SHIPPED -- index 39 where the better "
            "point was 40 -- and a pipeline test resolved the tie the same wrong way, so only "
            "the browser noticed",
        file="frontier.py",
        edits=[
            (
                '        min(range(len(frontier)), key=lambda i: frontier[i][EXACT][1]),\n'
                '        max(range(len(frontier)), key=lambda i: frontier[i][EXACT][2]),',
                '        min(range(len(frontier)), key=lambda i: frontier[i]["vol"]),\n'
                '        max(range(len(frontier)), key=lambda i: frontier[i]["sharpe"]),',
            )
        ],
        kills=["test_extremum_indices_break_a_transport_rounding_tie_by_the_unrounded_value"],
    ),
    Mutation(
        name="working-values-shipped-in-the-json",
        why="forget to strip the unrounded values `build` uses to find the extrema: every point "
            "then ships a second, higher-precision copy of itself, which a reader would believe "
            "over the rounded fields the tests and the browser are written against",
        file="frontier.py",
        edits=[("    for p in frontier:\n        del p[EXACT]", "    pass")],
        kills=["test_no_frontier_point_ships_a_working_field"],
    ),
    Mutation(
        name="vol-ignores-correlations",
        why="measure risk from the variances alone: the classic dropped off-diagonal, and the "
            "reported vol then describes a portfolio nobody holds",
        file="frontier.py",
        edits=[("    ret, vol, sharpe = performance(w, mu, cov, rf)",
                "    ret, vol, sharpe = performance(w, mu, np.diag(np.diag(cov)), rf)")],
        kills=["test_reported_performance_is_recomputable_from_the_shipped_weights"],
    ),
    Mutation(
        name="asset-dots-use-raw-sample-std",
        why="plot the dots in raw-sample coordinates and the curve in shrunk ones: assets "
            "appear to beat their own frontier by a percent",
        file="frontier.py",
        edits=[("    sigma = np.sqrt(np.diag(est.cov.to_numpy()))",
                "    sigma = (panel.pct_change().dropna().std() * np.sqrt(TRADING_DAYS_PER_YEAR)).to_numpy()")],
        kills=[
            "test_asset_dot_coordinates_are_the_shrunk_diagonal",
            "test_build_asset_table_uses_the_shrunk_diagonal",
        ],
    ),
    Mutation(
        name="prune-single-lookback",
        why="pop one dominated neighbour instead of the run: leaves a non-monotone array, and "
            "the browser's dragged handle jumps backwards through it",
        file="frontier.py",
        edits=[('        while kept and kept[-1]["vol"] >= p["vol"] - tol:',
                '        if kept and kept[-1]["vol"] >= p["vol"] - tol:')],
        kills=[
            "test_prune_pops_a_run_of_dominated_points_not_just_the_last",
            "test_prune_output_is_always_strictly_increasing_in_both_axes",
        ],
    ),
    Mutation(
        name="max-return-ignores-cap",
        why="return max(mu) as the frontier's right end: every top target goes infeasible and "
            "surfaces as a generic solver error rather than a bad target",
        file="frontier.py",
        edits=[("        take = min(cap, remaining)", "        take = remaining")],
        kills=[
            "test_max_feasible_return_matches_a_brute_force_linear_program",
            "test_build_produces_a_frontier_no_asset_can_beat",
        ],
    ),
    Mutation(
        name="panel-forward-filled",
        why="ffill the holes instead of intersecting: a fabricated zero return, which deflates "
            "every variance and pulls every correlation toward zero. INERT on the shipped store "
            "-- all 116 survivors share one calendar, so there is no hole to fill -- which is "
            "why the guard is a synthetic-store test and not the real-artifact one whose name "
            "sounds like it",
        file="fetch.py",
        edits=[('    wide = wide.dropna(how="any")', '    wide = wide.ffill().dropna(how="any")')],
        kills=["test_load_panel_intersects_trading_days_rather_than_forward_filling"],
    ),
    Mutation(
        name="reindexed-onto-a-calendar",
        why="reindex to daily frequency and ffill: 40% more rows, all of them Saturdays",
        file="fetch.py",
        edits=[('    df = df[~df.index.duplicated(keep="last")].sort_index()',
                '    df = df[~df.index.duplicated(keep="last")].sort_index()\n'
                '    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D")).ffill()')],
        kills=["test_parse_indexes_by_the_new_york_trading_date_not_the_raw_timestamp"],
    ),
    Mutation(
        name="start-filter-disabled",
        why="keep the late arrivals: the window stops being a parameter and becomes whatever "
            "the youngest fund allows",
        file="fetch.py",
        edits=[("        if start is not None and s.index.min() > pd.Timestamp(start):",
                "        if False and s.index.min() > pd.Timestamp(start):")],
        kills=["test_every_kept_asset_predates_the_requested_start"],
    ),
    Mutation(
        name="covariance-annualised-at-365",
        why="252 trading days vs 365 calendar days: a 20% error in every volatility, and the "
            "single easiest constant to get wrong",
        file="frontier.py",
        edits=[("    shrink = risk_models.CovarianceShrinkage(panel, frequency=TRADING_DAYS_PER_YEAR)",
                "    shrink = risk_models.CovarianceShrinkage(panel, frequency=365.0)")],
        kills=["test_covariance_is_symmetric_psd_and_annualised"],
    ),
    Mutation(
        name="arithmetic-mean-returns",
        why="arithmetic instead of geometric: +sigma^2/2, which is 2 percentage points on a "
            "20%-vol asset and reorders the cross-section",
        file="frontier.py",
        edits=[("        panel, compounding=True, frequency=TRADING_DAYS_PER_YEAR",
                "        panel, compounding=False, frequency=TRADING_DAYS_PER_YEAR")],
        kills=[
            "test_expected_return_is_the_geometric_mean_not_the_arithmetic_one",
            "test_monthly_compounding_reproduces_the_annualised_mean_return",
        ],
    ),
    Mutation(
        name="parse-collapses-close-onto-adjclose",
        why="return the raw close as both columns: the store's audit sample then records the "
            "PRICE return as the vendor's total return, and the reconstruction check loses the "
            "only independent reference it has",
        file="fetch.py",
        edits=[('        {"close": quote["close"], "adjclose": adj[0]["adjclose"]}, index=pd.Index(idx, name="date")',
                '        {"close": quote["close"], "adjclose": quote["close"]}, index=pd.Index(idx, name="date")')],
        # Only the parse test: the rebuild runs --skip-fetch, so the sample on disk is the real
        # one and the reconstruction check still has its reference. Caught at the boundary the
        # damage enters through, which is the earliest place it CAN be caught.
        kills=["test_parse_keeps_close_and_adjclose_separately"],
    ),
    Mutation(
        name="dividends-never-applied",
        why="return the split-adjusted close as the total return: the whole dividend adjustment "
            "gone, which is the same error as reading `close` was before the store change",
        file="store.py",
        edits=[("    return close.astype(float) * factor", "    return close.astype(float)")],
        kills=[
            "test_reconstruction_matches_the_vendors_adjclose",
            "test_every_dividend_payer_in_the_store_actually_moves_the_series",
            "test_the_adjustment_scales_bars_before_the_ex_date_and_leaves_the_rest",
            "test_the_adjustment_is_visible_at_the_start_of_a_dividend_payer",
            "test_total_return_compounds_at_least_as_fast_as_price_return_for_every_asset",
        ],
    ),
    Mutation(
        name="ex-date-bar-adjusted-too",
        why="off by one at the ex-date boundary: the ex-date's own close already reflects the "
            "distribution, so adjusting it fabricates a return on exactly that day -- one bar "
            "in 4,000 per event, invisible in aggregate",
        file="store.py",
        edits=[("        factor[:pos] *= 1.0 - amount / prior",
                "        factor[:pos + 1] *= 1.0 - amount / prior")],
        kills=[
            "test_the_adjustment_scales_bars_before_the_ex_date_and_leaves_the_rest",
            "test_the_last_bar_is_never_adjusted_whatever_the_dividend_history",
            "test_reconstruction_matches_the_vendors_adjclose",
        ],
    ),
    Mutation(
        name="dividend-factors-summed-not-compounded",
        why="subtract each yield instead of multiplying the factors: right to first order and "
            "wrong by several percent of terminal wealth over 135 SPY distributions, in the "
            "direction that overstates the total return",
        file="store.py",
        edits=[("        factor[:pos] *= 1.0 - amount / prior",
                "        factor[:pos] -= amount / prior")],
        kills=[
            "test_dividends_compound_across_events_rather_than_summing",
            "test_reconstruction_matches_the_vendors_adjclose",
        ],
    ),
    Mutation(
        name="unchanged-partitions-rewritten-anyway",
        why="write every partition the incoming data covers, changed or not. The collector "
            "refetches full histories, so this rewrites all 34 year files weekly -- 34 new "
            "compressed blobs in the history for 250 new bars, i.e. the entire saving gone",
        file="store.py",
        edits=[("        if old.shape == merged.shape and old.equals(merged):\n            return False",
                "        if False:\n            return False")],
        kills=["test_rewriting_unchanged_history_touches_no_file"],
    ),
    Mutation(
        name="close-partition-appends-instead-of-upserting",
        why="keep the old rows when re-writing a symbol: a refetch stops correcting a bad print "
            "and starts duplicating the date, which `read_close` then resolves to one of the two "
            "silently",
        file="store.py",
        edits=[('            chunk = pd.concat([old[~old["symbol"].isin(syms)], chunk], ignore_index=True)',
                "            chunk = pd.concat([old, chunk], ignore_index=True)")],
        kills=["test_rewriting_a_symbol_replaces_its_rows_instead_of_duplicating_them"],
    ),
    Mutation(
        name="event-log-appends-instead-of-upserting",
        why="never clear a symbol's old event rows: a withdrawn or mis-parsed distribution stays "
            "in the log forever and keeps being applied to every earlier bar",
        file="store.py",
        edits=[('        incoming = pd.concat([old[~old["symbol"].isin(symbols)], incoming], ignore_index=True)',
                "        incoming = pd.concat([old, incoming], ignore_index=True)")],
        kills=["test_a_symbol_that_stops_paying_has_its_old_events_cleared"],
    ),
    Mutation(
        name="everything-written-to-the-current-year",
        why="partition on the run date rather than the bar date -- the plausible version of the "
            "mistake, since the new bars ARE this year's. Historical years stop being frozen, "
            "which is the entire reason the layout is partitioned",
        file="store.py",
        edits=[('    return {int(y): chunk for y, chunk in long.groupby(long["date"].dt.year, sort=True)}',
                "    return {int(y): chunk for y, chunk in "
                "long.groupby(pd.Series(pd.Timestamp.today().year, index=long.index), sort=True)}")],
        kills=["test_a_year_partition_contains_only_that_year"],
    ),
    Mutation(
        name="freshness-reads-only-the-newest-partition",
        why="a symbol whose last bar is in an earlier year reports no date at all, so on the "
            "first trading days of January every symbol looks stale and the whole universe is "
            "refetched -- a bug that can only appear one week a year",
        file="store.py",
        edits=[("    paths = _year_paths(price_dir)[-2:]", "    paths = _year_paths(price_dir)[-1:]")],
        kills=["test_last_dates_reports_the_newest_bar_per_symbol"],
    ),
]


FAILED_RE = re.compile(r"^FAILED\s+\S+::([^\s]+)")


def _failed_tests(stdout: str) -> set[str]:
    """Test names from pytest's `-rf` short summary, with any parametrisation suffix stripped."""
    out = set()
    for line in stdout.splitlines():
        m = FAILED_RE.match(line.strip())
        if m:
            out.add(m.group(1).split("[")[0])
    return out


def _apply(tree: Path, mut: Mutation) -> None:
    path = tree / "pipeline" / mut.file
    text = path.read_text()
    for old, new in mut.edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"\nMUTATION HARNESS IS STALE: {mut.name!r} expected exactly one occurrence of\n\n"
                f"{old}\n\nin {mut.file}, found {n}. A mutation that no longer applies would be "
                "reported as 'caught' while testing nothing -- fix the harness before trusting it."
            )
        text = text.replace(old, new)
    path.write_text(text)


def _stage(tmp: Path, name: str) -> Path:
    tree = tmp / name
    (tree / "pipeline").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        PIPELINE,
        tree / "pipeline",
        ignore=shutil.ignore_patterns("data", "__pycache__", ".pytest_cache", "*.pyc"),
    )
    return tree


def _run(tree: Path) -> tuple[bool, set[str], str]:
    """Rebuild the artifacts, then run the ORIGINAL suite against them."""
    data = tree / "data"
    env = {
        "MARKOWITZ_DATA_DIR": str(data),
        "MARKOWITZ_PRICE_DIR": str(PRICES),
        "PATH": __import__("os").environ["PATH"],
        "HOME": __import__("os").environ.get("HOME", str(tree)),
    }
    build = subprocess.run(
        [sys.executable, "-u", str(tree / "pipeline" / "build.py"), "--out", str(data), *BUILD_ARGS],
        capture_output=True,
        text=True,
        cwd=tree,
        env=env,
    )
    if build.returncode != 0:
        tail = (build.stderr or build.stdout).strip().splitlines()[-1:]
        return False, set(), f"build failed: {tail[0] if tail else '(no output)'}"

    test = subprocess.run(
        [sys.executable, "-m", "pytest", str(tree / "pipeline" / "tests"), "-q", "--tb=no", "-rf", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=tree,
        env=env,
    )
    summary = test.stdout.strip().splitlines()[-1] if test.stdout.strip() else "(no output)"
    return True, _failed_tests(test.stdout), summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="substring filter on the mutant name")
    ap.add_argument("--keep", action="store_true", help="leave the mutant trees on disk")
    args = ap.parse_args(argv)

    # Checked through the store's own API rather than by globbing, so a layout change cannot
    # turn this precondition into a silent pass. `--skip-fetch` on an empty store would build a
    # frontier from nothing and every mutant would be "caught" by the same missing-data error.
    sys.path.insert(0, str(PIPELINE))
    import store  # noqa: PLC0415 - after the path insert, by necessity

    if not store.exists(PRICES) or len(store.stored_symbols(PRICES)) < 20:
        raise SystemExit(f"no price store at {PRICES} -- run `python -u pipeline/build.py` first")

    muts = [m for m in MUTATIONS if not args.only or args.only in m.name]
    if not muts:
        raise SystemExit(f"no mutant matches {args.only!r}")

    tmp = Path(tempfile.mkdtemp(prefix="markowitz-mutants-"))
    print(f"mutants in {tmp}", flush=True)
    survivors: list[str] = []
    unexpected: list[tuple[str, set[str]]] = []
    t0 = time.time()

    try:
        # BASELINE. An unmutated copy must be fully green on the subset, or every "caught"
        # below is meaningless -- it would be catching the subset, not the mutation.
        print("\n=== baseline (no mutation) ===", flush=True)
        built, failed, summary = _run(_stage(tmp, "baseline"))
        print(f"  {summary}", flush=True)
        if not built or failed:
            raise SystemExit(
                f"BASELINE IS NOT GREEN ({summary}); failures: {sorted(failed) or 'build error'}.\n"
                "Nothing below can be interpreted until the unmutated subset passes."
            )

        for i, mut in enumerate(muts, 1):
            tree = _stage(tmp, mut.name)
            _apply(tree, mut)
            built, failed, summary = _run(tree)

            print(f"\n=== [{i}/{len(muts)}] {mut.name} ===", flush=True)
            print(f"  {mut.why}", flush=True)

            if not built:
                # Still caught, but by a crash rather than by an assertion. Reported
                # separately because a crash says nothing about whether the invariant holds.
                print(f"  CAUGHT (build refused to run) -- {summary}", flush=True)
                continue

            missed = [k for k in mut.kills if k not in failed]
            extra = failed - set(mut.kills)
            print(f"  {summary}", flush=True)
            for k in mut.kills:
                print(f"  {'KILLED BY ' if k not in missed else 'SURVIVED  '} {k}", flush=True)
            if extra:
                print(f"  also failed ({len(extra)}): {', '.join(sorted(extra))}", flush=True)
                unexpected.append((mut.name, extra))
            if missed:
                survivors.append(f"{mut.name}: not caught by {', '.join(missed)}")
            elif not failed:
                survivors.append(f"{mut.name}: the whole suite stayed green")
    finally:
        if args.keep:
            print(f"\nmutant trees left at {tmp}", flush=True)
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 78}")
    print(f"{len(muts)} mutants in {time.time() - t0:.0f}s")
    if unexpected:
        print("\ncollateral (a mutation killed tests it was not aimed at -- usually fine, but")
        print("it means those tests overlap, so read it before assuming coverage is broad):")
        for name, extra in unexpected:
            print(f"  {name}: {len(extra)} extra")
    if survivors:
        print(f"\n{len(survivors)} SURVIVOR(S) -- a test claims to guard something it does not:")
        for s in survivors:
            print(f"  {s}")
        return 1
    print("\nevery mutant was killed by the test named for it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
