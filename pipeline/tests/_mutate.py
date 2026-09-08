"""Mutation harness: break the pipeline on purpose, one line at a time, and check the suite notices.

    python -u pipeline/tests/_mutate.py            # all mutants
    python -u pipeline/tests/_mutate.py --only cap  # substring filter on the mutant name
    python -u pipeline/tests/_mutate.py --check     # every needle still matches; run nothing
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
asserted to appear exactly once -- twice over, and the two checks are not redundant.
`_preflight` reads `pipeline/` and reports EVERY stale needle before a single mutant runs;
`_apply` re-checks the staged copy, which is the one that actually gets mutated. Without the
first, a needle that went stale in the ninetieth record costs forty minutes to find and takes
the other ninety-nine reports down with it -- which is how four of them (three in
`backtest.py`, one in `delisted.py`) sat stale after the cost-model and `fetch_longest`
refactors on 2026-09-06 without anyone learning about more than the first. Without the second,
the guarantee weakens from "this edit landed" to "an edit like it would have landed somewhere".
`--check` is the first one on its own, and it is seconds.

Each mutant declares the tests it must kill. "Some test failed" is too weak a bar -- a
mutation that makes the build crash at import would satisfy it while proving nothing about
the invariants. So the expected test names must appear in the failure list, and anything
ELSE that failed is reported as collateral rather than silently accepted.

WHY A 26-SYMBOL SUBSET AND A CACHED PRICE DIRECTORY
---------------------------------------------------
Full runs are 116 assets x 3 caps x 62 quadratic programs per rebuild, and there are ~100
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

The `universe.py` mutants are the same shape for a different reason. They remove a VALIDATION,
and the shipped universe file is valid -- so the rebuild produces byte-identical artifacts and
no artifact test can possibly notice. They are caught by `test_universe_invariants.py`, which
writes a deliberately broken universe to `tmp_path` and requires the raise. A validator can only
be tested with input the shipped data by definition does not contain, and the corollary is that
adding a `raise` to `universe.py` without adding a case there guards nothing.

The `backtest.py` mutants take that furthest -- `backtest.py` is not in the rebuild's call graph
at all, and the property they exist for (that no weight depends on data from after the cutoff) is
one the shipped panel cannot express in either direction. See the comment above that group.

WHAT THIS HARNESS CANNOT REACH AT ALL
-------------------------------------
`test_results_invariants.py` reads `pipeline/results/*.json`, and `_stage` COPIES those files into
the mutant tree rather than regenerating them -- the rebuild runs `build.py`, never `backtest.py`.
So no mutation here can be caught by that file: the committed evidence is identical in every
mutant tree, and adding a mutant that expects it to fail would be reported SURVIVED for a reason
that has nothing to do with the code.

`_stage` also ignores `data/`, which has the same consequence one directory over: the second
vendor's cache at `pipeline/data/delisted/` is absent from every mutant tree, so the two tests in
`test_delisted_invariants.py` that read it -- the served-span check and the provenance check --
SKIP rather than run. Every `delisted-*` mutant below is therefore killed by one of that file's
SYNTHETIC tests, and a mutant aimed at the cache would be reported MISKILLED. Its one falsifiable assertion (a block whose `benchmark`
field names a symbol its periods never measured) was mutation-checked by hand on 2026-09-06, and
the check is repeatable -- point the WINDOW block's label at `u.benchmark` in `main` and run

    python -u pipeline/backtest.py --universe sp500_2023 --benchmark SPY --start 2019-08-01 \\
      --years 3 --window --train-years 4 --cutoff-years-back 3 --window-holds 1,2,3 \\
      --no-rolling --out pipeline/results/_mut.json

The same edit to the ROLLING block is not worth automating either, for the opposite reason: it is
already fatal. `summary_table` subscripts the row it looked up and raises before `write_json`, so
that half of the bug cannot reach a file. The window printer is None-tolerant by design and writes
one. Verified both ways the same day.
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
        name="frontier-stops-short-of-its-right-end",
        why="drop the constructed right endpoint (= the pre-2026-09-05 build): the curve ends "
            "one grid step below r_hi, so the highest-return assets plot past the end of the "
            "line and read as beating the frontier, and the maximum-return portfolio is not "
            "selectable in the browser at all. Nothing else in the artifact looks wrong -- "
            "n_points is off by one and `pruned` and `failed_targets` are unchanged",
        file="frontier.py",
        edits=[("    points.append(_point(_clean(\n"
                "        dict(zip(symbols, max_return_weights(est.mu, cap))), symbols, cap,\n"
                "    ), symbols, mu_v, cov_v, rf))",
                "    _ = max_return_weights(est.mu, cap)")],
        kills=["test_build_frontier_reaches_the_maximum_feasible_return"],
    ),
    Mutation(
        name="max-return-weights-rot-behind-a-correct-scalar",
        why="give max_feasible_return back its own independent greedy loop and break the "
            "vector's sort order. This is the shape the LP test CANNOT catch: the scalar stays "
            "exactly right, so `test_max_feasible_return_matches_a_brute_force_linear_program` "
            "passes, and only a test that looks at the WEIGHTS notices that the frontier now "
            "ends at the lowest-return corner. It is the reason the two share one function",
        file="frontier.py",
        edits=[
            ('    for i in np.argsort(-mu.to_numpy(), kind="stable"):',
             '    for i in np.argsort(mu.to_numpy(), kind="stable"):'),
            ("    return float(max_return_weights(mu, cap) @ mu.to_numpy())",
             "    order = np.sort(mu.to_numpy())[::-1]\n"
             "    remaining, total = 1.0, 0.0\n"
             "    for m in order:\n"
             "        take = min(cap, remaining)\n"
             "        total += take * m\n"
             "        remaining -= take\n"
             "        if remaining <= 1e-12:\n"
             "            break\n"
             "    return float(total)"),
        ],
        kills=["test_max_return_weights_is_a_feasible_portfolio_attaining_that_return"],
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
        name="frontier-files-shipped-unstamped",
        why="write the frontier payloads without the run stamp -- the state this shipped in. The "
            "browser then cannot detect a MIXED BUNDLE (a cached frontier against a fresh "
            "covariance), which is the weekly cron's own failure mode and is invisible whenever "
            "the symbol set has not changed",
        file="build.py",
        edits=[('        payload = {"generated_at": generated_at, **fr.build(est, rf=rf_value, cap=cap, n_points=args.points)}',
                "        payload = fr.build(est, rf=rf_value, cap=cap, n_points=args.points)")],
        kills=["test_every_artifact_carries_the_same_generated_at"],
    ),
    Mutation(
        name="one-artifact-stamped-from-another-run",
        why="the stamp present on all six files but not AGREEING -- what a stale HTTP cache "
            "actually produces. Distinct from the mutation above, which removes the field: this "
            "one leaves every field in place and only the cross-file comparison can see it",
        file="build.py",
        edits=[('    manifest = {\n        "generated_at": generated_at,',
                '    manifest = {\n        "generated_at": "1970-01-01T00:00:00+00:00",')],
        kills=["test_every_artifact_carries_the_same_generated_at"],
    ),
    Mutation(
        name="cap-slug-collision-unchecked",
        why="drop the distinct-slug guard: 0.125 and 0.124 both slug to `cap12`, the second solve "
            "overwrites the first, and the manifest advertises two caps that are one file -- so "
            "every cross-cap invariant compares a file with itself and passes",
        file="build.py",
        edits=[("    if len(set(slugs)) != len(slugs):", "    if False:")],
        kills=["test_two_caps_that_share_a_slug_are_refused_rather_than_silently_merged"],
    ),
    Mutation(
        name="group-labels-dropped-from-the-manifest",
        why="ship `groups` without `group_labels`: the SPA has no map of its own by design, so "
            "every filter button, tooltip and table cell renders an empty group name",
        file="build.py",
        edits=[('        "group_labels": dict(u.group_labels),', "")],
        kills=["test_the_manifest_names_every_group_it_asks_the_page_to_show"],
    ),
    Mutation(
        name="deliberate-exclusions-not-shipped",
        why="validate the universe's exclusions and their evidence, then drop them -- the state "
            "this shipped in. The argument for leaving an instrument out is the whole reason to "
            "record leaving it out, and it never reached the page",
        file="build.py",
        edits=[('            "deliberate": dict(u.excluded),', "")],
        kills=["test_deliberate_exclusions_ship_with_their_evidence"],
    ),
    Mutation(
        name="undeclared-asset-group-accepted",
        why="drop the group check in the loader: the asset is optimised over and held, but has no "
            "label, no filter button and no legend entry -- present in the portfolio and absent "
            "from the chart that is supposed to explain it",
        file="universe.py",
        edits=[('        if row["group"] not in groups:', "        if False:")],
        kills=["test_an_asset_whose_group_is_not_declared_is_rejected"],
    ),
    Mutation(
        name="exclusion-without-a-reason-accepted",
        why="allow an empty exclusion reason: an instrument leaves the universe with nothing to "
            "say whether that was a measured data fault, a judgement, or an oversight",
        file="universe.py",
        edits=[("    if blank:", "    if False:")],
        kills=["test_an_exclusion_with_no_reason_is_rejected"],
    ),
    Mutation(
        name="mislabelled-group-silently-ignored",
        why="accept a [group_labels] key that names no declared group: the typo is inert and the "
            "group it was meant for keeps the derived label, so the only symptom is the button "
            "the label was added to fix",
        file="universe.py",
        edits=[("    if stray:", "    if False:")],
        kills=["test_a_label_for_an_undeclared_group_is_rejected"],
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
    Mutation(
        name="solver-left-to-the-dependency-default",
        # The one mutant here whose damage is a function of PROBLEM SIZE, which is why it needs
        # two killers and why one of them is a monkeypatch rather than a number. PyPortfolioOpt's
        # default for a QP is OSQP, a first-order method: at this harness's 26 assets and at the
        # shipped 116 it converges and nothing changes, and at 475 it returns `user_limit` -- its
        # iteration budget -- which `_solve` correctly reports as "not attainable", because an
        # exhausted solver and an infeasible problem are indistinguishable from outside. So the
        # 500-asset test is the one that shows the harm, and it is seed-dependent (see its
        # fixture); the monkeypatch test is the one that always fires. Both must fail, or the
        # guard rests on a random draw.
        why="drop the named solver and take the dependency's default: the answer then depends on "
            "how many assets the universe happens to have, and a solver upgrade upstream can move "
            "a published number with nothing in this repo changing",
        file="frontier.py",
        edits=[('ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, cap), solver=SOLVER)',
                'ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, cap))')],
        kills=["test_solve_names_its_solver_rather_than_taking_the_default",
               "test_every_objective_solves_on_a_universe_of_five_hundred_assets"],
    ),

    # ------------------------------------------------------------------------- backtest.py
    #
    # THESE ARE THE `universe.py` SHAPE TAKEN FURTHER. Not one of them can be caught by a
    # rebuild: `backtest.py` writes nothing into `web/public/data` and the artifact tests never
    # import it. They are caught by `test_backtest_invariants.py`, which builds a synthetic
    # panel -- and they are the reason it has to. The shipped panel has 116 assets trading on
    # the same days with no gaps, so it cannot tell a monthly rebalance from a buy-and-hold,
    # and nothing on disk can tell an honest backtest from a leaking one.
    #
    # A leak is the failure this whole group exists for. Every number `backtest.py` prints
    # stays plausible and stays ordered the way the reader expects when the weights are chosen
    # with knowledge of the test window -- the table looks BETTER, which is the trap. The first
    # two mutants are that bug in its two forms.
    Mutation(
        name="backtest-lookahead-estimate-over-the-whole-panel",
        why="estimate mu and Sigma on the full panel instead of the training half: the tangency "
            "portfolio is then chosen knowing what the test window did, and every realized "
            "number in the table becomes a description of the fit rather than a measurement",
        file="backtest.py",
        edits=[("    est = fr.estimate(train)", "    est = fr.estimate(panel)")],
        kills=["test_the_solved_weights_do_not_depend_on_a_single_bar_after_the_cutoff"],
    ),
    Mutation(
        name="backtest-lookahead-rf-from-the-end-of-the-window",
        why="take the risk-free rate from the END of the holding period rather than the cutoff: a "
            "subtler leak than the estimates, because it moves only the tangency point -- and "
            "rates fell by more than 100bp inside these windows, so it moves it",
        file="backtest.py",
        edits=[("else rf_asof(irx, cutoff)", "else rf_asof(irx, test.index.max())")],
        kills=[
            "test_the_solved_weights_do_not_depend_on_a_single_bar_after_the_cutoff",
            "test_the_realized_sharpe_uses_the_test_window_rate_not_the_cutoff_rate",
        ],
    ),
    Mutation(
        name="backtest-rf-backfilled-so-a-future-quote-answers-for-a-missing-one",
        why="`bfill` instead of `asof`: on any date without a quote of its own the investor is "
            "handed the NEXT rate, and a date before the series starts gets one silently "
            "instead of raising",
        file="backtest.py",
        edits=[(
            "    v = irx.asof(when)",
            "    v = irx.reindex(irx.index.union([when])).bfill().loc[when]",
        )],
        kills=["test_rf_asof_never_returns_a_rate_from_the_future"],
    ),
    Mutation(
        name="backtest-cutoff-bar-dropped-from-training",
        why="train on everything strictly before the cutoff: the entry bar is then in neither "
            "half's returns, which loses a day and, more to the point, makes the two halves' "
            "bar counts stop adding up -- the one arithmetic check on the split",
        file="backtest.py",
        edits=[("    train = panel.loc[:cutoff]", "    train = panel.loc[:cutoff].iloc[:-1]")],
        kills=["test_the_cutoff_bar_belongs_to_both_halves_and_is_counted_once_each"],
    ),
    Mutation(
        name="backtest-monthly-rebalance-never-resets",
        why="the month-boundary test never fires, so 'monthly' silently returns buy-and-hold: "
            "the two columns of the JSON become one number printed twice, and the comparison "
            "with the SPA's growth curve is no longer a comparison",
        file="backtest.py",
        edits=[("        if period[t] != period[t - 1]:", "        if t == 0:")],
        kills=["test_monthly_rebalancing_actually_resets_the_weights_at_the_month_boundary"],
    ),
    Mutation(
        name="backtest-monthly-rebalance-on-the-first-bar-of-the-new-month",
        why="price the new segment off bar t instead of t-1: one day of drift leaks into every "
            "segment, in the direction of whatever moved on the month's first bar",
        file="backtest.py",
        edits=[
            ("            base_row, base_val = t - 1, v[t - 1]", "            base_row, base_val = t, v[t - 1]"),
        ],
        kills=["test_monthly_rebalancing_actually_resets_the_weights_at_the_month_boundary"],
    ),
    Mutation(
        name="backtest-hold-path-as-a-daily-rebalanced-sum-of-returns",
        why="compound `sum_i w_i r_i` and call it buy-and-hold: the classic overstatement, worth "
            "0.25 on a hand-computed 2-asset case and growing with both the window and the "
            "dispersion -- it is the 'simplification' this function's docstring exists to refuse",
        file="backtest.py",
        edits=[(
            "        return pd.Series((P / P[0]) @ w, index=prices.index)",
            "        return pd.Series(np.cumprod(np.r_[1.0, (P[1:] / P[:-1] - 1.0) @ w + 1.0]),"
            " index=prices.index)",
        )],
        kills=["test_the_monthly_path_is_not_the_daily_rebalanced_weighted_sum_of_returns"],
    ),
    Mutation(
        name="backtest-annualised-over-bars-not-returns",
        why="the same off-by-one `test_annualisation_is_over_returns_not_bars` guards in the "
            "pipeline, in the other file that annualises: 10.00% a year is reported as 9.98%",
        file="backtest.py",
        edits=[("    n = len(v) - 1\n", "    n = len(v)\n")],
        kills=["test_annualisation_round_trips_at_252_over_returns_not_bars"],
    ),
    Mutation(
        name="backtest-annualised-at-365",
        why="calendar days on a trading-day series: the realized column is inflated by ~45% of "
            "itself while staying the right order of magnitude, so it reads as a good year",
        file="backtest.py",
        edits=[(
            "    ret = growth ** (fetch.TRADING_DAYS_PER_YEAR / n) - 1.0",
            "    ret = growth ** (365.0 / n) - 1.0",
        )],
        kills=["test_annualisation_round_trips_at_252_over_returns_not_bars"],
    ),
    Mutation(
        name="backtest-drawdown-measured-from-the-start-not-the-peak",
        why="`v.iloc[0]` for `v.cummax()`: a portfolio that rose 25% and gave it all back reports "
            "no drawdown at all, and the column is silently zero for every strategy that ended "
            "above where it began -- which is most of them",
        file="backtest.py",
        edits=[(
            '"max_drawdown": round(float((1.0 - v / v.cummax()).max()), 6),',
            '"max_drawdown": round(float((1.0 - v / v.iloc[0]).max()), 6),',
        )],
        kills=["test_max_drawdown_is_a_positive_fraction_of_the_peak"],
    ),
    Mutation(
        name="backtest-drawdown-shipped-negative",
        why="the sign convention `export.ts` fixes, broken in the other direction: a bare "
            "drawdown is ambiguous in sign the moment it leaves the file that produced it, and "
            "-0.21 against 0.21 is exactly the pair nobody notices in a table",
        file="backtest.py",
        edits=[(
            '"max_drawdown": round(float((1.0 - v / v.cummax()).max()), 6),',
            '"max_drawdown": round(float((v / v.cummax() - 1.0).min()), 6),',
        )],
        kills=["test_max_drawdown_is_a_positive_fraction_of_the_peak"],
    ),
    Mutation(
        name="backtest-rolling-training-window-grows-instead-of-rolling",
        why="train on everything before the cutoff rather than a fixed window: the rolling mode "
            "silently becomes the nested mode, so the ten periods stop being ten answers to the "
            "same question and the early ones are estimated from 5 years while the late ones get 14",
        file="backtest.py",
        edits=[("        train = panel.loc[train_from:cutoff]", "        train = panel.loc[:cutoff]")],
        kills=["test_a_rolling_training_window_is_fixed_length_and_ends_at_the_cutoff"],
    ),
    Mutation(
        name="backtest-rolling-periods-overlap",
        why="step the cutoffs by one year regardless of the holding period: at a 2-year hold the "
            "windows then overlap by half, and the whole reason this mode exists -- that the "
            "periods are separate observations -- is gone with nothing looking wrong",
        file="backtest.py",
        edits=[(
            "        cutoff = _bar_at_or_before(panel, end - pd.DateOffset(years=hold_years * k),",
            "        cutoff = _bar_at_or_before(panel, end - pd.DateOffset(years=k),",
        )],
        kills=["test_rolling_holding_periods_are_laid_end_to_end_and_never_overlap"],
    ),
    Mutation(
        name="backtest-rolling-holding-period-runs-to-the-end-of-the-panel",
        why="hold every period to today instead of for `hold_years`: the early periods then cover "
            "ten years and the last covers one, all of them overlapping, and the average of the "
            "column is an average over windows of different lengths",
        file="backtest.py",
        edits=[(
            "        test_to = _bar_at_or_before(panel, cutoff + pd.DateOffset(years=hold_years),",
            "        test_to = _bar_at_or_before(panel, end,",
        )],
        kills=["test_rolling_holding_periods_are_laid_end_to_end_and_never_overlap"],
    ),
    Mutation(
        name="backtest-rolling-lookahead-trains-past-the-cutoff",
        why="extend the training window a year PAST the cutoff: the leak in the mode where it is "
            "hardest to see, because every period still has a plausible forecast and the periods "
            "still do not overlap",
        file="backtest.py",
        edits=[(
            "        train = panel.loc[train_from:cutoff]",
            "        train = panel.loc[train_from:cutoff + pd.DateOffset(years=1)]",
        )],
        kills=["test_a_rolling_period_cannot_see_past_its_own_cutoff"],
    ),
    Mutation(
        name="backtest-realized-sharpe-charged-the-cutoff-rate",
        why="use the cutoff's yield as the hurdle for the realized return: the investor is "
            "charged a rate from the wrong end of the window, which flatters or penalises every "
            "realized Sharpe by however far rates moved",
        file="backtest.py",
        edits=[(
            '            "realized_monthly": realized(v_month, rf_test),',
            '            "realized_monthly": realized(v_month, rf_cut),',
        )],
        kills=["test_the_realized_sharpe_uses_the_test_window_rate_not_the_cutoff_rate"],
    ),

    # The matched-risk comparison, which is the one place in this file where a leak would look
    # like SCRUPULOUSNESS. Aiming at the benchmark's realized volatility is the obviously fairer
    # target and is unknowable at the cutoff; sizing a levered position with the volatility it
    # turned out to have is the same mistake wearing the word "fair". Both make the table more
    # defensible and the result meaningless, and neither moves a single weight vector -- which is
    # why `leverage` and `risk_target` had to go into the no-lookahead test's comparison tuple.
    Mutation(
        name="backtest-matched-risk-target-from-the-test-window",
        why="aim at the benchmark's REALIZED volatility instead of its forecast: the position is "
            "then sized with knowledge of the window it is measured over, and every weight "
            "vector in the file is untouched",
        file="backtest.py",
        edits=[(
            "    target_vol = predicted_vol(est, benchmark) if benchmark in est.mu.index else None",
            "    target_vol = predicted_vol(fr.estimate(test), benchmark) "
            "if benchmark in est.mu.index else None",
        )],
        kills=["test_the_solved_weights_do_not_depend_on_a_single_bar_after_the_cutoff",
               "test_the_risk_target_is_the_benchmarks_forecast_volatility_not_its_realized_one"],
    ),
    Mutation(
        name="backtest-leverage-from-the-realized-volatility",
        why="size the levered position by what the portfolio's volatility turned out to be, so "
            "it lands on the risk target every time -- by cheating",
        file="backtest.py",
        edits=[("            k = target_vol / own_vol",
                '            k = target_vol / row["realized_monthly"]["vol"]')],
        kills=["test_the_solved_weights_do_not_depend_on_a_single_bar_after_the_cutoff"],
    ),
    Mutation(
        name="backtest-matched-risk-accepts-a-portfolio-short-of-the-target",
        why="`efficient_risk` constrains volatility to be AT MOST the target, so dropping the "
            "reachability check ships a row labelled 'matched risk' that carries less risk than "
            "it claims whenever the cap puts the target out of reach",
        file="backtest.py",
        edits=[("            if w_mr is not None and reaches(w_mr, est, target):",
                "            if w_mr is not None:")],
        kills=["test_a_cap_that_cannot_reach_the_benchmarks_risk_omits_the_row_rather_than_missing_it"],
    ),
    Mutation(
        name="backtest-leverage-applied-to-the-total-return-not-the-excess",
        why="scale the whole return rather than the excess over rf: the borrowing is never paid "
            "for in the predicted column, so leverage looks free and the predicted Sharpe rises "
            "with k -- the identity that makes the comparison fair is exactly this one",
        file="backtest.py",
        edits=[('                    "ret": round(rf_cut + k * (row["predicted"]["ret"] - rf_cut), 6),',
                '                    "ret": round(k * row["predicted"]["ret"], 6),')],
        kills=["test_leverage_scales_the_excess_return_and_the_risk_by_the_same_factor"],
    ),
    Mutation(
        name="backtest-lever-charged-the-first-rate-for-the-whole-window",
        why="finance a year of borrowing at the rate quoted on the cutoff: a subsidy when rates "
            "rise, and the error scales with k - 1, so it is invisible on the unlevered rows and "
            "worth several percent on the 3x ones",
        file="backtest.py",
        edits=[('the value path\'s bars")\n'
                "    cash = (1.0 + rf_annual) ** (1.0 / fetch.TRADING_DAYS_PER_YEAR) - 1.0",
                'the value path\'s bars")\n'
                "    cash = (1.0 + rf_annual.iloc[0]) ** (1.0 / fetch.TRADING_DAYS_PER_YEAR) - 1.0")],
        kills=["test_levering_charges_the_rate_that_moved_not_the_one_at_the_cutoff"],
    ),
    Mutation(
        name="backtest-lever-financed-for-free",
        why="drop the cash leg: borrowing costs nothing and holding cash earns nothing, which "
            "turns the capital market line into a plain multiplication",
        file="backtest.py",
        edits=[("    mixed = (k * v.pct_change() + (1.0 - k) * cash).fillna(0.0)",
                "    mixed = (k * v.pct_change()).fillna(0.0)")],
        kills=["test_levering_by_one_is_the_unlevered_path_and_by_zero_is_cash"],
    ),
    Mutation(
        name="backtest-lever-accepts-a-misaligned-financing-series",
        why="drop the index check: pandas then aligns the financing cost onto whatever dates it "
            "was given, charging the wrong days by an amount no other assertion could reveal",
        file="backtest.py",
        edits=[(
            "    if len(rf_annual) != len(v) or not (rf_annual.index == v.index).all():\n"
            '        raise ValueError("the financing rate must be quoted on exactly the value path\'s bars")',
            "    pass",
        )],
        kills=["test_levering_refuses_a_financing_rate_not_quoted_on_the_paths_own_bars"],
    ),
    # The momentum forecast. `momentum_mu` reads exactly TWO BARS of the training panel, and
    # which two is its entire specification -- so every mutant here moves one edge of that window
    # and none of them changes the output's shape, units, sign or plausibility. A 12/0 window and
    # a 12/1 window both return annualised returns for the right assets in the right order; the
    # difference is a documented several percent a year, in the direction that makes the signal
    # look absent. This is the `universe.py` shape once more: the failure is invisible in every
    # artifact and is caught only by a test that perturbs one bar and asserts the exact factor.
    Mutation(
        name="backtest-momentum-does-not-skip-the-recent-month",
        why="measure momentum right up to the cutoff: the most recent month reverses, so the "
            "basket buys assets that just gapped and earns the giveback -- the single most "
            "common way a momentum backtest is quietly built to fail, and it fails by enough to "
            "read as the signal not existing",
        file="backtest.py",
        edits=[("cutoff - pd.DateOffset(months=skip_months)", "cutoff")],
        kills=["test_momentum_stops_short_of_the_cutoff_by_the_skip"],
    ),
    Mutation(
        name="backtest-momentum-measures-the-whole-training-window",
        why="run the lookback back to the first bar of the training panel: momentum is then the "
            "five-year mean under a different name, so `--mu momentum` reproduces `--mu history` "
            "and the comparison the flag exists for is vacuous",
        file="backtest.py",
        edits=[("cutoff - pd.DateOffset(months=lookback_months),", "train.index.min(),")],
        kills=["test_momentum_ignores_everything_before_its_lookback_window"],
    ),
    Mutation(
        name="backtest-momentum-not-annualised",
        why="return the window's total return instead of an annualised rate: it lands in the same "
            "`predicted.ret` column as `frontier.estimate`'s output, ~8% low on a 10%/yr asset, "
            "and because it shifts every asset the same way no ranking assertion can see it",
        file="backtest.py",
        edits=[("    return growth ** (fetch.TRADING_DAYS_PER_YEAR / n) - 1.0",
                "    return growth - 1.0")],
        kills=["test_momentum_is_annualised_on_the_same_252_day_convention_as_everything_else"],
    ),
    Mutation(
        name="backtest-momentum-basket-holds-the-worst-scorers",
        why="`nsmallest` for the comparator basket: still a legal equal-weight portfolio of N "
            "assets with the right label, so it is the control that makes a momentum result "
            "attributable, silently inverted",
        file="backtest.py",
        edits=[(".nlargest(k).index", ".nsmallest(k).index")],
        kills=["test_the_momentum_basket_is_equal_weight_in_the_highest_scoring_assets"],
    ),
    Mutation(
        name="backtest-momentum-forecast-silently-ignored",
        why="the substitution falls through, so `--mu momentum` runs the historical mean while "
            "every table header, `method.mu_model` and the JSON all say momentum -- a whole "
            "experiment reported as its own control",
        file="backtest.py",
        edits=[("        est = replace(est, mu=mom)", "        est = replace(est)")],
        kills=["test_the_momentum_model_replaces_the_return_forecast_and_nothing_else"],
    ),
    Mutation(
        name="backtest-momentum-model-refits-the-covariance-too",
        why="re-estimate the covariance over the momentum window as well: the two runs then "
            "differ in TWO inputs, so neither result can be attributed to the return forecast "
            "-- and it looks like consistency rather than a confound",
        file="backtest.py",
        edits=[("        est = replace(est, mu=mom)",
                "        est = replace(est, mu=mom, cov=fr.estimate(train.tail(230)).cov)")],
        kills=["test_the_momentum_model_replaces_the_return_forecast_and_nothing_else"],
    ),
    Mutation(
        name="backtest-momentum-comparator-only-under-the-momentum-model",
        why="gate the unoptimised momentum basket on the model in use: the history run then has "
            "nothing to attribute its difference to, which is exactly the comparison the basket "
            "was added for, and every row it does print is still correct",
        file="backtest.py",
        edits=[("strategies(est_s, caps, rf_cut, benchmark, target_vol, mom_s, model.top_n)",
                "strategies(est_s, caps, rf_cut, benchmark, target_vol, "
                'mom_s if model.kind == "momentum" else None, model.top_n)')],
        kills=["test_the_momentum_comparator_and_diagnostic_are_present_under_both_forecasts"],
    ),
    Mutation(
        name="backtest-momentum-window-shorter-than-asked-for-passes-silently",
        why="drop the length check: a training window too short for the lookback returns a score "
            "measured over whatever bars happened to be there, labelled as a twelve-month one",
        file="backtest.py",
        edits=[("    if n < 20:", "    if False:")],
        kills=["test_a_training_window_too_short_for_the_momentum_lookback_is_refused"],
    ),
    Mutation(
        name="backtest-win-counts-compacted-and-misaligned",
        why="drop the None placeholders before zipping, so a strategy absent from one period has "
            "its period 4 compared against the benchmark's period 3 -- every win count in the "
            "matched-risk table is then against the wrong years, and in range",
        file="backtest.py",
        edits=[("            both = [(a, b) for a, b in zip(cols[k], cols[\"bm\"]) "
                "if a is not None and b is not None]",
                "            both = list(zip([a for a in cols[k] if a is not None],\n"
                "                            [b for b in cols[\"bm\"] if b is not None]))")],
        kills=["test_a_strategy_missing_from_some_periods_is_compared_against_the_right_years"],
    ),
    Mutation(
        name="backtest-a-partial-row-is-printed-as-if-it-were-complete",
        why="drop the `(N of M periods)` annotation, so a return compounded over 7 periods sits in "
            "the same column as one compounded over 10 with nothing to say they differ",
        file="backtest.py",
        edits=[("f\"{'' if len(rows_k) == len(periods) else f'   ({len(rows_k)} of "
                "{len(periods)} periods)'}\"", "\"\"")],
        kills=["test_a_strategy_missing_from_some_periods_is_compared_against_the_right_years"],
    ),
    Mutation(
        name="backtest-the-benchmark-is-left-in-the-candidate-set",
        why="drop the hold-out, so the optimiser may buy the very index fund it is being measured "
            "against -- the min-variance solve takes it on sight, and 'did mean-variance beat the "
            "index' is then asked of a portfolio allowed to BE the index",
        file="backtest.py",
        edits=[("    est_s = est if investable is None else restrict(est, investable)",
                "    est_s = est")],
        kills=["test_holding_out_the_benchmark_prices_it_without_letting_anything_buy_it"],
    ),
    Mutation(
        name="backtest-restrict-changes-the-shrinkage-it-inherited",
        why="scale the shrinkage while taking the submatrix, so the held-out run's covariance is "
            "not the one the full run estimated -- the two runs would then differ in something "
            "other than the hold-out and could not be compared",
        file="backtest.py",
        edits=[("shrinkage=est.shrinkage, n_obs=est.n_obs)",
                "shrinkage=est.shrinkage * 2.0, n_obs=est.n_obs)")],
        kills=["test_restrict_takes_the_submatrix_and_does_not_re_estimate_or_re_shrink"],
    ),
    Mutation(
        name="backtest-a-window-hold-runs-to-the-end-of-the-panel",
        why="take the hold as everything after the cutoff, so a row labelled a 1-year hold ran for "
            "however long the panel happens to be -- and lengthens every week the data refreshes",
        file="backtest.py",
        edits=[("        test = panel.loc[cutoff:test_to]\n        row = evaluate(train, test, irx, caps, benchmark, rf_override, top_n, model, investable, costs)\n        out.append({\"hold_years\": h,",
                "        test = panel.loc[cutoff:]\n        row = evaluate(train, test, irx, caps, benchmark, rf_override, top_n, model, investable, costs)\n        out.append({\"hold_years\": h,")],
        kills=["test_a_window_hold_ends_at_the_requested_length_not_at_the_panel_end"],
    ),
    Mutation(
        name="backtest-a-window-hold-past-the-panel-is-silently-shortened",
        why="drop the refusal, so a 5-year hold from 3 years back is reported as a 5-year hold and "
            "is three -- the annualised return is then computed over the wrong denominator",
        file="backtest.py",
        edits=[("        if wanted > end:", "        if False:")],
        kills=["test_a_window_hold_that_runs_past_the_panel_is_refused_rather_than_truncated"],
    ),
    Mutation(
        name="backtest-the-window-training-window-grows-with-the-panel",
        why="train on everything before the cutoff instead of a fixed length, so the nested rows "
            "stop being one stated training window and quietly lengthen every week",
        file="backtest.py",
        edits=[("    train = panel.loc[train_from:cutoff]\n\n    out = []\n    for h in sorted(hold_years):",
                "    train = panel.loc[:cutoff]\n\n    out = []\n    for h in sorted(hold_years):")],
        kills=["test_a_window_hold_ends_at_the_requested_length_not_at_the_panel_end"],
    ),
    Mutation(
        name="backtest-averaged-arithmetically-instead-of-compounded",
        why="report the arithmetic mean of the per-period returns: it exceeds the compounded rate "
            "by about half the variance, so it pays a bonus for volatility -- in the tables whose "
            "entire purpose is comparing strategies at different volatilities",
        file="backtest.py",
        edits=[("    return float(np.prod([1.0 + r for r in rets]) ** (1.0 / len(rets)) - 1.0)",
                "    return float(np.mean(rets))")],
        kills=["test_the_reported_average_compounds_and_is_not_the_arithmetic_mean"],
    ),

    # ----------------------------------------------------------- delisted.py + join_extra
    #
    # The `backtest.py` shape once more, and for one reason more: `_stage` copies `pipeline/`
    # with `ignore_patterns("data", ...)`, so `pipeline/data/delisted/` is NOT in a mutant tree.
    # The two tests in `test_delisted_invariants.py` that read the cache -- the served-span check
    # and the provenance check -- therefore SKIP inside every mutant, and a mutant that expected
    # either of them would be reported MISKILLED for a reason that has nothing to do with the
    # code. Every kill named below is one of that file's synthetic tests, which is also why they
    # are synthetic: all twelve cached series have a bad final bar and a clean penultimate one,
    # so nothing on disk distinguishes "drops the last bar" from "keeps it".
    #
    # What makes this group worth having is that none of these mutations produces a number that
    # looks wrong. A kept final bar, an unnormalised splice, an outer join or a basket that drops
    # a wiped-out name instead of zeroing it all return a complete result of the right shape --
    # and three of the four move the answer in the direction that flatters the method.
    Mutation(
        name="delisted-keeps-the-last-traded-bar",
        why="trust the final print of a delisted series: it is an index-deletion auction or a "
            "stub, and where an independent check exists it fails -- CMA held 1.89542 x FITB to "
            "+-0.10% for eleven bars and then printed its last 5.34% below that ratio",
        file="delisted.py",
        edits=[("    kept, dropped = df.index[-2], df.index[-1]",
                "    kept, dropped = df.index[-1], df.index[-1]")],
        kills=["test_the_last_traded_bar_is_dropped_and_the_one_before_it_is_the_terminal_level"],
    ),
    Mutation(
        name="delisted-a-stub-terminal-bar-is-used-anyway",
        why="drop one bar even when the bar behind it is also a stub, so the terminal level is a "
            "price nobody traded at -- three of the twelve print volume 0 or 1 on their last bar, "
            "so a series with two of them is not hypothetical",
        file="delisted.py",
        edits=[('    if not df["volume"].loc[kept] > 0:', "    if False:")],
        kills=["test_a_stub_penultimate_bar_is_refused_rather_than_used"],
    ),
    Mutation(
        name="delisted-splice-is-not-level-continuous",
        why="continue at the continuation asset's own LEVEL instead of rescaling to the terminal "
            "one: the series stays monotone and NaN-free, and one bar carries a return of a "
            "hundred-fold because the two vendors' price scales are unrelated",
        file="delisted.py",
        edits=[("            tail = lvl * after / float(calendar.asof(end))",
                "            tail = lvl * after")],
        kills=["test_the_continuation_never_overwrites_a_real_bar_and_is_continuous_at_the_splice"],
    ),
    Mutation(
        name="delisted-keeps-a-vendor-bar-the-calendar-does-not-have",
        why="take the second vendor's trading days as given: a bar on a day the store has no "
            "trading on is a return the portfolio could not have had, and no vendor's holiday "
            "handling has been checked here -- which is what `fetch.load_panel`'s loudest rule is "
            "about, one vendor over",
        file="delisted.py",
        edits=[("        real = real.reindex(calendar.index.intersection(real.index))\n", "")],
        kills=["test_a_vendor_holiday_the_calendar_does_not_have_is_dropped_rather_than_carried"],
    ),
    Mutation(
        name="delisted-range-is-requested-but-never-checked",
        why="treat the range string as a parameter rather than a request: an unrecognised range "
            "returns 252 rows with a 200, which is indistinguishable from a young company -- the "
            "training window would silently start after the cutoff",
        file="delisted.py",
        edits=[("(needs is not None and df.index.min() > pd.Timestamp(needs))",
                "(False and df.index.min() > pd.Timestamp(needs))")],
        kills=["test_a_range_the_vendor_does_not_recognise_reads_as_a_short_history_and_raises"],
    ),
    Mutation(
        name="delisted-join-extra-does-not-reindex-onto-the-panels-days",
        why="join on the extra column's own index, so restoring twelve names moves the window and "
            "the restored run stops being comparable with the baseline bar for bar -- the property "
            "the whole comparison rests on. TWO edits, because the panel's days are pinned twice "
            "(at `s.reindex` and again at `pd.DataFrame(..., index=panel.index)`) and removing "
            "either one alone is neutral: the first version of this mutant dropped only the "
            "reindex and SURVIVED the window test, which is the harness reporting that the second "
            "pin was doing the work",
        file="backtest.py",
        edits=[("        on_panel = s.reindex(panel.index)", "        on_panel = s"),
               ("pd.concat([panel, pd.DataFrame(kept, index=panel.index)], axis=1)",
                "pd.concat([panel, pd.DataFrame(kept)], axis=1)")],
        kills=["test_joining_extra_columns_cannot_move_the_panels_window"],
    ),
    Mutation(
        name="delisted-join-extra-tolerates-a-clashing-column",
        why="let an extra column shadow a stored one: the panel is still the right shape and the "
            "right length, and one asset now comes from a vendor with no event log behind it",
        file="backtest.py",
        edits=[("    if clash:", "    if False:")],
        kills=["test_an_extra_column_that_shadows_a_stored_one_is_refused"],
    ),
    Mutation(
        name="delisted-join-extra-skips-the-panels-start-filter",
        why="admit an extra column whatever its history, so an asset is in the candidate set "
            "because of which FILE it came from -- `load_panel` drops the store's own late "
            "arrivals with a reason",
        file="backtest.py",
        edits=[("        if s.index.min() > pd.Timestamp(start):", "        if False:")],
        kills=["test_an_extra_column_is_held_to_the_panels_own_start_and_coverage_filters"],
    ),
    Mutation(
        name="delisted-join-extra-deletes-bars-to-fit-a-hole",
        why="let `dropna` shorten the panel: two bars missing from one restored column would "
            "delete those days from all 478 stored assets, which `min_coverage` tolerating a "
            "small gap makes reachable by design",
        file="backtest.py",
        edits=[("    if len(joined) != before:", "    if False:")],
        kills=["test_a_hole_inside_a_tolerated_column_refuses_the_join_rather_than_deleting_a_bar"],
    ),
    Mutation(
        name="delisted-stress-span-from-a-key-the-period-lacks",
        why="read the hold's start from `test['start']`, which a test block does not carry: "
            "`pd.Timestamp(None)` is NaT and propagates into every bound as NaN, so the failure "
            "mode is a complete file of nulls rather than an error",
        file="delisted.py",
        edits=[('    a = pd.Timestamp(periods[0]["cutoff"])',
                '    a = pd.Timestamp(periods[0]["test"].get("start"))')],
        kills=["test_the_bounds_span_comes_from_the_cutoff_and_not_a_key_the_period_lacks"],
    ),
    Mutation(
        name="delisted-stress-a-wiped-out-name-leaves-the-basket-instead-of-zeroing",
        why="divide the wiped-out case by the names that WERE priced: that is the assumption the "
            "bound exists to remove -- a missing constituent treated as never having existed, "
            "which returns the measured number and calls it a lower bound",
        file="delisted.py",
        edits=[('        "all_wiped_out": annual((n_held * ew_growth) / (n_held + n_extra)),',
                '        "all_wiped_out": annual(ew_growth),')],
        kills=["test_the_equal_weight_bound_brackets_the_measured_value_from_both_sides"],
    ),

    # --------------------------------------------------------------- pit.py: the membership walk
    #
    # The `universe.py` shape and the `backtest.py` shape at once, which is why every kill below is
    # in `test_pit_invariants.py` and none is an artifact test. Half of these mutations remove a
    # VALIDATION or a STRIP, and `sp500_pit.toml` is valid -- so the rebuild is byte-identical and no
    # file on disk changes. The other half are lookaheads and window arithmetic, and the shipped
    # membership's eleven snapshots share ~95% of their names, so a run over it cannot tell "each
    # window used its own list" from "every window used the 2016 list" wherever they agree.
    #
    # `pipeline/results/backtest_pit*.json` cannot catch any of them either: `_stage` copies those
    # files rather than regenerating them (see WHAT THIS HARNESS CANNOT REACH AT ALL), so a mutant
    # aimed at them would be reported SURVIVED for a reason unrelated to the code. And
    # `test_store_invariants.py`'s point-in-time coverage test SKIPS in a mutant tree, because it
    # reads the price store and `data/` is ignored -- so it is named nowhere below.
    #
    # What makes the group worth having is that the two strip mutations are survivorship errors in
    # OPPOSITE directions and both produce a complete, plausible result: not stripping a recycled
    # ticker puts a micro-cap in the candidate set labelled as a 2016 constituent, and stripping a
    # reissued one deletes Dow Inc from seven snapshots of an index it was actually in.
    #
    # Two of these report heavy collateral and the reason is worth knowing rather than skimming:
    # `pit-the-training-window-runs-past-its-own-cutoff` also trips `evaluate`'s own seam check
    # (`test starts at X, not at the cutoff`), and `pit-training-window-is-a-year-longer-than-it-says`
    # trips `run`'s start assertion on the synthetic store. In both cases every OTHER test that
    # calls `pit.run` fails too, because `run` raises before returning. That is two guards agreeing,
    # not a test overlapping itself -- but it does mean neither mutant is evidence that the named
    # test alone would catch a subtler version of the same slip.
    Mutation(
        name="pit-a-recycled-ticker-is-not-stripped",
        why="leave a ticker whose current occupant is a different company in every snapshot: it "
            "fetches a clean full-length filter-passing series and the optimiser holds it as the "
            "constituent it is not -- the PARA case, generalised",
        file="pit.py",
        edits=[('            if s in recycled:\n                gone[s] = "recycled"',
                '            if False:\n                gone[s] = "recycled"')],
        kills=["test_a_recycled_ticker_leaves_every_snapshot_and_a_reissued_one_only_the_early_ones",
               "test_every_removal_is_reported_per_snapshot_with_the_table_that_made_it"],
    ),
    Mutation(
        name="pit-a-reissued-ticker-is-stripped-from-every-snapshot",
        why="apply the recycled remedy to a reissued ticker: Dow Inc, Fox Corporation and both Fox "
            "classes vanish from every snapshot of an index that held them, which is survivorship "
            "bias in the other direction and equally invisible -- the candidate set just gets "
            "smaller",
        file="pit.py",
        edits=[("            elif s in cuts and pd.Timestamp(d) < cuts[s]:",
                "            elif s in cuts:")],
        kills=["test_a_recycled_ticker_leaves_every_snapshot_and_a_reissued_one_only_the_early_ones",
               "test_every_removal_is_reported_per_snapshot_with_the_table_that_made_it"],
    ),
    Mutation(
        name="pit-a-reissued-ticker-is-stripped-on-the-wrong-side-of-its-date",
        why="strip the snapshots AFTER the reissue date instead of before it: exactly inverted, so "
            "the legitimate member is deleted and the other company is kept, and the count of names "
            "removed is about the same either way",
        file="pit.py",
        edits=[("pd.Timestamp(d) < cuts[s]", "pd.Timestamp(d) > cuts[s]")],
        kills=["test_a_recycled_ticker_leaves_every_snapshot_and_a_reissued_one_only_the_early_ones"],
    ),
    Mutation(
        name="pit-removals-are-applied-but-never-reported",
        why="strip correctly and discard the report: a stripped name is a constituent nobody prices "
            "and it never reaches `load_panel`, so it appears in no `dropped` reason -- the stress "
            "bound would then leave out precisely the names it is about, and be too narrow with "
            "every number in it plausible",
        file="pit.py",
        edits=[("        snaps[d], removed[d] = keep, gone",
                "        snaps[d], removed[d] = keep, {}")],
        kills=["test_every_removal_is_reported_per_snapshot_with_the_table_that_made_it"],
    ),
    Mutation(
        name="pit-a-reissue-declaration-needs-no-reason",
        why="accept `[reissued].X = { from = ... }` with no `why`: a declaration that strips real "
            "snapshots with no measurement recorded behind it, which is the thing every table in "
            "that membership file exists to prevent",
        file="pit.py",
        edits=[('        if not isinstance(spec, dict) or "from" not in spec or "why" not in spec:',
                '        if not isinstance(spec, dict) or "from" not in spec:')],
        kills=["test_a_reissue_declaration_without_a_date_or_a_reason_raises"],
    ),
    Mutation(
        name="pit-a-symbol-may-be-both-recycled-and-reissued",
        why="allow the two contradictory remedies on one ticker: which one wins is then decided by "
            "the order of the branches in the strip loop, silently, and differently if that loop is "
            "ever rewritten",
        file="pit.py",
        edits=[('        if sym in recycled:\n            raise ValueError(f"{path.name}: {sym} is '
                'in both [recycled] and [reissued]")',
                "        if False:\n            pass")],
        kills=["test_a_symbol_declared_both_recycled_and_reissued_raises"],
    ),
    Mutation(
        name="pit-held-between-reports-the-inner-dates",
        why="take the LAST snapshot holding a symbol as its first: the membership interval collapses "
            "to a point, and the overlap screen below then rejects nearly every real series -- a "
            "quieter failure than it sounds, because a rejected series is a name silently missing "
            "from the candidate set",
        file="pit.py",
        edits=[("            first = out.get(s, (d, d))[0]", "            first = d")],
        kills=["test_held_between_reports_the_first_and_last_snapshot_that_holds_a_symbol"],
    ),
    Mutation(
        name="pit-overlap-screen-is-one-sided",
        why="check only that a series does not start after the index last held the ticker, not that "
            "it ends after the index first held it: a predecessor's series that stops before the "
            "membership begins then passes, and the two directions catch different real cases (CA "
            "and DNB are the first, a truncated predecessor the second)",
        file="pit.py",
        edits=[('    if last_bar < lo:\n        return (f"served {first_bar.date()} -> '
                '{last_bar.date()}, entirely before the index first "\n'
                '                f"held it ({lo.date()})")',
                "    if False:\n        pass")],
        kills=["test_a_series_is_rejected_exactly_when_it_cannot_be_the_constituents_history"],
    ),
    Mutation(
        name="pit-overlap-screen-admits-a-symbol-no-snapshot-holds",
        why="return None for an unknown symbol: `held` is built from the snapshots AFTER the strips, "
            "so this lets the screen undo the very removals it exists to catch -- a recycled ticker "
            "would come back in through the second vendor's panel",
        file="pit.py",
        edits=[('    if sym not in held:\n        return "no snapshot holds this symbol"',
                "    if sym not in held:\n        return None")],
        kills=["test_a_symbol_no_snapshot_holds_is_rejected_rather_than_admitted"],
    ),
    Mutation(
        name="pit-screen-skips-a-symbol-with-no-cached-file",
        why="`continue` without recording the rejection: the count of names the second vendor "
            "delivered becomes the count of files that happen to exist, and a symbol nobody fetched "
            "is indistinguishable from one nobody prices",
        file="pit.py",
        edits=[('            rejected[sym] = f"{dl.SOURCE} serves nothing usable for it"\n'
                "            continue", "            continue")],
        kills=["test_the_screen_rejects_a_cached_series_that_cannot_be_the_constituent"],
    ),
    Mutation(
        name="pit-training-window-is-a-year-longer-than-it-says",
        why="open the training window a year early: every estimate is made from more data than the "
            "run claims, and the only visible sign is a bar count nobody reads",
        file="pit.py",
        edits=[("    return c - pd.DateOffset(years=train_years), c + pd.DateOffset(years=hold_years)",
                "    return c - pd.DateOffset(years=train_years + 1), "
                "c + pd.DateOffset(years=hold_years)")],
        kills=["test_the_training_window_is_fixed_length_and_measured_back_from_the_cutoff"],
    ),
    Mutation(
        name="pit-a-window-uses-the-first-cutoffs-membership",
        why="load one membership list for every window: THE POINT OF THE WHOLE FILE removed, and it "
            "shows up nowhere -- the shipped snapshots share ~95% of their names, so the run "
            "completes with the right shape and measures the 2016 index's survivors forward, which "
            "is the survivorship bias this walk exists to eliminate",
        file="pit.py",
        edits=[("        members = list(snaps[cutoff])", "        members = list(snaps[cutoffs[0]])")],
        kills=["test_each_window_chooses_from_the_membership_of_its_own_cutoff"],
    ),
    Mutation(
        name="pit-the-benchmark-is-in-its-own-candidate-set",
        why="let the optimiser hold SPY: an index fund among its own constituents has lower variance "
            "than almost any one of them, so the minimum-variance solve buys it on sight and 'did "
            "mean-variance beat the index' becomes a question about a portfolio allowed to BE the "
            "index. `test_a_held_out_benchmark_is_priced_and_not_investable` reads the document's "
            "`benchmark_investable` flag, which this edit does not touch, so the behavioural test "
            "in `test_pit_invariants.py` is the only thing that can see it",
        file="pit.py",
        edits=[("        investable = [c for c in panel.columns if c != benchmark]",
                "        investable = list(panel.columns)")],
        kills=["test_the_benchmark_is_priced_but_never_investable_even_when_a_snapshot_lists_it"],
    ),
    Mutation(
        name="pit-the-training-window-runs-past-its-own-cutoff",
        why="estimate over the whole panel instead of stopping at the rebalance bar: the lookahead, "
            "in the one place `test_backtest_invariants.py` cannot reach it -- `run` does its own "
            "date arithmetic and `evaluate` is handed whatever it slices",
        file="pit.py",
        edits=[("        train, test = panel.loc[from_bar:cut_bar], panel.loc[cut_bar:to_bar]",
                "        train, test = panel.loc[from_bar:to_bar], panel.loc[cut_bar:to_bar]")],
        kills=["test_the_weights_of_every_window_ignore_the_prices_after_its_own_cutoff"],
    ),
    Mutation(
        name="pit-a-short-hold-is-reported-at-its-requested-length",
        why="drop the end assertion: a one-year hold whose year is not in the panel yet is measured "
            "over whatever bars exist and labelled `1.0y`, which is the nested-window trap in "
            "miniature -- a number that reads as an answer to the question asked",
        file="pit.py",
        edits=[("        if to_bar < test_to - pd.Timedelta(days=SLACK_DAYS):", "        if False:")],
        kills=["test_a_hold_that_would_run_past_the_last_bar_raises_rather_than_being_reported_short"],
    ),
    Mutation(
        name="pit-a-short-training-window-is-reported-at-its-requested-length",
        why="drop the start assertion: `load_panel` drops a symbol whose own history starts late, so "
            "the survivors all predate the window -- and the panel index is their INTERSECTION, so "
            "a calendar mismatch shortens it with nothing dropped and nothing said. A 1-year "
            "training window reported as 5",
        file="pit.py",
        edits=[("        if from_bar > train_from + pd.Timedelta(days=SLACK_DAYS):",
                "        if False:")],
        kills=["test_a_training_window_the_panel_starts_inside_raises_rather_than_being_reported_at_length"],
    ),
    Mutation(
        name="pit-unpriced-constituents-omit-the-stripped-names",
        why="count only the names the store had no prices for: the ones the membership stripped were "
            "never asked for, so they are in no `dropped` reason -- the stress bound then covers 87 "
            "names instead of 99 and is too narrow by exactly the set of tickers that now mean "
            "someone else",
        file="pit.py",
        edits=[("        unpriced = sorted((absent - restored_here) | set(gone))",
                "        unpriced = sorted(absent - restored_here)")],
        kills=["test_a_constituent_whose_ticker_now_means_another_company_is_counted_among_the_unpriced"],
    ),
    Mutation(
        name="pit-unpriced-constituents-keep-the-restored-names",
        why="ignore what the second vendor supplied: a restored run is then bounded as if the "
            "restoration had never happened, which widens the bound in the direction that flatters "
            "the method",
        file="pit.py",
        edits=[("        unpriced = sorted((absent - restored_here) | set(gone))",
                "        unpriced = sorted(absent | set(gone))")],
        kills=["test_a_restored_name_is_no_longer_an_unpriced_constituent"],
    ),

    # ---------------------------------------------------------------------------- the cost model
    #
    # `Costs` is the newest thing in `backtest.py` and the only one whose failure mode is a NUMBER
    # THAT IS TOO GOOD rather than a crash. A cost model that undercharges is a subsidy to whichever
    # side trades more, which is never the index -- so these mutants are all in the direction the
    # result would like, and none of them can be caught by a committed artifact: the shipped files
    # were generated at one `--trade-bps`, and the load-bearing claims are about the RELATION
    # between two settings (zero reproduces the pre-cost run; the convention is buys plus sells).
    Mutation(
        name="costs-the-opening-purchase-is-free",
        why="start the turnover path at zero: the portfolio appears already held, so a one-year hold "
            "pays nothing at all and the annually-reoptimised strategy's entire bill vanishes -- "
            "which is the one cost every strategy in the file certainly pays",
        file="backtest.py",
        edits=[("    t[0] = float(np.abs(w).sum())", "    t[0] = 0.0")],
        kills=["test_turnover_is_buys_plus_sells_and_bar_zero_is_the_purchase"],
    ),
    Mutation(
        name="costs-turnover-is-the-one-way-half",
        why="charge |dw| once instead of on both legs: every cost in every table is then exactly "
            "half of what it should be, which is the size of the whole effect and is invisible in "
            "any single number's plausibility",
        file="backtest.py",
        edits=[("                t[i - 1] += float(np.abs(w - drifted).sum())",
                "                t[i - 1] += 0.5 * float(np.abs(w - drifted).sum())")],
        kills=["test_the_cost_haircut_is_multiplicative_so_net_weights_equal_gross_weights"],
    ),
    Mutation(
        name="costs-the-rebalance-is-charged-on-the-bar-after-it-happened",
        why="record the month-boundary trade on bar i instead of i-1, where `value_path` prices the "
            "new segment: the bill is right and the day is wrong, so the two functions disagree "
            "about what the portfolio looked like when it traded",
        file="backtest.py",
        edits=[("                t[i - 1] += float(np.abs(w - drifted).sum())",
                "                t[i] += float(np.abs(w - drifted).sum())")],
        kills=["test_turnover_is_buys_plus_sells_and_bar_zero_is_the_purchase"],
    ),
    Mutation(
        name="costs-a-zero-cost-run-emits-net-columns-anyway",
        why="drop the `zero()` gate on the strategy rows: every result published before costs "
            "existed grows a net column of the same numbers, and a reader cannot tell a run with no "
            "frictions from a run whose frictions were measured at zero",
        file="backtest.py",
        edits=[('        if not costs.zero():\n            row["net_hold"] = net(v_hold, test_s, w, "hold")',
                '        if True:\n            row["net_hold"] = net(v_hold, test_s, w, "hold")')],
        kills=["test_a_zero_cost_run_is_bit_identical_to_one_made_before_costs_existed"],
    ),
    Mutation(
        name="costs-the-annual-fee-is-charged-once-rather-than-accrued",
        why="a flat daily factor instead of one compounded over the bars held: the fee then applies "
            "in full on the first day and never again, so a 0.0945% expense ratio over three years "
            "costs a third of what it should",
        file="backtest.py",
        edits=[("    fee = daily ** np.arange(len(turnover), dtype=float)",
                "    fee = np.full(len(turnover), daily)")],
        kills=["test_the_cost_haircut_is_multiplicative_so_net_weights_equal_gross_weights"],
    ),
    Mutation(
        name="costs-the-fund-fee-is-never-charged",
        why="zero the weighted expense ratio inside `net`: the only side that pays a management fee "
            "here is the benchmark, so this is precisely the asymmetry the parameter was added to "
            "remove, restored while every net column still reads as a net column",
        file="backtest.py",
        edits=[("        er = scale * weighted_expense(prices.columns, w, costs)", "        er = 0.0")],
        kills=["test_costs_charge_the_optimiser_and_the_benchmark_or_neither"],
    ),
    Mutation(
        name="costs-the-daily-mix-reset-is-free",
        why="`lever`'s docstring says the reset \"is free here, and it is free nowhere\"; this makes "
            "that true again. The levered rows are the ones sized to the index's risk, so a free "
            "reset flatters exactly the comparison that is supposed to settle the question",
        file="backtest.py",
        edits=[("    return 2.0 * drift", "    return 0.0 * drift")],
        kills=["test_levering_charges_the_daily_reset_it_documents_as_free"],
    ),
    Mutation(
        name="costs-the-mix-reset-is-charged-one-way",
        why="the same halving as `costs-turnover-is-the-one-way-half`, in the other function: the "
            "levered rows would then be charged on a different convention from the unlevered ones, "
            "which is worse than either convention consistently applied",
        file="backtest.py",
        edits=[("    return 2.0 * drift", "    return drift")],
        kills=["test_levering_charges_the_daily_reset_it_documents_as_free"],
    ),
    Mutation(
        name="costs-a-negative-trading-cost-is-accepted",
        why="`--trade-bps -5` is a typo that pays the investor to trade, so every net column comes "
            "out better than its gross one and the table reads as an argument for rebalancing more",
        file="backtest.py",
        edits=[("        if self.trade_bps < 0:", "        if False:")],
        kills=["test_a_negative_trading_cost_or_an_impossible_fee_is_refused"],
    ),
    Mutation(
        name="costs-an-expense-ratio-of-one-is-accepted",
        why="close the interval at 1.0: a fee of 100% a year is a position wiped out inside the "
            "window, reported as a measurement, and `(1 - 1.0) ** (1/252)` is 0.0 rather than an "
            "error",
        file="backtest.py",
        edits=[("        bad = {s: e for s, e in self.expense.items() if not 0.0 <= e < 1.0}",
                "        bad = {s: e for s, e in self.expense.items() if not 0.0 <= e <= 1.0}")],
        kills=["test_a_negative_trading_cost_or_an_impossible_fee_is_refused"],
    ),

    # ------------------------------------------------------- the second vendor's range, and the join
    #
    # `fetch_longest` cannot be mutation-tested against the cache: `_stage` ignores `data/`, and the
    # cached parquets are the OUTPUT of a fetch that already chose a range. Every mutant below is
    # caught by a test that monkeypatches the vendor, which is the only way to state "the longest
    # span wins" as a property rather than as a fact about what was served on one afternoon.
    Mutation(
        name="delisted-the-first-range-tried-wins",
        why="keep the first response instead of the longest: `MAX` fell back to one year for AVB on "
            "the afternoon this was written while `10Y` served 2,513 bars, so first-wins is a panel "
            "that starts in 2025 for a window that opens in 2011",
        file="delisted.py",
        edits=[("        if best is None or _span_years(df) > _span_years(best[0]):",
                "        if best is None:")],
        kills=["test_the_longest_range_wins_even_when_the_range_that_serves_it_comes_last"],
    ),
    Mutation(
        name="delisted-the-span-floor-is-checked-per-range-instead-of-once",
        why="check every range against the floor as it arrives: a fallback is the EXPECTED outcome "
            "for most symbols, so this turns the normal case into a hard failure and the symbols "
            "the vendor does serve well go unmeasured",
        file="delisted.py",
        edits=[("        df = _parse(symbol, _get(symbol, rng=rng))",
                "        df = fetch_series(symbol, rng=rng, min_span_years=min_span_years)")],
        kills=["test_the_longest_range_wins_even_when_the_range_that_serves_it_comes_last"],
    ),
    Mutation(
        name="delisted-the-winning-range-is-never-checked",
        why="return the longest response without checking it against the floor: when every range "
            "falls back to one year the longest of them is still one year, and it arrives with a "
            "200 and plausible prices",
        file="delisted.py",
        edits=[('    return _check_served(symbol, best[0], "+".join(ranges), None, min_span_years), best[1]',
                "    return best[0], best[1]")],
        kills=["test_a_one_year_fallback_on_every_range_is_refused_rather_than_returned"],
    ),
    Mutation(
        name="delisted-a-fetch-that-checks-nothing-is-allowed",
        why="drop the exactly-one rule: a call with neither `needs` nor `min_span_years` then "
            "returns whatever the vendor sent, which is the one call shape this guard exists to "
            "make impossible",
        file="delisted.py",
        edits=[("    if (needs is None) == (min_span_years is None):", "    if False:")],
        kills=["test_a_call_that_checks_neither_the_span_nor_the_date_is_refused"],
    ),
    Mutation(
        name="delisted-restoring-nothing-is-never-an-error",
        why="allow an empty join unconditionally: `--extra-panel` on a single-window run then "
            "produces a file that names the restoration in its provenance and is bit-identical to "
            "the baseline",
        file="backtest.py",
        edits=[("    if not kept and not allow_empty:", "    if False:")],
        kills=["test_restoring_nothing_raises_unless_the_caller_joins_once_per_window"],
    ),
    Mutation(
        name="delisted-a-per-window-join-refuses-its-empty-window",
        why="ignore `allow_empty`: the error is correct for the one-shot caller and fatal for the "
            "point-in-time one, whose earliest window legitimately has no restorable name in it -- "
            "so the honest outcome becomes the unrunnable one",
        file="backtest.py",
        edits=[("    if not kept and not allow_empty:", "    if not kept:")],
        kills=["test_restoring_nothing_raises_unless_the_caller_joins_once_per_window"],
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


def _stale(mut: Mutation, old: str, n: int) -> str:
    return (f"MUTATION HARNESS IS STALE: {mut.name!r} expected exactly one occurrence of\n\n"
            f"{old}\n\nin {mut.file}, found {n}. A mutation that no longer applies would be "
            "reported as 'caught' while testing nothing -- fix the harness before trusting it.")


def _preflight(muts: list[Mutation]) -> None:
    """Every needle, checked against the real source before a single mutant runs.

    `_apply` already refuses a stale needle, and that is the load-bearing check -- but it fires
    one mutant at a time, ~26 s apart, so a needle that went stale in the ninetieth record costs
    forty minutes to discover and the run dies with the other ninety-nine unreported. This says
    all of them at once.

    It is not a substitute for the check in `_apply`: this one reads `pipeline/`, that one reads
    the staged copy, and it is the staged copy that gets mutated. Both, or the guarantee moves
    from "the edit landed" to "an edit like it would have landed somewhere".
    """
    text = {f: (PIPELINE / f).read_text() for f in {m.file for m in muts}}
    stale = [_stale(m, old, text[m.file].count(old))
             for m in muts for old, _ in m.edits if text[m.file].count(old) != 1]
    if stale:
        raise SystemExit("\n" + f"\n\n{'-' * 78}\n\n".join(stale) +
                         f"\n\n{len(stale)} stale needle(s) of "
                         f"{sum(len(m.edits) for m in muts)}; nothing was run.")


def _apply(tree: Path, mut: Mutation) -> None:
    path = tree / "pipeline" / mut.file
    text = path.read_text()
    for old, new in mut.edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit("\n" + _stale(mut, old, n))
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
    ap.add_argument("--check", action="store_true",
                    help="verify every needle still matches its source exactly once, and exit "
                         "without running anything -- seconds instead of forty minutes")
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

    _preflight(muts)
    if args.check:
        print(f"{len(muts)} mutants, {sum(len(m.edits) for m in muts)} needles, all matching "
              "exactly once", flush=True)
        return 0

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
