# CLAUDE.md

Guidance for Claude Code working in `~/dev/phd/markowitz`. See `README.md` for what the project
is and how to run it; this file is the part that is not obvious from the code.

## Shape of the repo

Two subprojects and one artifact between them. **This repo does not follow the workspace's
`development/` split** — the two halves are `pipeline/` and `web/`, and commands run from the
repo root (Python) or from `web/` (npm).

```
pipeline/                 Python 3.11, pyenv virtualenv "markowitz" (.python-version)
  universes/*.toml        one file per universe: symbols, groups, benchmark, exclusions+evidence
  universe.py             the LOADER for those files -- validates, does not contain a universe
  store.py                the price store: year-partitioned closes + an event log
  fetch.py                Yahoo chart endpoint -> store -> common-window total-return panel
  frontier.py             mu, Sigma, the QP, the frontier, weight cleaning
  build.py                the one entry point: fetch -> estimate -> solve -> write JSON
  backtest.py             the walk-forward test. OFF TO THE SIDE: reads the store, writes
                          pipeline/results/, ships nothing to the browser
  pit.py                  the POINT-IN-TIME walk: one membership snapshot per rebalance, its own
                          panel and candidate set per window. Calls backtest.evaluate unchanged
  delisted.py             the SECOND vendor, for tickers the first one 404s. Off to the side of
                          the side: never enters the store, reaches backtest.py via --extra-panel
  results/*.json          committed BY HAND, never rebuilt by CI. Three kinds, `kind` says which
  tests/                  invariants + _mutate.py, the mutation harness (--check is the fast path)
  data/prices/            the store. gitignored today, but see .gitignore -- that is now a choice
  data/delisted/          raw/*.csv COMMITTED (unrefetchable dead tickers), panel_*.parquet not
  data/delisted_pit/      the same, for sp500_pit.toml's 23 second-vendor names -- 25 raw/*.csv,
                          because CA and DNB were fetched and then REJECTED by the overlap screen
                          and the evidence for rejecting them is the series itself
web/                      Vite + React 19 + TS, no chart library
  public/data/*.json      COMMITTED. This is the deployment artifact.
  src/portfolio.ts        the browser's own maths -- interpolation, tangency, growth
  src/export.ts           the portfolio download: CSV + JSON, serialised in the browser
  src/config.ts           the view state: URL fragment, localStorage, a file on disk
  src/*.test.ts           vitest, reading public/data -- the SHIPPED artifacts, not fixtures
  tests/_mutate.mjs       the web mutation harness; stages web/ into web/.mutate/
  tools/shoot.mjs         headless-Chrome screenshotter, for the render-and-look step
```

## Constraints that are decisions, not accidents

- **No AWS. No server of any kind.** Not even serverless. The pipeline commits JSON and Pages
  serves it. If a task seems to need a backend, the answer is almost certainly to precompute more
  and ship it in the JSON.
- **The GitHub side went live on 2026-09-04.** `git@github.com:benlvovsky/markowitz.git`, public,
  default branch `main`, serving at https://benlvovsky.github.io/markowitz/. Both workflows are
  active. Two of the three switches that make it work are **repository settings, not files** —
  Pages source = "GitHub Actions", and Actions workflow permissions = "Read and write" (without
  the second, `update-data`'s final `git push` 403s no matter what `permissions:` says in the
  workflow). If CI breaks in a way the YAML cannot explain, check those two first.
- **`rf` is browser-side, never a pipeline parameter.** The frontier does not depend on it. Any
  change that makes the pipeline emit per-rf files is a regression in the design, not a feature.
  The one place the two halves do meet is its *precision*: `build.fetch_rf` rounds the T-bill
  yield to 4 dp, `config.RF_STEP` is 1 basis point, and `pct(rf, 2)` prints two decimals of
  percent. All three have to agree, or the seeded default is a slider position the reader cannot
  return to and the URL then carries an `rf=` forever. `config.test.ts` asserts the shipped
  `rf_default` is on that grid; nothing else fails, because both values are individually fine.
  **`backtest.py` is the one exception and is allowed to be**: a backtest is a claim about the
  past, and the investor standing at the cutoff faced that day's T-bill yield rather than the one
  today's reader drags to. There `rf` is pinned to `^IRX` as of the cutoff, which is what makes
  the tangency portfolio a measurement instead of a view setting. It emits no per-rf file and
  nothing it computes reaches the browser, so the rule above is intact.
- **`backtest.py` has two modes and the rolling one is the answer; the fixed-cutoff one is kept
  because it is the trap.** `--years 1,2,3` trains on everything before each cutoff and holds to
  the last bar, so the 3-year window *contains* the 2-year window, which contains the 1-year one:
  one observation with three end dates, reported as three. It says the tangency portfolio beat the
  benchmark at all three, and that is a fact about which end dates this panel has. `run_rolling`
  walks a **fixed-length** training window forward in **non-overlapping** holds — ten one-year
  periods that can come out 6–4, and did come out 4–6 — and reverses the conclusion: 9.9%/yr
  against the benchmark's 15.7%, with a worst year of −6.3% against its −12.2%. Both modes ship in
  `pipeline/results/backtest.json`. If a future answer quotes the nested numbers as evidence the
  method works, that is this trap, not a result. The rank correlations are what to quote: mean
  +0.25 for return, +0.91 for volatility. `evaluate()` is the single shared core precisely so there
  is exactly one place a lookahead could enter, and `test_backtest_invariants.py` perturbs the
  prices *and* the rf series after the cutoff and requires the weights unchanged — perturbing
  prices alone left `backtest-lookahead-rf-from-the-end-of-the-window` alive, since rescaling
  prices does not move `panel.index.max()`.
- **Any return comparison in `backtest.py` must be made at matched RISK, and the target is always
  the training window's forecast.** The tangency portfolio ran ~7.6% annualised volatility against
  SPY's ~16.8%, so the headline "9.9%/yr against 15.7%" is mostly a statement about position size.
  `matched_risk@cap*` (solve the frontier at the benchmark's forecast volatility, long-only) and
  `tangency_levered@cap*` (scale up the CML, financed at `^IRX` as it moved) are the comparison
  that settles it, and the answer is **no edge**: 17.6% at 19.1% risk against SPY's 15.7% at 16.8%
  — 0.92 return per unit of risk against 0.93 — and the levered version is worse than the index
  outright at 0.73, with a 44% worst drop on mean 3.4× leverage. Three traps live here:
  - **The target is the benchmark's *forecast* volatility, never its realized one.** Realized is
    the obviously fairer number and is a lookahead that moves no weight vector, so it reads as
    scrupulousness. `evaluate` computes it once and passes it down (`strategies` takes
    `risk_target`) so there is one site, and `leverage`/`risk_target` are in the no-lookahead
    test's comparison tuple for the same reason.
  - **`efficient_risk` constrains volatility to be at most the target, so solved ≠ matched.** An
    unreachable target does not fail — the constraint goes slack and it returns a lower-risk
    portfolio still labelled `matched_risk`. `reaches()` checks and drops the row. This was a real
    bug caught by its own test.
  - **Never average per-period returns arithmetically** (`_compound`, not `np.mean`). The two differ
    by about half the variance, so the mean pays a bonus for volatility — in the tables whose whole
    purpose is comparing strategies at different volatilities.
  The finding worth keeping: forecast risk is accurate on portfolios the optimiser was asked to
  *describe* (SPY 1.03, equal-weight 1.01 realized/forecast) and 12–25% low on the ones it *built*
  (tangency 1.25, matched-risk 1.15). It selects the assets whose estimated variances are lowest,
  and the lowest estimates are disproportionately errors. So "the covariance half works" is true
  only as description; as an optimisation input it is biased the one way that flatters the result.
- **`--mu` chooses the return forecast, `estimates()` is the only place it is chosen, and the
  covariance is deliberately untouched by both models.** `--mu momentum` swaps `frontier.estimate`'s
  sample mean for each asset's return over the past 12 months stopping 1 month short, leaving the
  Ledoit-Wolf covariance, the caps, the solver, the windows and the matched-risk comparison
  identical — one input differs, so the two runs' rank correlations are directly comparable.
  Re-estimating the covariance too would change two things at once, and risk is the half that
  already measures +0.91. The result, `pipeline/results/backtest_momentum.json` against
  `backtest.json`: momentum is **worse**, +0.07 ranking against the sample mean's +0.25, and 0.53
  return per unit of risk at matched risk against 0.92. Three things about that number:
  - **The `skip` is the point.** The most recent month reverses, so omitting it is the standard way
    a momentum backtest is quietly built to fail. `backtest-momentum-does-not-skip-the-recent-month`
    is the mutant, and the tests that catch it perturb **a single bar or a half-open span hanging
    off one edge** — momentum reads exactly two bars, so it is path-independent by construction and
    a perturbation applied to the *middle* of the window would pass on any window at all.
  - **A weight vector is chosen once per holding period, so this is momentum refreshed ANNUALLY**
    and the effect is documented at monthly refresh. `pd.DateOffset` refuses a fractional year, so
    a 1-year hold is the shortest `run_rolling` can walk. Recorded in `method.caveats`, not
    corrected for — do not report the +0.07 as a verdict on monthly momentum.
  - **`momentum_top{N}` exists so a bad result is attributable**: the signal held equal-weight with
    no optimiser at all. It came out 0.62, so momentum was weak here *before* mean-variance touched
    it. It is emitted under **both** models (`estimates` returns `mom` either way), which is also
    the cross-check that the two runs differ in exactly one input — the basket is identical in both.
- **`matched_risk` is absent from some periods, and every printer has to be aligned to `periods`
  rather than compacted.** `reaches()` drops the row wherever the cap put the target out of reach,
  and the shipped ETF universe reaches it at every cap in all ten periods — so the hole first
  appeared under `--mu momentum` (cap 10%, 9 of 10) and took two latent bugs with it. Compacting a
  strategy's returns and the benchmark's independently, then zipping, compares one strategy's
  period 4 against the benchmark's period 3: **wrong years, plausible number, silent.** Keep the
  `None` placeholders, count wins only over pairs where both ran, and annotate a partial row with
  `(N of M periods)` — a return compounded over 9 periods must not sit in the same column as one
  compounded over 10 with nothing to say so. `test_a_strategy_missing_from_some_periods_is_compared_against_the_right_years`
  is the guard, and it hand-builds a block with a hole for the same reason the rest of that file is
  synthetic: no run over the shipped data produces one.
- **A benchmark of single stocks has to be PRICED WITHOUT BEING INVESTABLE, and the hold-out is
  `evaluate(..., investable=)`.** An index fund among its own constituents has lower variance than
  almost any one of them, so the minimum-variance solve buys it on sight and "did mean-variance
  beat the index" becomes a question about a portfolio allowed to *be* the index. `--benchmark SPY`
  puts SPY in the panel and out of the candidate set; `restrict(est, keep)` takes the **submatrix**
  of the already-estimated `mu`/`cov` and carries the shrinkage across unchanged, because
  re-estimating over the subset would make the held-out run differ from the full run in something
  other than the hold-out. Four things here are not obvious:
  - **`strategies`'s risk target must not be gated on `benchmark in symbols`.** It was, so holding
    the benchmark out silently deleted `matched_risk` — the one comparison that settles the
    question — and every table still printed.
  - **The extra benchmark row is gated on `benchmark not in est_s`, not on `investable is not
    None`.** An `investable` list that happens to contain the benchmark is the ETF case spelled out
    longhand and `strategies` emits the row itself; the looser gate appended a *second* identical
    row, which every printer hid by taking the first match.
  - `investable=None` and `investable=list(panel.columns)` must be `==`, bit for bit. That is what
    makes the feature safe to add to already-published ETF results, and
    `test_holding_nothing_out_is_the_same_run_as_not_holding_out_at_all` is what found the
    duplicate row.
  - **Every printed label that names the universe or the benchmark comes off the block**, not out
    of a string literal. `'all 116 equal'` and `'S&P 500'` were literals, and a 480-stock universe
    printed a column headed 116.
- **`run_window` is the third mode and it is nested BY DESIGN: one cutoff, a fixed-length training
  window behind it, several hold lengths in front.** It answers "how did this one portfolio's lead
  over the index evolve as it was held", which is a real question, and it cannot be turned into a
  batting average: the 3-year row's returns *contain* the 2-year row's, and all rows share one
  weight vector per strategy (`test_the_nested_window_rows_are_one_weight_vector_measured_at_several_dates`
  asserts exactly that). Any claim about whether the method *works* belongs in `run_rolling`. Both
  ends are stated rather than taken from the panel — a hold that would run past the last bar
  **raises** instead of being reported at its requested length and being shorter, and the training
  window is fixed-length so it does not lengthen every week the data refreshes. Its no-lookahead
  test is not redundant with `run_cutoff`'s: `run_window` does its own date arithmetic for both ends
  of both windows, which is a second place a bar can land on the wrong side of the cutoff.
- **The S&P 500 run answers the opposite way to the ETF run, and the crux is that its training
  window contains COVID.** `pipeline/results/backtest_sp500.json` — 4-year window ending 3 years
  back, weights chosen on 2023-09-01 over 477 investable single stocks (478 in the panel) with SPY
  priced but not investable:

  ```
  python -u pipeline/backtest.py --universe sp500_2023 --benchmark SPY --start 2019-08-01 \
    --years 3 --window --train-years 4 --cutoff-years-back 3 --window-holds 1,2,3 \
    --hold-years 1 --span-years 3 --out pipeline/results/backtest_sp500.json
  ```

  On the three non-overlapping one-year holds — the only non-nested read this file permits, all of
  them after the as-of date — compounded return per unit of **realized** risk: `matched_risk@cap10`
  28.9% at 16.2% = 1.79 and `tangency@cap10` 26.9% at 14.9% = 1.80 against **SPY's 22.3% at 14.9% =
  1.49**, with equal-weight at 1.17. `matched_risk@cap10` beat SPY in **3 of 3** periods. That is a
  genuine edge, unlike the ETF universe's 0.92 against 0.93 — and it is **three observations**, with
  the return forecast still ranking assets at only **+0.13** (risk +0.74). The trap: SPY's *forecast*
  volatility from the training window was **23.08%** against the 15.34% it realized over the three
  years, because 2020 is in the training data. "Matched to the benchmark's forecast vol" therefore
  meant aiming above the risk the index actually ran in two of the three periods — matched-risk
  realized 15.4%, 19.3% and 13.7% against SPY's 12.4%, 19.6% and 12.8%, so 1.25×, 0.98× and 1.07×.
  **Quote the return-per-unit-of-realized-risk column, never the raw returns**, and never the nested
  window rows (cap 20% reads 38.0% vs 26.9% at one year) as evidence the method works — that is
  `run_window`'s nesting, documented above. These figures move every time the file is regenerated,
  because the third one-year hold ends at the panel's last bar; the file carries its own
  `panel.end`, and a number quoted here without one is from some other run.
- **`pit.py` SUPERSEDES the S&P 500 result above, and it reverses it. Ten one-year holds say there
  is no edge.** `sp500_2023.toml` is one cutoff, so the only honest read it permits is three
  non-overlapping holds — three observations. `pipeline/universes/sp500_pit.toml` is the membership
  as it stood on **each of ten rebalance dates** (2016-09-01 … 2025-08-29, reconstructed from a
  change log), and `pit.py` walks it: a fixed 5-year training window behind each cutoff, one
  non-overlapping 1-year hold in front, the candidate set rebuilt from *that date's* snapshot, SPY
  priced and never investable.

  ```bash
  python -u pipeline/pit.py --train-years 5 --hold-years 1 --from-year 2016 --to-year 2025 \
    --caps 1.0,0.2,0.1 --benchmark SPY --out pipeline/results/backtest_pit.json
  ```

  Compounded return per unit of **realized** risk over the ten holds, from
  `pipeline/results/backtest_pit.json`: **SPY 15.31%/yr at 16.88% = 0.907**, against
  `matched_risk@cap100` 14.42% at 19.22% = **0.750**, `equal_weight` 0.753, `min_variance@cap100`
  0.748, `tangency@cap100` 0.605, `momentum_top10` 0.645. **Every optimised strategy at every cap is
  below the index per unit of risk.** `matched_risk` beat SPY's *return* in 5 of 10 periods, which is
  the number to distrust: it ran 19.2% volatility against the index's 16.9%, so the wins are position
  size. The forecast bias documented under `backtest.py` is the mechanism and it is larger here — the
  matched-risk target is the benchmark's *forecast* vol, and it overshot in most windows.

  **The rank correlations are the number to quote here too, and over ten cutoffs they are the
  cleanest statement this repo has: the return forecast ranks at +0.06 and the risk forecast at
  +0.72.** Ten cutoffs is the widest evidence for the split the whole project keeps finding — the
  covariance half describes the cross-section, the mean half is a coin flip — and it is *narrower*
  than the ETF run's +0.25 rather than wider, on single stocks, which is where idiosyncratic variance
  was supposed to give the mean something to find. `momentum_top10`'s own score is **−0.02**: over
  these ten years the signal ranked slightly backwards, so its 17.69% raw return is the highest
  number in the table earned by the worst-ranked forecast in it, at 27.4% volatility. Quote the ranks
  before the returns; a return column over ten observations is one market.

  **This does not mean the `backtest_sp500.json` numbers were wrong; it means three observations
  from one cutoff were three observations.** That file's `matched_risk@cap10` reads 1.79 against
  SPY's 1.49 over 2023-09-01 → 2026, and the 2023 cutoff is one of the ten cutoffs here. Do not
  quote it as the S&P 500 answer. Quote this one, and say it is ten.

  Eight things about this run are decisions, and the last three are the ones a future answer is
  most likely to get wrong:
  - **Costs are charged to BOTH sides or not at all.** `--trade-bps 5` writes
    `backtest_pit_net5bp.json`: the strategies pay 5bp per dollar traded on a buys-plus-sells
    turnover, and SPY pays its **published 9.45bp expense ratio**, because charging the optimiser's
    frictions against a frictionless index is the bias the flag exists to remove. Net: SPY 0.897,
    `matched_risk@cap100` 0.745. **The gap is not a friction artifact** — annual rebalancing barely
    trades — so `--trade-bps 0` reproducing a pre-flag run is the default and the 5bp file is the
    check, not the headline. Three things in the model are load-bearing and each has its own test:
    the charge is **proportional**, so it commutes with returns and the net path is the gross path
    times a cumulative product — which is why both readings ship from **one** solve and the net
    reading needs no weight trajectory of its own; turnover is **buys plus sells** (`sum |dw|`, and
    `mix_turnover`'s factor of 2 for the levered rows' daily reset), because the two conventions
    differ by exactly 2× and that is the size of the whole effect; and `Costs.zero()` gates the net
    columns so a zero-cost run emits none at all, which is what makes every result published before
    the flag existed reproduce **bit for bit**. That last one is asserted on the two runs' **JSON
    text**, not field by field — a new key holding a zero is exactly the difference a field-by-field
    loop is written not to notice.
  - **Both sides are total returns.** Every column of the panel is `store.total_return()`, dividends
    reinvested at the close, SPY included. There is no version of this comparison where one side
    reinvests and the other does not.
  - **Three ticker-identity failures, three tables, three DIFFERENT remedies**, and confusing them is
    survivorship bias in whichever direction the wrong remedy points. `[recycled]` — the ticker now
    means another company entirely (APC is ARKO Petroleum, not Anadarko) — is stripped from **every**
    snapshot. `[reissued]` — another company before a date, a legitimate member after (DOW, FOX,
    FOXA, Q) — is stripped only from snapshots **before** `from`; applying the recycled rule to it
    deletes Dow Inc from seven snapshots of an index it was in. `[truncated]` — right company, stub
    series — is stripped from **none** and is exempt from the adjclose reconstruction check only.
  - **`pit.overlaps_its_membership` is the general form of the PARA check and it is not manual.** The
    index held a ticker between two known dates, so a series that never reaches into that interval
    cannot be that constituent's history whatever it is named. It caught MON, CA and DNB — all three
    clean, full-length, filter-passing series belonging to someone else. Both directions are needed
    (`entirely after` is a reissue, `entirely before` is a predecessor's stub) and it is a
    **necessary, not sufficient** condition: a ticker reissued fast enough to overlap would pass.
  - **A stress bound over this run must take its set of missing names FROM THE RUN, never from the
    membership file's `unpriceable` literal.** The literal is 67 names and is wrong in two directions
    at once: it omits the constituents the identity tables *stripped* (they never reach `load_panel`,
    so they are in no `dropped` reason) and it includes the 23 the second vendor serves (missing from
    a baseline run, present in a restored one — one literal cannot be right for both). Each period
    reports its own `membership.unpriced_constituents`; the union is **102** baseline and **99**
    restored. `test_a_bound_over_a_point_in_time_run_bounds_that_runs_own_missing_names` is the guard,
    and nothing else can see the failure — a bound built from the wrong set has a right span, a
    reconciling benchmark ratio, and is simply too narrow.
  - **The bound does not rescue the result.** In `backtest_pit_restored_cash_stress.json` (so the
    12.60% below is the restored run's equal-weight over its own 9.999-year span, not the baseline
    table's 12.77% over ten periods — check which file a number came from before comparing two):
    equal-weight is 12.60%/yr as measured, 10.12% with all 99 wiped out and 13.30% with all 99 at the
    best outcome anywhere in the panel (ANSS, 4.33×) — a 3.18 pp/yr swing whose most favourable end
    is still ~2 pp/yr below SPY's 15.31%. For
    `matched_risk@cap100` to reach the index the 99 would have to **outperform everything else by
    29.7% terminal**, while being selected as lowest-risk and highest-forecast-return — and the 23
    that *were* restored were held by it **0 times**, in any period, under either continuation.
  - **`--extra-panel` restores the 23 the second vendor serves and changes nothing.**
    `backtest_pit_restored_cash.json` and `..._spy.json` are the two continuations, and every
    `beats_the_benchmark_per_unit_of_risk` is `false` in both. "The conclusion holds under both" is
    the claim; if it ever holds under only one, the file to change is not this one. The cache is
    `pipeline/data/delisted_pit/` and it reaches the optimiser only through `join_extra`.
  - **Two caveats live in `method.caveats` and are not corrected for.** The holds are
    non-overlapping but consecutive *training* windows overlap by 4 years, so a persistent regime is
    estimated much the same way several times running — the ten holds are ten observations of the
    hold, not ten independent estimates. And the membership is reconstructed from a change log, not
    licensed from the index provider: it reproduces 498 of the 503 hand-verified 2023 names, the 5
    misses being ticker spelling.
- **`frontier.SOLVER` is pinned to CLARABEL, and the reason is a SIZE-DEPENDENT SILENT FAILURE.**
  PyPortfolioOpt's default for a QP is OSQP, a first-order method. At 116 assets it converges; at
  the S&P 500 universe's ~477 it returns `user_limit` — its iteration budget — on `max_sharpe`, and
  `_solve` reports that as
  "the objective is not attainable at this cap", because from outside an exhausted solver and an
  infeasible problem are the same thing. On a frontier that shortens the curve; on a backtest it
  raises. CLARABEL is interior-point and does the same several-hundred-asset problem in 0.1 s. Pinning it means
  the answer does not depend on how many assets a universe happens to have, and a solver default
  changing upstream cannot move a published number without this line changing too. Rebuilding the
  six shipped artifacts under CLARABEL moved weights by 1e-6 to 6.7e-4 (MUB crosses `WEIGHT_FLOOR`
  at cap 1.0) and improved cap-100 min-variance from 0.00868693 to 0.00868586 — same 63 points, same
  `max_sharpe_index`, no `failed_targets`, so it is a strict improvement and not a re-tuning. Two
  tests guard it and they are a pair on purpose: the 500-asset one shows the *harm* but is
  seed-dependent (OSQP is at a knife edge — seed 1 exhausts it, 20260906 does not), and
  `test_solve_names_its_solver_rather_than_taking_the_default` monkeypatches `EfficientFrontier` and
  fires always. `SOLVER` may be renamed; it may not be removed.
- **The weight cap IS a pipeline parameter** (a constraint on the QP), so each cap is its own
  solve and its own file.
- **Weights are interpolated, never curves.** The browser takes convex combinations of solved
  long-only portfolios, so every draggable position is genuinely feasible. Do not replace this
  with a spline through the frontier points — the numbers in the weight table would stop
  describing the point on the chart.
- **Both ends of the frontier are supplied, not asked for, and the right one is CONSTRUCTED.**
  `np.linspace(r_lo, r_hi, n_points + 2)[1:-1]` skips both endpoints because neither is a variance
  minimum the QP can be pointed at: `r_lo` is the `min_volatility` solve, and `r_hi` is an LP
  vertex where `w'mu == r_hi` admits exactly one feasible point, so cvxpy returns infeasible on
  the wrong side of a rounding step as often as not. `frontier.max_return_weights` builds that
  vertex greedily instead — pour `cap` into the highest-mu asset, then the next — and
  `max_feasible_return` is now `w @ mu` over it, so the scalar and the portfolio cannot drift
  apart. Until 2026-09-05 the vertex was simply omitted and the curve stopped one grid step short
  of `r_hi`, which is a defect in the *picture* (at cap 1.0 the curve ended below and left of
  SMH's own dot, so the top assets read as beating the frontier) **and** in the interaction, since
  the interpolated path stops wherever the point list stops and the maximum-return portfolio was
  therefore not selectable at all. The frontier is 63 points per cap, not 62, for that reason.
- **The store holds `close` + an event log, never `adjclose`.** Not a storage detail — the
  dividend adjustment is *backward*, so an adjusted column is retroactive and can never be
  append-only, and year-partitioning it would buy nothing. Total returns are rebuilt at load by
  `store.total_return()` and checked against the vendor's own field to ~1e-6. A change that
  stores the adjusted series undoes the entire point of the layout. The reverse mistake is
  equally bad: **do not materialise the adjusted panel** to disk. It is 0.35 s to derive, it is
  retroactive, and a stored derivative can drift from the inputs it claims to summarise.
- **"`close` is split-adjusted" is two claims, and they need two assertions.** The adjustment is
  checked ACROSS each split (the bar before against the bar after — two adjusted bars whatever the
  vendor did with the one between them), and a split-day *spike* is checked separately. Adding the
  S&P 500 constituents turned up the case that forces the split: BRO's 1983 3:2 split day prints
  0.222222 between neighbours of 0.148148 and 0.150000 — exactly 1.5×, reverting the next day. The
  level across it moves 1.3%, so the series *is* adjusted; one bar is not. A single unadjusted bar
  is a pair of equal-and-opposite daily returns and a variance does not care that they cancel, so it
  is a real defect — and it is listed by symbol and date in `KNOWN_UNADJUSTED_SPLIT_BARS` rather
  than absorbed by raising the threshold to 0.51, which would have hidden it and the next one.
- **The download is browser-side, and its numbers are the page's numbers.** `export.ts` writes
  weights at the pipeline's own transport precision (6 dp), which is why the total is written out
  as measured instead of asserted to be 1 — interpolating two 6-dp vectors and dropping sub-1bp
  positions loses a little, and a file claiming 1.000000 would be the one number on the page the
  reader could not reproduce. `max_drawdown` ships as a *positive* fraction and says so in the
  file, because a bare drawdown number is ambiguous in sign once it leaves the page.
- **View state is the reader's, kept in three places with one precedence: URL, then storage, then
  defaults.** The URL fragment has to win or a shared link would render differently for the
  recipient than for the sender. Three things in `config.ts` look like tidying and are not: `pos`
  is a *fraction* of the path, never a point index, because switching the cap switches to a
  frontier with a different point count; `pos: null` means "track the tangency" and is not the
  same value as `0`; and `writeStored` **removes** the key when the config equals the defaults,
  since Reset clears it and the state change that follows would otherwise write it straight back.
  `coerce` validates against the *manifest*, so it drops a cap or a group this build did not ship
  and reports what it dropped — a load that silently adjusts something is a load that lied.
  There is no account and nothing is uploaded. If a backend is ever added, `parseFile`/`serialise`
  are the seam; nothing else in the file needs to know.
- **A universe is a file, not code — in the browser too.** `universe.py` loads and validates; the
  symbols live in `pipeline/universes/*.toml`. One store keyed by symbol serves all of them, so a
  symbol two universes share is fetched once. Adding a universe must not require touching
  `universe.py` — and must not require touching `web/src` either. **No group name of any universe
  appears anywhere in `web/src`.** Labels come from `manifest.group_labels`, which the pipeline
  fills for every declared group, read through the single `viz.groupLabel`; `types.Group` is
  `string` rather than a union of this universe's three, because a compile-time union enforces the
  opposite of the rule. There were four hardcoded `{equity: 'Equity', …}` maps, two disagreeing
  about capitalisation, and all four would have rendered blank for a second universe with no error
  anywhere. `viz.test.ts` scans `src/` for a group name in either shape a copy takes (quoted, or
  bare before a colon) — that scan is the only guard that can fail when a fifth copy appears,
  since a fifth copy renders correctly for *this* universe. It strips comment lines, or it flags
  the comments written to prevent the thing.
- **A POINT-IN-TIME universe (`sp500_2023.toml`) is only clean FORWARD of its as-of date, and the
  three ways it breaks are in increasing order of danger.** The file is the 503 lines that made up
  the S&P 500 on **2023-08-30**, as the index stood that day — a list downloaded today knows which
  companies did well enough to still be in it, and a backtest over that measures the selection. The
  consequence that constrains every run: **a rolling test on this file may only cover periods after
  the as-of date**, because before it the membership was chosen with knowledge of what followed.
  That caps the honest non-nested test at three one-year holds, and it is why the same file is also
  run through `run_window`, whose nesting is a known and stated limitation rather than a hidden one.
  The three failure modes, and only the first is loud:
  1. **The vendor 404s the ticker.** 17 of the 503 — it serves *nothing* for a symbol that stopped
     trading rather than a series that ends. Excluded from the **store**, which is not the same as
     unmeasurable: a second vendor serves 12 of the 17, and `delisted.py` (next bullet) measures
     what they did. Five are priced by nobody and are bounded instead.
  2. **The ticker changed.** A symbol lookup cannot distinguish a rename from a merger that adopted
     the partner's ticker, so each successor is verified **by price level at a date years before the
     change**. BK→BNY, FI→FISV, FLT→CPAY, MMC→MRSH, PARA→PSKY and PEAK→DOC reproduce the old
     ticker's own path and are held under the new spelling; WRK→SW does not (SW carries Smurfit
     Kappa's history from 2008, not WestRock's) and was rejected as a splice of two companies.
  3. **The ticker was RECYCLED**, and there is no automatic screen for it. `PARA` fetches 1,100+
     clean bars with full coverage and is **Banzai International**, not Paramount Global — Paramount
     merged into Paramount Skydance (PSKY) and Nasdaq reissued the symbol to a micro-cap. The fetch,
     the coverage filter and the adjclose cross-check all pass; the optimiser would have held a
     micro-cap labelled as a 2023 constituent. It surfaced only because the new occupant's three
     reverse splits tripped the store's split test. The check is the vendor's current `longName`
     against the constituent the ticker is supposed to be, and it is **manual** — all 486 were
     screened this way and PARA is the only case.
  Two more things about the file: start dates
  are deliberately absent (ten constituents begin after the training window opens, and `load_panel`'s
  measured filter drops them and says so, where a hand-written inception date goes stale silently),
  and it declares `benchmark = ""` so SPY can only enter via `--benchmark`, priced and not
  investable. Its generator is `/tmp/make_sp500_universe.py` — deliberately not committed, since the
  file is the artifact and the script would invite regenerating it from *today's* membership.
- **The delisted constituents are MEASURED now, and the measurement reversed the sign of the
  survivorship-bias argument this file used to make.** `pipeline/delisted.py` fetches the 12 of
  `sp500_2023.toml`'s 17 exclusions that a *second* vendor (stockanalysis.com) still serves, and
  `backtest.py --extra-panel` runs the whole S&P test again with them restored. The result: from the
  cutoff to each name's own last traded bar the twelve returned **+21.8% against SPY's +46.4%** over
  matched sub-spans, and only **4 of 12** beat the index (WBA −90.5pp, IPG −69.9pp, CDAY −64.9pp;
  CMA +57.2pp is the one large winner). Deletion from the index is not a synonym for a takeover
  premium. Restoring them moves the equal-weighted comparator from 16.16%/yr to 16.14% (spy
  continuation) or 15.98% (cash), i.e. **−0.02 to −0.18 pp/yr**, where the old argument from the
  premium said +0.18 to +0.47 pp/yr in the other direction. The conclusion is untouched:
  `matched_risk@cap10` is 1.79 return per unit of realized risk against SPY's 1.49 in the baseline
  and in **both** restored runs, all 1782 bars, and no tangency or matched-risk portfolio held any
  of the twelve in any period under either treatment. Seven things here are decisions:
  - **This vendor must never enter the store.** `store.py`'s contract is one vendor shipping
    `close` + an event log + an `adjclose` to check the reconstruction against; this one ships a
    level and nothing else, so writing it into `data/prices/` would leave the store's tests passing
    on symbols they cannot test. It lives in `data/delisted/` and reaches the optimiser only
    through `join_extra`, which records the provenance in the result JSON.
  - **The gate is a RATIO test, not a level test.** Two total-return series agree up to one
    arbitrary scale factor, so the question is whether `second/first` is constant: ≤7.3e-4 on nine
    controls. The tenth, **T at 1.0e-1, is the 2022 WarnerMedia spinoff** — a distribution in kind
    is not a cash dividend and the two vendors credit it differently. Named in `SPINOFF_CONTROLS`
    rather than absorbed by loosening the threshold. K carries the same ambiguity with no second
    series to resolve it and is a stated provenance caveat.
  - **The final traded bar of every series is dropped, uniformly.** It is an index-deletion auction
    or a stub: three of the twelve print volume 0 or 1. Where an independent check exists it
    sometimes fails and sometimes does not — **CMA held 1.89542 × FITB to ±0.10% for eleven bars
    then printed its last 5.34% below that ratio; IPG held 0.35386 × OMC to ±0.21% including its
    last.** With only 3 of 12 having a checkable all-stock acquirer, the uniform rule is the one
    that needs no case list, and its cost is *stated* per symbol in `provenance_*.json` (−8.62% on
    CTRA to +1.36% on CDAY) rather than asserted to be small.
  - **`range=MAX` returns one year with a 200.** A range string is a request, so `fetch_series`
    asserts the served span reaches the date the caller needs.
  - **Two continuations, because what happens after the terminal bar is an assumption**: cash at
    `^IRX` as it moved, or SPY. Both ship. "The conclusion holds under both" is the claim; if it
    ever holds under only one, the file to change is not this one.
  - **The one artefact, and it must not be reported as a result**: `min_variance@cap10` improves
    from 1.34 to 1.52 (cash) / 1.61 (spy), because a continuation is *smooth* and therefore has
    artificially low variance, which is exactly what a minimum-variance solve buys. It is the only
    strategy that ever holds a restored name, and only once the training window has gone mostly
    synthetic.
  - **The five nobody prices are bounded, and the bound lives in its own artifact kind.**
    `delisted.py --stress` writes `*_stress.json` with `"kind": "stress_bound"`, which
    `test_results_invariants.py` splits its fixture on — without the discriminator every assertion
    in that file runs over a document with no `periods` and passes vacuously. Equal-weighted: 16.16%
    as measured, 15.75% with the five wiped out, 16.32% with all five at the best outcome observed
    anywhere in the panel (CMA, 2.21×) — a 0.565 pp/yr swing. Optimised: for `matched_risk@cap10` at
    1.77 to fall to the benchmark's 1.48 the five would need a **10.6% terminal shortfall**, and at
    cap 10% they can occupy at most 50% of the portfolio, so **21.2% of underperformance against
    everything else** while being selected as lowest-risk and highest-forecast-return — and the
    twelve that *were* restored were held by that strategy **0 times**.
- **The store's split tests are scoped to `_held_symbols()` — symbols some universe's `[assets]`
  actually lists — not to everything in the store.** The store is keyed by symbol and shared across
  universes, so it accumulates symbols no universe holds, including ones excluded *because* they are
  broken. PARA is the case: it stays on disk, it is a different company, and its reverse splits fail
  the split test correctly while telling us nothing about any estimate. Scoping narrows the tests to
  exactly the claim they make. It is not a way to silence one — add `PARA` to any universe's
  `[assets]` and the failure comes straight back.
- **One `generated_at` per run, on all six artifacts, and `data.loadBundle` refuses a bundle where
  they disagree.** The counts cannot see the failure this catches: `update-data` runs weekly over
  a symbol set that rarely changes, so last week's cached `stats.json` against this week's
  frontier has exactly the right number of symbols in exactly the right order, and every
  calculation on the page succeeds while pricing today's weights with last week's covariance. Six
  files fetched separately with per-file cache lifetimes is exactly how that happens, and
  `no-cache` on the request is a request. Throwing is right rather than picking the newest: there
  is no second source, and a reload is the fix. `data.test.ts` tests the refusal **one stale file
  at a time** — a stale frontier is as likely as a stale `stats.json`.

## Two pieces of algebra the browser depends on

Both are in `web/src/portfolio.ts`, both have cross-checks in `web/src/portfolio.test.ts`, and
both are easy to "simplify" into something subtly wrong.

1. **The segment quadratic.** Risk between solved points `i` and `j` is
   `var(u) = (1-u)²Sii + 2u(1-u)Sij + u²Sjj`, with `Sii = wᵢ'Cwᵢ` and `Sij = wᵢ'Cwⱼ`
   precomputed once per segment. This is not an approximation of `w(u)'Cw(u)` — it *is*
   `w(u)'Cw(u)`, which is why interpolated risk is exact in O(1). The test checks it against a
   brute-force quadratic form to 1e-12.
2. **The closed-form tangency.** Maximising `(A+Bu)/√(C+Du+Eu²)` over a segment: the `u²` terms
   cancel identically, leaving one linear root. So moving the rf slider re-derives the exact
   tangency point rather than sampling for it. The test checks it against a 400-step dense scan
   at seven rf values.

## Testing discipline

Beyond the workspace rule (never weaken a test to make it pass): **both suites read the shipped
artifacts, not fixtures.** That is deliberate — they fail when the data is wrong and not only
when the code is. Keep it that way when adding tests.

**Five files are deliberately synthetic, and the test for whether a sixth should be is whether
the shipped data can express the property at all.** `test_universe_invariants.py` (a validator
needs invalid input), the store's layout tests (a store can only be wrong on the *second* write),
`test_backtest_invariants.py` (116 assets trading on the same 3,941 days cannot distinguish a
monthly rebalance from a buy-and-hold, and no file on disk can distinguish an honest backtest from
a leaking one), `test_delisted_invariants.py` (all twelve cached series have a bad-looking final
bar and a clean penultimate one, so nothing on disk distinguishes "drops the last bar" from "keeps
it" — and `join_extra`'s load-bearing claim is that it CANNOT move the panel's window, which is a
statement about what does not happen), and `test_pit_invariants.py` (three reasons at once:
`sp500_pit.toml` is *valid*, so nothing built from it shows what a malformed `[reissued]` entry
does; its eleven snapshots share ~95% of their names, so no run over it distinguishes "each window
used its own membership" from "every window used the 2016 list"; and `run`'s two window-length
assertions fire on a panel that does not reach its requested ends, which is the one panel the
shipped store never produces). Reaching for a fixture because the real data is awkward is
the thing this rule forbids; reaching for one because the real data is *uniform in the dimension
being tested* is the only way to test it. `test_frontier_invariants.py`'s `wide_est` fixture is the
extra case and passes the same test: the property is a function of **problem size**, and no artifact
built from a 116-instrument universe can express what happens at 500.

`test_results_invariants.py` is the other direction — it reads `pipeline/results/*.json`, which are
**committed by hand and never rebuilt by CI**, so it is the only test that opens them at all. Read
its docstring before adding to it: two of its three original assertions cannot be violated by any
run `main` can perform, and it says which and why. That is unusual enough to be worth stating rather
than quietly leaving green tests that guard nothing. It splits its fixture on the `kind` field, and
`test_every_committed_result_has_a_shape_this_file_knows_about` is what makes a *third* artifact
kind fail loudly instead of being tested by nothing. **That guard has since fired for real**:
`pit.py`'s four files carry `kind: "pit_backtest"` and put `benchmark`/`periods` at the top level
rather than under `rolling[0]`, and the test named them on the first run. `_blocks` was taught the
shape — deliberately by SHAPE and not by kind, so a fourth document that also carries its periods
at the top level cannot silently yield zero blocks, which is a vacuous pass in every assertion
there.

`pipeline/tests/_mutate.py` is the reason to trust the pipeline suite. It stages a copy of the
package, applies one source-text edit, and asserts a *named* test catches it; a mutation whose
pattern no longer matches is a hard error, because a stale mutation would be reported as caught
while testing nothing. It has already found five tests that guarded less than their names claimed
(`_prune_dominated`'s `while`-vs-`if`; `_clean`'s rounding; "the last bar is never adjusted
*whatever* the dividend history", which had no case with a dividend on the last bar; "total
return compounds at least as fast as price return", whose `>=` was satisfied by deleting the
dividend adjustment outright; and "joining extra columns cannot move the panel's window", which
survived dropping `join_extra`'s reindex because `pd.DataFrame(kept, index=panel.index)` pins the
days a second time — the mutant has to remove **both** pins, and that is what the survivor told
us). Run it after touching `frontier.py`, `fetch.py`, `store.py`, `backtest.py`, `pit.py` or
`delisted.py`:

```bash
python -u pipeline/tests/_mutate.py --check   # every needle still matches, in seconds
python -u pipeline/tests/_mutate.py     # 119 mutants, 56 min measured; exits 1 if any survives
python -u pipeline/tests/_mutate.py --only backtest    # the 39 backtest mutants, ~18 min
python -u pipeline/tests/_mutate.py --only pit         # the 18 point-in-time mutants, ~9 min
python -u pipeline/tests/_mutate.py --only cost        # the 10 cost-model mutants, ~5 min
python -u pipeline/tests/_mutate.py --only delisted    # the 17 second-vendor mutants, ~8 min
node web/tests/_mutate.mjs              # 38 mutants, ~40 s; --only <substring>, --list, --keep
```

**Run `--check` first, always.** It is the whole set's needles against the real source and it takes
seconds; the per-mutant check inside `_apply` cannot replace it, because it fires on the mutant that
happens to be reached first and aborts, hiding every other stale needle behind it. Four had gone
stale simultaneously after the cost-model and `fetch_longest` refactors on 2026-09-06, and finding
them one 40-minute run at a time is how a harness stops being run. Both checks stay: the preflight
reads `pipeline/`, `_apply` reads the staged copy that is actually mutated, and those are two
different files.

Each mutant is ~27 s now (the suite is 262 tests), so the timings above move as tests are added;
the counts are what to keep honest. `_stage` ignores `data/`, so `pipeline/data/delisted/` is
absent from every mutant tree and the two cache-reading tests in `test_delisted_invariants.py`
**skip** there — every `delisted-*` mutant is killed by one of that file's synthetic tests, and one
aimed at the cache would be reported MISKILLED.

`web/tests/_mutate.mjs` is the same instrument over `config.ts`, `export.ts`, `data.ts` and
`viz.ts` — run it after touching any of those four. Same four-part contract, and each part exists
to stop the harness reporting success while measuring nothing: a needle that no longer matches
**exactly once** exits 2, a mutant that does not parse exits 3 (it fails every test in its file,
which looks exactly like a kill), the expected test's *name* has to appear among the failures or it
is reported MISKILLED, and it mutates a staged copy at `web/.mutate/` rather than the working tree.
The last one is not paranoia about `finally`: an interrupted in-place run leaves a source file
broken in a way that looks like something someone meant. All four are themselves checked — break
one deliberately and confirm the exit code, the same way the tests are.

Neither harness is in `.github/workflows/`, on purpose. They are slow, they are tests *of* the
tests rather than of the code, and a green CI run is not what they are evidence for.

Two mutants in the web set guard rules that no *behavioural* test can reach, and they are the
reason the source scans exist: `group-labels-hardcoded-again` puts a hardcoded
`{<group>: '<Label>', …}` map back into `viz.ts`, which renders **correctly** for this universe and
is caught only by `viz.test.ts`'s scan of `src/`. The harness builds that map from
`manifest.group_labels` rather than writing the literal out, because a harness containing the
literal would be one more copy of the thing the scan forbids — in the file whose job is to prove
the scan works.

It also found a claim the shipped data **cannot** exercise: all 116 survivors trade on exactly the
same 3,941 days, so `ffill().dropna()` and `dropna()` return the identical panel and the loudest
rule in `fetch.py`'s docstring was guarded by nothing. Guards for that class of property have to
be built on a synthetic store — see `test_load_panel_intersects_trading_days_rather_than_forward_filling`.
When a mutation survives, ask which of the two it is before touching a test.

The `universe.py` mutants are that same shape for a different reason: they remove a *validation*,
and the shipped universe file is valid, so the rebuild produces byte-identical artifacts and no
artifact test can notice. They are caught by `test_universe_invariants.py`, which writes a
deliberately broken universe to `tmp_path` and requires the raise. A validator can only be tested
with input the shipped data by definition does not contain — so adding a `raise` to `universe.py`
without adding a case there guards nothing.

Three known false-comparison traps, all already fixed — do not reintroduce them:

- The JSON's `sharpe`, `ret`, `vol`, weights and covariance are **rounded for transport** (6, 8,
  8, 6 and 9 decimals). Comparing a fresh recompute against a shipped field at 1e-9 fails on
  correct code. Compare recompute-to-recompute, and check the rounding separately at its own
  precision.
- **Never pick an extremum by argmax over a rounded field.** The frontier is stationary at the
  tangency, so its neighbour ships the *same* 6-decimal `sharpe`; `max` then returns whichever
  came first, which is a fact about point order. This shipped `max_sharpe_index = 39` where the
  better point was 40 (by 1.1e-9), and the pipeline test agreed because it resolved the tie the
  same wrong way — only the browser, which recomputes from weights and covariance, disagreed.
  `frontier._extremum_indices` now works from unrounded values; two tests and a mutation guard it.
- A test that asserts only `w.sum() == 1` passes on a `_clean` that never rounds. Assert the sum
  *and* `np.array_equal(w, np.round(w, 6))`.

## Charts

Built to the `dataviz` skill; the non-obvious consequences:

- **Only two categorical hues exist**: `--cat-1` frontier, `--cat-2` CML/tangency/rf. Both
  validated at `--pairs all` in light and dark. The 116 asset dots are deliberately **gray** —
  colouring them by group would need five slots, which fails the CVD gate. Group identity lives
  in the filter row, legend, tooltip and table, so it is never colour-alone because it is never
  colour. Filtering changes **opacity**, never hue.
- `index.html`'s `data-palette` lists only the two categorical slots. Gray is excluded on
  purpose: it makes no categorical claim and would fail the chroma floor.
- Chart labels carry `class="mark-label"` (a surface halo via `paint-order: stroke`) and go
  through one `declutter()` pass in priority order. The bottom-left corner stacks the
  minimum-variance portfolio, `rf` and the lowest-volatility asset by construction, so labels
  there collide unless decluttered.
- The tangency marker is a **ring** larger than the handle, because the handle's default
  position is the tangency and a filled disc was invisible underneath it.
- **The whole plot is a pointer target, and there are four of them going through one
  `beginDrag`.** A transparent `<rect>` over `PLOT` (first painted child, so every mark stays on
  top of it and keeps its own handlers), the 26px band over the curve, the asset dots, and the
  handle. All four snap to `nearestOnPolyline` — *except* the handle, which passes `snap: false`
  because it is already under the cursor and seeking would nudge the portfolio before the drag
  began. The dots need their own `onPointerDown` rather than relying on the background rect: they
  are painted above it and are not its descendants, so nothing bubbles, and 116 dots over the
  middle of the plot would be a lot of pixels where clicking quietly did nothing. Verify this over
  CDP if you touch it — `aria-valuenow` on the handle before and after an
  `Input.dispatchMouseEvent` is the whole check, and no unit test reaches it.
- **The default view is fitted to the FRONTIER, not to every asset** (`config.defaults`). Fitting
  to every asset lets one outlier set the domain — UNG at −24% return and 49% volatility costs the
  bottom half of the plot and a third of its width. The cost of the default is stated in
  `config.ts` and is real: an off-scale asset is clipped and only its clamped edge *label*
  survives, so UNG reads as a name in the corner with no dot under it. "Fitted to every asset" is
  one click away.
- **The root `<svg>` is `role="group"`, and `role="img"` sits on an inner `<g>` holding every mark
  *except* the handle.** Not a stylistic choice: WAI-ARIA makes every descendant of a `role="img"`
  presentational — the subtree is replaced by the label — so with `role="img"` on the root, the
  handle's `role="slider"` was inside a node telling assistive technology to ignore its children,
  and the page's only interactive control could be pruned from the accessibility tree. The split
  keeps both the long numeric description and the slider. Moving `role="img"` back to the `<svg>`
  reintroduces it silently, since nothing about the picture changes.
- `.axis-title` is **not** uppercased: `text-transform: uppercase` renders "annualised σ" as
  "annualised Σ", the summation operator.

After any chart change, render it and look at it, then screenshot both themes. The palette
validator checks colour, not layout. `web/tools/shoot.mjs` is that step — headless Chrome over the
DevTools protocol, no dependencies, and it **exits nonzero if the page complained**, which is the
reason it is a committed tool rather than a snippet: a chart that fails to mount and a chart that
mounts wrong produce the same blank PNG, and the difference is only ever in the console.

```bash
npm run build && mkdir -p /tmp/serve && ln -sfn "$PWD/dist" /tmp/serve/markowitz
(cd /tmp/serve && python3 -m http.server 4455) &
node tools/shoot.mjs http://localhost:4455/markowitz/ /tmp/light.png '' light
node tools/shoot.mjs http://localhost:4455/markowitz/ /tmp/dark.png  '.chart' dark
```

Serve under the `/markowitz/` prefix rather than at the root, so a base-path mistake fails here
instead of after a deploy. Three things in the tool that look like slack and are not: the viewport
is emulated at a *realistic* 1200px and the full-page height is measured from `scrollHeight`
afterwards, because emulating a 4200px-tall window makes `scrollHeight` report 4200 whatever the
content is; `/favicon.ico` 404s are the one HTTP failure it ignores, because Chrome requests
that at the origin root on its own whether the page asks or not; and it **clears
`local_storage` for the origin before every navigate**. That last one is the tool lying in the
one way it exists to prevent: the profile at `/tmp/markowitz-shoot-profile` is persistent (it has
to be — a shared profile is locked by any Chrome the user has open), `config.writeStored` saves
any departure from the defaults, and passing `#theme=` is itself a departure, so a run stores the
whole config and the *next* run renders it. On 2026-09-05 two screenshots taken to check a changed
default rendered the **old** default from a store written minutes earlier — internally consistent,
completely wrong, nothing in the console and nothing in the PNG.

A blank white `<div id="root">` with the CSS loaded has happened here and looks exactly like a
chart bug. It was diagnosed on 2026-09-04 as `npx vite preview` returning 404 for `/assets/*.js`
when the request carries `Sec-Fetch-Dest: script`; **that diagnosis did not reproduce on re-test**
the same day (Vite 7.3.6, all three `Sec-Fetch-Dest` values → 200, and headless Chrome renders the
preview build in full). So do not trust the cause, only the symptom: an empty `#root` is JS that
never ran, and a wrong base path is the likelier reason. The `http.server` recipe above stands on
its own merits.

## Reporting numbers

The universe is 132 symbols; **116** survive the common-window filter, and the 16 dropped funds
are named in the manifest with the reason. Measured shrinkage intensity ≈0.006 — quote it, do not
claim shrinkage is what makes the answer trustworthy. It is not: the expected returns are the weak
link, and the three weight caps are the honest read on that.

**Read the window off `web/public/data/manifest.json`, do not restate it here.** `update-data`
runs every Saturday, so the end date, the trading-day count and the year count all move weekly;
a number written into prose is stale by the next cron and there is nothing that fails when it
drifts. This section said "2026-09-02, 3,939 trading days" one day after the first scheduled run
made it 2026-09-03 and 3,941. `manifest.window` carries `start`, `end`, `trading_days` and
`years`, and both test suites read it.
