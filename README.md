# Markowitz

An interactive efficient frontier over 116 cross-asset ETFs. A Python pipeline solves the
mean-variance problem and writes JSON; a React SPA reads that JSON and lets you drag a handle
along the frontier to pick a portfolio.

Two subprojects, one artifact between them:

```
pipeline/  fetch prices -> estimate mu and Sigma -> solve the QP -> write JSON
                                                                      |
web/       React + TypeScript SPA  <---- web/public/data/*.json <------+
```

## Why there is no server

The frontier is expensive to solve and cheap to *read*: three solved frontiers, 116 assets and
15+ years of monthly history come to 432 KB of JSON. So the pipeline runs on a schedule, commits
its output, and the SPA fetches static files. **The repo is the database.** No AWS, no Lambda, no
S3, no DynamoDB, and nothing to keep running or pay for.

What makes that work is that the **risk-free rate is not baked into the frontier**. The frontier
solves "minimise variance subject to a return target", in which `rf` does not appear; only the
tangency portfolio and the capital market line depend on it, and both are recoverable from the
frontier the browser already has. So `rf` is a slider that re-derives them exactly, with no
re-solve and no refetch. The weight cap is genuinely a constraint on the QP, so each cap is a
separate solve and a separate file — which is why it is a three-way selector, not a slider.

Taking a portfolio away is client-side for the same reason: the CSV and JSON downloads under the
weight table are a `Blob` and an anchor click, with no endpoint to render them. The CSV body is
plain `symbol,name,group,weight` for pasting into a spreadsheet and carries the cap, the rf and
the window in leading `#` lines; the JSON carries the full estimation metadata, so the file states
what it is without the page it came from.

The reader's own settings work the same way, with three durabilities and no account. The cap,
the rf, the handle position, the group filter and the theme live in the **URL fragment** — so
"Copy link" shares an exact view, and a fragment is never sent to a server — are mirrored into
`localStorage` so a reload comes back where you left it, and can be saved to a JSON file you own.
On load the URL wins, then storage, then the defaults: a shared link has to render the same for
the recipient as for the sender. Nothing is uploaded, nothing is tracked, and there is nowhere for
it to go. See `web/src/config.ts`.

The second thing that makes it work: **the browser interpolates weights, not curves.** Dragging
between two solved points takes a convex combination of two long-only portfolios, which is
itself feasible; return is then exact (linear in the weights) and risk is exact from the shipped
covariance via a three-scalar segment quadratic. Every position on the curve is a portfolio you
could actually hold, not a point on a fitted line. See `web/src/portfolio.ts`.

## Running it

```bash
# Pipeline (Python 3.11, pyenv virtualenv "markowitz" via .python-version)
pip install -r pipeline/requirements.txt
python -u pipeline/build.py                    # fetch, solve, write web/public/data/*.json
python -m pytest pipeline/tests -q             # invariants on the artifacts just written

# SPA
cd web && npm ci
npm run dev                                    # http://localhost:5173
npm test                                       # browser-side maths, against the committed JSON
npm run build
```

To look at the built page rather than the dev server — which is the step a chart change needs,
since a palette validator cannot see a label collision:

```bash
cd web && npm run build
mkdir -p /tmp/serve && ln -sfn "$PWD/dist" /tmp/serve/markowitz
(cd /tmp/serve && python3 -m http.server 4455) &          # /markowitz/, matching Pages
node tools/shoot.mjs http://localhost:4455/markowitz/ /tmp/light.png '' light
node tools/shoot.mjs http://localhost:4455/markowitz/ /tmp/dark.png  '.chart' dark
```

`tools/shoot.mjs` is a headless-Chrome screenshotter over the DevTools protocol, no dependencies.
It prints the console and any failed request and exits nonzero if the page complained, which is
the part that matters: a chart that fails to mount and a chart that mounts wrong produce the same
blank PNG. With a selector it clips to one element (opening any `<details>` first); without one,
the whole page at its real scroll height.

`build.py --skip-fetch` reuses the local price store in `pipeline/data/prices/` (gitignored,
3.7 MB, reproducible). Useful flags: `--universe`, `--out`, `--symbols` for a subset, `--caps`,
`--points`, `--start`, `--rf auto|0.042`.

## The price store, and why it is shaped like that

`pipeline/data/prices/` is one store keyed by symbol, not one file per universe:

```
close/<year>.parquet     symbol, date, close   -- one file per calendar year
events.parquet           the dividend and split log, by ex-date
adjclose_sample.parquet  24 vendor adjclose bars per symbol, for auditing the reconstruction
```

Two decisions in there, and the second is the reason the first works.

**Partitioned by year** so a refresh rewrites one file instead of all of them. Parquet is
compressed binary and git stores it whole, so a rewritten file is a whole new blob — under the
old one-file-per-symbol layout a weekly cron would have added 14 MB to the history every week to
record 250 new bars. Only partitions whose *content* actually changed are written, which matters
because the collector has no incremental endpoint: it refetches each symbol's full history every
run and hands most partitions a chunk identical to what they already hold.

**Storing `close` plus dividends rather than `adjclose`**, because partitioning an adjusted price
by year saves nothing. The dividend adjustment is applied *backward* — a distribution with an
ex-date this week multiplies every earlier bar of that symbol by `(1 − D/P)` — so one dividend
rewrites a whole history, and across ~120 payers something has an ex-date most weeks. `close` and
an event log are genuinely append-only; the total-return series is rebuilt at load time by
`store.total_return()` and matches Yahoo's own `adjclose` to ~1e-6 relative, which is the
precision that field ships at (float32). Measured churn: ~216 KB per weekly run.

## Universes

A universe is a TOML file in `pipeline/universes/` — symbols with group tags, the benchmark, the
group order, and exclusions *with the evidence for excluding them*. `etf_global.toml` is the
default: 132 symbols, 116 of which survive the common-window filter. `sp500_2023.toml` is the other
one that is used for a published number — the S&P 500 as it stood on 2023-08-30, 484 assets, for the
[single-stock backtest](#single-stocks-chosen-as-of-the-cutoff--where-the-answer-changes). It is a
`backtest.py` universe and is not built for the page: the price endpoint carries neither a sector nor
a company name, so it is one group and ticker-as-name, and the file records only what was read.

`sp500_pit.toml` is a third shape and it is not a universe of assets at all — it is **ten
memberships**, one per rebalance date from 2016 to 2025, plus the three ticker-identity tables
(`[recycled]`, `[reissued]`, `[truncated]`) that say which symbols do not mean what they appear to.
`pipeline/pit.py` is its only reader; `universe.load()` never sees it, and neither does `build.py`.
It exists because a single snapshot can only be read forward of its own date — see
[ten cutoffs](#ten-cutoffs-membership-rebuilt-at-each-one--where-the-edge-disappears).

```bash
python -u pipeline/build.py --universe etf_global --out web/public/data
```

Adding a universe is adding a file. The store is keyed by symbol, so a symbol two universes share
is fetched once, and `universe.load()` validates on the way in (every asset has a name, class and
group; the group is one of the declared groups; the benchmark is a member; nothing is both
included and excluded) — a typo fails the build instead of quietly shipping an unlabelled dot.

`benchmark = ""` is how a universe says it deliberately has none, which is not the same as omitting
the key: TOML has no null, and a universe of single stocks *must* not carry an index fund among its
candidates. An index fund has lower variance than almost any one of its own constituents, so the
minimum-variance solve buys it on sight, and "did mean-variance beat the index" would then be a
question about a portfolio that is allowed to *be* the index. `backtest.py --benchmark SPY` supplies
one that is **priced but not investable** — in the panel so its returns and its forecast volatility
can be measured, out of the candidate set so nothing can hold it.

## What the data files are

| File | Contents |
|---|---|
| `manifest.json` | Universe, window, per-asset return/vol/Sharpe, exclusions and why, estimation metadata |
| `stats.json` | `mu` and the full shrunk covariance matrix — the browser needs it to recompute risk |
| `history.json` | Monthly total returns per asset, for the growth curve |
| `frontier_cap{100,20,10}.json` | 63 points each: weights, return, vol, Sharpe, plus the min-variance and max-Sharpe indices |

## Method, and where it is weak

Expected returns are the sample **geometric** mean, annualised at 252 trading days. Risk is a
Ledoit-Wolf shrunk covariance matrix from daily returns; the measured shrinkage intensity is
≈0.006 — nearly inactive, because at ~3,900 daily returns for 116 assets the sample covariance
needs little help. Long-only with a box constraint, solved as a QP through PyPortfolioOpt/cvxpy.

The weak link is the expected returns, and no amount of covariance care fixes it: the standard
error on a 15-year annualised mean for a 20%-volatility asset is about 5 percentage points, the
same order as the equity risk premium being estimated. **That is what the three weight caps are
for** — how far the portfolio moves between 100% and 10% is a direct read on how much of the
"optimal" answer was estimation error.

Shrinkage also cannot fix a near-exact linear dependency, and this universe has several by
construction: UUP is short the basket FXE/FXY/FXB are long, IAU and GLD hold the same bullion.
The minimum-variance portfolio duly nets a currency basket down to under 1% volatility. That is
the optimiser exploiting a collinearity, not a low-risk portfolio anyone found — it is left in
view because seeing it is the point.

Prices are total returns — split-adjusted closes with the stored dividend log applied backward
over each symbol's full history, reconstructed at load rather than stored (see the price store
above). No transaction costs, no taxes, no slippage. The growth chart is in-sample and is not a
backtest. Not investment advice.

### The out-of-sample check, which does not ship

`pipeline/backtest.py` is the missing half of that: estimate on everything up to a cutoff, take
the T-bill yield of that day, solve the tangency portfolio, then hold it forward and measure. It
writes `pipeline/results/backtest.json` and nothing into `web/public/data` — the page's contract
is exactly six files under one `generated_at`, and a seventh would have to be either inside that
invariant or exempt from it.

It runs in three modes, and the second exists because the first cannot answer the question.

```bash
python -u pipeline/backtest.py            # fixed cutoffs + rolling, the defaults below
python -u pipeline/backtest.py --years 1,2,3 --caps 1.0,0.2,0.1 --rf auto     # fixed cutoffs
python -u pipeline/backtest.py --train-years 5 --hold-years 1,2 --span-years 10 --no-rolling
python -u pipeline/backtest.py --window --train-years 4 --cutoff-years-back 3 --window-holds 1,2,3
```

**Fixed cutoffs** (`--years`) train on everything before a cutoff 1, 2 or 3 years back and hold to
the last bar. Three of them are not three trials: the 3-year window contains the 2-year window,
which contains the 1-year one, so it is one observation with three end dates. It flattered the
method — the tangency portfolio beat the benchmark at all three cutoffs — and that was a fact
about which three end dates the panel happens to have.

**One stated window** (`--window`) is the same nested shape with both ends pinned instead of
inherited: `--train-years` behind a cutoff `--cutoff-years-back` years back, held forward for each
of `--window-holds`. It is the shape of "train on 2019–2023, then how did it do over the next 1, 2
and 3 years", and it is kept for a question the rolling mode does not answer — how one portfolio's
lead over the index evolves as it is held. The rows still share one training window and one weight
vector per strategy, so three rows beating the benchmark is one win. A hold that would run past the
last bar raises rather than being reported at its requested length and being shorter.

**Rolling** (`--train-years/--hold-years/--span-years`) is the honest version: a *fixed-length*
5-year training window walked forward in **non-overlapping** holding periods, so ten one-year
holds over the last decade are ten separate results that can come out 6–4. They came out 4–6. At
cap 100% the portfolio compounded 9.9%/yr against the benchmark's 15.7% and an equal-weight basket
of the same 116 assets at 9.4%; £100 became £257 where the benchmark made £429. It beat the
benchmark in 4 of 10 years, and at caps 20% and 10% in 3 of 10. Two-year holds (five periods) are
worse: 9.4%/yr against 15.9%, ahead in 1 of 5. What it did do is arrive with much smaller
drawdowns — a worst year of −6.3% against the benchmark's −12.2%.

What survives is not the return column but the **cross-sectional rank correlation between what the
training window said and what the test window did**, averaged over the ten periods: **+0.25 for
return, +0.91 for volatility**. Risk persists and is worth estimating; the expected returns the
optimiser sorts on are close to noise at this horizon. That is the same conclusion the weight caps
are there for, arrived at from the other direction. At a 5-year window the shrinkage intensity is
also genuinely active (0.0159, against ≈0.006 on the shipped 15-year estimate).

#### Comparing at the same risk, which is the comparison that settles it

Those return columns are not comparable as they stand: the tangency portfolio carried ~7.6%
annualised volatility against SPY's ~16.8%, so most of the shortfall is a difference in how much
risk was taken rather than in how well anything was chosen. `matched_risk@cap*` and
`tangency_levered@cap*` remove that. Both aim at the **benchmark's forecast volatility from the
same training window** — the realized volatility over the test window is the tempting target and is
not knowable at the cutoff — the first by solving the frontier for the most return available at
that risk (long-only, no borrowing), the second by scaling the tangency portfolio up the capital
market line, financed at the T-bill rate as it moved.

| Over ten 1-year holds, cap 100% | Compounded | Risk it ran | Return per unit | Worst drop |
|---|---|---|---|---|
| Long-only at the benchmark's risk | 17.6% | 19.1% | 0.92 | 31.9% |
| Tangency levered to it (mean 3.4×) | 15.1% | 20.6% | 0.73 | 44.2% |
| Just holding SPY | 15.7% | 16.8% | 0.93 | 33.7% |

**No edge.** The long-only version's extra 1.9 points of return are bought with 2.3 points of extra
risk at the same ratio the index already offered, and the levered version is worse than the index
outright — 3.4× leverage at the bill rate, reset daily for free, and it still loses. At 2-year
holds the long-only version falls to 0.77 against 0.91. Caps 20% and 10% land at 0.88 and 0.92.

The reason the risk columns overshoot is worth more than the return columns. Split the ten periods
by whether the optimiser *chose* the portfolio:

| | Forecast risk | Realized risk | Ratio |
|---|---|---|---|
| SPY | 17.1% | 16.8% | 1.03 |
| Equal-weight, all 116 | 12.7% | 12.3% | 1.01 |
| Tangency | 6.9% | 7.6% | 1.25 |
| Long-only at the benchmark's risk | 17.1% | 19.1% | 1.15 |

On portfolios it was merely asked to *describe*, the covariance matrix is accurate to about 3%. On
the ones it was asked to *build*, realized risk runs 12–25% above forecast — the QP selects the
assets whose estimated variances and correlations are lowest, and the lowest estimates are
disproportionately the errors. So "the covariance half works" needs that qualifier: it is a good
description and a biased optimisation input, and the bias runs the one direction that flatters the
result.

#### Trying a better return forecast, which did not help

+0.25 is close enough to a coin flip that the obvious question is whether a different forecast does
better. `--mu momentum` answers it for the one with the strongest published record — each asset's
return over the past 12 months, **stopping a month short**, because the most recent month reverses
and omitting the skip is the standard way a momentum backtest is quietly built to succeed.

```bash
python -u pipeline/backtest.py --mu momentum --out pipeline/results/backtest_momentum.json
python -u pipeline/backtest.py --mu momentum --mom-lookback 6 --mom-skip 1 --mom-top 20
```

Everything else is held fixed: same Ledoit-Wolf covariance, same caps, same solver, same ten
periods, same matched-risk comparison. One input differs, so the diagnostics compare directly.

| Over ten 1-year holds | Ranked next year's returns | Long-only at the benchmark's risk | Return per unit |
|---|---|---|---|
| Average return over the training window | **+0.25** | 17.6% at 19.1% | **0.92** |
| 12-1 momentum | **+0.07** | 10.1% at 19.1% | **0.53** |
| Just holding SPY | — | 15.7% at 16.8% | 0.93 |

Momentum is *worse*. +0.07 is a coin flip, and feeding a near-random ranking to the optimiser buys
the index's risk for two-thirds of its return. The `momentum_top10` row — the signal held
equal-weight with no optimiser at all — came out 0.62, so the signal was weak here before
mean-variance touched it; that row ships under **both** forecasts, which is also the check that the
two runs differ in exactly one input.

Two reasons not to read this as a verdict on momentum. A weight vector is chosen **once per holding
period**, and `pd.DateOffset` refuses a fractional year, so the shortest walk this harness can do
refreshes annually while the published effect is monthly — by month 11 the ranking is nearly two
years stale. And the universe is 116 mutually 0.8–0.95-correlated sector and asset-class funds,
not the individual stocks the effect is documented on. Both are in `method.caveats` in the JSON.

The risk forecast is +0.91 under both. That number has not moved through any of this.

Consecutive training windows still overlap by `train_years − hold_years`, and every asset in the
universe is one that still exists today with quotes back to 2011, so all of them survived the
period being tested. The comparators are in the table for that reason: an equal-weight basket of
the same assets and the benchmark are what the machinery has to beat before any of it has earned
anything.

#### Single stocks, chosen as of the cutoff — where the answer changes

> **Superseded.** This section is one cutoff and three holds. The next one is ten cutoffs with the
> index membership rebuilt at each of them, and it reverses the conclusion. Read both — the
> difference between them is the whole lesson — but do not quote this section's 1.79 as the answer.
>
> The training window here is four years. `backtest_sp500_train5.json` is the same run at **five**,
> which is the length the ten-cutoff test uses: `matched_risk@cap100` 29.95%/yr at 17.04% = 1.76
> against SPY's 22.17% at 14.94% = 1.48 over the same three holds. So the window length is not what
> the next section changes — the number of cutoffs is.

Everything above is 116 broad funds that are 0.8–0.95 correlated with each other. Mean-variance has
much less to work with there than it does across 500 individual companies, and the survivorship
caveat in the paragraph above is unanswerable while the universe is "funds that still exist". So the
same machinery was pointed at `sp500_2023.toml` — **the 503 lines that made up the S&P 500 on
2023-08-30, as the index stood that day**, not as it stands now. A constituent list downloaded today
already knows which companies did well enough to still be in the index.

```bash
python -u pipeline/backtest.py --universe sp500_2023 --benchmark SPY --start 2019-08-01 \
  --years 3 --window --train-years 4 --cutoff-years-back 3 --window-holds 1,2,3 \
  --hold-years 1 --span-years 3 --out pipeline/results/backtest_sp500.json
```

478 of 486 assets survive the common window, 477 are investable, and SPY is priced without being one
of them. Because the membership list is only clean *forward* of its as-of date, the non-nested test
is three one-year holds and no more — a rolling window reaching back before 2023-08-30 would be
selecting its candidates with knowledge of what followed. Compounded over those three holds, with
return per unit of the risk each strategy **actually ran**:

| Over three 1-year holds | Compounded | Risk it ran | Return per unit | Beat SPY |
|---|---|---|---|---|
| Long-only at the benchmark's risk, cap 10% | 28.9% | 16.2% | **1.79** | 3 of 3 |
| Tangency, cap 10% | 26.9% | 14.9% | **1.80** | 2 of 3 |
| Tangency, cap 100% | 28.6% | 19.3% | 1.48 | 2 of 3 |
| Equal-weight, all 477 | 16.3% | 13.9% | 1.17 | 0 of 3 |
| Just holding SPY | 22.3% | 14.9% | 1.49 | — |

That is an edge, where the ETF universe had none (1.79 against 1.49, versus 0.92 against 0.93). It
is also **three observations**, and the return forecast still ranks next year's winners at only
**+0.13** — the risk forecast at +0.74 is again the half that works. The third hold ends at the
panel's last bar, so every figure in that table moves when the file is regenerated; it carries its
own `panel.end`. Read the caveats as part of the result, not as hedging:

- **The training window contains 2020.** SPY's *forecast* volatility from 2019-08 to 2023-09 was
  23.08%, against the 15.34% it went on to realize. So "matched to the benchmark's forecast
  volatility" meant aiming above the risk the index actually ran in two of the three periods —
  15.4%, 19.3% and 13.7% against SPY's 12.4%, 19.6% and 12.8%. The forecast is what was knowable at
  the cutoff and aiming at the realized number would be a lookahead — but it means the raw return
  columns are not the comparison. The per-unit column is.
- **17 of the 503 are not in the price store**, because the vendor serves nothing at all for a
  symbol that stopped trading rather than a series that ends. Twelve of them have since been
  measured from a second vendor and it changed nothing — [see below](#what-happened-to-the-ones-that-were-delisted).
- **Six tickers changed and one was recycled.** BK→BNY, FI→FISV, FLT→CPAY, MMC→MRSH, PARA→PSKY and
  PEAK→DOC are the same company renamed, each verified by price level years before the change rather
  than by the rename (WRK→SW failed that check — SW carries Smurfit Kappa's history, not WestRock's —
  and is excluded). The ticker `PARA` is the dangerous one: it fetches a clean full series and is
  **Banzai International**, not Paramount Global, which merged into Paramount Skydance and is held
  as PSKY. Every automatic check passes on a recycled ticker; it surfaced only because the new
  occupant's reverse splits tripped the store's split test. There is no automatic screen — the check
  is the vendor's current company name against the constituent it should be.
- The `--window` rows in the same file (cap 20%: 38.0% against SPY's 26.9% at one year) are **nested
  by construction** — one weight vector reported at three end dates — and are not three wins.

#### What happened to the ones that were delisted

The bullet above used to end with an argument: takeovers happen at a premium, so the unpriceable
names are disproportionately winners and leaving them out most likely understates the result — 3.6%
of an equal-weighted basket, +0.18 to +0.47 pp/yr. That is an argument, not a measurement, and when
it was measured it came out with **the opposite sign**.

A second vendor (stockanalysis.com) still serves 12 of the 17. `pipeline/delisted.py` fetches them,
caches the raw CSVs in the repo (dead tickers are not refetchable), states what happened to each,
and stitches each one to the end of the panel under a declared assumption about the proceeds. Then
the whole test runs again with them restored:

```bash
python -u pipeline/delisted.py --refresh          # fetch + cross-check against the store
python -u pipeline/backtest.py --universe sp500_2023 --benchmark SPY --start 2019-08-01 \
  --years 3 --hold-years 1 --span-years 3 --extra-panel pipeline/data/delisted/panel_cash.parquet \
  --out pipeline/results/backtest_sp500_delisted_cash.json
python -u pipeline/delisted.py --stress pipeline/results/backtest_sp500_delisted_cash.json
```

From the cutoff to each name's own last traded bar, the twelve returned **+21.8% against SPY's
+46.4%** over the same sub-spans, and only **4 of 12** beat the index — WBA −90.5pp, IPG −69.9pp and
CDAY −64.9pp against CMA's +57.2pp. Being dropped from the S&P 500 is not a synonym for being taken
over at a premium. Restoring all twelve moves the equal-weighted comparator by **−0.02 to −0.18
pp/yr** (depending on whether the proceeds sit in T-bills or go into SPY), and it leaves the
conclusion exactly where it was: `matched_risk@cap10` is 1.79 against SPY's 1.49 in the baseline and
in both restored runs, over the same 1,782 bars, and **no tangency or matched-risk portfolio held any
of the twelve in any period under either treatment**.

Three things about how that is done, because each of them is a way the number could have been made up
instead:

- **The second vendor never enters the price store.** The store's guarantees rest on one vendor
  shipping closes, an event log and its own adjusted series to check the reconstruction against; this
  one ships a level and nothing else. It lives in `pipeline/data/delisted/` and reaches the optimiser
  only through `--extra-panel`, which joins on the panel's own trading days and refuses to move its
  window. The restored run is the baseline run plus twelve columns, bar for bar.
- **The last traded bar of every delisted series is dropped.** It is an index-deletion auction or a
  stub — three of the twelve print volume 0 or 1. Where an independent check exists it sometimes
  fails: CMA tracked 1.89542 × Fifth Third for eleven straight bars to within ±0.10% and then printed
  its final bar 5.34% below that ratio, while IPG tracked 0.35386 × Omnicom to ±0.21% *including* its
  last. The uniform rule is the one that needs no case list, and what it cost each symbol is written
  into `provenance_*.json` rather than asserted to be small.
- **The five nobody prices are bounded, not ignored.** `--stress` writes what they would have to have
  done to overturn the result. Equal-weighted, the whole range from "all five went to zero" to "all
  five did as well as the best outcome anywhere in the panel" is 0.565 pp/yr. For `matched_risk@cap10`
  at 1.77 to fall to SPY's 1.48, the five would need to end 10.6% short — and at a 10% cap they can
  be at most half the portfolio, so 21.2% worse than everything else over three years, while being
  picked as the lowest-risk highest-forecast-return names available.

One artefact, recorded so it does not get reported as a finding: `min_variance@cap10` *improves* when
the twelve are restored (1.34 → 1.51 or 1.61). A stitched continuation is smooth, so it has
artificially low variance, which is precisely what a minimum-variance solve reaches for. It is the
only strategy that ever buys one of them.

The 477-asset run is also what turned up a solver bug worth knowing about: PyPortfolioOpt's default
QP solver is OSQP, and at that size it exhausts its iteration budget and returns a status
indistinguishable from "this target is unattainable". `frontier.SOLVER` now names CLARABEL, which
solves the same problem in 0.1 s. Re-solving the shipped frontiers under it moved weights by at most
6.7e-4 and slightly improved the cap-100 minimum variance, so the published numbers were never
wrong — but at 116 assets nothing in the repo would have noticed the day a dependency changed its
default.

#### Ten cutoffs, membership rebuilt at each one — where the edge disappears

The section above has one honest complaint against it: three observations. `sp500_2023.toml` is the
index as it stood on one day, so every window it permits ends after that day, and the cleanest read
it allows is three non-overlapping years. Three years is not enough to tell an edge from a market.

`pipeline/universes/sp500_pit.toml` is the same idea done ten times: the S&P 500's membership **as it
stood on each of ten rebalance dates**, 2016-09-01 through 2025-08-29, reconstructed from a change
log. `pipeline/pit.py` walks it — a fixed **5-year** training window behind each cutoff, one
non-overlapping **1-year** hold in front, the candidate set rebuilt from *that date's* snapshot, SPY
priced on every bar and investable on none:

```bash
python -u pipeline/pit.py --train-years 5 --hold-years 1 --from-year 2016 --to-year 2025 \
  --caps 1.0,0.2,0.1 --benchmark SPY --out pipeline/results/backtest_pit.json
```

Compounded over the ten holds, per unit of the risk each strategy actually ran:

| Over ten 1-year holds | Compounded/yr | Risk it ran | Return per unit | Beat SPY |
|---|---|---|---|---|
| Just holding SPY | **15.31%** | 16.88% | **0.907** | — |
| Long-only at the benchmark's risk, cap 100% | 14.42% | 19.22% | 0.750 | 5 of 10 |
| Long-only at the benchmark's risk, cap 20% | 13.93% | 19.03% | 0.732 | 5 of 10 |
| Long-only at the benchmark's risk, cap 10% | 12.93% | 18.97% | 0.682 | 4 of 10 |
| Equal-weight, every investable member | 12.77% | 16.95% | 0.753 | 2 of 10 |
| Minimum variance, cap 100% | 9.80% | 13.10% | 0.748 | 2 of 10 |
| Tangency, cap 100% | 12.92% | 21.37% | 0.605 | 4 of 10 |
| Tangency levered to the benchmark's risk, cap 100% | 13.88% | 19.78% | 0.702 | 5 of 10 |
| Momentum, top 10 equal-weight | 17.69% | 27.43% | 0.645 | 6 of 10 |

**Every optimised strategy at every cap is below the index per unit of risk.** The two columns that
look like wins are the two to distrust: `matched_risk@cap100` beat SPY's return in 5 of 10 periods
while running 19.2% volatility against the index's 16.9%, and momentum's 17.69% — the highest raw
return in the table — was earned at 27.4% volatility. This is the forecast-bias mechanism described
above, and it is worse here than on the ETF universe: the matched-risk target is the benchmark's
*forecast* volatility, which is what was knowable at the cutoff, and it overshot in most windows.

The column worth reading before any of those is the one the table cannot show. Averaged over the ten
cutoffs, the cross-sectional rank correlation between the forecast and what the next year actually
paid is **+0.06 for the return forecast and +0.72 for the risk forecast** — the same split as the ETF
result, measured on single stocks over ten independent years, and *narrower* on the return side than
the ETF universe's +0.25. Single companies have the idiosyncratic variance an index fund does not, so
this was the universe where a return forecast had the most room to find something; it found less.
The momentum score's own ranking is **−0.02**, slightly backwards, which is what makes
`momentum_top10`'s 17.69% the highest raw return in the table and the least defensible number in it.

The three-hold result in the previous section was not wrong. Its 2023 cutoff is one of the ten here,
and it is one of the good ones. Ten cutoffs is what a batting average needs.

What makes the point-in-time version harder than it sounds, in the order the mistakes bite:

- **A ticker is not an identity, and there are three different failures with three different
  remedies.** `[recycled]` — the symbol now belongs to another company entirely (`APC` is ARKO
  Petroleum, not Anadarko) — is stripped from **every** snapshot. `[reissued]` — someone else before a
  date, a real member after it (`DOW`, `FOX`, `FOXA`, `Q`) — is stripped only from snapshots **before**
  that date; applying the recycled rule to it would delete Dow Inc from seven snapshots of an index it
  was in, which is survivorship bias pointing the other way. `[truncated]` — right company, stub
  series — is stripped from **none**. `pit.overlaps_its_membership` is the general form of the check
  and it is not manual: the index held a ticker between two known dates, so a series that never
  reaches into that interval cannot be that constituent's history whatever it is named. It caught
  three more (`MON`, `CA`, `DNB`) — all clean, full-length, filter-passing series belonging to someone
  else.
- **The early windows are the thin ones, and that is inherent.** In the 2016 window 397 of 507 members
  are investable: 84 constituents are priced by nobody because they stopped trading a decade ago, and
  26 more begin after the training window opens. By the 2025 window it is 486 of 504 with 5 unpriced.
  The count is per period in the file, and it is the honest reason to read the later holds as stronger
  evidence than the earlier ones.
- **The bound over the missing names now runs the other way, and it does not close the gap.** For the
  ten-cutoff run, 99 constituents are priced by no vendor. Wiping all of them out takes equal-weight
  from 12.60%/yr to 10.12%; giving every one of them the best outcome observed anywhere in the panel
  takes it to 13.30% — a 3.18 pp/yr swing, and **the favourable end is still below the index's
  15.31%**. (12.60% and not the table's 12.77%: the bound annualises one growth factor over the
  9.999-year span from the first cutoff, the table compounds ten one-year period returns. Both are in
  their own files with their own spans, which is the only way to tell which run a number came from.) For `matched_risk@cap100` to reach SPY per unit of risk, the 99 would have had to
  *outperform* everything else by **29.7%**, and at cap 100% the bound on their exposure is vacuous,
  so that is the most generous form of the question. No tangency or matched-risk portfolio held a
  restored name in any period.
- **Costs are charged to both sides or to neither.** `--trade-bps 5` writes
  `backtest_pit_net5bp.json`: the strategies pay 5 basis points per dollar traded on a buys-plus-sells
  turnover, and SPY pays its published 9.45bp expense ratio, because measuring the optimiser's
  frictions against a frictionless index is the bias the flag exists to remove. It moves SPY from
  0.907 to 0.897 and `matched_risk@cap100` from 0.750 to 0.745 — annual rebalancing barely trades, so
  **the gap is not a friction artifact**. `--trade-bps 0` is the default and reproduces a pre-flag run
  bit for bit.
- **Both sides are total returns.** Every column of the panel is `store.total_return()` — dividends
  reinvested at the close — SPY included. There is no version of this comparison where one side
  reinvests and the other does not.
- **Two caveats are stated and not corrected for**, and they are in `method.caveats` in the file. The
  holds do not overlap, but consecutive *training* windows overlap by four years, so a persistent
  regime is estimated the same way several times running; and a weight vector is chosen once per hold,
  so this is annual rebalancing — momentum in particular is documented at monthly refresh, and the
  `momentum_top10` row here is not a verdict on that.


## Tests

Both suites test the shipped artifacts rather than fixtures, so they fail when the *data* is
wrong and not only when the code is.

- `pipeline/tests/` — invariants on the JSON: the panel is a real common window with no
  forward-fill, annualisation is over returns at 252, weights sum to one **at the precision they
  ship at** and respect the cap, the frontier is monotone, the monthly series telescopes to the
  same total growth the optimiser measured. Plus the store's own: a year partition holds only its
  year, re-writing unchanged history touches no file, and the reconstructed total return matches
  the vendor's `adjclose` at the 24 sampled bars per symbol.
- `pipeline/tests/test_backtest_invariants.py` — the walk-forward test's own invariants, and one of
  two files here built on a **synthetic** panel rather than the shipped one. It has to be: 116
  assets trading on the same days cannot tell a monthly rebalance from a buy-and-hold, and no
  file on disk can tell an honest backtest from a leaking one. The test that matters perturbs the
  prices *and* the risk-free rate after the cutoff and requires the solved weights to be
  identical, because a backtest with lookahead in it produces numbers that are plausible, ordered
  the way you expect, and worthless. It runs under both `--mu` models, since momentum locates its
  two bars by date arithmetic from the cutoff and that is a second, independent way to end up a
  month on the wrong side of it. The momentum tests perturb a single bar or a span hanging off one
  edge of the window, never the middle: momentum reads exactly two prices, so a mid-window
  perturbation changes nothing and a test built that way would pass on any window at all.
- `pipeline/tests/test_delisted_invariants.py` — the second-vendor path, and the second synthetic
  file. Same justification: all twelve cached series have a bad-looking final bar and a clean one
  behind it, so nothing on disk distinguishes "drops the last bar" from "keeps it", and no committed
  panel has a column that clashes with the store, starts late, or has a hole in the middle. The
  load-bearing test is that joining a restored column **cannot move the panel's window** — a claim
  about what does not happen, which needs a panel whose window is known. One test there is not
  synthetic and reads the real cache: `range=MAX` returns one year with a 200 response, so the
  served span is a fact about the committed files rather than a parameter the code sets.
- `pipeline/tests/test_results_invariants.py` — the only test that opens `pipeline/results/*.json`,
  which are generated by hand and committed as evidence rather than rebuilt by CI. Its subject is
  the one field in those files that is a **join key rather than a label**: every printer looks its
  benchmark row up by `f"benchmark:{block['benchmark']}"`, so a block recording one symbol while its
  periods measured another compares against nothing. That is fatal in the rolling block and merely
  cosmetic in the window block, which prints `--` and writes the file — and the asymmetry is the
  reason the test exists. Its docstring is explicit that two of its three assertions cannot be
  violated by any run the CLI can perform; read it before adding to it.
- `pipeline/tests/test_pit_invariants.py` — the point-in-time walk, and the third synthetic file.
  A membership file has to be *invalid* to test the loader that rejects it; consecutive snapshots of
  the real index share ~95% of their names, so nothing on disk distinguishes "rebuilds the candidate
  set per cutoff" from "uses the first one ten times"; and both window-length assertions need a panel
  that starts inside the window it was asked for, which the shipped store never produces.
- `pipeline/tests/_mutate.py` — a mutation harness: 119 mutants, each one source-text edit (drop
  the monthly anchor, annualise at 365, forward-fill the panel, never apply the dividends, adjust
  the ex-date bar too, partition on the run date, use arithmetic means, ship a frontier file with
  no run stamp, accept an asset in a group the universe never declared, charge the trading cost one
  way instead of on both legs, strip a *reissued* ticker from every snapshot instead of the ones
  before its date, …), each asserted to be caught by a *named* test. It exists because "262 passed"
  is not evidence, and it has found five tests that guarded less than their names claimed and one
  rule the shipped data cannot exercise at all. The fifth was the window test above: `join_extra`
  pins the panel's days twice, so a single-edit mutant removing one pin was behaviourally neutral
  and survived a test that was nonetheless correct — the mutant now makes both edits.

  `--check` verifies every needle still matches its source exactly once and exits in seconds. Run it
  first, always: the per-mutant check aborts on whichever stale needle is reached first and hides the
  rest behind it, and four of them went stale together in one afternoon's refactoring.
- `web/src/export.test.ts` — the downloaded file, parsed back with an RFC 4180 reader: the
  weights match the table at the precision the file claims, the total is the one actually
  written rather than 1, the provenance lines carry the cap/rf/window, and a name containing a
  comma survives — a case the shipped universe cannot produce and so is built synthetically.
- `web/src/config.test.ts` — the view state through all three of its round trips: every field
  survives the URL fragment (including `pos: null`, "track the tangency", which is not the same
  as `pos: 0`), the URL beats storage beats the defaults, a config naming a cap this build does
  not have degrades and *says* what it changed, and a browser that throws on every storage access
  still draws a chart.
- `web/src/portfolio.test.ts` — the browser's own arithmetic: it reproduces every solved point
  at its integer index, checks the three-scalar segment quadratic against a brute-force
  quadratic form to 1e-12, and checks the closed-form tangency root against a dense scan at
  seven risk-free rates.
- `web/src/data.test.ts` — the load, with `fetch` serving the real `public/data` off disk: six
  files fetched separately can arrive from two different pipeline runs, and last week's
  `stats.json` against this week's frontier has exactly the right number of symbols in exactly
  the right order, so every count agrees and the page silently prices today's weights with last
  week's covariance. The one field that can see it is `generated_at`, and the refusal is tested
  one stale file at a time — a stale frontier is as likely as a stale `stats.json`.
- `web/src/viz.test.ts` — half behaviour, half a scan of `src/` itself. A group's display name
  used to be a hardcoded map in four files, two of them disagreeing about capitalisation; the
  labels now come from `manifest.group_labels`, and only a source scan can fail when someone adds
  a fifth copy, because a fifth copy renders correctly for *this* universe.
- `web/tests/_mutate.mjs` — the same instrument as `_mutate.py`, pointed at the other half: 38
  mutants over `config.ts`, `export.ts`, `data.ts` and `viz.ts` (write the default cap into a
  shared link, treat `pos: 0` as absent, let storage beat the URL, stop comparing the six run
  stamps, drop the CSV quoting, assert the weight total as 1, put a hardcoded group map back, take
  the rf slider off the basis-point grid, …), each asserted to be caught by a *named* test. 38 s,
  and it runs against a staged copy of `web/` so an interrupted run cannot leave a mutated source
  in the tree.

Neither harness runs in CI. They are tests **of** the test suites, run by hand after touching what
they cover; `_mutate.py` has already found five pipeline tests that guarded less than their names
claimed. A survivor means one of two things, and they need opposite fixes: a test that guards less
than it says, or a property the shipped data cannot exercise — the second needs a synthetic
fixture, not a weakened assertion.

```bash
python -u pipeline/tests/_mutate.py              # 85 mutants, ~37 min
python -u pipeline/tests/_mutate.py --only backtest    # the 39 backtest mutants, ~17 min
python -u pipeline/tests/_mutate.py --only delisted    # the 11 second-vendor mutants, ~5 min
node web/tests/_mutate.mjs                      # 38 mutants, ~40 s; --only <substring>, --list, --keep
```

## Deployment

Live at **https://benlvovsky.github.io/markowitz/**.

`.github/workflows/` holds both halves: `update-data.yml` (weekly cron, Saturday 07:00 UTC —
fetch, solve, test, commit the JSON) and `deploy-web.yml` (typecheck, test, build, publish to
Pages). `update-data` has no `push` trigger on purpose, so pushing code never spends 90 seconds
refetching prices; run it by hand with `gh workflow run update-data` when you want fresh data now.

Two of the three things that make this work are **repository settings rather than files**, and
neither workflow can set them for itself:

| Setting | Value | What breaks without it |
|---|---|---|
| Pages → Source | GitHub Actions | `configure-pages` fails with "Get Pages site failed" *after* a green build |
| Actions → Workflow permissions | Read and write | `update-data` fetches, solves and tests, then 403s on `git push` |
