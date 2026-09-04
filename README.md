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
default: 132 symbols, 116 of which survive the common-window filter.

```bash
python -u pipeline/build.py --universe etf_global --out web/public/data
```

Adding a universe is adding a file. The store is keyed by symbol, so a symbol two universes share
is fetched once, and `universe.load()` validates on the way in (every asset has a name, class and
group; the group is one of the declared groups; the benchmark is a member; nothing is both
included and excluded) — a typo fails the build instead of quietly shipping an unlabelled dot.

## What the data files are

| File | Contents |
|---|---|
| `manifest.json` | Universe, window, per-asset return/vol/Sharpe, exclusions and why, estimation metadata |
| `stats.json` | `mu` and the full shrunk covariance matrix — the browser needs it to recompute risk |
| `history.json` | Monthly total returns per asset, for the growth curve |
| `frontier_cap{100,20,10}.json` | 62 solved points each: weights, return, vol, Sharpe, plus the min-variance and max-Sharpe indices |

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

## Tests

Both suites test the shipped artifacts rather than fixtures, so they fail when the *data* is
wrong and not only when the code is.

- `pipeline/tests/` — invariants on the JSON: the panel is a real common window with no
  forward-fill, annualisation is over returns at 252, weights sum to one **at the precision they
  ship at** and respect the cap, the frontier is monotone, the monthly series telescopes to the
  same total growth the optimiser measured. Plus the store's own: a year partition holds only its
  year, re-writing unchanged history touches no file, and the reconstructed total return matches
  the vendor's `adjclose` at the 24 sampled bars per symbol.
- `pipeline/tests/_mutate.py` — a mutation harness: 32 mutants, each one source-text edit (drop
  the monthly anchor, annualise at 365, forward-fill the panel, never apply the dividends, adjust
  the ex-date bar too, partition on the run date, use arithmetic means, ship a frontier file with
  no run stamp, accept an asset in a group the universe never declared, …), each asserted to be
  caught by a *named* test. It exists because "89 passed" is not evidence, and it has found four
  tests that guarded less than their names claimed and one rule the shipped data cannot exercise
  at all.
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
