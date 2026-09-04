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
  tests/                  invariants + _mutate.py, the mutation harness
  data/prices/            the store. gitignored today, but see .gitignore -- that is now a choice
web/                      Vite + React 19 + TS, no chart library
  public/data/*.json      COMMITTED. This is the deployment artifact.
  src/portfolio.ts        the browser's own maths -- interpolation, tangency, growth
  src/export.ts           the portfolio download: CSV + JSON, serialised in the browser
  src/config.ts           the view state: URL fragment, localStorage, a file on disk
```

## Constraints that are decisions, not accidents

- **No AWS. No server of any kind.** Not even serverless. The pipeline commits JSON and Pages
  serves it. If a task seems to need a backend, the answer is almost certainly to precompute more
  and ship it in the JSON.
- **The GitHub side is not live yet.** `.github/workflows/*.yml` are scaffolded but no repo has
  been created, nothing has been pushed, and Pages is off. Do not create a remote, push, or
  enable Pages without being asked.
- **`rf` is browser-side, never a pipeline parameter.** The frontier does not depend on it. Any
  change that makes the pipeline emit per-rf files is a regression in the design, not a feature.
- **The weight cap IS a pipeline parameter** (a constraint on the QP), so each cap is its own
  solve and its own file.
- **Weights are interpolated, never curves.** The browser takes convex combinations of solved
  long-only portfolios, so every draggable position is genuinely feasible. Do not replace this
  with a spline through the frontier points — the numbers in the weight table would stop
  describing the point on the chart.
- **The store holds `close` + an event log, never `adjclose`.** Not a storage detail — the
  dividend adjustment is *backward*, so an adjusted column is retroactive and can never be
  append-only, and year-partitioning it would buy nothing. Total returns are rebuilt at load by
  `store.total_return()` and checked against the vendor's own field to ~1e-6. A change that
  stores the adjusted series undoes the entire point of the layout. The reverse mistake is
  equally bad: **do not materialise the adjusted panel** to disk. It is 0.35 s to derive, it is
  retroactive, and a stored derivative can drift from the inputs it claims to summarise.
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
- **A universe is a file, not code.** `universe.py` loads and validates; the symbols live in
  `pipeline/universes/*.toml`. One store keyed by symbol serves all of them, so a symbol two
  universes share is fetched once. Adding a universe must not require touching `universe.py`.

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

`pipeline/tests/_mutate.py` is the reason to trust the pipeline suite. It stages a copy of the
package, applies one source-text edit, and asserts a *named* test catches it; a mutation whose
pattern no longer matches is a hard error, because a stale mutation would be reported as caught
while testing nothing. It has already found four tests that guarded less than their names claimed
(`_prune_dominated`'s `while`-vs-`if`; `_clean`'s rounding; "the last bar is never adjusted
*whatever* the dividend history", which had no case with a dividend on the last bar; and "total
return compounds at least as fast as price return", whose `>=` was satisfied by deleting the
dividend adjustment outright). Run it after touching `frontier.py`, `fetch.py` or `store.py`:

```bash
python -u pipeline/tests/_mutate.py     # exits 1 if any mutation survives
```

It also found a claim the shipped data **cannot** exercise: all 116 survivors trade on exactly the
same 3,941 days, so `ffill().dropna()` and `dropna()` return the identical panel and the loudest
rule in `fetch.py`'s docstring was guarded by nothing. Guards for that class of property have to
be built on a synthetic store — see `test_load_panel_intersects_trading_days_rather_than_forward_filling`.
When a mutation survives, ask which of the two it is before touching a test.

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
- `.axis-title` is **not** uppercased: `text-transform: uppercase` renders "annualised σ" as
  "annualised Σ", the summation operator.

After any chart change, render it and look at it — `npm run build && npx vite preview`, then
screenshot both themes. The palette validator checks colour, not layout.

## Reporting numbers

The universe is 132 symbols; **116** survive the common-window filter, and the 16 dropped funds
are named in the manifest with the reason. Window: 2011-01-03 → 2026-09-02, 3,939 trading days,
15.6 years. Measured shrinkage intensity 0.0058 — quote it, do not claim shrinkage is what makes
the answer trustworthy. It is not: the expected returns are the weak link, and the three weight
caps are the honest read on that.
