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

`pipeline/tests/_mutate.py` is the reason to trust the pipeline suite. It stages a copy of the
package, applies one source-text edit, and asserts a *named* test catches it; a mutation whose
pattern no longer matches is a hard error, because a stale mutation would be reported as caught
while testing nothing. It has already found four tests that guarded less than their names claimed
(`_prune_dominated`'s `while`-vs-`if`; `_clean`'s rounding; "the last bar is never adjusted
*whatever* the dividend history", which had no case with a dividend on the last bar; and "total
return compounds at least as fast as price return", whose `>=` was satisfied by deleting the
dividend adjustment outright). Run it after touching `frontier.py`, `fetch.py` or `store.py`:

```bash
python -u pipeline/tests/_mutate.py     # 32 mutants, ~4.5 min; exits 1 if any survives
node web/tests/_mutate.mjs              # 38 mutants, ~40 s; --only <substring>, --list, --keep
```

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
instead of after a deploy. Two things in the tool that look like slack and are not: the viewport is
emulated at a *realistic* 1200px and the full-page height is measured from `scrollHeight`
afterwards, because emulating a 4200px-tall window makes `scrollHeight` report 4200 whatever the
content is; and `/favicon.ico` 404s are the one HTTP failure it ignores, because Chrome requests
that at the origin root on its own whether the page asks or not.

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
