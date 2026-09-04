import { useEffect, useMemo, useState } from 'react'
import {
  clearStored,
  defaults,
  initial,
  toHash,
  writeStored,
  type Config,
} from './config'
import { loadBundle } from './data'
import { downloadCsv, downloadJson, type Selection } from './export'
import {
  annualised,
  at,
  buildModel,
  buildPath,
  growth,
  maxDrawdown,
  realisedVol,
  tangency as findTangency,
  weightsAt,
} from './portfolio'
import type { Bundle, Group } from './types'
import { AssetTable } from './components/AssetTable'
import { Controls, FrontierLegend } from './components/Controls'
import { FrontierChart } from './components/FrontierChart'
import { GrowthChart } from './components/GrowthChart'
import { SettingsBar } from './components/SettingsBar'
import { StatTiles, tile } from './components/StatTiles'
import { WeightTable } from './components/WeightTable'
import { mult, pct } from './viz'

export default function App() {
  const [bundle, setBundle] = useState<Bundle | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadBundle().then(setBundle, (e: Error) => setError(e.message))
  }, [])

  if (error) {
    return (
      <div className="app">
        <h1>Efficient frontier</h1>
        <div className="error">
          <strong>The data files could not be loaded.</strong>
          <pre>{error}</pre>
          <p className="note" style={{ marginTop: 10 }}>
            The SPA reads six JSON files from <code>public/data/</code>, written by{' '}
            <code>python -u pipeline/build.py</code>. Run it once and reload.
          </p>
        </div>
      </div>
    )
  }
  if (!bundle) return <div className="app status">Loading the frontier…</div>
  return <Frontier bundle={bundle} />
}

function Frontier({ bundle }: { bundle: Bundle }) {
  const { manifest, stats, history, frontiers } = bundle

  /** Read ONCE, at mount: the URL fragment, else `localStorage`, else the defaults. Read once
   *  rather than subscribed to, because this component then owns the state and writes back --
   *  reacting to our own `replaceState` would be a loop. */
  const [start] = useState(() => initial(manifest, window.location.hash))

  const [capSlug, setCapSlug] = useState(start.cap)
  const [rf, setRf] = useState(start.rf)
  const [visibleGroups, setVisibleGroups] = useState<Set<Group>>(new Set(start.groups))
  const [fitFrontier, setFitFrontier] = useState(start.fitFrontier)
  const [logScale, setLogScale] = useState(start.logScale)
  /** `themeSet` distinguishes "the reader chose light" from "the OS is light". Only an explicit
   *  choice is saved, so a config made on a dark machine does not force dark on a light one. */
  const [themeSet, setThemeSet] = useState(start.theme !== undefined)
  const [dark, setDark] = useState(
    () =>
      start.theme !== undefined
        ? start.theme === 'dark'
        : (window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false),
  )
  /** Position along the frontier, as a FRACTION of the path rather than a point index.
   *
   * A fraction rather than an index because switching the weight cap switches to a different
   * frontier with a different point count, and a stored index would mean the handle jumped an
   * arbitrary distance. The fraction keeps it where the reader left it, which makes the cap
   * selector a genuine before-and-after comparison instead of a reset. */
  const [frac, setFrac] = useState<number | null>(start.pos)

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? 'dark' : 'light'
  }, [dark])

  const config: Config = {
    cap: capSlug,
    rf,
    pos: frac,
    groups: manifest.groups.filter((g) => visibleGroups.has(g)),
    fitFrontier,
    logScale,
    ...(themeSet ? { theme: dark ? ('dark' as const) : ('light' as const) } : {}),
  }
  const configKey = JSON.stringify(config)

  /** Persist to the URL and to `localStorage`, DEBOUNCED, which is not a performance nicety:
   *  dragging the handle changes `frac` on every pointer move, and Safari throttles
   *  `history.replaceState` to roughly 100 calls per 30 seconds before it throws. `replaceState`
   *  rather than `pushState` for the same reason -- a drag must not fill the back button with
   *  every position it passed through. */
  useEffect(() => {
    const id = window.setTimeout(() => {
      const cfg = JSON.parse(configKey) as Config
      writeStored(cfg, manifest)
      const hash = toHash(cfg, manifest)
      const { pathname, search } = window.location
      window.history.replaceState(null, '', `${pathname}${search}${hash ? `#${hash}` : ''}`)
    }, 250)
    return () => window.clearTimeout(id)
  }, [configKey, manifest])

  const applyConfig = (c: Config) => {
    setCapSlug(c.cap)
    setRf(c.rf)
    setFrac(c.pos)
    setVisibleGroups(new Set(c.groups))
    setFitFrontier(c.fitFrontier)
    setLogScale(c.logScale)
    setThemeSet(c.theme !== undefined)
    if (c.theme !== undefined) setDark(c.theme === 'dark')
  }

  const model = useMemo(() => buildModel(stats), [stats])
  const path = useMemo(() => buildPath(frontiers[capSlug], model), [frontiers, capSlug, model])
  const tangency = useMemo(() => findTangency(path, rf), [path, rf])

  const last = path.points.length - 1
  // Default to the tangency portfolio: it is the answer the construction exists to give.
  const t = frac === null ? tangency.t : frac * last
  const here = at(path, t, rf)

  const weights = useMemo(() => weightsAt(path, t, model), [path, t, model])
  const held = useMemo(() => new Map(weights.map((w) => [w.symbol, w.weight])), [weights])
  const bySymbol = useMemo(() => new Map(manifest.assets.map((a) => [a.symbol, a])), [manifest])

  const port = useMemo(() => growth(weights, history), [weights, history])
  const bench = useMemo(
    () =>
      manifest.benchmark
        ? growth([{ symbol: manifest.benchmark, weight: 1 }], history)
        : null,
    [manifest.benchmark, history],
  )

  const counts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const a of manifest.assets) c[a.group] = (c[a.group] ?? 0) + 1
    return c
  }, [manifest.assets])

  const toggleGroup = (g: Group) =>
    setVisibleGroups((prev) => {
      const next = new Set(prev)
      // Never allow an empty selection: a scatter with nothing in it looks like a load
      // failure rather than a filter state.
      if (next.has(g) && next.size > 1) next.delete(g)
      else next.add(g)
      return next
    })

  const shown = manifest.assets.filter((a) => visibleGroups.has(a.group))
  const capValue = frontiers[capSlug].weight_cap
  const dd = maxDrawdown(port)
  const benchDd = bench ? maxDrawdown(bench) : null
  const monthlyVol = realisedVol(weights, history)

  /** What the download writes. Assembled here rather than inside the button because every
   *  field is state this component already holds, and a file of weights without the cap, the
   *  rf and the window they were solved under is not reproducible. */
  const selection: Selection = {
    manifest,
    cap: capValue,
    capSlug,
    rf,
    position: here,
    nPoints: path.points.length,
    atTangency: Math.abs(t - tangency.t) < 1e-6,
    weights,
    bySymbol,
    inSample: {
      growthOf1: port[port.length - 1],
      maxDrawdown: dd,
      realisedVol: monthlyVol,
    },
  }

  return (
    <div className="app">
      <div className="masthead">
        <div>
          <h1>The efficient frontier of {manifest.n_assets} cross-asset ETFs</h1>
        </div>
        <button
          className="toggle"
          aria-pressed={dark}
          onClick={() => {
            setDark(!dark)
            setThemeSet(true)
          }}
        >
          {dark ? 'Light' : 'Dark'}
        </button>
      </div>
      <p className="subtitle">
        Mean-variance optimisation over {manifest.window.start} → {manifest.window.end} (
        {manifest.window.trading_days.toLocaleString()} trading days,{' '}
        {manifest.window.years.toFixed(1)} years). Expected returns are the sample geometric
        mean; risk is a Ledoit-Wolf shrunk covariance matrix, both annualised at 252 trading
        days. Drag the handle to choose a portfolio.
      </p>

      <Controls
        caps={manifest.caps}
        cap={capSlug}
        onCap={setCapSlug}
        rf={rf}
        rfDefault={manifest.rf_default}
        onRf={setRf}
        groups={manifest.groups}
        visibleGroups={visibleGroups}
        onToggleGroup={toggleGroup}
        counts={counts}
        fitFrontier={fitFrontier}
        onFit={setFitFrontier}
        extra={
          <SettingsBar
            config={config}
            manifest={manifest}
            onLoad={applyConfig}
            onReset={() => {
              clearStored()
              applyConfig(defaults(manifest))
            }}
          />
        }
      />

      <FrontierChart
        assets={manifest.assets}
        path={path}
        model={model}
        rf={rf}
        t={t}
        onT={(next) => setFrac(last > 0 ? next / last : 0)}
        tangency={tangency}
        visibleGroups={visibleGroups}
        fitFrontier={fitFrontier}
        benchmark={manifest.benchmark}
      />
      <FrontierLegend nAssets={shown.length} hidden={manifest.assets.length - shown.length} />

      <StatTiles
        tiles={[
          {
            label: 'Expected return',
            value: tile.pct(here.ret),
            sub: 'annualised, in sample',
          },
          { label: 'Volatility', value: tile.pct(here.vol), sub: 'model σ, from the shrunk matrix' },
          {
            label: 'Sharpe',
            value: tile.ratio(here.sharpe),
            sub: `at rf ${pct(rf, 2)}${Math.abs(t - tangency.t) < 1e-6 ? ' · the maximum' : ` · max is ${tangency.sharpe.toFixed(2)}`}`,
          },
          {
            label: 'Holdings',
            value: String(weights.length),
            sub: `of ${manifest.n_assets}, capped at ${pct(capValue, 0)}`,
          },
          {
            label: 'Growth of 1',
            value: mult(port[port.length - 1]),
            sub: bench ? `${manifest.benchmark}: ${mult(bench[bench.length - 1])}` : undefined,
          },
          {
            label: 'Worst drawdown',
            value: tile.pct(dd, 1),
            sub: benchDd ? `${manifest.benchmark}: ${pct(benchDd, 1)}` : 'month-end, in sample',
          },
        ]}
      />
      <p className="note" style={{ marginTop: 10 }}>
        Realised σ of the monthly series is {pct(monthlyVol, 2)} against the
        model's {pct(here.vol, 2)}, and compounding the monthly path gives{' '}
        {pct(annualised(port, history.years), 2)} against the model's {pct(here.ret, 2)}. Return
        agrees by construction — the monthly series telescopes to the same total growth the
        optimiser measured. Volatility does not, and is not meant to: the model number is
        annualised from daily returns, this one from monthly, and the gap is the autocorrelation
        daily sampling annualises away.
      </p>

      <div className="section">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <h2>What you hold</h2>
          <div className="segmented">
            <button onClick={() => downloadCsv(selection)} title="Weights as CSV, with the cap, rf and window in comment lines">
              Download CSV
            </button>
            <button onClick={() => downloadJson(selection)} title="Weights plus the full estimation metadata">
              Download JSON
            </button>
          </div>
        </div>
        <WeightTable weights={weights} bySymbol={bySymbol} cap={capValue} />
      </div>

      <div className="section">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <h2>Growth of 1 unit, monthly rebalanced</h2>
          <button className="toggle" aria-pressed={logScale} onClick={() => setLogScale(!logScale)}>
            {logScale ? 'Log scale' : 'Linear scale'}
          </button>
        </div>
        <p className="caption">
          <strong>This is in-sample and cannot be read as a test.</strong> The weights were
          chosen using the returns and covariances of this exact window, so the optimiser knew
          the answer; the curve shows the shape of the ride, not evidence that the method works.
          A log scale is the default because equal vertical distances are then equal compound
          rates, which is what comparing two growth curves means.
        </p>
        <GrowthChart
          portfolio={port}
          benchmark={bench}
          benchmarkSymbol={manifest.benchmark}
          history={history}
          logScale={logScale}
        />
      </div>

      <div className="section">
        <h2>Every asset in the universe</h2>
        <p className="caption">
          The same data as the scatter, addressable by name and sortable. Rows with a dot are in
          the current portfolio. Sort by Sharpe to see which assets a mean-variance optimiser
          over this window was always going to pick.
        </p>
        <AssetTable assets={manifest.assets} held={held} visibleGroups={visibleGroups} />
      </div>

      <div className="section">
        <h2>What this does and does not show</h2>
        <p className="note">
          <strong>Expected returns are the weak link, and no amount of covariance care fixes
          it.</strong>{' '}
          The standard error on a {manifest.window.years.toFixed(0)}-year annualised mean return
          for a 20%-volatility asset is about 5 percentage points — the same order as the equity
          risk premium being estimated. So the tangency portfolio is a statement about which
          assets happened to do well since {manifest.window.start.slice(0, 4)}, and over that
          window the answer is known in advance: US large-cap technology.
        </p>
        <p className="note" style={{ marginTop: 10 }}>
          <strong>That is what the weight cap is for.</strong> Compare the three settings above.
          At 100% the optimiser concentrates into a handful of positions; at 10% it cannot. How
          much the picture moves between them is a direct read on how much of the "optimal"
          portfolio was estimation error rather than signal.
        </p>
        <p className="note" style={{ marginTop: 10 }}>
          <strong>Shrinkage was measured, not assumed.</strong> Ledoit-Wolf chose an intensity of{' '}
          {manifest.estimation.shrinkage_delta.toFixed(4)} on {manifest.estimation.n_returns.toLocaleString()}{' '}
          daily returns for {manifest.n_assets} assets — nearly inactive, because at that sample
          size the sample covariance needs little help. It is not what holds this answer
          together. What shrinkage cannot fix at any intensity is a near-exact linear
          dependency, and this universe has several by construction (UUP is a short basket of
          the currencies FXE, FXY and FXB are long; IAU and GLD hold the same bullion). The
          minimum-variance portfolio duly nets a currency basket down to under 1% volatility.
          That is the optimiser exploiting a collinearity, not a low-risk portfolio anyone
          found, and it is left in view because seeing it is the point.
        </p>
        <p className="note" style={{ marginTop: 10 }}>
          {Object.keys(manifest.excluded.window_or_coverage).length} funds in the universe were
          dropped for starting after {manifest.window.requested_start} — a covariance matrix
          needs one common window, so a 2015 launch either truncates every other asset's history
          or leaves:{' '}
          {Object.keys(manifest.excluded.window_or_coverage).sort().join(', ') || 'none'}.
        </p>
        <p className="note" style={{ marginTop: 14, color: 'var(--ink-muted)' }}>
          Prices are dividend-adjusted daily closes. Data generated {manifest.generated_at}, rf
          from {manifest.rf_source}. No transaction costs, no taxes, no slippage; monthly
          rebalancing at zero cost flatters the capped variants most. Not investment advice.
        </p>
      </div>
    </div>
  )
}
