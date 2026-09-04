/** Serialising the portfolio at the handle to a file the reader can take away.
 *
 * Entirely browser-side, which is not a limitation here but the same constraint the whole
 * project runs under: there is no server to POST to and nothing to render a file remotely.
 * A Blob and an anchor click is the whole mechanism.
 *
 * TWO FORMATS BECAUSE THEY HAVE DIFFERENT JOBS, and the split is the reason both exist:
 *
 * - CSV is for pasting into a spreadsheet or a broker's basket upload, so the body is plain
 *   `symbol,name,group,weight` with nothing clever in it. Its provenance rides in leading `#`
 *   lines: a file of weights with no record of the window, the cap and the rf they came from
 *   is a number without its estimate, and this construction's whole caveat is that the
 *   estimate is the weak part. `pandas.read_csv(p, comment='#')` skips them.
 * - JSON is for reproducing the row: the full window, the estimation metadata, the position
 *   along the path and the in-sample outcome, so the file states what it is without the page
 *   it came from.
 *
 * ROUNDING MATCHES THE PIPELINE'S TRANSPORT PRECISION (weights 6, ret/vol 8, sharpe 6). Not
 * arbitrary: the shipped weights are themselves 6-decimal, so writing more digits of an
 * interpolation between two 6-decimal vectors would be printing precision the inputs never
 * had. It follows that the weights do NOT sum to exactly 1 -- interpolation plus the 1bp
 * floor drops a little -- so the total is written out as measured rather than asserted, for
 * the same reason the weight table prints it.
 */
import type { Asset, Manifest } from './types'
import type { Position } from './portfolio'
import { groupLabel, yearsText } from './viz'

const W_DP = 6
const RV_DP = 8
const S_DP = 6

export interface Holding {
  symbol: string
  weight: number
}

export interface Selection {
  manifest: Manifest
  /** The per-asset weight cap this frontier was solved under, e.g. 1.0. */
  cap: number
  /** The cap's slug, as the manifest names it, e.g. `cap100`. Used in the filename. */
  capSlug: string
  rf: number
  /** Where the handle is: continuous index along the path, plus its exact ret/vol/sharpe. */
  position: Position
  /** Path length, so `t` can be reported against the range it lives in. */
  nPoints: number
  atTangency: boolean
  weights: Holding[]
  bySymbol: Map<string, Asset>
  /** Outcomes of the monthly in-sample path, which the page shows beside the model numbers. */
  inSample: { growthOf1: number; maxDrawdown: number; realisedVol: number }
}

const round = (v: number, dp: number) => Number(v.toFixed(dp))
const pctStr = (v: number, dp = 2) => `${(v * 100).toFixed(dp)}%`

/** RFC 4180 field. No asset name in the shipped universe needs quoting today; one added to a
 *  universe TOML tomorrow might, and a comma in a name silently shifts every later column. */
function field(v: string): string {
  return /[",\r\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v
}

function provenance(s: Selection): string[] {
  const m = s.manifest
  const w = m.window
  return [
    `efficient frontier of ${m.n_assets} cross-asset ETFs -- portfolio export`,
    `exported_at=${new Date().toISOString()}`,
    `data_generated_at=${m.generated_at}`,
    `window=${w.start}..${w.end} (${w.trading_days} trading days, ${yearsText(w.years)} years)`,
    `universe=${m.universe.key} (${m.universe.name}, from ${m.universe.file})`,
    `weight_cap=${pctStr(s.cap, 0)}`,
    `risk_free_rate=${pctStr(s.rf)} (${m.rf_source})`,
    `position=${s.atTangency ? 'tangency portfolio, max Sharpe at this rf' : `t=${s.position.t.toFixed(3)} of 0..${s.nPoints - 1}`}`,
    `expected_return=${pctStr(s.position.ret)} volatility=${pctStr(s.position.vol)} sharpe=${s.position.sharpe.toFixed(4)}`,
    `expected returns are the sample geometric mean; covariance is Ledoit-Wolf shrunk at delta=${m.estimation.shrinkage_delta.toFixed(4)}`,
    'IN SAMPLE: these weights were chosen on this exact window. No transaction costs, taxes or slippage. Not investment advice.',
  ]
}

export function portfolioCsv(s: Selection): string {
  const lines = provenance(s).map((l) => `# ${l}`)
  lines.push('symbol,name,group,weight')
  let total = 0
  for (const h of s.weights) {
    const a = s.bySymbol.get(h.symbol)
    total += round(h.weight, W_DP)
    lines.push(
      [
        field(h.symbol),
        field(a?.name ?? ''),
        field(a ? groupLabel(s.manifest.group_labels, a.group) : ''),
        h.weight.toFixed(W_DP),
      ].join(','),
    )
  }
  lines.push(['TOTAL', '', '', total.toFixed(W_DP)].join(','))
  return `${lines.join('\n')}\n`
}

export function portfolioJson(s: Selection): string {
  const m = s.manifest
  const holdings = s.weights.map((h) => {
    const a = s.bySymbol.get(h.symbol)
    return {
      symbol: h.symbol,
      name: a?.name ?? null,
      asset_class: a?.asset_class ?? null,
      group: a?.group ?? null,
      weight: round(h.weight, W_DP),
    }
  })
  const doc = {
    kind: 'markowitz-portfolio',
    exported_at: new Date().toISOString(),
    data_generated_at: m.generated_at,
    universe: {
      key: m.universe.key,
      name: m.universe.name,
      file: m.universe.file,
      n_universe: m.n_universe,
      n_assets: m.n_assets,
      benchmark: m.benchmark,
    },
    window: m.window,
    estimation: m.estimation,
    selection: {
      weight_cap: s.cap,
      risk_free_rate: s.rf,
      rf_source: m.rf_source,
      path_position: round(s.position.t, 6),
      path_points: s.nPoints,
      at_tangency: s.atTangency,
    },
    performance: {
      expected_return: round(s.position.ret, RV_DP),
      volatility: round(s.position.vol, RV_DP),
      sharpe: round(s.position.sharpe, S_DP),
      basis: 'annualised at 252 trading days, from the shrunk covariance matrix',
    },
    in_sample: {
      growth_of_1: round(s.inSample.growthOf1, 6),
      max_drawdown: round(s.inSample.maxDrawdown, 6),
      realised_vol_monthly: round(s.inSample.realisedVol, RV_DP),
      basis:
        'monthly rebalanced at these fixed weights, zero cost; max_drawdown is a POSITIVE ' +
        'fraction (0.25 means a 25% peak-to-trough fall)',
    },
    n_holdings: holdings.length,
    weights_sum: round(
      holdings.reduce((t, h) => t + h.weight, 0),
      W_DP,
    ),
    weights_precision_dp: W_DP,
    weights: holdings,
    // The window length and the cap count are READ, not written into the string. Both were
    // literals ("15-year", "the three weight caps") describing the shipped build on the day it
    // was written, in the one artifact a reader keeps after the page is gone -- and the weekly
    // cron moves the window, so the sentence was going to become false with nothing to catch it.
    caveat:
      'IN SAMPLE. The weights were chosen using the returns and covariances of this exact ' +
      `window, so the optimiser knew the answer. Expected returns are the weak input: the ` +
      `standard error on a ${yearsText(m.window.years)}-year annualised mean for a ` +
      '20%-volatility asset is about 5 percentage points. Compare the ' +
      `${m.caps.length} weight caps before believing any single row. ` +
      'No transaction costs, taxes or slippage. Not investment advice.',
  }
  return `${JSON.stringify(doc, null, 2)}\n`
}

/** `portfolio-cap100-vol11.19-rf3.74-20260904.csv`. The cap, vol and rf are in the name
 *  because the interesting use is downloading several and comparing them in one directory. */
export function filename(s: Selection, ext: 'csv' | 'json'): string {
  const d = new Date()
  const stamp = [
    d.getFullYear(),
    String(d.getMonth() + 1).padStart(2, '0'),
    String(d.getDate()).padStart(2, '0'),
  ].join('')
  const n = (v: number) => (v * 100).toFixed(2)
  return `portfolio-${s.capSlug}-vol${n(s.position.vol)}-rf${n(s.rf)}-${stamp}.${ext}`
}

/** Hand a string to the browser as a download. The only part of this file that touches the
 *  DOM, kept separate so the serialisers above are testable in node. */
export function save(name: string, mime: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: `${mime};charset=utf-8` }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.rel = 'noopener'
  document.body.appendChild(a)
  a.click()
  a.remove()
  // Not revoked synchronously: Safari has cancelled the download when the URL went away in
  // the same task as the click.
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

export function downloadCsv(s: Selection): void {
  save(filename(s, 'csv'), 'text/csv', portfolioCsv(s))
}

export function downloadJson(s: Selection): void {
  save(filename(s, 'json'), 'application/json', portfolioJson(s))
}
