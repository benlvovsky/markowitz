/** The download is a file someone acts on, so it gets tested like an artifact, not like a button.
 *
 * Three failure modes worth guarding, none of them visible on screen:
 *
 * 1. The file disagrees with the table. The exported weights come from the same `weightsAt`
 *    the page renders, so they agree by construction -- but the ROUNDING is the export's own,
 *    and a file whose weights sum to something the page never showed is the kind of error a
 *    reader discovers in a spreadsheet a week later. Parsed back and compared here.
 * 2. A name breaks the CSV. No asset in the shipped universe contains a comma today, so the
 *    real data cannot exercise the quoting at all -- a `,` added to a universe TOML tomorrow
 *    would shift every later column silently. Synthetic case below, for the same reason
 *    `pipeline/tests` builds a synthetic store for the forward-fill rule.
 * 3. The provenance goes stale. The file's whole claim to being reproducible is that it
 *    carries the cap, the rf and the window; asserted field by field rather than by eyeball.
 *
 * Reads `public/data/*.json`, like the rest of this suite, so it fails when the data is wrong
 * and not only when the code is.
 */
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { filename, portfolioCsv, portfolioJson, type Selection } from './export'
import { at, buildModel, buildPath, growth, maxDrawdown, realisedVol, tangency, weightsAt } from './portfolio'
import type { Asset, FrontierDoc, History, Manifest, Stats } from './types'

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data')
const read = <T>(name: string): T => JSON.parse(readFileSync(join(DATA, name), 'utf8')) as T

const manifest = read<Manifest>('manifest.json')
const stats = read<Stats>('stats.json')
const history = read<History>('history.json')
const model = buildModel(stats)
const bySymbol = new Map(manifest.assets.map((a) => [a.symbol, a]))

/** RFC 4180, so the test cannot pass a file that only a `split(',')` would read. */
function parseCsv(text: string): { comments: string[]; rows: string[][] } {
  const comments: string[] = []
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  let atLineStart = true
  let i = 0
  const endCell = () => {
    row.push(cell)
    cell = ''
  }
  const endRow = () => {
    endCell()
    rows.push(row)
    row = []
    atLineStart = true
  }
  while (i < text.length) {
    const c = text[i]
    if (atLineStart && !quoted && c === '#') {
      const nl = text.indexOf('\n', i)
      comments.push(text.slice(i + 1, nl < 0 ? text.length : nl).trim())
      i = nl < 0 ? text.length : nl + 1
      continue
    }
    atLineStart = false
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        cell += '"'
        i += 2
      } else if (c === '"') {
        quoted = false
        i++
      } else {
        cell += c
        i++
      }
      continue
    }
    if (c === '"') {
      quoted = true
      i++
    } else if (c === ',') {
      endCell()
      i++
    } else if (c === '\n') {
      endRow()
      i++
    } else if (c === '\r') {
      i++
    } else {
      cell += c
      i++
    }
  }
  if (cell !== '' || row.length) endRow()
  return { comments, rows }
}

function selectionFor(capIndex: number, rf: number, tOverride?: number): Selection {
  const entry = manifest.caps[capIndex]
  const doc = read<FrontierDoc>(entry.file)
  const path = buildPath(doc, model)
  const tan = tangency(path, rf)
  const t = tOverride ?? tan.t
  const weights = weightsAt(path, t, model)
  const port = growth(weights, history)
  return {
    manifest,
    cap: doc.weight_cap,
    capSlug: entry.slug,
    rf,
    position: at(path, t, rf),
    nPoints: path.points.length,
    atTangency: Math.abs(t - tan.t) < 1e-6,
    weights,
    bySymbol,
    inSample: {
      growthOf1: port[port.length - 1],
      maxDrawdown: maxDrawdown(port),
      realisedVol: realisedVol(weights, history),
    },
  }
}

describe('the exported CSV is the portfolio the page shows', () => {
  it('carries every holding, at the precision it claims, in table order', () => {
    for (let c = 0; c < manifest.caps.length; c++) {
      const s = selectionFor(c, manifest.rf_default)
      const { rows } = parseCsv(portfolioCsv(s))
      expect(rows[0]).toEqual(['symbol', 'name', 'group', 'weight'])
      const body = rows.slice(1, -1)
      expect(body.length).toBe(s.weights.length)
      body.forEach((r, k) => {
        expect(r).toHaveLength(4)
        expect(r[0]).toBe(s.weights[k].symbol)
        expect(r[1]).toBe(bySymbol.get(s.weights[k].symbol)!.name)
        // The file's own claim: six decimals, no more. A raw `String(weight)` would print the
        // interpolation's full binary tail, which is precision the 6dp inputs never had.
        expect(r[3]).toMatch(/^\d\.\d{6}$/)
        expect(Number(r[3])).toBeCloseTo(s.weights[k].weight, 6)
      })
      // Descending weight, the order the table renders.
      const ws = body.map((r) => Number(r[3]))
      for (let k = 1; k < ws.length; k++) expect(ws[k]).toBeLessThanOrEqual(ws[k - 1])
    }
  })

  it('states the total it actually wrote rather than 1', () => {
    for (let c = 0; c < manifest.caps.length; c++) {
      const s = selectionFor(c, manifest.rf_default)
      const { rows } = parseCsv(portfolioCsv(s))
      const total = rows[rows.length - 1]
      expect(total[0]).toBe('TOTAL')
      const body = rows.slice(1, -1).map((r) => Number(r[3]))
      const summed = body.reduce((a, b) => a + b, 0)
      expect(Number(total[3])).toBeCloseTo(summed, 6)
      // Interpolation plus the 1bp floor loses a little; it must be a little, not a lot.
      expect(Number(total[3])).toBeGreaterThan(0.99)
      expect(Number(total[3])).toBeLessThanOrEqual(1.0000005)
    }
  })

  it('respects the cap it was solved under, so the file cannot come from another frontier', () => {
    for (let c = 0; c < manifest.caps.length; c++) {
      const s = selectionFor(c, manifest.rf_default)
      for (const h of s.weights) expect(h.weight).toBeLessThanOrEqual(s.cap + 1e-6)
    }
  })

  it('puts the cap, the rf and the window in comment lines a reader can find', () => {
    const s = selectionFor(1, 0.05)
    const csv = portfolioCsv(s)
    const { comments, rows } = parseCsv(csv)
    const joined = comments.join('\n')
    expect(joined).toContain(`window=${manifest.window.start}..${manifest.window.end}`)
    expect(joined).toContain(`weight_cap=${(s.cap * 100).toFixed(0)}%`)
    expect(joined).toContain('risk_free_rate=5.00%')
    expect(joined).toContain(`data_generated_at=${manifest.generated_at}`)
    expect(joined.toLowerCase()).toContain('in sample')
    expect(joined.toLowerCase()).toContain('not investment advice')
    // Nothing in the body may start with `#`, or `read_csv(comment='#')` would eat a holding.
    for (const r of rows) expect(r[0].startsWith('#')).toBe(false)
  })

  it('says whether the handle is at the tangency, because the default position is', () => {
    const tan = portfolioCsv(selectionFor(0, manifest.rf_default))
    expect(tan).toContain('position=tangency portfolio')
    const moved = portfolioCsv(selectionFor(0, manifest.rf_default, 3))
    expect(moved).toContain('position=t=3.000 of 0..')
    expect(moved).not.toContain('position=tangency')
  })

  it('quotes a name containing a comma or a quote -- which the shipped universe cannot test', () => {
    const s = selectionFor(0, manifest.rf_default)
    const hostile = 'Vanguard "Total", World'
    const first = s.weights[0].symbol
    const patched: Selection = {
      ...s,
      bySymbol: new Map(s.bySymbol).set(first, {
        ...(s.bySymbol.get(first) as Asset),
        name: hostile,
      }),
    }
    const { rows } = parseCsv(portfolioCsv(patched))
    expect(rows[1]).toHaveLength(4)
    expect(rows[1][1]).toBe(hostile)
    expect(rows[1][0]).toBe(first)
    expect(Number(rows[1][3])).toBeCloseTo(s.weights[0].weight, 6)
  })
})

describe('the exported JSON reproduces the row without the page', () => {
  it('agrees with a fresh recompute of return, risk and Sharpe', () => {
    for (let c = 0; c < manifest.caps.length; c++) {
      const rf = 0.031
      const s = selectionFor(c, rf)
      const doc = JSON.parse(portfolioJson(s))
      expect(doc.kind).toBe('markowitz-portfolio')
      expect(doc.selection.weight_cap).toBe(s.cap)
      expect(doc.selection.risk_free_rate).toBe(rf)
      expect(doc.selection.at_tangency).toBe(true)
      // Recompute-to-recompute, at the precision the file rounds to -- never file-to-file at
      // 1e-9, the trap CLAUDE.md names.
      expect(doc.performance.expected_return).toBeCloseTo(s.position.ret, 8)
      expect(doc.performance.volatility).toBeCloseTo(s.position.vol, 8)
      expect(doc.performance.sharpe).toBeCloseTo((s.position.ret - rf) / s.position.vol, 6)
      expect(doc.n_holdings).toBe(s.weights.length)
      expect(doc.weights.map((w: { symbol: string }) => w.symbol)).toEqual(
        s.weights.map((w) => w.symbol),
      )
      expect(doc.weights_sum).toBeCloseTo(
        s.weights.reduce((a, h) => a + Number(h.weight.toFixed(6)), 0),
        6,
      )
    }
  })

  it('carries the window and estimation metadata verbatim from the manifest', () => {
    const doc = JSON.parse(portfolioJson(selectionFor(0, manifest.rf_default)))
    expect(doc.window).toEqual(manifest.window)
    expect(doc.estimation).toEqual(manifest.estimation)
    expect(doc.universe).toEqual({
      n_universe: manifest.n_universe,
      n_assets: manifest.n_assets,
      benchmark: manifest.benchmark,
    })
    expect(doc.caveat.toLowerCase()).toContain('in sample')
    // A bare drawdown number is ambiguous in sign, so the convention is asserted here AND
    // stated in the file: positive fraction, matching `maxDrawdown`.
    expect(doc.in_sample.max_drawdown).toBeGreaterThan(0)
    expect(doc.in_sample.max_drawdown).toBeLessThan(1)
    expect(doc.in_sample.basis).toContain('POSITIVE')
    expect(doc.in_sample.growth_of_1).toBeGreaterThan(1)
  })

  it('names every holding it lists', () => {
    const doc = JSON.parse(portfolioJson(selectionFor(2, manifest.rf_default)))
    for (const w of doc.weights) {
      expect(w.name).toBeTruthy()
      expect(manifest.groups).toContain(w.group)
      expect(w.weight).toBeGreaterThanOrEqual(1e-4)
    }
  })
})

describe('the filename', () => {
  it('is filesystem-safe and identifies the cap and the settings', () => {
    for (const ext of ['csv', 'json'] as const) {
      const s = selectionFor(1, 0.042)
      const name = filename(s, ext)
      expect(name).toMatch(/^[A-Za-z0-9._-]+$/)
      expect(name.endsWith(`.${ext}`)).toBe(true)
      expect(name).toContain(s.capSlug)
      expect(name).toContain('rf4.20')
    }
  })

  it('differs between two caps, so downloading several does not overwrite one file', () => {
    const names = manifest.caps.map((_, c) => filename(selectionFor(c, 0.042), 'csv'))
    expect(new Set(names).size).toBe(names.length)
  })
})
