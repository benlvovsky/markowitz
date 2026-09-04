/** Invariants for the arithmetic the BROWSER does, against the artifacts it actually reads.
 *
 * The pipeline's own suite proves the JSON is self-consistent. It says nothing about this
 * code, and this code is not a renderer -- it recomputes risk and return, interpolates
 * portfolios the solver never saw, and solves for the tangency point in closed form. Three
 * classes of bug live here and nowhere else:
 *
 * 1. The segment shortcut. Risk along a segment is computed from three precomputed scalars
 *    rather than from the covariance matrix, which is a real algebraic claim and is checked
 *    below against a brute-force quadratic form over the interpolated weight vector.
 * 2. The closed-form tangency root. `tangency` claims the u^2 terms cancel and one linear
 *    root suffices. Checked against a dense scan of every segment.
 * 3. Disagreement with the pipeline. If `at(path, i)` did not reproduce the shipped ret/vol
 *    at integer i, the chart would draw a curve through points other than the solved ones --
 *    and every number on the page would be subtly wrong in a way no rendering test sees.
 *
 * Reads `public/data/*.json` directly, so it tests the committed artifacts rather than a
 * fixture that can drift from them.
 */
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import {
  annualised,
  at,
  bilinear,
  buildModel,
  buildPath,
  dotMu,
  growth,
  tangency,
  toSparse,
  weightsAt,
  type Model,
  type Path,
  type Sparse,
} from './portfolio'
import type { FrontierDoc, History, Manifest, Stats } from './types'
import { nearestOnPolyline } from './viz'

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data')
const read = <T>(name: string): T => JSON.parse(readFileSync(join(DATA, name), 'utf8')) as T

const manifest = read<Manifest>('manifest.json')
const stats = read<Stats>('stats.json')
const history = read<History>('history.json')
const model = buildModel(stats)

/** The same tolerance the pipeline's own consistency tests use: weights ship at 1e-6, and a
 * 116-term dot product of them carries ~1e-6*sqrt(116) of slack. Asserted from the other
 * side of the contract here. */
const TOL = 2e-5

const caps = manifest.caps.map((c) => ({
  ...c,
  doc: read<FrontierDoc>(c.file),
}))

/** Interpolated weights as a dense vector -- the slow, obvious way, for cross-checking. */
function denseAt(path: Path, t: number, model: Model): Float64Array {
  const i = Math.min(Math.floor(t), path.points.length - 2)
  const u = t - i
  const w = new Float64Array(model.n)
  for (const [sym, x] of Object.entries(path.points[i].w)) w[model.index.get(sym)!] += (1 - u) * x
  for (const [sym, x] of Object.entries(path.points[i + 1].w)) w[model.index.get(sym)!] += u * x
  return w
}

function denseQuad(w: Float64Array, model: Model): number {
  let s = 0
  for (let i = 0; i < model.n; i++) {
    if (w[i] === 0) continue
    for (let j = 0; j < model.n; j++) s += w[i] * w[j] * model.cov[i * model.n + j]
  }
  return s
}

describe('the model matches the artifacts it was built from', () => {
  it('indexes every symbol the frontiers hold', () => {
    expect(model.n).toBe(manifest.assets.length)
    for (const { doc, slug } of caps) {
      for (const p of doc.frontier) {
        for (const sym of Object.keys(p.w)) {
          expect(model.index.get(sym), `${slug} holds ${sym}`).toBeDefined()
        }
      }
    }
  })

  it('reproduces the asset table from mu and the covariance diagonal', () => {
    manifest.assets.forEach((a, i) => {
      expect(a.ret).toBeCloseTo(model.mu[i], 8)
      expect(a.vol).toBeCloseTo(Math.sqrt(model.cov[i * model.n + i]), 6)
    })
  })
})

describe.each(caps)('cap $slug', ({ doc }) => {
  const path = buildPath(doc, model)
  const rf = doc.risk_free_rate

  it('reproduces every solved point at its integer index', () => {
    doc.frontier.forEach((p, i) => {
      const q = at(path, i, rf)
      expect(q.ret, `point ${i} return`).toBeCloseTo(p.ret, 6)
      expect(Math.abs(q.vol - p.vol), `point ${i} volatility`).toBeLessThan(TOL)
      expect(Math.abs(q.sharpe - p.sharpe), `point ${i} sharpe`).toBeLessThan(1e-4)
    })
  })

  it('gets interpolated risk from three scalars, exactly as the full quadratic form does', () => {
    // THE claim of the segment shortcut: var(u) = (1-u)^2 Sii + 2u(1-u) Sij + u^2 Sjj is not
    // an approximation of w(u)' C w(u), it IS w(u)' C w(u). A single dropped cross term here
    // would move every displayed volatility between solved points, and would be invisible in
    // the chart because the curve would still look like a frontier.
    for (let i = 0; i + 1 < path.points.length; i++) {
      for (const u of [0.13, 0.37, 0.5, 0.82]) {
        const fast = at(path, i + u, rf)
        const slow = Math.sqrt(denseQuad(denseAt(path, i + u, model), model))
        expect(Math.abs(fast.vol - slow), `segment ${i} at u=${u}`).toBeLessThan(1e-12)
      }
    }
  })

  it('interpolates return linearly, because return IS linear in the weights', () => {
    for (let i = 0; i + 1 < path.points.length; i++) {
      const u = 0.4
      const w = denseAt(path, i + u, model)
      let ret = 0
      for (let j = 0; j < model.n; j++) ret += w[j] * model.mu[j]
      expect(at(path, i + u, rf).ret).toBeCloseTo(ret, 12)
    }
  })

  it('never leaves the feasible set between solved points', () => {
    const cap = doc.weight_cap
    for (let i = 0; i + 1 < path.points.length; i++) {
      for (const u of [0.05, 0.5, 0.95]) {
        const w = weightsAt(path, i + u, model, 0)
        const total = w.reduce((s, x) => s + x.weight, 0)
        expect(total, `segment ${i}`).toBeCloseTo(1, 6)
        expect(Math.min(...w.map((x) => x.weight))).toBeGreaterThanOrEqual(0)
        expect(Math.max(...w.map((x) => x.weight))).toBeLessThanOrEqual(cap + 1e-9)
      }
    }
  })

  it('finds the tangency point by closed form where a dense scan finds it', () => {
    // The algebra being checked: setting d/du of (A+Bu)/sqrt(C+Du+Eu^2) to zero cancels the
    // u^2 terms and leaves ONE linear root. If that were wrong, the tangency marker would sit
    // near but not at the maximum -- close enough to look right, and wrong by a real amount
    // at the caps where the frontier is most curved.
    for (const testRf of [0, 0.01, 0.02, 0.03, rf, 0.05, 0.075]) {
      const closed = tangency(path, testRf)
      let scanned = -Infinity
      for (let i = 0; i + 1 < path.points.length; i++) {
        for (let k = 0; k <= 400; k++) {
          const s = at(path, i + k / 400, testRf).sharpe
          if (Number.isFinite(s) && s > scanned) scanned = s
        }
      }
      expect(closed.sharpe, `rf=${testRf}`).toBeGreaterThanOrEqual(scanned - 1e-12)
      expect(closed.sharpe - scanned, `rf=${testRf}`).toBeLessThan(1e-6)
    }
  })

  it('agrees with the pipeline about which solved point is the tangency', () => {
    const k = doc.max_sharpe_index
    // Compared against the shipped point RECOMPUTED here, not against its shipped `sharpe`
    // field. The pipeline rounds that field to 6 decimals for transport, so comparing an exact
    // recompute to it at 1e-9 fails by ~2e-7 on a pipeline that is entirely correct -- which
    // is what this test did on its first run. The rounding is checked separately, below.
    const shipped = at(path, k, rf)
    const found = tangency(path, rf)

    // The browser searches the continuous path; the pipeline searched solved points only. So
    // the browser can only do better, and by no more than the interpolation gap -- otherwise
    // the point the JSON labels "tangency" is not the tangency.
    expect(found.sharpe).toBeGreaterThanOrEqual(shipped.sharpe - 1e-12)
    expect(found.sharpe - shipped.sharpe).toBeLessThan(5e-4)

    // And it is the same place, not merely the same value: the continuous optimum must lie in
    // a segment touching the solved one.
    expect(Math.abs(found.t - k)).toBeLessThanOrEqual(1)

    // The exact claim about the field, stated exactly: it is the argmax over SOLVED points.
    // The bound above only catches a gross error -- an off-by-one index survives it, because
    // one point either side of the tangency has a Sharpe well inside the interpolation gap.
    let argmax = 0
    for (let i = 1; i < path.points.length; i++) {
      if (at(path, i, rf).sharpe > at(path, argmax, rf).sharpe) argmax = i
    }
    expect(k).toBe(argmax)

    // The shipped rounded field, against the exact recompute. Slack is transport precision on
    // three quantities at once: sharpe at 1e-6, mu at 1e-8, and the covariance at 1e-9.
    expect(Math.abs(doc.frontier[k].sharpe - shipped.sharpe)).toBeLessThan(1e-5)
  })

  it('tightens nothing when the risk-free rate moves, because the frontier does not depend on it', () => {
    // rf appears nowhere in "minimise variance subject to a return target". If a future change
    // let it leak into the path, this is where it would show.
    const a = buildPath(doc, model)
    const b = buildPath(doc, model)
    for (let i = 0; i < a.points.length; i++) {
      expect(at(a, i, 0).ret).toBe(at(b, i, 0.09).ret)
      expect(at(a, i, 0).vol).toBe(at(b, i, 0.09).vol)
    }
  })
})

describe('the growth curve is the same number as the frontier axis', () => {
  const path = buildPath(caps[0].doc, model)

  it('compounds a single asset to exactly its expected return', () => {
    // The telescoping identity, asserted through the browser's own code path. The monthly
    // series is anchored at the daily panel's first bar, so this is an equality and not an
    // approximation; 2e-6 is the transport precision of the 1e-6-rounded returns.
    for (const a of manifest.assets) {
      const g = growth([{ symbol: a.symbol, weight: 1 }], history)
      expect(Math.abs(annualised(g, history.years) - a.ret), a.symbol).toBeLessThan(2e-6)
    }
  })

  it('is a weighted average of its holdings in every month', () => {
    const w = weightsAt(path, path.points.length / 2, model)
    const g = growth(w, history)
    for (let t = 0; t < history.dates.length; t++) {
      const expected = w.reduce((s, x) => s + x.weight * history.returns[x.symbol][t], 0)
      expect(g[t + 1] / g[t] - 1).toBeCloseTo(expected, 12)
    }
  })

  it('starts at one', () => {
    expect(growth(weightsAt(path, 0, model), history)[0]).toBe(1)
  })
})

describe('the drag geometry', () => {
  const path = buildPath(caps[0].doc, model)

  it('projects onto the path rather than snapping to a vertex', () => {
    // A cursor a third of the way along a segment must return a fractional index. Snapping to
    // the nearest solved point instead would make the handle stutter between ~60 positions and
    // would silently disable weight interpolation, which is the whole feature.
    const xs = Float64Array.from([0, 100, 200])
    const ys = Float64Array.from([0, 0, 0])
    expect(nearestOnPolyline({ xs, ys }, 33, 9)).toBeCloseTo(0.33, 10)
    expect(nearestOnPolyline({ xs, ys }, 150, -40)).toBeCloseTo(1.5, 10)
  })

  it('clamps to the ends instead of extrapolating off the frontier', () => {
    const xs = Float64Array.from([0, 100])
    const ys = Float64Array.from([0, 0])
    expect(nearestOnPolyline({ xs, ys }, -500, 0)).toBe(0)
    expect(nearestOnPolyline({ xs, ys }, 5000, 0)).toBe(1)
    expect(at(path, -3, 0.03).ret).toBe(at(path, 0, 0.03).ret)
    expect(at(path, 1e6, 0.03).ret).toBe(at(path, path.points.length - 1, 0.03).ret)
  })
})

describe('the sparse primitives', () => {
  it('computes a bilinear form symmetrically', () => {
    const p = caps[0].doc.frontier
    const a: Sparse = toSparse(p[0], model)
    const b: Sparse = toSparse(p[p.length - 1], model)
    expect(bilinear(a, b, model)).toBeCloseTo(bilinear(b, a, model), 15)
  })

  it('computes the return of a solved point from mu alone', () => {
    for (const { doc } of caps) {
      for (const p of doc.frontier) {
        expect(Math.abs(dotMu(toSparse(p, model), model) - p.ret)).toBeLessThan(TOL)
      }
    }
  })
})
