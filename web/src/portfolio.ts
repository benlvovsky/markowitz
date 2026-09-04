/** Portfolio arithmetic in the browser. Nothing here is fitted, smoothed or approximated.
 *
 * THE ONE IDEA THIS FILE IS BUILT ON
 * ---------------------------------
 * The pipeline ships ~60 solved frontier points, each carrying its FULL weight vector. The
 * curve between two adjacent points is therefore not a drawing -- it is the family of
 * portfolios w(u) = (1-u)*w_i + u*w_{i+1}, and every one of them is a portfolio somebody
 * could hold: a convex combination of two long-only vectors summing to one is itself
 * long-only and sums to one, and it respects the same per-asset cap. So the dragged handle
 * always has a real answer to "what do I hold", which is the only question worth asking of
 * a point on an efficient frontier.
 *
 * That also makes the displayed risk EXACT rather than interpolated. Along a segment:
 *
 *     ret(u) = (1-u) r_i + u r_{i+1}                                  (linear in w)
 *     var(u) = (1-u)^2 S_ii + 2u(1-u) S_ij + u^2 S_jj                 (quadratic in w)
 *
 * with S_ii = w_i' C w_i, S_ij = w_i' C w_j, S_jj = w_j' C w_j precomputed once per
 * segment from the covariance matrix in stats.json. Three scalars per segment, and then
 * risk at any u is two multiplies. The chord lies very slightly ABOVE the true frontier --
 * by construction, since it is a chord of a convex curve -- and that gap is the honest cost
 * of a finite point count, not a rendering artifact. It is what `--points 60` buys down.
 *
 * The alternative that was rejected: fit a curve through the points and interpolate the
 * CURVE. It draws the same picture and produces a (vol, ret) pair with no portfolio behind
 * it, so the weight table would have to be faked or hidden.
 *
 * WHY THE RISK-FREE RATE IS A SLIDER AND NOT A REBUILD
 * ---------------------------------------------------
 * The frontier does not depend on rf at all -- it solves "minimise variance subject to a
 * return target", where rf does not appear. Only the tangency portfolio and the capital
 * market line do, and both are recoverable here: see `tangency`, which finds the exact
 * maximum-Sharpe point along the piecewise weight path in closed form.
 */
import type { FrontierDoc, FrontierPoint, History, Stats } from './types'

/** mu and the covariance matrix, flattened row-major for cache-friendly indexing. */
export interface Model {
  symbols: string[]
  index: Map<string, number>
  mu: Float64Array
  cov: Float64Array
  n: number
}

export function buildModel(stats: Stats): Model {
  const n = stats.symbols.length
  const cov = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    const row = stats.cov[i]
    for (let j = 0; j < n; j++) cov[i * n + j] = row[j]
  }
  return {
    symbols: stats.symbols,
    index: new Map(stats.symbols.map((s, i) => [s, i])),
    mu: Float64Array.from(stats.mu),
    cov,
    n,
  }
}

/** A portfolio as (asset index, weight) pairs.
 *
 * Sparse because the shipped vectors are: a 116-asset tangency portfolio holds 7 positions,
 * and the quadratic form over the sparse support is 49 multiplies instead of 13,456. That is
 * what keeps a drag at 60fps without a worker.
 */
export interface Sparse {
  idx: Int32Array
  val: Float64Array
}

export function toSparse(point: FrontierPoint, model: Model): Sparse {
  const entries = Object.entries(point.w)
  const idx = new Int32Array(entries.length)
  const val = new Float64Array(entries.length)
  entries.forEach(([sym, w], k) => {
    const i = model.index.get(sym)
    if (i === undefined) throw new Error(`frontier holds ${sym}, which is not in stats.json`)
    idx[k] = i
    val[k] = w
  })
  return { idx, val }
}

export function dotMu(w: Sparse, model: Model): number {
  let s = 0
  for (let k = 0; k < w.idx.length; k++) s += w.val[k] * model.mu[w.idx[k]]
  return s
}

/** a' C b. Symmetric in a and b; used for both the diagonal and cross segment terms. */
export function bilinear(a: Sparse, b: Sparse, model: Model): number {
  const { cov, n } = model
  let s = 0
  for (let p = 0; p < a.idx.length; p++) {
    const row = a.idx[p] * n
    const wa = a.val[p]
    let inner = 0
    for (let q = 0; q < b.idx.length; q++) inner += b.val[q] * cov[row + b.idx[q]]
    s += wa * inner
  }
  return s
}

/** The frontier as a continuously-traversable path. `t` in [0, n-1] indexes it. */
export interface Path {
  points: FrontierPoint[]
  w: Sparse[]
  ret: Float64Array
  /** w_i' C w_i, per point. */
  sii: Float64Array
  /** w_i' C w_{i+1}, per segment (length points.length - 1). */
  sij: Float64Array
  cap: number
  minVolIndex: number
}

export function buildPath(doc: FrontierDoc, model: Model): Path {
  const points = doc.frontier
  const w = points.map((p) => toSparse(p, model))
  const ret = Float64Array.from(w.map((v) => dotMu(v, model)))
  const sii = Float64Array.from(w.map((v) => bilinear(v, v, model)))
  const sij = new Float64Array(Math.max(0, w.length - 1))
  for (let i = 0; i + 1 < w.length; i++) sij[i] = bilinear(w[i], w[i + 1], model)
  return { points, w, ret, sii, sij, cap: doc.weight_cap, minVolIndex: doc.min_vol_index }
}

export interface Position {
  /** Continuous index along the path. */
  t: number
  ret: number
  vol: number
  sharpe: number
}

/** Split a continuous `t` into (segment, fraction), clamped to the path. */
function split(path: Path, t: number): { i: number; u: number } {
  const last = path.points.length - 1
  const c = Math.min(Math.max(t, 0), last)
  const i = Math.min(Math.floor(c), Math.max(0, last - 1))
  return { i, u: c - i }
}

/** Exact return and volatility of the interpolated portfolio at `t`. */
export function at(path: Path, t: number, rf: number): Position {
  const { i, u } = split(path, t)
  if (path.points.length === 1) {
    const vol = Math.sqrt(Math.max(path.sii[0], 0))
    return { t: 0, ret: path.ret[0], vol, sharpe: (path.ret[0] - rf) / vol }
  }
  const ret = (1 - u) * path.ret[i] + u * path.ret[i + 1]
  const varr =
    (1 - u) * (1 - u) * path.sii[i] + 2 * u * (1 - u) * path.sij[i] + u * u * path.sii[i + 1]
  const vol = Math.sqrt(Math.max(varr, 0))
  return { t: i + u, ret, vol, sharpe: vol > 0 ? (ret - rf) / vol : NaN }
}

/** The interpolated weight vector at `t`, largest holding first, sub-1bp positions dropped.
 *
 * The 1e-4 floor is the same one the pipeline uses: below it a weight is solver noise
 * rather than a position, and a table row saying "0.003%" invites a reader to believe the
 * optimiser meant it.
 */
export function weightsAt(
  path: Path,
  t: number,
  model: Model,
  floor = 1e-4,
): { symbol: string; weight: number }[] {
  const { i, u } = split(path, t)
  const acc = new Map<number, number>()
  const add = (w: Sparse, scale: number) => {
    if (scale === 0) return
    for (let k = 0; k < w.idx.length; k++) {
      acc.set(w.idx[k], (acc.get(w.idx[k]) ?? 0) + scale * w.val[k])
    }
  }
  if (path.points.length === 1) add(path.w[0], 1)
  else {
    add(path.w[i], 1 - u)
    add(path.w[i + 1], u)
  }
  return [...acc.entries()]
    .filter(([, v]) => v >= floor)
    .map(([j, v]) => ({ symbol: model.symbols[j], weight: v }))
    .sort((a, b) => b.weight - a.weight)
}

/** The exact maximum-Sharpe point on the path, for an arbitrary rf. Closed form.
 *
 * On segment i, with A = r_i - rf, B = r_{i+1} - r_i and var(u) = C + D u + E u^2
 * (C = S_ii, D = 2(S_ij - S_ii), E = S_ii - 2 S_ij + S_jj), the Sharpe ratio is
 * (A + Bu)/sqrt(C + Du + Eu^2). Setting the derivative to zero gives
 *
 *     B*var(u) - (A + Bu)*var'(u)/2 = 0
 *
 * and the u^2 terms cancel identically, leaving ONE linear root:
 *
 *     u* = (A D / 2 - B C) / (B D / 2 - A E)
 *
 * So the tangency portfolio is found by scanning segments, not by searching along them, and
 * the rf slider stays exact at any resolution rather than snapping to a sampled grid. E is
 * (w_i - w_j)' C (w_i - w_j) >= 0, so a zero denominator means the segment's Sharpe is
 * constant in u; the endpoints are always candidates, which covers it.
 */
export function tangency(path: Path, rf: number): Position {
  let best = at(path, 0, rf)
  const consider = (t: number) => {
    const p = at(path, t, rf)
    if (Number.isFinite(p.sharpe) && p.sharpe > best.sharpe) best = p
  }
  for (let i = 0; i < path.points.length; i++) consider(i)
  for (let i = 0; i + 1 < path.points.length; i++) {
    const A = path.ret[i] - rf
    const B = path.ret[i + 1] - path.ret[i]
    const C = path.sii[i]
    const D = 2 * (path.sij[i] - path.sii[i])
    const E = path.sii[i] - 2 * path.sij[i] + path.sii[i + 1]
    const den = (B * D) / 2 - A * E
    if (Math.abs(den) < 1e-18) continue
    const u = ((A * D) / 2 - B * C) / den
    if (u > 0 && u < 1) consider(i + u)
  }
  return best
}

/** Growth of one unit, monthly-rebalanced at fixed weights. Returns the cumulative path.
 *
 * FIXED WEIGHTS, REBALANCED MONTHLY, and that is a real assumption rather than a detail:
 * the portfolio return in month t is w . r_t, which is what holding those weights through
 * the month and resetting them at the end produces. Buy-and-hold would drift toward whatever
 * won, and would not be the portfolio the frontier point describes. Costs are zero here,
 * which flatters monthly rebalancing of a 12-holding portfolio slightly and flatters the
 * high-turnover capped variants more.
 *
 * The series is anchored at the daily panel's first bar (see fetch.monthly_returns), so the
 * final value of a single-asset portfolio compounds to exactly that asset's total growth
 * over the same window the optimiser measured. That identity is what makes this curve and
 * the frontier's `ret` axis two views of one number instead of two estimates.
 */
export function growth(
  weights: { symbol: string; weight: number }[],
  history: History,
): Float64Array {
  const series = weights
    .map((w) => ({ w: w.weight, r: history.returns[w.symbol] }))
    .filter((s): s is { w: number; r: number[] } => Array.isArray(s.r))
  const out = new Float64Array(history.dates.length + 1)
  out[0] = 1
  for (let t = 0; t < history.dates.length; t++) {
    let r = 0
    for (const s of series) r += s.w * s.r[t]
    out[t + 1] = out[t] * (1 + r)
  }
  return out
}

/** Annualised growth from a cumulative path, using the DAILY window's length in 252-day
 * years. `history.years` is that number; the monthly path telescopes to the same total
 * growth, so this reproduces the optimiser's `mu` for a single-asset portfolio. */
export function annualised(path: Float64Array, years: number): number {
  return Math.pow(path[path.length - 1], 1 / years) - 1
}

/** Realised annualised standard deviation of the monthly portfolio returns.
 *
 * Reported alongside the model volatility on purpose. The model number is a shrunk estimate
 * from DAILY returns; this one is the sample deviation of the 189 MONTHLY returns actually
 * plotted. They differ -- monthly sampling picks up autocorrelation that daily sampling
 * annualises away -- and showing both is more honest than picking one.
 */
export function realisedVol(
  weights: { symbol: string; weight: number }[],
  history: History,
): number {
  const series = weights
    .map((w) => ({ w: w.weight, r: history.returns[w.symbol] }))
    .filter((s): s is { w: number; r: number[] } => Array.isArray(s.r))
  const n = history.dates.length
  const rs = new Float64Array(n)
  for (let t = 0; t < n; t++) {
    let r = 0
    for (const s of series) r += s.w * s.r[t]
    rs[t] = r
  }
  const mean = rs.reduce((a, b) => a + b, 0) / n
  let ss = 0
  for (let t = 0; t < n; t++) ss += (rs[t] - mean) ** 2
  return Math.sqrt(ss / (n - 1)) * Math.sqrt(12)
}

/** Largest peak-to-trough fall in a cumulative path, as a positive fraction. */
export function maxDrawdown(path: Float64Array): number {
  let peak = path[0]
  let worst = 0
  for (let i = 1; i < path.length; i++) {
    if (path[i] > peak) peak = path[i]
    else worst = Math.max(worst, 1 - path[i] / peak)
  }
  return worst
}
