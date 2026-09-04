/** Scales, ticks, formatting and the geometry the drag needs. No chart library.
 *
 * Hand-rolled because the entire charting requirement is two linear scales, a polyline and
 * a hit test, and every library that draws those also brings its own opinions about the
 * DOM, its own tooltip, and a bundle larger than the data. The one non-obvious piece is
 * `nearestOnPolyline`, which is what makes the handle drag along the curve instead of
 * jumping to whichever solved point is closest.
 */

export interface Scale {
  (v: number): number
  invert: (px: number) => number
  domain: [number, number]
  range: [number, number]
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain
  const [r0, r1] = range
  const span = d1 - d0 || 1
  const f = ((v: number) => r0 + ((v - d0) / span) * (r1 - r0)) as Scale
  f.invert = (px: number) => d0 + ((px - r0) / (r1 - r0 || 1)) * span
  f.domain = domain
  f.range = range
  return f
}

/** 1-2-5 ticks. `count` is a target, not a promise -- the step is snapped to a round number
 * so the labels read as 0.05 / 0.10 / 0.15 rather than 0.043 / 0.086. */
export function ticks(domain: [number, number], count = 6): number[] {
  const [lo, hi] = domain
  if (!(hi > lo)) return [lo]
  const raw = (hi - lo) / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const norm = raw / mag
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  }
  return out
}

/** Pad a domain by a fraction of its span, optionally pinning one end (vol starts at 0). */
export function pad(
  domain: [number, number],
  frac = 0.06,
  opts: { pinLow?: number } = {},
): [number, number] {
  const [lo, hi] = domain
  const p = (hi - lo) * frac || Math.abs(hi) * frac || 0.01
  return [opts.pinLow !== undefined ? opts.pinLow : lo - p, hi + p]
}

/** A group's display name, read from the build that shipped it. THE ONLY GROUP NAMING IN THE APP.
 *
 * There were four copies of a hardcoded `{equity: 'Equity', ...}` map -- one each in the filter
 * row, the chart tooltip, the weight table and the CSV export -- and two of them disagreed about
 * capitalisation, so the same asset was "fixed income" in the download and "Fixed income" in the
 * table. Worse, they made "a universe is a file" false everywhere the reader could see: a second
 * universe with different groups would have rendered four blank labels with no error. The labels
 * now come from `manifest.group_labels`, which the pipeline fills for every declared group.
 *
 * The fallback is for a bundle written by an older pipeline than this build expects, which is a
 * real state on a Pages deploy: the raw key is ugly and legible, which is the right failure. `||`
 * rather than `??`, because an empty-string label is the same failure as a missing one and a
 * blank filter button is the one outcome with no clue in it.
 */
export const groupLabel = (labels: Record<string, string>, g: string) => labels[g] || g

/** The window's length in years, with ONE rounding wherever the page says it.
 *
 * It was `.toFixed(1)` in the subtitle and `.toFixed(0)` in the caveat five screens down, so the
 * same window read as "15.7 years" in one paragraph and a "16-year annualised mean return" in
 * the other, and a reader comparing them cannot tell which is the window they are looking at.
 * Worse, the JSON export's caveat had the number written into the string as "15-year", so it was
 * wrong the moment the cron rolled the window past 15.5. One decimal at every site: the subtitle
 * and the caveat then make the same claim, and neither can go stale.
 */
export const yearsText = (years: number) => years.toFixed(1)

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`
export const pctSigned = (v: number, digits = 1) =>
  `${v >= 0 ? '+' : '−'}${(Math.abs(v) * 100).toFixed(digits)}%`
export const num = (v: number, digits = 2) =>
  Number.isFinite(v) ? v.toFixed(digits) : '—'
export const mult = (v: number) => `${v.toFixed(2)}×`

/** A polyline in pixel space, plus the parameter each vertex corresponds to. */
export interface Polyline {
  xs: Float64Array
  ys: Float64Array
}

/** Closest point on a polyline to (px, py), returned as a CONTINUOUS vertex index.
 *
 * This is what "drag along the frontier" means geometrically: the cursor is projected onto
 * the drawn path rather than snapped to a vertex, so the handle moves smoothly and lands on
 * an interpolated portfolio. Returning a continuous index rather than a coordinate matters,
 * because the index is what indexes the WEIGHTS -- the interpolation happens in weight
 * space and the pixel position is derived from it, never the other way round.
 *
 * Projecting in PIXEL space, not data space: risk and return have different units and wildly
 * different numeric ranges, so "nearest" in data space would be dominated by whichever axis
 * happened to have larger numbers. Pixels are what the user is actually pointing at.
 */
export function nearestOnPolyline(line: Polyline, px: number, py: number): number {
  const { xs, ys } = line
  let bestT = 0
  let bestD = Infinity
  for (let i = 0; i + 1 < xs.length; i++) {
    const ax = xs[i]
    const ay = ys[i]
    const dx = xs[i + 1] - ax
    const dy = ys[i + 1] - ay
    const len2 = dx * dx + dy * dy
    const u = len2 > 0 ? Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0
    const qx = ax + u * dx
    const qy = ay + u * dy
    const d = (px - qx) ** 2 + (py - qy) ** 2
    if (d < bestD) {
      bestD = d
      bestT = i + u
    }
  }
  return bestT
}

/** Where a ray from (0, rf) through (vol, ret) leaves the plot on the right. */
export function cmlEnd(rf: number, vol: number, ret: number, xMax: number) {
  const slope = vol > 0 ? (ret - rf) / vol : 0
  return { x: xMax, y: rf + slope * xMax }
}

/** Push labels apart vertically until none overlaps an already-placed one.
 *
 * Direct labels are the whole reason a scatter is readable, and they collide the moment two
 * marks are close -- which in this chart is guaranteed, not unlucky: the minimum-variance
 * portfolio, the risk-free rate and the lowest-volatility asset are all pinned to the same
 * bottom-left corner by construction, and the first render duly stacked "rf 3.77%", "SHV" and
 * the min-variance square into one illegible pile.
 *
 * Greedy in INPUT ORDER, so the caller's order is a priority: the first label keeps its ideal
 * position and later ones move. Vertical-only, because moving a label sideways detaches it
 * from the mark it names -- the reader has no leader lines to follow.
 *
 * Widths are estimated from the character count at the mono font's advance rather than
 * measured, which is exact for a monospace face and avoids a layout read per frame.
 */
const CHAR_W = 6.62
const LINE_H = 13

export function declutter<T extends { x: number; y: number; text: string; anchor?: 'start' | 'end' }>(
  labels: T[],
  box: { y0: number; y1: number },
): (T & { y: number })[] {
  const placed: { x0: number; x1: number; y0: number; y1: number }[] = []
  const out: (T & { y: number })[] = []
  for (const l of labels) {
    const w = l.text.length * CHAR_W
    const x0 = l.anchor === 'end' ? l.x - w : l.x
    let y = l.y
    for (const dy of [0, -LINE_H, LINE_H, -2 * LINE_H, 2 * LINE_H, -3 * LINE_H, 3 * LINE_H, -4 * LINE_H, 4 * LINE_H]) {
      const cand = Math.min(Math.max(l.y + dy, box.y0 + 10), box.y1 - 2)
      const hit = placed.some(
        (p) => x0 < p.x1 && x0 + w > p.x0 && cand - LINE_H + 3 < p.y1 && cand + 3 > p.y0,
      )
      y = cand
      if (!hit) break
    }
    placed.push({ x0, x1: x0 + w, y0: y - LINE_H + 3, y1: y + 3 })
    out.push({ ...l, y })
  }
  return out
}

/** Nudge label anchors so a mark's text does not fall off the plot. */
export function clampLabel(
  x: number,
  y: number,
  box: { x0: number; x1: number; y0: number; y1: number },
): { x: number; y: number; anchor: 'start' | 'end' } {
  return {
    x: Math.min(Math.max(x, box.x0 + 4), box.x1 - 4),
    y: Math.min(Math.max(y, box.y0 + 12), box.y1 - 4),
    anchor: x > box.x1 - 90 ? 'end' : 'start',
  }
}
