import { useMemo, useRef, useState } from 'react'
import type { History } from '../types'
import { linear, mult, pct, ticks } from '../viz'
import { Tooltip } from './Tooltip'

/** What the selected portfolio actually did, against buy-and-hold in the benchmark.
 *
 * This chart exists to undercut the one above it. The frontier is built from the sample
 * geometric mean and a shrunk sample covariance over the SAME window plotted here, so this
 * curve is not an out-of-sample test and cannot be read as one -- it is the in-sample fit,
 * and the optimiser knew the answer. Its job is to show the SHAPE of what "efficient" bought
 * you: where the drawdowns were, and whether beating the benchmark took a smooth ride or one
 * good decade. A reader who wants evidence needs a holdout, which this artifact does not have.
 *
 * COLOUR IS CONTINUOUS WITH THE FRONTIER CHART BY CONSTRUCTION. The portfolio is categorical
 * slot 1, the same hue as the frontier it was dragged from; the benchmark is the same gray as
 * the asset dots, because SPY IS one of those dots. Slot 2 stays reserved for the risk-free
 * rate and the capital market line across the whole page, so orange never means two things.
 */

const W = 920
const H = 300
const M = { top: 16, right: 74, bottom: 40, left: 66 }
const PLOT = { x0: M.left, x1: W - M.right, y0: M.top, y1: H - M.bottom }

interface Props {
  portfolio: Float64Array
  benchmark: Float64Array | null
  benchmarkSymbol: string | null
  history: History
  logScale: boolean
}

export function GrowthChart({ portfolio, benchmark, benchmarkSymbol, history, logScale }: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [cursor, setCursor] = useState<number | null>(null)

  // One label per year, drawn from the actual month strings rather than a generated range,
  // so a change in the pipeline's window cannot desynchronise the axis from the data.
  const yearTicks = useMemo(() => {
    const out: { i: number; year: string }[] = []
    let seen = ''
    history.dates.forEach((d, i) => {
      const y = d.slice(0, 4)
      if (y !== seen) {
        seen = y
        out.push({ i: i + 1, year: y })
      }
    })
    return out.filter((_, k) => k % Math.ceil(out.length / 9) === 0)
  }, [history.dates])

  const { sx, sy, yTicks } = useMemo(() => {
    const all = benchmark ? [...portfolio, ...benchmark] : [...portfolio]
    const lo = Math.min(...all)
    const hi = Math.max(...all)
    const x = linear([0, portfolio.length - 1], [PLOT.x0, PLOT.x1])
    if (logScale) {
      const y = linear([Math.log(lo), Math.log(hi)], [PLOT.y1, PLOT.y0])
      // Round multiples, spaced by powers of 2 so the labels stay short over a wide range.
      const vals: number[] = []
      for (let v = 2 ** Math.floor(Math.log2(lo)); v <= hi * 1.001; v *= 2) vals.push(v)
      return { sx: x, sy: (v: number) => y(Math.log(v)), yTicks: vals }
    }
    const y = linear([Math.min(lo, 1) * 0.98, hi * 1.02], [PLOT.y1, PLOT.y0])
    return { sx: x, sy: y, yTicks: ticks(y.domain, 6) }
  }, [portfolio, benchmark, logScale])

  const line = (series: Float64Array) => {
    let d = ''
    for (let i = 0; i < series.length; i++) {
      d += `${i === 0 ? 'M' : 'L'}${sx(i).toFixed(2)} ${sy(series[i]).toFixed(2)}`
    }
    return d
  }

  const onMove = (e: React.PointerEvent) => {
    const r = svgRef.current!.getBoundingClientRect()
    const x = ((e.clientX - r.left) * W) / r.width
    const i = Math.round(sx.invert(x))
    setCursor(i >= 0 && i < portfolio.length ? i : null)
  }

  // Index 0 is the anchor (the panel's first bar), so month label i corresponds to point i+1.
  const labelFor = (i: number) => (i === 0 ? history.anchor : history.dates[i - 1])

  return (
    <div className="chart-wrap">
      <div style={{ position: 'relative' }}>
        <svg
          ref={svgRef}
          className="chart"
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Growth of one unit invested. The selected portfolio ends at ${mult(portfolio[portfolio.length - 1])}${
            benchmark ? `, ${benchmarkSymbol} at ${mult(benchmark[benchmark.length - 1])}` : ''
          }.`}
          onPointerMove={onMove}
          onPointerLeave={() => setCursor(null)}
        >
          {yTicks.map((v) => (
            <g key={`y${v}`}>
              <line className="grid" x1={PLOT.x0} x2={PLOT.x1} y1={sy(v)} y2={sy(v)} />
              <text x={PLOT.x0 - 10} y={sy(v) + 4} textAnchor="end">
                {mult(v)}
              </text>
            </g>
          ))}
          {yearTicks.map((t) => (
            <text key={t.year} x={sx(t.i)} y={PLOT.y1 + 18} textAnchor="middle">
              {t.year}
            </text>
          ))}
          <line className="axis" x1={PLOT.x0} x2={PLOT.x1} y1={sy(1)} y2={sy(1)} />

          {benchmark && (
            <path
              d={line(benchmark)}
              fill="none"
              stroke="var(--dot)"
              strokeWidth={2}
              strokeLinejoin="round"
            />
          )}
          <path
            d={line(portfolio)}
            fill="none"
            stroke="var(--cat-1)"
            strokeWidth={2}
            strokeLinejoin="round"
          />

          {/* Direct labels at the right end: two series, so a legend AND direct labels. */}
          <text x={PLOT.x1 + 8} y={sy(portfolio[portfolio.length - 1]) + 4} fill="var(--cat-1)" style={{ fontWeight: 600 }}>
            {mult(portfolio[portfolio.length - 1])}
          </text>
          {benchmark && (
            <text x={PLOT.x1 + 8} y={sy(benchmark[benchmark.length - 1]) + 4} fill="var(--ink-muted)">
              {mult(benchmark[benchmark.length - 1])}
            </text>
          )}

          {cursor !== null && (
            <g>
              <line
                x1={sx(cursor)}
                x2={sx(cursor)}
                y1={PLOT.y0}
                y2={PLOT.y1}
                stroke="var(--border-strong)"
                strokeWidth={1}
              />
              <circle
                cx={sx(cursor)}
                cy={sy(portfolio[cursor])}
                r={4.5}
                fill="var(--cat-1)"
                stroke="var(--surface-raised)"
                strokeWidth={2}
              />
              {benchmark && (
                <circle
                  cx={sx(cursor)}
                  cy={sy(benchmark[cursor])}
                  r={4.5}
                  fill="var(--dot)"
                  stroke="var(--surface-raised)"
                  strokeWidth={2}
                />
              )}
            </g>
          )}
        </svg>

        {cursor !== null && (
          <Tooltip
            x={(sx(cursor) / W) * 100}
            y={(sy(portfolio[cursor]) / H) * 100}
            title={labelFor(cursor)}
            rows={
              [
                ['Portfolio', mult(portfolio[cursor])],
                ...(benchmark && benchmarkSymbol
                  ? ([
                      [benchmarkSymbol, mult(benchmark[cursor])],
                      ['Difference', pct(portfolio[cursor] / benchmark[cursor] - 1, 1)],
                    ] as [string, string][])
                  : []),
              ] as [string, string][]
            }
          />
        )}
      </div>

      <div className="legend">
        <span className="legend-item">
          <svg width="18" height="10" aria-hidden="true">
            <line x1="0" y1="5" x2="18" y2="5" stroke="var(--cat-1)" strokeWidth="2" />
          </svg>
          Selected portfolio
        </span>
        {benchmarkSymbol && (
          <span className="legend-item">
            <svg width="18" height="10" aria-hidden="true">
              <line x1="0" y1="5" x2="18" y2="5" stroke="var(--dot)" strokeWidth="2" />
            </svg>
            {benchmarkSymbol}, buy and hold
          </span>
        )}
      </div>
    </div>
  )
}
