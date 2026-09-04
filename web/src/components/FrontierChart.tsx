import { useMemo, useRef, useState } from 'react'
import type { Asset, Group } from '../types'
import type { Model, Path, Position } from '../portfolio'
import { at } from '../portfolio'
import { clampLabel, cmlEnd, declutter, linear, nearestOnPolyline, pad, pct, ticks } from '../viz'
import { Tooltip } from './Tooltip'

/** The efficient frontier, the capital market line, every asset as a dot, and a handle you drag.
 *
 * ENCODING, IN THE ORDER THE DECISIONS WERE MADE
 * ---------------------------------------------
 * Form: a scatter with an overlaid path. The data's job is a TRADE-OFF between two
 * continuous measures, which is the one job scatter does and bars cannot.
 *
 * Colour: only two things get a hue, and they are the two marks carrying the argument --
 * the frontier (categorical slot 1) and the capital market line plus its tangency portfolio
 * (slot 2). Both slots were validated against both surfaces at `--pairs all`.
 *
 * The 116 asset dots are deliberately GRAY. The tempting alternative is to colour them by
 * group, which is what `universe.py` records `group` for; three group hues plus the two mark
 * hues is five categorical slots, and five does not clear the all-pairs CVD gate. Rather
 * than ship a palette that fails, group identity is carried by the legend's filter buttons,
 * the tooltip and the table -- so it is never colour-alone, because it is never colour at
 * all -- and hue stays with the frontier. Filtering changes OPACITY, never hue, so a dot's
 * appearance never depends on what else is currently on screen.
 *
 * The dragged handle is primary ink with a 2px surface ring, so it reads as the cursor's
 * object rather than as a third data series.
 */

const W = 920
const H = 520
const M = { top: 20, right: 116, bottom: 54, left: 66 }
const PLOT = { x0: M.left, x1: W - M.right, y0: M.top, y1: H - M.bottom }

interface Props {
  assets: Asset[]
  path: Path
  model: Model
  rf: number
  t: number
  onT: (t: number) => void
  tangency: Position
  visibleGroups: Set<Group>
  fitFrontier: boolean
  benchmark: string | null
}

const GROUP_LABEL: Record<Group, string> = {
  equity: 'Equity',
  fixed_income: 'Fixed income',
  real_asset_fx: 'Real assets & FX',
}

export function FrontierChart({
  assets,
  path,
  model,
  rf,
  t,
  onT,
  tangency,
  visibleGroups,
  fitFrontier,
  benchmark,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [dragging, setDragging] = useState(false)
  const [hover, setHover] = useState<Asset | null>(null)

  const here = at(path, t, rf)
  const minVol = path.points[path.minVolIndex]

  const { sx, sy } = useMemo(() => {
    const fVols = Array.from(path.points, (p) => p.vol)
    const fRets = Array.from(path.points, (p) => p.ret)
    // Fitting to the frontier still has to include rf: the capital market line starts at
    // (0, rf), and a y-axis that excluded it would draw the CML leaving the plot.
    const volHi = fitFrontier ? Math.max(...fVols) : Math.max(...fVols, ...assets.map((a) => a.vol))
    const retLo = fitFrontier
      ? Math.min(...fRets, rf)
      : Math.min(...fRets, rf, ...assets.map((a) => a.ret))
    const retHi = fitFrontier
      ? Math.max(...fRets)
      : Math.max(...fRets, ...assets.map((a) => a.ret))
    return {
      sx: linear(pad([0, volHi], 0.06, { pinLow: 0 }), [PLOT.x0, PLOT.x1]),
      sy: linear(pad([retLo, retHi], 0.09), [PLOT.y1, PLOT.y0]),
    }
  }, [assets, path, rf, fitFrontier])

  /** The frontier in pixel space. Also the drag target: the handle's position is derived
   * from a continuous index into THIS array, and the weights from the same index. */
  const line = useMemo(() => {
    const xs = new Float64Array(path.points.length)
    const ys = new Float64Array(path.points.length)
    path.points.forEach((p, i) => {
      xs[i] = sx(p.vol)
      ys[i] = sy(p.ret)
    })
    return { xs, ys }
  }, [path, sx, sy])

  const linePath = useMemo(() => {
    let d = ''
    for (let i = 0; i < line.xs.length; i++) {
      d += `${i === 0 ? 'M' : 'L'}${line.xs[i].toFixed(2)} ${line.ys[i].toFixed(2)}`
    }
    return d
  }, [line])

  const toLocal = (e: { clientX: number; clientY: number }) => {
    const r = svgRef.current!.getBoundingClientRect()
    const k = W / r.width
    return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k }
  }

  const seek = (e: { clientX: number; clientY: number }) => {
    const { x, y } = toLocal(e)
    onT(nearestOnPolyline(line, x, y))
  }

  const last = path.points.length - 1
  const step = Math.max(last / 120, 1e-6)
  const onKey = (e: React.KeyboardEvent) => {
    const k = e.key
    const jump =
      k === 'ArrowRight' || k === 'ArrowUp'
        ? step
        : k === 'ArrowLeft' || k === 'ArrowDown'
          ? -step
          : k === 'PageUp'
            ? step * 10
            : k === 'PageDown'
              ? -step * 10
              : 0
    if (jump !== 0) {
      e.preventDefault()
      onT(Math.min(last, Math.max(0, t + jump)))
    } else if (k === 'Home') {
      e.preventDefault()
      onT(0)
    } else if (k === 'End') {
      e.preventDefault()
      onT(last)
    } else if (k === 't' || k === 'T') {
      e.preventDefault()
      onT(tangency.t)
    }
  }

  const cml = cmlEnd(rf, tangency.vol, tangency.ret, sx.domain[1])
  const hx = sx(here.vol)
  const hy = sy(here.ret)

  // Selective direct labels: the two frontier landmarks, the benchmark, and the extremes of
  // each axis. A label on every one of 116 dots is unreadable, and a label on none of them
  // makes the scatter an anonymous cloud.
  const labelled = useMemo(() => {
    const shown = assets.filter((a) => visibleGroups.has(a.group))
    if (!shown.length) return []
    const pick = [
      shown.reduce((b, a) => (a.ret > b.ret ? a : b)),
      shown.reduce((b, a) => (a.vol > b.vol ? a : b)),
      shown.reduce((b, a) => (a.vol < b.vol ? a : b)),
      shown.reduce((b, a) => (a.sharpe > b.sharpe ? a : b)),
      benchmark ? shown.find((a) => a.symbol === benchmark) : undefined,
    ]
    return [...new Map(pick.filter((a): a is Asset => !!a).map((a) => [a.symbol, a])).values()]
  }, [assets, visibleGroups, benchmark])

  const placedLabels = useMemo(() => {
    const wanted = [
      { key: 'tangency', text: 'tangency', x: sx(tangency.vol) + 13, y: sy(tangency.ret) - 11, fill: 'var(--cat-2)', bold: true },
      { key: 'minvar', text: 'min variance', x: sx(minVol.vol) + 10, y: sy(minVol.ret) + 16, fill: 'var(--cat-1)', bold: false },
      // BELOW the marker, not above: the capital market line leaves (0, rf) heading up and to
      // the right, so a label above the marker is drawn straight through the dashed line.
      { key: 'rf', text: `rf ${pct(rf, 2)}`, x: sx(0) + 8, y: sy(rf) + 15, fill: 'var(--cat-2)', bold: false },
      ...labelled.map((a) => ({
        key: a.symbol,
        text: a.symbol,
        x: sx(a.vol) + 8,
        y: sy(a.ret) - 7,
        fill: 'var(--ink-muted)',
        bold: false,
      })),
    ].map((l) => {
      const c = clampLabel(l.x, l.y, PLOT)
      return { ...l, x: c.x, y: c.y, anchor: c.anchor }
    })
    return declutter(wanted, PLOT)
  }, [labelled, minVol, rf, sx, sy, tangency])

  return (
    <div className={`chart-wrap${dragging ? ' dragging' : ''}`}>
      <div style={{ position: 'relative' }}>
        <svg
          ref={svgRef}
          className="chart"
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={
            `Efficient frontier of ${assets.length} assets. Risk on the horizontal axis, ` +
            `expected return on the vertical. The selected portfolio has volatility ` +
            `${pct(here.vol)} and expected return ${pct(here.ret)}.`
          }
          onPointerMove={(e) => dragging && seek(e)}
          onPointerUp={() => setDragging(false)}
          onPointerCancel={() => setDragging(false)}
        >
          <defs>
            <clipPath id="plot">
              <rect x={PLOT.x0} y={PLOT.y0} width={PLOT.x1 - PLOT.x0} height={PLOT.y1 - PLOT.y0} />
            </clipPath>
          </defs>

          {/* Recessive grid: it is a reading aid, not data. */}
          {ticks(sy.domain, 6).map((v) => (
            <g key={`gy${v}`}>
              <line className="grid" x1={PLOT.x0} x2={PLOT.x1} y1={sy(v)} y2={sy(v)} />
              <text x={PLOT.x0 - 10} y={sy(v) + 4} textAnchor="end">
                {pct(v, 0)}
              </text>
            </g>
          ))}
          {ticks(sx.domain, 7).map((v) => (
            <g key={`gx${v}`}>
              <line className="grid" x1={sx(v)} x2={sx(v)} y1={PLOT.y0} y2={PLOT.y1} />
              <text x={sx(v)} y={PLOT.y1 + 18} textAnchor="middle">
                {pct(v, 0)}
              </text>
            </g>
          ))}
          <line className="axis" x1={PLOT.x0} x2={PLOT.x1} y1={PLOT.y1} y2={PLOT.y1} />
          <line className="axis" x1={PLOT.x0} x2={PLOT.x0} y1={PLOT.y0} y2={PLOT.y1} />
          <text className="axis-title" x={PLOT.x0} y={H - 12}>
            Volatility (annualised σ)
          </text>
          <text
            className="axis-title"
            transform={`rotate(-90) translate(${-PLOT.y1} ${16})`}
            textAnchor="start"
          >
            Expected return (annualised)
          </text>

          <g clipPath="url(#plot)">
            {/* Capital market line: the ray from the risk-free rate through the tangency
                portfolio. Dashed because the part beyond the tangency point requires
                LEVERAGE, which this long-only universe cannot express -- the dash is the
                visual admission that the right-hand extension is hypothetical. */}
            <line
              x1={sx(0)}
              y1={sy(rf)}
              x2={sx(cml.x)}
              y2={sy(cml.y)}
              stroke="var(--cat-2)"
              strokeWidth={2}
              strokeDasharray="6 4"
              strokeLinecap="round"
            />

            {/* Asset dots. Gray, thin, and behind everything -- context, not subject. */}
            {assets.map((a) => {
              const on = visibleGroups.has(a.group)
              const isHover = hover?.symbol === a.symbol
              return (
                <circle
                  key={a.symbol}
                  cx={sx(a.vol)}
                  cy={sy(a.ret)}
                  r={isHover ? 5.5 : 3.5}
                  fill="var(--dot)"
                  fillOpacity={on ? (isHover ? 0.95 : 0.5) : 0.1}
                  stroke={isHover ? 'var(--surface-raised)' : 'none'}
                  strokeWidth={2}
                  style={{ cursor: on ? 'pointer' : 'default' }}
                  onPointerEnter={() => on && setHover(a)}
                  onPointerLeave={() => setHover(null)}
                />
              )
            })}

            <path
              d={linePath}
              fill="none"
              stroke="var(--cat-1)"
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />

            {/* Invisible fat stroke over the curve: a 2px line is a 2px hit target, which is
                not draggable on a trackpad and not remotely usable on a touch screen. */}
            <path
              d={linePath}
              fill="none"
              stroke="transparent"
              strokeWidth={26}
              strokeLinecap="round"
              style={{ cursor: 'grab' }}
              onPointerDown={(e) => {
                e.currentTarget.setPointerCapture(e.pointerId)
                setDragging(true)
                seek(e)
              }}
            />

            {/* Minimum-variance portfolio: the frontier's left end. Square, so it is not
                confused with the round tangency marker in a black-and-white print. */}
            <rect
              x={sx(minVol.vol) - 4.5}
              y={sy(minVol.ret) - 4.5}
              width={9}
              height={9}
              fill="var(--cat-1)"
              stroke="var(--surface-raised)"
              strokeWidth={2}
            />

            {/* Tangency portfolio: the highest Sharpe ratio at the current rf, found exactly
                along the interpolated path rather than snapped to a solved point.

                A RING, not a disc, and larger than the handle. The handle's default position
                IS the tangency, so on first load the two marks coincide -- and a filled disc
                of the same size was completely hidden underneath, leaving the legend promising
                an orange marker that was nowhere on the chart. A ring reads as orange with the
                dark handle sitting in its middle. */}
            <circle
              cx={sx(tangency.vol)}
              cy={sy(tangency.ret)}
              r={9}
              fill="none"
              stroke="var(--surface-raised)"
              strokeWidth={5}
            />
            <circle
              cx={sx(tangency.vol)}
              cy={sy(tangency.ret)}
              r={9}
              fill="none"
              stroke="var(--cat-2)"
              strokeWidth={2.5}
            />

            {/* Guides from the handle to both axes: this is the readout, so the position has
                to be legible without arithmetic. */}
            <line
              x1={PLOT.x0}
              x2={hx}
              y1={hy}
              y2={hy}
              stroke="var(--ink-muted)"
              strokeWidth={1}
              strokeDasharray="2 3"
            />
            <line
              x1={hx}
              x2={hx}
              y1={hy}
              y2={PLOT.y1}
              stroke="var(--ink-muted)"
              strokeWidth={1}
              strokeDasharray="2 3"
            />
          </g>

          {/* Direct labels, outside the clip so they can sit in the right margin. All of them
              go through one declutter pass, in priority order: the two frontier landmarks and
              the risk-free rate name marks the argument depends on, so they hold their ideal
              position and the asset symbols move around them. */}
          <circle cx={sx(0)} cy={sy(rf)} r={3} fill="var(--cat-2)" />
          {placedLabels.map((l) => (
            <text
              key={l.key}
              className="mark-label"
              x={l.x}
              y={l.y}
              textAnchor={l.anchor}
              fill={l.fill}
              style={l.bold ? { fontWeight: 600 } : undefined}
            >
              {l.text}
            </text>
          ))}

          {/* The handle. A real slider as far as assistive technology is concerned: it has a
              role, a range, and a value expressed in the units on the axes. */}
          <g
            className="handle"
            role="slider"
            tabIndex={0}
            aria-label="Portfolio position along the efficient frontier"
            aria-valuemin={0}
            aria-valuemax={last}
            aria-valuenow={Number(t.toFixed(3))}
            aria-valuetext={`volatility ${pct(here.vol)}, expected return ${pct(here.ret)}, Sharpe ${here.sharpe.toFixed(2)}`}
            onKeyDown={onKey}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId)
              setDragging(true)
            }}
          >
            <circle cx={hx} cy={hy} r={dragging ? 11 : 9} fill="var(--ink)" fillOpacity={0.12} />
            <circle
              cx={hx}
              cy={hy}
              r={6.5}
              fill="var(--ink)"
              stroke="var(--surface-raised)"
              strokeWidth={2}
            />
          </g>
        </svg>

        {hover && (
          <Tooltip
            x={(sx(hover.vol) / W) * 100}
            y={(sy(hover.ret) / H) * 100}
            title={hover.symbol}
            sub={`${hover.name} · ${GROUP_LABEL[hover.group]}`}
            rows={[
              ['Return', pct(hover.ret, 2)],
              ['Volatility', pct(hover.vol, 2)],
              ['Sharpe', hover.sharpe.toFixed(2)],
              ['In portfolio', weightOf(hover.symbol, path, t, model)],
            ]}
          />
        )}
      </div>

      <p className="note" style={{ padding: '10px 8px 2px' }}>
        Drag the handle along the curve, or focus it and use the arrow keys ({'←'}/
        {'→'}), <kbd>Home</kbd>/<kbd>End</kbd>, or <kbd>T</kbd> to jump to the tangency
        portfolio. Every position on the curve is a real long-only portfolio — the weights are
        interpolated, then risk and return are recomputed from the covariance matrix, so the
        numbers describe the holdings in the table below and not a point on a fitted line.
      </p>
    </div>
  )
}

function weightOf(symbol: string, path: Path, t: number, model: Model): string {
  const i = Math.min(Math.floor(t), path.points.length - 2)
  const u = t - i
  const j = model.index.get(symbol)
  if (j === undefined || path.points.length < 2) return '—'
  const a = path.points[i].w[symbol] ?? 0
  const b = path.points[i + 1].w[symbol] ?? 0
  const w = (1 - u) * a + u * b
  return w >= 1e-4 ? pct(w, 2) : 'not held'
}
