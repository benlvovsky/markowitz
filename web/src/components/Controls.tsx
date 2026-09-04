import type { ReactNode } from 'react'
import { RF_MAX, RF_MIN, RF_STEP } from '../config'
import type { CapEntry, Group } from '../types'
import { groupLabel, pct } from '../viz'

interface Props {
  caps: CapEntry[]
  cap: string
  onCap: (slug: string) => void
  rf: number
  rfDefault: number
  onRf: (rf: number) => void
  groups: Group[]
  /** `manifest.group_labels`. Passed in rather than looked up here: this component renders
   *  whatever groups the build shipped and holds no opinion about what they are called. */
  groupLabels: Record<string, string>
  visibleGroups: Set<Group>
  onToggleGroup: (g: Group) => void
  counts: Record<string, number>
  fitFrontier: boolean
  onFit: (v: boolean) => void
  /** Rendered as the last cell of the same row, so the settings buttons live with the knobs
   *  they save rather than in a bar of their own. */
  extra?: ReactNode
}

/** All the controls in ONE row above the charts, which is where a reader looks for them.
 *
 * The two knobs are not the same kind of thing, and the layout should not pretend they are:
 *
 *   - The RISK-FREE RATE is free. The frontier is the solution to "minimise variance subject
 *     to a return target", in which rf does not appear, so moving this slider re-derives the
 *     tangency portfolio and the capital market line from data already in the browser. No
 *     refetch, no re-solve, exact at any slider position.
 *   - The WEIGHT CAP is a constraint on the quadratic program, so each value is a separate
 *     solve and a separate file the pipeline had to write. That is why it is a three-way
 *     selector and not a slider: those are the three frontiers that exist.
 */
export function Controls({
  caps,
  cap,
  onCap,
  rf,
  rfDefault,
  onRf,
  groups,
  groupLabels,
  visibleGroups,
  onToggleGroup,
  counts,
  fitFrontier,
  onFit,
  extra,
}: Props) {
  return (
    <div className="controls">
      <div className="control">
        <span className="control-label" id="cap-label">
          Max weight per asset
        </span>
        <div className="segmented" role="group" aria-labelledby="cap-label">
          {caps.map((c) => (
            <button
              key={c.slug}
              aria-pressed={cap === c.slug}
              onClick={() => onCap(c.slug)}
              title={`Re-solved frontier with every weight capped at ${pct(c.cap, 0)}`}
            >
              {pct(c.cap, 0)}
            </button>
          ))}
        </div>
      </div>

      <div className="control">
        <label htmlFor="rf">
          Risk-free rate <span className="control-value">{pct(rf, 2)}</span>
        </label>
        <input
          id="rf"
          type="range"
          min={RF_MIN}
          max={RF_MAX}
          step={RF_STEP}
          value={rf}
          onChange={(e) => onRf(Number(e.target.value))}
        />
        <span className="control-value" style={{ fontSize: 11 }}>
          T-bill today: {pct(rfDefault, 2)}
        </span>
      </div>

      <div className="control">
        <span className="control-label" id="group-label">
          Show asset groups
        </span>
        <div style={{ display: 'flex', gap: 6 }} role="group" aria-labelledby="group-label">
          {groups.map((g) => (
            <button
              key={g}
              className="toggle"
              aria-pressed={visibleGroups.has(g)}
              onClick={() => onToggleGroup(g)}
            >
              {groupLabel(groupLabels, g)}{' '}
              <span style={{ color: 'var(--ink-muted)' }}>{counts[g] ?? 0}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="control">
        <span className="control-label">Axes</span>
        <button className="toggle" aria-pressed={fitFrontier} onClick={() => onFit(!fitFrontier)}>
          {fitFrontier ? 'Fitted to the frontier' : 'Fitted to every asset'}
        </button>
      </div>

      {extra}
    </div>
  )
}

/** The mark legend for the frontier chart. Shape as well as colour on every entry, so the
 * chart stays readable in grayscale, in forced-colors mode and under any CVD. */
export function FrontierLegend({ nAssets, hidden }: { nAssets: number; hidden: number }) {
  return (
    <div className="legend">
      <span className="legend-item">
        <svg width="20" height="10" aria-hidden="true">
          <line x1="0" y1="5" x2="20" y2="5" stroke="var(--cat-1)" strokeWidth="2" />
        </svg>
        Efficient frontier
      </span>
      <span className="legend-item">
        <svg width="12" height="12" aria-hidden="true">
          <rect x="1.5" y="1.5" width="9" height="9" fill="var(--cat-1)" />
        </svg>
        Minimum variance
      </span>
      <span className="legend-item">
        <svg width="20" height="10" aria-hidden="true">
          <line
            x1="0"
            y1="5"
            x2="20"
            y2="5"
            stroke="var(--cat-2)"
            strokeWidth="2"
            strokeDasharray="6 4"
          />
        </svg>
        Capital market line
      </span>
      <span className="legend-item">
        <svg width="14" height="14" aria-hidden="true">
          <circle cx="7" cy="7" r="5" fill="var(--cat-2)" />
        </svg>
        Tangency (max Sharpe)
      </span>
      <span className="legend-item">
        <svg width="14" height="14" aria-hidden="true">
          <circle cx="7" cy="7" r="5.5" fill="var(--ink)" />
        </svg>
        Your portfolio — drag it
      </span>
      <span className="legend-item">
        <svg width="12" height="12" aria-hidden="true">
          <circle cx="6" cy="6" r="3.5" fill="var(--dot)" fillOpacity="0.5" />
        </svg>
        Individual asset ({nAssets}
        {hidden > 0 ? `, ${hidden} hidden` : ''})
      </span>
    </div>
  )
}
