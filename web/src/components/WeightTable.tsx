import type { Asset } from '../types'
import { pct } from '../viz'

interface Props {
  weights: { symbol: string; weight: number }[]
  bySymbol: Map<string, Asset>
  cap: number
}

const GROUP_LABEL: Record<string, string> = {
  equity: 'Equity',
  fixed_income: 'Fixed income',
  real_asset_fx: 'Real assets & FX',
}

/** What you actually hold at the handle's position.
 *
 * The bar is a magnitude encoding on a shared baseline scaled to the CAP, not to the largest
 * holding: rescaling to the largest would make a 9% position at cap=10% look identical to a
 * 95% position at cap=100%, which hides the single most important thing the cap selector
 * does. A "pinned" marker calls out positions sitting exactly at the constraint, because a
 * portfolio whose weights are mostly determined by the cap rather than by the data is one a
 * reader should discount, and that is not visible from the numbers alone.
 */
export function WeightTable({ weights, bySymbol, cap }: Props) {
  const total = weights.reduce((s, w) => s + w.weight, 0)
  const pinned = weights.filter((w) => w.weight >= cap - 5e-4).length
  return (
    <>
      <p className="caption">
        {weights.length} position{weights.length === 1 ? '' : 's'} summing to {pct(total, 2)}
        {pinned > 0 && (
          <>
            {' '}· <strong>{pinned}</strong> pinned at the {pct(cap, 0)} cap, so {pinned === 1 ? 'that weight is' : 'those weights are'} set by
            the constraint rather than by the estimates
          </>
        )}
        .
      </p>
      <div className="table-scroll" style={{ maxHeight: 340 }}>
        <table>
          <thead>
            <tr>
              <th className="left">Symbol</th>
              <th className="left">Name</th>
              <th className="left">Group</th>
              <th>Weight</th>
              <th className="left" style={{ width: '30%' }}>
                &nbsp;
              </th>
            </tr>
          </thead>
          <tbody>
            {weights.map((w) => {
              const a = bySymbol.get(w.symbol)
              const atCap = w.weight >= cap - 5e-4
              return (
                <tr key={w.symbol}>
                  <td className="sym">{w.symbol}</td>
                  <td className="left">{a?.name ?? '—'}</td>
                  <td className="left" style={{ color: 'var(--ink-muted)' }}>
                    {a ? (GROUP_LABEL[a.group] ?? a.group) : '—'}
                  </td>
                  <td>
                    {pct(w.weight, 2)}
                    {atCap && (
                      <span title="at the per-asset cap" style={{ color: 'var(--cat-2)' }}>
                        {' '}
                        ●
                      </span>
                    )}
                  </td>
                  <td className="bar-cell">
                    <span
                      className="bar"
                      style={{ width: `${Math.max(1.5, (w.weight / cap) * 100)}%` }}
                      aria-hidden="true"
                    />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </>
  )
}
