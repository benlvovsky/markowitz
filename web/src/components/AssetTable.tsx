import { useMemo, useState } from 'react'
import type { Asset, Group } from '../types'
import { pct } from '../viz'

type Key = 'symbol' | 'name' | 'asset_class' | 'ret' | 'vol' | 'sharpe' | 'first_date' | 'weight'

interface Props {
  assets: Asset[]
  held: Map<string, number>
  visibleGroups: Set<Group>
}

const COLS: { key: Key; label: string; left?: boolean }[] = [
  { key: 'symbol', label: 'Symbol', left: true },
  { key: 'name', label: 'Name', left: true },
  { key: 'asset_class', label: 'Asset class', left: true },
  { key: 'ret', label: 'Return' },
  { key: 'vol', label: 'Volatility' },
  { key: 'sharpe', label: 'Sharpe' },
  { key: 'weight', label: 'Weight' },
  { key: 'first_date', label: 'History from' },
]

/** THE TABLE VIEW. Not an extra: the scatter above encodes 116 assets in position alone, and
 * a reader who cannot resolve overlapping dots -- or who is using a screen reader, or reading
 * a printout -- needs the same data addressable by name. Sorting by Sharpe is also the fastest
 * way to see the thing the chart makes you squint at: which assets the frontier is built from.
 *
 * `asset_class` is here rather than in the chart on purpose. It is the finer 11-way label, and
 * 11 categories cannot be encoded in hue at all; a table column is the honest place for it.
 */
export function AssetTable({ assets, held, visibleGroups }: Props) {
  const [sort, setSort] = useState<{ key: Key; desc: boolean }>({ key: 'sharpe', desc: true })

  const rows = useMemo(() => {
    const value = (a: Asset, k: Key): string | number =>
      k === 'weight' ? (held.get(a.symbol) ?? -1) : (a[k as keyof Asset] as string | number)
    return assets
      .filter((a) => visibleGroups.has(a.group))
      .slice()
      .sort((p, q) => {
        const x = value(p, sort.key)
        const y = value(q, sort.key)
        const c = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))
        return sort.desc ? -c : c
      })
  }, [assets, held, visibleGroups, sort])

  const click = (key: Key) =>
    setSort((s) => ({ key, desc: s.key === key ? !s.desc : key !== 'symbol' && key !== 'name' }))

  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {COLS.map((c) => (
              <th
                key={c.key}
                className={c.left ? 'left' : undefined}
                aria-sort={sort.key === c.key ? (sort.desc ? 'descending' : 'ascending') : 'none'}
              >
                <button onClick={() => click(c.key)}>
                  {c.label}
                  {sort.key === c.key ? (sort.desc ? ' ↓' : ' ↑') : ''}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((a) => {
            const w = held.get(a.symbol)
            return (
              <tr key={a.symbol} className={w ? 'held' : undefined}>
                <td className="sym">{a.symbol}</td>
                <td className="left">{a.name}</td>
                <td className="left" style={{ color: 'var(--ink-muted)' }}>
                  {a.asset_class.replace(/_/g, ' ')}
                </td>
                <td>{pct(a.ret, 2)}</td>
                <td>{pct(a.vol, 2)}</td>
                <td>{a.sharpe.toFixed(2)}</td>
                <td style={{ color: w ? 'var(--ink)' : 'var(--ink-muted)' }}>
                  {w ? pct(w, 2) : '—'}
                </td>
                <td style={{ color: 'var(--ink-muted)' }}>{a.first_date}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
