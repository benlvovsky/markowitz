import { mult, pct } from '../viz'

interface Tile {
  label: string
  value: string
  sub?: string
}

/** The dragged portfolio's numbers. A stat tile rather than a chart, because a single
 * current value has no shape to plot -- see the dataviz "is it even a chart" question. */
export function StatTiles({ tiles }: { tiles: Tile[] }) {
  return (
    <div className="tiles">
      {tiles.map((t) => (
        <div className="tile" key={t.label}>
          <div className="tile-label">{t.label}</div>
          <div className="tile-value">{t.value}</div>
          {t.sub && <div className="tile-sub">{t.sub}</div>}
        </div>
      ))}
    </div>
  )
}

export const tile = {
  pct: (v: number, d = 2) => (Number.isFinite(v) ? pct(v, d) : '—'),
  ratio: (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '—'),
  mult,
}
