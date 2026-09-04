/** Positioned in PERCENT of the chart box, not pixels.
 *
 * The SVG scales with the container (`viewBox` plus `width: 100%`), so an internal
 * coordinate is not a CSS pixel. Percentages of the same box are, which keeps the tooltip
 * pinned to its mark at every window width without reading layout back from the DOM.
 */
interface Props {
  /** Percent of the chart box, 0-100. */
  x: number
  y: number
  title: string
  sub?: string
  rows: [string, string][]
}

export function Tooltip({ x, y, title, sub, rows }: Props) {
  // Flip to the left of the cursor past the midpoint so the box never leaves the plot.
  const flip = x > 62
  return (
    <div
      className="tooltip"
      style={{
        left: `${x}%`,
        top: `${y}%`,
        transform: `translate(${flip ? 'calc(-100% - 14px)' : '14px'}, -50%)`,
      }}
      role="status"
    >
      <div className="tooltip-title">{title}</div>
      {sub && <div className="tooltip-sub">{sub}</div>}
      {rows.map(([k, v]) => (
        <div className="tooltip-row" key={k}>
          <span>{k}</span>
          <b>{v}</b>
        </div>
      ))}
    </div>
  )
}
