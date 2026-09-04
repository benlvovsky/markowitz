/** The JSON contract, mirrored from `pipeline/build.py`.
 *
 * These types are hand-written rather than generated, and they are the place a pipeline
 * change has to be reflected. The pipeline's own suite guarantees the SHAPE (every field
 * present, weights summing to one, reported performance recomputable); this file only
 * states what the browser expects to find.
 */

export interface Asset {
  symbol: string
  name: string
  asset_class: string
  group: Group
  ret: number
  vol: number
  sharpe: number
  first_date: string
  last_date: string
}

/** A group NAME, not an enumeration of the ones that exist.
 *
 * It was a union of the three literals in `etf_global.toml`, which made the compiler enforce
 * the opposite of the pipeline's rule that a universe is a file: a second universe partitions
 * the world differently, and `--universe sp500` would have needed this line edited before its
 * artifacts could be read at all. The groups a build ships are in `manifest.groups`, their
 * labels in `manifest.group_labels`, and `config.coerce` validates a stored filter against
 * that list at runtime -- which is where the check belongs, since the data arrives over HTTP.
 */
export type Group = string

export interface CapEntry {
  cap: number
  file: string
  slug: string
}

export interface Manifest {
  generated_at: string
  window: {
    start: string
    end: string
    requested_start: string
    trading_days: number
    n_returns: number
    years: number
  }
  estimation: {
    expected_returns: string
    covariance: string
    shrinkage_delta: number
    n_returns: number
  }
  rf_default: number
  rf_source: string
  /** Which universe file this bundle is a statement about, and the argument for its selection. */
  universe: {
    key: string
    name: string
    file: string
    description: string
  }
  benchmark: string | null
  groups: Group[]
  /** Complete: the pipeline derives a label for every declared group, so there is no fallback
   * here and nothing in this app holds a group map of its own. See `groupLabel`. */
  group_labels: Record<Group, string>
  n_universe: number
  n_assets: number
  caps: CapEntry[]
  /** Three separate claims, kept apart. `deliberate` is a judgement recorded in the universe
   * file with its evidence and is not counted in `n_universe`; the other two are this run's
   * measured failures of a symbol the universe did ask for. */
  excluded: {
    fetch_failed: Record<string, string>
    window_or_coverage: Record<string, string>
    deliberate: Record<string, string>
  }
  assets: Asset[]
}

/** `mu` and the full covariance matrix, so risk is recomputed here rather than trusted. */
export interface Stats {
  generated_at: string
  symbols: string[]
  mu: number[]
  cov: number[][]
}

export interface History {
  generated_at: string
  freq: 'monthly'
  /** The daily panel's first bar. The chained returns telescope to its total growth. */
  anchor: string
  /** The DAILY window in 252-day years -- the exponent that reproduces `mu`. */
  years: number
  partial_months: string[]
  dates: string[]
  symbols: string[]
  returns: Record<string, number[]>
}

/** One solved frontier point. `w` is sparse: only positions the optimiser actually opened. */
export interface FrontierPoint {
  ret: number
  vol: number
  sharpe: number
  w: Record<string, number>
}

export interface FrontierDoc {
  /** The run that produced it. Every file in a bundle carries the same stamp, and `loadBundle`
   * refuses a bundle where they disagree -- see there for why a weekly cron needs this. */
  generated_at: string
  weight_cap: number
  risk_free_rate: number
  n_requested: number
  n_points: number
  pruned: number
  failed_targets: number[]
  return_range: [number, number]
  frontier: FrontierPoint[]
  min_vol_index: number
  max_sharpe_index: number
}

export interface Bundle {
  manifest: Manifest
  stats: Stats
  history: History
  frontiers: Record<string, FrontierDoc>
}
