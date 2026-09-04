import type { Bundle, FrontierDoc, History, Manifest, Stats } from './types'

/** Data lives beside the bundle in `public/data/`, committed by the pipeline.
 *
 * There is no API. The whole back end is a cron job that writes six JSON files into the
 * repo, and this is the entire client for it. BASE_URL rather than a leading slash so the
 * same build works at the domain root, under a GitHub Pages project subpath, and from
 * `vite preview` -- see vite.config.ts.
 */
const url = (name: string) => `${import.meta.env.BASE_URL}data/${name}`

async function getJson<T>(name: string): Promise<T> {
  const res = await fetch(url(name), { cache: 'no-cache' })
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status} ${res.statusText}`)
  return (await res.json()) as T
}

export async function loadBundle(): Promise<Bundle> {
  const manifest = await getJson<Manifest>('manifest.json')

  // The manifest names the frontier files, so a change to `--caps` in the pipeline needs no
  // change here. Fetched in parallel: they are independent, and the four of them together
  // are smaller than one photograph.
  const [stats, history, ...docs] = await Promise.all([
    getJson<Stats>('stats.json'),
    getJson<History>('history.json'),
    ...manifest.caps.map((c) => getJson<FrontierDoc>(c.file)),
  ])

  const frontiers: Record<string, FrontierDoc> = {}
  manifest.caps.forEach((c, i) => {
    frontiers[c.slug] = docs[i]
  })

  assertOneRun(manifest, stats, history, frontiers)
  if (stats.symbols.length !== manifest.assets.length) {
    throw new Error(
      `stats.json has ${stats.symbols.length} symbols, manifest has ${manifest.assets.length}: ` +
        'the two files are from different pipeline runs',
    )
  }
  return { manifest, stats, history, frontiers }
}

/** Refuse a MIXED BUNDLE: files from two different pipeline runs, loaded together.
 *
 * The counts above cannot see it. `update-data` runs weekly over a universe whose symbol set
 * rarely changes, so last week's cached `stats.json` against this week's frontier has exactly
 * the right number of symbols in exactly the right order, and every calculation on this page
 * succeeds while silently pricing today's weights with last week's covariance. Six files
 * fetched separately from a CDN with per-file cache lifetimes is precisely the situation that
 * produces it, and `no-cache` on the request is a request, not a guarantee.
 *
 * `generated_at` is one stamp taken once per run, so equality is the whole test. Throwing is
 * the right answer rather than picking the newest: there is nothing to fall back to -- the app
 * has no second source -- and a reload is what fixes it, which the error message says.
 */
function assertOneRun(
  manifest: Manifest,
  stats: Stats,
  history: History,
  frontiers: Record<string, FrontierDoc>,
): void {
  const stamps: Record<string, string> = {
    'manifest.json': manifest.generated_at,
    'stats.json': stats.generated_at,
    'history.json': history.generated_at,
  }
  for (const [slug, doc] of Object.entries(frontiers)) stamps[`frontier_${slug}.json`] = doc.generated_at

  const missing = Object.entries(stamps).filter(([, s]) => !s).map(([n]) => n)
  if (missing.length) throw new Error(`no generated_at in ${missing.join(', ')}`)

  const distinct = [...new Set(Object.values(stamps))]
  if (distinct.length > 1) {
    const detail = Object.entries(stamps)
      .map(([name, s]) => `${name} ${s}`)
      .join('; ')
    throw new Error(
      `these files are from ${distinct.length} different pipeline runs, so the numbers on the ` +
        `page would mix them: ${detail}. Reload to pick up one consistent set.`,
    )
  }
}
