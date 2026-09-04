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

  if (stats.symbols.length !== manifest.assets.length) {
    throw new Error(
      `stats.json has ${stats.symbols.length} symbols, manifest has ${manifest.assets.length}: ` +
        'the two files are from different pipeline runs',
    )
  }
  return { manifest, stats, history, frontiers }
}
