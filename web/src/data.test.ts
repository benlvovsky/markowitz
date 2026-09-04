/** The loader, and the one class of failure it exists to catch.
 *
 * `loadBundle` fetches six files independently, and the counts it used to check cannot see the
 * failure that actually happens on a static deploy: a stale copy of ONE of them. The universe's
 * symbol set rarely changes between weekly runs, so last week's `stats.json` has the right number
 * of symbols in the right order, every calculation on the page succeeds, and the reader is shown
 * today's weights priced with last week's covariance. `generated_at` is the only field that can
 * tell the two apart, which is why the pipeline stamps all six with one value.
 *
 * `fetch` is stubbed to serve the SHIPPED files from disk rather than a fixture, so the positive
 * test fails if the pipeline stops writing a stamp -- and the negative tests perturb one field of
 * one file, which is exactly the state a CDN produces.
 */
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { loadBundle } from './data'
import type { Manifest } from './types'

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data')
const readText = (name: string) => readFileSync(join(DATA, name), 'utf8')

const manifest = JSON.parse(readText('manifest.json')) as Manifest
const FILES = ['manifest.json', 'stats.json', 'history.json', ...manifest.caps.map((c) => c.file)]

/** Serve `public/data` over the stubbed `fetch`, with `override` replacing a file's text. */
function serve(override: Record<string, string> = {}) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const name = FILES.find((f) => url.endsWith(f))
      if (!name) return { ok: false, status: 404, statusText: 'Not Found' }
      const text = override[name] ?? readText(name)
      return { ok: true, status: 200, statusText: 'OK', json: async () => JSON.parse(text) }
    }),
  )
}

/** One file, with one field changed. Returns the text, so the rest of it is untouched. */
function withField(name: string, field: string, value: unknown): string {
  const doc = JSON.parse(readText(name)) as Record<string, unknown>
  if (value === undefined) delete doc[field]
  else doc[field] = value
  return JSON.stringify(doc)
}

afterEach(() => vi.unstubAllGlobals())

describe('loadBundle', () => {
  it('loads the shipped bundle, keyed by the slugs the manifest names', async () => {
    serve()
    const bundle = await loadBundle()
    expect(Object.keys(bundle.frontiers).sort()).toEqual(manifest.caps.map((c) => c.slug).sort())
    expect(bundle.stats.symbols.length).toBe(manifest.assets.length)
    for (const c of manifest.caps) {
      expect(bundle.frontiers[c.slug].weight_cap, c.slug).toBe(c.cap)
    }
  })

  it('every shipped file carries the same generated_at, which is what makes the check possible', async () => {
    serve()
    const b = await loadBundle()
    const stamps = [
      b.manifest.generated_at,
      b.stats.generated_at,
      b.history.generated_at,
      ...Object.values(b.frontiers).map((f) => f.generated_at),
    ]
    expect(stamps.length).toBe(FILES.length)
    expect(new Set(stamps).size).toBe(1)
    expect(stamps[0]).toMatch(/^\d{4}-\d{2}-\d{2}T/)
  })

  it('refuses a bundle whose files come from different runs, one file at a time', async () => {
    // Every file in turn, because the check has to be symmetric: a stale FRONTIER is as likely
    // as a stale `stats.json`, and a comparison written against the manifest alone would pass
    // whenever the manifest is the odd one out.
    for (const name of FILES) {
      serve({ [name]: withField(name, 'generated_at', '1970-01-01T00:00:00+00:00') })
      await expect(loadBundle(), name).rejects.toThrow(/different pipeline runs/)
    }
  })

  it('refuses a bundle with a file that carries no stamp at all, naming the file', async () => {
    for (const name of FILES) {
      serve({ [name]: withField(name, 'generated_at', undefined) })
      await expect(loadBundle(), name).rejects.toThrow(new RegExp(`no generated_at in .*${name}`))
    }
  })

  it('still catches a symbol-count mismatch, which is the other way the files can disagree', async () => {
    const stats = JSON.parse(readText('stats.json')) as { symbols: string[] }
    serve({ 'stats.json': withField('stats.json', 'symbols', stats.symbols.slice(1)) })
    await expect(loadBundle()).rejects.toThrow(/different pipeline runs/)
  })

  it('reports which file the HTTP error was for, since six requests fail identically otherwise', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 404, statusText: 'Not Found' })))
    await expect(loadBundle()).rejects.toThrow(/manifest\.json: HTTP 404/)
  })
})
