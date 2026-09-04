/** The reader's view state: what they chose, in a form that survives a reload, a shared link
 *  and a file on their own disk.
 *
 * THREE PLACES, ONE SHAPE, AND A DELIBERATE PRECEDENCE. The same `Config` is written to the URL
 * fragment, to `localStorage` and to a downloadable file, and on load the URL wins, then
 * storage, then the defaults. The URL has to win or a shared link would be silently overridden
 * by whatever the recipient happened to look at last, which is the one thing a shared link
 * must not do.
 *
 * THE POSITION IS A FRACTION, NOT A POINT INDEX. `pos` is 0..1 along the path, or null meaning
 * "wherever the tangency is". Switching the weight cap switches to a frontier with a different
 * point count, so a stored index would move the portfolio by an arbitrary amount -- the same
 * reason `App` holds `frac` rather than `t`. A null is not the same as 0: it tracks the
 * tangency as `rf` moves, which is the default the page opens on.
 *
 * VALIDATED AGAINST THE MANIFEST, NOT AGAINST A SCHEMA. A cap slug or a group name is only
 * meaningful if this build's data actually has it, so `coerce` takes the manifest and drops
 * anything it cannot honour. A config from a run with different caps therefore degrades to the
 * defaults instead of leaving the page in a state with no frontier in it.
 *
 * There is no backend here and no account. If one is added later, `parseFile`/`serialise` are
 * the seam: a store that returns a `Config` from anywhere at all slots in where `read` is
 * called, and nothing else in this file has to know.
 */
import type { Group, Manifest } from './types'

export const CONFIG_VERSION = 1
export const STORAGE_KEY = 'markowitz:config:v1'

export interface Config {
  /** Cap slug, e.g. `cap20`. Must be one this build shipped. */
  cap: string
  rf: number
  /** Position along the frontier as a fraction of the path, or null for "at the tangency". */
  pos: number | null
  groups: Group[]
  fitFrontier: boolean
  logScale: boolean
  /** Absent means "follow the operating system", which is the page's own default. */
  theme?: 'light' | 'dark'
}

export const RF_MIN = 0
export const RF_MAX = 0.08

export function defaults(manifest: Manifest): Config {
  return {
    cap: manifest.caps[0].slug,
    rf: manifest.rf_default,
    pos: null,
    groups: [...manifest.groups],
    fitFrontier: false,
    logScale: true,
  }
}

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)

/** Anything -> a Config this build can honour. Unknown fields ignored, bad fields defaulted.
 *  Returns the reasons it had to change something, so a file load can say so out loud rather
 *  than appearing to work. */
export function coerce(
  raw: unknown,
  manifest: Manifest,
): { config: Config; complaints: string[] } {
  const base = defaults(manifest)
  const complaints: string[] = []
  if (typeof raw !== 'object' || raw === null) {
    return { config: base, complaints: ['not an object'] }
  }
  const r = raw as Record<string, unknown>
  const out: Config = { ...base }

  if (typeof r.cap === 'string') {
    if (manifest.caps.some((c) => c.slug === r.cap)) out.cap = r.cap
    else complaints.push(`unknown weight cap "${r.cap}"`)
  }
  if (r.rf !== undefined) {
    const rf = Number(r.rf)
    if (Number.isFinite(rf)) {
      out.rf = clamp(rf, RF_MIN, RF_MAX)
      if (out.rf !== rf) complaints.push(`risk-free rate ${rf} clamped to ${out.rf}`)
    } else complaints.push('risk-free rate was not a number')
  }
  if (r.pos !== undefined) {
    if (r.pos === null) out.pos = null
    else {
      const p = Number(r.pos)
      if (Number.isFinite(p)) out.pos = clamp(p, 0, 1)
      else complaints.push('position was not a number')
    }
  }
  if (r.groups !== undefined) {
    const known = new Set<string>(manifest.groups)
    const picked = Array.isArray(r.groups)
      ? manifest.groups.filter((g) => (r.groups as unknown[]).includes(g))
      : []
    const unknown = Array.isArray(r.groups)
      ? (r.groups as unknown[]).filter((g) => typeof g !== 'string' || !known.has(g))
      : []
    if (unknown.length) complaints.push(`unknown group(s) ${unknown.join(', ')}`)
    // Never an empty selection: a scatter with nothing in it reads as a load failure, which is
    // the same rule `App.toggleGroup` enforces against the buttons.
    if (picked.length) out.groups = picked
    else complaints.push('no known groups selected; showing all')
  }
  if (typeof r.fitFrontier === 'boolean') out.fitFrontier = r.fitFrontier
  if (typeof r.logScale === 'boolean') out.logScale = r.logScale
  if (r.theme === 'light' || r.theme === 'dark') out.theme = r.theme
  else if (r.theme !== undefined) complaints.push(`unknown theme "${String(r.theme)}"`)

  return { config: out, complaints }
}

/* ------------------------------------------------------------------ the URL fragment */

/** `#cap=cap20&rf=0.0374&pos=0.6452&groups=equity,real_asset_fx&fit=0&log=1`
 *
 *  The FRAGMENT rather than the query string, for one reason that matters here: a fragment is
 *  never sent to the server, so a shared link cannot end up in someone's access log. It also
 *  survives GitHub Pages' static serving without any rewrite rule. Kept short and readable so
 *  it can be edited by hand -- `rf=0.05` in the address bar is a legitimate way to use this. */
export function toHash(c: Config, manifest: Manifest): string {
  const d = defaults(manifest)
  const p = new URLSearchParams()
  if (c.cap !== d.cap) p.set('cap', c.cap)
  if (c.rf !== d.rf) p.set('rf', String(Number(c.rf.toFixed(6))))
  if (c.pos !== null) p.set('pos', String(Number(c.pos.toFixed(6))))
  if (c.groups.length !== d.groups.length) p.set('groups', c.groups.join(','))
  if (c.fitFrontier) p.set('fit', '1')
  if (!c.logScale) p.set('log', '0')
  if (c.theme) p.set('theme', c.theme)
  return p.toString()
}

export function fromHash(hash: string, manifest: Manifest): Config | null {
  const body = hash.replace(/^#/, '')
  if (!body) return null
  const p = new URLSearchParams(body)
  if ([...p.keys()].length === 0) return null
  const raw: Record<string, unknown> = {}
  if (p.has('cap')) raw.cap = p.get('cap')
  if (p.has('rf')) raw.rf = Number(p.get('rf'))
  if (p.has('pos')) raw.pos = Number(p.get('pos'))
  if (p.has('groups')) raw.groups = (p.get('groups') as string).split(',').filter(Boolean)
  if (p.has('fit')) raw.fitFrontier = p.get('fit') === '1'
  if (p.has('log')) raw.logScale = p.get('log') === '1'
  if (p.has('theme')) raw.theme = p.get('theme')
  return coerce(raw, manifest).config
}

/* ------------------------------------------------------------------ the file on disk */

/** The saved file carries the DATA VERSION it was made against as well as its own format
 *  version. Not for validation -- a config from an older run is still perfectly usable -- but
 *  because a settings file whose numbers came from a different window is exactly the sort of
 *  thing someone will otherwise have to guess about a year from now. */
export function serialise(c: Config, manifest: Manifest): string {
  return `${JSON.stringify(
    {
      kind: 'markowitz-config',
      version: CONFIG_VERSION,
      saved_at: new Date().toISOString(),
      data_generated_at: manifest.generated_at,
      window: { start: manifest.window.start, end: manifest.window.end },
      config: c,
    },
    null,
    2,
  )}\n`
}

export interface ParseResult {
  config: Config | null
  /** Fatal reason the file was rejected outright. */
  error?: string
  /** Non-fatal adjustments, worth showing but not worth refusing over. */
  complaints: string[]
}

export function parseFile(text: string, manifest: Manifest): ParseResult {
  let doc: unknown
  try {
    doc = JSON.parse(text)
  } catch {
    return { config: null, error: 'that file is not JSON', complaints: [] }
  }
  if (typeof doc !== 'object' || doc === null) {
    return { config: null, error: 'that file is not a settings file', complaints: [] }
  }
  const d = doc as Record<string, unknown>
  if (d.kind !== 'markowitz-config') {
    // The portfolio export is the file most likely to be dropped here by mistake, since it is
    // the other JSON this page hands out. Say which one it is rather than "invalid".
    const hint =
      d.kind === 'markowitz-portfolio'
        ? 'that is a portfolio export, not a settings file'
        : 'that file is not a settings file'
    return { config: null, error: hint, complaints: [] }
  }
  if (d.version !== CONFIG_VERSION) {
    return {
      config: null,
      error: `settings file version ${String(d.version)}, but this page reads version ${CONFIG_VERSION}`,
      complaints: [],
    }
  }
  const { config, complaints } = coerce(d.config, manifest)
  return { config, complaints }
}

export function configFilename(): string {
  const d = new Date()
  const stamp = [
    d.getFullYear(),
    String(d.getMonth() + 1).padStart(2, '0'),
    String(d.getDate()).padStart(2, '0'),
  ].join('')
  return `frontier-settings-${stamp}.json`
}

/* ------------------------------------------------------------------ localStorage */

/** Storage is best-effort on purpose. Private browsing, a full quota and a blocked
 *  third-party-storage setting all throw here, and none of them is a reason to fail to draw a
 *  chart -- the URL and the file are the durable copies. */
export function readStored(manifest: Manifest): Config | null {
  try {
    const text = localStorage.getItem(STORAGE_KEY)
    if (!text) return null
    return coerce(JSON.parse(text), manifest).config
  } catch {
    return null
  }
}

/** Stores only a DEPARTURE from the default view, by the same test the URL uses: if `toHash` has
 *  nothing to say, the key is removed rather than written. Without that, Reset cannot work --
 *  it clears the key, then the state change it caused writes the defaults straight back, and
 *  "reset" would leave a stored copy behind forever. */
export function writeStored(c: Config, manifest: Manifest): void {
  try {
    if (toHash(c, manifest) === '') localStorage.removeItem(STORAGE_KEY)
    else localStorage.setItem(STORAGE_KEY, JSON.stringify(c))
  } catch {
    /* ignored -- see readStored */
  }
}

export function clearStored(): void {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    /* ignored */
  }
}

/** URL, then storage, then defaults. See the precedence note at the top of the file. */
export function initial(manifest: Manifest, hash: string): Config {
  return fromHash(hash, manifest) ?? readStored(manifest) ?? defaults(manifest)
}
