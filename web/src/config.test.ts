/** The view state is persisted in three places, and the bugs all live in the seams.
 *
 * What is actually at risk here, none of it visible on screen:
 *
 * 1. A ROUND TRIP THAT LOSES SOMETHING. `toHash` deliberately omits any field equal to the
 *    default, so the link stays short and hand-editable; the failure mode of that optimisation
 *    is a field that reads back as the default when it was not. `pos` is the sharp case,
 *    because `null` ("track the tangency") and `0` ("the minimum-variance end") are both
 *    falsy and mean opposite things.
 * 2. PRECEDENCE. URL beats storage beats defaults. If storage ever leaked through a partial
 *    hash, a shared link would render differently for the recipient than for the sender, which
 *    is the one thing a shared link must not do.
 * 3. A LOAD THAT SILENTLY ADJUSTS SOMETHING. `coerce` is allowed to change any field it cannot
 *    honour, and every such change must arrive as a complaint the UI can show. A config from a
 *    build with different weight caps must degrade, not leave the page with no frontier in it.
 *
 * Validated against the shipped `manifest.json` rather than a fixture, like the rest of this
 * suite: the cap slugs and group names it accepts are the ones this build actually has, so a
 * pipeline change that renames a cap fails here.
 */
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { beforeEach, describe, expect, it } from 'vitest'
import {
  CONFIG_VERSION,
  RF_MAX,
  RF_MIN,
  RF_STEP,
  STORAGE_KEY,
  clearStored,
  coerce,
  configFilename,
  defaults,
  fromHash,
  initial,
  parseFile,
  readStored,
  serialise,
  toHash,
  writeStored,
  type Config,
} from './config'
import { portfolioJson, type Selection } from './export'
import { at, buildModel, buildPath, growth, maxDrawdown, realisedVol, tangency, weightsAt } from './portfolio'
import type { FrontierDoc, Group, History, Manifest, Stats } from './types'

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data')
const read = <T>(name: string): T => JSON.parse(readFileSync(join(DATA, name), 'utf8')) as T

const manifest = read<Manifest>('manifest.json')

/** An in-memory `localStorage`. Vitest runs in node here, where the global is absent -- and
 *  absent is a case the real code must survive too, so it is tested rather than stubbed away
 *  in `survives a browser with no usable storage at all`. */
class MemoryStorage {
  private map = new Map<string, string>()
  getItem(k: string) {
    return this.map.get(k) ?? null
  }
  setItem(k: string, v: string) {
    this.map.set(k, v)
  }
  removeItem(k: string) {
    this.map.delete(k)
  }
}

/** Every non-empty subset of the shipped groups, in manifest order. `toHash` decides "the
 *  groups are the default" by comparing LENGTH, which is only sound because `coerce` filters
 *  `manifest.groups` in order and cannot produce a short list twice over -- so the round trip
 *  is checked on all of them rather than on one. */
function groupSubsets(groups: Group[]): Group[][] {
  const out: Group[][] = []
  for (let mask = 1; mask < 1 << groups.length; mask++) {
    out.push(groups.filter((_, i) => mask & (1 << i)))
  }
  return out
}

describe('the URL fragment', () => {
  it('is empty for the default view, so a shared link only ever carries a departure from it', () => {
    expect(toHash(defaults(manifest), manifest)).toBe('')
    expect(fromHash('', manifest)).toBeNull()
    expect(fromHash('#', manifest)).toBeNull()
  })

  it('round-trips every field, including the two that mean opposite things', () => {
    const cases: Config[] = [
      { ...defaults(manifest), pos: null, rf: 0.055 },
      { ...defaults(manifest), pos: 0 },
      { ...defaults(manifest), pos: 1 },
      { ...defaults(manifest), pos: 0.641_025, rf: 0, fitFrontier: true, logScale: false },
      { ...defaults(manifest), theme: 'dark' },
      { ...defaults(manifest), theme: 'light' },
      { ...defaults(manifest), cap: manifest.caps[manifest.caps.length - 1].slug },
    ]
    for (const c of cases) {
      const back = fromHash(`#${toHash(c, manifest)}`, manifest)
      expect(back, JSON.stringify(c)).toEqual(c)
    }
  })

  it('distinguishes "at the tangency" from "at the low-risk end"', () => {
    const atTangency = { ...defaults(manifest), rf: 0.05, pos: null }
    const atTheEnd = { ...defaults(manifest), rf: 0.05, pos: 0 }
    expect(toHash(atTangency, manifest)).not.toBe(toHash(atTheEnd, manifest))
    expect(fromHash(`#${toHash(atTangency, manifest)}`, manifest)!.pos).toBeNull()
    expect(fromHash(`#${toHash(atTheEnd, manifest)}`, manifest)!.pos).toBe(0)
  })

  it('round-trips every group selection this build can produce', () => {
    for (const groups of groupSubsets(manifest.groups)) {
      const c: Config = { ...defaults(manifest), groups }
      const hash = toHash(c, manifest)
      // Only the FULL selection is the default, so only it may drop out of the link. Asserted
      // rather than assumed, because `toHash` decides that by length.
      expect(hash.includes('groups='), groups.join('+')).toBe(groups.length !== manifest.groups.length)
      // Through `initial`, the real entry point, over EMPTY storage -- stated rather than left
      // to the runner's environment, since `initial` falls back to storage before the defaults.
      const got = withStorage(new MemoryStorage(), () => initial(manifest, `#${hash}`))
      expect(got.groups, groups.join('+')).toEqual(groups)
    }
  })

  it('is short and hand-editable rather than an encoded blob', () => {
    const c: Config = { ...defaults(manifest), rf: 0.05, pos: 0.5, fitFrontier: true }
    const hash = toHash(c, manifest)
    expect(hash).toMatch(/^[\w=&.,-]+$/)
    expect(hash.length).toBeLessThan(80)
    expect(fromHash('#rf=0.05', manifest)!.rf).toBe(0.05)
  })

  it('determines the whole view rather than patching whatever the reader last had', () => {
    const store = new MemoryStorage()
    const stored: Config = {
      ...defaults(manifest),
      cap: manifest.caps[manifest.caps.length - 1].slug,
      pos: 0.9,
      fitFrontier: true,
      theme: 'dark',
    }
    store.setItem(STORAGE_KEY, JSON.stringify(stored))
    // A link that names only `rf` must not inherit the recipient's cap, position or theme.
    const got = withStorage(store, () => initial(manifest, '#rf=0.06'))
    expect(got).toEqual({ ...defaults(manifest), rf: 0.06 })
  })
})

describe('coerce', () => {
  it('leaves the defaults alone and complains about nothing', () => {
    const d = defaults(manifest)
    const { config, complaints } = coerce(d, manifest)
    expect(config).toEqual(d)
    expect(complaints).toEqual([])
  })

  it('accepts every cap and group the shipped data actually has', () => {
    for (const cap of manifest.caps) {
      const { config, complaints } = coerce({ cap: cap.slug }, manifest)
      expect(config.cap, cap.slug).toBe(cap.slug)
      expect(complaints, cap.slug).toEqual([])
    }
    const { config, complaints } = coerce({ groups: manifest.groups }, manifest)
    expect(config.groups).toEqual(manifest.groups)
    expect(complaints).toEqual([])
  })

  it('says out loud every field it could not honour', () => {
    const { config, complaints } = coerce(
      { cap: 'cap37', rf: 3.74, pos: 12, groups: ['equity', 'crypto'], theme: 'sepia' },
      manifest,
    )
    expect(config.cap).toBe(defaults(manifest).cap)
    expect(config.rf).toBe(RF_MAX)
    expect(config.pos).toBe(1)
    expect(config.groups).toEqual(['equity'])
    expect(config.theme).toBeUndefined()
    const said = complaints.join(' | ')
    expect(said).toContain('cap37')
    expect(said).toContain('3.74')
    expect(said).toContain('crypto')
    expect(said).toContain('sepia')
    // The position too, which used to clamp in silence. `#pos=12` in a hand-edited link moves the
    // reader's selection to the frontier's far end, and a load that relocated it without saying
    // so is a load that lied about honouring the link.
    expect(said).toContain('position 12')
  })

  it('never leaves the page with nothing to draw', () => {
    for (const groups of [[], ['nonsense'], 'equity', 42, null]) {
      const { config, complaints } = coerce({ groups }, manifest)
      expect(config.groups.length, JSON.stringify(groups)).toBeGreaterThan(0)
      if (groups !== null) expect(complaints.length, JSON.stringify(groups)).toBeGreaterThan(0)
    }
    expect(coerce(null, manifest).config).toEqual(defaults(manifest))
    expect(coerce('a string', manifest).config).toEqual(defaults(manifest))
    expect(coerce(undefined, manifest).config).toEqual(defaults(manifest))
  })

  it('rejects a risk-free rate that is not a number rather than propagating NaN', () => {
    for (const rf of ['soon', {}, NaN, Infinity]) {
      const { config, complaints } = coerce({ rf }, manifest)
      expect(Number.isFinite(config.rf), String(rf)).toBe(true)
      expect(complaints.length, String(rf)).toBeGreaterThan(0)
    }
  })

  it('keeps the risk-free rate inside the range the slider can express', () => {
    // Out of range and back is a silent lie otherwise: the page would render an rf the slider
    // cannot show, so the CML would not match the control that claims to set it.
    expect(coerce({ rf: -1 }, manifest).config.rf).toBe(0)
    expect(coerce({ rf: 99 }, manifest).config.rf).toBe(RF_MAX)
    expect(coerce({ rf: 0.037 }, manifest).config.rf).toBe(0.037)
  })

  it('opens on a risk-free rate the slider can actually return to', () => {
    // A CROSS-LAYER assertion, and the only place the two halves meet. `build.fetch_rf` rounds
    // the T-bill yield to 4 dp precisely because the slider steps in basis points; at 5 dp the
    // default lands between two positions, so a reader who nudges the slider can never get back
    // to the page's own default and the URL then carries an `rf=` forever. Nothing else fails if
    // either side changes its precision, because both values are individually reasonable.
    const rf = defaults(manifest).rf
    expect(rf).toBe(manifest.rf_default)
    expect(rf).toBeGreaterThanOrEqual(RF_MIN)
    expect(rf).toBeLessThanOrEqual(RF_MAX)
    expect(Math.abs(Math.round(rf / RF_STEP) * RF_STEP - rf)).toBeLessThan(RF_STEP / 100)
    // And the slider's own domain has to be an exact number of steps, or its top end is a
    // position the reader can approach and not reach.
    expect(Number.isInteger(Math.round((RF_MAX - RF_MIN) / RF_STEP))).toBe(true)
    expect(Math.abs((RF_MAX - RF_MIN) / RF_STEP - Math.round((RF_MAX - RF_MIN) / RF_STEP)))
      .toBeLessThan(1e-6)
  })
})

describe('the settings file', () => {
  it('round-trips through serialise and parseFile', () => {
    const c: Config = {
      cap: manifest.caps[manifest.caps.length - 1].slug,
      rf: 0.0615,
      pos: 0.25,
      groups: [manifest.groups[0]],
      fitFrontier: true,
      logScale: false,
      theme: 'dark',
    }
    const result = parseFile(serialise(c, manifest), manifest)
    expect(result.error).toBeUndefined()
    expect(result.complaints).toEqual([])
    expect(result.config).toEqual(c)
  })

  it('records what build it was made against, so a stale file can be recognised later', () => {
    const doc = JSON.parse(serialise(defaults(manifest), manifest)) as Record<string, unknown>
    expect(doc.kind).toBe('markowitz-config')
    expect(doc.version).toBe(CONFIG_VERSION)
    expect(doc.data_generated_at).toBe(manifest.generated_at)
    expect(doc.window).toEqual({ start: manifest.window.start, end: manifest.window.end })
    expect(String(doc.saved_at)).toMatch(/^\d{4}-\d{2}-\d{2}T/)
  })

  it('names the mistake when the portfolio export is dropped on it by accident', () => {
    // The actual shipped portfolio file, not a hand-made stand-in: this is the other JSON the
    // page hands out, and the two are one Downloads folder apart.
    const result = parseFile(portfolioJson(portfolioSelection()), manifest)
    expect(result.config).toBeNull()
    expect(result.error).toContain('portfolio export')
  })

  it('refuses a file that is not one of ours instead of guessing', () => {
    const cases: Array<[string, string]> = [
      ['not json at all {', 'not JSON'],
      ['[1, 2, 3]', 'not a settings file'],
      ['"a string"', 'not a settings file'],
      ['{"config": {"rf": 0.05}}', 'not a settings file'],
      [JSON.stringify({ kind: 'markowitz-config', version: 99, config: {} }), 'version'],
    ]
    for (const [text, expected] of cases) {
      const result = parseFile(text, manifest)
      expect(result.config, text.slice(0, 30)).toBeNull()
      expect(result.error, text.slice(0, 30)).toContain(expected)
    }
  })

  it('loads a config from a build with different caps, and says what it changed', () => {
    const foreign = JSON.stringify({
      kind: 'markowitz-config',
      version: CONFIG_VERSION,
      config: { cap: 'cap45', rf: 0.04, pos: 0.5, groups: ['equity', 'commodities'] },
    })
    const result = parseFile(foreign, manifest)
    expect(result.error).toBeUndefined()
    expect(result.config).not.toBeNull()
    expect(result.config!.cap).toBe(defaults(manifest).cap)
    expect(result.config!.rf).toBe(0.04)
    expect(result.config!.pos).toBe(0.5)
    expect(result.complaints.length).toBeGreaterThan(0)
  })

  it('is filesystem-safe and dated', () => {
    expect(configFilename()).toMatch(/^frontier-settings-\d{8}\.json$/)
  })
})

describe('storage', () => {
  let store: MemoryStorage
  beforeEach(() => {
    store = new MemoryStorage()
  })

  it('round-trips, and clearing it puts the page back to the defaults', () => {
    const c: Config = { ...defaults(manifest), rf: 0.02, pos: 0.75, theme: 'light' }
    withStorage(store, () => {
      writeStored(c, manifest)
      expect(readStored(manifest)).toEqual(c)
      clearStored()
      expect(readStored(manifest)).toBeNull()
      expect(initial(manifest, '')).toEqual(defaults(manifest))
    })
  })

  it('holds a departure from the default view or nothing, so Reset can actually forget', () => {
    // Reset clears the key and then changes state, and that state change re-persists. If the
    // defaults were storable, the second write would undo the reset -- so writing the defaults
    // has to be a removal. Same test the URL uses: an empty hash means nothing to say.
    withStorage(store, () => {
      writeStored({ ...defaults(manifest), rf: 0.02 }, manifest)
      expect(store.getItem(STORAGE_KEY)).not.toBeNull()
      writeStored(defaults(manifest), manifest)
      expect(store.getItem(STORAGE_KEY)).toBeNull()
      expect(readStored(manifest)).toBeNull()
    })
  })

  it('coerces what it read, because the reader can edit it and an old build can have written it', () => {
    store.setItem(STORAGE_KEY, JSON.stringify({ cap: 'cap37', rf: 99, groups: [] }))
    withStorage(store, () => {
      const got = readStored(manifest)
      expect(got).not.toBeNull()
      expect(got!.cap).toBe(defaults(manifest).cap)
      expect(got!.rf).toBe(RF_MAX)
      expect(got!.groups).toEqual(manifest.groups)
    })
  })

  it('survives a browser with no usable storage at all', () => {
    // Private browsing, a full quota and blocked site data all throw on access. None of them is
    // a reason to fail to draw a chart, so every path here is best-effort by design.
    const hostile = {
      getItem() {
        throw new DOMException('denied')
      },
      setItem() {
        throw new DOMException('quota')
      },
      removeItem() {
        throw new DOMException('denied')
      },
    }
    withStorage(hostile, () => {
      expect(readStored(manifest)).toBeNull()
      // Both paths: a departure hits `setItem`, the defaults hit `removeItem`.
      expect(() => writeStored({ ...defaults(manifest), rf: 0.03 }, manifest)).not.toThrow()
      expect(() => writeStored(defaults(manifest), manifest)).not.toThrow()
      expect(() => clearStored()).not.toThrow()
      expect(initial(manifest, '#rf=0.05')).toEqual({ ...defaults(manifest), rf: 0.05 })
      expect(initial(manifest, '')).toEqual(defaults(manifest))
    })
    // And with no `localStorage` global at all, which is a `TypeError` rather than a `DOMException`
    // and so takes a different path into the same `catch`.
    withStorage(undefined, () => {
      expect(readStored(manifest)).toBeNull()
      // Both paths: a departure hits `setItem`, the defaults hit `removeItem`.
      expect(() => writeStored({ ...defaults(manifest), rf: 0.03 }, manifest)).not.toThrow()
      expect(() => writeStored(defaults(manifest), manifest)).not.toThrow()
      expect(initial(manifest, '')).toEqual(defaults(manifest))
    })
  })

  it('prefers the URL, then storage, then the defaults', () => {
    const stored: Config = { ...defaults(manifest), rf: 0.01, pos: 0.3 }
    store.setItem(STORAGE_KEY, JSON.stringify(stored))
    withStorage(store, () => {
      expect(initial(manifest, '')).toEqual(stored)
      expect(initial(manifest, '#rf=0.07')).toEqual({ ...defaults(manifest), rf: 0.07 })
    })
  })

  it('does not survive a version bump silently', () => {
    // The key carries the version, so a v2 build reads nothing rather than misreading v1.
    expect(STORAGE_KEY).toContain(`v${CONFIG_VERSION}`)
  })
})

/* -------------------------------------------------------------- helpers */

/** Swaps `globalThis.localStorage` for the duration of `fn`.
 *
 * The DESCRIPTOR is saved and restored rather than the value: node ships `localStorage` as a
 * lazy accessor, so reading it to put it back is what prints the `--localstorage-file` warning,
 * and on a run where the flag were set it would open a file this suite has no business touching. */
function withStorage<T>(store: unknown, fn: () => T): T {
  const prev = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', {
    value: store,
    configurable: true,
    writable: true,
  })
  try {
    return fn()
  } finally {
    if (prev) Object.defineProperty(globalThis, 'localStorage', prev)
    else delete (globalThis as { localStorage?: unknown }).localStorage
  }
}

/** A real portfolio export, built the way `App` builds one, for the wrong-file test. */
function portfolioSelection(): Selection {
  const stats = read<Stats>('stats.json')
  const history = read<History>('history.json')
  const cap = manifest.caps[0]
  const doc = read<FrontierDoc>(cap.file)
  const model = buildModel(stats)
  const path = buildPath(doc, model)
  const rf = manifest.rf_default
  const t = tangency(path, rf).t
  const weights = weightsAt(path, t, model)
  const port = growth(weights, history)
  return {
    manifest,
    cap: doc.weight_cap,
    capSlug: cap.slug,
    rf,
    position: at(path, t, rf),
    nPoints: path.points.length,
    atTangency: true,
    weights,
    bySymbol: new Map(manifest.assets.map((a) => [a.symbol, a])),
    inSample: {
      growthOf1: port[port.length - 1],
      maxDrawdown: maxDrawdown(port),
      realisedVol: realisedVol(weights, history),
    },
  }
}
