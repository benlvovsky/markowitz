/** The two formatters that had duplicates, and the guard against the duplicates coming back.
 *
 * Neither is interesting arithmetic. What is interesting is that both used to exist in several
 * places at once, and the copies disagreed:
 *
 *  - a hardcoded `{equity: 'Equity', ...}` map in four files, two of them differing in case, so
 *    the same asset was "fixed income" in the CSV and "Fixed income" in the table -- and all four
 *    would have rendered blank for a universe with different groups, which is the exact promise
 *    `pipeline/universes/*.toml` is supposed to keep;
 *  - the window length as `.toFixed(1)` in the subtitle and `.toFixed(0)` in the caveat, so one
 *    page stated two different window lengths.
 *
 * So the tests are half behaviour and half a SOURCE SCAN. The scan is the only thing that can
 * fail when someone adds a fifth copy, because a fifth copy renders correctly for this universe.
 */
import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import type { Manifest } from './types'
import { groupLabel, yearsText } from './viz'

const SRC = dirname(fileURLToPath(import.meta.url))
const DATA = join(SRC, '..', 'public', 'data')
const manifest = JSON.parse(readFileSync(join(DATA, 'manifest.json'), 'utf8')) as Manifest

/** Every .ts/.tsx under src/, recursively, excluding the tests themselves.
 *
 * `text` has COMMENT LINES REMOVED. The scans below are looking for a literal that would be
 * rendered, and the comments explaining why not to write one contain the literal by necessity --
 * without this the guards flag the documentation that exists to prevent the thing. Crude
 * line-level stripping, which is exactly right for a needle that has to appear in emitted code.
 */
function sources(dir = SRC): { path: string; text: string }[] {
  const out: { path: string; text: string }[] = []
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name)
    if (e.isDirectory()) out.push(...sources(p))
    else if (/\.tsx?$/.test(e.name) && !/\.test\.tsx?$/.test(e.name)) {
      const text = readFileSync(p, 'utf8')
        .split('\n')
        .filter((l) => !/^\s*(\/\/|\/?\*)/.test(l))
        .join('\n')
      out.push({ path: p.slice(SRC.length + 1), text })
    }
  }
  return out
}

describe('group labels', () => {
  it('names every group the shipped build asks the page to show', () => {
    for (const g of manifest.groups) {
      expect(manifest.group_labels[g], g).toBeTruthy()
      expect(groupLabel(manifest.group_labels, g)).toBe(manifest.group_labels[g])
    }
    // Every asset's group is one of them, or a filter button exists that nothing lands in.
    for (const a of manifest.assets) expect(manifest.groups, a.symbol).toContain(a.group)
  })

  it('falls back to the raw key rather than rendering an empty label', () => {
    // The state on a Pages deploy where a cached page is newer than the data it fetched. Ugly
    // and legible is the right failure; a blank filter button is not.
    expect(groupLabel({}, 'real_asset_fx')).toBe('real_asset_fx')
    expect(groupLabel({ equity: '' }, 'equity')).toBe('equity')
  })

  it('names no group of this universe in the code, in either shape a copy takes', () => {
    // Two alternatives, and both are needed. A QUOTED name is a comparison or a lookup key; a
    // BARE name followed by a colon is an object literal, which is the shape all four of the
    // copies actually had (`equity: 'Equity'`) -- a needle requiring quotes would have missed
    // every one of them. What the pattern deliberately does not match is the word in prose:
    // "the equity risk premium" is English, and `manifest.groups` happens to contain an ordinary
    // English word, so a bare `\bequity\b` flags the page's own text and guards nothing.
    const needle = new RegExp(
      manifest.groups.map((g) => `(['"]${g}['"]|\\b${g}\\s*:)`).join('|'),
    )
    const offenders = sources()
      .filter(({ text }) => needle.test(text))
      .map(({ path }) => path)
    expect(offenders).toEqual([])
  })
})

describe('the window length', () => {
  it('is stated with one rounding, so two places on the page cannot disagree', () => {
    expect(yearsText(15.44)).toBe('15.4')
    expect(yearsText(15.66)).toBe('15.7')
    expect(yearsText(16)).toBe('16.0')
    expect(yearsText(manifest.window.years)).toBe(manifest.window.years.toFixed(1))
  })

  it('is never written into a string as a literal', () => {
    // `export.ts`'s JSON caveat said "15-year", in the one artifact a reader keeps after the page
    // is gone, and the weekly cron was going to make it false with nothing to notice.
    const offenders = sources()
      .filter(({ text }) => /\b1\d-year\b/.test(text))
      .map(({ path }) => path)
    expect(offenders).toEqual([])
  })
})
