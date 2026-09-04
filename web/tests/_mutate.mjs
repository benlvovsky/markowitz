/** Mutation harness for the web suite: break one thing, confirm a NAMED test catches it.
 *
 *     node tests/_mutate.mjs                 # every mutant, ~40 s
 *     node tests/_mutate.mjs --only storage  # the ones whose name contains "storage"
 *     node tests/_mutate.mjs --list          # what it would run, and what should catch each
 *     node tests/_mutate.mjs --keep          # leave the staged copy behind to look at
 *
 * WHY THIS EXISTS. `web/src/*.test.ts` makes load-bearing claims -- that a shared link round-trips,
 * that a mixed bundle is refused, that no group name is hardcoded, that the download's numbers are
 * the page's numbers -- and a green suite is evidence of none of them until it has been seen to go
 * red for the right reason. `pipeline/tests/_mutate.py` has already found four pipeline tests that
 * guarded less than their names claimed; this is the same instrument pointed at the other half.
 * It is not a test suite and is not run in CI. It is a test OF the test suite, run by hand after
 * touching `config.ts`, `export.ts`, `data.ts` or `viz.ts`.
 *
 * THE CONTRACT, all four parts of it, copied from the pipeline harness because each one exists to
 * stop the harness reporting success while measuring nothing:
 *
 *  1. A needle that no longer appears EXACTLY ONCE is a hard error, not a skip. A stale mutation
 *     silently tests nothing while printing as caught, which is worse than no mutation at all.
 *  2. A mutant that does not PARSE is a hard error. It fails every test in its file, which looks
 *     exactly like a kill.
 *  3. The expected test's NAME has to appear in the failure output. "Something went red" is not
 *     the claim; "the test that says it guards this guards this" is.
 *  4. It runs against a STAGED COPY of the tree (`web/.mutate/`, gitignored), never the working
 *     tree. The pipeline harness does the same, for the same reason: an in-place mutation plus a
 *     restore in a `finally` still leaves the source broken if the process is killed between the
 *     two, and the diff that results looks like something someone meant.
 *
 * Needles are LITERAL SUBSTRINGS, not regexes. This source is full of `?`, `(`, `[`, `.`, `|` and
 * `$`, and an escaping slip in a mutation harness is indistinguishable from a passing check.
 */
import { spawnSync } from 'node:child_process'
import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const WEB = dirname(dirname(fileURLToPath(import.meta.url)))
const STAGE = join(WEB, '.mutate')
/** Everything the suite reads. `public` is in here because both suites deliberately test the
 *  SHIPPED artifacts rather than fixtures, so a staged tree without the data has no tests in it. */
const STAGED = ['src', 'public', 'index.html', 'package.json', 'tsconfig.json', 'vite.config.ts']

const manifest = JSON.parse(readFileSync(join(WEB, 'public', 'data', 'manifest.json'), 'utf8'))

/** A hardcoded group-label map, built from the manifest so it cannot go stale.
 *
 * Generated rather than written out because this file must not contain the literal either: the
 * guard it checks is a scan of `src/` for a group name, and a harness that hardcoded
 * `equity: 'Equity'` would be one more copy of the thing the scan exists to forbid -- in a file
 * whose whole purpose is to prove the scan works. */
const hardcodedLabels = manifest.groups
  .map((g) => `${g}: ${JSON.stringify(manifest.group_labels[g])}`)
  .join(', ')

/** name, file, needle, replacement, the test that must catch it. */
const MUTATIONS = [
  /* ------------------------------------------------------------ config.ts: the URL fragment */
  [
    'pos-zero-treated-as-absent',
    'config.ts',
    "  if (c.pos !== null) p.set('pos', String(Number(c.pos.toFixed(6))))",
    "  if (c.pos) p.set('pos', String(Number(c.pos.toFixed(6))))",
    'distinguishes "at the tangency" from "at the low-risk end"',
  ],
  [
    'groups-omitted-unless-a-single-one',
    'config.ts',
    "  if (c.groups.length !== d.groups.length) p.set('groups', c.groups.join(','))",
    "  if (c.groups.length < d.groups.length - 1) p.set('groups', c.groups.join(','))",
    'round-trips every group selection this build can produce',
  ],
  [
    'default-cap-written-into-the-link',
    'config.ts',
    "  if (c.cap !== d.cap) p.set('cap', c.cap)",
    "  p.set('cap', c.cap)",
    'is empty for the default view',
  ],
  [
    'leading-hash-not-stripped',
    'config.ts',
    "  const body = hash.replace(/^#/, '')",
    '  const body = hash',
    'round-trips every field',
  ],
  /* ------------------------------------------------------------ config.ts: coerce */
  [
    'rf-not-clamped-to-the-slider-range',
    'config.ts',
    '      out.rf = clamp(rf, RF_MIN, RF_MAX)',
    '      out.rf = rf',
    'keeps the risk-free rate inside the range the slider can express',
  ],
  [
    'rf-step-off-the-printed-grid',
    'config.ts',
    'export const RF_STEP = 0.0001',
    'export const RF_STEP = 0.0005',
    'opens on a risk-free rate the slider can actually return to',
  ],
  [
    'unknown-cap-accepted-silently',
    'config.ts',
    '    if (manifest.caps.some((c) => c.slug === r.cap)) out.cap = r.cap\n' +
      '    else complaints.push(`unknown weight cap "${r.cap}"`)',
    '    out.cap = r.cap',
    'says out loud every field it could not honour',
  ],
  [
    'position-not-clamped',
    'config.ts',
    '        out.pos = clamp(p, 0, 1)',
    '        out.pos = p',
    'says out loud every field it could not honour',
  ],
  [
    'empty-group-selection-allowed',
    'config.ts',
    '    if (picked.length) out.groups = picked\n' +
      "    else complaints.push('no known groups selected; showing all')",
    '    out.groups = picked',
    'never leaves the page with nothing to draw',
  ],
  [
    'groups-taken-from-the-file-unfiltered',
    'config.ts',
    '      ? manifest.groups.filter((g) => (r.groups as unknown[]).includes(g))',
    '      ? (r.groups as Group[])',
    'says out loud every field it could not honour',
  ],
  [
    'theme-not-validated',
    'config.ts',
    "  if (r.theme === 'light' || r.theme === 'dark') out.theme = r.theme",
    "  if (r.theme !== undefined) out.theme = r.theme as 'light' | 'dark'",
    'says out loud every field it could not honour',
  ],
  [
    'non-numeric-rf-propagates',
    'config.ts',
    '    if (Number.isFinite(rf)) {',
    '    if (true) {',
    'rejects a risk-free rate that is not a number',
  ],
  /* ------------------------------------------------------------ config.ts: the settings file */
  [
    'kind-not-checked',
    'config.ts',
    "  if (d.kind !== 'markowitz-config') {",
    '  if (false) {',
    'names the mistake when the portfolio export is dropped on it',
  ],
  [
    'version-not-checked',
    'config.ts',
    '  if (d.version !== CONFIG_VERSION) {',
    '  if (false) {',
    'refuses a file that is not one of ours',
  ],
  [
    'build-stamp-dropped-from-the-saved-file',
    'config.ts',
    '      data_generated_at: manifest.generated_at,',
    '',
    'records what build it was made against',
  ],
  [
    'a-foreign-config-refused-outright',
    'config.ts',
    '  const { config, complaints } = coerce(d.config, manifest)\n  return { config, complaints }',
    '  const { config, complaints } = coerce(d.config, manifest)\n' +
      "  if (complaints.length) return { config: null, error: 'unusable', complaints }\n" +
      '  return { config, complaints }',
    'loads a config from a build with different caps',
  ],
  /* ------------------------------------------------------------ config.ts: localStorage */
  [
    'storage-key-loses-its-version',
    'config.ts',
    "export const STORAGE_KEY = 'markowitz:config:v1'",
    "export const STORAGE_KEY = 'markowitz:config'",
    'does not survive a version bump silently',
  ],
  [
    'storage-read-not-guarded',
    'config.ts',
    '    return coerce(JSON.parse(text), manifest).config\n  } catch {\n    return null\n  }',
    '    return coerce(JSON.parse(text), manifest).config\n  } catch (e) {\n    throw e\n  }',
    'survives a browser with no usable storage at all',
  ],
  [
    'storage-write-not-guarded',
    'config.ts',
    '    else localStorage.setItem(STORAGE_KEY, JSON.stringify(c))\n  } catch {',
    '    else localStorage.setItem(STORAGE_KEY, JSON.stringify(c))\n  } catch (e) {\n' +
      '    throw e\n  } finally {',
    'survives a browser with no usable storage at all',
  ],
  [
    'stored-value-trusted-as-read',
    'config.ts',
    '    return coerce(JSON.parse(text), manifest).config',
    '    return JSON.parse(text) as Config',
    'coerces what it read',
  ],
  [
    'storage-beats-the-url',
    'config.ts',
    '  return fromHash(hash, manifest) ?? readStored(manifest) ?? defaults(manifest)',
    '  return readStored(manifest) ?? fromHash(hash, manifest) ?? defaults(manifest)',
    'prefers the URL, then storage, then the defaults',
  ],
  /* ------------------------------------------------------------ export.ts */
  [
    'csv-quoting-removed',
    'export.ts',
    '  return /[",\\r\\n]/.test(v) ? `"${v.replace(/"/g, \'""\')}"` : v',
    '  return v',
    'quotes a name containing a comma',
  ],
  [
    'weights-written-at-full-precision',
    'export.ts',
    '        h.weight.toFixed(W_DP),',
    '        String(h.weight),',
    'at the precision it claims',
  ],
  [
    'total-asserted-as-one',
    'export.ts',
    "  lines.push(['TOTAL', '', '', total.toFixed(W_DP)].join(','))",
    "  lines.push(['TOTAL', '', '', (1).toFixed(W_DP)].join(','))",
    'states the total it actually wrote',
  ],
  [
    'rf-dropped-from-provenance',
    'export.ts',
    '    `risk_free_rate=${pctStr(s.rf)} (${m.rf_source})`,',
    '',
    'puts the cap, the rf and the window',
  ],
  [
    'tangency-flag-always-set',
    'export.ts',
    '    `position=${s.atTangency ?',
    '    `position=${true ?',
    'says whether the handle is at the tangency',
  ],
  [
    'window-replaced-by-a-literal',
    'export.ts',
    '    window: m.window,',
    "    window: { ...m.window, start: '2000-01-01' },",
    'carries the window and estimation metadata verbatim',
  ],
  [
    'sharpe-computed-at-rf-zero',
    'export.ts',
    '      sharpe: round(s.position.sharpe, S_DP),',
    '      sharpe: round(s.position.ret / s.position.vol, S_DP),',
    'agrees with a fresh recompute',
  ],
  [
    'cap-omitted-from-the-filename',
    'export.ts',
    '  return `portfolio-${s.capSlug}-',
    '  return `portfolio-',
    'differs between two caps',
  ],
  /* ------------------------------------------------------------ data.ts */
  [
    'run-stamps-never-compared',
    'data.ts',
    '  assertOneRun(manifest, stats, history, frontiers)',
    '',
    'refuses a bundle whose files come from different runs',
  ],
  [
    'missing-stamp-falls-through-to-the-wrong-error',
    'data.ts',
    '  const missing = Object.entries(stamps).filter(([, s]) => !s).map(([n]) => n)',
    '  const missing = Object.entries(stamps).filter(() => false).map(([n]) => n)',
    'carries no stamp at all, naming the file',
  ],
  [
    'symbol-count-no-longer-checked',
    'data.ts',
    '  if (stats.symbols.length !== manifest.assets.length) {',
    '  if (false) {',
    'still catches a symbol-count mismatch',
  ],
  [
    'http-failure-does-not-name-the-file',
    'data.ts',
    '  if (!res.ok) throw new Error(`${name}: HTTP ${res.status} ${res.statusText}`)',
    '  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`)',
    'reports which file the HTTP error was for',
  ],
  /* ------------------------------------------------------------ viz.ts */
  [
    'group-labels-hardcoded-again',
    'viz.ts',
    'export const groupLabel = (labels: Record<string, string>, g: string) => labels[g] || g',
    `const LABELS: Record<string, string> = { ${hardcodedLabels} }\n` +
      'export const groupLabel = (labels: Record<string, string>, g: string) =>\n' +
      '  LABELS[g] || labels[g] || g',
    'names no group of this universe in the code',
  ],
  [
    'empty-group-label-passed-through',
    'viz.ts',
    '=> labels[g] || g',
    '=> labels[g] ?? g',
    'falls back to the raw key rather than rendering an empty label',
  ],
  [
    'paragraph-breaks-collapsed',
    'viz.ts',
    '  return text\n    .trim()\n    .split(/\\n\\s*\\n/)',
    '  return text\n    .trim()\n    .split(/\\n\\n\\n\\n/)',
    'keeps the paragraph breaks and drops the file wrapping',
  ],
  [
    'file-wrapping-left-in-the-paragraph',
    'viz.ts',
    "    .map((p) => p.replace(/\\s*\\n\\s*/g, ' ').trim())",
    '    .map((p) => p.trim())',
    'keeps the paragraph breaks and drops the file wrapping',
  ],
  [
    'window-length-rounded-two-ways-again',
    'viz.ts',
    'export const yearsText = (years: number) => years.toFixed(1)',
    'export const yearsText = (years: number) => years.toFixed(0)',
    'is stated with one rounding',
  ],
]

/* ------------------------------------------------------------------------------- the harness */

const args = process.argv.slice(2)
const only = args.includes('--only') ? args[args.indexOf('--only') + 1] : null
const keep = args.includes('--keep')
const selected = only ? MUTATIONS.filter(([name]) => name.includes(only)) : MUTATIONS

if (args.includes('--list')) {
  for (const [name, file, , , kills] of selected) console.log(`${name}\t${file}\t<- ${kills}`)
  process.exit(0)
}
if (selected.length === 0) {
  console.error(`no mutation matches --only ${only}`)
  process.exit(2)
}

function stage() {
  rmSync(STAGE, { recursive: true, force: true })
  mkdirSync(STAGE, { recursive: true })
  for (const entry of STAGED) {
    cpSync(join(WEB, entry), join(STAGE, entry), { recursive: true })
  }
}

/** The staged suite. `--root` rather than a copied node_modules: module resolution walks up from
 *  `web/.mutate/src` and finds `web/node_modules`, so the stage is 650 KB rather than 75 MB. */
function run() {
  const p = spawnSync('npx', ['vitest', 'run', '--root', '.mutate', '--reporter=basic'], {
    cwd: WEB,
    encoding: 'utf8',
  })
  return { ok: p.status === 0, out: `${p.stdout ?? ''}${p.stderr ?? ''}` }
}

stage()
const pristine = new Map()
for (const [, file] of selected) {
  if (!pristine.has(file)) pristine.set(file, readFileSync(join(STAGE, 'src', file), 'utf8'))
}

const baseline = run()
if (!baseline.ok) {
  console.error('BASELINE IS RED -- nothing to mutate')
  console.error(baseline.out.slice(-3000))
  process.exit(1)
}
const n = (k) => `${k} mutant${k === 1 ? '' : 's'}`
console.log(`baseline green, ${n(selected.length)}\n`)

const survivors = []
for (const [name, file, needle, replacement, kills] of selected) {
  const path = join(STAGE, 'src', file)
  const source = pristine.get(file)
  const hits = source.split(needle).length - 1
  if (hits !== 1) {
    // Part 1 of the contract. Exit rather than skip: the whole run is now untrustworthy, because
    // whatever this needle used to break is no longer being broken by anything.
    console.error(`\nSTALE MUTATION '${name}': its needle appears ${hits} times in ${file}`)
    console.error('Fix the needle or delete the mutation -- do not leave it unmatched.')
    if (!keep) rmSync(STAGE, { recursive: true, force: true })
    process.exit(2)
  }
  writeFileSync(path, source.replace(needle, replacement))
  const { ok, out } = run()
  writeFileSync(path, source)

  if (/Transform failed|Failed Suites|Error: Failed to load/.test(out)) {
    console.error(`\nINVALID MUTATION '${name}': the mutant does not parse, so it fails every`)
    console.error('test in its file and a kill would mean nothing. Rewrite it to compile.')
    if (!keep) rmSync(STAGE, { recursive: true, force: true })
    process.exit(3)
  }
  if (ok) {
    survivors.push([name, 'SURVIVED -- the whole suite stayed green'])
    console.log(`  SURVIVED  ${name}`)
  } else if (out.toLowerCase().includes(kills.toLowerCase())) {
    console.log(`  killed    ${name}  <- '${kills}'`)
  } else {
    survivors.push([name, `killed, but NOT by a test named '${kills}'`])
    console.log(`  MISKILLED ${name}: expected '${kills}' among the failures`)
  }
}

if (keep) console.log(`\nstage left at ${STAGE}`)
else rmSync(STAGE, { recursive: true, force: true })

console.log()
if (survivors.length) {
  for (const [name, why] of survivors) console.log(`FAIL ${name}: ${why}`)
  console.log(
    '\nA survivor is one of two things, and they need opposite fixes: a test that guards less\n' +
      'than its name claims, or a property the SHIPPED DATA cannot exercise. Decide which before\n' +
      'touching a test -- the second kind needs a synthetic fixture, not a weakened assertion.',
  )
  process.exit(1)
}
console.log(`every one of ${n(selected.length)} was killed by the test named for it`)
