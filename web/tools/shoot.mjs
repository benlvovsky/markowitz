/** Screenshot the built page in a headless browser, so "render it and look at it" is one command.
 *
 *     node tools/shoot.mjs <url> <out.png> [selector] [theme]
 *
 *     node tools/shoot.mjs http://localhost:4455/markowitz/ /tmp/light.png '' light
 *     node tools/shoot.mjs http://localhost:4455/markowitz/ /tmp/chart.png '.chart' dark
 *
 * WHY THIS EXISTS AT ALL. `CLAUDE.md` requires looking at the chart after any change to it,
 * because the palette validator checks colour and cannot see a label collision, a clipped mark or
 * an overflowing row. That step only ever actually happens if it is one command, and screenshotting
 * a React SPA is not one command by default -- the page has to have mounted, fetched six JSON
 * files and laid out ~3,500px of content before the pixels mean anything.
 *
 * Serve `dist` under the `/markowitz/` prefix Pages uses, so a base-path mistake shows up here
 * rather than after a deploy:
 *
 *     npm run build && mkdir -p /tmp/serve && ln -sfn "$PWD/dist" /tmp/serve/markowitz
 *     (cd /tmp/serve && python3 -m http.server 4455)
 *
 * NO DEPENDENCIES, deliberately: the Chrome DevTools Protocol over Node's global `WebSocket` is
 * about forty lines, and a screenshot tool is not worth adding puppeteer and a second browser
 * download to a repo whose entire runtime dependency list is react and react-dom.
 *
 * IT PRINTS THE CONSOLE AND ANY EXCEPTION, which is the part that earns its keep. A chart that
 * fails to mount and a chart that mounts wrong produce the same blank PNG, and the difference is
 * always in the console -- so the tool that captures the picture is also the one that says why
 * there isn't one.
 *
 * `SELECTOR` clips to one element's box (`details` are opened first, or a clip of a collapsed
 * disclosure is a screenshot of its summary). Without it, the whole page at its full scroll
 * height, which is what `captureBeyondViewport` is for.
 */
import { spawn } from 'node:child_process'
import { writeFileSync } from 'node:fs'

const CHROME =
  process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const WIDTH = Number(process.env.SHOOT_WIDTH ?? 1360)
/** A REALISTIC viewport height, not a tall one. The obvious shortcut -- emulate a 4200px-high
 *  window so everything is on screen at once -- reports `scrollHeight` as 4200 whatever the
 *  content is, so the full-page capture below cannot then measure what it is capturing. It would
 *  also lay the page out at a viewport height no reader has. */
const HEIGHT = Number(process.env.SHOOT_HEIGHT ?? 1200)
const SCALE = Number(process.env.SHOOT_SCALE ?? 1)
/** The data is six fetches and the frontier is 3 x 62 quadratic programs' worth of JSON; 3.5 s is
 *  measured slack over a localhost load, not a guess at one. */
const SETTLE_MS = Number(process.env.SHOOT_SETTLE ?? 3500)

const [rawUrl, out, selector, theme] = process.argv.slice(2)
if (!rawUrl || !out) {
  console.error('usage: node tools/shoot.mjs <url> <out.png> [selector] [light|dark]')
  process.exit(64)
}
// The theme is a fragment parameter, so it is appended rather than set: `#theme=dark` on a URL
// that already carries a config would otherwise throw the reader's whole view state away.
const url = theme ? (rawUrl.includes('#') ? `${rawUrl}&theme=${theme}` : `${rawUrl}#theme=${theme}`) : rawUrl

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
// A fixed port would collide with a previous run that has not exited yet, and the failure looks
// like a hung browser rather than a busy socket.
const PORT = 9300 + Math.floor(Math.random() * 400)

const chrome = spawn(
  CHROME,
  [
    '--headless=new',
    '--disable-gpu',
    '--hide-scrollbars',
    `--remote-debugging-port=${PORT}`,
    // Its own profile: a shared one is locked by any real Chrome the user has open, and the job
    // then fails with nothing on stderr to say why.
    '--user-data-dir=/tmp/markowitz-shoot-profile',
    `--window-size=${WIDTH},1200`,
    'about:blank',
  ],
  { stdio: 'ignore' },
)

async function attach() {
  for (let i = 0; i < 60; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
      const page = list.find((t) => t.type === 'page')
      if (page) return page
    } catch {
      /* not listening yet */
    }
    await sleep(200)
  }
  throw new Error(`chrome never opened a debugging port on ${PORT}`)
}

const page = await attach()
const ws = new WebSocket(page.webSocketDebuggerUrl)
await new Promise((resolve, reject) => {
  ws.onopen = resolve
  ws.onerror = reject
})

let id = 0
const pending = new Map()
const urls = new Map()
let failed = false
/** `/favicon.ico` is requested by the BROWSER, not by the page, at the origin root rather than
 *  under the base -- so it 404s on every static serve of a subpath and says nothing about whether
 *  the chart rendered. Everything else that 404s is a missing asset or a missing data file, which
 *  is exactly what exiting nonzero is for. */
const chromesOwn = (url) => /\/favicon\.ico$/.test(url)
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data)
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m.result)
    pending.delete(m.id)
    return
  }
  // Everything below is why this is not just `puppeteer.screenshot`.
  if (m.method === 'Runtime.consoleAPICalled') {
    const args = (m.params.args ?? []).map((a) => a.value ?? a.description).join(' ')
    console.error(`console.${m.params.type}: ${args}`)
    if (m.params.type === 'error') failed = true
  }
  if (m.method === 'Runtime.exceptionThrown') {
    const d = m.params.exceptionDetails
    console.error(`EXCEPTION: ${d.exception?.description ?? d.text}`)
    failed = true
  }
  // `/favicon.ico` is requested by the BROWSER, not by the page, at the origin root rather than
  // under the base -- so it 404s on every static serve of a subpath and has nothing to do with
  // whether the chart rendered. Everything else that 404s is a missing asset or a missing data
  // file, which is exactly the failure worth exiting nonzero for.
  // `Network.loadingFailed` carries a requestId and no URL, so the URLs are kept as they are
  // requested -- without them a failure reads "REQUEST FAILED: net::ERR_ABORTED" with no way to
  // tell a missing data file from the favicon.
  if (m.method === 'Network.requestWillBeSent') urls.set(m.params.requestId, m.params.request.url)
  if (m.method === 'Network.responseReceived' && m.params.response.status >= 400) {
    if (!chromesOwn(m.params.response.url)) {
      console.error(`HTTP ${m.params.response.status} ${m.params.response.url}`)
      failed = true
    }
  }
  if (m.method === 'Network.loadingFailed') {
    const url = urls.get(m.params.requestId) ?? '(unknown url)'
    if (!chromesOwn(url)) {
      console.error(`REQUEST FAILED: ${m.params.errorText} ${url}`)
      failed = true
    }
  }
})
const send = (method, params = {}) =>
  new Promise((resolve) => {
    const n = ++id
    pending.set(n, resolve)
    ws.send(JSON.stringify({ id: n, method, params }))
  })

try {
  await send('Page.enable')
  await send('Runtime.enable')
  await send('Network.enable')
  await send('Emulation.setDeviceMetricsOverride', {
    width: WIDTH,
    height: HEIGHT,
    deviceScaleFactor: 1,
    mobile: false,
  })
  await send('Page.navigate', { url })
  await sleep(SETTLE_MS)

  // A blank page is the failure this catches, and it is invisible in the PNG: an empty `#root`
  // renders as a plain surface-coloured rectangle that looks like a deliberate layout.
  const body = await send('Runtime.evaluate', {
    expression: 'document.body.innerText.trim().length',
    returnByValue: true,
  })
  if (!body.result?.value) {
    console.error(`NOTHING RENDERED at ${url} -- the page has no text in it.`)
    console.error('Check the requests above: an empty `#root` with the CSS loaded is a JS that')
    console.error('never ran, which is a wrong base path far more often than it is a chart bug.')
    failed = true
  }

  let clip
  if (selector) {
    // Opened before measuring: a clip of a closed `<details>` is a screenshot of its summary,
    // which is a correct picture of the wrong thing.
    await send('Runtime.evaluate', {
      expression: `document.querySelectorAll('details').forEach((d) => (d.open = true))`,
    })
    await sleep(400)
    const box = await send('Runtime.evaluate', {
      expression: `(() => {
        const el = document.querySelector(${JSON.stringify(selector)})
        if (!el) return null
        const r = el.getBoundingClientRect()
        return { x: r.x + scrollX, y: r.y + scrollY, width: r.width, height: r.height }
      })()`,
      returnByValue: true,
    })
    const b = box.result?.value
    if (!b) throw new Error(`no element matches ${selector}`)
    const pad = 16
    clip = {
      x: Math.max(0, b.x - pad),
      y: Math.max(0, b.y - pad),
      width: b.width + pad * 2,
      height: b.height + pad * 2,
      scale: SCALE,
    }
  } else {
    const size = await send('Runtime.evaluate', {
      expression:
        '({ w: document.documentElement.clientWidth, h: document.documentElement.scrollHeight })',
      returnByValue: true,
    })
    const { w, h } = size.result.value
    clip = { x: 0, y: 0, width: w, height: h, scale: SCALE }
  }

  const shot = await send('Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: true,
    clip,
  })
  writeFileSync(out, Buffer.from(shot.data, 'base64'))
  console.log(`${out}  ${Math.round(clip.width)}x${Math.round(clip.height)} css @ ${SCALE}x  ${url}`)
} finally {
  ws.close()
  chrome.kill()
}

// Nonzero when the page complained, so this can sit in front of a manual look without turning a
// broken render into a green step. The PNG is still written -- it is evidence either way.
process.exit(failed ? 1 : 0)
