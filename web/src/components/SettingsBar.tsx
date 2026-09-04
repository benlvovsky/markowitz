import { useRef, useState } from 'react'
import { configFilename, parseFile, serialise, type Config } from '../config'
import { save } from '../export'
import type { Manifest } from '../types'

interface Props {
  config: Config
  manifest: Manifest
  onLoad: (c: Config) => void
  onReset: () => void
}

/** Save, load, share and reset the view state.
 *
 * FOUR BUTTONS, THREE DIFFERENT DURABILITIES, and the labels have to make that visible: "Copy
 * link" shares a state that lives in the URL, "Save" writes a file the reader owns, and the
 * page is already persisting to `localStorage` continuously without being asked. Reset is here
 * because a shared link can put the page into a state the recipient did not choose, and
 * without it the only way out is editing the address bar.
 *
 * A LOAD THAT SILENTLY ADJUSTS SOMETHING IS A LOAD THAT LIED. `parseFile` returns complaints
 * separately from a fatal error, and both are shown: a settings file from a run with different
 * weight caps still loads, but the reader is told which fields could not be honoured rather
 * than left to wonder why the cap looks wrong. The most likely mistake -- dropping the
 * portfolio export here, since it is the other JSON this page hands out -- is named explicitly.
 */
export function SettingsBar({ config, manifest, onLoad, onReset }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [msg, setMsg] = useState<{ kind: 'ok' | 'bad'; text: string } | null>(null)
  const [over, setOver] = useState(false)

  const ingest = async (file: File) => {
    const result = parseFile(await file.text(), manifest)
    if (!result.config) {
      setMsg({ kind: 'bad', text: result.error ?? 'that file could not be read' })
      return
    }
    onLoad(result.config)
    setMsg(
      result.complaints.length
        ? { kind: 'bad', text: `loaded, with changes: ${result.complaints.join('; ')}` }
        : { kind: 'ok', text: `loaded ${file.name}` },
    )
  }

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      setMsg({ kind: 'ok', text: 'link copied' })
    } catch {
      // Clipboard access is refused without a secure context or a user gesture the browser
      // recognises. The URL is in the address bar either way, so say so instead of failing.
      setMsg({ kind: 'bad', text: 'could not copy; the link is in the address bar' })
    }
  }

  return (
    <div
      className="control settings-cell"
      style={{
        // Dashed and on `--border-strong`, not on a categorical hue: both of those are spoken
        // for by the chart, and the DASH is what says "drop target" anyway.
        outline: over ? '2px dashed var(--border-strong)' : 'none',
        outlineOffset: 6,
      }}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const f = e.dataTransfer.files[0]
        if (f) void ingest(f)
      }}
    >
      <span className="control-label">Settings</span>
      <div className="segmented" role="group" aria-label="Settings">
        <button onClick={() => save(configFilename(), 'application/json', serialise(config, manifest))}>
          Save
        </button>
        <button onClick={() => input.current?.click()}>Load</button>
        <button onClick={() => void copyLink()}>Copy link</button>
        <button
          onClick={() => {
            onReset()
            setMsg({ kind: 'ok', text: 'reset to defaults' })
          }}
        >
          Reset
        </button>
      </div>
      <input
        ref={input}
        type="file"
        accept="application/json,.json"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void ingest(f)
          // Cleared so re-picking the same file fires `change` again.
          e.target.value = ''
        }}
      />
      <span className="settings-status" data-bad={msg?.kind === 'bad'} role="status">
        {msg ? msg.text : 'or drop a settings file here'}
      </span>
    </div>
  )
}
