import { useMemo, useState } from 'react'
import { Button } from 'antd'
import { PaperGuideReaderDrawer } from '../components/chat/PaperGuideReaderDrawer'
import type { ReaderSessionHighlight } from '../components/chat/reader/readerTypes'
import {
  buildReaderRegressionPayload,
  type ReaderRegressionScenario,
} from '../testing/readerRegressionFixtures'

function parseScenario(input: string | null): ReaderRegressionScenario {
  const raw = String(input || '').trim().toLowerCase()
  if (raw === 'evidence-nav') return 'evidence-nav'
  if (raw === 'candidate-fallback') return 'candidate-fallback'
  if (raw === 'strict-missing-exact') return 'strict-missing-exact'
  if (raw === 'discussion-only') return 'discussion-only'
  if (raw === 'limitations-only') return 'limitations-only'
  if (raw === 'future-work-only') return 'future-work-only'
  if (raw === 'equation') return 'equation'
  if (raw === 'figure') return 'figure'
  if (raw === 'multi-panel') return 'multi-panel'
  return 'strict-quote'
}

export default function ReaderRegressionPage() {
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const scenario = parseScenario(params.get('scenario'))
  const payload = useMemo(() => buildReaderRegressionPayload(scenario), [scenario])
  const [sessionHighlights, setSessionHighlights] = useState<ReaderSessionHighlight[]>([])
  const [appendLog, setAppendLog] = useState('')

  return (
    <div className="flex h-screen min-h-0 flex-col bg-[var(--bg)]">
      <div className="border-b border-[var(--border)] bg-[var(--panel)]/75 px-5 py-3">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="text-sm font-medium text-black/80 dark:text-white/80">
              Reader regression harness
            </div>
            <div className="text-xs text-black/45 dark:text-white/45" data-testid="reader-test-scenario">
              {scenario}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-full border border-[var(--border)] px-2 py-1 text-xs text-black/55 dark:text-white/55" data-testid="highlight-count">
              {sessionHighlights.length} highlights
            </span>
            <Button size="small" onClick={() => setSessionHighlights([])}>
              Clear highlights
            </Button>
            <Button size="small" onClick={() => setAppendLog('')}>
              Clear ask log
            </Button>
          </div>
        </div>
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_340px]">
        <div className="min-h-0 border-r border-[var(--border)]">
          <PaperGuideReaderDrawer
            open
            payload={payload}
            onClose={() => {}}
            onAppendSelection={(text) => {
              setAppendLog((current) => (current ? `${current}\n---\n${text}` : text))
            }}
            presentation="inline"
            sessionHighlights={sessionHighlights}
            onAddSessionHighlight={(highlight) => {
              setSessionHighlights((current) => {
                if (current.some((item) => item.id === highlight.id)) return current
                return [...current, highlight]
              })
            }}
            onRemoveSessionHighlight={(highlightId) => {
              setSessionHighlights((current) => current.filter((item) => item.id !== highlightId))
            }}
          />
        </div>
        <aside className="min-h-0 overflow-y-auto bg-[var(--panel)]/35 px-4 py-4">
          <div className="space-y-4">
            <section className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-black/40 dark:text-white/40">
                Ask output
              </div>
              <pre
                className="min-h-24 whitespace-pre-wrap rounded-2xl border border-[var(--border)] bg-[var(--panel)] px-3 py-3 text-xs text-black/70 dark:text-white/70"
                data-testid="append-output"
              >
                {appendLog || '(empty)'}
              </pre>
            </section>
            <section className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-black/40 dark:text-white/40">
                Highlight texts
              </div>
              <div className="space-y-2" data-testid="highlight-list">
                {sessionHighlights.length > 0 ? sessionHighlights.map((item) => (
                  <div
                    key={item.id}
                    className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] px-3 py-2 text-xs text-black/70 dark:text-white/70"
                  >
                    {item.text}
                  </div>
                )) : (
                  <div className="rounded-2xl border border-dashed border-[var(--border)] px-3 py-4 text-xs text-black/45 dark:text-white/45">
                    No highlights yet.
                  </div>
                )}
              </div>
            </section>
          </div>
        </aside>
      </div>
    </div>
  )
}
