import { useCallback, useMemo, useState } from 'react'
import { Button } from 'antd'
import { PaperGuideReaderDrawer } from '../components/chat/PaperGuideReaderDrawer'
import type {
  ReaderLocateResult,
  ReaderSelectionShelfPayload,
  ReaderSessionHighlight,
} from '../components/chat/reader/readerTypes'
import type { CiteDetail } from '../components/chat/citationState'
import {
  buildReaderRegressionDocResponse,
  buildReaderRegressionPayload,
  type ReaderRegressionScenario,
} from '../testing/readerRegressionFixtures'

function parseScenario(input: string | null): ReaderRegressionScenario {
  const raw = String(input || '').trim().toLowerCase()
  if (raw === 'evidence-nav') return 'evidence-nav'
  if (raw === 'duplicate-sections') return 'duplicate-sections'
  if (raw === 'duplicate-images') return 'duplicate-images'
  if (raw === 'candidate-fallback') return 'candidate-fallback'
  if (raw === 'strict-missing-exact') return 'strict-missing-exact'
  if (raw === 'discussion-only') return 'discussion-only'
  if (raw === 'limitations-only') return 'limitations-only'
  if (raw === 'future-work-only') return 'future-work-only'
  if (raw === 'equation') return 'equation'
  if (raw === 'figure') return 'figure'
  if (raw === 'multi-panel') return 'multi-panel'
  if (raw === 'render-polish') return 'render-polish'
  if (raw === 'citation-links') return 'citation-links'
  return 'strict-quote'
}

export default function ReaderRegressionPage() {
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const scenario = parseScenario(params.get('scenario'))
  const payload = useMemo(() => buildReaderRegressionPayload(scenario), [scenario])
  const documentOverride = useMemo(() => buildReaderRegressionDocResponse(scenario), [scenario])
  const [sessionHighlights, setSessionHighlights] = useState<ReaderSessionHighlight[]>(() => (
    params.get('seedHighlight') === '1'
      ? [{
        id: 'seed-highlight-quote-1',
        text: 'SCI compresses a short video into one coded measurement.',
        noteKind: 'highlight',
        sourcePath: '__reader_regression__/fixture.md',
        sourceName: 'Fixture Paper',
        headingPath: 'Fixture Paper / 1. Introduction',
        blockId: 'quote-1',
        anchorId: 'a-quote-1',
        createdAt: Date.now(),
      }]
      : []
  ))
  const [appendLog, setAppendLog] = useState('')
  const [locateResult, setLocateResult] = useState<ReaderLocateResult | null>(null)
  const [readerCitationShelf, setReaderCitationShelf] = useState<CiteDetail[]>([])
  const [readerSelectionShelf, setReaderSelectionShelf] = useState<ReaderSelectionShelfPayload[]>([])
  const appendSelectionLog = useCallback((text: string) => {
    setAppendLog((current) => (current ? `${current}\n---\n${text}` : text))
  }, [])
  const addSessionHighlight = useCallback((highlight: ReaderSessionHighlight) => {
    setSessionHighlights((current) => {
      if (current.some((item) => item.id === highlight.id)) return current
      return [...current, highlight]
    })
  }, [])
  const removeSessionHighlight = useCallback((highlightId: string) => {
    setSessionHighlights((current) => current.filter((item) => item.id !== highlightId))
  }, [])
  const updateSessionHighlight = useCallback((highlight: ReaderSessionHighlight) => {
    const targetId = String(highlight?.id || '').trim()
    if (!targetId) return
    setSessionHighlights((current) => current.map((item) => (
      item.id === targetId ? { ...item, ...highlight } : item
    )))
  }, [])
  const recordLocateResult = useCallback((result: ReaderLocateResult) => {
    setLocateResult((current) => {
      if (
        current
        && current.locateRequestId === result.locateRequestId
        && current.status === result.status
        && current.precision === result.precision
        && current.hint === result.hint
      ) {
        return current
      }
      return result
    })
  }, [])
  const addReaderCitationToShelf = useCallback((detail: CiteDetail) => {
    setReaderCitationShelf((current) => {
      const key = String(detail.anchor || '').trim()
      if (key && current.some((item) => String(item.anchor || '').trim() === key)) return current
      return [detail, ...current]
    })
  }, [])
  const addReaderSelectionToShelf = useCallback((payload: ReaderSelectionShelfPayload) => {
    setReaderSelectionShelf((current) => {
      const key = [
        payload.sourcePath,
        payload.blockId || payload.anchorId || '',
        payload.anchorKind || '',
        payload.text,
      ].join('|')
      if (current.some((item) => [
        item.sourcePath,
        item.blockId || item.anchorId || '',
        item.anchorKind || '',
        item.text,
      ].join('|') === key)) return current
      return [payload, ...current]
    })
  }, [])

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
            <span className="rounded-full border border-[var(--border)] px-2 py-1 text-xs text-black/55 dark:text-white/55" data-testid="reader-citation-shelf-count">
              {readerCitationShelf.length} citation refs
            </span>
            <span className="rounded-full border border-[var(--border)] px-2 py-1 text-xs text-black/55 dark:text-white/55" data-testid="reader-selection-shelf-count">
              {readerSelectionShelf.length} selections
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
      <div className="flex min-h-0 flex-1 flex-col xl:grid xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="min-h-0 flex-1 border-r border-[var(--border)]">
          <PaperGuideReaderDrawer
            open
            payload={payload}
            onClose={() => {}}
            onAppendSelection={appendSelectionLog}
            presentation="inline"
            sessionHighlights={sessionHighlights}
            onAddSessionHighlight={addSessionHighlight}
            onUpdateSessionHighlight={updateSessionHighlight}
            onRemoveSessionHighlight={removeSessionHighlight}
            onLocateResult={recordLocateResult}
            documentOverride={documentOverride}
            onAddCitationToShelf={addReaderCitationToShelf}
            onAddSelectionToShelf={addReaderSelectionToShelf}
          />
        </div>
        <aside className="max-h-72 min-h-0 overflow-y-auto bg-[var(--panel)]/35 px-4 py-4 xl:max-h-none">
          <div className="space-y-4">
            <section className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-black/40 dark:text-white/40">
                Locate result
              </div>
              <pre
                className="min-h-20 whitespace-pre-wrap rounded-2xl border border-[var(--border)] bg-[var(--panel)] px-3 py-3 text-xs text-black/70 dark:bg-black/20 dark:text-white/70"
                data-testid="reader-locate-result-json"
              >
                {locateResult ? JSON.stringify(locateResult, null, 2) : '(empty)'}
              </pre>
            </section>
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
                Reader shelf selections
              </div>
              <div className="space-y-2" data-testid="reader-selection-shelf-list">
                {readerSelectionShelf.length > 0 ? readerSelectionShelf.map((item, idx) => (
                  <div
                    key={`${item.blockId || item.anchorId || idx}-${idx}`}
                    className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] px-3 py-2 text-xs text-black/70 dark:text-white/70"
                  >
                    <div data-testid={`reader-selection-shelf-kind-${idx}`}>{item.anchorKind || 'selection'}</div>
                    <div>{item.text}</div>
                  </div>
                )) : (
                  <div className="rounded-2xl border border-dashed border-[var(--border)] px-3 py-4 text-xs text-black/45 dark:text-white/45">
                    No reader selections yet.
                  </div>
                )}
              </div>
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
