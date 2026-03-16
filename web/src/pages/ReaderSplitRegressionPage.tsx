import { useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { PaperGuideReaderDrawer } from '../components/chat/PaperGuideReaderDrawer'
import type { ReaderSessionHighlight } from '../components/chat/reader/readerTypes'
import { buildReaderRegressionPayload } from '../testing/readerRegressionFixtures'

const DEFAULT_WIDTH = 560
const MIN_WIDTH = 420
const MAX_WIDTH = 760

function clampWidth(value: number) {
  if (!Number.isFinite(value)) return DEFAULT_WIDTH
  return Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, Math.round(value)))
}

export default function ReaderSplitRegressionPage() {
  const payload = buildReaderRegressionPayload('strict-quote')
  const [readerWidth, setReaderWidth] = useState(DEFAULT_WIDTH)
  const [previewWidth, setPreviewWidth] = useState(DEFAULT_WIDTH)
  const [readerResizing, setReaderResizing] = useState(false)
  const [sessionHighlights, setSessionHighlights] = useState<ReaderSessionHighlight[]>([])
  const splitLayoutRef = useRef<HTMLDivElement | null>(null)
  const readerResizeGuideRef = useRef<HTMLDivElement | null>(null)
  const readerResizeRef = useRef<{ startX: number; startWidth: number } | null>(null)
  const activePointerIdRef = useRef<number | null>(null)

  const updateGuide = (nextWidth: number) => {
    const layout = splitLayoutRef.current
    const guide = readerResizeGuideRef.current
    const clamped = clampWidth(nextWidth)
    setPreviewWidth(clamped)
    if (!layout || !guide) return
    guide.style.left = `${Math.max(0, layout.clientWidth - clamped)}px`
  }

  const finishResize = (commit: boolean) => {
    const finalWidth = clampWidth(commit ? previewWidth : readerWidth)
    readerResizeRef.current = null
    activePointerIdRef.current = null
    setReaderResizing(false)
    if (commit) {
      setReaderWidth(finalWidth)
      setPreviewWidth(finalWidth)
    } else {
      setPreviewWidth(readerWidth)
    }
  }

  const beginResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!event.isPrimary) return
    const currentWidth = clampWidth(readerWidth)
    readerResizeRef.current = { startX: event.clientX, startWidth: currentWidth }
    activePointerIdRef.current = event.pointerId
    setReaderResizing(true)
    updateGuide(currentWidth)
    event.currentTarget.setPointerCapture(event.pointerId)
    event.preventDefault()
  }

  const moveResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (activePointerIdRef.current !== event.pointerId) return
    const state = readerResizeRef.current
    if (!state) return
    updateGuide(state.startWidth + (state.startX - event.clientX))
    event.preventDefault()
  }

  const commitResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (activePointerIdRef.current !== event.pointerId) return
    finishResize(true)
    event.preventDefault()
  }

  const cancelResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (activePointerIdRef.current !== event.pointerId) return
    finishResize(false)
    event.preventDefault()
  }

  return (
    <div className="flex h-screen min-h-0 flex-col bg-[var(--bg)]">
      <div className="border-b border-[var(--border)] bg-[var(--panel)]/75 px-5 py-3">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="text-sm font-medium text-black/80 dark:text-white/80">
              Reader split-pane regression harness
            </div>
            <div className="text-xs text-black/45 dark:text-white/45">
              Preview follows drag, width commits on release.
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-black/55 dark:text-white/55">
            <span data-testid="split-preview-width">{previewWidth}</span>
            <span>/</span>
            <span data-testid="split-committed-width">{readerWidth}</span>
          </div>
        </div>
      </div>
      <div ref={splitLayoutRef} className="relative flex min-h-0 flex-1">
        <main className="flex min-h-0 min-w-0 flex-1 flex-col" data-testid="split-main-panel">
          <div className="flex-1 overflow-y-auto px-6 py-6">
            <div className="mx-auto max-w-3xl space-y-4">
              {Array.from({ length: 24 }).map((_, idx) => (
                <section
                  key={idx}
                  className="rounded-3xl border border-[var(--border)] bg-[var(--panel)]/65 px-5 py-4"
                >
                  <div className="text-xs uppercase tracking-[0.18em] text-black/35 dark:text-white/35">
                    Conversation
                  </div>
                  <div className="mt-2 text-sm text-black/75 dark:text-white/75">
                    Resize regression content block {idx + 1}. The left panel should stay stable while the preview divider moves.
                  </div>
                </section>
              ))}
            </div>
          </div>
        </main>
        <div
          ref={readerResizeGuideRef}
          className={`pointer-events-none absolute inset-y-0 z-20 hidden w-0 xl:block ${
            readerResizing ? 'opacity-100' : 'opacity-0'
          }`}
          aria-hidden="true"
          data-testid="split-resize-guide"
        >
          <div className="absolute inset-y-0 -translate-x-1/2 border-l-2 border-[var(--accent)]/75 shadow-[0_0_0_1px_rgba(22,119,255,0.15)]" />
        </div>
        <aside className="hidden h-full shrink-0 xl:flex">
          <div
            className={`w-2 shrink-0 cursor-col-resize transition ${
              readerResizing
                ? 'bg-[var(--accent)]/30'
                : 'bg-transparent hover:bg-black/[0.05] dark:hover:bg-white/[0.05]'
            }`}
            onPointerDown={beginResize}
            onPointerMove={moveResize}
            onPointerUp={commitResize}
            onPointerCancel={cancelResize}
            data-testid="split-resize-handle"
          />
          <div
            className={`flex h-full shrink-0 border-l border-[var(--border)] ${
              readerResizing ? 'bg-[var(--panel)]' : 'bg-[var(--panel)]/70 backdrop-blur-sm'
            }`}
            style={{ width: `${readerWidth}px` }}
            data-testid="split-reader-pane"
          >
            <PaperGuideReaderDrawer
              open
              payload={payload}
              onClose={() => {}}
              onAppendSelection={() => {}}
              presentation="inline"
              sessionHighlights={sessionHighlights}
              onAddSessionHighlight={(highlight) => {
                setSessionHighlights((current) => [...current, highlight])
              }}
              onRemoveSessionHighlight={(highlightId) => {
                setSessionHighlights((current) => current.filter((item) => item.id !== highlightId))
              }}
            />
          </div>
        </aside>
      </div>
    </div>
  )
}
