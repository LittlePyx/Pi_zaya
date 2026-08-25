import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react'
import type { ReaderOpenPayload } from './reader/readerTypes'

const DESKTOP_READER_BREAKPOINT = 1280
const DESKTOP_DOCK_DEFAULT_WIDTH = 392
const DESKTOP_DOCK_MIN_WIDTH = 320
const DESKTOP_DOCK_MAX_WIDTH = 760
const DESKTOP_DOCK_COLLAPSED_WIDTH = 48
const DESKTOP_DOCK_WIDTH_TRANSITION = 'width 160ms cubic-bezier(0.2, 0, 0, 1)'
const RIGHT_DOCK_WIDTH_STORAGE_KEY = 'kb:chat-side-dock-width'
const RIGHT_DOCK_COLLAPSED_STORAGE_KEY = 'kb:chat-side-dock-collapsed'

export type RightDockPanel = 'timeline' | 'shelf' | 'notes' | 'reader'

function clampRightDockWidth(value: number) {
  if (!Number.isFinite(value)) return DESKTOP_DOCK_DEFAULT_WIDTH
  return Math.max(DESKTOP_DOCK_MIN_WIDTH, Math.min(DESKTOP_DOCK_MAX_WIDTH, Math.round(value)))
}

function loadStoredRightDockWidth() {
  if (typeof window === 'undefined') return DESKTOP_DOCK_DEFAULT_WIDTH
  try {
    const raw = Number(window.localStorage.getItem(RIGHT_DOCK_WIDTH_STORAGE_KEY) || 0)
    return clampRightDockWidth(raw || DESKTOP_DOCK_DEFAULT_WIDTH)
  } catch {
    return DESKTOP_DOCK_DEFAULT_WIDTH
  }
}

function loadStoredRightDockCollapsed() {
  if (typeof window === 'undefined') return false
  try {
    return window.localStorage.getItem(RIGHT_DOCK_COLLAPSED_STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

export function useReaderDock() {
  const [readerOpen, setReaderOpen] = useState(false)
  const [readerPayload, setReaderPayload] = useState<ReaderOpenPayload | null>(null)
  const [rightDockCollapsed, setRightDockCollapsed] = useState(loadStoredRightDockCollapsed)
  const [rightDockWidth, setRightDockWidth] = useState(loadStoredRightDockWidth)
  const [rightDockPanel, setRightDockPanel] = useState<RightDockPanel>('timeline')
  const [desktopReaderEligible, setDesktopReaderEligible] = useState(
    () => (typeof window !== 'undefined' ? window.innerWidth >= DESKTOP_READER_BREAKPOINT : false),
  )
  const [rightDockResizing, setRightDockResizing] = useState(false)

  const readerPayloadRef = useRef<ReaderOpenPayload | null>(null)
  const readerOpenRef = useRef(false)
  const splitLayoutRef = useRef<HTMLDivElement | null>(null)
  const rightDockResizeGuideRef = useRef<HTMLDivElement | null>(null)
  const rightDockResizeRef = useRef<{ startX: number; startWidth: number } | null>(null)
  const rightDockActivePointerIdRef = useRef<number | null>(null)
  const rightDockResizeFocusRestoreRef = useRef<HTMLElement | null>(null)
  const rightDockWidthLiveRef = useRef(rightDockWidth)
  const rightDockResizePreviewWidthRef = useRef(rightDockWidth)
  const rightDockCollapsedRef = useRef(rightDockCollapsed)

  useEffect(() => {
    readerOpenRef.current = readerOpen
  }, [readerOpen])

  useEffect(() => {
    readerPayloadRef.current = readerPayload
  }, [readerPayload])

  useEffect(() => {
    rightDockCollapsedRef.current = rightDockCollapsed
  }, [rightDockCollapsed])

  useEffect(() => {
    if (typeof window === 'undefined') return undefined
    const syncLayout = () => {
      setDesktopReaderEligible(window.innerWidth >= DESKTOP_READER_BREAKPOINT)
    }
    window.addEventListener('resize', syncLayout)
    return () => {
      window.removeEventListener('resize', syncLayout)
    }
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      window.localStorage.setItem(RIGHT_DOCK_WIDTH_STORAGE_KEY, String(clampRightDockWidth(rightDockWidth)))
    } catch {
      // Storage can fail in private mode; the in-memory width still works.
    }
  }, [rightDockWidth])

  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      window.localStorage.setItem(RIGHT_DOCK_COLLAPSED_STORAGE_KEY, rightDockCollapsed ? '1' : '0')
    } catch {
      // Storage can fail in private mode; the in-memory collapsed state still works.
    }
  }, [rightDockCollapsed])

  useEffect(() => {
    rightDockWidthLiveRef.current = rightDockWidth
    if (!rightDockResizing) {
      rightDockResizePreviewWidthRef.current = rightDockWidth
    }
  }, [rightDockResizing, rightDockWidth])

  const restoreRightDockResizeFocus = useCallback(() => {
    const target = rightDockResizeFocusRestoreRef.current
    rightDockResizeFocusRestoreRef.current = null
    if (!target || !target.isConnected) return
    try {
      target.focus({ preventScroll: true })
    } catch {
      target.focus()
    }
  }, [])

  const clearRightDockResizeSession = useCallback(() => {
    rightDockResizeRef.current = null
    rightDockActivePointerIdRef.current = null
    rightDockResizePreviewWidthRef.current = rightDockWidthLiveRef.current
    if (typeof document !== 'undefined') {
      document.body.classList.remove('kb-right-dock-resizing')
      document.body.style.removeProperty('cursor')
      document.body.style.removeProperty('user-select')
    }
    const guide = rightDockResizeGuideRef.current
    if (guide) {
      guide.style.removeProperty('left')
    }
  }, [])

  useEffect(() => () => {
    clearRightDockResizeSession()
  }, [clearRightDockResizeSession])

  const updateRightDockResizeGuide = useCallback((nextWidth: number) => {
    const guide = rightDockResizeGuideRef.current
    const layout = splitLayoutRef.current
    const clampedWidth = clampRightDockWidth(nextWidth)
    rightDockResizePreviewWidthRef.current = clampedWidth
    if (!guide || !layout || rightDockCollapsedRef.current) return
    const nextLeft = Math.max(0, layout.clientWidth - clampedWidth)
    guide.style.left = `${Math.round(nextLeft)}px`
  }, [])

  const finishRightDockResize = useCallback((commit: boolean) => {
    const finalWidth = clampRightDockWidth(
      commit ? rightDockResizePreviewWidthRef.current || rightDockWidthLiveRef.current : rightDockWidthLiveRef.current,
    )
    clearRightDockResizeSession()
    setRightDockResizing(false)
    if (commit && !rightDockCollapsedRef.current) {
      setRightDockWidth(finalWidth)
    }
    window.requestAnimationFrame(restoreRightDockResizeFocus)
  }, [clearRightDockResizeSession, restoreRightDockResizeFocus])

  const beginRightDockResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (rightDockCollapsedRef.current || !event.isPrimary) return
    const currentWidth = clampRightDockWidth(rightDockWidthLiveRef.current)
    rightDockResizeRef.current = {
      startX: event.clientX,
      startWidth: currentWidth,
    }
    rightDockActivePointerIdRef.current = event.pointerId
    rightDockResizePreviewWidthRef.current = currentWidth
    updateRightDockResizeGuide(currentWidth)
    const activeElement = document.activeElement
    if (
      activeElement instanceof HTMLElement
      && (activeElement.tagName === 'TEXTAREA' || activeElement.tagName === 'INPUT')
    ) {
      rightDockResizeFocusRestoreRef.current = activeElement
      activeElement.blur()
    } else {
      rightDockResizeFocusRestoreRef.current = null
    }
    setRightDockResizing(true)
    document.body.classList.add('kb-right-dock-resizing')
    document.body.style.setProperty('cursor', 'col-resize')
    document.body.style.setProperty('user-select', 'none')
    event.currentTarget.setPointerCapture(event.pointerId)
    event.preventDefault()
  }, [updateRightDockResizeGuide])

  const handleRightDockResizeMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (rightDockActivePointerIdRef.current !== event.pointerId) return
    const state = rightDockResizeRef.current
    if (!state) return
    updateRightDockResizeGuide(state.startWidth + (state.startX - event.clientX))
    event.preventDefault()
  }, [updateRightDockResizeGuide])

  const commitRightDockResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (rightDockActivePointerIdRef.current !== event.pointerId) return
    finishRightDockResize(true)
    event.preventDefault()
  }, [finishRightDockResize])

  const cancelRightDockResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (rightDockActivePointerIdRef.current !== event.pointerId) return
    finishRightDockResize(false)
    event.preventDefault()
  }, [finishRightDockResize])

  const openReaderDock = useCallback((payload: ReaderOpenPayload) => {
    readerPayloadRef.current = payload
    readerOpenRef.current = true
    setReaderPayload(payload)
    setRightDockPanel('reader')
    setRightDockCollapsed(false)
    setReaderOpen(true)
  }, [])

  const closeReader = useCallback(() => {
    readerOpenRef.current = false
    setReaderOpen(false)
  }, [])

  const resetReaderDock = useCallback(() => {
    readerOpenRef.current = false
    readerPayloadRef.current = null
    setReaderOpen(false)
    setReaderPayload(null)
  }, [])

  const showDockPanel = useCallback((panel: RightDockPanel) => {
    setRightDockPanel(panel)
    setRightDockCollapsed(false)
  }, [])

  const collapseRightDock = useCallback(() => {
    setRightDockCollapsed(true)
  }, [])

  const toggleRightDockCollapsed = useCallback(() => {
    setRightDockCollapsed((value) => !value)
  }, [])

  const rightDockStyle = useMemo<CSSProperties>(() => ({
    width: rightDockCollapsed ? `${DESKTOP_DOCK_COLLAPSED_WIDTH}px` : `${rightDockWidth}px`,
    transition: rightDockResizing ? 'none' : DESKTOP_DOCK_WIDTH_TRANSITION,
  }), [rightDockCollapsed, rightDockResizing, rightDockWidth])

  return {
    readerOpen,
    readerPayload,
    readerOpenRef,
    readerPayloadRef,
    openReaderDock,
    closeReader,
    resetReaderDock,
    rightDockCollapsed,
    rightDockPanel,
    setRightDockPanel,
    showDockPanel,
    collapseRightDock,
    toggleRightDockCollapsed,
    desktopReaderEligible,
    rightDockResizing,
    rightDockStyle,
    splitLayoutRef,
    rightDockResizeGuideRef,
    beginRightDockResize,
    handleRightDockResizeMove,
    commitRightDockResize,
    cancelRightDockResize,
  }
}
