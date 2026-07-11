import { useEffect, useLayoutEffect, useState } from 'react'
import type { CSSProperties, RefObject } from 'react'
/* eslint-disable react-hooks/set-state-in-effect */

export interface CitationPopoverPoint {
  x: number
  y: number
}

export interface CitationPopoverRect {
  height: number
  width: number
}

export interface CitationPopoverViewport {
  height: number
  width: number
}

export interface CitationPopoverPositionOptions {
  margin?: number
  offsetX?: number
  offsetY?: number
}

interface UseCitationPopoverPositionOptions<T extends HTMLElement> {
  active: boolean
  popoverRef: RefObject<T | null>
  position: CitationPopoverPoint | null
}

interface UseCitationPopoverDismissOptions<T extends HTMLElement> {
  active: boolean
  onClose: () => void
  popoverRef: RefObject<T | null>
}

const DEFAULT_MARGIN = 12
const DEFAULT_OFFSET_X = 10
const DEFAULT_OFFSET_Y = 28
const HIDDEN_OFFSET = 10

export const CITATION_POPOVER_DISMISS_IGNORE_SELECTOR = [
  '.kb-md-locate-inline-btn',
  '.kb-prov-locate-chip',
  '[data-kb-locate-block-id]',
].join(', ')

function samePlacement(left: number, top: number, current: CSSProperties | null): boolean {
  return current?.left === left && current?.top === top && current.visibility === undefined
}

export function getCitationPopoverPositionStyle(
  position: CitationPopoverPoint,
  rect: CitationPopoverRect,
  viewport: CitationPopoverViewport,
  options: CitationPopoverPositionOptions = {},
): CSSProperties {
  const margin = options.margin ?? DEFAULT_MARGIN
  const offsetX = options.offsetX ?? DEFAULT_OFFSET_X
  const offsetY = options.offsetY ?? DEFAULT_OFFSET_Y
  const maxLeft = Math.max(margin, viewport.width - rect.width - margin)
  const maxTop = Math.max(margin, viewport.height - rect.height - margin)
  return {
    left: Math.min(Math.max(margin, position.x + offsetX), maxLeft),
    top: Math.min(Math.max(margin, position.y + offsetY), maxTop),
  }
}

export function getHiddenCitationPopoverStyle(position: CitationPopoverPoint): CSSProperties {
  return {
    left: position.x + HIDDEN_OFFSET,
    top: position.y + HIDDEN_OFFSET,
    visibility: 'hidden',
  }
}

export function isCitationPopoverDismissIgnoredTarget(target: EventTarget | null): boolean {
  const targetEl = target instanceof Element ? target : null
  return Boolean(targetEl?.closest(CITATION_POPOVER_DISMISS_IGNORE_SELECTOR))
}

export function useCitationPopoverPosition<T extends HTMLElement>({
  active,
  popoverRef,
  position,
}: UseCitationPopoverPositionOptions<T>): CSSProperties | null {
  const [style, setStyle] = useState<CSSProperties | null>(null)

  useLayoutEffect(() => {
    if (!active || !position || !popoverRef.current) {
      setStyle(null)
      return undefined
    }

    const updatePosition = () => {
      const el = popoverRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const next = getCitationPopoverPositionStyle(position, rect, {
        height: window.innerHeight,
        width: window.innerWidth,
      })
      setStyle((current) => (samePlacement(Number(next.left), Number(next.top), current) ? current : next))
    }

    updatePosition()
    const resizeObserver = typeof ResizeObserver === 'undefined'
      ? null
      : new ResizeObserver(updatePosition)
    resizeObserver?.observe(popoverRef.current)
    window.addEventListener('resize', updatePosition)
    window.addEventListener('scroll', updatePosition, true)
    return () => {
      resizeObserver?.disconnect()
      window.removeEventListener('resize', updatePosition)
      window.removeEventListener('scroll', updatePosition, true)
    }
  }, [active, popoverRef, position])

  return style
}

export function useCitationPopoverDismiss<T extends HTMLElement>({
  active,
  onClose,
  popoverRef,
}: UseCitationPopoverDismissOptions<T>) {
  useEffect(() => {
    if (!active) return undefined

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    const onPointerDown = (event: MouseEvent) => {
      const el = popoverRef.current
      if (!el) return
      if (isCitationPopoverDismissIgnoredTarget(event.target)) return
      if (event.target instanceof Node && !el.contains(event.target)) onClose()
    }

    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('mousedown', onPointerDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('mousedown', onPointerDown)
    }
  }, [active, onClose, popoverRef])
}
