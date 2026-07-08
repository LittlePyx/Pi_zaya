import { useCallback, useRef, useState } from 'react'
import type {
  Dispatch,
  MutableRefObject,
  SetStateAction,
} from 'react'

import {
  mergeCiteMeta,
  toShelfItem,
  type CiteDetail,
} from './citationState'

export interface CitationPopoverPoint {
  x: number
  y: number
}

export interface CitationPopoverOpenOptions {
  guideLoading?: boolean
  loading?: boolean
  pinned?: boolean
  requestKey?: string
}

export interface CitationPopoverStateController {
  activeRequestKeyRef: MutableRefObject<string>
  close: () => void
  detail: CiteDetail | null
  guideLoading: boolean
  loading: boolean
  mergeDetailForKey: (itemKey: string, metas: Array<Record<string, unknown>>) => Array<Record<string, unknown>>
  open: (detail: CiteDetail, position: CitationPopoverPoint, options?: CitationPopoverOpenOptions) => void
  pinned: boolean
  position: CitationPopoverPoint | null
  setDetail: Dispatch<SetStateAction<CiteDetail | null>>
  setGuideLoading: Dispatch<SetStateAction<boolean>>
  setLoading: Dispatch<SetStateAction<boolean>>
  setPinned: Dispatch<SetStateAction<boolean>>
}

export function getUsableCitationPopoverMetas(
  metas: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  return metas.filter((meta) => meta && Object.keys(meta).length > 0)
}

export function useCitationPopoverState(): CitationPopoverStateController {
  const [detail, setDetail] = useState<CiteDetail | null>(null)
  const [position, setPosition] = useState<CitationPopoverPoint | null>(null)
  const [loading, setLoading] = useState(false)
  const [guideLoading, setGuideLoading] = useState(false)
  const [pinned, setPinned] = useState(false)
  const activeRequestKeyRef = useRef('')

  const open = useCallback((
    nextDetail: CiteDetail,
    nextPosition: CitationPopoverPoint,
    options: CitationPopoverOpenOptions = {},
  ) => {
    activeRequestKeyRef.current = options.requestKey ?? toShelfItem(nextDetail).key
    setPinned(Boolean(options.pinned))
    setDetail(nextDetail)
    setPosition(nextPosition)
    setLoading(Boolean(options.loading))
    setGuideLoading(Boolean(options.guideLoading))
  }, [])

  const close = useCallback(() => {
    activeRequestKeyRef.current = ''
    setPinned(false)
    setDetail(null)
    setPosition(null)
    setLoading(false)
    setGuideLoading(false)
  }, [])

  const mergeDetailForKey = useCallback((itemKey: string, metas: Array<Record<string, unknown>>) => {
    const usable = getUsableCitationPopoverMetas(metas)
    if (!usable.length) return usable
    setDetail((current) => {
      if (!current) return current
      if (toShelfItem(current).key !== itemKey) return current
      let merged = current
      for (const meta of usable) {
        merged = mergeCiteMeta(merged, meta)
      }
      return merged
    })
    return usable
  }, [])

  return {
    activeRequestKeyRef,
    close,
    detail,
    guideLoading,
    loading,
    mergeDetailForKey,
    open,
    pinned,
    position,
    setDetail,
    setGuideLoading,
    setLoading,
    setPinned,
  }
}
