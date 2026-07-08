import { useCallback } from 'react'

import {
  toShelfItem,
  type CiteDetail,
} from './citationState'
import {
  buildReaderCitationPopoverMetadataPlan,
  loadReaderCitationPopoverMetadata,
  type LoadReaderCitationPopoverMetadataOptions,
  type ReaderCitationPopoverMetadataResult,
} from './readerCitationPopoverMetadata'
import {
  useCitationPopoverState,
  type CitationPopoverPoint,
} from './useCitationPopoverState'

export interface ReaderCitationPopoverOpenEvent {
  clientX: number
  clientY: number
}

export type ReaderCitationPopoverMetadataLoader = (
  detail: CiteDetail,
  options?: LoadReaderCitationPopoverMetadataOptions,
) => Promise<ReaderCitationPopoverMetadataResult>

export interface UseReaderCitationPopoverOptions {
  loadMetadata?: ReaderCitationPopoverMetadataLoader
}

export interface ReaderCitationPopoverController {
  close: () => void
  detail: CiteDetail | null
  loading: boolean
  position: CitationPopoverPoint | null
  showCitation: (detail: CiteDetail, event: ReaderCitationPopoverOpenEvent) => void
}

export function useReaderCitationPopover(
  options: UseReaderCitationPopoverOptions = {},
): ReaderCitationPopoverController {
  const {
    activeRequestKeyRef,
    close,
    detail,
    loading,
    mergeDetailForKey,
    open,
    position,
    setLoading,
  } = useCitationPopoverState()
  const loadMetadata = options.loadMetadata ?? loadReaderCitationPopoverMetadata

  const mergeMetadata = useCallback((itemKey: string, metas: Array<Record<string, unknown>>) => {
    if (metas.length <= 0) return
    mergeDetailForKey(itemKey, metas)
  }, [mergeDetailForKey])

  const showCitation = useCallback((nextDetail: CiteDetail, event: ReaderCitationPopoverOpenEvent) => {
    const itemKey = toShelfItem(nextDetail).key
    open(nextDetail, { x: event.clientX, y: event.clientY }, { requestKey: itemKey })
    const metadataPlan = buildReaderCitationPopoverMetadataPlan(nextDetail, itemKey)
    if (metadataPlan.requestCount <= 0) {
      setLoading(false)
      return
    }

    setLoading(true)
    loadMetadata(nextDetail, { plan: metadataPlan })
      .then(({ metas }) => {
        if (activeRequestKeyRef.current !== itemKey) return
        mergeMetadata(itemKey, metas)
      })
      .finally(() => {
        if (activeRequestKeyRef.current === itemKey) {
          setLoading(false)
        }
      })
  }, [activeRequestKeyRef, loadMetadata, mergeMetadata, open, setLoading])

  return {
    close,
    detail,
    loading,
    position,
    showCitation,
  }
}
