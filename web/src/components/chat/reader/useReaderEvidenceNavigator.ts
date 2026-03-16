import { useEffect, useMemo } from 'react'
import type { ReaderLocateCandidate } from './readerTypes'
import { candidateDisplayLabel, candidateIdentityKey } from './readerDomUtils'

export interface ReaderEvidenceNavItem {
  id: string
  label: string
  headingPath: string
  blockId: string
  anchorId: string
  anchorKind: string
  anchorNumber: number
  snippet: string
  index: number
  targetIndex: number
}

interface UseReaderEvidenceNavigatorArgs {
  open: boolean
  sourcePath: string
  title: string
  evidenceAlternatives: ReaderLocateCandidate[]
  alternatives: ReaderLocateCandidate[]
  activeAltIndex: number
  setActiveAltIndex: (idx: number) => void
}

export function useReaderEvidenceNavigator({
  open,
  sourcePath,
  title,
  evidenceAlternatives,
  alternatives,
  activeAltIndex,
  setActiveAltIndex,
}: UseReaderEvidenceNavigatorArgs) {
  const internalIndexByKey = useMemo(() => {
    const out = new Map<string, number>()
    alternatives.forEach((item, index) => {
      out.set(candidateIdentityKey(item), index)
    })
    return out
  }, [alternatives])

  const evidenceItems = useMemo<ReaderEvidenceNavItem[]>(() => {
    return evidenceAlternatives.flatMap((item, index) => {
      const targetIndex = internalIndexByKey.get(candidateIdentityKey(item))
      if (!Number.isFinite(targetIndex)) return []
      return [{
        id: String(item.blockId || item.anchorId || `evidence-${index + 1}`).trim(),
        label: candidateDisplayLabel(item, title) || `Evidence ${index + 1}`,
        headingPath: String(item.headingPath || '').trim(),
        blockId: String(item.blockId || '').trim(),
        anchorId: String(item.anchorId || '').trim(),
        anchorKind: String(item.anchorKind || '').trim().toLowerCase(),
        anchorNumber: Number.isFinite(Number(item.anchorNumber || 0))
          ? Math.floor(Number(item.anchorNumber || 0))
          : 0,
        snippet: String(item.highlightSnippet || item.snippet || '').trim(),
        index,
        targetIndex: Number(targetIndex),
      }]
    })
  }, [evidenceAlternatives, internalIndexByKey, title])

  const activeEvidenceIndex = useMemo(() => {
    if (evidenceItems.length <= 0) return -1
    return evidenceItems.findIndex((item) => item.targetIndex === activeAltIndex)
  }, [evidenceItems, activeAltIndex])

  useEffect(() => {
    if (!open) return
    if (evidenceItems.length <= 0) return
    if (activeEvidenceIndex >= 0) return
    setActiveAltIndex(evidenceItems[0].targetIndex)
  }, [open, sourcePath, evidenceItems, activeEvidenceIndex, setActiveAltIndex])

  const safeIndex = evidenceItems.length > 0
    ? (activeEvidenceIndex >= 0 ? activeEvidenceIndex : 0)
    : 0

  return {
    evidenceItems,
    activeEvidenceIndex: safeIndex,
    activeEvidenceItem: evidenceItems[safeIndex] || null,
    hasEvidenceNav: evidenceItems.length > 1,
    canGoPrevEvidence: safeIndex > 0,
    canGoNextEvidence: safeIndex < evidenceItems.length - 1,
    evidencePositionLabel: evidenceItems.length > 0 ? `${safeIndex + 1} / ${evidenceItems.length}` : '',
    goPrevEvidence: () => {
      if (safeIndex <= 0) return
      setActiveAltIndex(evidenceItems[safeIndex - 1].targetIndex)
    },
    goNextEvidence: () => {
      if (safeIndex >= evidenceItems.length - 1) return
      setActiveAltIndex(evidenceItems[safeIndex + 1].targetIndex)
    },
  }
}
