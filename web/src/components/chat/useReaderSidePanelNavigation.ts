import type { RefObject } from 'react'

import type { ReaderDocBlock } from '../../api/references'
import { useReaderEvidenceNavigator, type ReaderEvidenceNavItem } from './reader/useReaderEvidenceNavigator'
import { useReaderHighlightWorkspace } from './reader/useReaderHighlightWorkspace'
import type { ReaderLocateCandidate, ReaderSessionHighlight } from './reader/readerTypes'

export interface ReaderSidePanelLabelOptions {
  activeEvidenceItem?: Pick<ReaderEvidenceNavItem, 'label'> | null
  highlightsOpen: boolean
  isPageSurface: boolean
  outlineOpen: boolean
  sessionHighlightCount: number
  S: Record<string, string>
}

export interface UseReaderSidePanelNavigationOptions {
  activeAltIndex: number
  alternatives: ReaderLocateCandidate[]
  contentRef: RefObject<HTMLDivElement | null>
  evidenceAlternatives: ReaderLocateCandidate[]
  isPageSurface: boolean
  onRemoveSessionHighlight?: (highlightId: string) => void
  open: boolean
  outlineOpen: boolean
  readerBlocks: ReaderDocBlock[]
  sessionHighlights: ReaderSessionHighlight[]
  setActiveAltIndex: (idx: number) => void
  sourcePath: string
  title: string
  S: Record<string, string>
}

export function buildReaderSourceLabel(title: string, activeHeadingPath: string): string {
  return [title, activeHeadingPath].filter(Boolean).join(' / ')
}

export function buildReaderSidePanelLabels({
  activeEvidenceItem,
  highlightsOpen,
  isPageSurface,
  outlineOpen,
  sessionHighlightCount,
  S,
}: ReaderSidePanelLabelOptions) {
  return {
    activeEvidenceLabel: String(activeEvidenceItem?.label || '').trim(),
    outlineToggleLabel: outlineOpen && !isPageSurface
      ? (S.reader_hide_sections || 'Hide sections')
      : (S.reader_sections || 'Sections'),
    highlightsToggleLabel: highlightsOpen && !isPageSurface
      ? (S.reader_hide_highlights || 'Hide highlights')
      : (S.reader_highlights_count || '{n} highlights').replace('{n}', String(sessionHighlightCount)),
  }
}

export function useReaderSidePanelNavigation({
  activeAltIndex,
  alternatives,
  contentRef,
  evidenceAlternatives,
  isPageSurface,
  onRemoveSessionHighlight,
  open,
  outlineOpen,
  readerBlocks,
  sessionHighlights,
  setActiveAltIndex,
  sourcePath,
  title,
  S,
}: UseReaderSidePanelNavigationOptions) {
  const highlightWorkspace = useReaderHighlightWorkspace({
    open,
    sourcePath,
    contentRef,
    readerBlocks,
    sessionHighlights,
    onRemoveSessionHighlight,
  })

  const evidenceNavigator = useReaderEvidenceNavigator({
    open,
    sourcePath,
    title,
    evidenceAlternatives,
    alternatives,
    activeAltIndex,
    setActiveAltIndex,
  })

  const labels = buildReaderSidePanelLabels({
    activeEvidenceItem: evidenceNavigator.activeEvidenceItem,
    highlightsOpen: highlightWorkspace.highlightsOpen,
    isPageSurface,
    outlineOpen,
    sessionHighlightCount: sessionHighlights.length,
    S,
  })

  return {
    ...highlightWorkspace,
    ...evidenceNavigator,
    ...labels,
  }
}
