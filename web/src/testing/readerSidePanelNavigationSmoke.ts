import {
  buildReaderSidePanelLabels,
  buildReaderSourceLabel,
} from '../components/chat/useReaderSidePanelNavigation'

export interface ReaderSidePanelNavigationSmokeResult {
  closedDock: ReturnType<typeof buildReaderSidePanelLabels>
  openDock: ReturnType<typeof buildReaderSidePanelLabels>
  pageSurface: ReturnType<typeof buildReaderSidePanelLabels>
  sourceLabel: string
  sourceOnlyLabel: string
}

const labels = {
  reader_hide_highlights: 'Hide highlights',
  reader_hide_sections: 'Hide sections',
  reader_highlights_count: '{n} notes',
  reader_sections: 'Sections',
}

export function runReaderSidePanelNavigationSmoke(): ReaderSidePanelNavigationSmokeResult {
  return {
    closedDock: buildReaderSidePanelLabels({
      activeEvidenceItem: null,
      highlightsOpen: false,
      isPageSurface: false,
      outlineOpen: false,
      sessionHighlightCount: 0,
      S: labels,
    }),
    openDock: buildReaderSidePanelLabels({
      activeEvidenceItem: { label: ' Evidence A ' },
      highlightsOpen: true,
      isPageSurface: false,
      outlineOpen: true,
      sessionHighlightCount: 3,
      S: labels,
    }),
    pageSurface: buildReaderSidePanelLabels({
      activeEvidenceItem: { label: 'Page Evidence' },
      highlightsOpen: true,
      isPageSurface: true,
      outlineOpen: true,
      sessionHighlightCount: 4,
      S: labels,
    }),
    sourceLabel: buildReaderSourceLabel('Reader Paper', 'Methods'),
    sourceOnlyLabel: buildReaderSourceLabel('Reader Paper', ''),
  }
}
