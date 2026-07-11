import { useRef } from 'react'

import type { CiteDetail } from './citationState'
import { CitationPopoverActions } from './CitationPopoverActions'
import { SystemAEvidenceCard, SystemBLiteratureCard } from './CitationPopoverCards'
import { CitationPopoverFlowStrip } from './CitationPopoverFlowStrip'
import { CitationPopoverHeader } from './CitationPopoverHeader'
import { CitationPopoverMetaPanels } from './CitationPopoverMetaPanels'
import { CitationPopoverStatusPanels } from './CitationPopoverStatusPanels'
import { buildCitationPopoverViewModel } from './citationPopoverViewModel'
import {
  getHiddenCitationPopoverStyle,
  useCitationPopoverDismiss,
  useCitationPopoverPosition,
} from './useCitationPopoverPosition'

import { useT } from '../../i18n'

interface Props {
  detail: CiteDetail | null
  position: { x: number; y: number } | null
  loading: boolean
  guideLoading: boolean
  inShelf: boolean
  onClose: () => void
  onAddToShelf: (detail: CiteDetail) => void
  onOpenShelf: () => void
  onOpenReader: (detail: CiteDetail) => void
  onStartGuide: (detail: CiteDetail) => void
  onMouseEnter?: () => void
  onMouseLeave?: () => void
  showOpenReaderAction?: boolean
  showStartGuideAction?: boolean
}

export function CitationPopover({
  detail,
  position,
  loading,
  guideLoading,
  inShelf,
  onClose,
  onAddToShelf,
  onOpenShelf,
  onOpenReader,
  onStartGuide,
  onMouseEnter,
  onMouseLeave,
  showOpenReaderAction = true,
  showStartGuideAction = true,
}: Props) {
  const S = useT()
  const ref = useRef<HTMLDivElement>(null)
  const style = useCitationPopoverPosition({
    active: Boolean(detail && position),
    popoverRef: ref,
    position,
  })
  useCitationPopoverDismiss({
    active: Boolean(detail),
    onClose,
    popoverRef: ref,
  })

  if (!detail || !position) return null

  const {
    explainText,
    frame,
    isSystemB,
    status,
    systemA,
    systemB,
  } = buildCitationPopoverViewModel({
    detail,
    S,
    loading,
  })

  return (
    <div
      ref={ref}
      className={`kb-cite-pop ${isSystemB ? 'kb-cite-pop-system-b w-[480px]' : 'kb-cite-pop-system-a w-[460px]'} fixed z-50 max-w-[calc(100vw-20px)]`}
      data-testid="citation-popover"
      style={style ?? getHiddenCitationPopoverStyle(position)}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="kb-cite-pop-scroll">
        <CitationPopoverHeader
          isSystemB={isSystemB}
          kindLabel={frame.kindLabel}
          badgeLabel={frame.badgeLabel}
          title={isSystemB ? frame.systemBTitle : frame.systemATitle}
          subtitle={frame.headerSubtitle}
          compactMetaItems={frame.compactMetaItems}
          onClose={onClose}
        />

        <CitationPopoverFlowStrip
          explainText={explainText}
          flowSteps={frame.flowSteps}
          flowAriaLabel={S.cite_flow_aria}
        />
        <CitationPopoverStatusPanels
          bindingState={status.bindingState}
          bindingOverlapText={status.bindingOverlapText}
          showBindingReason={status.showBindingReason}
          bindingReason={status.bindingReason}
          showCardQuality={status.showCardQuality}
          cardQualityFlags={status.cardQualityFlags}
          cardQualityLabel={status.cardQualityLabel}
          cardQualityScore={status.cardQualityScore}
          showCardWarning={status.showCardWarning}
          cardWarning={status.cardWarning}
          showExternalMetadataWarning={status.showExternalMetadataWarning}
          externalMetadataWarningText={status.externalMetadataWarningText}
          externalMetadataTitleHint={status.externalMetadataTitleHint}
        />
        {!isSystemB ? (
          <SystemAEvidenceCard
            showTakeaway={systemA.showTakeaway}
            takeawayLabel={systemA.takeawayLabel}
            takeawayText={systemA.takeawayText}
            card={systemA.contentCard}
            showClaim={systemA.showClaim}
            excerptLabel={S.cite_excerpt}
            showSupport={systemA.showSupport}
          />
        ) : (
          <SystemBLiteratureCard
            {...systemB}
            excerptLabel={S.cite_excerpt}
          />
        )}
        <CitationPopoverMetaPanels
          showMetaGrid={frame.showMetaGrid}
          metaRows={frame.metaRows}
          doiLabel={frame.doiLabel}
          doiHref={frame.doiHref}
          loading={loading}
          isSystemB={isSystemB}
          loadingLabel={S.cite_loading}
          showMetrics={frame.showMetrics}
          metrics={frame.metrics}
        />
      </div>

      <CitationPopoverActions
        detail={detail}
        showOpenReaderAction={showOpenReaderAction}
        canOpenReader={frame.canOpenReader}
        openReaderLabel={frame.primaryActionLabel}
        onOpenReader={onOpenReader}
        showStartGuideAction={showStartGuideAction}
        guideLoading={guideLoading}
        startGuideLabel={S.cite_start_guide}
        startingGuideLabel={S.cite_starting_guide}
        onStartGuide={onStartGuide}
        openShelfLabel={S.cite_open_shelf}
        onOpenShelf={onOpenShelf}
        inShelf={inShelf}
        addToShelfLabel={S.cite_add_to_shelf}
        inShelfLabel={S.cite_in_shelf}
        onAddToShelf={onAddToShelf}
      />
    </div>
  )
}
