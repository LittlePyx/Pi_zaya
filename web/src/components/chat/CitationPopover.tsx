import { useEffect, useLayoutEffect, useRef, useState } from 'react'
/* eslint-disable react-hooks/set-state-in-effect */

import type { CiteDetail } from './citationState'
import {
  citationCardView,
} from './citationState'
import { CitationPopoverActions } from './CitationPopoverActions'
import { SystemAEvidenceCard, SystemBLiteratureCard } from './CitationPopoverCards'
import { CitationPopoverFlowStrip } from './CitationPopoverFlowStrip'
import { CitationPopoverHeader } from './CitationPopoverHeader'
import { CitationPopoverMetaPanels } from './CitationPopoverMetaPanels'
import { CitationPopoverStatusPanels } from './CitationPopoverStatusPanels'
import {
  compact,
  looksNarrativeMetadataText,
} from './citationPopoverUtils'
import { buildCitationPopoverFrameModel } from './citationPopoverFrameModel'
import { buildCitationPopoverStatusModel } from './citationPopoverStatusModel'
import { buildSystemAEvidenceCardModel } from './citationPopoverSystemA'
import { buildSystemBLiteratureCardModel } from './citationPopoverSystemB'

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
  const [style, setStyle] = useState<{ left: number; top: number } | null>(null)
  const localizeKnownLabel = (value: string): string => {
    const text = compact(value)
    if (!text) return ''
    const labels: Record<string, string> = {
      上游引用: S.cite_kind_upstream,
      答案依据: S.cite_kind_evidence,
      答案中的话: S.cite_answer_point,
      对应回答: S.cite_answer_point,
      答案要点: S.cite_answer_point,
      引用语境: S.cite_context,
      语境摘要: S.cite_context_summary,
      链路已闭合: S.cite_trace_complete,
      链路需核对: S.cite_trace_review,
      疑似错配: S.cite_binding_mismatch,
      候选依据: S.cite_binding_candidate,
      上游参考文献: S.cite_upstream_reference,
      引用所在论文: S.cite_location_paper,
      当前论文引用处: S.cite_location_current,
      来源: S.cite_meta_source,
      发表: S.cite_meta_published,
      作者: S.cite_meta_author,
      位置: S.cite_position,
      锚点: S.cite_anchor_label,
      证据重点: S.cite_evidence_focus,
      原文证据: S.cite_original_evidence,
      可靠度: S.cite_reliability,
      证据链: S.cite_evidence_chain,
      上游作用: S.cite_upstream_role,
      上游文献条目: S.cite_reference_entry,
      说明: S.cite_note,
      'Missing reference entry': S.cite_missing_reference_entry,
    }
    return labels[text] || text
  }
  const localizeKnownBody = (value: string): string => {
    const text = compact(value)
    if (!text) return ''
    const missingReferenceMatch = text.match(/^Reference \[(\d{1,4})]\s+is cited in the opened Reader document, but the converted References section does not contain a matching bibliography entry\.?$/i)
    if (missingReferenceMatch) {
      return S.cite_missing_reference_entry_body.replace('{n}', missingReferenceMatch[1])
    }
    if (/前端缺少后端 cite_details/.test(text)) return S.cite_frontend_candidate_reason
    if (/前端根据本轮 References 临时补齐/.test(text)) return S.cite_candidate_support_default
    if (/这条引用只能作为候选依据/.test(text)) return S.cite_candidate_support_default
    return text
  }

  useEffect(() => {
    if (!detail) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    const onPointerDown = (event: MouseEvent) => {
      const el = ref.current
      if (!el) return
      const targetEl = event.target instanceof Element ? event.target : null
      if (targetEl?.closest('.kb-md-locate-inline-btn, .kb-prov-locate-chip, [data-kb-locate-block-id]')) return
      if (event.target instanceof Node && !el.contains(event.target)) onClose()
    }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('mousedown', onPointerDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('mousedown', onPointerDown)
    }
  }, [detail, onClose])

  useLayoutEffect(() => {
    if (!detail || !position || !ref.current) {
      setStyle(null)
      return
    }
    const rect = ref.current.getBoundingClientRect()
    const margin = 12
    const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin)
    const maxTop = Math.max(margin, window.innerHeight - rect.height - margin)
    setStyle({
      left: Math.min(Math.max(margin, position.x + 10), maxLeft),
      top: Math.min(Math.max(margin, position.y + 28), maxTop),
    })
  }, [detail, position])

  if (!detail || !position) return null

  const isSystemB = Boolean(detail.isInpaper)
  const view = citationCardView(detail)
  const viewSection = (id: string) => view.sections.find((item) => item.id === id)
  const takeawaySection = viewSection('takeaway')
  const claimSection = viewSection('claim')
  const locatorSection = viewSection('locator')
  const contextSummarySection = viewSection('context_summary')
  const evidenceSection = viewSection('evidence')
  const referenceSection = viewSection('reference')
  const supportSection = viewSection('support')
  const warningSection = viewSection('warning')
  const cardTakeawayLabel = localizeKnownLabel(takeawaySection?.label || detail.cardTakeawayLabel)
  const rawCardTakeaway = compact(takeawaySection?.text || detail.cardTakeaway)
  const cardTakeaway = looksNarrativeMetadataText(rawCardTakeaway, detail) ? '' : rawCardTakeaway
  const cardClaimLabel = localizeKnownLabel(claimSection?.label || detail.cardClaimLabel)
  const cardEvidenceLabel = localizeKnownLabel(evidenceSection?.label || detail.cardEvidenceLabel)
  const cardLocatorLabel = localizeKnownLabel(locatorSection?.label || detail.cardLocatorLabel)
  const frame = buildCitationPopoverFrameModel({
    detail,
    S,
    isSystemB,
    viewHeader: view.header,
    locatorSection,
    cardLocatorLabel,
    localizeKnownLabel,
  })
  const cardReferenceLabel = localizeKnownLabel(referenceSection?.label || detail.cardReferenceLabel)
  const cardSupportLabel = localizeKnownLabel(supportSection?.label || detail.cardSupportLabel)
  const status = buildCitationPopoverStatusModel({
    detail,
    S,
    isSystemB,
    supportSection,
    warningSection,
    displayMain: frame.displayMain,
    localizeKnownBody,
    localizeKnownLabel,
  })
  const systemA = buildSystemAEvidenceCardModel({
    detail,
    S,
    isSystemB,
    claimSection,
    evidenceSection,
    supportSection,
    cardTakeaway,
    cardTakeawayLabel,
    cardClaimLabel,
    cardEvidenceLabel,
    cardSupportLabel,
    cardQualityFlags: status.cardQualityFlags,
    cardWarning: status.cardWarning,
    hasBindingState: Boolean(status.bindingState),
    supportText: status.supportText,
  })
  const explainText = ''
  const systemB = buildSystemBLiteratureCardModel({
    detail,
    S,
    isSystemB,
    loading,
    locatorSection,
    contextSummarySection,
    referenceSection,
    cardTakeaway,
    cardEvidenceLabel,
    cardReferenceLabel,
    cardSupportLabel,
    cardQualityFlags: status.cardQualityFlags,
    sourcePaperText: frame.sourcePaperText,
    headingPath: frame.headingPath,
    pageLabel: frame.pageLabel,
    badgeLabel: frame.badgeLabel,
    doiLabel: frame.doiLabel,
    systemBTitle: frame.systemBTitle,
    systemBTitleMissing: frame.systemBTitleMissing,
    headerSubtitle: frame.headerSubtitle,
    metrics: frame.metrics,
    explicitSupportText: status.explicitSupportText,
    displaySource: frame.displaySource,
    localizeKnownBody,
    localizeKnownLabel,
  })

  return (
    <div
      ref={ref}
      className={`kb-cite-pop ${isSystemB ? 'kb-cite-pop-system-b w-[480px]' : 'kb-cite-pop-system-a w-[460px]'} fixed z-50 max-w-[calc(100vw-20px)]`}
      data-testid="citation-popover"
      style={style ?? { left: position.x + 10, top: position.y + 10, visibility: 'hidden' }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
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
