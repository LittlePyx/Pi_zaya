import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Collapse, Modal, Tabs, Typography, message } from 'antd'
import { useNavigate } from 'react-router-dom'
import { useT } from '../../i18n'
import { referenceSourcePathCacheKey, referencesApi } from '../../api/references'
import { useChatStore } from '../../stores/chatStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { basenameFromSourcePath } from '../../utils/sourcePath'
import { internalDebugBrowserEnabled } from '../../utils/internalDebug'
import type { ReaderOpenPayload } from '../chat/reader/readerTypes'
import { buildBasicReaderOpenPayload } from '../chat/reader/readerOpenPayloadUtils'
import {
  buildCiteDetailFromMeta,
  citationDisplay,
  citationFormats,
  citeMetricSummary,
  type CiteDetail,
} from '../chat/citationState'
import {
  prepareRefsPanelHits,
  type RefsPanelRefEntry as RefEntry,
  type RefsPanelRefHit as RefHit,
  type RefsPanelRefUiMeta as RefUiMeta,
} from './refsPanelDisplay'
import { selectLocalizedRefCardText, selectRefRelevanceText } from './refCardCopy'

const { Link, Text } = Typography
const expandedRefsPanelKeys = new Set<string>()


function refsPanelExpansionKey(msgId: number) {
  return String(Number(msgId || 0) || 0)
}

function refSourceStateKey(hit: RefHit, index: number) {
  const ui = hit.ui_meta || {}
  const readerOpen = (ui.reader_open && typeof ui.reader_open === 'object') ? ui.reader_open : {}
  const sourcePath = String(ui.source_path || readerOpen.sourcePath || hit.meta?.source_path || '').trim()
  const sourceKey = referenceSourcePathCacheKey(sourcePath)
  return sourceKey ? `source:${sourceKey}` : `row:${index}`
}

interface RefCardViewSection {
  id?: string
  label?: string
  title?: string
  text?: string
  kind?: string
  tone?: string
  source?: string
}

interface RefCardView {
  version?: number
  route?: string
  kind?: string
  header?: {
    kicker?: string
    title?: string
    subtitle?: string
  }
  sections?: RefCardViewSection[]
  summary?: string
  quality?: Record<string, unknown>
}

interface Props {
  refs: Record<string, unknown>
  msgId: number
  onOpenReader?: (payload: ReaderOpenPayload) => void
  activeSourcePath?: string
  activeSourceName?: string
}


function hasResolvedCitationMeta(meta: Record<string, unknown> | null | undefined) {
  const rec = meta || {}
  const title = String(rec.title || '').trim()
  const venue = String(rec.venue || '').trim()
  const year = String(rec.year || '').trim()
  const doi = String(rec.doi || rec.doi_url || '').trim()
  const conferenceTier = String(rec.conference_tier || '').trim()
  const journalIf = String(rec.journal_if || '').trim()
  const citationCount = Number(rec.citation_count || 0)
  return Boolean(title || venue || year || doi || conferenceTier || journalIf || citationCount > 0)
}

function positiveNumber(input: unknown): number {
  const value = Number(input)
  return Number.isFinite(value) && value > 0 ? value : 0
}

function normalizeUiText(input: string) {
  return String(input || '').replace(/\s+/g, ' ').trim().toLowerCase()
}



function shouldShowSemanticBadge(text: string) {
  const low = normalizeUiText(text)
  if (!low) return false
  const blocked = [
    '语义直连',
    '文档语义',
    'semantic',
    'vector',
    'embedding',
    'dense',
    'sparse',
    'bm25',
    'keyword',
    '关键词',
    'lexical',
    'rerank',
    'cross encoder',
    'cross-encoder',
  ]
  return !blocked.some((token) => low.includes(token))
}

function normalizePolishStatus(input: unknown) {
  const status = String(input || '').trim().toLowerCase()
  if (status === 'full' || status === 'heuristic' || status === 'pending' || status === 'failed') return status
  return ''
}

function cleanRefViewText(input: unknown) {
  return String(input || '').replace(/\s+/g, ' ').trim()
}

function normalizeRefCardView(input: unknown): RefCardView | null {
  if (!input || typeof input !== 'object') return null
  const rec = input as RefCardView
  const sections = Array.isArray(rec.sections)
    ? rec.sections
        .map((section) => ({
          id: cleanRefViewText(section?.id),
          label: cleanRefViewText(section?.label),
          title: cleanRefViewText(section?.title),
          text: cleanRefViewText(section?.text),
          kind: cleanRefViewText(section?.kind),
          tone: cleanRefViewText(section?.tone),
          source: cleanRefViewText(section?.source),
        }))
        .filter((section) => section.id && section.text)
    : []
  if (!sections.length && !cleanRefViewText(rec.header?.title)) return null
  return {
    ...rec,
    header: {
      kicker: cleanRefViewText(rec.header?.kicker),
      title: cleanRefViewText(rec.header?.title),
      subtitle: cleanRefViewText(rec.header?.subtitle),
    },
    sections,
    summary: cleanRefViewText(rec.summary),
  }
}

function refCardSection(view: RefCardView | null, id: string): RefCardViewSection | null {
  return view?.sections?.find((section) => section.id === id) || null
}

function polishStatusLabel(status: string, S: ReturnType<typeof useT>) {
  if (status === 'full') return S.refs_polish_full
  if (status === 'pending') return S.refs_polish_pending
  if (status === 'failed') return S.refs_polish_failed
  if (status === 'heuristic') return S.refs_polish_heuristic
  return ''
}

export function RefsPanel({ refs, msgId, onOpenReader, activeSourcePath, activeSourceName }: Props) {
  const S = useT()
  const uiLocale = useSettingsStore((state) => state.uiLocale)
  const refsCardLocale = useSettingsStore((state) => state.refsCardLocale)
  const cardCopyLocale = refsCardLocale === 'auto' ? uiLocale : refsCardLocale
  const showInternalRefDiagnostics = internalDebugBrowserEnabled()
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const nav = useNavigate()
  const expansionKey = refsPanelExpansionKey(msgId)
  const [activeKeys, setActiveKeys] = useState<string[]>(() => (
    expandedRefsPanelKeys.has(expansionKey) ? ['refs'] : []
  ))
  const entry = refs[String(msgId)] as RefEntry | undefined
  const displayState = String(entry?.display_state || '').trim().toLowerCase()
  const suppressionReason = String(entry?.suppression_reason || '').trim().toLowerCase()
  const hasBackendDisplayState = Boolean(displayState)
  const suggestionText = String(entry?.suggestion || '').trim()
  const rawHitCount = Array.isArray(entry?.hits) ? entry.hits.length : 0
  const preparedHits = useMemo(
    () => prepareRefsPanelHits(entry, { activeSourcePath, activeSourceName }),
    [activeSourceName, activeSourcePath, entry],
  )
  const visibleHits = preparedHits.hits
  const hiddenActiveSourceCount = preparedHits.hiddenActiveSourceCount
  const guideFilter = entry?.guide_filter || {}
  const pendingCount = visibleHits.filter((hit) => String(hit?.meta?.ref_pack_state || '').trim().toLowerCase() === 'pending').length
  const hasPending = displayState === 'pending' || pendingCount > 0
  const filteredSelfCount = positiveNumber(guideFilter.filtered_hit_count)
  const isActiveSourceFilteredOnly = hiddenActiveSourceCount > 0 && visibleHits.length === 0 && Boolean(String(activeSourcePath || activeSourceName || '').trim())
  const shouldShowGuideFilterNote = !isActiveSourceFilteredOnly && !hasPending && (
    displayState === 'hidden_by_guide'
    || ((!hasBackendDisplayState) && rawHitCount === 0 && Boolean(guideFilter.hidden_self_source))
  )
  const shouldShowNegativeSuppressedNote = displayState === 'suppressed'
  const shouldShowEmptyNote = !hasPending && displayState === 'empty'
  const suppressionNoteText = suppressionReason === 'focus_filter_removed_all'
    ? S.refs_suppressed_focus
    : suppressionReason === 'llm_filter_removed_all'
      ? S.refs_suppressed_llm
      : suppressionReason === 'score_gate_removed_all'
        ? S.refs_suppressed_score
        : suppressionReason === 'render_failed'
          ? S.refs_suppressed_render
          : S.refs_suppressed_default
  const [citeSourceKey, setCiteSourceKey] = useState<string | null>(null)
  const [loadingSourceKey, setLoadingSourceKey] = useState<string | null>(null)
  const [guideLoadingSourceKey, setGuideLoadingSourceKey] = useState<string | null>(null)
  const [remoteMeta, setRemoteMeta] = useState<Record<string, Record<string, unknown>>>({})
  const autoFetchedCitationMetaRef = useRef<Set<string>>(new Set())

  const handleCollapseChange = (keys: string | string[]) => {
    const nextKeys = (Array.isArray(keys) ? keys : [keys])
      .map((key) => String(key || '').trim())
      .filter(Boolean)
    setActiveKeys(nextKeys)
    if (nextKeys.includes('refs')) {
      expandedRefsPanelKeys.add(expansionKey)
    } else {
      expandedRefsPanelKeys.delete(expansionKey)
    }
  }

  const fetchCitationMeta = useCallback(async (sourceKey: string, ui: RefUiMeta, options?: { silent?: boolean }) => {
    const sourcePath = String(ui.source_path || '').trim()
    if (!sourcePath) return
    const silent = Boolean(options?.silent)
    if (!silent) {
      setLoadingSourceKey(sourceKey)
    }
    try {
      const meta = await referencesApi.citationMetaCached(sourcePath)
      setRemoteMeta((current) => ({ ...current, [sourceKey]: meta }))
    } catch (err) {
      if (!silent) {
        message.error(err instanceof Error ? err.message : S.refs_fetch_meta_failed)
      }
    } finally {
      if (!silent) {
        setLoadingSourceKey((current) => (current === sourceKey ? null : current))
      }
    }
  }, [S.refs_fetch_meta_failed])

  useEffect(() => {
    // Bibliographic metadata is supplementary to the already-visible evidence
    // card. Do not spend API/SQLite capacity on collapsed shelves while the
    // answer and its citation packet are still settling.
    if (!activeKeys.includes('refs') || hasPending || visibleHits.length <= 0) return
    for (const [index, hit] of visibleHits.entries()) {
      const ui = hit.ui_meta || {}
      const sourceKey = refSourceStateKey(hit, index)
      const sourcePath = String(ui.source_path || '').trim()
      if (!sourcePath) continue
      const existingMeta = (remoteMeta[sourceKey] || ui.citation_meta || {}) as Record<string, unknown>
      if (hasResolvedCitationMeta(existingMeta)) continue
      const fetchKey = `${msgId}:${sourceKey}`
      if (autoFetchedCitationMetaRef.current.has(fetchKey)) continue
      autoFetchedCitationMetaRef.current.add(fetchKey)
      void fetchCitationMeta(sourceKey, ui, { silent: true })
    }
  }, [activeKeys, fetchCitationMeta, hasPending, msgId, remoteMeta, visibleHits])

  const citeDetail = useMemo<CiteDetail | null>(() => {
    if (citeSourceKey === null) return null
    const citeIndex = visibleHits.findIndex((hit, index) => refSourceStateKey(hit, index) === citeSourceKey)
    if (citeIndex < 0 || !visibleHits[citeIndex]) return null
    const ui = visibleHits[citeIndex]?.ui_meta || {}
    const meta = remoteMeta[citeSourceKey] || ui.citation_meta
    return buildCiteDetailFromMeta(meta as Record<string, unknown>, {
      sourceName: basenameFromSourcePath(ui.display_name || ui.source_path),
      sourcePath: ui.source_path,
      num: citeIndex + 1,
      anchor: `ref-source-${msgId}-${citeIndex}`,
    })
  }, [citeSourceKey, visibleHits, msgId, remoteMeta])

  const startPaperGuideFromHit = async (sourceKey: string, ui: RefUiMeta) => {
    const sourcePath = String(ui.source_path || '').trim()
    if (!sourcePath) {
      message.info(S.refs_reader_missing)
      return
    }
    const sourceName = basenameFromSourcePath(ui.display_name || sourcePath) || S.default_source_fallback
    setGuideLoadingSourceKey(sourceKey)
    try {
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: `${S.timeline_guide_label} · ${sourceName}`,
      })
      nav('/')
      message.success(S.refs_guide_started)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.refs_guide_failed)
    } finally {
      setGuideLoadingSourceKey((current) => (current === sourceKey ? null : current))
    }
  }

  const openReaderFromHit = (ui: RefUiMeta) => {
    if (!onOpenReader) return
    const readerOpen = (ui.reader_open && typeof ui.reader_open === 'object') ? ui.reader_open : {}
    const sourcePath = String(readerOpen.sourcePath || ui.source_path || '').trim()
    if (!sourcePath) {
      message.info(S.refs_reader_missing)
      return
    }
    const payload = buildBasicReaderOpenPayload({
      sourcePath,
      sourceName: basenameFromSourcePath(readerOpen.sourceName || ui.display_name || sourcePath),
      headingPath: String(readerOpen.headingPath || ui.heading_path || ui.section_label || ui.subsection_label || '').trim(),
      snippet: String(readerOpen.snippet || ui.summary_line || ui.why_line || '').trim(),
      highlightSnippet: String(readerOpen.highlightSnippet || readerOpen.snippet || ui.summary_line || ui.why_line || '').trim(),
      anchorId: String((readerOpen as Partial<ReaderOpenPayload>).anchorId || '').trim(),
      blockId: String((readerOpen as Partial<ReaderOpenPayload>).blockId || '').trim(),
      relatedBlockIds: Array.isArray((readerOpen as Partial<ReaderOpenPayload>).relatedBlockIds)
        ? (readerOpen as Partial<ReaderOpenPayload>).relatedBlockIds
        : undefined,
      anchorKind: String(readerOpen.anchorKind || '').trim(),
      anchorNumber: Number(readerOpen.anchorNumber || 0),
      strictLocate: Boolean((readerOpen as Partial<ReaderOpenPayload>).strictLocate),
      locateTarget: (readerOpen as Partial<ReaderOpenPayload>).locateTarget || null,
      alternatives: Array.isArray(readerOpen.alternatives) ? readerOpen.alternatives : undefined,
      visibleAlternatives: Array.isArray(readerOpen.visibleAlternatives) ? readerOpen.visibleAlternatives : undefined,
      evidenceAlternatives: Array.isArray(readerOpen.evidenceAlternatives) ? readerOpen.evidenceAlternatives : undefined,
      initialAltIndex: Number.isFinite(Number(readerOpen.initialAltIndex)) ? Number(readerOpen.initialAltIndex) : undefined,
      fallbackSourceName: S.default_source_fallback,
    })
    if (!payload) return
    onOpenReader(payload)
  }

  if (!entry || (!hasPending && visibleHits.length === 0 && !shouldShowGuideFilterNote && !shouldShowNegativeSuppressedNote && !shouldShowEmptyNote)) return null

  return (
    <>
      <Collapse
        size="middle"
        activeKey={activeKeys}
        onChange={handleCollapseChange}
        className="kb-refs-panel overflow-hidden rounded-[14px] border border-[var(--border)] bg-[var(--panel)]"
        items={[
          {
            key: 'refs',
            label: (
              <span className="kb-refs-panel-title">
                <span>{S.refs}</span>
                {visibleHits.length > 0 ? <span className="kb-refs-panel-count">{visibleHits.length}</span> : null}
              </span>
            ),
            children: hasPending && visibleHits.length === 0 ? (
              <div className="rounded-[14px] border border-[var(--border)]/70 bg-[var(--panel-2)] px-4 py-3 text-[13px] text-[var(--muted-text)]">
                {S.refs_pending_filter}
              </div>
            ) : shouldShowGuideFilterNote ? (
              <div
                className="rounded-[14px] border border-[var(--border)]/70 bg-[var(--panel-2)] px-4 py-3 text-[13px] text-[var(--muted-text)]"
                data-testid="refs-panel-guide-filter-note"
              >
                {S.refs_guide_filter_note.replace('{count}', filteredSelfCount > 0 ? `（${filteredSelfCount}）` : '')}
              </div>
            ) : shouldShowNegativeSuppressedNote ? (
              <div
                className="rounded-[14px] border border-amber-200/80 bg-amber-50/80 px-4 py-3 text-[13px] text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-100"
                data-testid="refs-panel-negative-suppressed-note"
              >
                <div className="font-medium">{S.refs_suppressed_title}</div>
                <div className="mt-1 text-[13px] opacity-80">
                  {suppressionNoteText}
                </div>
                {suggestionText && (
                  <div className="mt-1.5 text-[12px] italic opacity-60">
                    {suggestionText}
                  </div>
                )}
              </div>
            ) : shouldShowEmptyNote ? (
              <div
                className="rounded-[14px] border border-[var(--border)]/70 bg-[var(--panel-2)] px-4 py-3 text-[13px] text-[var(--muted-text)]"
                data-testid="refs-panel-empty-note"
              >
                {S.refs_empty_note}
                {suggestionText && (
                  <div className="mt-1.5 text-[12px] italic opacity-60">
                    {suggestionText}
                  </div>
                )}
              </div>
            ) : (
              <>
                {hasPending ? (
                  <div
                    className="mb-3 rounded-[14px] border border-[var(--border)]/70 bg-[var(--panel-2)] px-4 py-3 text-[13px] text-[var(--muted-text)]"
                    data-testid="refs-panel-pending-note"
                  >
                    {S.refs_pending_note}
                  </div>
                ) : null}
                <div className="kb-ref-list">
                {visibleHits.map((hit, index) => {
                  const ui = hit.ui_meta || {}
                  const sourceKey = refSourceStateKey(hit, index)
                  const metaState = String(hit.meta?.ref_pack_state || '').trim().toLowerCase()
                  const isFailed = metaState === 'failed'
                  const title = basenameFromSourcePath(ui.display_name || hit.meta?.source_path) || 'Unknown PDF'
                  const heading = ui.heading_path || ui.section_label || ''
                  const scorePending = Boolean(ui.score_pending)
                  const score = typeof ui.score === 'number' ? ui.score.toFixed(2) : ''
                  const cardView = normalizeRefCardView(ui.card_view || ui.cardView)
                  const summarySection = refCardSection(cardView, 'summary')
                  const whySection = refCardSection(cardView, 'why')
                  const summary = selectLocalizedRefCardText({
                    cardText: summarySection?.text,
                    explicitTexts: [cardView?.summary, ui.summary_line],
                    locale: cardCopyLocale,
                  })
                  const summaryKind = String(ui.summary_kind || '').trim().toLowerCase()
                  const summaryRole = String(ui.summary_display_role || '').trim().toLowerCase()
                  const sourceKind = String(ui.source_kind || '').trim().toLowerCase()
                  const isResearchBasket = sourceKind === 'research_basket'
                    || title.toLowerCase().startsWith('research basket:')
                  const isSourceEvidence = summaryRole === 'source_evidence'
                    || summaryRole === 'evidence'
                    || summaryKind === 'evidence'
                    || summaryKind === 'source_evidence'
                  const isGuide = summaryRole === 'guide'
                    || summaryKind === 'guide'
                    || summaryKind === 'section_grounded'
                  const isMetadata = summaryKind === 'metadata'
                  const summaryLabel = isResearchBasket
                    ? S.refs_basket_label
                    : isSourceEvidence
                      ? S.refs_evidence_label
                      : isGuide
                        ? S.refs_guide_label
                        : isMetadata
                          ? S.refs_metadata_label
                          : S.refs_summary_label
                  const summaryTitle = isResearchBasket
                    ? S.refs_basket_title
                    : isSourceEvidence
                      ? S.refs_evidence_title
                      : isGuide
                        ? S.refs_guide_title
                        : isMetadata
                          ? S.refs_metadata_title
                          : S.refs_summary_title
                  const why = selectRefRelevanceText({
                    cardText: whySection?.text,
                    explicitTexts: [
                      ui.card_support_explanation,
                      ui.user_question_relation,
                      ui.support_relation,
                      ui.why_relevant,
                      ui.why_line,
                    ],
                    evidenceTexts: [summarySection?.text, cardView?.summary, ui.summary_line, hit.text],
                    locale: cardCopyLocale,
                  })
                  const whyLabel = S.refs_why_chip
                  const whyTitle = S.refs_why_title
                  const polishStatus = normalizePolishStatus(ui.polish_status)
                  const polishLabel = polishStatusLabel(polishStatus, S)
                  const semanticBadges = (Array.isArray(ui.semantic_badges) ? ui.semantic_badges : [])
                    .map((badge) => ({
                      text: String(badge?.text || '').trim(),
                      score: positiveNumber(badge?.score),
                    }))
                    .filter((badge) => shouldShowSemanticBadge(badge.text))
                    .slice(0, 1)
                  const detail = buildCiteDetailFromMeta(
                    (remoteMeta[sourceKey] || ui.citation_meta || {}) as Record<string, unknown>,
                    {
                      sourceName: title,
                      sourcePath: ui.source_path,
                      num: index + 1,
                      anchor: `ref-source-${msgId}-${index}`,
                    },
                  )
                  const metrics = detail ? citeMetricSummary(detail) : []
                  const doi = String(detail?.doi || '').trim()
                  const doiUrl = String(detail?.doiUrl || '').trim()
                  const pageText = ui.page_start
                    ? `P.${ui.page_start}${ui.page_end && ui.page_end !== ui.page_start ? `-${ui.page_end}` : ''}`
                    : ''
                  const canFetchMeta = Boolean(String(ui.source_path || '').trim())

                  return (
                    <div key={`${msgId}-${sourceKey}`} className="kb-ref-item">
                      <div className="kb-ref-header">
                        <div className="kb-ref-rank">#{index + 1}</div>
                        <div className="kb-ref-main">
                          <div className="kb-ref-title-row">
                            <div className="min-w-0 flex-1">
                              <div className="kb-ref-title">{title}</div>
                              <div className="kb-ref-meta-row mt-1">
                                {heading ? <span>{heading}</span> : null}
                                {showInternalRefDiagnostics && scorePending ? <span className="kb-ref-score">{S.refs_score_pending}</span> : null}
                                {showInternalRefDiagnostics && !scorePending && score ? <span className="kb-ref-score">{S.refs_score_label.replace('{score}', score)}</span> : null}
                                {showInternalRefDiagnostics && polishStatus && polishLabel ? (
                                  <span
                                    className={`kb-ref-polish is-${polishStatus}`}
                                    data-testid={`refs-panel-polish-status-${index}`}
                                    data-status={polishStatus}
                                    title={String(ui.polish_detail || '')}
                                  >
                                    {polishLabel}
                                  </span>
                                ) : null}
                                {semanticBadges.map((badge, badgeIndex) => {
                                  const text = String(badge.text || '').trim()
                                  if (!text) return null
                                  return (
                                    <span className="kb-ref-semantic" key={`semantic-${msgId}-${index}-${badgeIndex}`}>
                                      {text}
                                    </span>
                                  )
                                })}
                                {pageText ? <span>{pageText}</span> : null}
                              </div>
                            </div>
                            <div className="kb-ref-actions">
                              <Button
                                className="kb-ref-action is-primary"
                                disabled={!canFetchMeta || !onOpenReader}
                                onClick={() => openReaderFromHit(ui)}
                              >
                                {S.refs_locate_btn}
                              </Button>
                              <Button
                                className="kb-ref-action"
                                disabled={!ui.can_open || !ui.source_path}
                                onClick={async () => {
                                  if (!ui.source_path) return
                                  await referencesApi.open(ui.source_path, ui.page_start ?? null)
                                    .then(() => message.success(S.refs_open_pdf_success))
                                    .catch((err: Error) => message.error(err.message || S.refs_open_pdf_failed))
                                }}
                              >
                                {S.refs_pdf_btn}
                              </Button>
                              <Button
                                className="kb-ref-action"
                                loading={loadingSourceKey === sourceKey}
                                disabled={!canFetchMeta}
                                onClick={async () => {
                                  setCiteSourceKey(sourceKey)
                                  const existingMeta = (remoteMeta[sourceKey] || ui.citation_meta || {}) as Record<string, unknown>
                                  if (!hasResolvedCitationMeta(existingMeta)) {
                                    await fetchCitationMeta(sourceKey, ui)
                                  }
                                }}
                              >
                                {S.refs_cite_btn}
                              </Button>
                              <Button
                                className="kb-ref-action"
                                loading={guideLoadingSourceKey === sourceKey}
                                disabled={!canFetchMeta}
                                onClick={() => { void startPaperGuideFromHit(sourceKey, ui) }}
                              >
                                {S.refs_guide_btn}
                              </Button>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className={`kb-ref-evidence-grid${why ? '' : ' is-single'}`}>
                        <div className="kb-ref-card">
                          <div className="kb-ref-card-head">
                            <span className="kb-ref-chip">{summaryLabel}</span>
                            <span className="kb-ref-card-title">{summaryTitle}</span>
                          </div>
                          <Text className="kb-ref-card-text !block !whitespace-pre-wrap">
                            {summary || (isFailed ? S.refs_no_summary_failed : S.refs_no_summary)}
                          </Text>
                        </div>
                        {why ? (
                          <div className="kb-ref-card">
                            <div className="kb-ref-card-head">
                              <span className="kb-ref-chip">{whyLabel}</span>
                              <span className="kb-ref-card-title">{whyTitle}</span>
                            </div>
                            <Text className="kb-ref-card-text !block !whitespace-pre-wrap">{why}</Text>
                          </div>
                        ) : null}
                      </div>

                      {metrics.length > 0 || doiUrl ? (
                        <div className="kb-ref-metrics" data-testid={`refs-panel-metrics-${index}`}>
                          {metrics.map((item, idx) => (
                            <span key={item}>
                              {idx > 0 ? ' | ' : ''}
                              {item}
                            </span>
                          ))}
                          {doiUrl ? (
                            <>
                              {metrics.length > 0 ? ' | ' : ''}
                              DOI{' '}
                              <Link href={doiUrl} target="_blank">
                                {doi || doiUrl}
                              </Link>
                            </>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  )
                })}
                </div>
              </>
            ),
          },
        ]}
      />

      <Modal
        open={citeSourceKey !== null && citeDetail !== null}
        title={null}
        footer={null}
        onCancel={() => setCiteSourceKey(null)}
        width={640}
        className="kb-ref-cite-modal"
      >
        {loadingSourceKey === citeSourceKey ? (
          <div className="py-8 text-center text-sm text-neutral-500">{S.refs_cite_loading}</div>
        ) : citeDetail ? (
          <>
            <div className="kb-ref-cite-head">
              <div className="kb-ref-cite-label">{S.refs_cite_label}</div>
              <div className="kb-ref-cite-main">{citationDisplay(citeDetail).main}</div>
              {citationDisplay(citeDetail).authors ? (
                <div className="kb-ref-cite-sub">{citationDisplay(citeDetail).authors}</div>
              ) : null}
              {citationDisplay(citeDetail).source ? (
                <div className="kb-ref-cite-sub">source: {citationDisplay(citeDetail).source}</div>
              ) : null}
            </div>

            {citeMetricSummary(citeDetail).length > 0 ? (
              <div className="kb-cite-pop-metrics mb-4">
                {citeMetricSummary(citeDetail).map((item) => (
                  <span key={item} className="kb-cite-pop-metric">
                    {item}
                  </span>
                ))}
              </div>
            ) : null}

            {citeDetail.doiUrl ? (
              <div className="kb-ref-cite-doi">
                DOI:{' '}
                <Link href={citeDetail.doiUrl} target="_blank">
                  {citeDetail.doi || citeDetail.doiUrl}
                </Link>
              </div>
            ) : null}

            <Tabs
              items={[
                {
                  key: 'gbt',
                  label: S.refs_cite_gbt,
                  children: <pre className="kb-ref-cite-pre">{citationFormats(citeDetail).gbt || S.refs_cite_none_gbt}</pre>,
                },
                {
                  key: 'bib',
                  label: S.refs_cite_bib,
                  children: <pre className="kb-ref-cite-pre">{citationFormats(citeDetail).bibtex || S.refs_cite_none_bib}</pre>,
                },
              ]}
            />
          </>
        ) : (
          <div className="py-8 text-center text-sm text-neutral-500">{S.refs_cite_none}</div>
        )}
      </Modal>
    </>
  )
}
