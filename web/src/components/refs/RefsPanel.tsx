import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, Collapse, Modal, Tabs, Typography, message } from 'antd'
import { useNavigate } from 'react-router-dom'
import { useT } from '../../i18n'
import { referencesApi } from '../../api/references'
import { useChatStore } from '../../stores/chatStore'
import type { ReaderOpenPayload } from '../chat/reader/readerTypes'
import { buildBasicReaderOpenPayload } from '../chat/reader/readerOpenPayloadUtils'
import {
  buildCiteDetailFromMeta,
  citationDisplay,
  citationFormats,
  citeMetricSummary,
  type CiteDetail,
} from '../chat/citationState'

const { Link, Text } = Typography
const expandedRefsPanelKeys = new Set<string>()


function refsPanelExpansionKey(msgId: number) {
  return String(Number(msgId || 0) || 0)
}

interface RefUiMeta {
  display_name?: string
  heading_path?: string
  section_label?: string
  subsection_label?: string
  page_start?: number
  page_end?: number
  score?: number | null
  score_pending?: boolean
  summary_line?: string
  summary_kind?: string
  summary_label?: string
  summary_title?: string
  summary_generation?: string
  summary_basis?: string
  polish_status?: string
  polish_source?: string
  polish_detail?: string
  summary_polish_status?: string
  why_polish_status?: string
  why_line?: string
  why_generation?: string
  why_basis?: string
  semantic_badges?: Array<{
    text?: string
    score?: number
  }>
  can_open?: boolean
  citation_meta?: Record<string, unknown>
  source_path?: string
  reader_open?: Partial<ReaderOpenPayload>
}

interface RefHit {
  text?: string
  meta?: {
    source_path?: string
    ref_pack_state?: string
  }
  ui_meta?: RefUiMeta
}

interface RefEntry {
  prompt?: string
  hits?: RefHit[]
  display_state?: string
  suppression_reason?: string
  suggestion?: string
  guide_filter?: {
    active?: boolean
    hidden_self_source?: boolean
    filtered_hit_count?: number
    guide_source_name?: string
  }
}

interface Props {
  refs: Record<string, unknown>
  msgId: number
  onOpenReader?: (payload: ReaderOpenPayload) => void
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

function normalizeRefFocusText(input: unknown) {
  return String(input || '')
    .toLowerCase()
    .replace(/\.en\.md$/g, ' ')
    .replace(/\.md$|\.pdf$/g, ' ')
    .replace(/[_/\\]+/g, ' ')
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function promptFocusTerms(prompt: string) {
  const text = String(prompt || '').trim()
  if (!text) return [] as string[]
  const out: string[] = []
  const seen = new Set<string>()
  const push = (raw: string) => {
    const norm = normalizeRefFocusText(raw)
    if (!norm || norm.length < 3 || seen.has(norm)) return
    seen.add(norm)
    out.push(norm)
  }
  for (const m of text.matchAll(/[“"'‘’]([^“"'‘’]{2,80})[“"'‘’]/g)) {
    push(String(m[1] || ''))
  }
  const stop = new Set([
    'the', 'and', 'for', 'with', 'from', 'into', 'using', 'about', 'where', 'which', 'what',
    'that', 'this', 'these', 'those', 'paper', 'papers', 'library', 'source', 'sources',
    'section', 'please', 'point', 'directly', 'most', 'does', 'do', 'did', 'discuss', 'discusses',
    'mentioned', 'mention', 'other', 'besides', 'find', 'show', 'explain',
  ])
  for (const m of text.matchAll(/\b[A-Za-z][A-Za-z0-9_-]{1,40}\b/g)) {
    const raw = String(m[0] || '').trim()
    const low = raw.toLowerCase()
    if (stop.has(low)) continue
    const hasSignal = /[A-Z]/.test(raw.slice(1)) || raw === raw.toUpperCase() || /\d/.test(raw) || raw.includes('-')
    if (!hasSignal) continue
    push(raw)
  }
  return out.slice(0, 8)
}

function promptNeedsStrictRefEvidence(prompt: string) {
  const low = String(prompt || '').toLowerCase()
  if (!low) return false
  const patterns = [
    'where is', 'where was', 'where are', 'discuss', 'mention', 'point me',
    'which paper', 'which papers', 'what other papers', 'besides this paper',
    '哪篇', '哪些论文', '提到', '哪里', '定位',
  ]
  return patterns.some((pattern) => low.includes(pattern))
}

function hitIdentityTerms(hit: RefHit) {
  const ui = hit.ui_meta || {}
  const meta = hit.meta || {}
  const values = [
    String(ui.display_name || ''),
    String(ui.source_path || ''),
    String(meta.source_path || ''),
  ]
  const out = new Set<string>()
  for (const raw of values) {
    const norm = normalizeRefFocusText(raw)
    if (!norm) continue
    out.add(norm)
    for (const token of norm.split(' ')) {
      if (token.length >= 3) out.add(token)
    }
  }
  return out
}

function hitSurfaceText(hit: RefHit) {
  const ui = hit.ui_meta || {}
  const readerOpen = (ui.reader_open && typeof ui.reader_open === 'object') ? ui.reader_open : {}
  const parts = [
    String(hit.text || ''),
    String(ui.heading_path || ''),
    String(ui.summary_line || ''),
    String(readerOpen.snippet || ''),
    String(readerOpen.highlightSnippet || ''),
  ]
  return normalizeRefFocusText(parts.filter(Boolean).join(' '))
}

function nonSourceFocusMatchCount(prompt: string, hit: RefHit) {
  const focusTerms = promptFocusTerms(prompt)
  if (!focusTerms.length) return 0
  const surface = hitSurfaceText(hit)
  if (!surface) return 0
  const identities = hitIdentityTerms(hit)
  let count = 0
  for (const term of focusTerms) {
    if (!surface.includes(term)) continue
    const isIdentity = Array.from(identities).some((ident) => term === ident || term.includes(ident) || ident.includes(term))
    if (!isIdentity) count += 1
  }
  return count
}

function looksNegativeReasonText(text: string) {
  const low = String(text || '').toLowerCase()
  if (!low) return false
  return [
    'not mentioned',
    'not discuss',
    'not discussed',
    'not stated',
    'no external paper matched',
    'no papers in your library',
    'cannot point',
    '未提及',
    '未提到',
    '没有提到',
    '没有命中',
    '无法定位',
    '不能指向',
  ].some((token) => low.includes(token))
}

function normalizePolishStatus(input: unknown) {
  const status = String(input || '').trim().toLowerCase()
  if (status === 'full' || status === 'heuristic' || status === 'pending' || status === 'failed') return status
  return ''
}

function polishStatusLabel(status: string, S: ReturnType<typeof useT>) {
  if (status === 'full') return S.refs_polish_full
  if (status === 'pending') return S.refs_polish_pending
  if (status === 'failed') return S.refs_polish_failed
  if (status === 'heuristic') return S.refs_polish_heuristic
  return ''
}

function shouldSuppressRefHitCard(prompt: string, hit: RefHit) {
  if (!promptNeedsStrictRefEvidence(prompt)) return false
  const ui = hit.ui_meta || {}
  const why = String(ui.why_line || '').trim()
  const summary = String(ui.summary_line || '').trim()
  const focusTerms = promptFocusTerms(prompt)
  const nonSourceMatches = nonSourceFocusMatchCount(prompt, hit)
  if (focusTerms.length > 1 && nonSourceMatches <= 0) {
    return true
  }
  if (looksNegativeReasonText(why) && nonSourceMatches <= 0) {
    return true
  }
  if (looksNegativeReasonText(summary) && nonSourceMatches <= 0) {
    return true
  }
  return false
}





export function RefsPanel({ refs, msgId, onOpenReader }: Props) {
  const S = useT()
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const nav = useNavigate()
  const expansionKey = refsPanelExpansionKey(msgId)
  const [activeKeys, setActiveKeys] = useState<string[]>(() => (
    expandedRefsPanelKeys.has(expansionKey) ? ['refs'] : []
  ))
  const entry = refs[String(msgId)] as RefEntry | undefined
  const prompt = String(entry?.prompt || '').trim()
  const displayState = String(entry?.display_state || '').trim().toLowerCase()
  const suppressionReason = String(entry?.suppression_reason || '').trim().toLowerCase()
  const hasBackendDisplayState = Boolean(displayState)
  const suggestionText = String(entry?.suggestion || '').trim()
  const rawHits = entry?.hits
  const hits = useMemo(() => (Array.isArray(rawHits) ? rawHits : []), [rawHits])
  const visibleHits = useMemo(
    () => (hasBackendDisplayState ? hits : hits.filter((hit) => !shouldSuppressRefHitCard(prompt, hit))),
    [hasBackendDisplayState, hits, prompt],
  )
  const suppressedHitCount = Math.max(0, hits.length - visibleHits.length)
  const guideFilter = entry?.guide_filter || {}
  const pendingCount = visibleHits.filter((hit) => String(hit?.meta?.ref_pack_state || '').trim().toLowerCase() === 'pending').length
  const hasPending = displayState === 'pending' || pendingCount > 0
  const filteredSelfCount = positiveNumber(guideFilter.filtered_hit_count)
  const shouldShowGuideFilterNote = !hasPending && (
    displayState === 'hidden_by_guide'
    || ((!hasBackendDisplayState) && hits.length === 0 && Boolean(guideFilter.hidden_self_source))
  )
  const shouldShowNegativeSuppressedNote = !hasPending && (
    displayState === 'suppressed'
    || ((!hasBackendDisplayState) && visibleHits.length === 0 && suppressedHitCount > 0)
  )
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
  const [citeIndex, setCiteIndex] = useState<number | null>(null)
  const [loadingIndex, setLoadingIndex] = useState<number | null>(null)
  const [guideLoadingIndex, setGuideLoadingIndex] = useState<number | null>(null)
  const [remoteMeta, setRemoteMeta] = useState<Record<number, Record<string, unknown>>>({})
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

  const fetchCitationMeta = async (index: number, ui: RefUiMeta, options?: { silent?: boolean }) => {
    const sourcePath = String(ui.source_path || '').trim()
    if (!sourcePath) return
    const silent = Boolean(options?.silent)
    if (!silent) {
      setLoadingIndex(index)
    }
    try {
      const meta = await referencesApi.citationMetaCached(sourcePath)
      setRemoteMeta((current) => ({ ...current, [index]: meta }))
    } catch (err) {
      if (!silent) {
        message.error(err instanceof Error ? err.message : S.refs_fetch_meta_failed)
      }
    } finally {
      if (!silent) {
        setLoadingIndex((current) => (current === index ? null : current))
      }
    }
  }

  useEffect(() => {
    if (hasPending || visibleHits.length <= 0) return
    for (const [index, hit] of visibleHits.entries()) {
      const ui = hit.ui_meta || {}
      const sourcePath = String(ui.source_path || '').trim()
      if (!sourcePath) continue
      const existingMeta = (remoteMeta[index] || ui.citation_meta || {}) as Record<string, unknown>
      if (hasResolvedCitationMeta(existingMeta)) continue
      const fetchKey = `${msgId}:${index}:${sourcePath}`
      if (autoFetchedCitationMetaRef.current.has(fetchKey)) continue
      autoFetchedCitationMetaRef.current.add(fetchKey)
      void fetchCitationMeta(index, ui, { silent: true })
    }
  }, [hasPending, msgId, remoteMeta, visibleHits])

  const citeDetail = useMemo<CiteDetail | null>(() => {
    if (citeIndex === null || !visibleHits[citeIndex]) return null
    const ui = visibleHits[citeIndex]?.ui_meta || {}
    const meta = remoteMeta[citeIndex] || ui.citation_meta
    return buildCiteDetailFromMeta(meta as Record<string, unknown>, {
      sourceName: ui.display_name,
      sourcePath: ui.source_path,
      num: citeIndex + 1,
      anchor: `ref-source-${msgId}-${citeIndex}`,
    })
  }, [citeIndex, visibleHits, msgId, remoteMeta])

  const startPaperGuideFromHit = async (index: number, ui: RefUiMeta) => {
    const sourcePath = String(ui.source_path || '').trim()
    if (!sourcePath) {
      message.info(S.refs_reader_missing)
      return
    }
    const sourceName = String(ui.display_name || '').trim() || sourcePath.split(/[\\/]/).pop() || 'Paper'
    setGuideLoadingIndex(index)
    try {
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: `阅读指导 · ${sourceName}`,
      })
      nav('/')
      message.success(S.refs_guide_started)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.refs_guide_failed)
    } finally {
      setGuideLoadingIndex((current) => (current === index ? null : current))
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
      sourceName: String(readerOpen.sourceName || ui.display_name || '').trim(),
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
      fallbackSourceName: '文献',
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
            label: <span className="kb-refs-panel-title">{S.refs}</span>,
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
                  const metaState = String(hit.meta?.ref_pack_state || '').trim().toLowerCase()
                  const isFailed = metaState === 'failed'
                  const title = ui.display_name || hit.meta?.source_path?.split('\\').pop() || 'Unknown PDF'
                  const heading = ui.heading_path || ui.section_label || ''
                  const scorePending = Boolean(ui.score_pending)
                  const score = typeof ui.score === 'number' ? ui.score.toFixed(2) : ''
                  const summary = String(ui.summary_line || '').trim()
                  const summaryLabel = String(ui.summary_label || '').trim() || S.refs_summary_label
                  const summaryTitle = String(ui.summary_title || '').trim() || S.refs_summary_title
                  const why = String(ui.why_line || '').trim()
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
                    (remoteMeta[index] || ui.citation_meta || {}) as Record<string, unknown>,
                    {
                      sourceName: ui.display_name,
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
                    <div key={`${msgId}-${index}`} className="kb-ref-item">
                      <div className="kb-ref-header">
                        <div className="kb-ref-rank">#{index + 1}</div>
                        <div className="kb-ref-main">
                          <div className="kb-ref-title-row">
                            <div className="min-w-0 flex-1">
                              <div className="kb-ref-title">{title}</div>
                              <div className="kb-ref-meta-row mt-1">
                                {heading ? <span>{heading}</span> : null}
                                {scorePending ? <span className="kb-ref-score">{S.refs_score_pending}</span> : null}
                                {!scorePending && score ? <span className="kb-ref-score">{S.refs_score_label.replace('{score}', score)}</span> : null}
                                {polishStatus && polishLabel ? (
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
                                loading={loadingIndex === index}
                                disabled={!canFetchMeta}
                                onClick={async () => {
                                  setCiteIndex(index)
                                  const existingMeta = (remoteMeta[index] || ui.citation_meta || {}) as Record<string, unknown>
                                  if (!hasResolvedCitationMeta(existingMeta)) {
                                    await fetchCitationMeta(index, ui)
                                  }
                                }}
                              >
                                {S.refs_cite_btn}
                              </Button>
                              <Button
                                className="kb-ref-action"
                                loading={guideLoadingIndex === index}
                                disabled={!canFetchMeta}
                                onClick={() => { void startPaperGuideFromHit(index, ui) }}
                              >
                                {S.refs_guide_btn}
                              </Button>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="kb-ref-evidence-grid">
                        <div className="kb-ref-card">
                          <div className="kb-ref-card-head">
                            <span className="kb-ref-chip">{summaryLabel}</span>
                            <span className="kb-ref-card-title">{summaryTitle}</span>
                          </div>
                          <Text className="kb-ref-card-text !block !whitespace-pre-wrap">
                            {summary || (isFailed ? S.refs_no_summary_failed : S.refs_no_summary)}
                          </Text>
                        </div>
                        <div className="kb-ref-card">
                          <div className="kb-ref-card-head">
                            <span className="kb-ref-chip">{S.refs_why_chip}</span>
                            <span className="kb-ref-card-title">{S.refs_why_title}</span>
                          </div>
                          <Text className="kb-ref-card-text !block !whitespace-pre-wrap">
                            {why || (isFailed ? S.refs_no_why_failed : S.refs_no_why)}
                          </Text>
                        </div>
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
        open={citeIndex !== null}
        title={null}
        footer={null}
        onCancel={() => setCiteIndex(null)}
        width={640}
        className="kb-ref-cite-modal"
      >
        {loadingIndex === citeIndex ? (
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
