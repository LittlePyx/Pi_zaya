import { useMemo, useState } from 'react'
import { Button, Collapse, Modal, Tabs, Typography, message } from 'antd'
import { useNavigate } from 'react-router-dom'
import { S } from '../../i18n/zh'
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
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const nav = useNavigate()
  const entry = refs[String(msgId)] as RefEntry | undefined
  const prompt = String(entry?.prompt || '').trim()
  const rawHits = entry?.hits
  const hits = useMemo(() => (Array.isArray(rawHits) ? rawHits : []), [rawHits])
  const visibleHits = useMemo(
    () => hits.filter((hit) => !shouldSuppressRefHitCard(prompt, hit)),
    [hits, prompt],
  )
  const suppressedHitCount = Math.max(0, hits.length - visibleHits.length)
  const guideFilter = entry?.guide_filter || {}
  const pendingCount = visibleHits.filter((hit) => String(hit?.meta?.ref_pack_state || '').trim().toLowerCase() === 'pending').length
  const hasPending = pendingCount > 0
  const filteredSelfCount = positiveNumber(guideFilter.filtered_hit_count)
  const shouldShowGuideFilterNote = !hasPending && hits.length === 0 && Boolean(guideFilter.hidden_self_source)
  const shouldShowNegativeSuppressedNote = !hasPending && visibleHits.length === 0 && suppressedHitCount > 0
  const [citeIndex, setCiteIndex] = useState<number | null>(null)
  const [loadingIndex, setLoadingIndex] = useState<number | null>(null)
  const [guideLoadingIndex, setGuideLoadingIndex] = useState<number | null>(null)
  const [remoteMeta, setRemoteMeta] = useState<Record<number, Record<string, unknown>>>({})

  const fetchCitationMeta = async (index: number, ui: RefUiMeta) => {
    const sourcePath = String(ui.source_path || '').trim()
    if (!sourcePath) return
    setLoadingIndex(index)
    try {
      const meta = await referencesApi.citationMetaCached(sourcePath)
      setRemoteMeta((current) => ({ ...current, [index]: meta }))
    } catch (err) {
      message.error(err instanceof Error ? err.message : '拉取文献信息失败')
    } finally {
      setLoadingIndex((current) => (current === index ? null : current))
    }
  }

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
      message.info('当前引用缺少可绑定的文献路径')
      return
    }
    const sourceName = String(ui.display_name || '').trim() || sourcePath.split(/[\\/]/).pop() || '文献'
    setGuideLoadingIndex(index)
    try {
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: `阅读指导 · ${sourceName}`,
      })
      nav('/')
      message.success('已进入阅读指导会话')
    } catch (err) {
      message.error(err instanceof Error ? err.message : '创建阅读指导会话失败')
    } finally {
      setGuideLoadingIndex((current) => (current === index ? null : current))
    }
  }

  const openReaderFromHit = (ui: RefUiMeta) => {
    if (!onOpenReader) return
    const readerOpen = (ui.reader_open && typeof ui.reader_open === 'object') ? ui.reader_open : {}
    const sourcePath = String(readerOpen.sourcePath || ui.source_path || '').trim()
    if (!sourcePath) {
      message.info('当前引用缺少可绑定的文献路径')
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

  if (!entry || (!hasPending && visibleHits.length === 0 && !shouldShowGuideFilterNote && !shouldShowNegativeSuppressedNote)) return null

  return (
    <>
      <Collapse
        size="large"
        className="kb-refs-panel overflow-hidden rounded-[24px] border border-[var(--border)] bg-[var(--panel)]"
        items={[
          {
            key: 'refs',
            label: <span className="kb-refs-panel-title">{S.refs}</span>,
            children: hasPending ? (
              <div className="rounded-[18px] border border-[var(--border)]/70 bg-[var(--panel-2)] px-5 py-4 text-sm text-[var(--muted)]">
                正在筛选高相关参考文献，并生成摘要与相关性说明...
              </div>
            ) : shouldShowGuideFilterNote ? (
              <div
                className="rounded-[18px] border border-[var(--border)]/70 bg-[var(--panel-2)] px-5 py-4 text-sm text-[var(--muted)]"
                data-testid="refs-panel-guide-filter-note"
              >
                {`已过滤当前阅读指导文献${filteredSelfCount > 0 ? `（${filteredSelfCount} 条）` : ''}，但这次没有命中其它库内文章。`}
              </div>
            ) : shouldShowNegativeSuppressedNote ? (
              <div
                className="rounded-[18px] border border-amber-200/80 bg-amber-50/80 px-5 py-4 text-sm text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-100"
                data-testid="refs-panel-negative-suppressed-note"
              >
                <div className="font-medium">已隐藏可能误导的参考定位卡片。</div>
                <div className="mt-1 text-[13px] opacity-80">
                  当前没有足够可靠、能直接支撑这个问题的定位切口可供打开。
                </div>
              </div>
            ) : (
              <div className="space-y-5">
                {visibleHits.map((hit, index) => {
                  const ui = hit.ui_meta || {}
                  const metaState = String(hit.meta?.ref_pack_state || '').trim().toLowerCase()
                  const isFailed = metaState === 'failed'
                  const title = ui.display_name || hit.meta?.source_path?.split('\\').pop() || 'Unknown PDF'
                  const heading = ui.heading_path || ui.section_label || ''
                  const scorePending = Boolean(ui.score_pending)
                  const score = typeof ui.score === 'number' ? ui.score.toFixed(2) : ''
                  const summary = String(ui.summary_line || '').trim()
                  const summaryLabel = String(ui.summary_label || '').trim() || '摘要'
                  const summaryTitle = String(ui.summary_title || '').trim() || '这篇文献讲什么 / 提供什么'
                  const summaryBasis = String(ui.summary_basis || '').trim()
                  const why = String(ui.why_line || '').trim()
                  const whyBasis = String(ui.why_basis || '').trim()
                  const semanticBadges = Array.isArray(ui.semantic_badges) ? ui.semantic_badges : []
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
                    <div key={`${msgId}-${index}`} className="space-y-4 border-b border-[var(--border)]/60 pb-5 last:border-b-0 last:pb-0">
                      <div className="flex items-start gap-4">
                        <div className="kb-ref-rank">#{index + 1}</div>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-start gap-4">
                            <div className="min-w-0 flex-1">
                              <div className="kb-ref-title">{title}</div>
                              <div className="kb-ref-meta-row mt-1.5">
                                {heading ? <span>{heading}</span> : null}
                                {scorePending ? <span className="kb-ref-score">相关分评估中</span> : null}
                                {!scorePending && score ? <span className="kb-ref-score">相关分 {score}</span> : null}
                                {semanticBadges.map((badge, badgeIndex) => {
                                  const text = String(badge?.text || '').trim()
                                  if (!text) return null
                                  const scoreText = positiveNumber(badge?.score)
                                  return (
                                    <span className="kb-ref-semantic" key={`semantic-${msgId}-${index}-${badgeIndex}`}>
                                      {text}
                                      {scoreText > 0 ? ` (${scoreText.toFixed(1)})` : ''}
                                    </span>
                                  )
                                })}
                                {pageText ? <span>{pageText}</span> : null}
                              </div>
                            </div>
                            <div className="flex shrink-0 gap-2">
                              <Button
                                className="kb-ref-action"
                                disabled={!canFetchMeta || !onOpenReader}
                                onClick={() => openReaderFromHit(ui)}
                              >
                                定位
                              </Button>
                              <Button
                                className="kb-ref-action"
                                disabled={!ui.can_open || !ui.source_path}
                                onClick={async () => {
                                  if (!ui.source_path) return
                                  await referencesApi.open(ui.source_path, ui.page_start ?? null)
                                    .then(() => message.success('已打开 PDF'))
                                    .catch((err: Error) => message.error(err.message || '打开失败'))
                                }}
                              >
                                PDF
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
                                Cite
                              </Button>
                              <Button
                                className="kb-ref-action"
                                loading={guideLoadingIndex === index}
                                disabled={!canFetchMeta}
                                onClick={() => { void startPaperGuideFromHit(index, ui) }}
                              >
                                阅读
                              </Button>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="grid gap-3 md:grid-cols-2">
                        <div className="kb-ref-card">
                          <div className="mb-2 flex items-center gap-2">
                            <span className="kb-ref-chip">{summaryLabel}</span>
                            <span className="kb-ref-card-title">{summaryTitle}</span>
                          </div>
                          {summaryBasis ? (
                            <div className="mb-2 text-[12px] text-[var(--muted)]" data-testid="refs-panel-summary-basis">
                              {summaryBasis}
                            </div>
                          ) : null}
                          <Text className="kb-ref-card-text !block !whitespace-pre-wrap">
                            {summary || (isFailed ? '暂未生成摘要定位' : '未提供摘要定位')}
                          </Text>
                        </div>
                        <div className="kb-ref-card">
                          <div className="mb-2 flex items-center gap-2">
                            <span className="kb-ref-chip">相关</span>
                            <span className="kb-ref-card-title">为什么与当前问题相关</span>
                          </div>
                          {whyBasis ? (
                            <div className="mb-2 text-[12px] text-[var(--muted)]" data-testid="refs-panel-why-basis">
                              {whyBasis}
                            </div>
                          ) : null}
                          <Text className="kb-ref-card-text !block !whitespace-pre-wrap">
                            {why || (isFailed ? '暂未生成相关性说明' : '未提供相关性说明')}
                          </Text>
                        </div>
                      </div>

                      {metrics.length > 0 || doiUrl ? (
                        <div className="kb-ref-metrics">
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
          <div className="py-8 text-center text-sm text-neutral-500">正在拉取文献信息...</div>
        ) : citeDetail ? (
          <>
            <div className="kb-ref-cite-head">
              <div className="kb-ref-cite-label">来源引用</div>
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
                  label: 'GB/T 7714',
                  children: <pre className="kb-ref-cite-pre">{citationFormats(citeDetail).gbt || '暂无可用引用'}</pre>,
                },
                {
                  key: 'bib',
                  label: 'BibTeX',
                  children: <pre className="kb-ref-cite-pre">{citationFormats(citeDetail).bibtex || '暂无可用引用'}</pre>,
                },
              ]}
            />
          </>
        ) : (
          <div className="py-8 text-center text-sm text-neutral-500">暂无可用引用</div>
        )}
      </Modal>
    </>
  )
}
