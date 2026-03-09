import { useMemo, useState } from 'react'
import { Button, Collapse, Modal, Tabs, Typography, message } from 'antd'
import { useNavigate } from 'react-router-dom'
import { S } from '../../i18n/zh'
import { referencesApi } from '../../api/references'
import { useChatStore } from '../../stores/chatStore'
import type { ReaderOpenPayload } from '../chat/PaperGuideReaderDrawer'
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
  why_line?: string
  semantic_badges?: Array<{
    text?: string
    score?: number
  }>
  can_open?: boolean
  citation_meta?: Record<string, unknown>
  source_path?: string
}

interface RefHit {
  meta?: {
    source_path?: string
    ref_pack_state?: string
  }
  ui_meta?: RefUiMeta
}

interface RefEntry {
  hits?: RefHit[]
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

export function RefsPanel({ refs, msgId, onOpenReader }: Props) {
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const nav = useNavigate()
  const entry = refs[String(msgId)] as RefEntry | undefined
  const hits = Array.isArray(entry?.hits) ? entry.hits : []
  const pendingCount = hits.filter((hit) => String(hit?.meta?.ref_pack_state || '').trim().toLowerCase() === 'pending').length
  const hasPending = pendingCount > 0
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
    if (citeIndex === null || !hits[citeIndex]) return null
    const ui = hits[citeIndex]?.ui_meta || {}
    const meta = remoteMeta[citeIndex] || ui.citation_meta
    return buildCiteDetailFromMeta(meta as Record<string, unknown>, {
      sourceName: ui.display_name,
      sourcePath: ui.source_path,
      num: citeIndex + 1,
      anchor: `ref-source-${msgId}-${citeIndex}`,
    })
  }, [citeIndex, hits, msgId, remoteMeta])

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
    const sourcePath = String(ui.source_path || '').trim()
    if (!sourcePath) {
      message.info('当前引用缺少可绑定的文献路径')
      return
    }
    const sourceName = String(ui.display_name || '').trim() || sourcePath.split(/[\\/]/).pop() || '文献'
    const headingPath = String(ui.heading_path || ui.section_label || ui.subsection_label || '').trim()
    const snippet = String(ui.summary_line || ui.why_line || '').trim()
    onOpenReader({
      sourcePath,
      sourceName,
      headingPath,
      snippet,
    })
  }

  if (!entry || (!hasPending && hits.length === 0)) return null

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
            ) : (
              <div className="space-y-5">
                {hits.map((hit, index) => {
                  const ui = hit.ui_meta || {}
                  const metaState = String(hit.meta?.ref_pack_state || '').trim().toLowerCase()
                  const isFailed = metaState === 'failed'
                  const title = ui.display_name || hit.meta?.source_path?.split('\\').pop() || 'Unknown PDF'
                  const heading = ui.heading_path || ui.section_label || ''
                  const scorePending = Boolean(ui.score_pending)
                  const score = typeof ui.score === 'number' ? ui.score.toFixed(2) : ''
                  const summary = String(ui.summary_line || '').trim()
                  const why = String(ui.why_line || '').trim()
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
                            <span className="kb-ref-chip">摘要</span>
                            <span className="kb-ref-card-title">这篇文献讲什么 / 提供什么</span>
                          </div>
                          <Text className="kb-ref-card-text !block !whitespace-pre-wrap">
                            {summary || (isFailed ? '暂未生成摘要定位' : '未提供摘要定位')}
                          </Text>
                        </div>
                        <div className="kb-ref-card">
                          <div className="mb-2 flex items-center gap-2">
                            <span className="kb-ref-chip">相关</span>
                            <span className="kb-ref-card-title">为什么与当前问题相关</span>
                          </div>
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
