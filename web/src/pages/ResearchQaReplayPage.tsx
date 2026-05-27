import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { MessageList } from '../components/chat/MessageList'
import type { ReaderOpenPayload } from '../components/chat/reader/readerTypes'
import type { Message } from '../api/chat'
import type {
  LibraryQualityCitationDiagnostic,
  LibraryQualityDiagnosticIssue,
  LibraryQualityFailureCase,
  LibraryQualityRefDiagnostic,
} from '../api/library'
import {
  RESEARCH_LIBRARY_DOCS,
  RESEARCH_QA_CASES,
  RESEARCH_QA_MESSAGES,
  RESEARCH_QA_REFS,
} from '../testing/researchQaFixtures'

const REPLAY_FAILURE_STORAGE_KEY = 'kb.researchQaReplay.failureCase.v1'

function loadStoredFailureCase(caseId: string): LibraryQualityFailureCase | null {
  if (typeof window === 'undefined' || !caseId) return null
  try {
    const raw = window.sessionStorage.getItem(REPLAY_FAILURE_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as LibraryQualityFailureCase
    if (String(parsed?.id || '') !== caseId) return null
    return parsed
  } catch {
    return null
  }
}

function refsForMessages(messages: Message[]) {
  const ids = new Set(messages.filter((item) => item.role === 'user').map((item) => String(item.id)))
  return Object.fromEntries(
    Object.entries(RESEARCH_QA_REFS).filter(([key]) => ids.has(String(key))),
  )
}

function isSystemBCitation(item: Record<string, unknown>) {
  return Boolean(item.is_inpaper)
}

function textValue(value: unknown, fallback = '') {
  const text = String(value || '').trim()
  return text || fallback
}

function numberValue(value: unknown, fallback = 0) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function diagnosticIssues(value: unknown): LibraryQualityDiagnosticIssue[] {
  return Array.isArray(value)
    ? value.filter((item): item is LibraryQualityDiagnosticIssue => Boolean(item && typeof item === 'object'))
    : []
}

function issueLabel(issue: LibraryQualityDiagnosticIssue) {
  const name = textValue(issue.name, 'issue')
  const field = textValue(issue.field)
  return field ? `${name} / ${field}` : name
}

function issuePills(issues: LibraryQualityDiagnosticIssue[], prefix: string) {
  if (!issues.length) return null
  return (
    <div className="mt-2 flex flex-wrap gap-1" data-testid={`${prefix}-issues`}>
      {issues.slice(0, 4).map((issue, index) => (
        <span
          key={`${prefix}-${issue.name}-${issue.field || ''}-${index}`}
          className={`rounded-full border px-2 py-0.5 text-[11px] ${
            issue.severity === 'warning'
              ? 'border-amber-500/30 bg-amber-500/10 text-amber-700'
              : 'border-red-500/30 bg-red-500/10 text-red-700'
          }`}
          title={textValue(issue.detail) || undefined}
        >
          {issueLabel(issue)}
        </span>
      ))}
    </div>
  )
}

function citationDiagnosticsFromFixture(cites: Array<Record<string, unknown>>): LibraryQualityCitationDiagnostic[] {
  return cites.slice(0, 8).map((item) => {
    const isSystemB = isSystemBCitation(item)
    return {
      route: isSystemB ? 'system_b' : 'system_a',
      num: numberValue(item.num || item.ref_num),
      anchor: textValue(item.anchor || item.anchor_id || item.block_id),
      title: textValue(item.title || item.source_name),
      source_name: textValue(item.source_name),
      source_path: textValue(item.source_path),
      heading_path: textValue(item.heading_path || item.location_label),
      evidence_quote: textValue(item.evidence_quote || item.citation_context),
      answer_claim: textValue(item.answer_claim),
      support_relation: textValue(item.support_relation || item.user_question_relation),
      trace: textValue(item.citation_context_source || item.mapping_source || item.anchor_kind),
      quality_issues: diagnosticIssues(item.quality_issues),
      shelf_quality_issues: diagnosticIssues(item.shelf_quality_issues),
      quality_issue_count: numberValue(item.quality_issue_count),
    }
  })
}

function refDiagnosticsFromFixture(refs: Array<Record<string, unknown>>): LibraryQualityRefDiagnostic[] {
  return refs.slice(0, 8).map((hit) => {
    const meta = (hit.meta && typeof hit.meta === 'object' ? hit.meta : {}) as Record<string, unknown>
    const uiMeta = (hit.ui_meta && typeof hit.ui_meta === 'object' ? hit.ui_meta : {}) as Record<string, unknown>
    const citationMeta = (uiMeta.citation_meta && typeof uiMeta.citation_meta === 'object' ? uiMeta.citation_meta : {}) as Record<string, unknown>
    return {
      title: textValue(uiMeta.display_name || citationMeta.title || meta.source_name),
      source_name: textValue(uiMeta.source_name || meta.source_name),
      source_path: textValue(uiMeta.source_path || citationMeta.source_path || meta.source_path),
      heading_path: textValue(uiMeta.heading_path || meta.heading_path),
      score: numberValue(uiMeta.score || hit.score),
      summary_line: textValue(uiMeta.summary_line),
      why_line: textValue(uiMeta.why_line),
      polish_status: textValue(uiMeta.polish_status),
      ref_pack_state: textValue(meta.ref_pack_state),
      evidence_quote: textValue(hit.text),
      quality_issues: diagnosticIssues(hit.quality_issues),
      quality_issue_count: numberValue(hit.quality_issue_count),
    }
  })
}

export default function ResearchQaReplayPage() {
  const [searchParams] = useSearchParams()
  const [payload, setPayload] = useState<ReaderOpenPayload | null>(null)
  const selectedCaseId = String(searchParams.get('case') || '').trim()
  const selectedCase = useMemo(
    () => RESEARCH_QA_CASES.find((item) => item.id === selectedCaseId) || null,
    [selectedCaseId],
  )
  const [failureCase, setFailureCase] = useState<LibraryQualityFailureCase | null>(() => loadStoredFailureCase(selectedCaseId))
  useEffect(() => {
    setFailureCase(loadStoredFailureCase(selectedCaseId))
  }, [selectedCaseId])
  const replayCases = selectedCase ? [selectedCase] : RESEARCH_QA_CASES
  const replayMessages = useMemo(() => {
    if (!selectedCase) return RESEARCH_QA_MESSAGES
    const ids = new Set([selectedCase.userMessageId, selectedCase.assistantMessageId])
    return RESEARCH_QA_MESSAGES.filter((item) => ids.has(Number(item.id)))
  }, [selectedCase])
  const replayRefs = useMemo(() => refsForMessages(replayMessages), [replayMessages])
  const selectedRefs = useMemo(() => selectedCase ? selectedCase.refs : [], [selectedCase])
  const selectedCites = useMemo(() => selectedCase ? selectedCase.citeDetails : [], [selectedCase])
  const diagnosticDocIds = useMemo(() => {
    const ids = new Set<string>()
    for (const value of selectedCase?.docIds || []) ids.add(value)
    for (const value of failureCase?.doc_ids || []) ids.add(value)
    for (const value of failureCase?.expected_doc_ids || []) ids.add(value)
    for (const value of failureCase?.ref_doc_ids || []) ids.add(value)
    for (const value of failureCase?.citation_doc_ids || []) ids.add(value)
    return ids
  }, [selectedCase, failureCase])
  const coveredDocIds = useMemo(
    () => (selectedCase || failureCase ? diagnosticDocIds : new Set(RESEARCH_QA_CASES.flatMap((item) => item.docIds))),
    [selectedCase, failureCase, diagnosticDocIds],
  )
  const coveredDocs = useMemo(
    () => RESEARCH_LIBRARY_DOCS.filter((doc) => coveredDocIds.has(doc.id)),
    [coveredDocIds],
  )
  const citationDiagnostics = useMemo(() => {
    if (failureCase?.citation_diagnostics?.length) return failureCase.citation_diagnostics
    return citationDiagnosticsFromFixture(selectedCites)
  }, [failureCase, selectedCites])
  const refDiagnostics = useMemo(() => {
    if (failureCase?.ref_diagnostics?.length) return failureCase.ref_diagnostics
    return refDiagnosticsFromFixture(selectedRefs)
  }, [failureCase, selectedRefs])
  const missingExpectedDocIds = useMemo(() => {
    if (failureCase?.missing_expected_doc_ids?.length) return failureCase.missing_expected_doc_ids
    if (!failureCase) return []
    const observed = new Set([...(failureCase.ref_doc_ids || []), ...(failureCase.citation_doc_ids || [])])
    return (failureCase.expected_doc_ids || []).filter((item) => !observed.has(item))
  }, [failureCase])
  const routeSummary = useMemo(() => {
    const fromCase = failureCase?.diagnostic_summary?.citation_routes
    if (fromCase) return fromCase
    return {
      system_a: citationDiagnostics.filter((item) => item.route !== 'system_b').length,
      system_b: citationDiagnostics.filter((item) => item.route === 'system_b').length,
    }
  }, [failureCase, citationDiagnostics])
  const qualityGateSummary = useMemo(() => {
    const summary = failureCase?.diagnostic_summary || {}
    const citationIssues = citationDiagnostics.reduce(
      (total, item) => total + diagnosticIssues(item.quality_issues).length,
      0,
    )
    const shelfIssues = citationDiagnostics.reduce(
      (total, item) => total + diagnosticIssues(item.shelf_quality_issues).length,
      0,
    )
    const refIssues = refDiagnostics.reduce(
      (total, item) => total + diagnosticIssues(item.quality_issues).length,
      0,
    )
    return {
      citationFailures: numberValue(summary.citation_card_failure_count, citationIssues),
      citationWarnings: numberValue(summary.citation_card_warning_count, 0),
      shelfFailures: numberValue(summary.shelf_failure_count, shelfIssues),
      shelfWarnings: numberValue(summary.shelf_warning_count, 0),
      shelfMetadataReady: numberValue(summary.shelf_metadata_ready_count, 0),
      shelfDoi: numberValue(summary.shelf_doi_count, 0),
      shelfSourceClickable: numberValue(summary.shelf_source_clickable_count, 0),
      shelfReview: numberValue(summary.shelf_review_count, 0),
      refFailures: numberValue(summary.ref_card_failure_count, refIssues),
      refWarnings: numberValue(summary.ref_card_warning_count, 0),
      systemBReview: numberValue(summary.system_b_needs_review_count, 0),
      systemBAnswerContextOnly: numberValue(summary.system_b_answer_context_only_count, 0),
      systemBFallback: numberValue(summary.system_b_reference_index_fallback_count, 0),
    }
  }, [failureCase, citationDiagnostics, refDiagnostics])

  return (
    <div className="min-h-screen bg-[var(--bg)] px-5 py-5 text-[var(--text)]">
      <div className="mx-auto flex max-w-7xl flex-col gap-4">
        <section className="rounded-[8px] border border-[var(--border)] bg-[var(--panel)] p-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h1 className="m-0 text-[20px] font-semibold">Research QA replay</h1>
              <p className="mt-1 text-sm text-[var(--muted-text)]">
                基于当前本地文献库抽样构造的真实科研学习问题，用来检查回答质量、原文证据、文内参考和参考定位卡。
              </p>
            </div>
            <div className="flex gap-2 text-sm">
              <span className="rounded-full border border-[var(--border)] px-3 py-1" data-testid="research-qa-doc-count">
                文献 {RESEARCH_LIBRARY_DOCS.length}
              </span>
              <span className="rounded-full border border-[var(--border)] px-3 py-1" data-testid="research-qa-case-count">
                问题 {replayCases.length}
              </span>
            </div>
          </div>

          <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {coveredDocs.map((doc) => (
              <div
                key={doc.id}
                className="rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] px-3 py-2"
                data-testid={`research-qa-doc-${doc.id}`}
              >
                <div className="text-sm font-medium">{doc.shortLabel}</div>
                <div className="mt-1 line-clamp-2 text-xs text-[var(--muted-text)]">{doc.title}</div>
              </div>
            ))}
          </div>
        </section>

        {(selectedCaseId || selectedCase || failureCase) ? (
          <section className="rounded-[8px] border border-[var(--border)] bg-[var(--panel)] p-4" data-testid="research-qa-diagnostic">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--muted-text)]">Quality diagnostic</div>
                <h2 className="m-0 mt-1 text-[16px] font-semibold" data-testid="research-qa-diagnostic-case">
                  {selectedCaseId || selectedCase?.id || failureCase?.id || 'all'}
                </h2>
                <p className="mt-1 max-w-4xl text-sm text-[var(--muted-text)]" data-testid="research-qa-diagnostic-question">
                  {failureCase?.question || selectedCase?.question || 'No replay fixture found for this case id. Open the latest quality report for raw details.'}
                </p>
              </div>
              <div className="flex flex-wrap gap-2 text-xs">
                <span className="rounded-full border border-[var(--border)] px-2 py-1" data-testid="research-qa-diagnostic-cite-count">
                  citations {failureCase?.citation_count ?? selectedCites.length}
                </span>
                <span className="rounded-full border border-[var(--border)] px-2 py-1">
                  refs {failureCase?.ref_hit_count ?? selectedRefs.length}
                </span>
                <span className="rounded-full border border-[var(--border)] px-2 py-1">
                  system B {failureCase?.system_b_count ?? selectedCites.filter(isSystemBCitation).length}
                </span>
              </div>
            </div>

            {failureCase?.failures?.length ? (
              <div className="mt-3 flex flex-wrap gap-2" data-testid="research-qa-diagnostic-failures">
                {failureCase.failures.map((failure) => (
                  <span key={`${failure.name}-${failure.detail}`} className="rounded-full border border-red-500/30 bg-red-500/10 px-2 py-1 text-xs">
                    {failure.name}
                  </span>
                ))}
              </div>
            ) : null}

            <div className="mt-3 grid gap-3 md:grid-cols-3">
              <div className="rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] p-3">
                <div className="text-xs font-semibold text-[var(--muted-text)]">Expected docs</div>
                <div className="mt-2 text-xs" data-testid="research-qa-diagnostic-docs">
                  {Array.from(diagnosticDocIds).join(' / ') || 'none'}
                </div>
              </div>
              <div className="rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] p-3">
                <div className="text-xs font-semibold text-[var(--muted-text)]">Citation routes</div>
                <div className="mt-2 text-xs">
                  System A {routeSummary.system_a || 0} / System B {routeSummary.system_b || 0}
                </div>
              </div>
              <div className="rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] p-3">
                <div className="text-xs font-semibold text-[var(--muted-text)]">Replay scope</div>
                <div className="mt-2 text-xs">
                  {selectedCase ? 'single fixture case' : 'full fixture replay'}
                </div>
              </div>
            </div>

            <div
              className="mt-3 rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] p-3"
              data-testid="research-qa-diagnostic-quality-gates"
            >
              <div className="text-xs font-semibold text-[var(--muted-text)]">Card quality gates</div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs">
                <span className="rounded-full border border-[var(--border)] px-2 py-1">
                  citation failures {qualityGateSummary.citationFailures}
                </span>
                <span className="rounded-full border border-[var(--border)] px-2 py-1">
                  shelf failures {qualityGateSummary.shelfFailures}
                </span>
                <span className="rounded-full border border-[var(--border)] px-2 py-1">
                  ref failures {qualityGateSummary.refFailures}
                </span>
                <span className="rounded-full border border-[var(--border)] px-2 py-1">
                  metadata ready {qualityGateSummary.shelfMetadataReady}
                </span>
                <span className="rounded-full border border-[var(--border)] px-2 py-1">
                  DOI {qualityGateSummary.shelfDoi}
                </span>
                <span className="rounded-full border border-[var(--border)] px-2 py-1">
                  source open {qualityGateSummary.shelfSourceClickable}
                </span>
                {qualityGateSummary.shelfReview > 0 ? (
                  <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-amber-700">
                    shelf review {qualityGateSummary.shelfReview}
                  </span>
                ) : null}
                {qualityGateSummary.systemBReview > 0 || qualityGateSummary.systemBAnswerContextOnly > 0 || qualityGateSummary.systemBFallback > 0 ? (
                  <span className="rounded-full border border-red-500/30 bg-red-500/10 px-2 py-1 text-red-700">
                    system B review {qualityGateSummary.systemBReview}
                  </span>
                ) : null}
              </div>
            </div>

            {missingExpectedDocIds.length > 0 ? (
              <div
                className="mt-3 rounded-[8px] border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-700"
                data-testid="research-qa-diagnostic-missing-docs"
              >
                Missing expected docs: {missingExpectedDocIds.join(' / ')}
              </div>
            ) : null}

            <div className="mt-3 grid gap-3 md:grid-cols-2">
              <div
                className="rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] p-3"
                data-testid="research-qa-diagnostic-citations"
              >
                <div className="text-xs font-semibold text-[var(--muted-text)]">Citation diagnostic</div>
                <div className="mt-2 grid gap-2">
                  {citationDiagnostics.length > 0 ? citationDiagnostics.slice(0, 4).map((item, index) => (
                    <div key={`${item.route}-${item.anchor}-${index}`} className="min-w-0 rounded-[8px] border border-[var(--border)] bg-[var(--panel)] px-2 py-2 text-xs">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-full border border-[var(--border)] px-2 py-0.5 font-semibold">
                          {item.route === 'system_b' ? 'System B' : 'System A'}
                        </span>
                        <span className="min-w-0 flex-1 truncate font-medium">{item.title || item.source_name || item.source_path || 'Untitled source'}</span>
                      </div>
                      {item.heading_path ? <div className="mt-1 truncate text-[var(--muted-text)]">{item.heading_path}</div> : null}
                      {item.evidence_quote ? <div className="mt-1 line-clamp-2 text-[var(--muted-text)]">{item.evidence_quote}</div> : null}
                      {issuePills(
                        [
                          ...diagnosticIssues(item.quality_issues),
                          ...diagnosticIssues(item.shelf_quality_issues),
                        ],
                        `research-qa-diagnostic-citation-${index}`,
                      )}
                    </div>
                  )) : (
                    <div className="text-xs text-[var(--muted-text)]">No citation diagnostics captured.</div>
                  )}
                </div>
              </div>
              <div
                className="rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] p-3"
                data-testid="research-qa-diagnostic-refs"
              >
                <div className="text-xs font-semibold text-[var(--muted-text)]">Reference basket diagnostic</div>
                <div className="mt-2 grid gap-2">
                  {refDiagnostics.length > 0 ? refDiagnostics.slice(0, 4).map((item, index) => (
                    <div key={`${item.source_path}-${index}`} className="min-w-0 rounded-[8px] border border-[var(--border)] bg-[var(--panel)] px-2 py-2 text-xs">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="min-w-0 flex-1 truncate font-medium">{item.title || item.source_name || item.source_path || 'Untitled source'}</span>
                        <span className="rounded-full border border-[var(--border)] px-2 py-0.5">{numberValue(item.score).toFixed(1)}</span>
                      </div>
                      {item.heading_path ? <div className="mt-1 truncate text-[var(--muted-text)]">{item.heading_path}</div> : null}
                      {item.summary_line || item.why_line ? (
                        <div className="mt-1 line-clamp-2 text-[var(--muted-text)]">{item.summary_line || item.why_line}</div>
                      ) : null}
                      {item.polish_status || item.ref_pack_state ? (
                        <div className="mt-1 text-[var(--muted-text)]">{[item.polish_status, item.ref_pack_state].filter(Boolean).join(' / ')}</div>
                      ) : null}
                      {issuePills(diagnosticIssues(item.quality_issues), `research-qa-diagnostic-ref-${index}`)}
                    </div>
                  )) : (
                    <div className="text-xs text-[var(--muted-text)]">No reference basket diagnostics captured.</div>
                  )}
                </div>
              </div>
            </div>
          </section>
        ) : null}

        <section className="rounded-[8px] border border-[var(--border)] bg-[var(--panel)] p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold">验收问题集</div>
              <div className="mt-1 text-xs text-[var(--muted-text)]">
                每个问题都按“普通研究生/科研学习者会怎么问”设计，避免直接说功能名。
              </div>
            </div>
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            {replayCases.map((item, index) => (
              <div
                key={item.id}
                className="rounded-[8px] border border-[var(--border)] px-3 py-2"
                data-testid={`research-qa-case-${item.id}`}
              >
                <div className="text-xs text-[var(--muted-text)]">#{index + 1}</div>
                <div className="mt-1 text-sm font-medium">{item.question}</div>
                <div className="mt-2 text-xs text-[var(--muted-text)]">
                  {item.acceptance.join(' / ')}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="min-h-[720px] rounded-[8px] border border-[var(--border)] bg-[var(--panel)]">
          <MessageList
            messages={replayMessages}
            refs={replayRefs}
            onOpenReader={(nextPayload) => setPayload(nextPayload)}
            paperGuideSourcePath=""
            paperGuideSourceName=""
          />
        </section>

        <section className="rounded-[8px] border border-[var(--border)] bg-[var(--panel)] p-4">
          <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-[var(--muted-text)]">
            Last open payload
          </div>
          <pre
            className="min-h-24 whitespace-pre-wrap rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] px-3 py-3 text-xs"
            data-testid="research-qa-open-payload"
          >
            {payload ? JSON.stringify(payload, null, 2) : '(empty)'}
          </pre>
        </section>
      </div>
    </div>
  )
}
