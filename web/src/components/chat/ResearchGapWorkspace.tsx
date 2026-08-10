import { useCallback, useEffect, useMemo, useState } from 'react'
import { Alert, Button, Modal, Popconfirm, Select, Spin, Tag, message } from 'antd'
import {
  CheckCircleOutlined,
  DiffOutlined,
  EyeOutlined,
  FileAddOutlined,
  ReloadOutlined,
  SearchOutlined,
  StopOutlined,
  ToolOutlined,
} from '@ant-design/icons'
import {
  chatApi,
  type CitationShelfRecord,
  type ResearchGapCandidate,
  type ResearchGapRepairApplyResult,
  type ResearchGapRepairCandidate,
  type ResearchGapRecord,
  type ResearchGapSourceExpansionApplyResult,
  type ResearchGapSourceExpansionPreviewResult,
  type ResearchGapSummary,
} from '../../api/chat'
import { useT } from '../../i18n'


interface Props {
  open: boolean
  projectId: string
  onClose: () => void
  onOpenEvidence?: (evidence: Record<string, unknown>) => void
  onShelfChanged?: (shelf: CitationShelfRecord) => void
  onOpenBrief?: (briefId: string, matrixId: string) => void
  onOpenMatrix?: (matrixId: string) => void
}

type GapFilter = 'all' | 'high' | 'open' | 'in_progress'

const EMPTY_SUMMARY: ResearchGapSummary = {
  total: 0,
  open: 0,
  in_progress: 0,
  high: 0,
  medium: 0,
  low: 0,
  searchable: 0,
  affected_matrix_count: 0,
  affected_brief_count: 0,
}

function priorityColor(priority: string): string {
  if (priority === 'high') return 'red'
  if (priority === 'medium') return 'orange'
  return 'blue'
}

function statusColor(status: string): string {
  if (status === 'in_progress') return 'processing'
  if (status === 'open') return 'warning'
  return 'default'
}

function numeric(value: unknown): number {
  const result = Number(value)
  return Number.isFinite(result) ? result : 0
}

export function ResearchGapWorkspace({
  open,
  projectId,
  onClose,
  onOpenEvidence,
  onShelfChanged,
  onOpenBrief,
  onOpenMatrix,
}: Props) {
  const S = useT()
  const [items, setItems] = useState<ResearchGapRecord[]>([])
  const [summary, setSummary] = useState<ResearchGapSummary>(EMPTY_SUMMARY)
  const [filter, setFilter] = useState<GapFilter>('all')
  const [loading, setLoading] = useState(false)
  const [candidateLoadingId, setCandidateLoadingId] = useState('')
  const [confirmingCandidateId, setConfirmingCandidateId] = useState('')
  const [ignoringGapId, setIgnoringGapId] = useState('')
  const [candidatesByGap, setCandidatesByGap] = useState<Record<string, ResearchGapCandidate[]>>({})
  const [repairLoadingId, setRepairLoadingId] = useState('')
  const [applyingRepairId, setApplyingRepairId] = useState('')
  const [repairsByGap, setRepairsByGap] = useState<Record<string, ResearchGapRepairCandidate[]>>({})
  const [lastRepair, setLastRepair] = useState<ResearchGapRepairApplyResult | null>(null)
  const [expansionLoadingId, setExpansionLoadingId] = useState('')
  const [applyingExpansionId, setApplyingExpansionId] = useState('')
  const [expansionsByCandidate, setExpansionsByCandidate] = useState<Record<string, ResearchGapSourceExpansionPreviewResult>>({})
  const [lastExpansion, setLastExpansion] = useState<ResearchGapSourceExpansionApplyResult | null>(null)

  const scan = useCallback(async (quiet = false) => {
    if (!projectId) return
    setLoading(true)
    try {
      const result = await chatApi.scanResearchGaps(projectId)
      setItems(result.items || [])
      setSummary(result.summary || EMPTY_SUMMARY)
      setCandidatesByGap({})
      setRepairsByGap({})
      setLastRepair(null)
      setExpansionsByCandidate({})
      setLastExpansion(null)
      if (!quiet) {
        message.success(
          result.summary.total > 0
            ? S.research_gap_scan_found.replace('{n}', String(result.summary.total))
            : S.research_gap_scan_clear,
        )
      }
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_gap_scan_failed)
    } finally {
      setLoading(false)
    }
  }, [S.research_gap_scan_clear, S.research_gap_scan_failed, S.research_gap_scan_found, projectId])

  useEffect(() => {
    if (!open || !projectId) return
    void scan(true)
  }, [open, projectId, scan])

  const visibleItems = useMemo(() => items.filter((item) => {
    if (filter === 'high') return item.priority === 'high'
    if (filter === 'open') return item.status === 'open'
    if (filter === 'in_progress') return item.status === 'in_progress'
    return true
  }), [filter, items])

  const findCandidates = async (gap: ResearchGapRecord) => {
    setCandidateLoadingId(gap.id)
    try {
      const result = await chatApi.listResearchGapCandidates(projectId, gap.id)
      setCandidatesByGap((current) => ({ ...current, [gap.id]: result.items || [] }))
      if ((result.items || []).length === 0) message.info(S.research_gap_candidate_empty)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_gap_candidate_failed)
    } finally {
      setCandidateLoadingId('')
    }
  }

  const previewExpansion = async (gap: ResearchGapRecord, candidate: ResearchGapCandidate) => {
    setExpansionLoadingId(candidate.id)
    try {
      const result = await chatApi.previewResearchGapSourceExpansion(projectId, gap.id, candidate.id)
      setExpansionsByCandidate((current) => ({ ...current, [candidate.id]: result }))
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_gap_expansion_preview_failed)
    } finally {
      setExpansionLoadingId('')
    }
  }

  const confirmCandidate = async (gap: ResearchGapRecord, candidate: ResearchGapCandidate) => {
    setConfirmingCandidateId(candidate.id)
    try {
      const result = await chatApi.confirmResearchGapCandidate(projectId, gap.id, candidate.id)
      setItems((current) => current.map((item) => item.id === gap.id ? result.gap : item))
      setSummary((current) => ({
        ...current,
        open: Math.max(0, current.open - (gap.status === 'open' ? 1 : 0)),
        in_progress: current.in_progress + (gap.status === 'in_progress' ? 0 : 1),
      }))
      onShelfChanged?.(result.shelf)
      message.success(S.research_gap_candidate_confirmed)
      await previewExpansion(result.gap, candidate)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_gap_candidate_confirm_failed)
    } finally {
      setConfirmingCandidateId('')
    }
  }

  const applyExpansion = async (
    gap: ResearchGapRecord,
    candidate: ResearchGapCandidate,
    expansion: ResearchGapSourceExpansionPreviewResult,
  ) => {
    setApplyingExpansionId(candidate.id)
    try {
      const result = await chatApi.applyResearchGapSourceExpansion(
        projectId,
        gap.id,
        candidate.id,
        expansion.matrix_revision,
      )
      setItems(result.research_gaps.items || [])
      setSummary(result.research_gaps.summary || EMPTY_SUMMARY)
      setCandidatesByGap({})
      setRepairsByGap({})
      setExpansionsByCandidate({})
      setLastRepair(null)
      setLastExpansion(result)
      message.success(S.research_gap_expansion_applied)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_gap_expansion_apply_failed)
    } finally {
      setApplyingExpansionId('')
    }
  }

  const findRepairs = async (gap: ResearchGapRecord) => {
    setRepairLoadingId(gap.id)
    try {
      const result = await chatApi.listResearchGapRepairs(projectId, gap.id)
      setRepairsByGap((current) => ({ ...current, [gap.id]: result.items || [] }))
      if ((result.items || []).length === 0) message.info(S.research_gap_repair_empty)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_gap_repair_failed)
    } finally {
      setRepairLoadingId('')
    }
  }

  const applyRepair = async (gap: ResearchGapRecord, repair: ResearchGapRepairCandidate) => {
    setApplyingRepairId(repair.id)
    try {
      const result = await chatApi.applyResearchGapRepair(
        projectId,
        gap.id,
        repair.id,
        gap.matrix_revision,
      )
      setItems(result.research_gaps.items || [])
      setSummary(result.research_gaps.summary || EMPTY_SUMMARY)
      setRepairsByGap({})
      setCandidatesByGap({})
      setLastRepair(result)
      setLastExpansion(null)
      message.success(S.research_gap_repair_applied)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_gap_repair_apply_failed)
    } finally {
      setApplyingRepairId('')
    }
  }

  const ignoreGap = async (gap: ResearchGapRecord) => {
    setIgnoringGapId(gap.id)
    try {
      await chatApi.ignoreResearchGap(projectId, gap.id, 'Reviewed and intentionally deferred in the project gap queue.')
      setItems((current) => current.filter((item) => item.id !== gap.id))
      setSummary((current) => ({
        ...current,
        total: Math.max(0, current.total - 1),
        open: Math.max(0, current.open - (gap.status === 'open' ? 1 : 0)),
        in_progress: Math.max(0, current.in_progress - (gap.status === 'in_progress' ? 1 : 0)),
        high: Math.max(0, current.high - (gap.priority === 'high' ? 1 : 0)),
        medium: Math.max(0, current.medium - (gap.priority === 'medium' ? 1 : 0)),
        low: Math.max(0, current.low - (gap.priority === 'low' ? 1 : 0)),
      }))
      message.success(S.research_gap_ignored)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_gap_ignore_failed)
    } finally {
      setIgnoringGapId('')
    }
  }

  const expansionFieldLabels: Record<string, string> = {
    method: S.evidence_matrix_col_method,
    dataset_or_experiment: S.evidence_matrix_col_experiment,
    metric: S.evidence_matrix_col_metric,
    key_result: S.evidence_matrix_col_result,
    limitation: S.evidence_matrix_col_limitation,
  }

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={1080}
      destroyOnHidden
      title={S.research_gap_title}
      className="kb-research-gap-modal"
    >
      <div className="kb-research-gap-workspace" data-testid="research-gap-workspace">
        <Alert
          type="info"
          showIcon
          message={S.research_gap_contract_title}
          description={S.research_gap_contract_detail}
        />
        {lastRepair ? (
          <Alert
            type={lastRepair.matrix.quality_status === 'verified' ? 'success' : 'warning'}
            showIcon
            message={S.research_gap_repair_result_title}
            description={(
              <div className="kb-research-gap-repair-result" data-testid="research-gap-repair-result">
                <p>
                  {S.research_gap_repair_result_detail
                    .replace('{matrix}', lastRepair.matrix.title)
                    .replace('{revision}', String(lastRepair.matrix.revision))
                    .replace('{comparisons}', String(lastRepair.reaudited_comparison_count))}
                </p>
                {lastRepair.affected_briefs.length > 0 ? (
                  <div>
                    <strong>{S.research_gap_repair_briefs_title}</strong>
                    {lastRepair.affected_briefs.map((brief) => (
                      <Button
                        key={brief.id}
                        size="small"
                        icon={<DiffOutlined />}
                        disabled={!brief.update_ready || !onOpenBrief}
                        onClick={() => onOpenBrief?.(brief.id, lastRepair.matrix.id)}
                        data-testid="research-gap-open-affected-brief"
                      >
                        {brief.title} · {brief.update_ready
                          ? S.research_gap_repair_brief_ready
                          : S.research_gap_repair_brief_blocked}
                      </Button>
                    ))}
                  </div>
                ) : null}
              </div>
            )}
          />
        ) : null}
        {lastExpansion ? (
          <Alert
            type={lastExpansion.matrix.quality_status === 'verified' ? 'success' : 'warning'}
            showIcon
            message={S.research_gap_expansion_result_title}
            description={(
              <div className="kb-research-gap-repair-result" data-testid="research-gap-expansion-result">
                <p>
                  {S.research_gap_expansion_result_detail
                    .replace('{matrix}', lastExpansion.matrix.title)
                    .replace('{revision}', String(lastExpansion.matrix.revision))
                    .replace('{rows}', String(lastExpansion.preserved_row_count))
                    .replace('{comparisons}', String(lastExpansion.reaudited_comparison_count))}
                </p>
                <div className="kb-research-gap-candidate-actions">
                  <Button
                    icon={<FileAddOutlined />}
                    disabled={!onOpenMatrix}
                    onClick={() => onOpenMatrix?.(lastExpansion.matrix.id)}
                    data-testid="research-gap-open-expanded-matrix"
                  >
                    {S.research_gap_expansion_open_matrix}
                  </Button>
                  {lastExpansion.affected_briefs.map((brief) => (
                    <Button
                      key={brief.id}
                      size="small"
                      icon={<DiffOutlined />}
                      disabled={!brief.update_ready || !onOpenBrief}
                      onClick={() => onOpenBrief?.(brief.id, lastExpansion.matrix.id)}
                      data-testid="research-gap-expansion-open-brief"
                    >
                      {brief.title} · {brief.update_ready
                        ? S.research_gap_repair_brief_ready
                        : S.research_gap_repair_brief_blocked}
                    </Button>
                  ))}
                </div>
                {lastExpansion.original_gap_preserved ? (
                  <small>{S.research_gap_expansion_gap_preserved}</small>
                ) : null}
              </div>
            )}
          />
        ) : null}
        <div className="kb-research-gap-summary">
          <div><strong>{summary.total}</strong><span>{S.research_gap_summary_total}</span></div>
          <div><strong>{summary.high}</strong><span>{S.research_gap_priority_high}</span></div>
          <div><strong>{summary.in_progress}</strong><span>{S.research_gap_status_in_progress}</span></div>
          <div><strong>{summary.affected_matrix_count}</strong><span>{S.research_gap_summary_matrices}</span></div>
          <div><strong>{summary.affected_brief_count}</strong><span>{S.research_gap_summary_briefs}</span></div>
        </div>
        <div className="kb-research-gap-toolbar">
          <Select<GapFilter>
            value={filter}
            onChange={setFilter}
            options={[
              { value: 'all', label: S.research_gap_filter_all },
              { value: 'high', label: S.research_gap_filter_high },
              { value: 'open', label: S.research_gap_filter_open },
              { value: 'in_progress', label: S.research_gap_status_in_progress },
            ]}
          />
          <Button icon={<ReloadOutlined />} loading={loading} onClick={() => void scan()}>
            {S.research_gap_rescan}
          </Button>
        </div>

        <Spin spinning={loading}>
          <div className="kb-research-gap-list">
            {!loading && visibleItems.length === 0 ? (
              <div className="kb-research-gap-empty">
                <CheckCircleOutlined />
                <strong>{S.research_gap_empty_title}</strong>
                <span>{S.research_gap_empty_detail}</span>
              </div>
            ) : null}
            {visibleItems.map((gap) => {
              const candidates = candidatesByGap[gap.id]
              const repairs = repairsByGap[gap.id]
              const impact = gap.impact || {}
              return (
                <article key={gap.id} className={`kb-research-gap-card priority-${gap.priority}`} data-testid="research-gap-card">
                  <header>
                    <div>
                      <Tag color={priorityColor(gap.priority)}>{S[`research_gap_priority_${gap.priority}`] || gap.priority}</Tag>
                      <Tag color={statusColor(gap.status)}>{S[`research_gap_status_${gap.status}`] || gap.status}</Tag>
                      <Tag>{S[`research_gap_kind_${gap.kind}`] || gap.kind}</Tag>
                    </div>
                    <span className="kb-research-gap-score">{S.research_gap_score.replace('{n}', String(gap.priority_score))}</span>
                  </header>
                  <h3>{gap.title}</h3>
                  <p>{gap.detail}</p>
                  <div className="kb-research-gap-context">
                    {gap.matrix_title ? <span>{S.research_gap_matrix}: {gap.matrix_title} · v{gap.matrix_revision}</span> : null}
                    {gap.brief_title ? <span>{S.research_gap_brief}: {gap.brief_title} · v{gap.brief_revision}</span> : null}
                    {gap.field ? <span>{S.research_gap_field}: {gap.field}</span> : null}
                  </div>
                  <div className="kb-research-gap-impact">
                    <strong>{S.research_gap_impact}</strong>
                    <span>{S.research_gap_impact_briefs.replace('{n}', String(numeric(impact.affected_brief_count)))}</span>
                    <span>{S.research_gap_impact_citations.replace('{n}', String(numeric(impact.affected_citation_count)))}</span>
                    <span>{S.research_gap_impact_comparisons.replace('{n}', String(numeric(impact.affected_comparison_count)))}</span>
                  </div>
                  {gap.reasons?.length ? (
                    <div className="kb-research-gap-reasons">
                      {gap.reasons.map((reason) => <Tag key={reason}>{reason}</Tag>)}
                    </div>
                  ) : null}
                  <div className="kb-research-gap-actions">
                    {gap.kind === 'missing_cell' || gap.kind === 'unsupported_cell' ? (
                      <Button
                        type="primary"
                        ghost
                        icon={<ToolOutlined />}
                        loading={repairLoadingId === gap.id}
                        onClick={() => void findRepairs(gap)}
                        data-testid="research-gap-find-repairs"
                      >
                        {S.research_gap_find_repairs}
                      </Button>
                    ) : null}
                    {gap.candidate_searchable ? (
                      <Button
                        icon={<SearchOutlined />}
                        loading={candidateLoadingId === gap.id}
                        onClick={() => void findCandidates(gap)}
                        data-testid="research-gap-find-candidates"
                      >
                        {S.research_gap_find_candidates}
                      </Button>
                    ) : null}
                    {gap.dismissible ? (
                      <Popconfirm
                        title={S.research_gap_ignore_confirm}
                        description={S.research_gap_ignore_detail}
                        onConfirm={() => void ignoreGap(gap)}
                        okText={S.confirm_ok}
                        cancelText={S.confirm_cancel}
                      >
                        <Button danger icon={<StopOutlined />} loading={ignoringGapId === gap.id}>
                          {S.research_gap_ignore}
                        </Button>
                      </Popconfirm>
                    ) : (
                      <span className="kb-research-gap-source-lock">{S.research_gap_source_lock}</span>
                    )}
                  </div>

                  {repairs ? (
                    <div className="kb-research-gap-candidates is-repair" data-testid="research-gap-repairs">
                      <strong>{S.research_gap_repairs_title}</strong>
                      <p>{S.research_gap_repairs_detail}</p>
                      {repairs.length === 0 ? <span>{S.research_gap_repair_empty}</span> : null}
                      {repairs.map((repair) => (
                        <div key={repair.id} className="kb-research-gap-candidate">
                          <div>
                            <strong>{repair.title || repair.source_name}</strong>
                            <small>{repair.location_label || repair.heading_path || repair.source_path}</small>
                          </div>
                          <blockquote>{repair.evidence_quote}</blockquote>
                          <div className="kb-research-gap-candidate-actions">
                            {onOpenEvidence ? (
                              <Button icon={<EyeOutlined />} onClick={() => onOpenEvidence({ ...repair })}>
                                {S.research_gap_open_evidence}
                              </Button>
                            ) : null}
                            <Popconfirm
                              title={S.research_gap_repair_confirm_title}
                              description={S.research_gap_repair_confirm_detail}
                              onConfirm={() => void applyRepair(gap, repair)}
                              okText={S.research_gap_repair_apply}
                              cancelText={S.confirm_cancel}
                            >
                              <Button
                                type="primary"
                                loading={applyingRepairId === repair.id}
                                data-testid="research-gap-apply-repair"
                              >
                                {S.research_gap_repair_apply}
                              </Button>
                            </Popconfirm>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : null}

                  {candidates ? (
                    <div className="kb-research-gap-candidates" data-testid="research-gap-candidates">
                      <strong>{S.research_gap_candidates_title}</strong>
                      <p>{S.research_gap_candidates_detail}</p>
                      {candidates.length === 0 ? <span>{S.research_gap_candidate_empty}</span> : null}
                      {candidates.map((candidate) => {
                        const confirmed = String(gap.action?.candidate_id || '') === candidate.id
                        const expansion = expansionsByCandidate[candidate.id]
                        return (
                          <div key={candidate.id} className="kb-research-gap-candidate">
                            <div>
                              <strong>{candidate.title || candidate.source_name}</strong>
                              <small>{candidate.location_label || candidate.heading_path || candidate.source_path}</small>
                            </div>
                            <blockquote>{candidate.evidence_quote}</blockquote>
                            <div className="kb-research-gap-candidate-actions">
                              {onOpenEvidence ? (
                                <Button icon={<EyeOutlined />} onClick={() => onOpenEvidence({ ...candidate })}>
                                  {S.research_gap_open_evidence}
                                </Button>
                              ) : null}
                              {confirmed ? (
                                <Button
                                  type="primary"
                                  ghost
                                  icon={<FileAddOutlined />}
                                  loading={expansionLoadingId === candidate.id}
                                  onClick={() => void previewExpansion(gap, candidate)}
                                  data-testid="research-gap-preview-expansion"
                                >
                                  {S.research_gap_expansion_preview}
                                </Button>
                              ) : (
                                <Popconfirm
                                  title={S.research_gap_confirm_title}
                                  description={S.research_gap_confirm_detail}
                                  onConfirm={() => void confirmCandidate(gap, candidate)}
                                  okText={S.research_gap_confirm}
                                  cancelText={S.confirm_cancel}
                                >
                                  <Button
                                    type="primary"
                                    loading={confirmingCandidateId === candidate.id}
                                    data-testid="research-gap-confirm-candidate"
                                  >
                                    {S.research_gap_confirm}
                                  </Button>
                                </Popconfirm>
                              )}
                            </div>
                            {expansion ? (
                              <div className="kb-research-gap-expansion-preview" data-testid="research-gap-expansion-preview">
                                <strong>
                                  {S.research_gap_expansion_preview_title.replace(
                                    '{paper}',
                                    expansion.preview.row.paper || expansion.preview.row.source_name,
                                  )}
                                </strong>
                                <p>{S.research_gap_expansion_preview_detail}</p>
                                <div className="kb-research-gap-expansion-fields">
                                  {expansion.preview.grounded_fields.map((field) => (
                                    <div key={field}>
                                      <Tag color="green">{expansionFieldLabels[field] || field}</Tag>
                                      <span>{expansion.preview.row.cells[field]?.value}</span>
                                    </div>
                                  ))}
                                </div>
                                {expansion.preview.missing_fields.length > 0 ? (
                                  <div className="kb-research-gap-expansion-missing">
                                    <small>{S.research_gap_expansion_missing_fields}</small>
                                    {expansion.preview.missing_fields.map((field) => (
                                      <Tag key={field}>{expansionFieldLabels[field] || field}</Tag>
                                    ))}
                                  </div>
                                ) : null}
                                <Popconfirm
                                  title={S.research_gap_expansion_confirm_title}
                                  description={S.research_gap_expansion_confirm_detail}
                                  onConfirm={() => void applyExpansion(gap, candidate, expansion)}
                                  okText={S.research_gap_expansion_apply}
                                  cancelText={S.confirm_cancel}
                                >
                                  <Button
                                    type="primary"
                                    icon={<FileAddOutlined />}
                                    loading={applyingExpansionId === candidate.id}
                                    data-testid="research-gap-apply-expansion"
                                  >
                                    {S.research_gap_expansion_apply}
                                  </Button>
                                </Popconfirm>
                              </div>
                            ) : null}
                          </div>
                        )
                      })}
                    </div>
                  ) : null}
                </article>
              )
            })}
          </div>
        </Spin>
      </div>
    </Modal>
  )
}
