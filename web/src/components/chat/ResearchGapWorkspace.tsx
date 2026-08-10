import { useCallback, useEffect, useMemo, useState } from 'react'
import { Alert, Button, Modal, Popconfirm, Select, Spin, Tag, message } from 'antd'
import {
  CheckCircleOutlined,
  EyeOutlined,
  ReloadOutlined,
  SearchOutlined,
  StopOutlined,
} from '@ant-design/icons'
import {
  chatApi,
  type CitationShelfRecord,
  type ResearchGapCandidate,
  type ResearchGapRecord,
  type ResearchGapSummary,
} from '../../api/chat'
import { useT } from '../../i18n'


interface Props {
  open: boolean
  projectId: string
  onClose: () => void
  onOpenEvidence?: (evidence: Record<string, unknown>) => void
  onShelfChanged?: (shelf: CitationShelfRecord) => void
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

  const scan = useCallback(async (quiet = false) => {
    if (!projectId) return
    setLoading(true)
    try {
      const result = await chatApi.scanResearchGaps(projectId)
      setItems(result.items || [])
      setSummary(result.summary || EMPTY_SUMMARY)
      setCandidatesByGap({})
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
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_gap_candidate_confirm_failed)
    } finally {
      setConfirmingCandidateId('')
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

                  {candidates ? (
                    <div className="kb-research-gap-candidates" data-testid="research-gap-candidates">
                      <strong>{S.research_gap_candidates_title}</strong>
                      <p>{S.research_gap_candidates_detail}</p>
                      {candidates.length === 0 ? <span>{S.research_gap_candidate_empty}</span> : null}
                      {candidates.map((candidate) => (
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
                          </div>
                        </div>
                      ))}
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
