import { useCallback, useEffect, useMemo, useState } from 'react'
import { Alert, Button, Drawer, Spin, Tag, message } from 'antd'
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  RightOutlined,
  StopOutlined,
} from '@ant-design/icons'
import {
  chatApi,
  type ProjectResearchStatus,
  type ProjectResearchStatusAction,
} from '../../api/chat'
import { useT } from '../../i18n'


interface Props {
  open: boolean
  projectId: string
  projectName?: string
  onClose: () => void
  onAction: (action: ProjectResearchStatusAction) => void | Promise<void>
}

function numeric(value: unknown): number {
  const result = Number(value)
  return Number.isFinite(result) ? result : 0
}

function statusColor(status: string): string {
  if (status === 'ready') return 'green'
  if (status === 'blocked') return 'red'
  if (status === 'needs_review') return 'orange'
  if (status === 'needs_input') return 'blue'
  return 'default'
}

function readinessIcon(readiness: string) {
  if (readiness === 'ready') return <CheckCircleOutlined />
  if (readiness === 'blocked') return <StopOutlined />
  return <ExclamationCircleOutlined />
}

export function ProjectActionCenter({
  open,
  projectId,
  projectName = '',
  onClose,
  onAction,
}: Props) {
  const S = useT()
  const [status, setStatus] = useState<ProjectResearchStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [acting, setActing] = useState(false)

  const refresh = useCallback(async () => {
    if (!projectId) return
    setLoading(true)
    try {
      setStatus(await chatApi.refreshProjectResearchStatus(projectId))
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.project_status_load_failed)
    } finally {
      setLoading(false)
    }
  }, [S.project_status_load_failed, projectId])

  useEffect(() => {
    if (!open || !projectId) return
    setStatus(null)
    void refresh()
  }, [open, projectId, refresh])

  const stages = useMemo(() => {
    if (!status) return []
    return [
      {
        key: 'sources',
        title: S.project_status_sources,
        state: status.stages.sources.status,
        detail: S.project_status_sources_detail
          .replace('{total}', String(numeric(status.stages.sources.project_source_count)))
          .replace('{changed}', String(
            numeric(status.stages.sources.changed_source_count)
            + numeric(status.stages.sources.stale_index_matrix_count),
          )),
      },
      {
        key: 'matrices',
        title: S.project_status_matrices,
        state: status.stages.matrices.status,
        detail: S.project_status_matrices_detail
          .replace('{verified}', String(numeric(status.stages.matrices.verified)))
          .replace('{total}', String(numeric(status.stages.matrices.total))),
      },
      {
        key: 'evidence',
        title: S.project_status_evidence,
        state: status.stages.evidence.status,
        detail: S.project_status_evidence_detail
          .replace('{gaps}', String(numeric(status.stages.evidence.active_gap_count)))
          .replace('{unsupported}', String(numeric(status.stages.evidence.unsupported_count))),
      },
      {
        key: 'comparisons',
        title: S.project_status_comparisons,
        state: status.stages.comparisons.status,
        detail: S.project_status_comparisons_detail
          .replace('{pending}', String(numeric(status.stages.comparisons.pending_candidate_count)))
          .replace('{scanned}', String(numeric(status.stages.comparisons.scanned_matrix_count)))
          .replace('{eligible}', String(numeric(status.stages.comparisons.eligible_matrix_count))),
      },
      {
        key: 'briefs',
        title: S.project_status_briefs,
        state: status.stages.briefs.status,
        detail: S.project_status_briefs_detail
          .replace('{current}', String(numeric(status.stages.briefs.current)))
          .replace('{total}', String(numeric(status.stages.briefs.total))),
      },
    ]
  }, [S, status])

  const runPrimaryAction = async () => {
    const action = status?.recommended_action
    if (!action) return
    if (action.code === 'refresh_project_status') {
      await refresh()
      return
    }
    setActing(true)
    try {
      await onAction(action)
    } finally {
      setActing(false)
    }
  }

  const action = status?.recommended_action
  const actionLabel = action
    ? S[`project_status_action_${action.code}`] || action.code
    : ''
  const actionReason = action
    ? S[`project_status_reason_${action.reason}`] || action.reason
    : ''
  const readiness = status?.readiness || 'needs_review'
  const totalMs = numeric(status?.phase_timings_ms?.total).toFixed(1)
  const comparisonMs = numeric(status?.phase_timings_ms?.scan_comparison_candidates).toFixed(1)

  return (
    <Drawer
      open={open}
      onClose={onClose}
      width={620}
      title={S.project_status_title}
      destroyOnHidden
      data-testid="project-action-center"
      extra={(
        <Button
          icon={<ReloadOutlined />}
          loading={loading}
          onClick={() => { void refresh() }}
          data-testid="project-status-refresh"
        >
          {S.project_status_refresh}
        </Button>
      )}
    >
      <Spin spinning={loading && !status}>
        <div className="kb-project-status-stack">
          <div>
            <div className="kb-project-status-project" data-testid="project-status-project-name">
              {status?.project?.name || projectName || S.project_status_project_fallback}
            </div>
            <div className="kb-project-status-contract">{S.project_status_deterministic_note}</div>
          </div>

          {status ? (
            <Alert
              type={readiness === 'ready' ? 'success' : readiness === 'blocked' ? 'error' : 'warning'}
              showIcon
              icon={readinessIcon(readiness)}
              message={S[`project_status_readiness_${readiness}`] || readiness}
              description={S.project_status_readiness_detail}
              data-testid="project-status-readiness"
            />
          ) : null}

          {action ? (
            <section className="kb-project-status-primary" data-testid="project-status-primary-action">
              <div className="kb-project-status-eyebrow">{S.project_status_next_action}</div>
              <div className="kb-project-status-primary-title">{actionLabel}</div>
              <div className="kb-project-status-primary-reason">{actionReason}</div>
              <div className="kb-project-status-primary-meta">
                {action.gap_count > 0 ? (
                  <Tag color="orange">{S.project_status_gap_count.replace('{n}', String(action.gap_count))}</Tag>
                ) : null}
                {action.candidate_count > 0 ? (
                  <Tag color="blue">{S.project_status_candidate_count.replace('{n}', String(action.candidate_count))}</Tag>
                ) : null}
                {action.matrix_title ? <Tag>{action.matrix_title}</Tag> : null}
                {action.brief_title ? <Tag>{action.brief_title}</Tag> : null}
              </div>
              <Button
                type="primary"
                icon={<RightOutlined />}
                loading={acting}
                onClick={() => { void runPrimaryAction() }}
                data-testid={`project-status-action-${action.code}`}
              >
                {actionLabel}
              </Button>
            </section>
          ) : null}

          <section>
            <div className="kb-project-status-section-title">{S.project_status_pipeline}</div>
            <div className="kb-project-status-grid">
              {stages.map((stage) => (
                <article key={stage.key} className="kb-project-status-stage" data-testid={`project-status-stage-${stage.key}`}>
                  <div className="kb-project-status-stage-head">
                    <span>{stage.title}</span>
                    <Tag color={statusColor(stage.state)}>
                      {S[`project_status_state_${stage.state}`] || stage.state}
                    </Tag>
                  </div>
                  <div className="kb-project-status-stage-detail">{stage.detail}</div>
                </article>
              ))}
            </div>
          </section>

          {status ? (
            <section className="kb-project-status-coverage" data-testid="project-status-coverage">
              <div className="kb-project-status-section-title">{S.project_status_scan_coverage}</div>
              <div>
                {S.project_status_scan_coverage_detail
                  .replace('{scanned}', String(status.comparison_scan.scanned_matrix_count))
                  .replace('{eligible}', String(status.comparison_scan.eligible_matrix_count))
                  .replace('{pairs}', String(status.comparison_scan.examined_row_pairs))}
              </div>
              <div>
                {S.project_status_timing_detail
                  .replace('{total}', totalMs)
                  .replace('{comparisons}', comparisonMs)}
              </div>
            </section>
          ) : null}
        </div>
      </Spin>
    </Drawer>
  )
}
