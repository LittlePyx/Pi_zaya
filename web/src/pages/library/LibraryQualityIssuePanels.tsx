import { Button, Typography } from 'antd'
import type {
  LibraryQualityFailureCase,
  LibraryQualityOverviewResponse,
  LibraryQualityPriorityAction,
  LibraryQualityRepairAction,
  LibraryResearchQaRerunResponse,
} from '../../api/library'
import {
  normalizeTextValue,
  qualityActionText,
  qualityStatusText,
} from './libraryPageUtils'
import './LibraryQualityIssuePanels.css'

const { Text } = Typography

const DEFAULT_REPAIR_ACTIONS: LibraryQualityRepairAction[] = [
  { id: 'open_replay', kind: 'open_replay', label: 'Open replay', severity: 'warning', enabled: true, detail: '' },
  { id: 'rerun_case', kind: 'rerun_case', label: 'Rerun case', severity: 'warning', enabled: true, detail: '' },
  { id: 'open_report', kind: 'open_artifact', target: 'report', label: 'Open report', severity: 'warning', enabled: true, detail: '' },
]

type QualityFailureFilter = {
  name: string
  count: number
}

type LibraryQualityIssuePanelsProps = {
  S: Record<string, string>
  priorityActions: LibraryQualityPriorityAction[]
  rerunSummary: LibraryQualityOverviewResponse['rerun_summary']
  failureCases: LibraryQualityFailureCase[]
  failureFilters: QualityFailureFilter[]
  failureFilter: string
  visibleFailureCases: LibraryQualityFailureCase[]
  artifactOpening: string
  caseActionKey: string
  caseRerunResults: Record<string, LibraryResearchQaRerunResponse>
  onPriorityAction: (action: LibraryQualityPriorityAction) => void
  onOpenFailureReport: () => void
  onFailureFilterChange: (filter: string) => void
  onOpenReplayCase: (item: LibraryQualityFailureCase) => void
  onFailureAction: (item: LibraryQualityFailureCase, action: LibraryQualityRepairAction) => void
  onCopyFailureSummary: (item: LibraryQualityFailureCase) => void
}

export function LibraryQualityIssuePanels({
  S,
  priorityActions,
  rerunSummary,
  failureCases,
  failureFilters,
  failureFilter,
  visibleFailureCases,
  artifactOpening,
  caseActionKey,
  caseRerunResults,
  onPriorityAction,
  onOpenFailureReport,
  onFailureFilterChange,
  onOpenReplayCase,
  onFailureAction,
  onCopyFailureSummary,
}: LibraryQualityIssuePanelsProps) {
  return (
    <>
      {priorityActions.length > 0 ? (
        <div className="kb-lib-quality-priority-actions" data-testid="library-quality-priority-actions">
          <Text className="kb-lib-quality-report-section-title">{S.lib_quality_priority_actions}</Text>
          <div className="kb-lib-quality-priority-list">
            {priorityActions.map((action) => {
              const severity = normalizeTextValue(action.severity || 'warning').toLowerCase() || 'warning'
              const count = Number(action.count || 0)
              return (
                <button
                  key={`${action.domain}-${action.label}`}
                  type="button"
                  className={`kb-lib-quality-priority-pill is-${severity}`}
                  data-quality-action-domain={action.domain}
                  onClick={() => { onPriorityAction(action) }}
                >
                  <strong>{qualityActionText(action, S)}</strong>
                  <em>{count > 0 ? String(count) : qualityStatusText(severity, S)}</em>
                </button>
              )
            })}
          </div>
        </div>
      ) : null}
      {rerunSummary?.available ? (
        <div className="kb-lib-quality-rerun-summary" data-testid="library-quality-rerun-summary">
          <Text className="kb-lib-quality-report-section-title">Rerun history</Text>
          <div className="kb-lib-quality-rerun-summary-grid">
            <span>Runs <strong>{rerunSummary.total}</strong></span>
            <span>Passed <strong>{rerunSummary.passed}</strong></span>
            <span>Failed <strong>{rerunSummary.failed + rerunSummary.error}</strong></span>
            <span>Cases <strong>{rerunSummary.case_count}</strong></span>
          </div>
          {rerunSummary.top_failures?.length ? (
            <div className="kb-lib-quality-rerun-summary-failures">
              {rerunSummary.top_failures.slice(0, 3).map((item) => (
                <em key={item.name}>{item.name} x{item.count}</em>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
      {failureCases.length > 0 ? (
        <div className="kb-lib-quality-failure-cases" data-testid="library-quality-failure-cases">
          <div className="kb-lib-quality-failure-head">
            <Text className="kb-lib-quality-report-section-title">
              {S.lib_quality_failure_cases.replace('{n}', String(failureCases.length))}
            </Text>
            <Button
              size="small"
              className="kb-lib-quality-domain-action"
              loading={artifactOpening === 'research_qa:report'}
              onClick={onOpenFailureReport}
            >
              {S.lib_quality_failure_open_report}
            </Button>
          </div>
          {failureFilters.length > 0 ? (
            <div className="kb-lib-quality-failure-filters">
              <button
                type="button"
                className={`kb-lib-quality-failure-filter${!failureFilter ? ' is-active' : ''}`}
                data-testid="library-quality-failure-filter-all"
                onClick={() => onFailureFilterChange('')}
              >
                {S.lib_quality_failure_all}
              </button>
              {failureFilters.map((item) => (
                <button
                  key={item.name}
                  type="button"
                  className={`kb-lib-quality-failure-filter${failureFilter === item.name ? ' is-active' : ''}`}
                  data-testid="library-quality-failure-filter"
                  onClick={() => onFailureFilterChange(item.name)}
                >
                  <span>{item.name}</span>
                  <strong>{item.count}</strong>
                </button>
              ))}
            </div>
          ) : null}
          {visibleFailureCases.length > 0 ? (
            <div className="kb-lib-quality-failure-list">
              {visibleFailureCases.slice(0, 4).map((item) => {
                const docIds = Array.isArray(item.doc_ids) ? item.doc_ids.filter(Boolean) : []
                const failures = Array.isArray(item.failures) ? item.failures : []
                const missingDocIds = Array.isArray(item.missing_expected_doc_ids) ? item.missing_expected_doc_ids.filter(Boolean) : []
                const routeSummary = item.diagnostic_summary?.citation_routes || {}
                const rootCauses = Array.isArray(item.root_causes) && item.root_causes.length > 0
                  ? item.root_causes
                  : failures.slice(0, 2).map((failure) => ({
                    code: failure.name,
                    label: failure.name,
                    severity: failure.domain === 'citation_cards' ? 'error' : 'warning',
                    detail: failure.detail || '',
                    action: 'inspect_replay',
                  }))
                const sourceDiagnostics = Array.isArray(item.source_diagnostics) ? item.source_diagnostics : []
                const shelfMissingFields = Array.isArray(item.shelf_metadata_missing_fields) ? item.shelf_metadata_missing_fields : []
                const repairActions = Array.isArray(item.repair_actions) && item.repair_actions.length > 0
                  ? item.repair_actions
                  : DEFAULT_REPAIR_ACTIONS
                const rerunResult = caseRerunResults[item.id]
                const persistedRerun = item.rerun_status?.available ? item.rerun_status : null
                const rerunView = rerunResult
                  ? {
                    label: 'Rerun',
                    status: rerunResult.status,
                    failures: (rerunResult.failures || []).map((failure) => failure.name),
                    latencyMs: rerunResult.latency_ms,
                    consecutiveFailed: 0,
                  }
                  : persistedRerun
                    ? {
                      label: 'Last rerun',
                      status: persistedRerun.last_status,
                      failures: persistedRerun.failure_names || [],
                      latencyMs: persistedRerun.last_latency_ms,
                      consecutiveFailed: persistedRerun.consecutive_failed,
                    }
                    : null
                return (
                  <div
                    key={item.id}
                    role="button"
                    tabIndex={0}
                    className="kb-lib-quality-failure-case"
                    data-testid="library-quality-failure-case"
                    onClick={() => onOpenReplayCase(item)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        onOpenReplayCase(item)
                      }
                    }}
                  >
                    <span className="kb-lib-quality-failure-case-title">{item.id}</span>
                    <span className="kb-lib-quality-failure-case-question">{item.question || S.lib_quality_failure_question_empty}</span>
                    <span className="kb-lib-quality-failure-case-badges">
                      {failures.slice(0, 3).map((failure) => (
                        <em key={`${item.id}-${failure.name}`}>{failure.name}</em>
                      ))}
                    </span>
                    {docIds.length > 0 ? (
                      <span className="kb-lib-quality-failure-case-docs">
                        {S.lib_quality_failure_case_docs.replace('{docs}', docIds.slice(0, 4).join(' / '))}
                      </span>
                    ) : null}
                    <span className="kb-lib-quality-failure-case-diagnostics">
                      {missingDocIds.length > 0 ? <em>Missing {missingDocIds.slice(0, 3).join(' / ')}</em> : null}
                      <em>A {routeSummary.system_a || 0} / B {routeSummary.system_b || 0}</em>
                      {sourceDiagnostics.length > 0 ? <em>Sources {sourceDiagnostics.length}</em> : null}
                    </span>
                    {shelfMissingFields.length > 0 ? (
                      <span className="kb-lib-quality-shelf-fields" data-testid="library-quality-shelf-metadata-fields">
                        {shelfMissingFields.slice(0, 4).map((field) => (
                          <em key={`${item.id}-shelf-field-${field.name}`}>
                            {field.name} x{field.count}
                          </em>
                        ))}
                      </span>
                    ) : null}
                    <span className="kb-lib-quality-root-causes" data-testid="library-quality-root-causes">
                      {rootCauses.slice(0, 3).map((cause) => {
                        const severity = normalizeTextValue(cause.severity).toLowerCase() || 'warning'
                        return (
                          <em key={`${item.id}-${cause.code}`} className={`is-${severity}`}>
                            {cause.label}
                          </em>
                        )
                      })}
                    </span>
                    {sourceDiagnostics.length > 0 ? (
                      <span className="kb-lib-quality-source-diag" data-testid="library-quality-source-diagnostics">
                        {sourceDiagnostics.slice(0, 2).map((source) => {
                          const status = normalizeTextValue(source.quality_status).toLowerCase() || 'unknown'
                          const label = source.title || source.source_name || source.source_path || 'source'
                          return (
                            <em key={`${item.id}-${source.source_path || source.source_name}`} className={`is-${status}`}>
                              {label}{source.quality_score > 0 ? ` Q${source.quality_score}` : ''}
                            </em>
                          )
                        })}
                      </span>
                    ) : null}
                    {rerunView ? (
                      <span className={`kb-lib-quality-rerun-result is-${rerunView.status}`} data-testid="library-quality-rerun-result">
                        {rerunView.label} {rerunView.status}
                        {rerunView.consecutiveFailed ? ` · ${rerunView.consecutiveFailed}x failing` : ''}
                        {rerunView.failures?.length ? ` · ${rerunView.failures.slice(0, 2).join(' / ')}` : ''}
                        {rerunView.latencyMs ? ` · ${Math.round(rerunView.latencyMs / 1000)}s` : ''}
                      </span>
                    ) : null}
                    <span className="kb-lib-quality-failure-actions" data-testid="library-quality-failure-actions">
                      {repairActions.slice(0, 6).map((action) => {
                        const loadingKey = `${item.id}:${action.kind}:${action.target || ''}`
                        return (
                          <Button
                            key={`${item.id}-${action.id || action.kind}`}
                            size="small"
                            className="kb-lib-quality-failure-action"
                            disabled={action.enabled === false}
                            loading={
                              caseActionKey === loadingKey
                              || (action.kind === 'repair_sources' && caseActionKey === `${item.id}:repair_sources`)
                            }
                            onClick={(event) => {
                              event.stopPropagation()
                              onFailureAction(item, action)
                            }}
                          >
                            {action.label}
                          </Button>
                        )
                      })}
                      <Button
                        size="small"
                        className="kb-lib-quality-failure-action"
                        onClick={(event) => {
                          event.stopPropagation()
                          onCopyFailureSummary(item)
                        }}
                      >
                        Copy summary
                      </Button>
                    </span>
                  </div>
                )
              })}
            </div>
          ) : (
            <Text type="secondary" className="kb-lib-quality-report-empty">{S.lib_quality_failure_no_match}</Text>
          )}
        </div>
      ) : null}
    </>
  )
}
