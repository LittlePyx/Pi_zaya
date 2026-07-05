import { Button } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import type { LibraryFileItem } from '../../api/library'
import {
  conversionMetric,
  conversionQualityLabel,
  conversionQualityStatus,
  conversionQualityToneClass,
  conversionRepairAttemptLabel,
  conversionSourceReadiness,
  formatQualityRepairRecordSummary,
  hasConversionQualityIssue,
  libraryDocumentTypeView,
  normalizeTextList,
  normalizeTextValue,
  type QualityRepairHistoryRecord,
} from './libraryPageUtils'
import './LibraryFileQualityLine.css'

type LibraryFileQualityChipsProps = {
  S: Record<string, string>
  item: LibraryFileItem
  qualityStatusVisible: boolean
  qualityDiagnosticsVisible: boolean
}

type LibraryFileQualityLineProps = {
  S: Record<string, string>
  item: LibraryFileItem
  diagnosticsVisible: boolean
  repairing: boolean
  repairResult?: string
  repairRecord?: QualityRepairHistoryRecord
  onRepairQuality: () => void
  onReindex: () => void
}

function sourceQualityContext(item: LibraryFileItem) {
  const quality = item.conversion_quality
  const qualityReport = quality?.conversion_report || null
  const qualityCenter = qualityReport?.quality_center || null
  const sourceQuality = qualityReport?.source_quality || qualityCenter?.source_quality || null
  return { quality, qualityReport, qualityCenter, sourceQuality }
}

export function LibraryFileQualityChips({
  S,
  item,
  qualityStatusVisible,
  qualityDiagnosticsVisible,
}: LibraryFileQualityChipsProps) {
  const { quality, sourceQuality } = sourceQualityContext(item)
  const sourceReadiness = conversionSourceReadiness(item, S)
  const documentTypeView = libraryDocumentTypeView(sourceQuality?.document_type)

  return (
    <>
      {qualityStatusVisible && quality ? (
        <span
          className={`kb-lib-file-quality-chip ${conversionQualityToneClass(quality)}`}
          data-testid="library-file-quality-chip"
          data-quality-status={conversionQualityStatus(quality)}
          title={qualityDiagnosticsVisible ? quality.summary : conversionQualityLabel(quality)}
        >
          {conversionQualityLabel(quality)}
        </span>
      ) : null}
      {qualityStatusVisible ? (
        <span
          className={`kb-lib-source-readiness-chip is-${sourceReadiness.tone}`}
          data-testid="library-file-source-readiness"
          data-source-readiness={sourceReadiness.kind}
          title={sourceReadiness.detail}
        >
          {sourceReadiness.label}
        </span>
      ) : null}
      {documentTypeView ? (
        <span
          className={`kb-lib-file-submeta-chip is-doc-type ${documentTypeView.tone}`}
          title={documentTypeView.title}
        >
          {documentTypeView.label}
        </span>
      ) : null}
    </>
  )
}

export function LibraryFileQualityLine({
  S,
  item,
  diagnosticsVisible,
  repairing,
  repairResult,
  repairRecord,
  onRepairQuality,
  onReindex,
}: LibraryFileQualityLineProps) {
  const { quality, qualityReport, qualityCenter, sourceQuality } = sourceQualityContext(item)
  const finalRepairResult = String(
    repairResult || (repairRecord ? formatQualityRepairRecordSummary(repairRecord, S) : ''),
  ).trim()

  if (!diagnosticsVisible) return null

  const qualityIssues = Array.isArray(quality?.issues) ? quality.issues.slice(0, 3) : []
  const sourceDocumentType = normalizeTextValue(sourceQuality?.document_type).toLowerCase()
  const qualityCenterStatus = normalizeTextValue(qualityCenter?.status || qualityReport?.source_quality_status).toLowerCase()
  const qualityCenterMessage = normalizeTextValue(qualityCenter?.message || qualityReport?.source_quality_message)
  const qualityCenterBadges = normalizeTextList(qualityCenter?.badges || []).slice(0, 4)
  const qualityCenterIssueLabels = normalizeTextList(qualityCenter?.issue_labels || []).slice(0, 3)
  const qualityRepairPlan = qualityReport?.repair_plan || null
  const latestQualityRepairAttempt = qualityReport?.latest_repair_attempt || null
  const latestQualityRepairAttemptStatus = normalizeTextValue(latestQualityRepairAttempt?.status).toLowerCase()
  const latestQualityRepairAttemptLabel = conversionRepairAttemptLabel(latestQualityRepairAttempt, S)
  const latestQualityRepairAttemptTone =
    ['success', 'resolved', 'ready', 'autofixed', 'fixed'].includes(latestQualityRepairAttemptStatus)
      ? 'is-success'
      : ['error', 'failed', 'blocked'].includes(latestQualityRepairAttemptStatus)
        ? 'is-error'
        : 'is-warning'
  const qualityAutoRepairApplied = Array.isArray(qualityReport?.auto_repair_applied)
    ? qualityReport?.auto_repair_applied || []
    : []
  const mathCount = conversionMetric(quality, 'display_math') + conversionMetric(quality, 'inline_math')
  const referenceCount = conversionMetric(quality, 'references') || conversionMetric(quality, 'reference_lines')
  const referenceMetricLabel = sourceDocumentType === 'supplementary' && referenceCount <= 0
    ? 'refs n/a'
    : `refs ${referenceCount}`
  const qualityNeedsRepair = hasConversionQualityIssue(item)
  const sourceReadiness = conversionSourceReadiness(item, S)
  const sourceQualityHasActionableIssues = Boolean(
    sourceQuality?.source_text_loss
    || sourceQuality?.references_before_body
    || qualityCenterIssueLabels.length > 0
    || qualityReport?.needs_reconvert
  )
  const showSourceQualityBadges = Boolean(qualityCenterBadges.length)
    && (
      !['ready', 'none'].includes(qualityCenterStatus)
      || sourceQualityHasActionableIssues
      || !sourceReadiness.qaReady
    )
  const visibleQualityCenterBadges = showSourceQualityBadges ? qualityCenterBadges : []
  const showLatestQualityRepairAttempt = Boolean(latestQualityRepairAttempt)
    && (
      ['queued', 'running', 'partial', 'blocked', 'failed', 'error'].includes(latestQualityRepairAttemptStatus)
      || (!sourceReadiness.qaReady && Boolean(latestQualityRepairAttemptStatus))
    )
  const qualityRepairButtonLabel = sourceReadiness.action === 'reconvert'
    ? S.lib_btn_reconvert_quality
    : sourceReadiness.action === 'reindex'
      ? S.lib_btn_refresh_index
      : S.lib_btn_repair_quality
  const sourceReadinessActionAvailable = qualityNeedsRepair || sourceReadiness.action === 'reindex'

  return (
    <>
      {quality ? (
        <div className="kb-lib-quality-line" data-testid="library-file-quality-line">
          <span className="kb-lib-quality-metric">pages {conversionMetric(quality, 'page_markers')}</span>
          <span
            className="kb-lib-quality-metric"
            title={sourceDocumentType === 'supplementary' ? 'Supplementary material may not include a standalone references section.' : undefined}
          >
            {referenceMetricLabel}
          </span>
          <span className="kb-lib-quality-metric">fig {conversionMetric(quality, 'figures')}</span>
          <span className="kb-lib-quality-metric">math {mathCount}</span>
          {visibleQualityCenterBadges.map((badge) => (
            <span
              key={`${item.name}-source-badge-${badge}`}
              className={`kb-lib-quality-issue ${
                qualityCenterStatus === 'ready' || qualityCenterStatus === 'none'
                  ? 'is-success'
                  : qualityCenterStatus === 'reconvert'
                    ? 'is-error'
                    : 'is-warning'
              }`}
              title={qualityCenterMessage || badge}
              data-testid="library-file-source-quality-badge"
            >
              {badge}
            </span>
          ))}
          {sourceQuality?.source_text_loss ? (
            <span
              className="kb-lib-quality-issue is-error"
              title={qualityCenterMessage || S.lib_source_status_blocked_detail}
              data-testid="library-file-source-text-loss"
            >
              source text loss
            </span>
          ) : null}
          {sourceQuality?.references_before_body ? (
            <span
              className="kb-lib-quality-issue is-warning"
              title={qualityCenterMessage || 'References were detected before recovered body sections.'}
              data-testid="library-file-source-references-before-body"
            >
              references order
            </span>
          ) : null}
          {qualityIssues.map((issue) => (
            <span
              key={`${item.name}-${issue.code}`}
              className={`kb-lib-quality-issue ${String(issue.severity || '') === 'error' ? 'is-error' : 'is-warning'}`}
              title={issue.label}
            >
              {issue.label}
            </span>
          ))}
          {qualityReport?.auto_repair_changed ? (
            <span
              className="kb-lib-quality-issue is-success"
              title={qualityAutoRepairApplied.join(' / ') || S.lib_quality_gate_autofixed}
            >
              {S.lib_quality_auto_fixed_count.replace('{n}', String(qualityAutoRepairApplied.length || 1))}
            </span>
          ) : null}
          {qualityCenterIssueLabels.length > 0 ? (
            <span
              className={`kb-lib-quality-issue ${qualityCenterStatus === 'reconvert' ? 'is-error' : 'is-warning'}`}
              title={qualityCenterMessage}
              data-testid="library-file-source-quality-issues"
            >
              {qualityCenterIssueLabels.join(' / ')}
            </span>
          ) : null}
          {qualityReport?.needs_reconvert ? (
            <span
              className="kb-lib-quality-issue is-error"
              title={qualityRepairPlan?.reason || S.lib_source_status_blocked_detail}
            >
              {qualityRepairPlan?.scope
                ? S.lib_quality_reconvert_scope.replace('{scope}', qualityRepairPlan.scope)
                : S.lib_quality_gate_blocked}
            </span>
          ) : null}
          {showLatestQualityRepairAttempt && latestQualityRepairAttempt ? (
            <span
              className={`kb-lib-quality-issue ${latestQualityRepairAttemptTone}`}
              title={latestQualityRepairAttempt.detail || latestQualityRepairAttempt.reason || latestQualityRepairAttempt.event}
            >
              {latestQualityRepairAttemptLabel}
            </span>
          ) : null}
          {sourceReadinessActionAvailable ? (
            <Button
              size="small"
              icon={<ReloadOutlined />}
              className="kb-lib-quality-repair-btn"
              data-testid="library-quality-repair"
              loading={repairing}
              disabled={item.task_state !== 'idle'}
              onClick={sourceReadiness.action === 'reindex' ? onReindex : onRepairQuality}
            >
              {qualityRepairButtonLabel}
            </Button>
          ) : null}
        </div>
      ) : null}

      {finalRepairResult ? (
        <div className="kb-lib-quality-repair-result" data-testid="library-quality-repair-result">
          {finalRepairResult}
        </div>
      ) : null}
    </>
  )
}
