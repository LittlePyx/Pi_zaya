import { Button, Tag, Typography } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import type {
  LibraryConversionQualityBatchResponse,
  LibraryQualityRepairImpact,
  LibraryQualityRepairRun,
} from '../../api/library'
import {
  formatSignedNumber,
  qualityBatchIndexText,
  qualityBatchStatusText,
  qualityRepairImpactIndexText,
  qualityRepairRunCanAdvance,
  qualityRepairRunStatusText,
  qualityRepairRunTagColor,
  qualityVerificationText,
} from './libraryPageUtils'
import './LibraryQualityStatusPanels.css'

const { Text } = Typography

type QualityDomainView = {
  key: 'conversion' | 'research_qa' | 'citation_cards' | 'reader_locate'
  label: string
  available: boolean
  status: string
  statusLabel: string
  countText: string
  detailText: string
  failureText: string
}

type QualityArtifactDomain = 'research_qa' | 'citation_cards'
type QualityArtifactTarget = 'report' | 'folder' | 'runbook'

type LibraryQualityStatusPanelsProps = {
  S: Record<string, string>
  batchResult: LibraryConversionQualityBatchResponse | null
  repairImpact: LibraryQualityRepairImpact | null
  repairRun: LibraryQualityRepairRun | null
  repairAdvancing: boolean
  domains: QualityDomainView[]
  reviewCount: number
  readerLocateRepairCount: number
  artifactOpening: string
  onFocusReview: () => void
  onRepairReaderLocateSources: () => void
  onAdvanceRepairRun: () => void
  onOpenArtifact: (domain: QualityArtifactDomain, target: QualityArtifactTarget) => void
}

function basename(path: string) {
  return path.split(/[\\/]/).pop() || path
}

function qualityDomainTagColor(status: string) {
  if (status === 'good') return 'success'
  if (status === 'error') return 'error'
  if (status === 'warning') return 'warning'
  return 'default'
}

function LibraryQualityBatchResultPanel({ result }: { result: LibraryConversionQualityBatchResponse | null }) {
  if (!result) return null

  return (
    <div className="kb-lib-quality-batch-result" data-testid="library-quality-batch-result">
      <div className="kb-lib-quality-repair-impact-head">
        <Text className="kb-lib-quality-report-section-title">{qualityBatchStatusText(result)}</Text>
        <Tag color={result.failed > 0 ? 'warning' : (result.needs_reindex ? 'processing' : 'success')}>
          {qualityBatchIndexText(result)}
        </Tag>
      </div>
      <div className="kb-lib-quality-repair-impact-grid">
        <span>
          <em>Ready</em>
          <strong>{result.ready}</strong>
        </span>
        <span>
          <em>Autofix</em>
          <strong>{result.autofix}</strong>
        </span>
        <span>
          <em>Reconvert</em>
          <strong>{result.reconvert}</strong>
        </span>
        <span>
          <em>Review</em>
          <strong>{result.review}</strong>
        </span>
      </div>
      {(result.changed_paths || []).length > 0 || (result.reconvert_paths || []).length > 0 || (result.errors || []).length > 0 ? (
        <div className="kb-lib-quality-repair-impact-issues">
          {(result.changed_paths || []).slice(0, 4).map((path) => (
            <span key={`batch-changed-${path}`} className="is-fixed" title={path}>changed: {basename(path)}</span>
          ))}
          {(result.reconvert_paths || []).slice(0, 3).map((path) => (
            <span key={`batch-reconvert-${path}`} className="is-remaining" title={path}>reconvert: {basename(path)}</span>
          ))}
          {(result.errors || []).slice(0, 2).map((item) => (
            <span key={`batch-error-${item.path}`} className="is-remaining" title={item.error}>failed: {basename(item.path)}</span>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function LibraryQualityRepairImpactPanel({
  impact,
  run,
  advancing,
  onAdvanceRepairRun,
}: {
  impact: LibraryQualityRepairImpact | null
  run: LibraryQualityRepairRun | null
  advancing: boolean
  onAdvanceRepairRun: () => void
}) {
  if (!impact) return null

  const verificationText = qualityVerificationText(run?.verification as Record<string, unknown> | undefined)

  return (
    <div className="kb-lib-quality-repair-impact" data-testid="library-quality-repair-impact">
      <div className="kb-lib-quality-repair-impact-head">
        <Text className="kb-lib-quality-report-section-title">Repair impact</Text>
        <Tag color={impact.reindexed === true ? 'success' : (impact.needs_reindex ? (impact.reindexed === false ? 'warning' : 'processing') : 'success')}>
          {qualityRepairImpactIndexText(impact)}
        </Tag>
      </div>
      {run ? (
        <div className="kb-lib-quality-repair-run" data-testid="library-quality-repair-run">
          <Tag color={qualityRepairRunTagColor(run)}>
            {qualityRepairRunStatusText(run)}
          </Tag>
          <span>{run.run_id.slice(0, 8)}</span>
          {run.detail ? <em>{run.detail}</em> : null}
          {verificationText ? (
            <em className="kb-lib-quality-repair-run-verification">
              {verificationText}
            </em>
          ) : null}
          {qualityRepairRunCanAdvance(run) ? (
            <Button
              size="small"
              icon={<ReloadOutlined />}
              loading={advancing}
              data-testid="library-quality-repair-run-advance"
              onClick={onAdvanceRepairRun}
            >
              Continue
            </Button>
          ) : null}
        </div>
      ) : null}
      <div className="kb-lib-quality-repair-impact-grid">
        <span>
          <em>Repaired</em>
          <strong>{impact.repaired}</strong>
        </span>
        <span>
          <em>Queued</em>
          <strong>{impact.enqueued}</strong>
        </span>
        <span>
          <em>Improved</em>
          <strong>{impact.improved}</strong>
        </span>
        <span>
          <em>Score</em>
          <strong>Q{impact.before_avg_score} -&gt; Q{impact.after_avg_score} ({formatSignedNumber(impact.score_delta)})</strong>
        </span>
      </div>
      {(impact.fixed_issue_codes || []).length > 0 || (impact.remaining_issue_codes || []).length > 0 ? (
        <div className="kb-lib-quality-repair-impact-issues">
          {(impact.fixed_issue_codes || []).slice(0, 5).map((issue) => (
            <span key={`fixed-${issue.name}`} className="is-fixed">{issue.name} x{issue.count}</span>
          ))}
          {(impact.remaining_issue_codes || []).slice(0, 3).map((issue) => (
            <span key={`remaining-${issue.name}`} className="is-remaining">{issue.name} x{issue.count}</span>
          ))}
        </div>
      ) : null}
    </div>
  )
}

export function LibraryQualityStatusPanels({
  S,
  batchResult,
  repairImpact,
  repairRun,
  repairAdvancing,
  domains,
  reviewCount,
  readerLocateRepairCount,
  artifactOpening,
  onFocusReview,
  onRepairReaderLocateSources,
  onAdvanceRepairRun,
  onOpenArtifact,
}: LibraryQualityStatusPanelsProps) {
  return (
    <>
      <LibraryQualityBatchResultPanel result={batchResult} />
      <LibraryQualityRepairImpactPanel
        impact={repairImpact}
        run={repairRun}
        advancing={repairAdvancing}
        onAdvanceRepairRun={onAdvanceRepairRun}
      />
      <div className="kb-lib-quality-domain-section">
        <Text className="kb-lib-quality-report-section-title">{S.lib_quality_domains_title}</Text>
        <div className="kb-lib-quality-domain-grid" data-testid="library-quality-domains">
          {domains.map((domain) => {
            const artifactDomain: QualityArtifactDomain = domain.key === 'citation_cards' ? 'citation_cards' : 'research_qa'
            const artifactTarget: QualityArtifactTarget = domain.available ? 'report' : 'runbook'
            return (
              <div
                key={domain.key}
                className={`kb-lib-quality-domain-card is-${domain.status}`}
                data-quality-domain={domain.key}
              >
                <div className="kb-lib-quality-domain-head">
                  <span>{domain.label}</span>
                  <Tag color={qualityDomainTagColor(domain.status)}>
                    {domain.statusLabel}
                  </Tag>
                </div>
                <strong>{domain.countText}</strong>
                {domain.detailText ? <span>{domain.detailText}</span> : null}
                {domain.failureText ? <em>{domain.failureText}</em> : null}
                <div className="kb-lib-quality-domain-actions">
                  {domain.key === 'conversion' ? (
                    <Button
                      size="small"
                      className="kb-lib-quality-domain-action"
                      disabled={reviewCount <= 0}
                      onClick={onFocusReview}
                    >
                      {S.lib_quality_report_focus_review}
                    </Button>
                  ) : domain.key === 'reader_locate' ? (
                    <Button
                      size="small"
                      className="kb-lib-quality-domain-action"
                      disabled={readerLocateRepairCount <= 0}
                      onClick={onRepairReaderLocateSources}
                    >
                      {domain.status === 'good' ? 'Verified' : 'Repair sources'}
                    </Button>
                  ) : (
                    <>
                      <Button
                        size="small"
                        className="kb-lib-quality-domain-action"
                        loading={artifactOpening === `${artifactDomain}:${artifactTarget}`}
                        onClick={() => onOpenArtifact(artifactDomain, artifactTarget)}
                      >
                        {domain.available ? S.lib_quality_artifact_open_report : S.lib_quality_artifact_open_runbook}
                      </Button>
                      {domain.available ? (
                        <Button
                          size="small"
                          className="kb-lib-quality-domain-action"
                          loading={artifactOpening === `${artifactDomain}:folder`}
                          onClick={() => onOpenArtifact(artifactDomain, 'folder')}
                        >
                          {S.lib_quality_artifact_open_folder}
                        </Button>
                      ) : null}
                    </>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}
