import { Button } from 'antd'
import './LibraryQualityOverviewPanels.css'

type QualityReportStats = {
  good: number
  review: number
  unknown: number
  avgScore: number
}

type QualitySourceReadinessStats = {
  ready: number
  autofixed: number
  blocked: number
}

type LibraryQualityOverviewPanelsProps = {
  S: Record<string, string>
  scanRunning: boolean
  repairDisabled: boolean
  reportStats: QualityReportStats
  sourceReadinessStats: QualitySourceReadinessStats
  onScanSource: () => void
  onSafeRepairAll: () => void
  onFocusReview: () => void
}

export function LibraryQualityOverviewPanels({
  S,
  scanRunning,
  repairDisabled,
  reportStats,
  sourceReadinessStats,
  onScanSource,
  onSafeRepairAll,
  onFocusReview,
}: LibraryQualityOverviewPanelsProps) {
  return (
    <>
      <div className="kb-lib-quality-center-tools">
        <Button
          size="small"
          className="kb-lib-action-quiet"
          loading={scanRunning}
          data-testid="library-quality-scan-source"
          onClick={onScanSource}
        >
          Source scan
        </Button>
        <Button
          size="small"
          className="kb-lib-action-quiet"
          loading={scanRunning}
          disabled={repairDisabled}
          data-testid="library-quality-safe-repair-all"
          onClick={onSafeRepairAll}
        >
          Safe repair all
        </Button>
      </div>
      <div className="kb-lib-quality-report-metrics">
        <span className="kb-lib-quality-report-metric is-source-ready" data-testid="library-quality-report-source-ready">
          <span>{S.lib_quality_report_source_ready}</span>
          <strong>{sourceReadinessStats.ready}</strong>
        </span>
        <span className="kb-lib-quality-report-metric is-autofixed" data-testid="library-quality-report-autofixed">
          <span>{S.lib_quality_report_autofixed}</span>
          <strong>{sourceReadinessStats.autofixed}</strong>
        </span>
        <button
          type="button"
          className="kb-lib-quality-report-metric is-blocked"
          disabled={sourceReadinessStats.blocked <= 0}
          data-testid="library-quality-report-blocked"
          onClick={onFocusReview}
        >
          <span>{S.lib_quality_report_blocked}</span>
          <strong>{sourceReadinessStats.blocked}</strong>
        </button>
        <span className="kb-lib-quality-report-metric is-good" data-testid="library-quality-report-good">
          <span>{S.lib_quality_report_good}</span>
          <strong>{reportStats.good}</strong>
        </span>
        <button
          type="button"
          className="kb-lib-quality-report-metric is-review"
          disabled={reportStats.review <= 0}
          data-testid="library-quality-report-review"
          onClick={onFocusReview}
        >
          <span>{S.lib_quality_report_review}</span>
          <strong>{reportStats.review}</strong>
        </button>
        <span className="kb-lib-quality-report-metric is-unknown" data-testid="library-quality-report-unknown">
          <span>{S.lib_quality_report_unknown}</span>
          <strong>{reportStats.unknown}</strong>
        </span>
        <span className="kb-lib-quality-report-metric is-score" data-testid="library-quality-report-avg">
          <span>{S.lib_quality_report_avg.replace('{score}', String(reportStats.avgScore))}</span>
        </span>
      </div>
    </>
  )
}
