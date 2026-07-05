import { type ReactNode } from 'react'
import { Button, Card, Typography } from 'antd'
import './LibraryQualityCenter.css'

const { Text } = Typography

type QualityReportStats = {
  assessed: number
  converted: number
  review: number
  avgScore: number
}

type QualityCenterSignal = {
  key: string
  label: string
  value: string
}

type LibraryQualityCenterProps = {
  children: ReactNode
  open: boolean
  tone: string
  S: Record<string, string>
  stats: QualityReportStats
  statusLabel: string
  nextAction: string
  summary: string
  signals: QualityCenterSignal[]
  recommendedRepairCount: number
  recommendedRepairBusy: boolean
  onToggleOpen: () => void
  onFocusReview: () => void
  onRepairRecommended: () => void
}

export function LibraryQualityCenter({
  children,
  open,
  tone,
  S,
  stats,
  statusLabel,
  nextAction,
  summary,
  signals,
  recommendedRepairCount,
  recommendedRepairBusy,
  onToggleOpen,
  onFocusReview,
  onRepairRecommended,
}: LibraryQualityCenterProps) {
  return (
    <Card
      size="small"
      className={`kb-lib-card kb-lib-quality-report is-${tone}${open ? ' is-open' : ' is-compact'}`}
      data-testid="library-quality-report"
    >
      <div className="kb-lib-quality-report-head">
        <div className="kb-lib-quality-report-copy">
          <Text className="kb-lib-quality-report-title">{S.lib_quality_report_title}</Text>
          <Text type="secondary" className="kb-lib-quality-report-hint">
            {S.lib_quality_report_hint
              .replace('{assessed}', String(stats.assessed))
              .replace('{converted}', String(stats.converted))
              .replace('{review}', String(stats.review))
              .replace('{avg}', String(stats.avgScore))}
          </Text>
        </div>
        <div className="kb-lib-quality-report-actions">
          <Button
            size="small"
            className="kb-lib-action-quiet"
            data-testid="library-quality-center-toggle"
            onClick={onToggleOpen}
          >
            {open ? S.lib_quality_center_toggle_hide : S.lib_quality_center_toggle_show}
          </Button>
          {stats.review > 0 ? (
            <Button
              size="small"
              className="kb-lib-action-quiet"
              data-testid="library-quality-report-focus-review"
              onClick={onFocusReview}
            >
              {S.lib_quality_report_focus_review}
            </Button>
          ) : null}
          {recommendedRepairCount > 0 ? (
            <Button
              size="small"
              type="primary"
              loading={recommendedRepairBusy}
              data-testid="library-quality-report-repair-recommended"
              onClick={onRepairRecommended}
            >
              {S.lib_quality_report_repair_top.replace('{n}', String(recommendedRepairCount))}
            </Button>
          ) : null}
        </div>
      </div>

      <div className="kb-lib-quality-center-summary" data-testid="library-quality-center-summary">
        <div className="kb-lib-quality-center-state">
          <span className={`kb-lib-quality-center-status is-${tone}`}>{statusLabel}</span>
          <strong>{nextAction}</strong>
          <p>{summary}</p>
        </div>
        <div className="kb-lib-quality-center-signals">
          {signals.map((item) => (
            <span key={item.key} className={`kb-lib-quality-center-signal is-${item.key}`}>
              <em>{item.label}</em>
              <strong>{item.value}</strong>
            </span>
          ))}
        </div>
      </div>

      {children}
    </Card>
  )
}
