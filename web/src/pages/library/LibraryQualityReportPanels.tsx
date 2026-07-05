import { Typography } from 'antd'
import { stripKnownSourceExt } from './libraryPageUtils'
import './LibraryQualityReportPanels.css'

const { Text } = Typography

type QualityIssueStatView = {
  key: string
  label: string
  severity: string
  papers: number
  count: number
  repairStrategy?: string
}

type QualityReportRecommendationView = {
  name: string
  score: number
  issues: string[]
}

type LibraryQualityReportPanelsProps = {
  S: Record<string, string>
  issues: QualityIssueStatView[]
  recommendations: QualityReportRecommendationView[]
  onFocusIssue: (label: string) => void
  onFocusRecommendation: (name: string) => void
}

export function LibraryQualityReportPanels({
  S,
  issues,
  recommendations,
  onFocusIssue,
  onFocusRecommendation,
}: LibraryQualityReportPanelsProps) {
  return (
    <div className="kb-lib-quality-report-body">
      <div className="kb-lib-quality-report-section">
        <Text className="kb-lib-quality-report-section-title">{S.lib_quality_report_top_issues}</Text>
        {issues.length > 0 ? (
          <div className="kb-lib-quality-report-issues">
            {issues.map((issue) => (
              <button
                key={issue.key}
                type="button"
                className={`kb-lib-quality-report-issue is-${issue.severity || 'warning'}`}
                data-testid="library-quality-report-issue"
                onClick={() => onFocusIssue(issue.label)}
              >
                <span>{issue.label}</span>
                {issue.repairStrategy ? <em>{issue.repairStrategy}</em> : null}
                <strong>{S.lib_quality_report_papers.replace('{n}', String(issue.papers))}</strong>
              </button>
            ))}
          </div>
        ) : (
          <Text type="secondary" className="kb-lib-quality-report-empty">{S.lib_quality_report_no_issues}</Text>
        )}
      </div>
      <div className="kb-lib-quality-report-section">
        <Text className="kb-lib-quality-report-section-title">{S.lib_quality_report_recommended}</Text>
        {recommendations.length > 0 ? (
          <div className="kb-lib-quality-report-recommendations" data-testid="library-quality-report-recommended">
            {recommendations.slice(0, 3).map((item) => (
              <button
                key={item.name}
                type="button"
                className="kb-lib-quality-report-recommendation"
                onClick={() => onFocusRecommendation(item.name)}
              >
                <span className="kb-lib-quality-report-rec-title">{stripKnownSourceExt(item.name) || item.name}</span>
                <span className="kb-lib-quality-report-rec-meta">
                  Q{item.score}
                  {item.issues.length > 0 ? ` · ${item.issues.join(' / ')}` : ''}
                </span>
              </button>
            ))}
          </div>
        ) : (
          <Text type="secondary" className="kb-lib-quality-report-empty">{S.lib_quality_report_no_issues}</Text>
        )}
      </div>
    </div>
  )
}
