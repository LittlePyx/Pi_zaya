import { Button, Card, Typography } from 'antd'
import {
  formatQualityRepairHistoryTime,
  stripKnownSourceExt,
  type QualityRepairHistoryRecord,
} from './libraryPageUtils'
import './LibraryQualityHistoryPanel.css'

const { Text } = Typography

type QualityRepairHistoryStats = {
  total: number
  fixedCount: number
  improved: number
  avgDelta: number
}

type LibraryQualityHistoryPanelProps = {
  visible: boolean
  S: Record<string, string>
  records: QualityRepairHistoryRecord[]
  stats: QualityRepairHistoryStats
  remainingNames: string[]
  recommendedNames: string[]
  repairingNames: Record<string, boolean>
  focusNames: string[]
  onFocusRemaining: () => void
  onRepairRecommended: () => void
  onClearFocus: () => void
  onOpenRecord: (name: string) => void
}

export function LibraryQualityHistoryPanel({
  visible,
  S,
  records,
  stats,
  remainingNames,
  recommendedNames,
  repairingNames,
  focusNames,
  onFocusRemaining,
  onRepairRecommended,
  onClearFocus,
  onOpenRecord,
}: LibraryQualityHistoryPanelProps) {
  if (!visible || records.length <= 0) return null

  return (
    <Card size="small" className="kb-lib-card kb-lib-quality-history" data-testid="library-quality-history">
      <div className="kb-lib-quality-history-head">
        <div>
          <Text className="kb-lib-quality-history-title">{S.lib_quality_history_title}</Text>
          <Text type="secondary" className="kb-lib-quality-history-hint">
            {S.lib_quality_history_hint
              .replace('{n}', String(stats.total))
              .replace('{delta}', String(stats.avgDelta >= 0 ? `+${stats.avgDelta}` : stats.avgDelta))
              .replace('{issues}', String(stats.fixedCount))}
          </Text>
        </div>
        <div className="kb-lib-quality-history-side">
          <div className="kb-lib-quality-history-metrics">
            <span data-testid="library-quality-history-count">{S.lib_quality_history_count.replace('{n}', String(stats.total))}</span>
            <span>{S.lib_quality_history_improved.replace('{n}', String(stats.improved))}</span>
          </div>
          <div className="kb-lib-quality-history-actions">
            <Button
              size="small"
              className="kb-lib-action-quiet"
              disabled={remainingNames.length <= 0}
              data-testid="library-quality-history-focus-remaining"
              onClick={onFocusRemaining}
            >
              {S.lib_quality_history_focus_remaining}
            </Button>
            <Button
              size="small"
              type="primary"
              disabled={recommendedNames.length <= 0}
              loading={recommendedNames.some((name) => Boolean(repairingNames[name]))}
              data-testid="library-quality-history-repair-recommended"
              onClick={onRepairRecommended}
            >
              {S.lib_quality_history_repair_recommended.replace('{n}', String(recommendedNames.length))}
            </Button>
            {focusNames.length > 0 ? (
              <Button
                size="small"
                className="kb-lib-action-quiet"
                data-testid="library-quality-history-clear-focus"
                onClick={onClearFocus}
              >
                {S.lib_quality_history_clear_focus}
              </Button>
            ) : null}
          </div>
        </div>
      </div>
      <div className="kb-lib-quality-history-list">
        {records.slice(0, 4).map((record) => (
          <div key={`${record.name}-${record.updatedAt}`} className="kb-lib-quality-history-row" data-testid="library-quality-history-row">
            <button
              type="button"
              className="kb-lib-quality-history-paper"
              title={record.name}
              data-testid="library-quality-history-paper"
              onClick={() => onOpenRecord(record.name)}
            >
              {stripKnownSourceExt(record.name) || record.name}
            </button>
            <div className="kb-lib-quality-history-result">
              <span className="kb-lib-quality-history-score">Q{record.beforeScore} -&gt; Q{record.afterScore}</span>
              {record.fixedIssues.length > 0 ? (
                <span className="kb-lib-quality-history-fixed">
                  {S.lib_quality_history_fixed.replace('{issues}', record.fixedIssues.slice(0, 2).join(' / '))}
                </span>
              ) : null}
              {record.remainingIssues.length > 0 ? (
                <span className="kb-lib-quality-history-remaining">
                  {S.lib_quality_history_remaining.replace('{n}', String(record.remainingIssues.length))}
                </span>
              ) : null}
            </div>
            <div className="kb-lib-quality-history-time">{formatQualityRepairHistoryTime(record.updatedAt)}</div>
          </div>
        ))}
      </div>
    </Card>
  )
}
