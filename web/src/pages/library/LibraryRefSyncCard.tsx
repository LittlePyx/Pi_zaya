import { Progress, Typography } from 'antd'
import {
  WorkbenchMetricStrip,
  WorkbenchPanel,
  WorkbenchStatusPill,
  type WorkbenchMetricItem,
  type WorkbenchTone,
} from '../../components/library/WorkbenchPrimitives'
import './LibraryRefSyncCard.css'

const { Text } = Typography

type LibraryRefSyncCardProps = {
  title: string
  message: string
  metaText: string
  statusLabel: string
  statusTone: WorkbenchTone
  percent: number
  docsTotal: number
  running: boolean
  status: string
  metricItems: WorkbenchMetricItem[]
  queueItems: WorkbenchMetricItem[]
  error?: string
}

export function LibraryRefSyncCard({
  title,
  message,
  metaText,
  statusLabel,
  statusTone,
  percent,
  docsTotal,
  running,
  status,
  metricItems,
  queueItems,
  error,
}: LibraryRefSyncCardProps) {
  return (
    <WorkbenchPanel className="kb-lib-refsync-card">
      <div className="kb-lib-refsync-shell">
        <div className="kb-lib-refsync-head">
          <div className="kb-lib-refsync-copy">
            <Text className="kb-lib-refsync-title">{title}</Text>
            <Text type="secondary" className="kb-lib-refsync-hint">
              {message}
            </Text>
            <Text type="secondary" className="kb-lib-refsync-meta">
              {metaText}
            </Text>
          </div>
          <WorkbenchStatusPill tone={statusTone}>{statusLabel}</WorkbenchStatusPill>
        </div>
        {docsTotal > 0 ? (
          <Progress
            percent={percent}
            status={running ? 'active' : (status === 'error' ? 'exception' : 'normal')}
          />
        ) : null}
        <WorkbenchMetricStrip items={metricItems} className="kb-lib-refsync-metrics" />
        <WorkbenchMetricStrip items={queueItems} className="kb-lib-refsync-queues" />
        {error ? <Text type="danger" className="text-xs">{error}</Text> : null}
      </div>
    </WorkbenchPanel>
  )
}
