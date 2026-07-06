import { Button, Card, Progress, Tag, Typography } from 'antd'
import { StopOutlined } from '@ant-design/icons'
import './LibraryStickyStatus.css'

const { Text } = Typography

type StickyConvertProgress = {
  completed: number
  total: number
  current: string
}

type StickyPageProgress = {
  done: number
  total: number
}

type LibraryStickyStatusProps = {
  convertRunning: boolean
  convertProgress: StickyConvertProgress | null
  convertTitle: string
  convertActiveSummary: string
  convertStageLabel: string
  convertPageLabel: string
  convertPageProgress: StickyPageProgress
  convertPercent: number
  convertPagePercent: number
  stopLabel: string
  refSyncRunning: boolean
  refSyncTitle: string
  refSyncMessage: string
  refSyncPercent: number
  refSyncRunningLabel: string
  onStopConvert: () => void | Promise<unknown>
}

export function LibraryStickyStatus({
  convertRunning,
  convertProgress,
  convertTitle,
  convertActiveSummary,
  convertStageLabel,
  convertPageLabel,
  convertPageProgress,
  convertPercent,
  convertPagePercent,
  stopLabel,
  refSyncRunning,
  refSyncTitle,
  refSyncMessage,
  refSyncPercent,
  refSyncRunningLabel,
  onStopConvert,
}: LibraryStickyStatusProps) {
  if ((!convertRunning || !convertProgress) && !refSyncRunning) return null

  return (
    <Card size="small" className="kb-lib-card kb-lib-sticky-status">
      <div className="kb-lib-sticky-wrap">
        {convertRunning && convertProgress ? (
          <div className="kb-lib-sticky-item">
            <div className="kb-lib-sticky-main">
              <Text className="kb-lib-sticky-title">{convertTitle}</Text>
              {convertProgress.current ? <Text type="secondary" className="kb-lib-sticky-sub">{convertProgress.current}</Text> : null}
              {convertActiveSummary ? <Text type="secondary" className="kb-lib-sticky-sub">{convertActiveSummary}</Text> : null}
              {convertStageLabel ? <Text type="secondary" className="kb-lib-sticky-sub">{convertStageLabel}</Text> : null}
              {convertPageProgress.total > 0 ? (
                <Text type="secondary" className="kb-lib-sticky-sub">
                  {convertPageLabel} {convertPageProgress.done}/{convertPageProgress.total}
                </Text>
              ) : null}
            </div>
            <div className="kb-lib-sticky-progress-stack">
              <Progress className="kb-lib-sticky-progress" percent={convertPercent} status="active" size="small" />
              {convertPageProgress.total > 0 ? (
                <Progress className="kb-lib-sticky-progress kb-lib-sticky-progress-inner" percent={convertPagePercent} status="active" size="small" />
              ) : null}
            </div>
            <Button size="small" danger icon={<StopOutlined />} onClick={() => { void onStopConvert() }}>
              {stopLabel}
            </Button>
          </div>
        ) : null}

        {refSyncRunning ? (
          <div className="kb-lib-sticky-item">
            <div className="kb-lib-sticky-main">
              <Text className="kb-lib-sticky-title">{refSyncTitle}</Text>
              <Text type="secondary" className="kb-lib-sticky-sub">
                {refSyncMessage}
              </Text>
            </div>
            <Progress className="kb-lib-sticky-progress" percent={refSyncPercent} status="active" size="small" />
            <Tag color="processing">{refSyncRunningLabel}</Tag>
          </div>
        ) : null}
      </div>
    </Card>
  )
}
