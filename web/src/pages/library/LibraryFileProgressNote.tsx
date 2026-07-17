import { Progress, Typography } from 'antd'
import type { LibraryFileItem } from '../../api/library'
import { useT } from '../../i18n'
import { conversionStageLabel, derivePageProgress, runningPagesLabel } from './libraryPageUtils'
import './LibraryFileProgressNote.css'

const { Text } = Typography

type LibraryFileProgressNoteProps = {
  item: LibraryFileItem
}

export function LibraryFileProgressNote({ item }: LibraryFileProgressNoteProps) {
  const S = useT()
  const itemProgress = derivePageProgress(item.cur_page_done, item.cur_page_total, item.cur_page_msg)
  const stageLabel = conversionStageLabel(item.conversion_stage, S)
  const remainingPagesLabel = runningPagesLabel(
    item.conversion_stage,
    item.running_pages,
    Number(item.running_page_count || 0),
    itemProgress.total,
    S,
  )
  const itemProgressPercent = itemProgress.total > 0
    ? Math.round((itemProgress.done / Math.max(1, itemProgress.total)) * 100)
    : 0

  return (
    <>
      {item.note ? <div className="kb-lib-file-note">{item.note}</div> : null}
      {item.task_state === 'running' ? (
        <div className="kb-lib-file-progress-note">
          {itemProgress.total > 0 ? (
            <>
              <Progress percent={itemProgressPercent} status="active" size="small" showInfo={false} />
              <Text type="secondary" className="text-xs">
                {`\u9875\u8fdb\u5ea6 ${itemProgress.done}/${itemProgress.total}`}
              </Text>
            </>
          ) : null}
          {stageLabel ? (
            <div>
              <Text type="secondary" className="text-xs">{stageLabel}</Text>
            </div>
          ) : null}
          {remainingPagesLabel ? (
            <div>
              <Text type="secondary" className="text-xs">{remainingPagesLabel}</Text>
            </div>
          ) : null}
        </div>
      ) : null}
    </>
  )
}
