import { Progress, Tooltip, Typography } from 'antd'
import type { LibraryFileItem } from '../../api/library'
import { useT } from '../../i18n'
import { conversionStageLabel, derivePageProgress, formatSeconds, runningPagesLabel } from './libraryPageUtils'
import './LibraryFileProgressNote.css'

const { Text } = Typography

type LibraryFileProgressNoteProps = {
  item: LibraryFileItem
}

function conversionResultLabel(item: LibraryFileItem, S: Record<string, string>): string {
  const result = item.last_conversion
  if (!result) return ''
  if (result.outcome === 'success') {
    return result.operation === 'index_retry'
      ? S.lib_conversion_result_index_retry_success
      : S.lib_conversion_result_success
  }
  if (result.outcome === 'cancelled') return S.lib_conversion_result_cancelled
  if (result.outcome === 'quality_blocked') return S.lib_conversion_result_quality_blocked
  if (result.outcome === 'index_failed') return S.lib_conversion_result_index_failed
  return S.lib_conversion_result_failed
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
  const result = item.task_state === 'idle' ? item.last_conversion : null
  const resultLabel = conversionResultLabel(item, S)
  const resultType = result?.outcome === 'success'
    ? 'success'
    : result?.outcome === 'cancelled'
      ? 'secondary'
      : result?.outcome === 'quality_blocked'
        ? 'warning'
        : 'danger'
  const resultDuration = result && Number(result.duration_s || 0) > 0
    ? S.lib_conversion_result_duration.replace('{seconds}', formatSeconds(Number(result.duration_s || 0)))
    : ''
  const resultTime = result && Number(result.finished_at || 0) > 0
    ? new Date(Number(result.finished_at) * 1000).toLocaleString()
    : ''
  const resultTitle = result
    ? [resultLabel, String(result.detail || result.message || '').trim()].filter(Boolean).join('\n')
    : ''

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
      {result && resultLabel ? (
        <Tooltip title={resultTitle} placement="bottomLeft">
          <div
            className={`kb-lib-file-result-note is-${result.outcome}`}
            data-testid="library-conversion-result"
          >
            <Text type={resultType} className="text-xs">
              {[resultLabel, resultDuration, resultTime].filter(Boolean).join(' · ')}
            </Text>
          </div>
        </Tooltip>
      ) : null}
    </>
  )
}
