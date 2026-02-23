import { Collapse, Tag, Typography } from 'antd'
import { S } from '../../i18n/zh'

const { Text } = Typography

interface RefHit {
  meta?: { source_path?: string; top_heading?: string; page_start?: number }
  text?: string
  score?: number
}

interface Props {
  refs: Record<string, unknown>
  msgId: number
}

export function RefsPanel({ refs, msgId }: Props) {
  const entry = refs[String(msgId)] as { hits?: RefHit[] } | undefined
  const hits = entry?.hits
  if (!hits?.length) return null

  return (
    <Collapse
      size="small"
      className="mt-2"
      items={[{
        key: '1',
        label: <Text type="secondary" className="text-xs">{S.refs} ({hits.length})</Text>,
        children: (
          <div className="space-y-2">
            {hits.map((h, i) => {
              const m = h.meta || {}
              const name = m.source_path?.split('/').pop() || '未知'
              return (
                <div key={i} className="text-xs p-2 rounded bg-[var(--bg)] border border-[var(--border)]">
                  <div className="flex items-center gap-2 mb-1">
                    <Tag color="blue" className="!text-xs">#{i + 1}</Tag>
                    <Text strong className="!text-xs truncate">{name}</Text>
                    {m.top_heading && <Text type="secondary" className="!text-xs truncate">{m.top_heading}</Text>}
                    {h.score != null && <Tag className="!text-xs ml-auto">{(h.score as number).toFixed(1)}</Tag>}
                  </div>
                  {h.text && (
                    <Text type="secondary" className="!text-xs line-clamp-3">{h.text.slice(0, 200)}</Text>
                  )}
                </div>
              )
            })}
          </div>
        ),
      }]}
    />
  )
}
