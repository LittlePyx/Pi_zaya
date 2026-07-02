import { Typography } from 'antd'
import type { ChatImageAttachment } from '../../api/chat'

const { Text } = Typography

interface UserMessageBubbleProps {
  content: string
  imageAttachments: ChatImageAttachment[]
  showText: boolean
  imageOnly: boolean
}

export function UserMessageBubble({
  content,
  imageAttachments,
  showText,
  imageOnly,
}: UserMessageBubbleProps) {
  return (
    <div className={`kb-msg-bubble kb-msg-bubble-user ${imageOnly ? 'is-image-only' : ''}`}>
      {imageAttachments.length > 0 ? (
        <div
          className={`${
            showText ? 'mb-3' : ''
          } grid ${
            imageAttachments.length === 1 ? 'max-w-[18rem] grid-cols-1' : 'max-w-[38rem] grid-cols-2 sm:grid-cols-3'
          } gap-2`}
        >
          {imageAttachments.map((item) => {
            const src = String(item.url || '').trim()
            const key = `${item.sha1 || item.path}-${item.name}`
            const frameClass = 'block overflow-hidden rounded-2xl border border-[var(--border)] bg-white/70'
            if (src) {
              return (
                <a
                  key={key}
                  href={src}
                  target="_blank"
                  rel="noreferrer"
                  className={frameClass}
                >
                  <img
                    src={src}
                    alt={item.name}
                    className="block h-32 w-full object-cover"
                    loading="lazy"
                  />
                </a>
              )
            }
            return (
              <div key={key} className={frameClass}>
                <div className="flex h-32 items-center justify-center px-3 text-center text-xs text-black/45">
                  {item.name}
                </div>
              </div>
            )
          })}
        </div>
      ) : null}
      {showText ? (
        <Text className="whitespace-pre-wrap">{content}</Text>
      ) : null}
    </div>
  )
}
