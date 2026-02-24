import { useEffect, useRef } from 'react'
import { Typography } from 'antd'
import { UserOutlined, RobotOutlined } from '@ant-design/icons'
import { MarkdownRenderer } from './MarkdownRenderer'
import { CopyBar } from './CopyBar'
import type { Message } from '../../api/chat'

const { Text } = Typography

interface Props {
  messages: Message[]
  generationPartial?: string
  generationStage?: string
}

export function MessageList({ messages, generationPartial, generationStage }: Props) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, generationPartial])

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
      {messages.map(m => (
        <div
          key={m.id}
          className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : ''}`}
        >
          {m.role !== 'user' && (
            <div className="w-7 h-7 rounded-full bg-[var(--accent)] flex items-center justify-center shrink-0 mt-1">
              <RobotOutlined className="text-white text-xs" />
            </div>
          )}
          <div
            className={`max-w-[75%] rounded-xl px-4 py-3 ${
              m.role === 'user'
                ? 'bg-[var(--msg-user-bg)]'
                : 'bg-[var(--msg-ai-bg)]'
            }`}
          >
            {m.role === 'user' ? (
              <Text>{m.content}</Text>
            ) : (
              <>
                <MarkdownRenderer content={m.content} />
                <CopyBar content={m.content} />
              </>
            )}
          </div>
          {m.role === 'user' && (
            <div className="w-7 h-7 rounded-full bg-[var(--msg-user-bg)] flex items-center justify-center shrink-0 mt-1 border border-[var(--border)]">
              <UserOutlined className="text-xs" />
            </div>
          )}
        </div>
      ))}

      {generationPartial !== undefined && generationPartial !== null && (
        <div className="flex gap-3">
          <div className="w-7 h-7 rounded-full bg-[var(--accent)] flex items-center justify-center shrink-0 mt-1">
            <RobotOutlined className="text-white text-xs" />
          </div>
          <div className="max-w-[75%] rounded-xl px-4 py-3 bg-[var(--msg-ai-bg)]">
            {generationStage && (
              <div className="flex items-center gap-2 mb-2">
                <span className="inline-block w-2 h-2 rounded-full bg-[var(--accent)] animate-pulse" />
                <Text type="secondary" className="text-xs">{generationStage}</Text>
              </div>
            )}
            {generationPartial ? (
              <MarkdownRenderer content={generationPartial} />
            ) : (
              <div className="flex items-center gap-1 py-1">
                <span className="typing-dot" />
                <span className="typing-dot" style={{ animationDelay: '0.15s' }} />
                <span className="typing-dot" style={{ animationDelay: '0.3s' }} />
              </div>
            )}
          </div>
        </div>
      )}

      <div ref={endRef} />
    </div>
  )
}
