import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import { Input, Button, Typography } from 'antd'
import { SendOutlined, PauseOutlined } from '@ant-design/icons'
import { S } from '../../i18n/zh'

const { Text } = Typography

const { TextArea } = Input

interface Props {
  onSend: (text: string) => void
  onStop: () => void
  generating: boolean
}

export function ChatInput({ onSend, onStop, generating }: Props) {
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)
  const composingRef = useRef(false)

  useEffect(() => {
    if (!generating) ref.current?.focus()
  }, [generating])

  const submit = () => {
    const t = text.trim()
    if (!t || generating) return
    onSend(t)
    setText('')
  }

  const onKey = (e: KeyboardEvent) => {
    if (composingRef.current) return
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="p-4 border-t border-[var(--border)] bg-[var(--panel)]">
      <div className="flex gap-2 items-end max-w-4xl mx-auto">
        <TextArea
          ref={ref as never}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={onKey}
          onCompositionStart={() => { composingRef.current = true }}
          onCompositionEnd={() => { composingRef.current = false }}
          placeholder={S.prompt_label}
          autoSize={{ minRows: 1, maxRows: 6 }}
          className="flex-1"
          autoFocus
        />
        {generating ? (
          <Button icon={<PauseOutlined />} onClick={onStop} danger>{S.stop}</Button>
        ) : (
          <Button type="primary" icon={<SendOutlined />} onClick={submit} disabled={!text.trim()}>
            {S.send}
          </Button>
        )}
      </div>
      <div className="text-center mt-1.5">
        <Text type="secondary" className="text-xs">
          Enter {S.send} · Shift+Enter {S.newline}
        </Text>
      </div>
    </div>
  )
}
