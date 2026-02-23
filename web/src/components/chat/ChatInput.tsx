import { useState, useRef, type KeyboardEvent } from 'react'
import { Input, Button } from 'antd'
import { SendOutlined, PauseOutlined } from '@ant-design/icons'
import { S } from '../../i18n/zh'

const { TextArea } = Input

interface Props {
  onSend: (text: string) => void
  onStop: () => void
  generating: boolean
}

export function ChatInput({ onSend, onStop, generating }: Props) {
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  const submit = () => {
    const t = text.trim()
    if (!t || generating) return
    onSend(t)
    setText('')
  }

  const onKey = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="flex gap-2 items-end p-4 border-t border-[var(--border)] bg-[var(--panel)]">
      <TextArea
        ref={ref as never}
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={onKey}
        placeholder={S.prompt_label}
        autoSize={{ minRows: 1, maxRows: 6 }}
        className="flex-1"
      />
      {generating ? (
        <Button icon={<PauseOutlined />} onClick={onStop} danger>{S.stop}</Button>
      ) : (
        <Button type="primary" icon={<SendOutlined />} onClick={submit} disabled={!text.trim()}>
          {S.send}
        </Button>
      )}
    </div>
  )
}
