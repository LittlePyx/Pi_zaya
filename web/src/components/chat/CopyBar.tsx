import { Button, message } from 'antd'
import { CopyOutlined } from '@ant-design/icons'
import { S } from '../../i18n/zh'

interface Props {
  text: string
  markdown?: string
}

export function CopyBar({ text, markdown }: Props) {
  const copy = (value: string, doneLabel: string) => {
    navigator.clipboard.writeText(value).then(() => message.success(doneLabel))
  }

  return (
    <div className="mt-2 flex gap-1 opacity-40 transition-opacity hover:opacity-100">
      <Button
        size="small"
        type="text"
        icon={<CopyOutlined />}
        onClick={() => copy(text, '已复制文本')}
      >
        {S.copy_text}
      </Button>
      {markdown ? (
        <Button
          size="small"
          type="text"
          icon={<CopyOutlined />}
          onClick={() => copy(markdown, '已复制 Markdown')}
        >
          Copy Markdown
        </Button>
      ) : null}
    </div>
  )
}
