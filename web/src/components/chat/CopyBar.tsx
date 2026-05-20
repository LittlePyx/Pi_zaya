import { Button, message } from 'antd'
import { CopyOutlined } from '@ant-design/icons'
import { useT } from '../../i18n'

interface Props {
  text: string
  markdown?: string
}

export function CopyBar({ text, markdown }: Props) {
  const S = useT()
  const copy = (value: string, doneLabel: string) => {
    navigator.clipboard.writeText(value).then(() => message.success(doneLabel))
  }

  return (
    <div className="mt-2 flex gap-1 opacity-40 transition-opacity hover:opacity-100">
      <Button
        size="small"
        type="text"
        icon={<CopyOutlined />}
        onClick={() => copy(text, S.copied_text)}
      >
        {S.copy_text}
      </Button>
      {markdown ? (
        <Button
          size="small"
          type="text"
          icon={<CopyOutlined />}
          onClick={() => copy(markdown, S.copied_markdown)}
        >
          {S.copy_md}
        </Button>
      ) : null}
    </div>
  )
}
