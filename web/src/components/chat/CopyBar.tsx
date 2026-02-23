import { Button, message } from 'antd'
import { CopyOutlined } from '@ant-design/icons'
import { S } from '../../i18n/zh'

export function CopyBar({ content }: { content: string }) {
  const copy = () => {
    navigator.clipboard.writeText(content).then(() => message.success('已复制'))
  }

  return (
    <div className="flex gap-1 mt-2 opacity-40 hover:opacity-100 transition-opacity">
      <Button size="small" type="text" icon={<CopyOutlined />} onClick={copy}>
        {S.copy_text}
      </Button>
    </div>
  )
}
