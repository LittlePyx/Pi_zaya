import { Button, Dropdown, Tooltip } from 'antd'
import type { MenuProps } from 'antd'
import type { ReactNode } from 'react'
import { BulbOutlined, DatabaseOutlined, DownOutlined, GlobalOutlined } from '@ant-design/icons'
import type { ExternalSupplementMode } from '../../api/chat'
import { useT } from '../../i18n'

interface Props {
  value: ExternalSupplementMode
  disabled?: boolean
  onChange: (mode: ExternalSupplementMode) => void
}

export function ExternalSupplementControl({ value, disabled = false, onChange }: Props) {
  const S = useT()
  const meta = {
    disabled: {
      label: S.chat_external_disabled,
      detail: S.chat_external_disabled_tip,
      icon: <DatabaseOutlined />,
    },
    model: {
      label: S.chat_external_model,
      detail: S.chat_external_model_tip,
      icon: <BulbOutlined />,
    },
    web: {
      label: S.chat_external_web,
      detail: S.chat_external_web_tip,
      icon: <GlobalOutlined />,
    },
  } satisfies Record<ExternalSupplementMode, { label: string; detail: string; icon: ReactNode }>
  const selected = meta[value]
  const items: MenuProps['items'] = (Object.keys(meta) as ExternalSupplementMode[]).map((mode) => ({
    key: mode,
    icon: meta[mode].icon,
    label: (
      <div className="kb-external-source-option">
        <strong>{meta[mode].label}</strong>
        <span>{meta[mode].detail}</span>
      </div>
    ),
  }))

  return (
    <Tooltip title={`${S.chat_external_label} — ${selected.detail}`}>
      <Dropdown
        trigger={['click']}
        disabled={disabled}
        menu={{
          items,
          selectable: true,
          selectedKeys: [value],
          onClick: ({ key }) => onChange(key as ExternalSupplementMode),
        }}
      >
        <Button
          className={`kb-external-source-btn is-${value}`}
          icon={selected.icon}
          disabled={disabled}
          data-testid="chat-external-supplement-control"
          aria-label={`${S.chat_external_label}: ${selected.label}`}
        >
          {selected.label}
          <DownOutlined className="kb-external-source-chevron" />
        </Button>
      </Dropdown>
    </Tooltip>
  )
}
