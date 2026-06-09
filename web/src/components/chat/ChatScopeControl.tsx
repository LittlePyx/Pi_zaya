import { Segmented, Tooltip } from 'antd'
import type { QueryScope } from '../../api/chat'
import { useT } from '../../i18n'

interface Props {
  value: QueryScope
  options: Array<{ value: QueryScope; disabled?: boolean }>
  onChange: (scope: QueryScope) => void
}

export function ChatScopeControl({ value, options, onChange }: Props) {
  const S = useT()
  const enabledOptions = options.filter((item) => !item.disabled)

  if (enabledOptions.length <= 1) return null

  const scopeMeta = {
    current_paper: {
      label: S.chat_scope_current_paper,
      title: S.chat_scope_current_paper_tip,
    },
    basket: {
      label: S.chat_scope_basket,
      title: S.chat_scope_basket_tip,
    },
    library: {
      label: S.chat_scope_library,
      title: S.chat_scope_library_tip,
    },
  } satisfies Record<QueryScope, { label: string; title: string }>
  const enabledScopeValues = new Set(enabledOptions.map((item) => item.value))
  const segmentedValue = enabledScopeValues.has(value) ? value : enabledOptions[0].value
  const segmentedOptions = enabledOptions.map((item) => ({
    value: item.value,
    label: (
      <Tooltip title={scopeMeta[item.value].title}>
        <span>{scopeMeta[item.value].label}</span>
      </Tooltip>
    ),
  }))

  return (
    <div className="kb-chat-scope-control" aria-label={S.chat_scope_label}>
      <span className="kb-chat-scope-label">{S.chat_scope_label}</span>
      <Segmented
        size="small"
        value={segmentedValue}
        options={segmentedOptions}
        onChange={(nextValue) => {
          const next = nextValue as QueryScope
          if (!enabledScopeValues.has(next)) return
          onChange(next)
        }}
      />
    </div>
  )
}
