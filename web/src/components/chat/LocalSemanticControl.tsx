import { useMemo, useState } from 'react'
import { Button, Dropdown, Modal, Tooltip, message } from 'antd'
import type { MenuProps } from 'antd'
import {
  CloudDownloadOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  DownOutlined,
  PauseOutlined,
  PlayCircleOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import type { LocalSemanticComponentResponse } from '../../api/retrieval'
import { useT } from '../../i18n'
import type { LocalRetrievalMode } from './useLocalSemanticRetrieval'

interface Props {
  status: LocalSemanticComponentResponse | null
  preferredMode: LocalRetrievalMode
  scopeEligible: boolean
  disabled?: boolean
  busy?: boolean
  onModeChange: (mode: LocalRetrievalMode) => void
  onEnable: () => Promise<unknown>
  onResume: () => Promise<unknown>
  onCancel: () => Promise<unknown>
  onRemove: () => Promise<unknown>
}

function formatMib(bytes: number): string {
  return String(Math.max(1, Math.ceil(Math.max(0, Number(bytes || 0)) / 1_048_576)))
}

export function LocalSemanticControl({
  status,
  preferredMode,
  scopeEligible,
  disabled = false,
  busy = false,
  onModeChange,
  onEnable,
  onResume,
  onCancel,
  onRemove,
}: Props) {
  const S = useT()
  const [consentOpen, setConsentOpen] = useState(false)
  const [removeOpen, setRemoveOpen] = useState(false)
  const lifecycle = status?.hybrid_local
  const state = lifecycle?.state || 'unavailable'
  const visible = Boolean(status?.show_mode_control && status.install_available)
  const ready = state === 'ready' && lifecycle?.available === true
  const semanticInUse = ready && scopeEligible && preferredMode === 'hybrid_local'
  const progress = lifecycle?.total_bytes
    ? Math.min(100, Math.floor((lifecycle.completed_bytes / lifecycle.total_bytes) * 100))
    : 0

  const buttonLabel = useMemo(() => {
    if (state === 'preparing') {
      if (lifecycle?.reason_id === 'semantic_index_preparing' && lifecycle.total_bytes > 0) {
        return S.chat_retrieval_indexing_progress.replace('{done}', String(lifecycle.completed_bytes)).replace('{total}', String(lifecycle.total_bytes))
      }
      return progress > 0
        ? S.chat_retrieval_preparing_progress.replace('{percent}', String(progress))
        : S.chat_retrieval_preparing
    }
    if (state === 'removing') return S.chat_retrieval_removing
    if (state === 'paused') return S.chat_retrieval_paused
    if (state === 'needs_space') return S.chat_retrieval_needs_space
    if (semanticInUse) return S.chat_retrieval_semantic
    return S.chat_retrieval_bm25
  }, [S, lifecycle, progress, semanticInUse, state])

  const run = async (operation: () => Promise<unknown>) => {
    try {
      await operation()
      return true
    } catch {
      message.error(S.chat_retrieval_action_failed)
      return false
    }
  }

  const items: MenuProps['items'] = [
    {
      key: 'bm25',
      icon: <DatabaseOutlined />,
      label: (
        <div className="kb-external-source-option">
          <strong>{S.chat_retrieval_bm25}</strong>
          <span>{S.chat_retrieval_bm25_tip}</span>
        </div>
      ),
    },
  ]
  if (ready) {
    items.push({
      key: 'hybrid_local',
      icon: <ThunderboltOutlined />,
      disabled: !scopeEligible,
      label: (
        <div className="kb-external-source-option">
          <strong>{S.chat_retrieval_semantic}</strong>
          <span>{scopeEligible ? S.chat_retrieval_semantic_tip : S.chat_retrieval_scope_tip}</span>
        </div>
      ),
    })
    items.push({ type: 'divider' }, {
      key: 'remove',
      icon: <DeleteOutlined />,
      danger: true,
      label: S.chat_retrieval_remove,
    })
  } else if (state === 'not_installed') {
    items.push({
      key: 'enable',
      icon: <CloudDownloadOutlined />,
      label: (
        <div className="kb-external-source-option">
          <strong>{S.chat_retrieval_enable}</strong>
          <span>{status?.component?.network_required_for_install === false ? S.chat_retrieval_enable_tip_offline : S.chat_retrieval_enable_tip}</span>
        </div>
      ),
    })
  } else if (state === 'preparing' && lifecycle?.task_id) {
    items.push({
      key: 'cancel',
      icon: <PauseOutlined />,
      label: S.chat_retrieval_cancel,
    })
  } else if (state === 'paused') {
    items.push({ key: 'resume', icon: <PlayCircleOutlined />, label: S.chat_retrieval_resume })
    items.push({ type: 'divider' }, { key: 'remove', icon: <DeleteOutlined />, danger: true, label: S.chat_retrieval_remove })
  } else if (state === 'needs_space') {
    items.push({
      key: 'resume',
      icon: <PlayCircleOutlined />,
      label: (
        <div className="kb-external-source-option">
          <strong>{S.chat_retrieval_space_continue}</strong>
          <span>{S.chat_retrieval_space_tip.replace(
            '{size}',
            formatMib(lifecycle?.required_additional_bytes || 0),
          )}</span>
        </div>
      ),
    })
    items.push({ type: 'divider' }, { key: 'remove', icon: <DeleteOutlined />, danger: true, label: S.chat_retrieval_remove })
  } else if (state === 'bm25_available') {
    items.push({
      key: 'no_action',
      disabled: true,
      label: (
        <div className="kb-external-source-option">
          <strong>{S.chat_retrieval_auto_fallback}</strong>
          <span>{S.chat_retrieval_auto_fallback_tip}</span>
        </div>
      ),
    })
    items.push({ type: 'divider' }, { key: 'remove', icon: <DeleteOutlined />, danger: true, label: S.chat_retrieval_remove })
  }

  if (!visible) return null

  return (
    <>
      <Tooltip title={semanticInUse ? S.chat_retrieval_semantic_tip : S.chat_retrieval_bm25_tip}>
        <Dropdown
          trigger={['click']}
          disabled={disabled || busy}
          menu={{
            items,
            selectable: true,
            selectedKeys: [semanticInUse ? 'hybrid_local' : 'bm25'],
            onClick: ({ key }) => {
              if (key === 'bm25') onModeChange('bm25')
              else if (key === 'hybrid_local') onModeChange('hybrid_local')
              else if (key === 'enable') setConsentOpen(true)
              else if (key === 'cancel') void run(onCancel)
              else if (key === 'resume') void run(onResume)
              else if (key === 'remove') setRemoveOpen(true)
            },
          }}
        >
          <Button
            className={`kb-external-source-btn ${semanticInUse ? 'is-model' : ''}`}
            icon={semanticInUse ? <ThunderboltOutlined /> : <DatabaseOutlined />}
            loading={busy}
            data-testid="chat-local-retrieval-control"
            aria-label={`${S.chat_retrieval_label}: ${buttonLabel}`}
          >
            {buttonLabel}
            <DownOutlined className="kb-external-source-chevron" />
          </Button>
        </Dropdown>
      </Tooltip>
      <Modal
        open={consentOpen}
        title={S.chat_retrieval_consent_title}
        okText={status?.component.network_required_for_install === false ? S.chat_retrieval_consent_bundled_confirm : S.chat_retrieval_consent_confirm}
        cancelText={S.cancel}
        confirmLoading={busy}
        onCancel={() => setConsentOpen(false)}
        onOk={() => {
          void run(onEnable).then((completed) => {
            if (completed) setConsentOpen(false)
          })
        }}
      >
        <div className="kb-local-retrieval-consent" data-testid="local-semantic-consent">
          <p>{status?.component.network_required_for_install === false ? S.chat_retrieval_consent_bundled_intro : S.chat_retrieval_consent_intro}</p>
          <ul>
            <li>{(status?.component.network_required_for_install === false ? S.chat_retrieval_consent_bundled : S.chat_retrieval_consent_download).replace('{size}', formatMib(status?.component.archive_bytes || 0))}</li>
            <li>{S.chat_retrieval_consent_disk.replace('{size}', formatMib(status?.component.installed_bytes || 0))}</li>
            <li>{S.chat_retrieval_consent_local}</li>
            <li>{S.chat_retrieval_consent_external}</li>
          </ul>
        </div>
      </Modal>
      <Modal
        open={removeOpen}
        title={S.chat_retrieval_remove_title}
        okText={S.chat_retrieval_remove_confirm}
        cancelText={S.cancel}
        okButtonProps={{ danger: true }}
        confirmLoading={busy}
        onCancel={() => setRemoveOpen(false)}
        onOk={() => {
          void run(onRemove).then((completed) => {
            if (completed) setRemoveOpen(false)
          })
        }}
      >
        <p>{S.chat_retrieval_remove_tip}</p>
      </Modal>
    </>
  )
}
