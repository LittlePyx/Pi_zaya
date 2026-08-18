import { ApiOutlined, CloseOutlined } from '@ant-design/icons'
import { Button } from 'antd'
import { useState } from 'react'
import { useT } from '../../i18n'

const DISMISSED_STORAGE_KEY = 'kb_first_run_api_guide_dismissed_v1'

function readDismissed() {
  try {
    return window.localStorage.getItem(DISMISSED_STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

function persistDismissed() {
  try {
    window.localStorage.setItem(DISMISSED_STORAGE_KEY, '1')
  } catch {
    // A restricted browser can still dismiss the guide for this session.
  }
}

export function FirstRunApiGuide({
  visible,
  onConfigure,
}: {
  visible: boolean
  onConfigure: () => void
}) {
  const S = useT()
  const [dismissed, setDismissed] = useState(readDismissed)

  if (!visible || dismissed) return null

  const dismiss = () => {
    persistDismissed()
    setDismissed(true)
  }

  return (
    <section
      className="kb-first-run-api-guide"
      data-testid="first-run-api-guide"
      aria-label={S.first_run_api_guide_title}
    >
      <div className="kb-first-run-api-guide-head">
        <span className="kb-first-run-api-guide-icon" aria-hidden="true">
          <ApiOutlined />
        </span>
        <div className="kb-first-run-api-guide-heading">
          <strong>{S.first_run_api_guide_title}</strong>
          <span>{S.first_run_api_guide_desc}</span>
        </div>
        <Button
          type="text"
          size="small"
          icon={<CloseOutlined />}
          aria-label={S.first_run_api_guide_later}
          onClick={dismiss}
        />
      </div>
      <div className="kb-first-run-api-guide-steps">
        <div>
          <span className="is-required">1</span>
          <p>
            <strong>{S.first_run_api_guide_text_title}</strong>
            <small>{S.first_run_api_guide_text_desc}</small>
          </p>
        </div>
        <div>
          <span>2</span>
          <p>
            <strong>{S.first_run_api_guide_vision_title}</strong>
            <small>{S.first_run_api_guide_vision_desc}</small>
          </p>
        </div>
      </div>
      <div className="kb-first-run-api-guide-footer">
        <span>{S.first_run_api_guide_local_note}</span>
        <div>
          <Button size="small" onClick={dismiss}>
            {S.first_run_api_guide_later}
          </Button>
          <Button type="primary" size="small" onClick={onConfigure}>
            {S.first_run_api_guide_configure}
          </Button>
        </div>
      </div>
    </section>
  )
}
