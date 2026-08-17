import { Button, Select, Typography } from 'antd'
import { ReloadOutlined, StopOutlined } from '@ant-design/icons'
import { SCOPE_OPTIONS } from './libraryPageUtils'
import './LibraryProcessControls.css'

const { Text } = Typography

type LibraryProcessControlsProps = {
  S: Record<string, string>
  scope: string
  converting: boolean
  onScopeChange: (value: string) => void | Promise<unknown>
  onConvertPending: () => void | Promise<unknown>
  onRefresh: () => void | Promise<unknown>
  onStopConvert: () => void | Promise<unknown>
}

export function LibraryProcessControls({
  S,
  scope,
  converting,
  onScopeChange,
  onConvertPending,
  onRefresh,
  onStopConvert,
}: LibraryProcessControlsProps) {
  return (
    <section className="kb-lib-workbench-section kb-lib-workbench-section-process">
      <div className="kb-lib-section-head">
        <div className="kb-lib-section-copy">
          <Text className="kb-lib-section-title">{S.lib_section_batch}</Text>
        </div>
      </div>

      <div className="kb-lib-process-toolbar">
        <div className="kb-lib-process-toolbar-main">
          <Select
            value={scope}
            onChange={(value) => { void onScopeChange(String(value)) }}
            data-testid="library-process-scope"
            className="kb-lib-process-scope"
            options={SCOPE_OPTIONS(S)}
          />
          <Button className="kb-lib-action-tonal" type="primary" onClick={() => { void onConvertPending() }}>
            {S.lib_btn_convert_pending_short}
          </Button>
        </div>
        <div className="kb-lib-process-toolbar-side">
          <Button className="kb-lib-action-quiet kb-lib-process-refresh" icon={<ReloadOutlined />} onClick={() => { void onRefresh() }}>
            {S.lib_btn_refresh}
          </Button>
          {converting ? (
            <Button icon={<StopOutlined />} danger onClick={() => { void onStopConvert() }}>
              {S.lib_btn_stop_all}
            </Button>
          ) : null}
        </div>
      </div>
    </section>
  )
}
