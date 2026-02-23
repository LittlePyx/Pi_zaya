import { Drawer, Slider, Switch, Button, message, Typography } from 'antd'
import { useSettingsStore } from '../../stores/settingsStore'
import { settingsApi } from '../../api/settings'
import { S } from '../../i18n/zh'

const { Text } = Typography

export function SettingsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const s = useSettingsStore()

  const testLlm = async () => {
    const res = await settingsApi.testLlm()
    message[res.ok ? 'success' : 'error'](res.ok ? `OK: ${res.reply}` : `失败: ${res.error}`)
  }

  return (
    <Drawer title={S.settings} open={open} onClose={onClose} width={320}>
      <div className="space-y-4">
        <div>
          <Text type="secondary">模型: {s.model || '未配置'}</Text>
        </div>

        <div>
          <Text>{S.top_k}: {s.topK}</Text>
          <Slider min={2} max={20} value={s.topK} onChange={v => s.update({ topK: v })} />
        </div>

        <div>
          <Text>{S.temp}: {s.temperature}</Text>
          <Slider min={0} max={1} step={0.05} value={s.temperature} onChange={v => s.update({ temperature: v })} />
        </div>

        <div>
          <Text>{S.max_tokens}: {s.maxTokens}</Text>
          <Slider min={256} max={4096} step={64} value={s.maxTokens} onChange={v => s.update({ maxTokens: v })} />
        </div>

        <div className="flex items-center justify-between">
          <Text>{S.deep_read}</Text>
          <Switch checked={s.deepRead} onChange={v => s.update({ deepRead: v })} />
        </div>

        <div className="flex items-center justify-between">
          <Text>{S.show_ctx}</Text>
          <Switch checked={s.showContext} onChange={v => s.update({ showContext: v })} />
        </div>

        <Button onClick={testLlm}>测试 LLM 连接</Button>
      </div>
    </Drawer>
  )
}
