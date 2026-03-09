import { Drawer, Slider, Button, Switch, Select, message, Typography } from 'antd'
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
          <Slider min={512} max={3072} step={128} value={s.maxTokens} onChange={v => s.update({ maxTokens: v })} />
          <div className="mt-1">
            <Text type="secondary">建议区间 1024-2048；超过 3072 往往只会增加时延和跑偏概率。</Text>
          </div>
        </div>

        <div className="flex items-center justify-between gap-3">
          <Text>回答结构化（v1）</Text>
          <Switch checked={s.answerContractV1} onChange={(v) => s.update({ answerContractV1: v })} />
        </div>

        <div className="flex items-center justify-between gap-3">
          <Text>自动深度档位</Text>
          <Switch checked={s.answerDepthAuto} onChange={(v) => s.update({ answerDepthAuto: v })} />
        </div>

        <div>
          <Text type="secondary">回答模式提示（可选）</Text>
          <Select
            className="mt-2 w-full"
            value={s.answerModeHint || ''}
            onChange={(v) => s.update({ answerModeHint: String(v || '') })}
            options={[
              { label: '自动', value: '' },
              { label: '文献阅读', value: 'reading' },
              { label: '方法对比', value: 'compare' },
              { label: '想法探索', value: 'idea' },
              { label: '实验设计', value: 'experiment' },
              { label: '复现排障', value: 'troubleshoot' },
              { label: '写作表达', value: 'writing' },
            ]}
          />
        </div>

        <Button onClick={testLlm}>测试 LLM 连接</Button>
      </div>
    </Drawer>
  )
}
