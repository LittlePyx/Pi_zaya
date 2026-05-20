import { Drawer, Slider, Button, Switch, Select, message, Typography } from 'antd'
import { useSettingsStore } from '../../stores/settingsStore'
import { settingsApi } from '../../api/settings'
import { useT } from '../../i18n'

const { Text } = Typography

export function SettingsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const S = useT()
  const s = useSettingsStore()

  const testLlm = async () => {
    const res = await settingsApi.testLlm()
    message[res.ok ? 'success' : 'error'](res.ok ? `OK: ${res.reply}` : `失败: ${res.error}`)
  }

  return (
    <Drawer title={S.settings} open={open} onClose={onClose} size={320}>
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-3">
          <Text>{S.ui_lang}</Text>
          <Select
            className="w-28"
            value={s.uiLocale}
            onChange={(v) => s.update({ uiLocale: v as 'zh' | 'en' })}
            options={[
              { label: S.lang_zh, value: 'zh' },
              { label: S.lang_en, value: 'en' },
            ]}
          />
        </div>

        <div>
          <Text type="secondary">{S.model_label}: {s.model || S.model_not_configured}</Text>
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
            <Text type="secondary">{S.max_tokens_hint}</Text>
          </div>
        </div>

        <div className="flex items-center justify-between gap-3">
          <Text>{S.answer_contract}</Text>
          <Switch checked={s.answerContractV1} onChange={(v) => s.update({ answerContractV1: v })} />
        </div>

        <div className="flex items-center justify-between gap-3">
          <Text>{S.auto_depth}</Text>
          <Switch checked={s.answerDepthAuto} onChange={(v) => s.update({ answerDepthAuto: v })} />
        </div>

        <div>
          <Text type="secondary">{S.answer_mode_hint}</Text>
          <Select
            className="mt-2 w-full"
            value={s.answerModeHint || ''}
            onChange={(v) => s.update({ answerModeHint: String(v || '') })}
            options={[
              { label: S.mode_auto, value: '' },
              { label: S.mode_reading, value: 'reading' },
              { label: S.mode_compare, value: 'compare' },
              { label: S.mode_idea, value: 'idea' },
              { label: S.mode_experiment, value: 'experiment' },
              { label: S.mode_troubleshoot, value: 'troubleshoot' },
              { label: S.mode_writing, value: 'writing' },
            ]}
          />
        </div>

        <Button onClick={testLlm}>{S.test_llm}</Button>
      </div>
    </Drawer>
  )
}
