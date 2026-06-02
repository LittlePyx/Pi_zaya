import { type ReactNode } from 'react'
import { ApiOutlined } from '@ant-design/icons'
import { Button, Drawer, Select, Segmented, Slider, Switch, Typography, message } from 'antd'
import { useSettingsStore } from '../../stores/settingsStore'
import { settingsApi } from '../../api/settings'
import { useT } from '../../i18n'

const { Text } = Typography

function SettingsSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="kb-settings-section">
      <div className="kb-settings-section-title">{title}</div>
      <div className="kb-settings-section-body">{children}</div>
    </section>
  )
}

function SettingsRow({
  title,
  description,
  children,
}: {
  title: string
  description?: string
  children: ReactNode
}) {
  return (
    <div className="kb-settings-row">
      <div className="kb-settings-row-copy">
        <Text className="kb-settings-row-title">{title}</Text>
        {description ? <Text className="kb-settings-row-desc">{description}</Text> : null}
      </div>
      <div className="kb-settings-row-control">{children}</div>
    </div>
  )
}

function SettingsValue({ value, muted = false }: { value: ReactNode; muted?: boolean }) {
  return <span className={`kb-settings-value ${muted ? 'is-muted' : ''}`}>{value}</span>
}

export function SettingsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const S = useT()
  const s = useSettingsStore()

  const testLlm = async () => {
    const res = await settingsApi.testLlm()
    message[res.ok ? 'success' : 'error'](
      res.ok
        ? S.settings_test_ok.replace('{reply}', String(res.reply || S.settings_test_default_reply))
        : S.settings_test_failed.replace('{error}', String(res.error || S.settings_test_unknown_error)),
    )
  }

  return (
    <Drawer
      title={(
        <div className="kb-settings-title">
          <span>{S.settings}</span>
          <Text className="kb-settings-subtitle">{S.settings_subtitle}</Text>
        </div>
      )}
      open={open}
      onClose={onClose}
      size="default"
      rootClassName="kb-settings-drawer-root"
      className="kb-settings-drawer"
    >
      <div className="kb-settings-shell">
        <SettingsSection title={S.settings_section_interface}>
          <SettingsRow title={S.ui_lang} description={S.settings_language_desc}>
            <Select
              className="kb-settings-select"
              value={s.uiLocale}
              onChange={(v) => { void s.update({ uiLocale: v as 'zh' | 'en' }) }}
              options={[
                { label: S.lang_zh, value: 'zh' },
                { label: S.lang_en, value: 'en' },
              ]}
            />
          </SettingsRow>
          <SettingsRow title={S.settings_theme} description={S.settings_theme_desc}>
            <Segmented
              className="kb-settings-segmented"
              value={s.theme}
              onChange={(v) => { void s.update({ theme: v as 'light' | 'dark' }) }}
              options={[
                { label: S.theme_light, value: 'light' },
                { label: S.theme_dark, value: 'dark' },
              ]}
            />
          </SettingsRow>
        </SettingsSection>

        <SettingsSection title={S.settings_section_answer}>
          <SettingsRow title={S.answer_mode_hint} description={S.settings_answer_mode_desc}>
            <Select
              className="kb-settings-select"
              value={s.answerModeHint || ''}
              onChange={(v) => { void s.update({ answerModeHint: String(v || '') }) }}
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
          </SettingsRow>
          <SettingsRow title={S.settings_answer_shape} description={S.settings_answer_shape_desc}>
            <Select
              className="kb-settings-select"
              value={s.answerOutputMode || ''}
              onChange={(v) => { void s.update({ answerOutputMode: String(v || '') }) }}
              options={[
                { label: S.settings_shape_auto, value: '' },
                { label: S.settings_shape_reading_guide, value: 'reading_guide' },
                { label: S.settings_shape_fact_answer, value: 'fact_answer' },
                { label: S.settings_shape_critical_review, value: 'critical_review' },
              ]}
            />
          </SettingsRow>
          <SettingsRow title={S.settings_auto_depth} description={S.settings_auto_depth_desc}>
            <Switch checked={s.answerDepthAuto} onChange={(v) => { void s.update({ answerDepthAuto: v }) }} />
          </SettingsRow>
          <SettingsRow title={S.settings_citation_language} description={S.settings_citation_language_desc}>
            <Segmented
              className="kb-settings-segmented"
              value={s.refsCardLocale}
              onChange={(v) => { void s.update({ refsCardLocale: v as 'auto' | 'zh' | 'en' }) }}
              options={[
                { label: S.settings_citation_auto, value: 'auto' },
                { label: S.settings_citation_zh, value: 'zh' },
                { label: S.settings_citation_en, value: 'en' },
              ]}
            />
          </SettingsRow>
        </SettingsSection>

        <SettingsSection title={S.settings_section_connection}>
          <SettingsRow title={S.model_label} description={S.settings_model_desc}>
            <SettingsValue value={s.model || S.model_not_configured} muted={!s.model} />
          </SettingsRow>
          <SettingsRow title={S.settings_api_key} description={S.settings_api_key_desc}>
            <SettingsValue value={s.hasApiKey ? S.settings_api_key_ready : S.settings_api_key_missing} muted={!s.hasApiKey} />
          </SettingsRow>
          <div className="kb-settings-action-row">
            <Button icon={<ApiOutlined />} onClick={() => { void testLlm() }}>
              {S.settings_test_connection}
            </Button>
          </div>
        </SettingsSection>

        <details className="kb-settings-advanced">
          <summary>
            <span>{S.settings_section_advanced}</span>
            <Text>{S.settings_advanced_desc}</Text>
          </summary>
          <div className="kb-settings-advanced-body">
            <SettingsRow title={`${S.top_k}: ${s.topK}`} description={S.settings_top_k_desc}>
              <Slider min={2} max={20} value={s.topK} onChange={(v) => { void s.update({ topK: v }) }} />
            </SettingsRow>
            <SettingsRow title={`${S.temp}: ${s.temperature}`} description={S.settings_temperature_desc}>
              <Slider min={0} max={1} step={0.05} value={s.temperature} onChange={(v) => { void s.update({ temperature: v }) }} />
            </SettingsRow>
            <SettingsRow title={`${S.max_tokens}: ${s.maxTokens}`} description={S.settings_max_tokens_desc}>
              <Slider min={512} max={3072} step={128} value={s.maxTokens} onChange={(v) => { void s.update({ maxTokens: v }) }} />
            </SettingsRow>
            <SettingsRow title={S.answer_contract} description={S.settings_structured_answer_desc}>
              <Switch checked={s.answerContractV1} onChange={(v) => { void s.update({ answerContractV1: v }) }} />
            </SettingsRow>
          </div>
        </details>
      </div>
    </Drawer>
  )
}
