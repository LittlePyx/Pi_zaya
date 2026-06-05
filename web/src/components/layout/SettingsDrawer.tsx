import { type ReactNode, useEffect, useState } from 'react'
import { ApiOutlined } from '@ant-design/icons'
import { Alert, Button, Drawer, Input, Select, Segmented, Slider, Switch, Typography, message } from 'antd'
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
  const [textApiKey, setTextApiKey] = useState('')
  const [textBaseUrl, setTextBaseUrl] = useState('')
  const [textModel, setTextModel] = useState('')
  const [visionApiKey, setVisionApiKey] = useState('')
  const [visionBaseUrl, setVisionBaseUrl] = useState('')
  const [visionModel, setVisionModel] = useState('')
  const [savingConnection, setSavingConnection] = useState(false)
  const [testingTarget, setTestingTarget] = useState<'text' | 'vision' | null>(null)

  useEffect(() => {
    if (!open) return
    setTextApiKey('')
    setVisionApiKey('')
    setTextBaseUrl(s.textBaseUrl || '')
    setTextModel(s.textModel || s.model || '')
    setVisionBaseUrl(s.visionBaseUrl || '')
    setVisionModel(s.visionModel || '')
  }, [open, s.model, s.textBaseUrl, s.textModel, s.visionBaseUrl, s.visionModel])

  const saveConnection = async () => {
    setSavingConnection(true)
    try {
      await settingsApi.update({
        ...(textApiKey.trim() ? { textApiKey: textApiKey.trim() } : {}),
        textBaseUrl: textBaseUrl.trim(),
        textModel: textModel.trim(),
        ...(visionApiKey.trim() ? { visionApiKey: visionApiKey.trim() } : {}),
        visionBaseUrl: visionBaseUrl.trim(),
        visionModel: visionModel.trim(),
      })
      setTextApiKey('')
      setVisionApiKey('')
      await s.load()
      message.success(S.settings_api_settings_saved)
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.settings_save_api_settings_failed)
    } finally {
      setSavingConnection(false)
    }
  }

  const testLlm = async (target: 'text' | 'vision') => {
    setTestingTarget(target)
    try {
      const res = await settingsApi.testLlm(target, target === 'text'
        ? {
            apiKey: textApiKey.trim(),
            baseUrl: textBaseUrl.trim(),
            model: textModel.trim(),
          }
        : {
            apiKey: visionApiKey.trim(),
            baseUrl: visionBaseUrl.trim(),
            model: visionModel.trim(),
          })
      message[res.ok ? 'success' : 'error'](
        res.ok
          ? S.settings_test_ok.replace('{reply}', String(res.reply || S.settings_test_default_reply))
          : S.settings_test_failed.replace('{error}', String(res.error || S.settings_test_unknown_error)),
      )
    } finally {
      setTestingTarget(null)
    }
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
          <div className="kb-settings-connection-alerts">
            {!s.hasTextApiKey ? (
              <Alert
                type="warning"
                showIcon
                message={S.settings_missing_text_api_title}
                description={S.settings_missing_text_api_desc}
              />
            ) : null}
            {s.hasTextApiKey && s.visionUsesTextFallback ? (
              <Alert
                type="info"
                showIcon
                message={S.settings_missing_vision_api_title}
                description={S.settings_missing_vision_api_desc}
              />
            ) : null}
          </div>
          <div className="kb-settings-credential-grid">
            <section className="kb-settings-credential">
              <div className="kb-settings-credential-head">
                <div>
                  <Text className="kb-settings-credential-title">{S.settings_text_api_title}</Text>
                  <Text className="kb-settings-credential-desc">{S.settings_text_api_desc}</Text>
                </div>
                <SettingsValue value={s.hasTextApiKey ? S.settings_api_key_ready : S.settings_api_key_missing} muted={!s.hasTextApiKey} />
              </div>
              <div className="kb-settings-credential-fields">
                <Input.Password
                  value={textApiKey}
                  onChange={(event) => setTextApiKey(event.target.value)}
                  placeholder={S.settings_api_key_placeholder}
                  autoComplete="off"
                />
                <Input
                  value={textBaseUrl}
                  onChange={(event) => setTextBaseUrl(event.target.value)}
                  placeholder={S.settings_base_url}
                />
                <Input
                  value={textModel}
                  onChange={(event) => setTextModel(event.target.value)}
                  placeholder={S.settings_model_id}
                />
              </div>
              <div className="kb-settings-credential-actions">
                <Button
                  icon={<ApiOutlined />}
                  loading={testingTarget === 'text'}
                  onClick={() => { void testLlm('text') }}
                >
                  {S.settings_test_text_connection}
                </Button>
              </div>
            </section>

            <section className="kb-settings-credential">
              <div className="kb-settings-credential-head">
                <div>
                  <Text className="kb-settings-credential-title">{S.settings_vision_api_title}</Text>
                  <Text className="kb-settings-credential-desc">{S.settings_vision_api_desc}</Text>
                </div>
                <SettingsValue
                  value={s.visionUsesTextFallback ? S.settings_api_key_fallback : (s.hasVisionApiKey ? S.settings_api_key_ready : S.settings_api_key_missing)}
                  muted={!s.hasVisionApiKey || s.visionUsesTextFallback}
                />
              </div>
              <div className="kb-settings-credential-fields">
                <Input.Password
                  value={visionApiKey}
                  onChange={(event) => setVisionApiKey(event.target.value)}
                  placeholder={S.settings_api_key_placeholder}
                  autoComplete="off"
                />
                <Input
                  value={visionBaseUrl}
                  onChange={(event) => setVisionBaseUrl(event.target.value)}
                  placeholder={S.settings_base_url}
                />
                <Input
                  value={visionModel}
                  onChange={(event) => setVisionModel(event.target.value)}
                  placeholder={S.settings_model_id}
                />
              </div>
              <div className="kb-settings-credential-actions">
                <Button
                  icon={<ApiOutlined />}
                  loading={testingTarget === 'vision'}
                  onClick={() => { void testLlm('vision') }}
                >
                  {S.settings_test_vision_connection}
                </Button>
              </div>
            </section>
          </div>
          <div className="kb-settings-action-row">
            <Button type="primary" loading={savingConnection} onClick={() => { void saveConnection() }}>
              {S.settings_save_api_settings}
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
