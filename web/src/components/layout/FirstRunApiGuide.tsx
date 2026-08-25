import { ApiOutlined, CheckOutlined, FileAddOutlined, MessageOutlined, MinusOutlined, SyncOutlined } from '@ant-design/icons'
import { Button } from 'antd'
import { useEffect, useMemo, useState } from 'react'
import type { OnboardingStep } from '../../api/app'
import { useT } from '../../i18n'
import { useOnboardingStore } from '../../stores/onboardingStore'

type GuideStep = Exclude<OnboardingStep, 'completed'>

const STEP_ORDER: GuideStep[] = ['connect_model', 'prepare_document', 'ask_question']

export function FirstRunApiGuide({
  visible,
  hasTextApiKey,
  onConfigure,
  onOpenLibrary,
  onAskQuestion,
}: {
  visible: boolean
  hasTextApiKey: boolean
  onConfigure: () => void
  onOpenLibrary: () => void
  onAskQuestion: () => void
}) {
  const S = useT()
  const [collapsed, setCollapsed] = useState(false)
  const status = useOnboardingStore((state) => state.status)
  const loaded = useOnboardingStore((state) => state.loaded)
  const refresh = useOnboardingStore((state) => state.refresh)

  useEffect(() => {
    if (visible) void refresh()
  }, [refresh, visible])

  useEffect(() => {
    if (!visible || status?.completed) return
    const timer = window.setInterval(() => { void refresh() }, 2500)
    return () => window.clearInterval(timer)
  }, [refresh, status?.completed, visible])

  const fallbackStep: GuideStep | null = hasTextApiKey ? null : 'connect_model'
  const currentStep = status?.current_step === 'completed'
    ? null
    : (status?.current_step || fallbackStep)
  const documentPreparing = Boolean(
    currentStep === 'prepare_document'
    && status
    && status.imported_document_count > 0
    && status.ready_document_count <= 0,
  )

  const steps = useMemo(() => [
    {
      key: 'connect_model' as const,
      icon: <ApiOutlined />,
      title: S.first_run_api_guide_text_title,
      description: S.first_run_api_guide_text_desc,
    },
    {
      key: 'prepare_document' as const,
      icon: documentPreparing ? <SyncOutlined /> : <FileAddOutlined />,
      title: documentPreparing
        ? S.first_run_api_guide_document_wait_title
        : S.first_run_api_guide_document_title,
      description: documentPreparing
        ? S.first_run_api_guide_document_wait_desc.replace('{n}', String(status?.imported_document_count || 0))
        : S.first_run_api_guide_document_desc,
    },
    {
      key: 'ask_question' as const,
      icon: <MessageOutlined />,
      title: S.first_run_api_guide_question_title,
      description: S.first_run_api_guide_question_desc,
    },
  ], [S, documentPreparing, status?.imported_document_count])

  if (!visible || status?.completed || (!currentStep && loaded)) return null
  if (!currentStep) return null

  const currentIndex = STEP_ORDER.indexOf(currentStep)
  const current = steps[currentIndex]
  const action = currentStep === 'connect_model'
    ? onConfigure
    : currentStep === 'prepare_document'
      ? onOpenLibrary
      : onAskQuestion
  const actionLabel = currentStep === 'connect_model'
    ? S.first_run_api_guide_configure
    : currentStep === 'prepare_document'
      ? (documentPreparing ? S.first_run_api_guide_view_progress : S.first_run_api_guide_import)
      : S.first_run_api_guide_ask
  const footerNote = currentStep === 'connect_model'
    ? S.first_run_api_guide_local_note
    : currentStep === 'prepare_document'
      ? (documentPreparing
          ? S.first_run_api_guide_conversion_note
          : S.first_run_api_guide_sample_note)
      : S.first_run_api_guide_question_note

  if (collapsed) {
    return (
      <section className="kb-first-run-api-guide is-collapsed" data-testid="first-run-api-guide">
        <div className="kb-first-run-api-guide-compact-copy">
          <span>{currentIndex + 1}/3</span>
          <strong>{current.title}</strong>
        </div>
        <Button size="small" onClick={() => setCollapsed(false)}>{S.first_run_api_guide_continue}</Button>
      </section>
    )
  }

  return (
    <section
      className="kb-first-run-api-guide"
      data-testid="first-run-api-guide"
      data-current-step={currentStep}
      aria-label={S.first_run_api_guide_title}
    >
      <div className="kb-first-run-api-guide-head">
        <span className="kb-first-run-api-guide-icon" aria-hidden="true"><ApiOutlined /></span>
        <div className="kb-first-run-api-guide-heading">
          <strong>{S.first_run_api_guide_title}</strong>
          <span>{S.first_run_api_guide_desc}</span>
        </div>
        <Button
          type="text"
          size="small"
          icon={<MinusOutlined />}
          aria-label={S.first_run_api_guide_collapse}
          onClick={() => setCollapsed(true)}
        />
      </div>
      <div className="kb-first-run-api-guide-steps">
        {steps.map((step, index) => {
          const complete = index < currentIndex
          const active = index === currentIndex
          return (
            <div key={step.key} className={`${complete ? 'is-complete' : ''}${active ? ' is-active' : ''}`}>
              <span>{complete ? <CheckOutlined /> : step.icon}</span>
              <p>
                <strong>{index + 1}. {step.title}</strong>
                <small>{step.description}</small>
              </p>
            </div>
          )
        })}
      </div>
      <div className="kb-first-run-api-guide-footer">
        <span>{footerNote}</span>
        <Button type="primary" size="small" onClick={action}>
          {actionLabel}
        </Button>
      </div>
    </section>
  )
}
