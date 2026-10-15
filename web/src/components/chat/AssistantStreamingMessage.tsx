import { Typography } from 'antd'
import type { AgentTraceReferenceHandlers } from './agentTraceReferenceTypes'
import {
  AssistantRetrievalFallbackNotice,
  AssistantSourceNotice,
  AssistantSourceSummaryNotice,
} from './AssistantMessageNotices'
import { AgentTracePanel } from './AgentTracePanel'
import type { AnswerSourceNoticeViewModel } from './answerSourceNoticeViewModel'
import { sourceSummaryFromAnswerContract } from './answerSourceNoticeViewModel'
import { MarkdownRenderer } from './MarkdownRenderer'
import {
  cleanAssistantAnswerPresentationText,
  splitLeadingAssistantSourceNotice,
} from './messageRenderPacket'
import { ResearchTracePanel } from './ResearchTracePanel'

const { Text } = Typography

interface AssistantStreamingMessageProps extends AgentTraceReferenceHandlers {
  partial: string
  stage?: string
  trace?: Record<string, unknown>
  agentTrace?: Record<string, unknown>
  agentSourceSummary?: Record<string, unknown>
  answerContract?: Record<string, unknown>
  S: Record<string, string>
  onOpenEvidence?: (sourceNotice: AnswerSourceNoticeViewModel) => void
}

export function AssistantAvatar() {
  return (
    <div className="kb-msg-avatar kb-msg-avatar-assistant">
      <img src="/pi_logo.png" alt="Pi assistant" className="h-5 w-5 object-contain" loading="lazy" />
    </div>
  )
}

export function AssistantStreamingMessage({
  partial,
  stage,
  trace,
  agentTrace,
  agentSourceSummary,
  answerContract,
  S,
  onOpenEvidence,
  onOpenReference,
  onAddReferenceToShelf,
}: AssistantStreamingMessageProps) {
  const cleanPartial = cleanAssistantAnswerPresentationText(partial)
  const sourceNotice = splitLeadingAssistantSourceNotice(cleanPartial)
  const visiblePartial = sourceNotice.notice ? sourceNotice.body : cleanPartial
  const contractSourceSummary = sourceSummaryFromAnswerContract(answerContract)
  const effectiveSourceSummary = contractSourceSummary || agentSourceSummary
  const hasAnswerContract = Boolean(answerContract && Object.keys(answerContract).length > 0)
  const hasSourceSummary = Boolean(
    hasAnswerContract
      || (effectiveSourceSummary && Object.keys(effectiveSourceSummary).length > 0),
  )

  return (
    <div className="kb-message-row is-assistant">
      <AssistantAvatar />
      <div className="kb-msg-bubble kb-msg-bubble-assistant is-streaming">
        {stage ? (
          <div className="mb-2 flex items-center gap-2">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--accent)]" />
            <Text type="secondary" className="text-xs">
              {S.chat_activity_generation}
            </Text>
          </div>
        ) : null}
        {hasSourceSummary ? (
          <AssistantSourceSummaryNotice
            answerContract={answerContract}
            sourceSummary={effectiveSourceSummary}
            fallbackNoticeText={sourceNotice.notice}
            onOpenEvidence={onOpenEvidence}
            S={S}
          />
        ) : sourceNotice.notice ? (
          <AssistantSourceNotice noticeText={sourceNotice.notice} S={S} />
        ) : null}
        <AssistantRetrievalFallbackNotice trace={trace} S={S} />
        {visiblePartial ? (
          <div className="break-words text-sm leading-7 text-[var(--text)]" data-testid="streaming-markdown-answer">
            <MarkdownRenderer content={visiblePartial} citeDetails={[]} />
          </div>
        ) : (
          <div className="flex items-center gap-1 py-1">
            <span className="typing-dot" />
            <span className="typing-dot" style={{ animationDelay: '0.15s' }} />
            <span className="typing-dot" style={{ animationDelay: '0.3s' }} />
          </div>
        )}
        <AgentTracePanel
          trace={agentTrace}
          onOpenReference={onOpenReference}
          onAddReferenceToShelf={onAddReferenceToShelf}
        />
        <ResearchTracePanel trace={trace} />
      </div>
    </div>
  )
}
