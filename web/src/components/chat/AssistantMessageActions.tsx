import { FileTextOutlined, ReloadOutlined } from '@ant-design/icons'
import { Button } from 'antd'
import { CopyBar } from './CopyBar'

interface AssistantMessageActionsProps {
  canSaveAsEvidenceNote: boolean
  copyMarkdown?: string
  copyText: string
  messageId: number
  onOpenEvidenceNote: (messageId: number) => void
  onRetry?: (promptText: string) => void
  retryDisabled: boolean
  retryLabel: string
  retryPrompt: string
  saveEvidenceNoteLabel: string
}

export function AssistantMessageActions({
  canSaveAsEvidenceNote,
  copyMarkdown,
  copyText,
  messageId,
  onOpenEvidenceNote,
  onRetry,
  retryDisabled,
  retryLabel,
  retryPrompt,
  saveEvidenceNoteLabel,
}: AssistantMessageActionsProps) {
  return (
    <>
      <div
        className="kb-answer-export-actions"
        data-testid={`assistant-message-actions-${messageId}`}
      >
        <CopyBar text={copyText} markdown={copyMarkdown} />
        {canSaveAsEvidenceNote ? (
          <Button
            size="small"
            type="text"
            icon={<FileTextOutlined />}
            data-testid={`research-note-open-${messageId}`}
            onClick={() => onOpenEvidenceNote(messageId)}
          >
            {saveEvidenceNoteLabel}
          </Button>
        ) : null}
      </div>
      {retryPrompt && onRetry ? (
        <Button
          className="kb-generation-retry-btn"
          type="text"
          size="small"
          icon={<ReloadOutlined />}
          disabled={retryDisabled}
          onClick={() => onRetry(retryPrompt)}
        >
          {retryLabel}
        </Button>
      ) : null}
    </>
  )
}
