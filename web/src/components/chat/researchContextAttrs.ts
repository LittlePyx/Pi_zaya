import { useMemo } from 'react'
import type { ResearchRuntimeContext } from './researchContext'

export type ResearchContextDomAttrs = Readonly<{
  'data-research-conversation-id': string
  'data-research-project-id': string
  'data-research-mode': string
  'data-research-task-mode': string
  'data-research-source-kind': string
  'data-research-source-ready': '0' | '1'
  'data-research-reader-linked': '0' | '1'
  'data-research-shelf-scope': string
  'data-research-api-text': string
  'data-research-api-vision': string
  'data-research-api-block-target': string
}>

export function useResearchContextAttrs(researchContext: ResearchRuntimeContext): ResearchContextDomAttrs {
  return useMemo(() => ({
    'data-research-conversation-id': researchContext.conversationId,
    'data-research-project-id': researchContext.projectId,
    'data-research-mode': researchContext.mode,
    'data-research-task-mode': researchContext.taskMode,
    'data-research-source-kind': researchContext.activeSource.kind,
    'data-research-source-ready': researchContext.activeSource.ready ? '1' : '0',
    'data-research-reader-linked': researchContext.reader.linkedToConversation ? '1' : '0',
    'data-research-shelf-scope': researchContext.shelfScope,
    'data-research-api-text': researchContext.api.text.status,
    'data-research-api-vision': researchContext.api.vision.status,
    'data-research-api-block-target': researchContext.api.sendBlockTarget,
  }), [
    researchContext.activeSource.kind,
    researchContext.activeSource.ready,
    researchContext.api.sendBlockTarget,
    researchContext.api.text.status,
    researchContext.api.vision.status,
    researchContext.conversationId,
    researchContext.mode,
    researchContext.projectId,
    researchContext.reader.linkedToConversation,
    researchContext.shelfScope,
    researchContext.taskMode,
  ])
}
