import type { Conversation } from '../../api/chat'
import type { LlmProviderReadiness, LlmReadinessPayload } from '../../api/settings'
import { shelfProjectScopeId } from './citationState'
import type { ReaderOpenPayload } from './reader/readerTypes'

export type ResearchTaskMode = 'normal' | 'paper_guide' | 'paper_guide_reader' | 'reader_review'
export type ResearchSourceKind = 'none' | 'guide' | 'reader'

export interface ResearchSourceContext {
  kind: ResearchSourceKind
  sourcePath: string
  sourceName: string
  label: string
  ready: boolean
}

export interface ResearchApiProviderContext {
  blocked: boolean
  configured: boolean
  severity: 'ok' | 'warning' | 'error'
  status: string
  reason: string
  lastError: string
}

export interface ResearchApiContext {
  text: ResearchApiProviderContext
  vision: ResearchApiProviderContext
  sendBlocked: boolean
  sendBlockTarget: 'text' | 'vision' | ''
  connectionAlertTarget: 'text' | 'vision' | ''
  needsVisionForPendingImages: boolean
  showTextConnectionAlert: boolean
}

export interface ResearchRuntimeContext {
  conversationId: string
  projectId: string
  shelfProjectId: string
  shelfScope: string
  mode: 'normal' | 'paper_guide'
  taskMode: ResearchTaskMode
  guideSource: ResearchSourceContext
  readerSource: ResearchSourceContext
  activeSource: ResearchSourceContext
  reader: {
    open: boolean
    linkedToConversation: boolean
    locateFeedbackKey: string
  }
  api: ResearchApiContext
  fingerprint: string
}

interface BuildResearchContextInput {
  activeConvId?: string | null
  activeProjectId?: string | null
  activeConversation?: Conversation | null
  guideBinding?: { sourcePath?: string; sourceName?: string } | null
  readerOpen?: boolean
  readerPayload?: ReaderOpenPayload | null
  settingsLoaded?: boolean
  hasTextApiKey?: boolean
  hasVisionApiKey?: boolean
  visionUsesTextFallback?: boolean
  readiness?: LlmReadinessPayload | null
  pendingImageCount?: number
}

const cleanText = (value: unknown): string => String(value || '').trim()

const sourceLabel = (sourceName: string, sourcePath: string): string =>
  sourceName || sourcePath.split(/[\\/]/).filter(Boolean).pop() || sourcePath

const providerContext = (
  readiness: LlmProviderReadiness | undefined,
  configuredFallback: boolean,
  extraBlocked = false,
): ResearchApiProviderContext => {
  const configured = Boolean(readiness?.has_api_key ?? configuredFallback)
  const severity = readiness?.severity || (configured ? 'ok' : 'error')
  const status = cleanText(readiness?.status) || (configured ? 'configured' : 'missing')
  const lastError = cleanText(readiness?.last_test?.error)
  const reason = lastError || cleanText(readiness?.reason)
  return {
    blocked: Boolean(extraBlocked || !configured || severity === 'error'),
    configured,
    severity,
    status,
    reason,
    lastError,
  }
}

export function buildResearchContext(input: BuildResearchContextInput): ResearchRuntimeContext {
  const conversationId = cleanText(input.activeConvId || input.activeConversation?.id)
  const projectId = cleanText(input.activeConversation?.project_id || input.activeProjectId)
  const shelfProjectId = projectId
  const shelfScope = shelfProjectScopeId(shelfProjectId)
  const guideSourcePath = cleanText(input.activeConversation?.bound_source_path || input.guideBinding?.sourcePath)
  const guideSourceName = cleanText(input.activeConversation?.bound_source_name || input.guideBinding?.sourceName)
  const guideReady = Boolean(input.activeConversation?.bound_source_ready || guideSourcePath)
  const mode: 'normal' | 'paper_guide' = input.activeConversation?.mode === 'paper_guide' || guideSourcePath ? 'paper_guide' : 'normal'
  const guideSource: ResearchSourceContext = {
    kind: guideSourcePath ? 'guide' : 'none',
    sourcePath: guideSourcePath,
    sourceName: guideSourceName,
    label: sourceLabel(guideSourceName, guideSourcePath),
    ready: Boolean(guideSourcePath && guideReady),
  }

  const readerSourcePath = cleanText(input.readerPayload?.sourcePath)
  const readerSourceName = cleanText(input.readerPayload?.sourceName)
  const readerOpen = Boolean(input.readerOpen && readerSourcePath)
  const readerSource: ResearchSourceContext = {
    kind: readerOpen ? 'reader' : 'none',
    sourcePath: readerOpen ? readerSourcePath : '',
    sourceName: readerOpen ? readerSourceName : '',
    label: readerOpen ? sourceLabel(readerSourceName, readerSourcePath) : '',
    ready: readerOpen,
  }

  const activeSource = readerSource.ready ? readerSource : guideSource.ready ? guideSource : {
    kind: 'none' as const,
    sourcePath: '',
    sourceName: '',
    label: '',
    ready: false,
  }
  const taskMode: ResearchTaskMode = mode === 'paper_guide'
    ? (readerOpen ? 'paper_guide_reader' : 'paper_guide')
    : (readerOpen ? 'reader_review' : 'normal')

  const text = providerContext(input.readiness?.providers?.text, Boolean(input.hasTextApiKey))
  const visionExtraBlocked = Boolean(input.visionUsesTextFallback || input.readiness?.providers?.vision?.status === 'fallback')
  const vision = providerContext(input.readiness?.providers?.vision, Boolean(input.hasVisionApiKey), visionExtraBlocked)
  const needsVisionForPendingImages = Number(input.pendingImageCount || 0) > 0
  const settingsLoaded = Boolean(input.settingsLoaded)
  const sendBlockTarget = settingsLoaded && text.blocked
    ? 'text'
    : settingsLoaded && needsVisionForPendingImages && vision.blocked
      ? 'vision'
      : ''

  const fingerprint = [
    conversationId,
    projectId || '__default__',
    mode,
    taskMode,
    guideSource.sourcePath,
    readerSource.sourcePath,
    shelfScope,
    text.status,
    vision.status,
  ].join('|')

  return {
    conversationId,
    projectId,
    shelfProjectId,
    shelfScope,
    mode,
    taskMode,
    guideSource,
    readerSource,
    activeSource,
    reader: {
      open: readerOpen,
      linkedToConversation: Boolean(readerOpen && conversationId),
      locateFeedbackKey: cleanText(input.readerPayload?.locateFeedbackKey),
    },
    api: {
      text,
      vision,
      sendBlocked: Boolean(sendBlockTarget),
      sendBlockTarget,
      connectionAlertTarget: sendBlockTarget,
      needsVisionForPendingImages,
      showTextConnectionAlert: Boolean(settingsLoaded && text.blocked),
    },
    fingerprint,
  }
}
