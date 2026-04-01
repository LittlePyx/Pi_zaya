import { useEffect, useLayoutEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import { Typography, message } from 'antd'
import { UserOutlined } from '@ant-design/icons'
import { MarkdownRenderer } from './MarkdownRenderer'
import { CopyBar } from './CopyBar'
import { CitationPopover } from './CitationPopover'
import { CiteShelf } from './CiteShelf'
import type {
  ReaderLocateCandidate,
  ReaderLocateClaimGroup,
  ReaderLocateTarget,
  ReaderOpenPayload,
} from './reader/readerTypes'
import {
  mergeCiteMeta,
  normalizeCiteDetail,
  normalizeShelfNote,
  normalizeShelfTags,
  shelfStorageKey,
  toShelfItem,
  type CiteDetail,
  type CiteShelfItem,
} from './citationState'
import { RefsPanel } from '../refs/RefsPanel'
import type { ChatImageAttachment, Message } from '../../api/chat'
import { referencesApi, type ReaderDocAnchor } from '../../api/references'
import { useChatStore } from '../../stores/chatStore'

const { Text } = Typography
const SHELF_MAX_ITEMS = 120
const SHELF_SCHEMA_VERSION = 4
const SHELF_SAVED_SCHEMA_VERSION = 1
const SHELF_SAVED_MAX_ITEMS = 16
const SHELF_SAVED_SUFFIX = ':saved_snapshots'
const MESSAGE_LIST_PREP_PERF_LIMIT = 180

interface MessageListPrepPerfEvent {
  ts: number
  convId: string
  messageCount: number
  assistantCount: number
  heavyCount: number
  lightCount: number
  cacheHits: number
  durationMs: number
}

interface MessageListPrepPerfApi {
  getLogs: () => MessageListPrepPerfEvent[]
  clear: () => void
}

interface MessageListDebugWindow extends Window {
  __kbMessageListPerf?: MessageListPrepPerfApi
}

const messageListPrepPerfLog: MessageListPrepPerfEvent[] = []

function messageListPerfNow() {
  try {
    return performance.now()
  } catch {
    return Date.now()
  }
}

function ensureMessageListPerfApi() {
  if (typeof window === 'undefined') return
  const w = window as MessageListDebugWindow
  if (w.__kbMessageListPerf) return
  w.__kbMessageListPerf = {
    getLogs: () => messageListPrepPerfLog.slice(),
    clear: () => {
      messageListPrepPerfLog.length = 0
    },
  }
}

function pushMessageListPrepPerf(event: MessageListPrepPerfEvent) {
  messageListPrepPerfLog.push(event)
  if (messageListPrepPerfLog.length > MESSAGE_LIST_PREP_PERF_LIMIT) {
    messageListPrepPerfLog.splice(0, messageListPrepPerfLog.length - MESSAGE_LIST_PREP_PERF_LIMIT)
  }
  ensureMessageListPerfApi()
}

if (typeof window !== 'undefined') {
  ensureMessageListPerfApi()
}

interface Props {
  activeConvId?: string | null
  messages: Message[]
  refs: Record<string, unknown>
  generationPartial?: string
  generationStage?: string
  jumpTarget?: { messageId: number; token: number } | null
  onJumpHandled?: (jumpTarget: { messageId: number; token: number }) => void
  trackedMessageIds?: number[]
  onTrackedMessageActive?: (messageId: number | null) => void
  onOpenReader?: (payload: ReaderOpenPayload) => void
  paperGuideSourcePath?: string
  paperGuideSourceName?: string
}

interface RefUiMetaLite {
  display_name?: string
  heading_path?: string
  section_label?: string
  subsection_label?: string
  summary_line?: string
  why_line?: string
  source_path?: string
  anchor_target_kind?: string
  anchor_target_number?: number
}

interface RefMetaLite {
  source_path?: string
  heading_path?: string
  ref_best_heading_path?: string
  ref_headings?: unknown
  ref_locs?: unknown
  ref_show_snippets?: unknown
  ref_overview_snippets?: unknown
  ref_snippets?: unknown
}

interface RefHitLite {
  ui_meta?: RefUiMetaLite
  meta?: RefMetaLite
}

interface RefEntryLite {
  hits?: RefHitLite[]
}

interface AssistantLocatePrep {
  bodyContent: string
  refsUserMsgId: number
  locateSourcePath: string
  locateSourceName: string
  refsLocateCandidatesAll: LocateCandidate[]
  guideLocateCandidates: LocateCandidate[]
  refsScopedCandidates: LocateCandidate[]
  messageProvenance: Record<string, unknown> | null
  provenanceSourcePath: string
  provenanceSourceName: string
  provenanceBlockMap: Record<string, Record<string, unknown>>
  provenanceDirectSegments: Array<Record<string, unknown>>
  hasDirectProvenance: boolean
  hasStructuredProvenance: boolean
  effectiveGuideSourcePath: string
  strictProvenanceLocate: boolean
  structuredLocateButtonCap: number
  provenanceLocateEntries: ProvenanceLocateEntry[]
  structuredProvenanceSegmentsAll: StructuredProvenanceSegment[]
  provenanceStrictIdentityReady: boolean
  hasStrictMustLocateEntries: boolean
  strictStructuredLocateOnly: boolean
  strictStructuredInlineLocate: boolean
  provenanceModeLabel: string
  structuredRenderSlotMap: Map<number, StructuredRenderLocateSlot>
  structuredLocateOrderBySegmentId: Map<string, number>
  allowedStructuredRenderOrders: Set<number>
  locateCandidates: LocateCandidate[]
}

function createEmptyAssistantLocatePrep(bodyContent: string, refsUserMsgId = 0): AssistantLocatePrep {
  return {
    bodyContent,
    refsUserMsgId,
    locateSourcePath: '',
    locateSourceName: '',
    refsLocateCandidatesAll: [],
    guideLocateCandidates: [],
    refsScopedCandidates: [],
    messageProvenance: null,
    provenanceSourcePath: '',
    provenanceSourceName: '',
    provenanceBlockMap: {},
    provenanceDirectSegments: [],
    hasDirectProvenance: false,
    hasStructuredProvenance: false,
    effectiveGuideSourcePath: '',
    strictProvenanceLocate: false,
    structuredLocateButtonCap: 12,
    provenanceLocateEntries: [],
    structuredProvenanceSegmentsAll: [],
    provenanceStrictIdentityReady: false,
    hasStrictMustLocateEntries: false,
    strictStructuredLocateOnly: false,
    strictStructuredInlineLocate: false,
    provenanceModeLabel: '',
    structuredRenderSlotMap: new Map<number, StructuredRenderLocateSlot>(),
    structuredLocateOrderBySegmentId: new Map<string, number>(),
    allowedStructuredRenderOrders: new Set<number>(),
    locateCandidates: [],
  }
}

interface LocateCandidate {
  sourcePath: string
  sourceName: string
  headingPath: string
  focusSnippet: string
  matchText: string
  sourceType: 'guide' | 'refs'
  blockId?: string
  anchorId?: string
  anchorKind?: string
  anchorNumber?: number
}

interface ProvenanceLocateEntry {
  segmentId: string
  label: string
  segmentText: string
  evidenceQuote: string
  claimType?: string
  mustLocate?: boolean
  locatePolicy?: string
  locateSurfacePolicy?: string
  claimGroupId?: string
  claimGroupKind?: string
  formulaOrigin?: string
  anchorKind?: string
  anchorText?: string
  equationNumber?: number
  snippetKey: string
  snippetAliases: string[]
  primary: LocateCandidate
  alternatives: LocateCandidate[]
  relatedBlockIds?: string[]
  sourceSegmentId?: string
  groupLeadText?: string
  groupDistance?: number
}

type StructuredLocateKind = 'paragraph' | 'list_item' | 'quote' | 'blockquote' | 'equation' | 'figure'

interface StructuredRenderSegment {
  order: number
  kind: StructuredLocateKind
  text: string
  snippetKey: string
}

interface StructuredProvenanceSegment {
  index: number
  segmentId: string
  kind: string
  segmentType: string
  evidenceMode: string
  claimType: string
  mustLocate: boolean
  locatePolicy: string
  locateSurfacePolicy: string
  claimGroupId: string
  claimGroupKind: string
  claimGroupTargetSegmentId: string
  claimGroupTargetDistance: number
  claimGroupLeadText: string
  formulaOrigin: string
  anchorKind: string
  anchorText: string
  equationNumber: number
  text: string
  snippetKey: string
  snippetAliases: string[]
}

interface StructuredRenderLocateSlot {
  order: number
  kind: StructuredLocateKind
  renderText: string
  renderSnippetKey: string
  entry: ProvenanceLocateEntry
  provenanceIndex: number
  score: number
}

interface StructuredLocateResolution {
  entry: ProvenanceLocateEntry
  order: number
  fallback: boolean
}

interface LocateRenderMetaLite {
  kind?: string
  order?: number
}

const GUIDE_LOCATE_CANDIDATE_LIMIT = 1600
const REF_LOCATE_CANDIDATE_LIMIT = 900

function stripMarkdownInline(input: string): string {
  return String(input || '')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/~~([^~]+)~~/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function normalizeStrictAnchorText(input: string): string {
  return stripMarkdownInline(String(input || ''))
    .replace(/\s+/g, ' ')
    .trim()
}

function inferStrictAnchorKind(
  rawKind: string,
  claimType: string,
): string {
  const kind = String(rawKind || '').trim().toLowerCase()
  if (kind) return kind
  const claim = String(claimType || '').trim().toLowerCase()
  if (claim === 'formula_claim' || claim === 'inline_formula_claim' || claim === 'equation_explanation_claim') {
    return 'equation'
  }
  if (claim === 'figure_claim') return 'figure'
  if (claim === 'quote_claim') return 'quote'
  if (claim === 'blockquote_claim') return 'blockquote'
  if (
    claim === 'shell_sentence'
    || claim === 'critical_fact_claim'
    || claim === 'method_detail'
    || claim === 'prior_work'
    || claim === 'doc_map'
  ) {
    return 'paragraph'
  }
  return ''
}

function hasSegmentStrictLocateIdentity(
  segment: Record<string, unknown> | null | undefined,
  currentSegment?: StructuredProvenanceSegment | null,
): boolean {
  const primaryBlockId = String(segment?.primary_block_id || '').trim()
  const evidenceBlockIds = Array.isArray(segment?.evidence_block_ids)
    ? segment?.evidence_block_ids.map((item) => String(item || '').trim()).filter(Boolean)
    : []
  const anchorKindRaw = String(segment?.anchor_kind || currentSegment?.anchorKind || '').trim().toLowerCase()
  const claimType = String(segment?.claim_type || currentSegment?.claimType || '').trim().toLowerCase()
  const anchorKind = inferStrictAnchorKind(anchorKindRaw, claimType)
  const anchorText = normalizeStrictAnchorText(String(segment?.anchor_text || currentSegment?.anchorText || ''))
  const evidenceQuote = normalizeStrictAnchorText(String(segment?.evidence_quote || ''))
  if (!(primaryBlockId && evidenceBlockIds.length > 0)) return false
  if (anchorKindRaw && (anchorText || evidenceQuote)) return true
  const locatePolicy = String(segment?.locate_policy || currentSegment?.locatePolicy || '').trim().toLowerCase()
  const mustLocate = Boolean(segment?.must_locate ?? currentSegment?.mustLocate)
  if (!(mustLocate || locatePolicy === 'required')) return false
  const segmentText = normalizeStrictAnchorText(String(segment?.text || currentSegment?.text || ''))
  return Boolean(anchorKind && (anchorText || evidenceQuote || segmentText))
}

function isFormulaBundleLocateEntry(entry: Pick<ProvenanceLocateEntry, 'claimGroupKind' | 'claimGroupId' | 'anchorKind' | 'claimType' | 'primary'>): boolean {
  const groupKind = String(entry.claimGroupKind || '').trim().toLowerCase()
  if (groupKind !== 'formula_bundle') return false
  const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
  const claimType = String(entry.claimType || '').trim().toLowerCase()
  if (anchorKind === 'equation') return true
  return claimType === 'formula_claim' || claimType === 'equation_explanation_claim'
}

function formulaBundleLocateGroupKey(entry: Pick<ProvenanceLocateEntry, 'claimGroupKind' | 'claimGroupId' | 'anchorKind' | 'claimType' | 'primary'>): string {
  if (!isFormulaBundleLocateEntry(entry)) return ''
  const claimGroupId = String(entry.claimGroupId || '').trim()
  if (claimGroupId) return claimGroupId
  const sourcePath = String(entry.primary?.sourcePath || '').trim()
  const targetId = String(entry.primary?.blockId || entry.primary?.anchorId || '').trim()
  return (sourcePath && targetId) ? `${sourcePath}::${targetId}` : ''
}

function formulaBundleRepresentativeScore(entry: ProvenanceLocateEntry): number {
  let score = 0
  const claimType = String(entry.claimType || '').trim().toLowerCase()
  const locateSurfacePolicy = String(entry.locateSurfacePolicy || '').trim().toLowerCase()
  const formulaOrigin = String(entry.formulaOrigin || '').trim().toLowerCase()
  if (locateSurfacePolicy === 'primary') score += 4.5
  else if (locateSurfacePolicy === 'secondary') score += 1.25
  if (formulaOrigin === 'source') score += 1.8
  else if (formulaOrigin === 'explanation') score += 0.6
  else if (formulaOrigin === 'derived') score -= 2.4
  if (claimType === 'formula_claim') score += 4
  else if (claimType === 'inline_formula_claim') score += 1.2
  else if (claimType === 'equation_explanation_claim') score += 2
  const anchorRaw = String(entry.anchorText || entry.evidenceQuote || entry.segmentText || '').trim()
  if (/\$\$[\s\S]{8,}\$\$/.test(anchorRaw)) score += 1.4
  else if (hasFormulaSignal(anchorRaw)) score += 0.7
  if (String(entry.primary?.anchorId || '').trim()) score += 0.4
  if (String(entry.primary?.blockId || '').trim()) score += 0.3
  score -= Math.min(1, Math.max(0, Number(entry.groupDistance || 0)) * 0.08)
  return score
}

function buildGuideLocateCandidates(
  markdown: string,
  sourcePath: string,
  sourceName: string,
  sourceType: 'guide' | 'refs' = 'guide',
  readerAnchors?: ReaderDocAnchor[],
): LocateCandidate[] {
  const out: LocateCandidate[] = []
  const seen = new Set<string>()
  const pushCandidate = (
    headingPathRaw: string,
    snippetRaw: string,
    extra?: { anchorId?: string; anchorKind?: string; anchorNumber?: number },
  ) => {
    const headingPath = stripMarkdownInline(headingPathRaw)
    const text = stripMarkdownInline(snippetRaw)
    const formulaLike = hasFormulaSignal(text)
    if (text.length < 24 && !formulaLike) return
    if (formulaLike && text.length < 6) return
    const anchorId = String(extra?.anchorId || '').trim()
    const anchorKind = String(extra?.anchorKind || '').trim().toLowerCase()
    const anchorNumber = Number(extra?.anchorNumber || 0)
    const key = `${normalizeLocateText(sourcePath)}::${anchorId.toLowerCase()}::${normalizeLocateText(headingPath)}::${normalizeLocateText(text).slice(0, 260)}`
    if (seen.has(key)) return
    seen.add(key)
    out.push({
      sourcePath,
      sourceName,
      headingPath,
      focusSnippet: text,
      matchText: [headingPath, text].filter(Boolean).join('\n'),
      sourceType,
      anchorId: anchorId || undefined,
      anchorKind: anchorKind || undefined,
      anchorNumber: Number.isFinite(anchorNumber) && anchorNumber > 0 ? Math.floor(anchorNumber) : undefined,
    })
  }

  const pushSentenceCandidates = (
    headingPath: string,
    text: string,
    extra?: { anchorId?: string; anchorKind?: string; anchorNumber?: number },
  ) => {
    const src = stripMarkdownInline(text)
    if (src.length < 24 && !hasFormulaSignal(src)) return
    const sentenceList = src
      .split(/(?<=[\u3002\uff01\uff1f.!;:\uff1b\uff1a])\s+/)
      .map((item) => item.trim())
      .filter((item) => item.length >= 16)
      .slice(0, 14)
    for (let i = 0; i < sentenceList.length; i += 1) {
      const sentence = sentenceList[i]
      if (sentence.length >= 24) pushCandidate(headingPath, sentence, extra)
      const pair = [sentence, sentenceList[i + 1] || ''].filter(Boolean).join(' ').trim()
      if (pair.length >= 30 && pair.length <= 260) {
        pushCandidate(headingPath, pair, extra)
      }
    }
  }

  const anchorList = Array.isArray(readerAnchors) ? readerAnchors : []
  if (anchorList.length > 0) {
    for (const item of anchorList) {
      const anchorId = String(item?.anchor_id || '').trim()
      const headingPath = String(item?.heading_path || '').trim()
      const kind = String(item?.kind || '').trim().toLowerCase()
      const number = Number(item?.number || 0)
      const text = String(item?.text || '').trim()
      if (!text) continue
      pushCandidate(headingPath, text, {
        anchorId,
        anchorKind: kind,
        anchorNumber: number,
      })
      pushSentenceCandidates(headingPath, text, {
        anchorId,
        anchorKind: kind,
        anchorNumber: number,
      })
    }
    return out.slice(0, GUIDE_LOCATE_CANDIDATE_LIMIT)
  }

  const lines = String(markdown || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')
  const headingStack: Array<{ level: number; text: string }> = []
  let bucket: string[] = []

  const flush = () => {
    if (bucket.length <= 0) return
    const text = stripMarkdownInline(bucket.join(' ').trim())
    bucket = []
    if (text.length < 24) return
    const headingPath = headingStack.map((item) => item.text).filter(Boolean).join(' / ')
    pushCandidate(headingPath, text)
    pushSentenceCandidates(headingPath, text)
  }

  for (const raw of lines) {
    const line = String(raw || '')
    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/)
    if (heading) {
      flush()
      const level = heading[1].length
      const text = stripMarkdownInline(heading[2] || '')
      if (text) {
        while (headingStack.length > 0 && headingStack[headingStack.length - 1].level >= level) {
          headingStack.pop()
        }
        headingStack.push({ level, text })
      }
      continue
    }
    if (/^\s*([-*_]\s*){3,}\s*$/.test(line) || /^\s*```/.test(line) || /^\s*~~~/.test(line)) {
      flush()
      continue
    }
    if (!line.trim()) {
      flush()
      continue
    }
    if (/^\s*\|/.test(line) || /^\s*>/.test(line)) {
      flush()
      const text = stripMarkdownInline(line.replace(/^\s*[>|]+\s*/, ''))
      if (text.length >= 24) {
        const headingPath = headingStack.map((item) => item.text).filter(Boolean).join(' / ')
        pushCandidate(headingPath, text)
      }
      continue
    }
    bucket.push(line)
  }
  flush()
  if (out.length <= 0) return out
  // Keep a practical upper bound for runtime matching cost.
  return out.slice(0, GUIDE_LOCATE_CANDIDATE_LIMIT)
}

function hasRenderableRefs(refs: Record<string, unknown>, msgId: number) {
  const entry = refs[String(msgId)] as {
    hits?: Array<{ meta?: Record<string, unknown> }>
    guide_filter?: { hidden_self_source?: boolean }
  } | undefined
  if (!entry) return false
  const hits = Array.isArray(entry.hits) ? entry.hits : []
  return hits.length > 0
}

function normalizeLocateText(input: string): string {
  return String(input || '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

function stripProvenanceNoise(input: string): string {
  return String(input || '')
    .replace(/\[\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\]/g, ' ')
    .replace(/\(\s*\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\s*\)/g, ' ')
    .replace(/(?:see|\u53C2\u89C1)\s*\[\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\]/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function isLikelyRhetoricalLocateShell(input: string): boolean {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
    .replace(/^\s{0,3}#{1,6}\s+/g, ' ')
    .replace(/^[^\u4e00-\u9fffA-Za-z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return true
  const normalized = normalizeLocateText(raw)
  if (!normalized) return true
  if (/^(?:直接证据|间接证据|唯一且明确|延伸思考题|高价值问题|研究问题|讨论题|思考题)(?:\s*[(:：\uff08][^)\uff09:：]{0,48}[)\uff09]?)?\s*[:\uFF1A]?$/.test(raw)) {
    return true
  }
  if (/^(?:\u8bf4\u660e|\u8868\u660e|\u53ef\u89c1|\u56e0\u6b64|\u6240\u4ee5|\u603b\u4e4b|\u7efc\u4e0a|\u7531\u6b64\u53ef\u89c1|\u8fdb\u4e00\u6b65\u8bf4\u660e|\u8fdb\u4e00\u6b65\u8868\u660e|\u8fdb\u4e00\u6b65\u8bc1\u5b9e|\u63d0\u793a)\s*[:\uFF1A]?$/.test(raw)) {
    return true
  }
  if (/^(?:\u6587\u4e2d\u63d0\u5230|\u6587\u4e2d\u6307\u51fa|\u4f5c\u8005\u6307\u51fa|\u5b9e\u9a8c\u7ed3\u679c\u663e\u793a|\u7ed3\u679c\u663e\u793a|\u8868\u683c\u6807\u9898\u4e0e\u65b9\u6cd5\u547d\u540d\u660e\u786e\u4e3a).{0,160}(?:\u8bf4\u660e|\u8868\u660e|\u610f\u5473\u7740|\u63d0\u793a|\u53ef\u89c1|\u8bc1\u5b9e)\s*[:\uFF1A]$/.test(raw)) {
    return true
  }
  if (/[:\uFF1A]$/.test(raw)) {
    const informativeTail = raw
      .replace(/[:\uFF1A]\s*$/, '')
      .replace(/["'\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u300c\u300d\u300e\u300f]/g, ' ')
      .replace(/\b(?:we|our|method|paper|table|figure|fig)\b/gi, ' ')
      .replace(/\s+/g, ' ')
      .trim()
    if (informativeTail.length <= 32) return true
    if (/(?:\u8bf4\u660e|\u8868\u660e|\u610f\u5473\u7740|\u63d0\u793a|\u53ef\u89c1|\u8bc1\u5b9e)\s*$/.test(informativeTail)) return true
  }
  return false
}

function scoreLocateContentCore(
  input: string,
  opts?: {
    kind?: string
    segmentType?: string
    evidenceMode?: string
  },
): number {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return 0
  if (isLikelyRhetoricalLocateShell(raw)) return 0.04
  let score = 0.18
  const length = raw.length
  if (length >= 18) score += 0.12
  if (length >= 28) score += 0.12
  if (length >= 46) score += 0.08
  if (/[“"'`]/.test(raw)) score += 0.12
  if (/[()（）=]/.test(raw)) score += 0.08
  if (/\d/.test(raw)) score += 0.08
  if (/[A-Z][A-Za-z0-9+-]{1,}/.test(raw)) score += 0.08
  if (/\b(?:gt|ground[ -]?truth|pose|camera|train|training|input|output|pipeline|rendering|volume)\b/i.test(raw)) score += 0.08
  if (/(?:使用|采用|输入|输出|恢复|重建|估计|固定|训练|生成|表征|渲染|约束|对比|利用|先用|再将|来自|对应)/.test(raw)) score += 0.12

  const kind = String(opts?.kind || '').trim().toLowerCase()
  const segmentType = String(opts?.segmentType || '').trim().toLowerCase()
  const evidenceMode = String(opts?.evidenceMode || '').trim().toLowerCase()
  if (kind === 'list_item') score += 0.08
  if (segmentType === 'bullet' || segmentType === 'evidence' || segmentType === 'equation_explanation') score += 0.08
  if (evidenceMode === 'direct') score += 0.06
  if (/[:\uFF1A]$/.test(raw)) score -= 0.18
  return Math.max(0, Math.min(1.2, score))
}

function isLikelyClaimGroupLead(
  input: string,
  opts?: {
    kind?: string
    segmentType?: string
  },
): boolean {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return false
  if (isLikelyRhetoricalLocateShell(raw)) return true
  if (/[:\uFF1A]$/.test(raw)) return true
  const kind = String(opts?.kind || '').trim().toLowerCase()
  const segmentType = String(opts?.segmentType || '').trim().toLowerCase()
  if ((kind === 'list_item' || segmentType === 'bullet') && /(?:如下|包括|分为|步骤|流程|表明|说明|可见|因此)$/.test(raw)) {
    return true
  }
  return false
}

function isLikelySectionBoundarySegment(segment: StructuredProvenanceSegment | null): boolean {
  if (!segment) return true
  const segmentType = String(segment.segmentType || '').trim().toLowerCase()
  if (segmentType === 'claim' || segmentType === 'next_step') return true
  const text = stripProvenanceNoise(stripMarkdownInline(String(segment.text || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!text) return true
  return /^(?:\u7ed3\u8bba|\u6838\u5fc3\u7ed3\u8bba|\u4f9d\u636e|\u8bc1\u636e|\u539f\u6587|\u4e0b\u4e00\u6b65|\u5efa\u8bae|\u98ce\u9669|\u9650\u5236)\s*[:\uFF1A]?/.test(text)
}

void isLikelyClaimGroupLead
void isLikelySectionBoundarySegment

function mergeStructuredSnippetAliases(...groups: Array<string[] | null | undefined>): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const group of groups) {
    if (!Array.isArray(group)) continue
    for (const item of group) {
      const norm = normalizeStructuredLocateSnippet(String(item || ''))
      if (!norm || seen.has(norm)) continue
      seen.add(norm)
      out.push(norm)
      if (out.length >= 10) return out
    }
  }
  return out
}

function extractQuotedSpans(input: string, minLen = 10): string[] {
  const src = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
  if (!src) return []
  const out: string[] = []
  const seen = new Set<string>()
  const push = (raw: string) => {
    const text = String(raw || '').replace(/\s+/g, ' ').trim()
    if (!text || text.length < minLen) return
    const key = normalizeLocateText(text)
    if (!key || seen.has(key)) return
    seen.add(key)
    out.push(text)
  }
  const patterns = [
    /["\u201C\u201D]\s*([^"\u201C\u201D]{6,260}?)\s*["\u201C\u201D]/g,
    /[\u2018\u2019']\s*([^\u2018\u2019']{6,220}?)\s*[\u2018\u2019']/g,
    /[\u300C\u300D\u300E\u300F\u300A\u300B]\s*([^\u300C\u300D\u300E\u300F\u300A\u300B]{6,260}?)\s*[\u300D\u300F\u300B]/g,
  ]
  for (const re of patterns) {
    for (const m of src.matchAll(re)) {
      push(String(m[1] || ''))
      if (out.length >= 6) return out
    }
  }
  return out
}

function quoteMatchStats(quoteSpans: string[], ...texts: string[]): { hits: number; score: number } {
  if (!Array.isArray(quoteSpans) || quoteSpans.length <= 0) return { hits: 0, score: 0 }
  const normTexts = texts
    .map((text) => normalizeLocateText(stripProvenanceNoise(String(text || ''))))
    .filter(Boolean)
  if (normTexts.length <= 0) return { hits: 0, score: 0 }
  let hits = 0
  let score = 0
  for (const q of quoteSpans) {
    const qNorm = normalizeLocateText(q)
    if (!qNorm) continue
    let exact = false
    let bestOverlap = 0
    for (const t of normTexts) {
      if (t.includes(qNorm) || (qNorm.length >= 16 && qNorm.includes(t))) {
        exact = true
        break
      }
      bestOverlap = Math.max(bestOverlap, overlapScore(qNorm, t))
    }
    if (exact) {
      hits += 1
      score += 1.0
      continue
    }
    if (bestOverlap >= 0.66) {
      hits += 1
      score += 0.72
      continue
    }
    score += 0.45 * bestOverlap
  }
  return { hits, score }
}

function tokenizeLocateText(input: string): string[] {
  const src = normalizeLocateText(input)
  if (!src) return []
  const out: string[] = []
  const latin = src.match(/[a-z0-9]{2,}/g) || []
  out.push(...latin)
  const cjkSeq = src.match(/[\u4e00-\u9fff]{1,}/g) || []
  for (const seq of cjkSeq) {
    if (seq.length <= 2) {
      out.push(seq)
      continue
    }
    for (let i = 0; i < seq.length - 1; i += 1) {
      out.push(seq.slice(i, i + 2))
    }
  }
  return out
}

function overlapScore(a: string, b: string): number {
  const ta = tokenizeLocateText(a)
  const tb = tokenizeLocateText(b)
  if (ta.length === 0 || tb.length === 0) return 0
  const sa = new Set(ta)
  const sb = new Set(tb)
  let overlap = 0
  for (const token of sa) {
    if (sb.has(token)) overlap += 1
  }
  const denom = Math.sqrt(Math.max(1, sa.size) * Math.max(1, sb.size))
  return overlap / denom
}

function coerceStringArray(input: unknown, maxItems = 8, maxChars = 2200): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  const push = (value: unknown) => {
    if (out.length >= maxItems) return
    const text = String(value || '').replace(/\s+/g, ' ').trim()
    if (!text) return
    const normalized = normalizeLocateText(text)
    if (!normalized || seen.has(normalized)) return
    seen.add(normalized)
    out.push(text.length > maxChars ? `${text.slice(0, maxChars).trimEnd()}...` : text)
  }
  if (Array.isArray(input)) {
    for (const value of input) {
      if (out.length >= maxItems) break
      push(value)
    }
    return out
  }
  push(input)
  return out
}

function pickFirstRefText(loc: Record<string, unknown>): string {
  const keys = ['snippet', 'text', 'quote', 'content', 'summary', 'why']
  for (const key of keys) {
    const value = String(loc[key] || '').trim()
    if (value) return value
  }
  return ''
}

function formulaTokens(text: string): string[] {
  const src = String(text || '')
  if (!src) return []
  const out: string[] = []
  const texCmds = src.match(/\\[a-zA-Z]{2,}/g) || []
  out.push(...texCmds.map((item) => item.toLowerCase()))
  const symbols = src.match(/[A-Za-z](?:_[A-Za-z0-9]+)?(?:\^[A-Za-z0-9]+)?/g) || []
  out.push(...symbols.map((item) => item.toLowerCase()))
  const numbers = src.match(/\b\d{1,4}\b/g) || []
  out.push(...numbers)
  return out
}

function hasFormulaSignal(text: string): boolean {
  const src = String(text || '')
  if (!src) return false
  if (/[=^_]/.test(src)) return true
  if (/\\[a-zA-Z]{2,}/.test(src)) return true
  if (/\$[^$]{1,80}\$/.test(src) || /\$\$[^]{1,200}\$\$/.test(src)) return true
  return false
}

function formulaOverlapScore(a: string, b: string): number {
  const ta = new Set(formulaTokens(a))
  const tb = new Set(formulaTokens(b))
  if (ta.size <= 0 || tb.size <= 0) return 0
  let overlap = 0
  for (const token of ta) {
    if (tb.has(token)) overlap += 1
  }
  return overlap / Math.sqrt(ta.size * tb.size)
}

function hasDisplayFormulaSignal(text: string): boolean {
  const src = String(text || '')
  if (!src) return false
  if (/\$\$[^]{1,400}\$\$/.test(src)) return true
  if (/\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}/i.test(src)) return true
  const hasMathCore = /=/.test(src) || /\\tag\{\s*\d{1,4}\s*\}/.test(src)
  const hasMathToken = /\\[a-zA-Z]{2,}/.test(src) || /\$[^$]{1,220}\$/.test(src)
  return Boolean(hasMathCore && hasMathToken)
}

function isEquationLocateCandidate(cand: LocateCandidate | null): boolean {
  if (!cand) return false
  const kind = String(cand.anchorKind || '').trim().toLowerCase()
  if (kind === 'equation') return true
  return hasDisplayFormulaSignal(String(cand.focusSnippet || cand.matchText || ''))
}

function shortSegmentLabel(input: string, maxLen = 84): string {
  const text = stripMarkdownInline(String(input || '')).replace(/\s+/g, ' ').trim()
  if (!text) return ''
  if (text.length <= maxLen) return text
  return `${text.slice(0, Math.max(18, maxLen - 3)).trimEnd()}...`
}

function compactHeadingPath(input: string, maxLen = 56): string {
  const raw = String(input || '').replace(/\s+/g, ' ').trim()
  if (!raw) return ''
  // Prefer the leaf section label rather than repeating the full paper title path.
  const parts = raw.split('/').map((p) => p.trim()).filter(Boolean)
  const leaf = (parts[parts.length - 1] || raw).trim()
  const tail = parts.length >= 2 ? `${parts[parts.length - 2]} / ${leaf}` : leaf
  const pick = tail.length >= 12 ? tail : leaf
  if (pick.length <= maxLen) return pick
  return `${pick.slice(0, Math.max(18, maxLen - 3)).trimEnd()}...`
}

function normalizeSourcePathForMatch(input: string): string {
  return String(input || '').trim().replace(/\\/g, '/').toLowerCase()
}

function buildStructuredProvenanceLocateEntries(
  messageProvenance: Record<string, unknown> | null,
  opts: {
    guideSourcePath: string
    fallbackSourceName: string
    maxEntries?: number
    minConfidence?: number
  },
): ProvenanceLocateEntry[] {
  const guideSourcePath = String(opts?.guideSourcePath || '').trim()
  const fallbackSourceName = String(opts?.fallbackSourceName || '').trim()
  const maxEntriesRaw = Number(opts?.maxEntries || 3)
  const maxEntries = Number.isFinite(maxEntriesRaw) && maxEntriesRaw > 0 ? Math.floor(maxEntriesRaw) : 3
  const minConfidenceRaw = Number(
    opts?.minConfidence === undefined
      ? 0.62
      : opts?.minConfidence,
  )
  const minConfidence = Number.isFinite(minConfidenceRaw) ? Math.max(0, minConfidenceRaw) : 0.62
  if (!messageProvenance || typeof messageProvenance !== 'object') return []
  const strictIdentityReady = Boolean(messageProvenance.strict_identity_ready)
  const sourcePath = String(messageProvenance.source_path || '').trim()
  if (!sourcePath) return []
  if (guideSourcePath) {
    const guideNorm = normalizeSourcePathForMatch(guideSourcePath)
    const sourceNorm = normalizeSourcePathForMatch(sourcePath)
    if (guideNorm && sourceNorm && guideNorm !== sourceNorm) return []
  }
  const sourceName = String(messageProvenance.source_name || '').trim()
    || String(fallbackSourceName || '').trim()
    || sourcePath.split(/[\\/]/).pop()
    || 'paper'
  const blockMap = (messageProvenance.block_map && typeof messageProvenance.block_map === 'object')
    ? messageProvenance.block_map as Record<string, Record<string, unknown>>
    : {}
  const segmentsRaw = Array.isArray(messageProvenance.segments) ? messageProvenance.segments : []
  const segmentsAll = listStructuredProvenanceSegments(messageProvenance)
  if (segmentsRaw.length <= 0 || segmentsAll.length <= 0) return []
  const provenanceById = new Map(segmentsAll.map((segment) => [segment.segmentId, segment]))

  const scoredEntries: Array<{
    entry: ProvenanceLocateEntry
    score: number
    idx: number
  }> = []
  const seenSegment = new Set<string>()
  const seenContent = new Set<string>()
  for (let idx = 0; idx < segmentsAll.length; idx += 1) {
    const segment = segmentsRaw[idx] as Record<string, unknown> | null
    if (!segment || typeof segment !== 'object') continue
    const currentSegment = segmentsAll[idx] || null
    if (!currentSegment) continue
    const evidenceMode = String(segment.evidence_mode || '').trim().toLowerCase()
    const primaryBlockId = String(segment.primary_block_id || '').trim()
    const primaryAnchorId = String(segment.primary_anchor_id || '').trim()
    const supportBlockIdsRaw = Array.isArray(segment.support_block_ids) ? segment.support_block_ids : []
    const evidenceBlockIdsRaw = Array.isArray(segment.evidence_block_ids) ? segment.evidence_block_ids : []
    const claimType = String(segment.claim_type || currentSegment.claimType || '').trim().toLowerCase()
    const mustLocate = Boolean(segment.must_locate ?? currentSegment.mustLocate)
    const locatePolicy = String(segment.locate_policy || currentSegment.locatePolicy || '').trim().toLowerCase()
    const locateSurfacePolicy = String(segment.locate_surface_policy || currentSegment.locateSurfacePolicy || '').trim().toLowerCase()
    if (locatePolicy === 'hidden') continue
    const claimGroupId = String(segment.claim_group_id || currentSegment.claimGroupId || '').trim()
    const claimGroupKind = String(segment.claim_group_kind || currentSegment.claimGroupKind || '').trim().toLowerCase()
    const formulaOrigin = String(segment.formula_origin || currentSegment.formulaOrigin || '').trim().toLowerCase()
    const segmentAnchorKind = String(segment.anchor_kind || currentSegment.anchorKind || '').trim().toLowerCase()
    const segmentAnchorText = normalizeStrictAnchorText(
      String(segment.anchor_text || currentSegment.anchorText || ''),
    )
    const segmentEquationNumber = Number.isFinite(Number(segment.equation_number || currentSegment.equationNumber || 0))
      ? Math.max(0, Math.floor(Number(segment.equation_number || currentSegment.equationNumber || 0)))
      : 0
    const blockIdsRaw = [
      ...[primaryBlockId].filter(Boolean),
      ...supportBlockIdsRaw.map((item) => String(item || '').trim()).filter(Boolean),
      ...evidenceBlockIdsRaw.map((item) => String(item || '').trim()).filter(Boolean),
    ]
    if (evidenceMode !== 'direct' || blockIdsRaw.length <= 0) continue
    if (claimType === 'shell_sentence' && !mustLocate) continue

    const sourceSegmentId = String(segment.segment_id || '').trim() || `seg_${idx + 1}`
    const evidenceQuote = normalizeStrictAnchorText(String(segment.evidence_quote || segmentAnchorText || ''))
    const headingLikeQuote = claimType === 'quote_claim' && isHeadingLikeQuotedAnchor(segmentAnchorText || evidenceQuote || currentSegment.text)
    if (headingLikeQuote) continue
    const hasStrictIdentity = hasSegmentStrictLocateIdentity(segment, currentSegment)
    if (!hasStrictIdentity) continue
    const isRequiredPolicy = locatePolicy === 'required'
    const effectiveMustLocate = strictIdentityReady && (mustLocate || isRequiredPolicy) && !headingLikeQuote
    if (claimGroupKind === 'formula_bundle' && (locateSurfacePolicy === 'hidden' || formulaOrigin === 'derived')) {
      continue
    }
    const keepSelfTarget = effectiveMustLocate || ['quote_claim', 'blockquote_claim', 'formula_claim', 'inline_formula_claim', 'equation_explanation_claim', 'figure_claim'].includes(claimType)
    const targetSegmentId = String(
      segment.claim_group_target_segment_id
      || currentSegment.claimGroupTargetSegmentId
      || currentSegment.segmentId
      || sourceSegmentId,
    ).trim() || sourceSegmentId
    const targetDistanceRaw = Number(
      segment.claim_group_target_distance
      ?? currentSegment.claimGroupTargetDistance
      ?? 0,
    )
    const targetDistance = Number.isFinite(targetDistanceRaw) && targetDistanceRaw > 0
      ? Math.max(0, Math.floor(targetDistanceRaw))
      : 0
    const targetSegment = provenanceById.get(targetSegmentId) || currentSegment
    const segmentId = String(targetSegment.segmentId || sourceSegmentId).trim() || sourceSegmentId
    if (seenSegment.has(segmentId)) continue

    const sourceSegmentText = stripMarkdownInline(String(segment.text || '')).trim()
    const segmentText = stripMarkdownInline(
      String(
        (keepSelfTarget && segmentAnchorText)
        || targetSegment.anchorText
        || targetSegment.text
        || sourceSegmentText
        || '',
      ),
    ).trim()
    if (!segmentText) continue
    const targetSnippetAliases = Array.isArray(targetSegment.snippetAliases) ? targetSegment.snippetAliases : []
    const sourceSnippetAliases = Array.isArray(segment.snippet_aliases)
      ? segment.snippet_aliases.map((item) => String(item || ''))
      : []
    const snippetKey = normalizeStructuredLocateSnippet(
      String(targetSegment.snippetKey || segment.snippet_key || segmentText).trim(),
    )
    const snippetAliases = mergeStructuredSnippetAliases(
      targetSnippetAliases,
      sourceSnippetAliases,
      [segmentAnchorText],
      [segmentText],
    )
    const candidates: LocateCandidate[] = []
    const seenBlock = new Set<string>()
    for (const blockIdRaw of blockIdsRaw.slice(0, 5)) {
      const blockId = String(blockIdRaw || '').trim()
      if (!blockId || seenBlock.has(blockId)) continue
      const block = blockMap[blockId]
      if (!block || typeof block !== 'object') continue
      seenBlock.add(blockId)
      const blockText = stripMarkdownInline(String(block.text || '')).trim()
      const headingPath = String(block.heading_path || '').trim()
      const anchorId = String(block.anchor_id || '').trim()
      const blockKind = String(block.kind || '').trim().toLowerCase()
      let anchorKind = String(segmentAnchorKind || blockKind || '').trim().toLowerCase()
      if (blockKind === 'equation') anchorKind = 'equation'
      if (blockKind === 'figure') anchorKind = 'figure'
      const anchorNumberRaw = Number(segmentEquationNumber || block.number || 0)
      const focusSnippet = segmentAnchorText || evidenceQuote || blockText || segmentText || headingPath
      if (!focusSnippet) continue
      candidates.push({
        sourcePath,
        sourceName,
        headingPath,
        focusSnippet,
        matchText: [headingPath, segmentAnchorText || evidenceQuote || '', blockText || segmentText].filter(Boolean).join('\n'),
        sourceType: 'guide',
        blockId,
        anchorId: anchorId || undefined,
        anchorKind: anchorKind || undefined,
        anchorNumber: Number.isFinite(anchorNumberRaw) && anchorNumberRaw > 0
          ? Math.floor(anchorNumberRaw)
          : undefined,
      })
    }
    if (candidates.length <= 0) continue

    const rankedCandidates = [...candidates].sort((a, b) => {
      const scoreB = scoreStructuredPrimaryCandidate(b, {
        claimType,
        anchorKind: segmentAnchorKind,
        anchorText: segmentAnchorText,
        evidenceQuote,
        segmentText,
        equationNumber: segmentEquationNumber,
        primaryBlockId,
        primaryAnchorId,
      })
      const scoreA = scoreStructuredPrimaryCandidate(a, {
        claimType,
        anchorKind: segmentAnchorKind,
        anchorText: segmentAnchorText,
        evidenceQuote,
        segmentText,
        equationNumber: segmentEquationNumber,
        primaryBlockId,
        primaryAnchorId,
      })
      if (scoreB !== scoreA) return scoreB - scoreA
      return candidates.indexOf(a) - candidates.indexOf(b)
    })

    const primary = rankedCandidates[0]
    const alternatives = [
      primary,
      ...rankedCandidates.filter((cand) => cand !== primary),
    ]
    const entry: ProvenanceLocateEntry = {
      segmentId,
      label: shortSegmentLabel(segmentAnchorText || evidenceQuote || segmentText || primary.focusSnippet),
      segmentText,
      evidenceQuote,
      claimType,
      mustLocate: effectiveMustLocate,
      locatePolicy: locatePolicy || (effectiveMustLocate ? 'required' : ''),
      locateSurfacePolicy,
      claimGroupId,
      claimGroupKind,
      formulaOrigin,
      anchorKind: segmentAnchorKind || primary.anchorKind || '',
      anchorText: segmentAnchorText || evidenceQuote || '',
      equationNumber: segmentEquationNumber || primary.anchorNumber || 0,
      snippetKey,
      snippetAliases,
      primary,
      alternatives,
      relatedBlockIds: coerceStringArray((segment as Record<string, unknown>).related_block_ids, 8, 180),
      sourceSegmentId,
      groupLeadText: targetDistance > 0
        ? String(segment.claim_group_lead_text || currentSegment.claimGroupLeadText || sourceSegmentText || '').trim() || undefined
        : undefined,
      groupDistance: targetDistance,
    }
    const contentKey = normalizeLocateText(segmentAnchorText || evidenceQuote || segmentText || primary.focusSnippet).slice(0, 220)
    if (contentKey && seenContent.has(contentKey)) continue
    if (contentKey) seenContent.add(contentKey)
    const evidenceConfidenceRaw = Number(segment.evidence_confidence || 0)
    const evidenceConfidence = Number.isFinite(evidenceConfidenceRaw) ? evidenceConfidenceRaw : 0
    const segmentFormula = hasFormulaSignal(segmentText)
    const contentCoreScore = scoreLocateContentCore(segmentText, {
      kind: targetSegment.kind,
      segmentType: targetSegment.segmentType,
      evidenceMode: targetSegment.evidenceMode,
    })
    const score = evidenceConfidence
      + (segmentFormula ? 0.03 : 0)
      + Math.min(0.18, contentCoreScore * 0.16)
      - Math.min(0.16, targetDistance * 0.06)
      + (effectiveMustLocate ? 0.42 : 0)
      + (claimType === 'formula_claim' ? 0.18 : 0)
      + (claimType === 'inline_formula_claim' ? 0.17 : 0)
      + (claimType === 'equation_explanation_claim' ? 0.16 : 0)
      + (claimType === 'figure_claim' ? 0.16 : 0)
      + ((claimType === 'quote_claim' || claimType === 'blockquote_claim') ? 0.14 : 0)
      + (locateSurfacePolicy === 'primary' ? 0.18 : 0)
      + (locateSurfacePolicy === 'secondary' ? 0.08 : 0)
      + (formulaOrigin === 'source' ? 0.12 : 0)
      - (formulaOrigin === 'derived' ? 0.32 : 0)
      + ((segmentAnchorText || evidenceQuote).length >= 18 ? 0.05 : 0)
    scoredEntries.push({ entry, score, idx })
    seenSegment.add(segmentId)
  }
  if (scoredEntries.length <= 0) return []
  let ranked = scoredEntries
    .filter((item) => {
      if (item.score >= minConfidence) return true
      return Boolean(item.entry.mustLocate || item.entry.locatePolicy === 'required')
    })
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score
      return a.idx - b.idx
    })
  if (ranked.length <= 0) {
    ranked = [...scoredEntries].sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score
      return a.idx - b.idx
    })
  }
  const bestFormulaBundleByKey = new Map<string, (typeof ranked)[number]>()
  for (const item of ranked) {
    const groupKey = formulaBundleLocateGroupKey(item.entry)
    if (!groupKey) continue
    const prev = bestFormulaBundleByKey.get(groupKey)
    if (!prev) {
      bestFormulaBundleByKey.set(groupKey, item)
      continue
    }
    const itemRepScore = formulaBundleRepresentativeScore(item.entry)
    const prevRepScore = formulaBundleRepresentativeScore(prev.entry)
    if (itemRepScore !== prevRepScore) {
      if (itemRepScore > prevRepScore) bestFormulaBundleByKey.set(groupKey, item)
      continue
    }
    if (item.score !== prev.score) {
      if (item.score > prev.score) bestFormulaBundleByKey.set(groupKey, item)
      continue
    }
    if (item.idx < prev.idx) bestFormulaBundleByKey.set(groupKey, item)
  }
  if (bestFormulaBundleByKey.size > 0) {
    ranked = ranked.filter((item) => {
      const groupKey = formulaBundleLocateGroupKey(item.entry)
      if (!groupKey) return true
      return bestFormulaBundleByKey.get(groupKey) === item
    })
  }
  const mustLocateEntries = ranked.filter((item) => item.entry.mustLocate || item.entry.locatePolicy === 'required')
  const optionalEntries = ranked.filter((item) => !(item.entry.mustLocate || item.entry.locatePolicy === 'required'))
  const limited = [
    ...mustLocateEntries,
    ...optionalEntries.slice(0, Math.max(0, maxEntries - mustLocateEntries.length)),
  ]

  // Deduplicate by primary evidence block id to avoid repeating the same source block across multiple segments.
  // Prefer required/must-locate entries; otherwise prefer higher score.
  const bestByBlock = new Map<string, (typeof limited)[number]>()
  for (const item of limited) {
    const blockId = String(item.entry?.primary?.blockId || '').trim()
    const anchorId = String(item.entry?.primary?.anchorId || '').trim()
    const key = blockId || (anchorId ? `${item.entry.primary.sourcePath}::${anchorId}` : '')
    if (!key) continue
    const prev = bestByBlock.get(key)
    if (!prev) {
      bestByBlock.set(key, item)
      continue
    }
    const prevRequired = Boolean(prev.entry.mustLocate || prev.entry.locatePolicy === 'required')
    const curRequired = Boolean(item.entry.mustLocate || item.entry.locatePolicy === 'required')
    if (curRequired && !prevRequired) {
      bestByBlock.set(key, item)
      continue
    }
    if (curRequired === prevRequired) {
      if (item.score > prev.score + 1e-6) {
        bestByBlock.set(key, item)
        continue
      }
      if (Math.abs(item.score - prev.score) <= 1e-6 && item.idx < prev.idx) {
        bestByBlock.set(key, item)
        continue
      }
    }
  }
  const deduped = Array.from(bestByBlock.values())
  return deduped
    .sort((a, b) => a.idx - b.idx)
    .map((item) => item.entry)
}

function normalizeStructuredLocateKind(input: string): StructuredLocateKind | '' {
  const raw = String(input || '').trim().toLowerCase()
  if (!raw) return ''
  if (raw === 'equation' || raw === 'math') return 'equation'
  if (raw === 'figure' || raw === 'fig' || raw === 'image' || raw === 'img') return 'figure'
  if (raw === 'list_item' || raw === 'list-item' || raw === 'li') return 'list_item'
  if (raw === 'quote' || raw === 'quoted_text') return 'quote'
  if (raw === 'blockquote' || raw === 'bq') return 'blockquote'
  if (raw === 'paragraph' || raw === 'p') return 'paragraph'
  return ''
}

function buildRefsLocateCandidatesAll(refHits: RefHitLite[]): LocateCandidate[] {
  const out: LocateCandidate[] = []
  const seen = new Set<string>()
  const push = (candidate: LocateCandidate | null) => {
    if (!candidate) return
    if (out.length >= REF_LOCATE_CANDIDATE_LIMIT) return
    const sourcePath = String(candidate.sourcePath || '').trim()
    const matchText = String(candidate.matchText || '').trim()
    if (!sourcePath || !matchText) return
    const key = `${normalizeLocateText(sourcePath)}::${normalizeLocateText(candidate.headingPath || '')}::${normalizeLocateText(matchText).slice(0, 220)}`
    if (seen.has(key)) return
    seen.add(key)
    out.push(candidate)
  }

  for (const hit of refHits) {
    const ui = hit?.ui_meta || {}
    const meta = hit?.meta || {}
    const sourcePath = String(ui.source_path || meta.source_path || '').trim()
    if (!sourcePath) continue
    const sourceName = String(ui.display_name || '').trim() || sourcePath.split(/[\\/]/).pop() || 'paper'

    const headingCandidates = new Set<string>([
      String(ui.heading_path || '').trim(),
      String(ui.section_label || '').trim(),
      String(ui.subsection_label || '').trim(),
      String(meta.ref_best_heading_path || '').trim(),
      String(meta.heading_path || '').trim(),
    ].filter(Boolean))
    const anchorKind = String(ui.anchor_target_kind || '').trim().toLowerCase()
    const anchorNum = Number(ui.anchor_target_number || 0)
    for (const heading of coerceStringArray(meta.ref_headings, 8, 160)) {
      headingCandidates.add(String(heading || '').trim())
    }

    const refLocs = Array.isArray(meta.ref_locs) ? meta.ref_locs : []
    for (const loc0 of refLocs.slice(0, 10)) {
      const loc = (loc0 || {}) as Record<string, unknown>
      const headingPath = String(loc.heading_path || loc.heading || '').trim()
      if (headingPath) headingCandidates.add(headingPath)
      const locText = pickFirstRefText(loc)
      const locAnchorId = String(loc.anchor_id || loc.anchorId || '').trim()
      const locAnchorKind = String(loc.anchor_kind || loc.kind || anchorKind || '').trim().toLowerCase()
      const locAnchorNumber = Number(loc.anchor_number || loc.number || anchorNum || 0)
      if (locText) {
        push({
          sourcePath,
          sourceName,
          headingPath: headingPath || String(ui.heading_path || '').trim(),
          focusSnippet: locText,
          matchText: [headingPath, locText].filter(Boolean).join('\n'),
          sourceType: 'refs',
          anchorId: locAnchorId || undefined,
          anchorKind: locAnchorKind || undefined,
          anchorNumber: Number.isFinite(locAnchorNumber) && locAnchorNumber > 0 ? Math.floor(locAnchorNumber) : undefined,
        })
      }
    }

    const snippetSeeds = [
      ...coerceStringArray(ui.summary_line, 1, 360),
      ...coerceStringArray(ui.why_line, 1, 360),
      ...coerceStringArray(meta.ref_show_snippets, 4, 2600),
      ...coerceStringArray(meta.ref_snippets, 4, 2600),
      ...coerceStringArray(meta.ref_overview_snippets, 2, 2600),
    ]
    if (anchorKind === 'equation' && Number.isFinite(anchorNum) && anchorNum > 0) {
      snippetSeeds.push(
        `equation ${anchorNum}`,
        `eq ${anchorNum}`,
        `(${anchorNum})`,
      )
    }
    const headingFallback = Array.from(headingCandidates).find(Boolean) || ''
    for (const seed of snippetSeeds) {
      const pieces = buildGuideLocateCandidates(seed, sourcePath, sourceName, 'refs')
      if (pieces.length > 0) {
        for (const piece of pieces.slice(0, 40)) push(piece)
        continue
      }
      push({
        sourcePath,
        sourceName,
        headingPath: headingFallback,
        focusSnippet: seed,
        matchText: [headingFallback, seed].filter(Boolean).join('\n'),
        sourceType: 'refs',
        anchorKind: anchorKind || undefined,
        anchorNumber: Number.isFinite(anchorNum) && anchorNum > 0 ? Math.floor(anchorNum) : undefined,
      })
    }

    for (const headingPath of headingCandidates) {
      push({
        sourcePath,
        sourceName,
        headingPath,
        focusSnippet: headingPath,
        matchText: headingPath,
        sourceType: 'refs',
        anchorKind: anchorKind || undefined,
        anchorNumber: Number.isFinite(anchorNum) && anchorNum > 0 ? Math.floor(anchorNum) : undefined,
      })
    }
  }
  return out
}

function isPreferredStrictFigureRefSnippet(input: string): boolean {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return false
  if (/^(?:figure|图)\s*#?\s*\d{1,4}$/i.test(raw)) return true
  return false
}

function isHeadingLikeQuotedAnchor(text: string): boolean {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(text || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return true
  if (/[。！？!?;；:：]/.test(raw)) return false
  const latin = raw.match(/[A-Za-z]{3,}/g) || []
  if (latin.length > 0 && latin.length <= 8 && raw.length <= 80) {
    const verbLike = /\b(?:is|are|was|were|be|been|being|can|cannot|could|should|would|will|use|used|using|estimate|estimated|show|shown|train|training|feed|feeding|make|making|exploit|provide|compare)\b/i
    if (!verbLike.test(raw)) return true
  }
  return raw.length <= 28 && !/\d/.test(raw)
}

function scoreStructuredAnchorCompatibility(
  renderKindInput: string,
  entry: Pick<ProvenanceLocateEntry, 'anchorKind' | 'claimType' | 'mustLocate'> | null | undefined,
): number {
  const renderKind = normalizeStructuredLocateKind(renderKindInput)
  const anchorKind = String(entry?.anchorKind || '').trim().toLowerCase()
  const claimType = String(entry?.claimType || '').trim().toLowerCase()
  if (!renderKind) return 0
  if (claimType === 'equation_explanation_claim') {
    if (renderKind === 'paragraph' || renderKind === 'list_item') return 0.66
    if (renderKind === 'blockquote') return 0.18
    if (renderKind === 'equation') return 0.12
    return -0.4
  }
  if (claimType === 'inline_formula_claim' || anchorKind === 'inline_formula') {
    if (renderKind === 'equation') return 0.74
    if (renderKind === 'paragraph' || renderKind === 'list_item') return 0.58
    if (renderKind === 'blockquote') return 0.2
    return -0.28
  }
  if (anchorKind === 'blockquote' || claimType === 'blockquote_claim') {
    return renderKind === 'blockquote' ? 0.72 : -1.2
  }
  if (anchorKind === 'equation' || claimType === 'formula_claim') {
    if (renderKind === 'equation') return 0.86
    return -1.05
  }
  if (anchorKind === 'figure' || claimType === 'figure_claim') {
    if (renderKind === 'figure') return 0.8
    if (renderKind === 'paragraph' || renderKind === 'list_item') return 0.08
    return -0.92
  }
  if (anchorKind === 'quote' || claimType === 'quote_claim') {
    if (renderKind === 'quote') return 0.88
    if (renderKind === 'blockquote') return 0.58
    if (renderKind === 'paragraph' || renderKind === 'list_item') return 0.28
    return -0.5
  }
  return 0
}

function splitAnswerRenderSegments(answerMarkdown: string): StructuredRenderSegment[] {
  const lines = String(answerMarkdown || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')
  const segments: StructuredRenderSegment[] = []
  let buf: string[] = []
  let quoteBuf: string[] = []
  let inFence = false
  let inDisplayMath = false
  let mathBuf: string[] = []
  let order = 0

  const push = (kind: StructuredLocateKind, rawText: string) => {
    const text = stripMarkdownInline(String(rawText || '')).replace(/\s+/g, ' ').trim()
    if (text.length < 10) return
    order += 1
    segments.push({
      order,
      kind,
      text: text.slice(0, 1600),
      snippetKey: normalizeLocateText(text.slice(0, 360)),
    })
  }

  const flushParagraph = () => {
    if (buf.length <= 0) return
    push('paragraph', buf.join('\n'))
    buf = []
  }

  const flushBlockquote = () => {
    if (quoteBuf.length <= 0) return
    push('blockquote', quoteBuf.join('\n'))
    quoteBuf = []
  }

  const flushDisplayMath = () => {
    if (mathBuf.length <= 0) return
    push('equation', mathBuf.join('\n'))
    mathBuf = []
  }

  for (const raw of lines) {
    const line = String(raw || '')
    if (/^\s*(```+|~~~+)\s*$/.test(line)) {
      if (inFence) {
        inFence = false
        flushParagraph()
        flushBlockquote()
        flushDisplayMath()
      } else {
        flushParagraph()
        flushBlockquote()
        inFence = true
      }
      continue
    }
    if (inFence) continue
    const trimmed = line.trim()
    const imageMatch = trimmed.match(/^!\[([^\]]*)\]\([^)]+\)\s*$/)
    if (imageMatch) {
      flushParagraph()
      flushBlockquote()
      flushDisplayMath()
      push('figure', String(imageMatch[1] || '').trim() || trimmed)
      continue
    }
    const eqStart = /^\s*(?:\$\$|\\\[|\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\})/.test(line)
    const eqEnd = /(?:\$\$|\\\]|\\end\{(?:equation|align|gather|multline|eqnarray)\*?\})\s*$/.test(line)
    if (inDisplayMath) {
      mathBuf.push(line)
      if (eqEnd) {
        inDisplayMath = false
        flushDisplayMath()
      }
      continue
    }
    if (eqStart) {
      flushParagraph()
      flushBlockquote()
      mathBuf = [line]
      if (eqEnd && !/^\s*(?:\$\$|\\\[)\s*$/.test(line)) {
        flushDisplayMath()
      } else {
        inDisplayMath = true
      }
      continue
    }
    if (!line.trim()) {
      flushParagraph()
      flushBlockquote()
      flushDisplayMath()
      continue
    }
    if (/^\s{0,3}#{1,6}\s+/.test(line)) {
      flushParagraph()
      flushBlockquote()
      flushDisplayMath()
      continue
    }
    const listMatch = line.match(/^\s*(?:[-*+]|\d+[.)])\s+(.*)$/)
    if (listMatch) {
      flushParagraph()
      flushBlockquote()
      flushDisplayMath()
      push('list_item', String(listMatch[1] || ''))
      continue
    }
    if (/^\s*\|.*\|\s*$/.test(line)) {
      flushParagraph()
      flushBlockquote()
      flushDisplayMath()
      continue
    }
    const quoteMatch = line.match(/^\s*>\s?(.*)$/)
    if (quoteMatch) {
      flushParagraph()
      flushDisplayMath()
      quoteBuf.push(String(quoteMatch[1] || ''))
      continue
    }
    flushBlockquote()
    buf.push(line)
  }
  flushParagraph()
  flushBlockquote()
  flushDisplayMath()
  return segments
}

function listStructuredProvenanceSegments(
  messageProvenance: Record<string, unknown> | null,
): StructuredProvenanceSegment[] {
  if (!messageProvenance || typeof messageProvenance !== 'object') return []
  const segmentsRaw = Array.isArray(messageProvenance.segments) ? messageProvenance.segments : []
  const out: StructuredProvenanceSegment[] = []
  for (let idx = 0; idx < segmentsRaw.length; idx += 1) {
    const segment = segmentsRaw[idx] as Record<string, unknown> | null
    if (!segment || typeof segment !== 'object') continue
    const segmentId = String(segment.segment_id || '').trim() || `seg_${idx + 1}`
    const text = stripMarkdownInline(String(segment.text || '')).replace(/\s+/g, ' ').trim()
    const snippetKeyRaw = String(segment.snippet_key || '').trim()
    const snippetAliases = Array.isArray(segment.snippet_aliases)
      ? segment.snippet_aliases
        .map((item) => normalizeStructuredLocateSnippet(String(item || '').trim()))
        .filter(Boolean)
        .slice(0, 8)
      : []
    out.push({
      index: idx,
      segmentId,
      kind: String(segment.kind || '').trim().toLowerCase(),
      segmentType: String(segment.segment_type || '').trim().toLowerCase(),
      evidenceMode: String(segment.evidence_mode || '').trim().toLowerCase(),
      claimType: String(segment.claim_type || '').trim().toLowerCase(),
      mustLocate: Boolean(segment.must_locate),
      locatePolicy: String(segment.locate_policy || '').trim().toLowerCase(),
      locateSurfacePolicy: String(segment.locate_surface_policy || '').trim().toLowerCase(),
      claimGroupId: String(segment.claim_group_id || '').trim(),
      claimGroupKind: String(segment.claim_group_kind || '').trim().toLowerCase(),
      claimGroupTargetSegmentId: String(segment.claim_group_target_segment_id || '').trim(),
      claimGroupTargetDistance: Number.isFinite(Number(segment.claim_group_target_distance || 0))
        ? Math.max(0, Math.floor(Number(segment.claim_group_target_distance || 0)))
        : 0,
      claimGroupLeadText: stripMarkdownInline(String(segment.claim_group_lead_text || '')).replace(/\s+/g, ' ').trim(),
      formulaOrigin: String(segment.formula_origin || '').trim().toLowerCase(),
      anchorKind: String(segment.anchor_kind || '').trim().toLowerCase(),
      anchorText: stripMarkdownInline(String(segment.anchor_text || '')).replace(/\s+/g, ' ').trim(),
      equationNumber: Number.isFinite(Number(segment.equation_number || 0))
        ? Math.max(0, Math.floor(Number(segment.equation_number || 0)))
        : 0,
      text,
      snippetKey: normalizeStructuredLocateSnippet(snippetKeyRaw || text.slice(0, 360)),
      snippetAliases,
    })
  }
  return out
}

function scoreStructuredRenderBinding(
  renderSegment: StructuredRenderSegment,
  entry: ProvenanceLocateEntry,
  provenanceSegment: StructuredProvenanceSegment | null,
  targetOrder: number,
): number {
  const segText = String(provenanceSegment?.text || entry.segmentText || '').trim()
  const segKey = String(provenanceSegment?.snippetKey || entry.snippetKey || '').trim()
  let score = Math.max(
    scoreProvenanceSegment(renderSegment.text, segText, segKey),
    overlapScore(renderSegment.text, segText),
  )
  if (segKey && renderSegment.snippetKey === normalizeLocateText(segKey)) {
    score += 0.42
  }
  if (Array.isArray(entry.snippetAliases) && entry.snippetAliases.length > 0) {
    const aliasScore = entry.snippetAliases.reduce((acc, alias) => {
      return Math.max(acc, overlapScore(renderSegment.text, String(alias || '')))
    }, 0)
    score += 0.22 * aliasScore
  }
  if (Array.isArray(provenanceSegment?.snippetAliases) && provenanceSegment.snippetAliases.length > 0) {
    const aliasScore = provenanceSegment.snippetAliases.reduce((acc, alias) => {
      return Math.max(acc, overlapScore(renderSegment.text, String(alias || '')))
    }, 0)
    score += 0.14 * aliasScore
  }
  const figureNumbers = extractFigureNumbersFromText(
    `${entry.anchorText} ${entry.segmentText} ${provenanceSegment?.text || ''}`,
  )
  if (figureNumbers.length > 0) {
    score += 0.56 * figureNumberMatchScore(renderSegment.text, figureNumbers)
  }
  const renderKind = normalizeStructuredLocateKind(renderSegment.kind)
  const segKind = normalizeStructuredLocateKind(String(provenanceSegment?.kind || ''))
  const anchorCompat = scoreStructuredAnchorCompatibility(renderKind, entry)
  if (anchorCompat <= -0.9) return anchorCompat
  score += anchorCompat
  if (renderKind && segKind && renderKind === segKind) {
    score += 0.18
  }
  if (targetOrder > 0) {
    const distance = Math.abs(renderSegment.order - targetOrder)
    if (distance === 0) score += 0.26
    else score -= Math.min(0.48, distance * 0.1)
    if (distance <= 1) score += 0.05
  }
  return score
}

function buildStructuredRenderLocateSlotMap(
  answerMarkdown: string,
  messageProvenance: Record<string, unknown> | null,
  provenanceLocateEntries: ProvenanceLocateEntry[],
): Map<number, StructuredRenderLocateSlot> {
  const renderSegments = splitAnswerRenderSegments(answerMarkdown)
  const provenanceSegments = listStructuredProvenanceSegments(messageProvenance)
  if (renderSegments.length <= 0 || provenanceSegments.length <= 0 || provenanceLocateEntries.length <= 0) {
    return new Map()
  }

  const provenanceById = new Map(provenanceSegments.map((segment) => [segment.segmentId, segment]))
  const renderableOrdinalBySegmentId = new Map<string, number>()
  let renderableOrdinal = 0
  for (const segment of provenanceSegments) {
    if (normalizeStructuredLocateKind(segment.kind)) {
      renderableOrdinal += 1
      renderableOrdinalBySegmentId.set(segment.segmentId, renderableOrdinal)
    }
  }

  const slotMap = new Map<number, StructuredRenderLocateSlot>()
  const assignedOrders = new Set<number>()
  const orderedEntries = provenanceLocateEntries
    .map((entry, entryIndex) => ({
      entry,
      provenanceSegment: provenanceById.get(entry.segmentId) || null,
      entryIndex,
    }))
    .sort((a, b) => {
      const aIndex = a.provenanceSegment?.index ?? a.entryIndex
      const bIndex = b.provenanceSegment?.index ?? b.entryIndex
      return aIndex - bIndex
    })

  for (const item of orderedEntries) {
    const { entry, provenanceSegment } = item
    const targetOrder = Number(renderableOrdinalBySegmentId.get(entry.segmentId) || 0)
    const formulaQuery = hasFormulaSignal(entry.segmentText || provenanceSegment?.text || '')
    const figureQuery = String(entry.anchorKind || '').trim().toLowerCase() === 'figure'
      || String(entry.claimType || '').trim().toLowerCase() === 'figure_claim'
    let bestSegment: StructuredRenderSegment | null = null
    let bestScore = Number.NEGATIVE_INFINITY
    for (const renderSegment of renderSegments) {
      if (assignedOrders.has(renderSegment.order)) continue
      const score = scoreStructuredRenderBinding(renderSegment, entry, provenanceSegment, targetOrder)
      if (score > bestScore) {
        bestScore = score
        bestSegment = renderSegment
      }
    }
    if (!bestSegment) continue
    const distance = targetOrder > 0 ? Math.abs(bestSegment.order - targetOrder) : 0
    let floor = formulaQuery ? 0.3 : (figureQuery ? 0.26 : 0.44)
    if (targetOrder > 0 && distance === 0) floor -= 0.14
    else if (targetOrder > 0 && distance <= 1) floor -= 0.08
    if (bestScore < floor) continue
    assignedOrders.add(bestSegment.order)
    slotMap.set(bestSegment.order, {
      order: bestSegment.order,
      kind: bestSegment.kind,
      renderText: bestSegment.text,
      renderSnippetKey: bestSegment.snippetKey,
      entry,
      provenanceIndex: provenanceSegment?.index ?? item.entryIndex,
      score: bestScore,
    })
  }
  return slotMap
}

function resolveStructuredRenderLocateSlot(
  snippet: string,
  meta: LocateRenderMetaLite | undefined,
  slotMap: Map<number, StructuredRenderLocateSlot>,
): StructuredRenderLocateSlot | null {
  if (!(slotMap instanceof Map) || slotMap.size <= 0) return null
  const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || ''))).trim()
  const targetOrderRaw = Number(meta?.order || 0)
  const targetOrder = Number.isFinite(targetOrderRaw) && targetOrderRaw > 0 ? Math.floor(targetOrderRaw) : 0
  const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))

  const scoreSlot = (slot: StructuredRenderLocateSlot): number => {
    const compat = scoreStructuredAnchorCompatibility(targetKind || slot.kind, slot.entry)
    if (targetKind && compat <= -0.9) return Number.NEGATIVE_INFINITY
    let score = 0
    if (raw) {
      score = Math.max(
        scoreProvenanceSegment(raw, slot.renderText, slot.renderSnippetKey),
        scoreProvenanceSegment(raw, slot.entry.segmentText, slot.entry.snippetKey),
        overlapScore(raw, slot.renderText),
      )
      if (Array.isArray(slot.entry.snippetAliases) && slot.entry.snippetAliases.length > 0) {
        const aliasScore = slot.entry.snippetAliases.reduce((acc, alias) => {
          return Math.max(acc, overlapScore(raw, String(alias || '')))
        }, 0)
        score += 0.18 * aliasScore
      }
    }
    const figureNumbers = extractFigureNumbersFromText(`${raw} ${slot.entry.anchorText} ${slot.entry.segmentText}`)
    if (figureNumbers.length > 0) {
      score += 0.62 * Math.max(
        figureNumberMatchScore(raw, figureNumbers),
        figureNumberMatchScore(slot.renderText, figureNumbers),
      )
    }
    if (targetKind && slot.kind === targetKind) score += 0.12
    score += Math.max(-0.6, compat)
    if (targetOrder > 0) {
      const distance = Math.abs(slot.order - targetOrder)
      if (distance === 0) score += 0.5
      else score -= Math.min(0.44, distance * 0.18)
    }
    return score
  }

  if (targetOrder > 0) {
    const direct = slotMap.get(targetOrder)
    if (direct) {
      const directScore = scoreSlot(direct)
      const directFloor = raw ? (hasFormulaSignal(raw) ? 0.16 : 0.1) : -1
      if ((!targetKind || direct.kind === targetKind) && directScore >= directFloor) {
        return direct
      }
    }
  }

  let best: StructuredRenderLocateSlot | null = null
  let bestScore = Number.NEGATIVE_INFINITY
  for (const slot of slotMap.values()) {
    const score = scoreSlot(slot)
    if (score > bestScore) {
      best = slot
      bestScore = score
    }
  }
  if (!best) return null
  if (targetKind) {
    const compat = scoreStructuredAnchorCompatibility(targetKind, best.entry)
    if (compat <= -0.9) return null
  }
  const floor = raw ? (hasFormulaSignal(raw) ? 0.34 : 0.48) : 0.22
  return bestScore >= floor ? best : null
}

function resolveStructuredFallbackLocateEntry(
  snippet: string,
  meta: LocateRenderMetaLite | undefined,
  provenanceLocateEntries: ProvenanceLocateEntry[],
): ProvenanceLocateEntry | null {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || ''))).trim()
  const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
  if (!raw || provenanceLocateEntries.length <= 0) return null

  let best: ProvenanceLocateEntry | null = null
  let bestScore = Number.NEGATIVE_INFINITY
  for (const entry of provenanceLocateEntries) {
    const compat = scoreStructuredAnchorCompatibility(
      targetKind || normalizeStructuredLocateKind(String(entry.anchorKind || '')),
      entry,
    )
    if (targetKind && compat <= -0.9) continue
    let score = Math.max(
      scoreProvenanceSegment(raw, entry.segmentText, entry.snippetKey),
      overlapScore(raw, entry.anchorText || entry.segmentText),
    )
    if (Array.isArray(entry.snippetAliases) && entry.snippetAliases.length > 0) {
      const aliasScore = entry.snippetAliases.reduce((acc, alias) => {
        return Math.max(acc, overlapScore(raw, String(alias || '')))
      }, 0)
      score += 0.18 * aliasScore
    }
    const figureNumbers = extractFigureNumbersFromText(`${raw} ${entry.anchorText} ${entry.segmentText}`)
    if (figureNumbers.length > 0) {
      score += 0.72 * figureNumberMatchScore(`${entry.anchorText} ${entry.segmentText}`, figureNumbers)
    }
    if (targetKind && normalizeStructuredLocateKind(String(entry.anchorKind || '')) === targetKind) {
      score += 0.14
    }
    if (entry.mustLocate || entry.locatePolicy === 'required') {
      score += 0.08
    }
    score += Math.max(-0.4, compat)
    if (score > bestScore) {
      best = entry
      bestScore = score
    }
  }
  if (!best) return null
  const targetIsFigure = targetKind === 'figure'
  const floor = targetIsFigure ? 0.34 : (hasFormulaSignal(raw) ? 0.38 : 0.56)
  return bestScore >= floor ? best : null
}

function normalizeStructuredLocateSnippet(input: string): string {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(input || '')))
    .replace(/\s+/g, ' ')
    .trim()
  if (!raw) return ''
  const trimmed = raw
    .replace(/\.{3,}\s*$/, '')
    .replace(/\u2026+\s*$/, '')
    .trim()
  return normalizeLocateText(trimmed)
}

function extractFigureNumbersFromText(text: string): number[] {
  const src = String(text || '')
  if (!src) return []
  const out: number[] = []
  const seen = new Set<number>()
  const push = (raw: string) => {
    const n = Number(raw)
    if (!Number.isFinite(n) || n <= 0) return
    const k = Math.floor(n)
    if (seen.has(k)) return
    seen.add(k)
    out.push(k)
  }
  for (const m of src.matchAll(/\b(?:fig(?:ure)?\.?\s*#?\s*(\d{1,4})|图\s*(\d{1,4}))\b/gi)) {
    push(String(m[1] || m[2] || ''))
  }
  return out
}

function figureNumberMatchScore(text: string, numbers: number[]): number {
  const src = String(text || '')
  if (!src || numbers.length <= 0) return 0
  let best = 0
  for (const num of numbers) {
    if (new RegExp(`\\bfig(?:ure)?\\.?\\s*#?\\s*${num}\\b`, 'i').test(src)) best = Math.max(best, 1.0)
    if (new RegExp(`图\\s*${num}\\b`).test(src)) best = Math.max(best, 1.0)
  }
  return best
}

function scoreProvenanceSegment(snippet: string, segmentText: string, segmentKey: string): number {
  const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || '')))
  const query = normalizeLocateText(raw)
  const segNorm = normalizeLocateText(stripProvenanceNoise(String(segmentText || '')))
  const keyNorm = normalizeLocateText(stripProvenanceNoise(String(segmentKey || '')))
  if (!query || (!segNorm && !keyNorm)) return 0

  let score = 0
  if (keyNorm) {
    if (keyNorm === query) score += 1.2
    if (keyNorm.includes(query)) score += 0.92
    if (query.includes(keyNorm) && keyNorm.length >= 20) score += 0.78
  }
  if (segNorm) {
    if (segNorm === query) score += 1.15
    if (segNorm.includes(query)) score += 0.86
    const segHead = segNorm.slice(0, Math.min(220, segNorm.length))
    if (query.includes(segHead) && segHead.length >= 20) score += 0.72
  }

  score += 0.82 * Math.max(
    overlapScore(raw, segmentText),
    overlapScore(query, segNorm),
    overlapScore(query, keyNorm),
  )

  if (hasFormulaSignal(raw) || hasFormulaSignal(segmentText)) {
    score += 0.72 * formulaOverlapScore(raw, segmentText)
  }
  return score
}

function extractEquationNumbersFromText(text: string): number[] {
  const src = String(text || '')
  if (!src) return []
  const out: number[] = []
  const seen = new Set<number>()
  const push = (raw: string) => {
    const n = Number(raw)
    if (!Number.isFinite(n) || n <= 0) return
    const k = Math.floor(n)
    if (seen.has(k)) return
    seen.add(k)
    out.push(k)
  }
  for (const m of src.matchAll(/\b(?:eq|equation|\u516C\u5F0F)\s*[#(\uFF08]?\s*(\d{1,4})\s*[)\uFF09]?/gi)) {
    push(String(m[1] || ''))
  }
  for (const m of src.matchAll(/\((\d{1,4})\)/g)) {
    push(String(m[1] || ''))
  }
  return out
}

function scoreLocateCandidate(snippet: string, cand: LocateCandidate): number {
  const query = String(snippet || '').trim()
  if (!query) return 0
  const qNorm = normalizeLocateText(query)
  const focusText = String(cand.focusSnippet || '').trim()
  const matchText = String(cand.matchText || focusText).trim()
  if (!matchText) return 0
  const mNorm = normalizeLocateText(matchText)
  const fNorm = normalizeLocateText(focusText)

  let score = Math.max(
    overlapScore(query, matchText),
    overlapScore(query, focusText) * 1.08,
  )

  if (qNorm && mNorm) {
    if (mNorm.includes(qNorm)) score += 0.7
    const qHead = qNorm.slice(0, Math.min(64, qNorm.length))
    if (qHead.length >= 18 && mNorm.includes(qHead)) score += 0.26
    const mHead = mNorm.slice(0, Math.min(64, mNorm.length))
    if (mHead.length >= 18 && qNorm.includes(mHead)) score += 0.18
  }
  if (qNorm && fNorm) {
    if (fNorm.includes(qNorm)) score += 0.2
    const qHead = qNorm.slice(0, Math.min(48, qNorm.length))
    if (qHead.length >= 16 && fNorm.includes(qHead)) score += 0.14
  }

  const tokenSet = new Set(tokenizeLocateText(matchText))
  const keyTokens = Array.from(new Set(tokenizeLocateText(query))).filter((token) => token.length >= 3)
  let hitCount = 0
  for (const token of keyTokens) {
    if (tokenSet.has(token)) hitCount += 1
  }
  if (hitCount > 0) {
    score += Math.min(0.36, 0.03 * hitCount)
  }
  if (query.length >= 80 && focusText.length >= 80) {
    score += 0.05
  }
  if (hasFormulaSignal(query) || hasFormulaSignal(matchText)) {
    score += 0.72 * formulaOverlapScore(query, matchText)
  }
  const qEqNumbers = extractEquationNumbersFromText(query)
  const candEqNo = Number(cand.anchorNumber || 0)
  const candKind = String(cand.anchorKind || '').trim().toLowerCase()
  if (qEqNumbers.length > 0) {
    if (candEqNo > 0 && qEqNumbers.includes(candEqNo)) score += 1.05
    if (candKind === 'equation') score += 0.18
  }
  if (hasFormulaSignal(query) && candKind === 'equation') {
    score += 0.22
  }
  if (cand.anchorId) {
    score += 0.04
  }
  if (cand.sourceType === 'guide') {
    score += 0.07
  }
  return score
}

function scoreStructuredPrimaryCandidate(
  cand: LocateCandidate,
  opts: {
    claimType?: string
    anchorKind?: string
    anchorText?: string
    evidenceQuote?: string
    segmentText?: string
    equationNumber?: number
    primaryBlockId?: string
    primaryAnchorId?: string
  },
): number {
  const claimType = String(opts.claimType || '').trim().toLowerCase()
  const anchorKind = String(opts.anchorKind || '').trim().toLowerCase()
  const anchorText = String(opts.anchorText || '').trim()
  const evidenceQuote = String(opts.evidenceQuote || '').trim()
  const segmentText = String(opts.segmentText || '').trim()
  const seed = anchorText || evidenceQuote || segmentText || String(cand.focusSnippet || '').trim()
  let score = scoreLocateCandidate(seed, cand)

  const candKind = String(cand.anchorKind || '').trim().toLowerCase()
  const candHeading = String(cand.headingPath || '').trim().toLowerCase()
  const candNumber = Number.isFinite(Number(cand.anchorNumber || 0))
    ? Math.max(0, Math.floor(Number(cand.anchorNumber || 0)))
    : 0
  const equationNumber = Number.isFinite(Number(opts.equationNumber || 0))
    ? Math.max(0, Math.floor(Number(opts.equationNumber || 0)))
    : 0

  if (opts.primaryBlockId && String(cand.blockId || '').trim() === String(opts.primaryBlockId || '').trim()) {
    score += 0.12
  }
  if (opts.primaryAnchorId && String(cand.anchorId || '').trim() === String(opts.primaryAnchorId || '').trim()) {
    score += 0.08
  }
  if (anchorKind && candKind === anchorKind) {
    score += 0.42
  }

  if (claimType === 'formula_claim') {
    if (candKind === 'equation') score += 1.55
    else if (candKind) score -= 0.72
    if (equationNumber > 0 && candNumber === equationNumber) score += 0.95
    if (candHeading.includes('figure')) score -= 0.26
  } else if (claimType === 'inline_formula_claim') {
    if (candKind === 'equation') score += 1.1
    else if (candKind === 'paragraph' || candKind === 'list_item' || candKind === 'blockquote') score += 0.58
  } else if (claimType === 'equation_explanation_claim') {
    if (candKind === 'equation') score -= 0.62
    if (candKind === 'paragraph' || candKind === 'list_item' || candKind === 'blockquote') score += 0.74
    if (equationNumber > 0 && candNumber === equationNumber) score += 0.08
  } else if (claimType === 'figure_claim') {
    if (candKind === 'figure') score += 1.18
    else if (candKind) score -= 0.34
  } else if (claimType === 'quote_claim') {
    if (candKind === 'quote') score += 1.02
    else if (candKind === 'blockquote') score += 0.48
    else if (candKind) score -= 0.2
  } else if (claimType === 'blockquote_claim') {
    if (candKind === 'blockquote') score += 1.0
    else if (candKind === 'quote') score += 0.42
    else if (candKind) score -= 0.18
  } else if (claimType === 'method_detail' || claimType === 'prior_work' || claimType === 'doc_map') {
    if (candKind === 'paragraph' || candKind === 'list_item' || candKind === 'blockquote') score += 0.34
    if (candKind === 'equation' || candKind === 'figure') score -= 0.28
  }

  return score
}

function isImageOnlyPlaceholder(content: string) {
  return /^\[Image attachment x\d+\]$/i.test(String(content || '').trim())
}

function imageAttachmentsOf(message: Message): ChatImageAttachment[] {
  return Array.isArray(message.attachments)
    ? message.attachments.filter((item): item is ChatImageAttachment => Boolean(item && item.path))
    : []
}

function AssistantAvatar() {
  return (
    <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center overflow-hidden rounded-full border border-[var(--border)] bg-white/90 dark:bg-white/5">
      <img src="/pi_logo.png" alt="Pi assistant" className="h-5 w-5 object-contain" loading="lazy" />
    </div>
  )
}

const shelfStorageFallback = new Map<string, string>()
interface ShelfSnapshot {
  version: number
  revision: number
  updatedAt: number
  open: boolean
  items: CiteShelfItem[]
}
interface ShelfSavedSnapshot {
  id: string
  name: string
  createdAt: number
  items: CiteShelfItem[]
}
interface ShelfSavedSnapshotPayload {
  version: number
  updatedAt: number
  snapshots: ShelfSavedSnapshot[]
}
const shelfSnapshotMemory = new Map<string, ShelfSnapshot>()
const shelfSavedSnapshotMemory = new Map<string, ShelfSavedSnapshotPayload>()

function cloneShelfSnapshot(snapshot: ShelfSnapshot): ShelfSnapshot {
  return {
    version: Number(snapshot.version || 0),
    revision: Number(snapshot.revision || 0),
    updatedAt: Number(snapshot.updatedAt || 0),
    open: Boolean(snapshot.open),
    items: (snapshot.items || []).map((item) => ({ ...item })),
  }
}

function cloneSavedSnapshot(snapshot: ShelfSavedSnapshot): ShelfSavedSnapshot {
  return {
    id: String(snapshot.id || ''),
    name: String(snapshot.name || ''),
    createdAt: Number(snapshot.createdAt || 0),
    items: (snapshot.items || []).map((item) => ({ ...item })),
  }
}

function cloneSavedSnapshotPayload(payload: ShelfSavedSnapshotPayload): ShelfSavedSnapshotPayload {
  return {
    version: Number(payload.version || 0),
    updatedAt: Number(payload.updatedAt || 0),
    snapshots: (payload.snapshots || []).map((entry) => cloneSavedSnapshot(entry)),
  }
}

function listShelfStorages(): Storage[] {
  const out: Storage[] = []
  try {
    out.push(window.localStorage)
  } catch {
    // ignore
  }
  try {
    if (!out.includes(window.sessionStorage)) out.push(window.sessionStorage)
  } catch {
    // ignore
  }
  return out
}

function readShelfStorage(key: string): string {
  for (const storage of listShelfStorages()) {
    try {
      const raw = storage.getItem(key)
      if (typeof raw === 'string') {
        shelfStorageFallback.set(key, raw)
        return raw
      }
    } catch {
      // ignore
    }
  }
  return shelfStorageFallback.get(key) || ''
}

function writeShelfStorage(key: string, payload: string) {
  // Always keep in-memory raw snapshot first to survive temporary storage failures.
  shelfStorageFallback.set(key, payload)
  let wrote = false
  for (const storage of listShelfStorages()) {
    try {
      storage.setItem(key, payload)
      wrote = true
    } catch {
      // ignore
    }
  }
  if (!wrote) return
}

function removeShelfStorage(key: string) {
  shelfSnapshotMemory.delete(key)
  for (const storage of listShelfStorages()) {
    try {
      storage.removeItem(key)
    } catch {
      // ignore
    }
  }
  shelfStorageFallback.delete(key)
}

function shelfSavedStorageKey(convId?: string | null): string {
  return `${shelfStorageKey(convId)}${SHELF_SAVED_SUFFIX}`
}

function normalizeDoiLike(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
    .replace(/^[\s"'`([{<]+|[\s"'`)\]}>.,;:]+$/g, '')
    .trim()
}

function normalizeTitleLike(value: string): string {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function shelfPaperIdentity(item: CiteShelfItem): string {
  const doi = normalizeDoiLike(item.doi || item.doiUrl)
  if (doi) return `doi:${doi}`
  const title = normalizeTitleLike(item.title || item.main)
  const year = /^\d{4}$/.test(String(item.year || '').trim()) ? String(item.year).trim() : ''
  if (title) return `title:${title}|${year}`
  return `key:${String(item.key || '').trim()}`
}

function dedupeShelfItems(items: CiteShelfItem[]): CiteShelfItem[] {
  const seen = new Set<string>()
  const out: CiteShelfItem[] = []
  for (const item of items || []) {
    const key = shelfPaperIdentity(item)
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push(item)
    if (out.length >= SHELF_MAX_ITEMS) break
  }
  return out
}

function restoreShelfItems(rawItems: unknown[]): CiteShelfItem[] {
  const seen = new Set<string>()
  const seenIdentity = new Set<string>()
  const out: CiteShelfItem[] = []
  for (const raw of rawItems) {
    if (!raw || typeof raw !== 'object') continue
    const rec = raw as Record<string, unknown>
    const detail = normalizeCiteDetail(rec)
    if (!detail) continue
    const base = toShelfItem(detail)
    const key = String(rec.key || '').trim() || base.key
    const main = String(rec.main || '').trim() || base.main
    if (!key || seen.has(key)) continue
    const identity = shelfPaperIdentity({ ...base, key, main })
    if (seenIdentity.has(identity)) continue
    seen.add(key)
    seenIdentity.add(identity)
    out.push({
      ...base,
      key,
      main,
      tags: normalizeShelfTags(rec.tags),
      note: normalizeShelfNote(rec.note),
    })
    if (out.length >= SHELF_MAX_ITEMS) break
  }
  return out
}

function readShelfSnapshot(key: string, rawOverride?: string): ShelfSnapshot | null {
  if (typeof rawOverride !== 'string') {
    const mem = shelfSnapshotMemory.get(key)
    if (mem) return cloneShelfSnapshot(mem)
  }
  const raw = typeof rawOverride === 'string' ? rawOverride : readShelfStorage(key)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw)
    const itemsRaw: unknown[] = Array.isArray(parsed?.items) ? parsed.items : []
    const revision0 = Number(parsed?.revision || 0)
    const updatedAt0 = Number(parsed?.updatedAt || 0)
    const snapshot: ShelfSnapshot = {
      version: Number(parsed?.version || 0) || 0,
      revision: Number.isFinite(revision0) && revision0 > 0 ? Math.floor(revision0) : 0,
      updatedAt: Number.isFinite(updatedAt0) && updatedAt0 > 0 ? Math.floor(updatedAt0) : 0,
      open: Boolean(parsed?.open),
      items: restoreShelfItems(itemsRaw),
    }
    shelfSnapshotMemory.set(key, snapshot)
    return cloneShelfSnapshot(snapshot)
  } catch {
    // Corrupted payload: keep running and clear stale bad data.
    shelfSnapshotMemory.delete(key)
    removeShelfStorage(key)
    return null
  }
}

function removeSavedShelfStorage(key: string) {
  shelfSavedSnapshotMemory.delete(key)
  for (const storage of listShelfStorages()) {
    try {
      storage.removeItem(key)
    } catch {
      // ignore
    }
  }
  shelfStorageFallback.delete(key)
}

function readSavedShelfSnapshots(storageKey: string, rawOverride?: string): ShelfSavedSnapshot[] {
  if (typeof rawOverride !== 'string') {
    const mem = shelfSavedSnapshotMemory.get(storageKey)
    if (mem) return cloneSavedSnapshotPayload(mem).snapshots
  }
  const raw = typeof rawOverride === 'string' ? rawOverride : readShelfStorage(storageKey)
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    const snapshotsRaw: unknown[] = Array.isArray(parsed?.snapshots) ? parsed.snapshots : []
    const seen = new Set<string>()
    const snapshots: ShelfSavedSnapshot[] = []
    for (const rawItem of snapshotsRaw) {
      if (!rawItem || typeof rawItem !== 'object') continue
      const rec = rawItem as Record<string, unknown>
      const id = String(rec.id || '').trim()
      if (!id || seen.has(id)) continue
      const createdAt0 = Number(rec.createdAt || 0)
      const createdAt = Number.isFinite(createdAt0) && createdAt0 > 0 ? Math.floor(createdAt0) : Date.now()
      const name = String(rec.name || '').trim() || 'Untitled snapshot'
      const itemsRaw: unknown[] = Array.isArray(rec.items) ? rec.items : []
      snapshots.push({
        id,
        name,
        createdAt,
        items: restoreShelfItems(itemsRaw),
      })
      seen.add(id)
      if (snapshots.length >= SHELF_SAVED_MAX_ITEMS) break
    }
    const payload: ShelfSavedSnapshotPayload = {
      version: Number(parsed?.version || 0) || 0,
      updatedAt: Number(parsed?.updatedAt || 0) || 0,
      snapshots,
    }
    shelfSavedSnapshotMemory.set(storageKey, payload)
    return cloneSavedSnapshotPayload(payload).snapshots
  } catch {
    removeSavedShelfStorage(storageKey)
    return []
  }
}

function persistSavedShelfSnapshots(storageKey: string, snapshots: ShelfSavedSnapshot[]) {
  if (!Array.isArray(snapshots) || snapshots.length <= 0) {
    removeSavedShelfStorage(storageKey)
    return
  }
  const normalized = snapshots
    .slice(0, SHELF_SAVED_MAX_ITEMS)
    .map((entry) => ({
      id: String(entry.id || '').trim(),
      name: String(entry.name || '').trim() || 'Untitled snapshot',
      createdAt: Number(entry.createdAt || 0) > 0 ? Number(entry.createdAt) : Date.now(),
      items: dedupeShelfItems(entry.items || []).slice(0, SHELF_MAX_ITEMS).map((item) => ({ ...item })),
    }))
    .filter((entry) => Boolean(entry.id))
  if (normalized.length <= 0) {
    removeSavedShelfStorage(storageKey)
    return
  }
  const payload: ShelfSavedSnapshotPayload = {
    version: SHELF_SAVED_SCHEMA_VERSION,
    updatedAt: Date.now(),
    snapshots: normalized,
  }
  shelfSavedSnapshotMemory.set(storageKey, cloneSavedSnapshotPayload(payload))
  writeShelfStorage(storageKey, JSON.stringify(payload))
}

function isWeakTitle(text: string): boolean {
  const t = String(text || '').trim()
  if (!t) return true
  if (/(?:\bet\s+al\b|doi[:\s]|^https?:\/\/)/i.test(t)) return true
  if (/\bIn\s+[A-Z]/.test(t)) return true
  const tokens = t.match(/[A-Za-z0-9\u4e00-\u9fff]+/g) || []
  return tokens.length <= 2
}

function isWeakAuthors(text: string): boolean {
  const t = String(text || '').trim()
  if (!t) return true
  if (/\b(?:journal|conference|proceedings|vol\.?|pp\.?)\b/i.test(t)) return true
  const tokens = t.match(/[A-Za-z\u4e00-\u9fff]+/g) || []
  return tokens.length <= 1
}

function isWeakVenue(text: string): boolean {
  const t = String(text || '').trim()
  if (!t) return true
  const tokens = t.match(/[A-Za-z0-9\u4e00-\u9fff]+/g) || []
  return tokens.length <= 1
}

function preferRicherField(field: 'title' | 'authors' | 'venue' | 'year' | 'main', current: string, incoming: string): string {
  const cur = String(current || '').trim()
  const inc = String(incoming || '').trim()
  if (!cur) return inc
  if (!inc) return cur
  if (field === 'year') {
    const curOk = /^\d{4}$/.test(cur)
    const incOk = /^\d{4}$/.test(inc)
    if (curOk && !incOk) return cur
    if (!curOk && incOk) return inc
    return cur
  }
  if (field === 'title') {
    const curWeak = isWeakTitle(cur)
    const incWeak = isWeakTitle(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  } else if (field === 'authors') {
    const curWeak = isWeakAuthors(cur)
    const incWeak = isWeakAuthors(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  } else if (field === 'venue') {
    const curWeak = isWeakVenue(cur)
    const incWeak = isWeakVenue(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  } else if (field === 'main') {
    const curWeak = isWeakTitle(cur)
    const incWeak = isWeakTitle(inc)
    if (curWeak && !incWeak) return inc
    if (!curWeak && incWeak) return cur
  }
  return inc.length > cur.length + 12 ? inc : cur
}

function sameTags(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false
  }
  return true
}

function sameShelfItem(a: CiteShelfItem, b: CiteShelfItem): boolean {
  return (
    a.key === b.key
    && a.main === b.main
    && a.traceConvId === b.traceConvId
    && a.traceAssistantMsgId === b.traceAssistantMsgId
    && a.traceAssistantOrder === b.traceAssistantOrder
    && a.traceUserMsgId === b.traceUserMsgId
    && a.sourceName === b.sourceName
    && a.sourcePath === b.sourcePath
    && a.raw === b.raw
    && a.citeFmt === b.citeFmt
    && a.title === b.title
    && a.authors === b.authors
    && a.venue === b.venue
    && a.year === b.year
    && a.volume === b.volume
    && a.issue === b.issue
    && a.pages === b.pages
    && a.doi === b.doi
    && a.doiUrl === b.doiUrl
    && a.citationCount === b.citationCount
    && a.citationSource === b.citationSource
    && a.journalIf === b.journalIf
    && a.journalQuartile === b.journalQuartile
    && a.journalIfSource === b.journalIfSource
    && a.venueKind === b.venueKind
    && a.venueVerifiedBy === b.venueVerifiedBy
    && a.openalexVenue === b.openalexVenue
    && a.conferenceTier === b.conferenceTier
    && a.conferenceRankSource === b.conferenceRankSource
    && a.conferenceCcf === b.conferenceCcf
    && a.conferenceCcfSource === b.conferenceCcfSource
    && a.conferenceName === b.conferenceName
    && a.conferenceAcronym === b.conferenceAcronym
    && a.num === b.num
    && a.anchor === b.anchor
    && a.bibliometricsChecked === b.bibliometricsChecked
    && a.summaryLine === b.summaryLine
    && a.summarySource === b.summarySource
    && a.note === b.note
    && sameTags(a.tags || [], b.tags || [])
  )
}

function sameShelfItems(a: CiteShelfItem[], b: CiteShelfItem[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (!sameShelfItem(a[i], b[i])) return false
  }
  return true
}

function persistShelfSnapshot(
  storageKey: string,
  payload: { open: boolean; items: CiteShelfItem[] },
  currentRevision: number,
): number {
  const normalizedItems = payload.items.slice(0, SHELF_MAX_ITEMS)
  const existing = readShelfSnapshot(storageKey)
  if (existing && existing.open === payload.open && sameShelfItems(existing.items, normalizedItems)) {
    return Math.max(currentRevision, existing.revision)
  }
  const nextRevision = Math.max(currentRevision, existing?.revision || 0) + 1
  const snapshot: ShelfSnapshot = {
    version: SHELF_SCHEMA_VERSION,
    revision: nextRevision,
    updatedAt: Date.now(),
    open: payload.open,
    items: normalizedItems,
  }
  shelfSnapshotMemory.set(storageKey, cloneShelfSnapshot(snapshot))
  const raw = JSON.stringify(snapshot)
  writeShelfStorage(storageKey, raw)
  return nextRevision
}

function preferExistingText(current: string, incoming: string): string {
  const cur = String(current || '').trim()
  if (cur) return cur
  return String(incoming || '').trim()
}

function preferPositiveNumber(current: number, incoming: number): number {
  const cur = Number(current || 0)
  if (Number.isFinite(cur) && cur > 0) return cur
  const inc = Number(incoming || 0)
  if (Number.isFinite(inc) && inc > 0) return inc
  return 0
}

function mergeShelfItemWithLive(item: CiteShelfItem, live: CiteShelfItem): CiteShelfItem {
  const mergedLike = {
    ...item,
    traceConvId: preferExistingText(item.traceConvId, live.traceConvId),
    traceAssistantMsgId: preferPositiveNumber(item.traceAssistantMsgId, live.traceAssistantMsgId),
    traceAssistantOrder: preferPositiveNumber(item.traceAssistantOrder, live.traceAssistantOrder),
    traceUserMsgId: preferPositiveNumber(item.traceUserMsgId, live.traceUserMsgId),
    sourceName: preferExistingText(item.sourceName, live.sourceName),
    sourcePath: preferExistingText(item.sourcePath, live.sourcePath),
    raw: preferExistingText(item.raw, live.raw),
    citeFmt: preferExistingText(item.citeFmt, live.citeFmt),
    title: preferRicherField('title', item.title, live.title),
    authors: preferRicherField('authors', item.authors, live.authors),
    venue: preferRicherField('venue', item.venue, live.venue),
    year: preferRicherField('year', item.year, live.year),
    volume: preferExistingText(item.volume, live.volume),
    issue: preferExistingText(item.issue, live.issue),
    pages: preferExistingText(item.pages, live.pages),
    doi: preferExistingText(item.doi, live.doi),
    doiUrl: preferExistingText(item.doiUrl, live.doiUrl),
    citationSource: preferExistingText(item.citationSource, live.citationSource),
    venueKind: preferExistingText(item.venueKind, live.venueKind),
    venueVerifiedBy: preferExistingText(item.venueVerifiedBy, live.venueVerifiedBy),
    openalexVenue: preferExistingText(item.openalexVenue, live.openalexVenue),
    journalIf: preferExistingText(item.journalIf, live.journalIf),
    journalQuartile: preferExistingText(item.journalQuartile, live.journalQuartile),
    journalIfSource: preferExistingText(item.journalIfSource, live.journalIfSource),
    conferenceTier: preferExistingText(item.conferenceTier, live.conferenceTier),
    conferenceRankSource: preferExistingText(item.conferenceRankSource, live.conferenceRankSource),
    conferenceCcf: preferExistingText(item.conferenceCcf, live.conferenceCcf),
    conferenceCcfSource: preferExistingText(item.conferenceCcfSource, live.conferenceCcfSource),
    conferenceName: preferExistingText(item.conferenceName, live.conferenceName),
    conferenceAcronym: preferExistingText(item.conferenceAcronym, live.conferenceAcronym),
    summaryLine: preferRicherField('title', item.summaryLine, live.summaryLine),
    summarySource: preferExistingText(item.summarySource, live.summarySource),
    citationCount: preferPositiveNumber(item.citationCount, live.citationCount),
    num: preferPositiveNumber(item.num, live.num),
    bibliometricsChecked: Boolean(item.bibliometricsChecked || live.bibliometricsChecked),
  }

  const normalized = normalizeCiteDetail(mergedLike) || item
  const autoMain = toShelfItem(normalized).main
  return {
    ...item,
    ...normalized,
    key: item.key,
    main: preferRicherField('main', item.main, preferExistingText(live.main, autoMain)),
    tags: normalizeShelfTags(item.tags),
    note: normalizeShelfNote(item.note),
  }
}

function normalizeTextLite(value: string): string {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function firstAuthorToken(value: string): string {
  const firstChunk = String(value || '')
    .split(/[;,\uFF0C\uFF1B]/)[0]
    ?.trim()
    || ''
  const tokens = firstChunk.match(/[A-Za-z\u4e00-\u9fff]+/g) || []
  if (tokens.length <= 0) return ''
  const first = tokens.at(0)
  return first ? first.toLowerCase() : ''
}

function year4(value: string): string {
  const match = String(value || '').match(/\b(19|20)\d{2}\b/)
  return match ? match[0] : ''
}

function jaccardTokens(a: string, b: string): number {
  const aSet = new Set(normalizeTextLite(a).split(' ').filter(Boolean))
  const bSet = new Set(normalizeTextLite(b).split(' ').filter(Boolean))
  if (aSet.size <= 0 || bSet.size <= 0) return 0
  let inter = 0
  for (const t of aSet) {
    if (bSet.has(t)) inter += 1
  }
  const union = aSet.size + bSet.size - inter
  return union > 0 ? inter / union : 0
}

function strictRepairMerge(base: CiteShelfItem, candidateMeta: Record<string, unknown>): CiteShelfItem | null {
  if (!candidateMeta || Object.keys(candidateMeta).length <= 0) return null
  const merged = mergeCiteMeta(base, candidateMeta)
  const mergedItem = {
    ...toShelfItem(merged),
    key: base.key,
    tags: normalizeShelfTags(base.tags),
    note: normalizeShelfNote(base.note),
  }

  const baseDoi = normalizeDoiLike(base.doi || base.doiUrl)
  const mergedDoi = normalizeDoiLike(mergedItem.doi || mergedItem.doiUrl)
  if (baseDoi && mergedDoi && baseDoi !== mergedDoi) return null

  const titleSignal = jaccardTokens(base.title || base.main, mergedItem.title || mergedItem.main) >= 0.55
  const authorSignal = (
    Boolean(firstAuthorToken(base.authors))
    && firstAuthorToken(base.authors) === firstAuthorToken(mergedItem.authors)
  )
  const yearSignal = Boolean(year4(base.year) && year4(base.year) === year4(mergedItem.year))
  const venueSignal = jaccardTokens(base.venue, mergedItem.venue) >= 0.5
  const newDoiSignal = !baseDoi && Boolean(mergedDoi)

  let signalCount = 0
  if (titleSignal) signalCount += 1
  if (authorSignal) signalCount += 1
  if (yearSignal) signalCount += 1
  if (venueSignal) signalCount += 1

  const accepted = newDoiSignal ? signalCount >= 1 : signalCount >= 2
  if (!accepted) return null
  return mergedItem
}

function snapshotDiffCounts(currentItems: CiteShelfItem[], baselineItems: CiteShelfItem[]): { added: number; removed: number } {
  const current = new Set(currentItems.map((item) => shelfPaperIdentity(item)))
  const baseline = new Set(baselineItems.map((item) => shelfPaperIdentity(item)))
  let added = 0
  let removed = 0
  for (const id of current) {
    if (!baseline.has(id)) added += 1
  }
  for (const id of baseline) {
    if (!current.has(id)) removed += 1
  }
  return { added, removed }
}

export function MessageList({
  activeConvId,
  messages,
  refs,
  generationPartial,
  generationStage,
  jumpTarget,
  onJumpHandled,
  trackedMessageIds,
  onTrackedMessageActive,
  onOpenReader,
  paperGuideSourcePath,
  paperGuideSourceName,
}: Props) {
  const createPaperGuideConversation = useChatStore((s) => s.createPaperGuideConversation)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [popoverDetail, setPopoverDetail] = useState<CiteDetail | null>(null)
  const [popoverPos, setPopoverPos] = useState<{ x: number; y: number } | null>(null)
  const [popoverLoading, setPopoverLoading] = useState(false)
  const [popoverGuideLoading, setPopoverGuideLoading] = useState(false)
  const [shelfOpen, setShelfOpen] = useState(false)
  const [shelfItems, setShelfItems] = useState<CiteShelfItem[]>([])
  const [focusedShelfKey, setFocusedShelfKey] = useState('')
  const [shelfSummaryLoadingKey, setShelfSummaryLoadingKey] = useState('')
  const [shelfRepairLoadingKey, setShelfRepairLoadingKey] = useState('')
  const [savedShelfSnapshots, setSavedShelfSnapshots] = useState<ShelfSavedSnapshot[]>([])
  const [selectedSavedSnapshotId, setSelectedSavedSnapshotId] = useState('')
  const assistantLocatePrepCacheRef = useRef(new Map<string, AssistantLocatePrep>())
  const assistantLocatePrepPerfRef = useRef<MessageListPrepPerfEvent | null>(null)
  const [guideDocCandidates, setGuideDocCandidates] = useState<LocateCandidate[]>([])
  const skipShelfPersistOnceRef = useRef(false)
  const persistShelfTimerRef = useRef<number | null>(null)
  const activeStorageKeyRef = useRef(shelfStorageKey(activeConvId))
  const shelfRevisionByKeyRef = useRef<Record<string, number>>({})
  const latestShelfStateRef = useRef<{ convId?: string | null; open: boolean; items: CiteShelfItem[] }>({
    convId: activeConvId,
    open: false,
    items: [],
  })
  const flushShelfSnapshotRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    const sourcePath = String(paperGuideSourcePath || '').trim()
    const sourceName = String(paperGuideSourceName || '').trim()
    if (!sourcePath) {
      setGuideDocCandidates([])
      return
    }
    let cancelled = false
    referencesApi.readerDoc(sourcePath)
      .then((res) => {
        if (cancelled) return
        const markdown = String(res.markdown || '')
        if (!markdown.trim()) {
          setGuideDocCandidates([])
          return
        }
        const resolvedName = String(res.source_name || sourceName || '').trim()
        const anchors = Array.isArray(res.anchors) ? res.anchors : []
        setGuideDocCandidates(
          buildGuideLocateCandidates(
            markdown,
            sourcePath,
            resolvedName || sourceName || sourcePath,
            'guide',
            anchors,
          ),
        )
      })
      .catch(() => {
        if (!cancelled) setGuideDocCandidates([])
      })
    return () => {
      cancelled = true
    }
  }, [paperGuideSourcePath, paperGuideSourceName])

  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const timer = window.requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight
    })
    return () => window.cancelAnimationFrame(timer)
  }, [activeConvId, generationPartial])

  useEffect(() => {
    if (!jumpTarget || !Number.isFinite(jumpTarget.messageId)) return
    const el = scrollRef.current
    if (!el) return
    const target = el.querySelector<HTMLElement>(`[data-msg-id="${jumpTarget.messageId}"]`)
    if (!target) return
    const targetRect = target.getBoundingClientRect()
    const containerRect = el.getBoundingClientRect()
    const top = targetRect.top - containerRect.top + el.scrollTop - 12
    el.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
    try {
      target.animate(
        [
          { boxShadow: '0 0 0 0 rgba(24,144,255,0.0)', backgroundColor: 'rgba(24,144,255,0.0)' },
          { boxShadow: '0 0 0 3px rgba(24,144,255,0.24)', backgroundColor: 'rgba(24,144,255,0.10)' },
          { boxShadow: '0 0 0 0 rgba(24,144,255,0.0)', backgroundColor: 'rgba(24,144,255,0.0)' },
        ],
        { duration: 900, easing: 'ease-out' },
      )
    } catch {
      // no-op: Web Animations may not be available in all envs.
    }
    onJumpHandled?.(jumpTarget)
  }, [jumpTarget, messages, onJumpHandled])

  useEffect(() => {
    if (!onTrackedMessageActive) return
    const el = scrollRef.current
    if (!el) return
    const trackedIds = Array.isArray(trackedMessageIds)
      ? trackedMessageIds.filter((id) => Number.isFinite(id))
      : []
    if (trackedIds.length <= 0) {
      onTrackedMessageActive(null)
      return
    }
    let syncFrameId = 0
    let measureFrameId = 0
    let lastReported: number | null = null
    let lastActiveIndex = -1
    let lastScrollTop = el.scrollTop
    let trackedAnchors: Array<{ id: number; top: number }> = []
    const SWITCH_HYSTERESIS_PX = 28

    const transitionMargin = (leftIndex: number, rightIndex: number) => {
      const leftTop = trackedAnchors[leftIndex]?.top ?? 0
      const rightTop = trackedAnchors[rightIndex]?.top ?? leftTop
      const gap = Math.max(0, rightTop - leftTop)
      return Math.min(SWITCH_HYSTERESIS_PX, Math.max(10, gap * 0.2))
    }

    const syncActiveMessage = () => {
      syncFrameId = 0
      if (trackedAnchors.length <= 0) {
        lastActiveIndex = -1
        if (lastReported !== null) {
          lastReported = null
          onTrackedMessageActive(null)
        }
        return
      }

      const currentScrollTop = el.scrollTop
      const anchorTop = currentScrollTop + Math.min(120, Math.max(48, el.clientHeight * 0.2))
      let low = 0
      let high = trackedAnchors.length - 1
      let activeIndex = 0
      while (low <= high) {
        const mid = Math.floor((low + high) / 2)
        if (trackedAnchors[mid]!.top <= anchorTop) {
          activeIndex = mid
          low = mid + 1
        } else {
          high = mid - 1
        }
      }
      if (lastActiveIndex >= 0 && lastActiveIndex < trackedAnchors.length && activeIndex !== lastActiveIndex) {
        const direction = currentScrollTop - lastScrollTop
        if (activeIndex === lastActiveIndex + 1 && direction >= 0) {
          const nextTop = trackedAnchors[activeIndex]?.top ?? 0
          if (anchorTop < nextTop + transitionMargin(lastActiveIndex, activeIndex)) {
            activeIndex = lastActiveIndex
          }
        } else if (activeIndex === lastActiveIndex - 1 && direction <= 0) {
          const currentTop = trackedAnchors[lastActiveIndex]?.top ?? 0
          if (anchorTop >= currentTop - transitionMargin(activeIndex, lastActiveIndex)) {
            activeIndex = lastActiveIndex
          }
        }
      }
      const activeMessageId = trackedAnchors[activeIndex]?.id ?? null
      lastScrollTop = currentScrollTop
      lastActiveIndex = activeMessageId != null ? activeIndex : -1

      if (activeMessageId !== lastReported) {
        lastReported = activeMessageId
        onTrackedMessageActive(activeMessageId)
      }
    }

    const scheduleSync = () => {
      if (syncFrameId) return
      syncFrameId = window.requestAnimationFrame(syncActiveMessage)
    }

    const measureTrackedAnchors = () => {
      measureFrameId = 0
      const containerRect = el.getBoundingClientRect()
      const currentScrollTop = el.scrollTop
      trackedAnchors = trackedIds
        .map((id) => {
          const node = el.querySelector<HTMLElement>(`[data-msg-id="${id}"]`)
          if (!node) return null
          const rect = node.getBoundingClientRect()
          return {
            id,
            top: rect.top - containerRect.top + currentScrollTop,
          }
        })
        .filter((item): item is { id: number; top: number } => Boolean(item))
        .sort((left, right) => left.top - right.top)
      if (lastReported != null) {
        lastActiveIndex = trackedAnchors.findIndex((item) => item.id === lastReported)
      } else {
        lastActiveIndex = -1
      }
      scheduleSync()
    }

    const scheduleMeasure = () => {
      if (measureFrameId) return
      measureFrameId = window.requestAnimationFrame(measureTrackedAnchors)
    }

    const resizeObserver = typeof ResizeObserver !== 'undefined'
      ? new ResizeObserver(() => {
        scheduleMeasure()
      })
      : null

    el.addEventListener('scroll', scheduleSync, { passive: true })
    window.addEventListener('resize', scheduleMeasure)
    resizeObserver?.observe(el)
    if (el.firstElementChild instanceof HTMLElement) {
      resizeObserver?.observe(el.firstElementChild)
    }
    scheduleMeasure()

    return () => {
      el.removeEventListener('scroll', scheduleSync)
      window.removeEventListener('resize', scheduleMeasure)
      resizeObserver?.disconnect()
      if (syncFrameId) {
        window.cancelAnimationFrame(syncFrameId)
      }
      if (measureFrameId) {
        window.cancelAnimationFrame(measureFrameId)
      }
    }
  }, [messages, onTrackedMessageActive, trackedMessageIds])

  useEffect(() => {
    const nextStorageKey = shelfStorageKey(activeConvId)
    const nextSavedStorageKey = shelfSavedStorageKey(activeConvId)
    const prevStorageKey = activeStorageKeyRef.current
    if (persistShelfTimerRef.current !== null) {
      window.clearTimeout(persistShelfTimerRef.current)
      persistShelfTimerRef.current = null
    }
    if (prevStorageKey !== nextStorageKey) {
      const prevRevision = Number(shelfRevisionByKeyRef.current[prevStorageKey] || 0)
      const flushedRevision = persistShelfSnapshot(
        prevStorageKey,
        { open: shelfOpen, items: shelfItems },
        prevRevision,
      )
      shelfRevisionByKeyRef.current[prevStorageKey] = flushedRevision
    }

    // Switching conversation changes storage key; skip one persist cycle to avoid
    // writing previous conversation shelf state into the new key before hydration.
    skipShelfPersistOnceRef.current = true
    const savedSnapshots = readSavedShelfSnapshots(nextSavedStorageKey)
    setSavedShelfSnapshots(savedSnapshots)
    setSelectedSavedSnapshotId((current) => {
      if (current && savedSnapshots.some((item) => item.id === current)) return current
      return savedSnapshots[0]?.id || ''
    })
    const snapshot = readShelfSnapshot(nextStorageKey)
    if (!snapshot) {
      shelfRevisionByKeyRef.current[nextStorageKey] = 0
      setShelfItems([])
      setShelfOpen(false)
      setFocusedShelfKey('')
      activeStorageKeyRef.current = nextStorageKey
      return
    }
    shelfRevisionByKeyRef.current[nextStorageKey] = Math.max(0, snapshot.revision || 0)
    setShelfItems(snapshot.items)
    setShelfOpen(snapshot.open)
    setFocusedShelfKey('')
    activeStorageKeyRef.current = nextStorageKey
  }, [activeConvId])

  useEffect(() => {
    const storageKey = shelfStorageKey(activeConvId)
    const savedStorageKey = shelfSavedStorageKey(activeConvId)
    const onStorage = (event: StorageEvent) => {
      if (event.key === savedStorageKey) {
        if (event.newValue === null) {
          shelfSavedSnapshotMemory.delete(savedStorageKey)
          setSavedShelfSnapshots([])
          setSelectedSavedSnapshotId('')
          return
        }
        const snapshots = readSavedShelfSnapshots(savedStorageKey, event.newValue)
        setSavedShelfSnapshots(snapshots)
        setSelectedSavedSnapshotId((current) => {
          if (current && snapshots.some((item) => item.id === current)) return current
          return snapshots[0]?.id || ''
        })
        return
      }
      if (event.key !== storageKey) return
      if (event.newValue === null) {
        shelfSnapshotMemory.delete(storageKey)
        skipShelfPersistOnceRef.current = true
        shelfRevisionByKeyRef.current[storageKey] = 0
        setShelfItems([])
        setShelfOpen(false)
        setFocusedShelfKey('')
        return
      }
      const snapshot = readShelfSnapshot(storageKey, event.newValue)
      if (!snapshot) return
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      if (snapshot.revision <= currentRevision) return
      skipShelfPersistOnceRef.current = true
      shelfRevisionByKeyRef.current[storageKey] = snapshot.revision
      setShelfItems(snapshot.items)
      setShelfOpen(snapshot.open)
      setFocusedShelfKey('')
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [activeConvId])

  useLayoutEffect(() => {
    latestShelfStateRef.current = { convId: activeConvId, open: shelfOpen, items: shelfItems }
  }, [activeConvId, shelfItems, shelfOpen])

  useEffect(() => {
    flushShelfSnapshotRef.current = () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
      const latest = latestShelfStateRef.current
      const storageKey = shelfStorageKey(latest.convId)
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      const nextRevision = persistShelfSnapshot(
        storageKey,
        { open: latest.open, items: latest.items },
        currentRevision,
      )
      shelfRevisionByKeyRef.current[storageKey] = nextRevision
      activeStorageKeyRef.current = storageKey
    }
    return () => {
      if (flushShelfSnapshotRef.current) {
        flushShelfSnapshotRef.current()
      }
      flushShelfSnapshotRef.current = null
    }
  }, [])

  useEffect(() => {
    setSelectedSavedSnapshotId((current) => {
      if (current && savedShelfSnapshots.some((item) => item.id === current)) return current
      return savedShelfSnapshots[0]?.id || ''
    })
  }, [savedShelfSnapshots])

  useEffect(() => {
    return () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    const flushShelfSnapshot = () => {
      flushShelfSnapshotRef.current?.()
    }
    window.addEventListener('pagehide', flushShelfSnapshot)
    window.addEventListener('beforeunload', flushShelfSnapshot)
    return () => {
      window.removeEventListener('pagehide', flushShelfSnapshot)
      window.removeEventListener('beforeunload', flushShelfSnapshot)
    }
  }, [])

  useEffect(() => {
    if (skipShelfPersistOnceRef.current) {
      skipShelfPersistOnceRef.current = false
      return
    }
    const storageKey = shelfStorageKey(activeConvId)
    if (persistShelfTimerRef.current !== null) {
      window.clearTimeout(persistShelfTimerRef.current)
      persistShelfTimerRef.current = null
    }
    persistShelfTimerRef.current = window.setTimeout(() => {
      const latest = latestShelfStateRef.current
      const latestStorageKey = shelfStorageKey(latest.convId)
      if (latestStorageKey !== storageKey) {
        persistShelfTimerRef.current = null
        return
      }
      const currentRevision = Number(shelfRevisionByKeyRef.current[storageKey] || 0)
      const nextRevision = persistShelfSnapshot(
        storageKey,
        { open: latest.open, items: latest.items },
        currentRevision,
      )
      shelfRevisionByKeyRef.current[storageKey] = nextRevision
      persistShelfTimerRef.current = null
    }, 80)
    return () => {
      if (persistShelfTimerRef.current !== null) {
        window.clearTimeout(persistShelfTimerRef.current)
        persistShelfTimerRef.current = null
      }
    }
  }, [activeConvId, shelfItems, shelfOpen])

  const rows = useMemo(() => {
    const out: Array<
      | { kind: 'message'; message: Message }
      | { kind: 'refs'; userMsgId: number }
    > = []
    let lastUserMsgId = 0
    const renderedRefs = new Set<number>()

    for (const message of messages) {
      out.push({ kind: 'message', message })
      if (message.role === 'user') {
        lastUserMsgId = message.id
        continue
      }
      if (lastUserMsgId > 0 && !renderedRefs.has(lastUserMsgId) && hasRenderableRefs(refs, lastUserMsgId)) {
        out.push({ kind: 'refs', userMsgId: lastUserMsgId })
        renderedRefs.add(lastUserMsgId)
      }
    }

    return out
  }, [messages, refs])

  const assistantTraceByMsgId = useMemo(() => {
    const out = new Map<number, { answerOrder: number; userMsgId: number }>()
    let answerOrder = 0
    let lastUserMsgId = 0
    for (const message of messages) {
      if (message.role === 'user') {
        lastUserMsgId = message.id
        continue
      }
      if (message.role !== 'assistant') continue
      answerOrder += 1
      out.set(message.id, { answerOrder, userMsgId: lastUserMsgId })
    }
    return out
  }, [messages])

  const liveCiteMap = useMemo(() => {
    const map = new Map<string, CiteShelfItem>()
    const convTraceId = String(activeConvId || '')
    for (const message of messages) {
      if (message.role !== 'assistant' || !Array.isArray(message.cite_details)) continue
      const trace = assistantTraceByMsgId.get(message.id)
      for (const rawDetail of message.cite_details) {
        const detail = normalizeCiteDetail(rawDetail)
        if (!detail) continue
        const tracedDetail: CiteDetail = {
          ...detail,
          traceConvId: convTraceId,
          traceAssistantMsgId: message.id,
          traceAssistantOrder: Number(trace?.answerOrder || 0),
          traceUserMsgId: Number(trace?.userMsgId || 0),
        }
        const item = toShelfItem(tracedDetail)
        map.set(item.key, item)
      }
    }
    return map
  }, [activeConvId, assistantTraceByMsgId, messages])

  useEffect(() => {
    setShelfItems((current) => {
      let changed = false
      const next = current.map((item) => {
        const live = liveCiteMap.get(item.key)
        if (!live) return item
        const merged = mergeShelfItemWithLive(item, live)
        if (!sameShelfItem(merged, item)) {
          changed = true
          return merged
        }
        return item
      })
      const deduped = dedupeShelfItems(next)
      if (deduped.length !== current.length) changed = true
      return changed ? deduped : current
    })
  }, [liveCiteMap])

  const fetchShelfSummaryForItem = (item: CiteShelfItem) => {
    const summaryLine = String(item.summaryLine || '').trim()
    const summarySource = String(item.summarySource || '').trim().toLowerCase()
    if (summaryLine && (summarySource === 'abstract' || summarySource === 'fulltext')) return
    setShelfSummaryLoadingKey(item.key)
    referencesApi.bibliometricsCached(item as unknown as Record<string, unknown>)
      .then((meta) => {
        if (!meta || Object.keys(meta).length === 0) return
        setShelfItems((current) => current.map((entry) => {
          if (entry.key !== item.key) return entry
          const merged = mergeCiteMeta(entry, meta)
          return {
            ...toShelfItem(merged),
            key: entry.key,
            tags: normalizeShelfTags(entry.tags),
            note: normalizeShelfNote(entry.note),
          }
        }))
      })
      .finally(() => {
        setShelfSummaryLoadingKey((current) => (current === item.key ? '' : current))
      })
  }

  const repairShelfItemMeta = (item: CiteShelfItem) => {
    if (shelfRepairLoadingKey === item.key) return
    setShelfRepairLoadingKey(item.key)
    const basePayload = item as unknown as Record<string, unknown>
    const strictTitlePayload: Record<string, unknown> = {
      ...basePayload,
      raw: '',
      cite_fmt: '',
      citeFmt: '',
    }
    Promise.all([
      referencesApi.bibliometrics(basePayload).catch(() => ({})),
      referencesApi.bibliometrics(strictTitlePayload).catch(() => ({})),
    ])
      .then((metas) => {
        const candidates = metas.filter((meta) => meta && Object.keys(meta).length > 0)
        let didUpdate = false
        setShelfItems((current) => current.map((entry) => {
          if (entry.key !== item.key) return entry
          for (const meta of candidates) {
            const accepted = strictRepairMerge(entry, meta)
            if (!accepted) continue
            if (!sameShelfItem(accepted, entry)) {
              didUpdate = true
              return accepted
            }
            return entry
          }
          return entry
        }))
        if (didUpdate) message.success('Metadata repaired with strict rules')
        else message.info('Strict match did not pass; original metadata kept')
      })
      .catch(() => {
        message.error('Repair failed, please retry.')
      })
      .finally(() => {
        setShelfRepairLoadingKey((current) => (current === item.key ? '' : current))
      })
  }

  const openCitation = (detail: CiteDetail, event: MouseEvent<HTMLElement>) => {
    setPopoverDetail(detail)
    setPopoverPos({ x: event.clientX, y: event.clientY })
    setPopoverGuideLoading(false)
    const sourcePath = String(detail.sourcePath || '').trim()
    const isInPaperReference = Number(detail.num || 0) > 0
    const shouldFetchCitationMeta = Boolean(sourcePath) && !isInPaperReference
    const hasDoi = Boolean(String(detail.doi || '').trim())
    const shouldFetchBibliometrics = !detail.bibliometricsChecked && (
      isInPaperReference
        ? hasDoi
        : (detail.doi || detail.title || detail.venue || detail.raw || detail.citeFmt)
    )
    if (!shouldFetchCitationMeta && !shouldFetchBibliometrics) {
      setPopoverLoading(false)
      return
    }

    const reqs: Array<Promise<Record<string, unknown>>> = []
    if (shouldFetchCitationMeta && sourcePath) {
      reqs.push(referencesApi.citationMetaCached(sourcePath).catch(() => ({})))
    }
    if (shouldFetchBibliometrics) {
      reqs.push(referencesApi.bibliometricsCached(detail as unknown as Record<string, unknown>).catch(() => ({})))
    }

    const itemKey = toShelfItem(detail).key
    setPopoverLoading(true)
    Promise.all(reqs)
      .then((metas) => {
        setPopoverDetail((current) => {
          if (!current) return current
          let merged = current
          for (const meta of metas) {
            if (meta && Object.keys(meta).length > 0) {
              merged = mergeCiteMeta(merged, meta)
            }
          }
          return merged
        })
        setShelfItems((current) => current.map((item) => {
          if (item.key !== itemKey) return item
          let merged: CiteDetail = item
          for (const meta of metas) {
            if (meta && Object.keys(meta).length > 0) {
              merged = mergeCiteMeta(merged, meta)
            }
          }
          return {
            ...toShelfItem(merged),
            tags: normalizeShelfTags(item.tags),
            note: normalizeShelfNote(item.note),
          }
        }))
      })
      .finally(() => setPopoverLoading(false))
  }

  const addToShelf = (detail: CiteDetail) => {
    const item = toShelfItem(detail)
    setShelfItems((current) => {
      const identity = shelfPaperIdentity(item)
      const existing = current.find((entry) => entry.key === item.key || shelfPaperIdentity(entry) === identity)
      const mergedIncoming = existing ? mergeShelfItemWithLive(existing, item) : item
      const next = [
        {
          ...mergedIncoming,
          tags: normalizeShelfTags(existing?.tags || mergedIncoming.tags),
          note: normalizeShelfNote(existing?.note || mergedIncoming.note),
        },
        ...current.filter((entry) => entry.key !== item.key && shelfPaperIdentity(entry) !== identity),
      ]
      const deduped = dedupeShelfItems(next)
      if (deduped.length > SHELF_MAX_ITEMS) return deduped.slice(0, SHELF_MAX_ITEMS)
      if (deduped.length === next.length) return deduped.slice(0, SHELF_MAX_ITEMS)
      return deduped
    })
    setFocusedShelfKey(item.key)
    setShelfOpen(true)
    fetchShelfSummaryForItem(item)
  }

  const startPaperGuideFromDetail = async (detail: CiteDetail) => {
    const isInPaperReference = Number(detail.num || 0) > 0
    if (isInPaperReference) {
      message.info('This item is an in-answer citation. Upload the PDF in Library first, then start paper guide.')
      return
    }
    const sourcePath = String(detail.sourcePath || '').trim()
    if (!sourcePath) {
      message.info('Current citation has no bindable source path')
      return
    }
    const sourceName = String(detail.sourceName || detail.title || '').trim() || sourcePath.split(/[\\/]/).pop() || 'paper'
    setPopoverGuideLoading(true)
    try {
      await createPaperGuideConversation({
        sourcePath,
        sourceName,
        title: `Paper Guide 闂?${sourceName}`,
      })
      message.success('Entered paper guide conversation')
      setPopoverDetail(null)
      setPopoverPos(null)
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Failed to create paper guide conversation')
    } finally {
      setPopoverGuideLoading(false)
    }
  }

  const openReaderFromDetail = (detail: CiteDetail) => {
    if (!onOpenReader) return
    const sourcePath = String(detail.sourcePath || '').trim()
    if (!sourcePath) {
      message.info('Current citation has no bindable source path')
      return
    }
    const sourceName = String(detail.sourceName || detail.title || '').trim() || sourcePath.split(/[\\/]/).pop() || 'paper'
    const snippet = String(detail.summaryLine || detail.title || detail.raw || '').trim()
    onOpenReader({
      sourcePath,
      sourceName,
      snippet,
    })
  }

  const selectedSavedSnapshot = useMemo(
    () => savedShelfSnapshots.find((item) => item.id === selectedSavedSnapshotId) || null,
    [savedShelfSnapshots, selectedSavedSnapshotId],
  )

  const selectedSnapshotDiff = useMemo(() => {
    if (!selectedSavedSnapshot) return ''
    const diff = snapshotDiffCounts(shelfItems, selectedSavedSnapshot.items)
    if (diff.added <= 0 && diff.removed <= 0) return 'No diff from current shelf'
    return `Compared with current: +${diff.added} / -${diff.removed}`
  }, [selectedSavedSnapshot, shelfItems])

  const guideSourcePathSet = useMemo(() => {
    const out = new Set<string>()
    for (const item of guideDocCandidates) {
      const sourcePath = String(item.sourcePath || '').trim()
      if (sourcePath) out.add(sourcePath)
    }
    return out
  }, [guideDocCandidates])

  const guideDocCandidatesBySourcePath = useMemo(() => {
    const out = new Map<string, LocateCandidate[]>()
    for (const item of guideDocCandidates) {
      const sourcePath = String(item.sourcePath || '').trim()
      if (!sourcePath) continue
      const list = out.get(sourcePath) || []
      list.push(item)
      out.set(sourcePath, list)
    }
    return out
  }, [guideDocCandidates])

  const assistantLocatePrepByMsgId = useMemo(() => {
    const nextCache = new Map<string, AssistantLocatePrep>()
    const out = new Map<number, AssistantLocatePrep>()
    const guideSourcePath = String(paperGuideSourcePath || '').trim()
    const guideSourceName = String(paperGuideSourceName || '').trim()
    const prepStartedAt = messageListPerfNow()
    let assistantCount = 0
    let heavyCount = 0
    let lightCount = 0
    let cacheHits = 0
    for (const message of messages) {
      if (message.role !== 'assistant') continue
      assistantCount += 1
      const trace = assistantTraceByMsgId.get(message.id)
      const bodyContent = message.rendered_body || message.rendered_content || message.content
      const refsUserMsgId = Number(message.refs_user_msg_id || trace?.userMsgId || 0)
      const refEntry = refsUserMsgId > 0 ? (refs[String(refsUserMsgId)] as RefEntryLite | undefined) : undefined
      const refHits = Array.isArray(refEntry?.hits) ? refEntry.hits : []
      const hasRawCiteDetails = Array.isArray(message.cite_details) && message.cite_details.length > 0
      const hasProvenancePayload = Boolean(message.provenance && typeof message.provenance === 'object')
      const shouldBuildLocatePrep = Boolean(onOpenReader) && (
        Boolean(guideSourcePath)
        || hasRawCiteDetails
        || refHits.length > 0
        || hasProvenancePayload
      )
      if (!shouldBuildLocatePrep) {
        const prepKey = [
          message.id,
          String(message.render_cache_key || ''),
          'light',
          refsUserMsgId,
        ].join('::')
        const cached = assistantLocatePrepCacheRef.current.get(prepKey)
        if (cached) {
          cacheHits += 1
          nextCache.set(prepKey, cached)
          out.set(message.id, cached)
          continue
        }
        const prep = createEmptyAssistantLocatePrep(bodyContent, refsUserMsgId)
        lightCount += 1
        nextCache.set(prepKey, prep)
        out.set(message.id, prep)
        continue
      }
      const citeDetails = Array.isArray(message.cite_details)
        ? message.cite_details
          .map(normalizeCiteDetail)
          .filter((detail): detail is CiteDetail => Boolean(detail))
          .map((detail) => ({
            ...detail,
            traceConvId: String(activeConvId || ''),
            traceAssistantMsgId: message.id,
            traceAssistantOrder: Number(trace?.answerOrder || 0),
            traceUserMsgId: Number(trace?.userMsgId || 0),
          }))
        : []
      const uniqueSourcePaths = Array.from(
        new Set(
          citeDetails
            .map((detail) => String(detail.sourcePath || '').trim())
            .filter(Boolean),
        ),
      )
      const guideDocAvailable = Boolean(guideSourcePath && guideSourcePathSet.has(guideSourcePath))
      const locateSourcePath = (
        guideSourcePath && guideDocAvailable
          ? guideSourcePath
          : (uniqueSourcePaths.length === 1 ? uniqueSourcePaths[0] : guideSourcePath)
      )
      const locateSourceName = (
        (guideSourcePath && guideDocAvailable ? guideSourceName : '')
        || (citeDetails.find((detail) => String(detail.sourcePath || '').trim() === locateSourcePath)?.sourceName || '')
        || guideSourceName
      )
      const refSig = `${refsUserMsgId}:${String((refEntry as { prompt_sig?: string } | undefined)?.prompt_sig || '')}:${Number((refEntry as { updated_at?: number } | undefined)?.updated_at || 0)}:${refHits.length}`
      const prepKey = [
        message.id,
        String(message.render_cache_key || ''),
        guideSourcePath,
        locateSourcePath,
        refSig,
      ].join('::')
      const cached = assistantLocatePrepCacheRef.current.get(prepKey)
      if (cached) {
        cacheHits += 1
        nextCache.set(prepKey, cached)
        out.set(message.id, cached)
        continue
      }

      const refsLocateCandidatesAll = buildRefsLocateCandidatesAll(refHits)
      const guideLocateCandidates = guideSourcePath
        ? (guideDocCandidatesBySourcePath.get(guideSourcePath) || [])
        : []
      const refsScopedCandidates = guideSourcePath
        ? refsLocateCandidatesAll.filter((item) => item.sourcePath === guideSourcePath)
        : refsLocateCandidatesAll
      const messageProvenance = (message.provenance && typeof message.provenance === 'object')
        ? message.provenance as Record<string, unknown>
        : null
      const provenanceSourcePath = String(messageProvenance?.source_path || '').trim()
      const provenanceSourceName = String(messageProvenance?.source_name || '').trim()
      const provenanceBlockMap = (messageProvenance?.block_map && typeof messageProvenance.block_map === 'object')
        ? messageProvenance.block_map as Record<string, Record<string, unknown>>
        : {}
      const provenanceDirectSegments = Array.isArray(messageProvenance?.segments)
        ? messageProvenance.segments.filter((segment) => {
          if (!segment || typeof segment !== 'object') return false
          const evidenceMode = String(segment.evidence_mode || '').trim().toLowerCase()
          const locatePolicy = String(segment.locate_policy || '').trim().toLowerCase()
          const evidenceIds = Array.isArray(segment.evidence_block_ids) ? segment.evidence_block_ids : []
          return evidenceMode === 'direct' && locatePolicy !== 'hidden' && evidenceIds.length > 0
        }) as Array<Record<string, unknown>>
        : []
      const hasDirectProvenance = Boolean(provenanceSourcePath) && provenanceDirectSegments.length > 0
      const hasStructuredProvenance = Boolean(
        provenanceSourcePath
        && Array.isArray(messageProvenance?.segments),
      )
      const effectiveGuideSourcePath = String(
        guideSourcePath
        || provenanceSourcePath
        || locateSourcePath
        || '',
      ).trim()
      const strictProvenanceLocate = Boolean(effectiveGuideSourcePath)
      const structuredLocateButtonCap = 12
      const provenanceLocateEntries = buildStructuredProvenanceLocateEntries(
        messageProvenance,
        {
          guideSourcePath: effectiveGuideSourcePath,
          fallbackSourceName: locateSourceName,
          maxEntries: structuredLocateButtonCap,
          minConfidence: 0.62,
        },
      )
      const structuredProvenanceSegmentsAll = messageProvenance
        ? listStructuredProvenanceSegments(messageProvenance)
        : []
      const provenanceStrictIdentityReady = Boolean(messageProvenance?.strict_identity_ready)
      const hasStrictMustLocateEntries = provenanceLocateEntries.some((entry) => Boolean(entry.mustLocate || entry.locatePolicy === 'required'))
      const strictStructuredLocateOnly = Boolean(
        strictProvenanceLocate
        && hasStructuredProvenance
        && provenanceStrictIdentityReady
        && hasStrictMustLocateEntries,
      )
      const strictStructuredInlineLocate = Boolean(strictStructuredLocateOnly)
      const provenanceMappingMode = String(messageProvenance?.mapping_mode || '').trim().toLowerCase()
      const provenanceLlmCallsRaw = Number(messageProvenance?.llm_rerank_calls || 0)
      const provenanceLlmCalls = Number.isFinite(provenanceLlmCallsRaw) && provenanceLlmCallsRaw > 0
        ? Math.floor(provenanceLlmCallsRaw)
        : 0
      const provenanceModeLabel = (() => {
        if (!strictStructuredLocateOnly) return ''
        if (provenanceMappingMode === 'llm_refined') {
          if (provenanceLlmCalls > 0) return `\u5b9a\u4f4d\u6620\u5c04\uff1aLLM\u7cbe\u4fee\uff08${provenanceLlmCalls} \u6b21\uff09`
          return '\u5b9a\u4f4d\u6620\u5c04\uff1aLLM\u7cbe\u4fee'
        }
        if (provenanceMappingMode === 'fast') return '\u5b9a\u4f4d\u6620\u5c04\uff1a\u5feb\u901f\u6620\u5c04'
        if (hasStructuredProvenance) return '\u5b9a\u4f4d\u6620\u5c04\uff1a\u7ed3\u6784\u5316\u6620\u5c04'
        return ''
      })()
      const structuredRenderSlotMap = buildStructuredRenderLocateSlotMap(
        String(bodyContent || ''),
        messageProvenance,
        provenanceLocateEntries,
      )
      const structuredLocateOrderBySegmentId = (() => {
        const map = new Map<string, number>()
        for (const slot of structuredRenderSlotMap.values()) {
          const segmentId = String(slot.entry.segmentId || '').trim()
          if (!segmentId || map.has(segmentId)) continue
          map.set(segmentId, Number(slot.order || 0))
        }
        return map
      })()
      const allowedStructuredRenderOrders = (() => {
        const ordered = Array.from(structuredRenderSlotMap.values()).sort((a, b) => a.order - b.order)
        const allowed = new Set<number>()
        let optionalCount = 0
        for (const slot of ordered) {
          if (slot.entry.mustLocate || slot.entry.locatePolicy === 'required') {
            allowed.add(slot.order)
            continue
          }
          if (optionalCount >= structuredLocateButtonCap) continue
          allowed.add(slot.order)
          optionalCount += 1
        }
        return allowed
      })()
      const locateCandidates = (() => {
        if (guideLocateCandidates.length > 0) return [...guideLocateCandidates, ...refsScopedCandidates]
        if (refsScopedCandidates.length > 0) return refsScopedCandidates
        if (refsLocateCandidatesAll.length > 0) return refsLocateCandidatesAll
        if (guideSourcePath) return guideDocCandidates
        return []
      })()

      const prep: AssistantLocatePrep = {
        bodyContent,
        refsUserMsgId,
        locateSourcePath,
        locateSourceName,
        refsLocateCandidatesAll,
        guideLocateCandidates,
        refsScopedCandidates,
        messageProvenance,
        provenanceSourcePath,
        provenanceSourceName,
        provenanceBlockMap,
        provenanceDirectSegments,
        hasDirectProvenance,
        hasStructuredProvenance,
        effectiveGuideSourcePath,
        strictProvenanceLocate,
        structuredLocateButtonCap,
        provenanceLocateEntries,
        structuredProvenanceSegmentsAll,
        provenanceStrictIdentityReady,
        hasStrictMustLocateEntries,
        strictStructuredLocateOnly,
        strictStructuredInlineLocate,
        provenanceModeLabel,
        structuredRenderSlotMap,
        structuredLocateOrderBySegmentId,
        allowedStructuredRenderOrders,
        locateCandidates,
      }
      heavyCount += 1
      nextCache.set(prepKey, prep)
      out.set(message.id, prep)
    }
    assistantLocatePrepCacheRef.current = nextCache
    assistantLocatePrepPerfRef.current = {
      ts: Date.now(),
      convId: String(activeConvId || ''),
      messageCount: messages.length,
      assistantCount,
      heavyCount,
      lightCount,
      cacheHits,
      durationMs: Number((messageListPerfNow() - prepStartedAt).toFixed(2)),
    }
    return out
  }, [
    activeConvId,
    assistantTraceByMsgId,
    guideDocCandidates,
    guideDocCandidatesBySourcePath,
    guideSourcePathSet,
    messages,
    onOpenReader,
    paperGuideSourceName,
    paperGuideSourcePath,
    refs,
  ])

  useEffect(() => {
    const perf = assistantLocatePrepPerfRef.current
    if (!perf) return
    pushMessageListPrepPerf(perf)
  }, [activeConvId, assistantLocatePrepByMsgId])

  const saveShelfSnapshot = () => {
    const currentItems = dedupeShelfItems(shelfItems).slice(0, SHELF_MAX_ITEMS)
    if (currentItems.length <= 0) {
      message.info('Shelf is empty; cannot save snapshot')
      return
    }
    const now = Date.now()
    const d = new Date(now)
    const pad = (value: number) => String(value).padStart(2, '0')
    const entry: ShelfSavedSnapshot = {
      id: `s_${now.toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      name: `Snapshot ${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`,
      createdAt: now,
      items: currentItems.map((item) => ({ ...item })),
    }
    setSavedShelfSnapshots((current) => {
      const next = [entry, ...current].slice(0, SHELF_SAVED_MAX_ITEMS)
      persistSavedShelfSnapshots(shelfSavedStorageKey(activeConvId), next)
      return next
    })
    setSelectedSavedSnapshotId(entry.id)
    message.success('Shelf snapshot saved')
  }

  const loadShelfSnapshot = () => {
    if (!selectedSavedSnapshot) return
    const restored = dedupeShelfItems(selectedSavedSnapshot.items).slice(0, SHELF_MAX_ITEMS).map((item) => ({ ...item }))
    setShelfItems(restored)
    setFocusedShelfKey('')
    setShelfSummaryLoadingKey('')
    setShelfRepairLoadingKey('')
    message.success(`Loaded snapshot: ${selectedSavedSnapshot.name}`)
  }

  const deleteShelfSnapshot = () => {
    if (!selectedSavedSnapshot) return
    const removedName = selectedSavedSnapshot.name
    setSavedShelfSnapshots((current) => {
      const next = current.filter((item) => item.id !== selectedSavedSnapshot.id)
      persistSavedShelfSnapshots(shelfSavedStorageKey(activeConvId), next)
      return next
    })
    setSelectedSavedSnapshotId((current) => (current === selectedSavedSnapshot.id ? '' : current))
    message.success(`Deleted snapshot: ${removedName}`)
  }

  return (
    <>
      <div ref={scrollRef} className="h-full min-h-0 overflow-y-auto px-4 py-6 kb-main-scroll">
        <div className="mx-auto flex max-w-7xl flex-col gap-4">
          {rows.map((row, index) => {
            if (row.kind === 'refs') {
              return (
                <div key={`refs-${row.userMsgId}-${index}`} className="flex gap-3">
                  <div className="mt-1 h-7 w-7 shrink-0" />
                  <div className="max-w-[88%] flex-1">
                    <RefsPanel
                      refs={refs}
                      msgId={row.userMsgId}
                      onOpenReader={onOpenReader}
                    />
                  </div>
                </div>
              )
            }

            const message = row.message
            const isUser = message.role === 'user'
            const trace = assistantTraceByMsgId.get(message.id)
            const citeDetails = Array.isArray(message.cite_details)
              ? message.cite_details
                .map(normalizeCiteDetail)
                .filter((detail): detail is CiteDetail => Boolean(detail))
                .map((detail) => ({
                  ...detail,
                  traceConvId: String(activeConvId || ''),
                  traceAssistantMsgId: message.id,
                  traceAssistantOrder: Number(trace?.answerOrder || 0),
                  traceUserMsgId: Number(trace?.userMsgId || 0),
                }))
              : []
            const imageAttachments = imageAttachmentsOf(message)
            const showUserText = !(isUser && imageAttachments.length > 0 && isImageOnlyPlaceholder(message.content))
            const isImageOnlyUserMessage = isUser && imageAttachments.length > 0 && !showUserText
            const prep = !isUser ? assistantLocatePrepByMsgId.get(message.id) : undefined
            const bodyContent = prep?.bodyContent || message.rendered_body || message.rendered_content || message.content
            const guideSourcePath = String(paperGuideSourcePath || '').trim()
            const locateSourceName = prep?.locateSourceName || String(paperGuideSourceName || '').trim()
            const messageProvenance = prep?.messageProvenance || (
              message.provenance && typeof message.provenance === 'object'
                ? message.provenance as Record<string, unknown>
                : null
            )
            const provenanceSourcePath = prep?.provenanceSourcePath || ''
            const provenanceSourceName = prep?.provenanceSourceName || ''
            const provenanceBlockMap = prep?.provenanceBlockMap || {} as Record<string, Record<string, unknown>>
            const provenanceDirectSegments = prep?.provenanceDirectSegments || []
            const hasDirectProvenance = prep?.hasDirectProvenance || false
            const hasStructuredProvenance = prep?.hasStructuredProvenance || false
            const effectiveGuideSourcePath = prep?.effectiveGuideSourcePath || guideSourcePath
            const strictProvenanceLocate = prep?.strictProvenanceLocate || false
            const provenanceLocateEntries = prep?.provenanceLocateEntries || []
            const structuredProvenanceSegmentsAll = prep?.structuredProvenanceSegmentsAll || []
            const provenanceStrictIdentityReady = prep?.provenanceStrictIdentityReady || false
            const hasStrictMustLocateEntries = prep?.hasStrictMustLocateEntries || false
            const strictStructuredLocateOnly = prep?.strictStructuredLocateOnly || false
            const strictStructuredInlineLocate = prep?.strictStructuredInlineLocate || false
            const guideInlineTextTailLocate = Boolean(
              guideSourcePath
              && provenanceLocateEntries.length > 0
              && !strictStructuredLocateOnly,
            )
            const provenanceModeLabel = prep?.provenanceModeLabel || ''
            const structuredRenderSlotMap = prep?.structuredRenderSlotMap || new Map<number, StructuredRenderLocateSlot>()
            const structuredLocateOrderBySegmentId = prep?.structuredLocateOrderBySegmentId || new Map<string, number>()
            const allowedStructuredRenderOrders = prep?.allowedStructuredRenderOrders || new Set<number>()
            const resolveStructuredFigureEntry = (snippet: string): ProvenanceLocateEntry | null => {
              const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || ''))).trim()
              const figureNumbers = extractFigureNumbersFromText(raw)
              const figureEntries = provenanceLocateEntries.filter((entry) => {
                const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
                const claimType = String(entry.claimType || '').trim().toLowerCase()
                return anchorKind === 'figure' || claimType === 'figure_claim'
              })
              if (!raw || figureEntries.length <= 0) return null
              let best: ProvenanceLocateEntry | null = null
              let bestScore = Number.NEGATIVE_INFINITY
              for (const entry of figureEntries) {
                let score = Math.max(
                  scoreProvenanceSegment(raw, entry.segmentText, entry.snippetKey),
                  overlapScore(raw, entry.anchorText || entry.segmentText),
                )
                if (figureNumbers.length > 0) {
                  score += 0.84 * figureNumberMatchScore(`${entry.anchorText} ${entry.segmentText}`, figureNumbers)
                }
                if (entry.mustLocate || entry.locatePolicy === 'required') {
                  score += 0.08
                }
                if (score > bestScore) {
                  best = entry
                  bestScore = score
                }
              }
              if (bestScore >= 0.26) return best
              if (!messageProvenance || !Array.isArray(messageProvenance?.segments)) return null
              const rawSegments = Array.isArray(messageProvenance.segments) ? messageProvenance.segments : []
              let rawBest: ProvenanceLocateEntry | null = null
              let rawBestScore = Number.NEGATIVE_INFINITY
              for (let idx = 0; idx < rawSegments.length; idx += 1) {
                const segment = rawSegments[idx] as unknown as Record<string, unknown> | null
                const currentSegment = structuredProvenanceSegmentsAll[idx] || null
                if (!segment || !currentSegment) continue
                const claimType = String(segment.claim_type || currentSegment.claimType || '').trim().toLowerCase()
                const locatePolicy = String(segment.locate_policy || currentSegment.locatePolicy || '').trim().toLowerCase()
                if (claimType !== 'figure_claim' || locatePolicy === 'hidden') continue
                if (!hasSegmentStrictLocateIdentity(segment, currentSegment)) continue
                const primaryBlockId = String(segment.primary_block_id || '').trim()
                const supportBlockIds = Array.isArray(segment.support_block_ids) ? segment.support_block_ids : []
                const evidenceBlockIds = Array.isArray(segment.evidence_block_ids) ? segment.evidence_block_ids : []
                const blockIds = [
                  ...[primaryBlockId].filter(Boolean),
                  ...supportBlockIds.map((item) => String(item || '').trim()).filter(Boolean),
                  ...evidenceBlockIds.map((item) => String(item || '').trim()).filter(Boolean),
                ]
                const candidates: LocateCandidate[] = []
                const seenBlock = new Set<string>()
                for (const blockIdRaw of blockIds) {
                  const blockId = String(blockIdRaw || '').trim()
                  if (!blockId || seenBlock.has(blockId)) continue
                  const block = provenanceBlockMap[blockId]
                  if (!block || typeof block !== 'object') continue
                  seenBlock.add(blockId)
                  const headingPath = String(block.heading_path || '').trim()
                  const blockText = stripMarkdownInline(String(block.text || '')).trim()
                  const anchorId = String(block.anchor_id || '').trim()
                  const anchorText = normalizeStrictAnchorText(String(segment.anchor_text || currentSegment.anchorText || ''))
                  const evidenceQuote = normalizeStrictAnchorText(String(segment.evidence_quote || anchorText || ''))
                  const focusSnippet = anchorText || evidenceQuote || blockText || currentSegment.text || headingPath
                  if (!focusSnippet) continue
                  candidates.push({
                    sourcePath: provenanceSourcePath || effectiveGuideSourcePath,
                    sourceName: provenanceSourceName || locateSourceName || (provenanceSourcePath.split(/[\\/]/).pop() || 'paper'),
                    headingPath,
                    focusSnippet,
                    matchText: [headingPath, anchorText || evidenceQuote || '', blockText || currentSegment.text].filter(Boolean).join('\n'),
                    sourceType: 'guide',
                    blockId,
                    anchorId: anchorId || undefined,
                    anchorKind: 'figure',
                  })
                }
                if (candidates.length <= 0) continue
                const entry: ProvenanceLocateEntry = {
                  segmentId: String(segment.segment_id || currentSegment.segmentId || `seg_${idx + 1}`).trim(),
                  label: shortSegmentLabel(String(segment.anchor_text || currentSegment.anchorText || currentSegment.text || '')),
                  segmentText: String(currentSegment.text || '').trim(),
                  evidenceQuote: normalizeStrictAnchorText(String(segment.evidence_quote || segment.anchor_text || '')),
                  claimType,
                  mustLocate: Boolean(segment.must_locate || locatePolicy === 'required'),
                  locatePolicy,
                  claimGroupId: String(segment.claim_group_id || currentSegment.claimGroupId || '').trim(),
                  claimGroupKind: String(segment.claim_group_kind || currentSegment.claimGroupKind || '').trim().toLowerCase(),
                  anchorKind: 'figure',
                  anchorText: normalizeStrictAnchorText(String(segment.anchor_text || currentSegment.anchorText || '')),
                  equationNumber: 0,
                  snippetKey: normalizeStructuredLocateSnippet(String(currentSegment.snippetKey || currentSegment.text || '').trim()),
                  snippetAliases: Array.isArray(currentSegment.snippetAliases) ? currentSegment.snippetAliases : [],
                  primary: candidates[0],
                  alternatives: candidates,
                  sourceSegmentId: String(segment.segment_id || '').trim() || undefined,
                }
                let score = Math.max(
                  scoreProvenanceSegment(raw, entry.segmentText, entry.snippetKey),
                  overlapScore(raw, entry.anchorText || entry.segmentText),
                )
                if (figureNumbers.length > 0) {
                  score += 0.92 * figureNumberMatchScore(`${entry.anchorText} ${entry.segmentText}`, figureNumbers)
                }
                if (score > rawBestScore) {
                  rawBest = entry
                  rawBestScore = score
                }
              }
              return rawBestScore >= 0.26 ? rawBest : null
            }
            const resolveStructuredQuoteEntry = (
              snippet: string,
              targetKindInput?: string,
            ): StructuredLocateResolution | null => {
              const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || ''))).trim()
              const targetKind = normalizeStructuredLocateKind(String(targetKindInput || ''))
              if (!raw) return null
              const quoteEntries = provenanceLocateEntries.filter((entry) => {
                const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
                const claimType = String(entry.claimType || '').trim().toLowerCase()
                if (targetKind === 'quote') {
                  return anchorKind === 'quote' || claimType === 'quote_claim'
                }
                if (targetKind === 'blockquote') {
                  return anchorKind === 'blockquote' || claimType === 'blockquote_claim' || claimType === 'quote_claim'
                }
                return anchorKind === 'quote' || anchorKind === 'blockquote' || claimType === 'quote_claim' || claimType === 'blockquote_claim'
              })
              if (quoteEntries.length <= 0) return null

              let best: ProvenanceLocateEntry | null = null
              let bestScore = Number.NEGATIVE_INFINITY
              for (const entry of quoteEntries) {
                const anchorKind = normalizeStructuredLocateKind(String(entry.anchorKind || ''))
                const compat = scoreStructuredAnchorCompatibility(targetKind || anchorKind, entry)
                if (targetKind && compat <= -0.9) continue
                let score = Math.max(
                  scoreProvenanceSegment(raw, entry.segmentText, entry.snippetKey),
                  overlapScore(raw, entry.anchorText || entry.segmentText),
                )
                if (Array.isArray(entry.snippetAliases) && entry.snippetAliases.length > 0) {
                  const aliasScore = entry.snippetAliases.reduce((acc, alias) => {
                    return Math.max(acc, overlapScore(raw, String(alias || '')))
                  }, 0)
                  score += 0.18 * aliasScore
                }
                if (entry.mustLocate || entry.locatePolicy === 'required') {
                  score += 0.08
                }
                if (targetKind && anchorKind === targetKind) {
                  score += 0.16
                }
                score += Math.max(-0.4, compat)
                if (score > bestScore) {
                  best = entry
                  bestScore = score
                }
              }
              const floor = targetKind === 'quote' ? 0.46 : 0.44
              if (!best || bestScore < floor) return null
              const order = Number(structuredLocateOrderBySegmentId.get(String(best.segmentId || '').trim()) || 0)
              return {
                entry: best,
                order: order > 0 ? order : 10000 + Math.max(0, provenanceLocateEntries.findIndex((item) => item.segmentId === best.segmentId)),
                fallback: !(order > 0),
              }
            }
            const isStrictStructuredTargetCompatible = (
              entry: ProvenanceLocateEntry | null | undefined,
              targetKindInput?: string,
            ): boolean => {
              const targetKind = normalizeStructuredLocateKind(String(targetKindInput || ''))
              if (!entry) return false
              if (!targetKind) return true
              const claimType = String(entry.claimType || '').trim().toLowerCase()
              const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
              if (targetKind === 'quote') {
                return anchorKind === 'quote' || claimType === 'quote_claim'
              }
              if (targetKind === 'blockquote') {
                return anchorKind === 'blockquote' || claimType === 'blockquote_claim' || claimType === 'quote_claim'
              }
              if (targetKind === 'figure') {
                return anchorKind === 'figure' || claimType === 'figure_claim'
              }
              if (targetKind === 'equation') {
                return anchorKind === 'equation' && claimType === 'formula_claim'
              }
              return true
            }
            const resolveStructuredInlineResolution = (
              snippet: string,
              meta?: LocateRenderMetaLite,
            ): StructuredLocateResolution | null => {
              if (!strictStructuredInlineLocate) return null
              const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
              const quoteEntry = (targetKind === 'quote' || targetKind === 'blockquote')
                ? resolveStructuredQuoteEntry(snippet, targetKind)
                : null
              if (quoteEntry) return quoteEntry
              const slot = resolveStructuredRenderLocateSlot(snippet, meta, structuredRenderSlotMap)
              if (slot && isStrictStructuredTargetCompatible(slot.entry, targetKind)) {
                return {
                  entry: slot.entry,
                  order: slot.order,
                  fallback: false,
                }
              }
              if (targetKind === 'equation') return null
              const fallbackEntry = resolveStructuredFallbackLocateEntry(snippet, meta, provenanceLocateEntries)
              const figureEntry = targetKind === 'figure' ? resolveStructuredFigureEntry(snippet) : null
              const finalEntry = figureEntry || fallbackEntry
              if (!finalEntry || !isStrictStructuredTargetCompatible(finalEntry, targetKind)) return null
              return {
                entry: finalEntry,
                order: 10000 + Math.max(0, provenanceLocateEntries.findIndex((item) => item.segmentId === finalEntry.segmentId)),
                fallback: true,
              }
            }
            const resolveExactStructuredInlineResolution = (
              snippet: string,
              meta?: LocateRenderMetaLite,
            ): StructuredLocateResolution | null => {
              const resolved = resolveStructuredInlineResolution(snippet, meta)
              if (!resolved || resolved.fallback) return null
              return resolved
            }
            const resolveProvenanceLocateCandidates = (snippet: string, limit = 4): LocateCandidate[] => {
              const raw = stripProvenanceNoise(stripMarkdownInline(String(snippet || '')))
              const key = normalizeLocateText(raw).slice(0, 360)
              if (!key || !provenanceSourcePath) return []
              const formulaQuery = hasFormulaSignal(raw)
              const quoteSpans = formulaQuery ? [] : extractQuotedSpans(raw, 12)
              const rankedSegments: Array<{ segment: NonNullable<typeof provenanceDirectSegments[number]>; score: number }> = []
              for (const segment of provenanceDirectSegments) {
                const segmentText = String(segment.text || '')
                if (isLikelyRhetoricalLocateShell(segmentText)) continue
                const segmentKey = String(segment.snippet_key || '')
                const segmentConf = Number(segment.evidence_confidence || 0)
                const confFloor = formulaQuery ? 0.5 : 0.62
                if (segmentConf > 0 && segmentConf < confFloor) continue
                let score = scoreProvenanceSegment(raw, segmentText, segmentKey)
                if (quoteSpans.length > 0) {
                  const qSeg = quoteMatchStats(quoteSpans, segmentText, segmentKey)
                  if (qSeg.hits <= 0 && qSeg.score < 0.55) continue
                  score += 0.38 * qSeg.score + (qSeg.hits > 0 ? 0.22 : 0)
                }
                if (score > 0) rankedSegments.push({ segment, score })
              }
              rankedSegments.sort((a, b) => b.score - a.score)
              const scoreFloor = formulaQuery ? 0.45 : 0.5
              let matchedSegments = rankedSegments
                .filter((row) => row.score >= scoreFloor)
                .slice(0, 1)
              if (formulaQuery && matchedSegments.length <= 0) {
                matchedSegments = rankedSegments
                  .filter((row) => row.score >= 0.42)
                  .slice(0, 1)
              }
              const out: LocateCandidate[] = []
              const seen = new Set<string>()
              for (const row of matchedSegments) {
                const segment = row.segment
                const evidenceIds = Array.isArray(segment.evidence_block_ids) ? segment.evidence_block_ids : []
                for (const blockIdRaw of evidenceIds.slice(0, formulaQuery ? 2 : 1)) {
                  const blockId = String(blockIdRaw || '').trim()
                  if (!blockId) continue
                  const block = provenanceBlockMap[blockId]
                  if (!block) continue
                  const blockKind = String(block.kind || '').trim().toLowerCase()
                  if (formulaQuery && blockKind && blockKind !== 'equation') continue
                  if (!formulaQuery && blockKind === 'equation' && evidenceIds.length > 1) continue
                  const blockText = String(block.text || '').trim()
                  if (quoteSpans.length > 0) {
                    const qBlock = quoteMatchStats(quoteSpans, blockText, String(segment.text || ''), String(block.heading_path || ''))
                    if (qBlock.hits <= 0 && qBlock.score < 0.85) continue
                  }
                  const key0 = `${provenanceSourcePath}::${blockId}`
                  if (seen.has(key0)) continue
                  seen.add(key0)
                  const segmentFocus = String(segment.text || '').trim()
                  const blockFocus = blockText
                  const focusSnippet = (formulaQuery ? (blockFocus || segmentFocus) : (segmentFocus || blockFocus))
                  if (!focusSnippet) continue
                  out.push({
                    sourcePath: provenanceSourcePath,
                    sourceName: provenanceSourceName || locateSourceName || provenanceSourcePath.split(/[\\/]/).pop() || 'paper',
                    headingPath: String(block.heading_path || '').trim(),
                    focusSnippet,
                    matchText: [String(block.heading_path || '').trim(), String(block.text || segment.text || '').trim()].filter(Boolean).join('\n'),
                    sourceType: 'guide',
                    blockId,
                    anchorId: String(block.anchor_id || '').trim() || undefined,
                    anchorKind: String(block.kind || '').trim().toLowerCase() || undefined,
                    anchorNumber: Number(block.number || 0) > 0 ? Math.floor(Number(block.number || 0)) : undefined,
                  })
                  if (out.length >= Math.max(1, limit)) return out
                }
              }
              return out
            }
            const locateCandidates = prep?.locateCandidates || (guideSourcePath ? guideDocCandidates : [])
            const enableLocateUi = Boolean(onOpenReader) && (
              strictStructuredLocateOnly
              || strictStructuredInlineLocate
              || hasDirectProvenance
              || provenanceLocateEntries.length > 0
              || locateCandidates.length > 0
            )
            const resolveCache = new Map<string, LocateCandidate[]>()
            const usedCount = new Map<string, number>()
            const resolveLocateCandidates = (snippet: string, limit = 4) => {
              const key = String(snippet || '').trim()
              if (!key) return []
              if (resolveCache.has(key)) return (resolveCache.get(key) || []).slice(0, Math.max(1, limit))
              const formulaQuery = hasFormulaSignal(key)
              const guideOnly = locateCandidates.filter((item) => item.sourceType === 'guide')
              const strictDirectMode = hasDirectProvenance && !formulaQuery && guideOnly.length > 0
              const provenancePicked = resolveProvenanceLocateCandidates(key, limit)
              if (provenancePicked.length > 0) {
                const picked = formulaQuery
                  ? (() => {
                    const eqProv = provenancePicked.filter((cand) => isEquationLocateCandidate(cand))
                    return eqProv.length > 0 ? eqProv : provenancePicked
                  })()
                  : provenancePicked
                resolveCache.set(key, picked)
                return picked.slice(0, Math.max(1, limit))
              }
              if (strictProvenanceLocate && hasStructuredProvenance && provenanceStrictIdentityReady && hasStrictMustLocateEntries) {
                // Paper-guide mode should not fall back to fuzzy locate if this
                // message already has strict-ready structured provenance but no
                // direct evidence for the current snippet.
                resolveCache.set(key, [])
                return []
              }
              const quoteSpans = formulaQuery ? [] : extractQuotedSpans(key, 12)
              if (!formulaQuery && quoteSpans.length > 0) {
                const quotePool = guideOnly.length > 0 ? guideOnly : locateCandidates
                const quoteRank = quotePool
                  .map((cand) => {
                    const q = quoteMatchStats(quoteSpans, cand.matchText, cand.focusSnippet, cand.headingPath)
                    let score = q.score + (0.35 * scoreLocateCandidate(key, cand))
                    if (q.hits > 0) score += 0.35
                    if (cand.sourceType === 'guide') score += 0.08
                    if (cand.anchorId || cand.blockId) score += 0.1
                    return { cand, score, hits: q.hits }
                  })
                  .sort((a, b) => b.score - a.score)
                const bestQuote = quoteRank[0]
                if (bestQuote && bestQuote.hits > 0 && bestQuote.score >= 1.05) {
                  resolveCache.set(key, [bestQuote.cand])
                  return [bestQuote.cand]
                }
              }

              const rankIn = (cands: LocateCandidate[]) => {
                const scored: Array<{ cand: LocateCandidate; score: number }> = []
                for (const cand of cands) {
                  const base = scoreLocateCandidate(key, cand)
                  const candKey = `${cand.sourcePath}::${cand.anchorId || ''}::${cand.headingPath}::${cand.focusSnippet.slice(0, 96)}`
                  const penalty = 0.03 * Number(usedCount.get(candKey) || 0)
                  const score = base - penalty
                  scored.push({ cand, score })
                }
                scored.sort((a, b) => b.score - a.score)
                return scored
              }

              const picked: LocateCandidate[] = []
              const pickedKeySet = new Set<string>()
              const pickedHeadingSet = new Set<string>()
              const addPicked = (cand: LocateCandidate, preferNewHeading = false) => {
                const candKey = `${cand.sourcePath}::${cand.anchorId || ''}::${cand.headingPath}::${cand.focusSnippet.slice(0, 96)}`
                if (pickedKeySet.has(candKey)) return false
                const headingRaw = String(cand.headingPath || '').trim()
                const headingKey = headingRaw
                  ? normalizeLocateText(headingRaw)
                  : normalizeLocateText(String(cand.focusSnippet || '').slice(0, 56))
                if (preferNewHeading && headingKey && pickedHeadingSet.has(headingKey)) return false
                picked.push(cand)
                pickedKeySet.add(candKey)
                if (headingKey) pickedHeadingSet.add(headingKey)
                return true
              }
              const ingestRank = (
                rankRows: Array<{ cand: LocateCandidate; score: number }>,
                floor: number,
                preferNewHeading: boolean,
              ) => {
                for (const row of rankRows) {
                  if (row.score < floor) break
                  addPicked(row.cand, preferNewHeading)
                  if (picked.length >= limit) break
                }
              }

              if (hasDirectProvenance && formulaQuery) {
                const eqNums = extractEquationNumbersFromText(key)
                const eqGuide = guideOnly.filter((cand) => isEquationLocateCandidate(cand))
                if (eqGuide.length > 0) {
                  let bestEq: LocateCandidate | null = null
                  let bestEqScore = -1
                  for (const cand of eqGuide) {
                    let s = scoreLocateCandidate(key, cand)
                    if (eqNums.length > 0 && Number(cand.anchorNumber || 0) > 0 && eqNums.includes(Math.floor(Number(cand.anchorNumber || 0)))) {
                      s += 0.45
                    }
                    if (cand.anchorId) s += 0.2
                    if (s > bestEqScore) {
                      bestEq = cand
                      bestEqScore = s
                    }
                  }
                  if (bestEq && bestEqScore >= 0.58) {
                    resolveCache.set(key, [bestEq])
                    return [bestEq]
                  }
                }
              }
              if (guideOnly.length > 0) {
                const guideRank = rankIn(guideOnly)
                const guideFloor = strictDirectMode
                  ? 0.34
                  : (hasFormulaSignal(key) ? 0.32 : 0.2)
                ingestRank(guideRank, guideFloor, true)
                if (picked.length < limit) ingestRank(guideRank, guideFloor, false)
              }
              if (picked.length < limit) {
                const strictPool = strictDirectMode
                  ? locateCandidates.filter((item) => String(item.sourcePath || '').trim() === provenanceSourcePath)
                  : []
                const rankBase = (strictDirectMode && strictPool.length > 0) ? strictPool : locateCandidates
                const rankAll = rankIn(rankBase)
                const allFloor = strictDirectMode
                  ? (hasFormulaSignal(key) ? 0.34 : 0.24)
                  : (hasFormulaSignal(key) ? 0.3 : 0.2)
                ingestRank(rankAll, allFloor, true)
                if (picked.length < limit) ingestRank(rankAll, allFloor, false)
                if (picked.length <= 0 && rankAll.length > 0) {
                  const best = rankAll[0]
                  const preferAnchor = Boolean(best?.cand?.anchorId)
                  const bestFloor = preferAnchor
                    ? (hasFormulaSignal(key) ? 0.3 : 0.24)
                    : (hasFormulaSignal(key) ? 0.38 : 0.3)
                  if ((best?.score || 0) >= bestFloor) {
                    addPicked(best.cand, false)
                  }
                }
              }
              if (picked.length <= 0 && hasFormulaSignal(key) && guideOnly.length > 0) {
                const eqNums = extractEquationNumbersFromText(key)
                const eqCandidates = guideOnly.filter((cand) => isEquationLocateCandidate(cand))
                if (eqCandidates.length > 0) {
                  const preferByNum = eqNums.length > 0
                    ? eqCandidates.filter((cand) => {
                      const n = Number(cand.anchorNumber || 0)
                      return Number.isFinite(n) && n > 0 && eqNums.includes(Math.floor(n))
                    })
                    : []
                  const pool = preferByNum.length > 0 ? preferByNum : eqCandidates
                  let bestEq: LocateCandidate | null = null
                  let bestEqScore = -1
                  for (const cand of pool) {
                    let s = scoreLocateCandidate(key, cand)
                    if (eqNums.length > 0 && Number(cand.anchorNumber || 0) > 0) s += 0.4
                    if (cand.anchorId) s += 0.2
                    if (s > bestEqScore) {
                      bestEq = cand
                      bestEqScore = s
                    }
                  }
                  if (bestEq && bestEqScore >= 0.34) addPicked(bestEq, false)
                }
              }
              const unique: LocateCandidate[] = []
              const seen = new Set<string>()
              for (const cand of picked) {
                const candKey = `${cand.sourcePath}::${cand.anchorId || ''}::${cand.headingPath}::${cand.focusSnippet.slice(0, 96)}`
                if (seen.has(candKey)) continue
                seen.add(candKey)
                unique.push(cand)
                if (unique.length >= limit) break
              }
              if (unique.length <= 0 && guideOnly.length > 0) {
                const relaxed = rankIn(guideOnly)
                const best = relaxed[0]
                if (best && (best.score || 0) >= 0.08) {
                  unique.push(best.cand)
                }
              }
              const first = unique[0]
              if (first) {
                const pickKey = `${first.sourcePath}::${first.anchorId || ''}::${first.headingPath}::${first.focusSnippet.slice(0, 96)}`
                usedCount.set(pickKey, Number(usedCount.get(pickKey) || 0) + 1)
              }
              resolveCache.set(key, unique)
              return unique.slice(0, Math.max(1, limit))
            }
            const locateButtonShownKeys = new Set<string>()
            const locateButtonCap = 5
            let optionalLocateButtonCount = 0
            const locateCandidateKey = (cand: LocateCandidate | null) => {
              if (!cand) return ''
              if (cand.blockId) return `${cand.sourcePath}::block::${cand.blockId}`
              if (cand.anchorId) return `${cand.sourcePath}::anchor::${cand.anchorId}`
              const headingKey = normalizeLocateText(String(cand.headingPath || ''))
              const focusKey = normalizeLocateText(String(cand.focusSnippet || '')).slice(0, 64)
              return `${cand.sourcePath}::${headingKey}::${focusKey}`
            }
            const openReaderByCandidates = (
              pickedList: LocateCandidate[],
              snippet: string,
              opts?: { strictLocate?: boolean; highlightSnippet?: string; relatedBlockIds?: string[] },
            ) => {
              if (!onOpenReader) return
              const picked = pickedList[0] || null
              if (!picked) return
              const highlightSnippet = String(opts?.highlightSnippet || snippet).trim()
              onOpenReader({
                sourcePath: picked.sourcePath,
                sourceName: picked.sourceName,
                headingPath: picked.headingPath,
                snippet: picked.focusSnippet || snippet,
                highlightSnippet: highlightSnippet || picked.focusSnippet || snippet,
                blockId: picked.blockId,
                anchorId: picked.anchorId,
                anchorKind: picked.anchorKind,
                anchorNumber: picked.anchorNumber,
                strictLocate: Boolean(opts?.strictLocate),
                locateMode: 'heuristic',
                relatedBlockIds: Array.isArray(opts?.relatedBlockIds)
                  ? opts.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
                  : undefined,
                alternatives: pickedList.map((item) => ({
                  headingPath: item.headingPath,
                  snippet: item.focusSnippet || snippet,
                  highlightSnippet: highlightSnippet || item.focusSnippet || snippet,
                  blockId: item.blockId,
                  anchorId: item.anchorId,
                  anchorKind: item.anchorKind,
                  anchorNumber: item.anchorNumber,
                })),
                evidenceAlternatives: pickedList.map((item) => ({
                  headingPath: item.headingPath,
                  snippet: item.focusSnippet || snippet,
                  highlightSnippet: highlightSnippet || item.focusSnippet || snippet,
                  blockId: item.blockId,
                  anchorId: item.anchorId,
                  anchorKind: item.anchorKind,
                  anchorNumber: item.anchorNumber,
                })),
                visibleAlternatives: pickedList.map((item) => ({
                  headingPath: item.headingPath,
                  snippet: item.focusSnippet || snippet,
                  highlightSnippet: highlightSnippet || item.focusSnippet || snippet,
                  blockId: item.blockId,
                  anchorId: item.anchorId,
                  anchorKind: item.anchorKind,
                  anchorNumber: item.anchorNumber,
                })),
                initialAltIndex: 0,
              })
            }
            const openReaderByStructuredEntry = (entry: ProvenanceLocateEntry, snippet: string) => {
              if (!onOpenReader) return
              const queryRaw = stripProvenanceNoise(
                stripMarkdownInline(String(entry.evidenceQuote || snippet || entry.segmentText || entry.label || '')),
              ).trim()
              const primary = entry.primary
              if (!primary) return
              const structuredSnippet = String(queryRaw || primary.focusSnippet || snippet).trim()
              const structuredHighlight = String(
                entry.evidenceQuote
                || queryRaw
                || primary.focusSnippet
                || snippet,
              ).trim()
              const structuredAnchorKind = String(entry.anchorKind || primary.anchorKind || '').trim()
              const structuredAnchorNumber = Number.isFinite(Number(entry.equationNumber || primary.anchorNumber || 0))
                ? Math.floor(Number(entry.equationNumber || primary.anchorNumber || 0))
                : undefined
              const groupDistance = Number.isFinite(Number(entry.groupDistance || 0))
                ? Math.max(0, Math.floor(Number(entry.groupDistance || 0)))
                : 0
              const locateTarget: ReaderLocateTarget = {
                segmentId: String(entry.segmentId || '').trim() || undefined,
                sourceSegmentId: String(entry.sourceSegmentId || '').trim() || undefined,
                headingPath: String(primary.headingPath || '').trim() || undefined,
                snippet: structuredSnippet || undefined,
                highlightSnippet: structuredHighlight || undefined,
                evidenceQuote: String(entry.evidenceQuote || '').trim() || undefined,
                anchorText: String(entry.anchorText || '').trim() || undefined,
                blockId: String(primary.blockId || '').trim() || undefined,
                anchorId: String(primary.anchorId || '').trim() || undefined,
                anchorKind: structuredAnchorKind || undefined,
                anchorNumber: structuredAnchorNumber,
                claimType: String(entry.claimType || '').trim() || undefined,
                locatePolicy: String(entry.locatePolicy || '').trim() || undefined,
                locateSurfacePolicy: String(entry.locateSurfacePolicy || '').trim() || undefined,
                snippetAliases: Array.isArray(entry.snippetAliases)
                  ? entry.snippetAliases.map((item) => String(item || '').trim()).filter(Boolean)
                  : undefined,
                relatedBlockIds: Array.isArray(entry.relatedBlockIds)
                  ? entry.relatedBlockIds.map((item) => String(item || '').trim()).filter(Boolean)
                  : undefined,
              }
              const claimGroup: ReaderLocateClaimGroup | undefined = (
                entry.claimGroupId
                || entry.claimGroupKind
                || entry.groupLeadText
                || groupDistance > 0
              )
                ? {
                  id: String(entry.claimGroupId || '').trim() || undefined,
                  kind: String(entry.claimGroupKind || '').trim() || undefined,
                  leadText: String(entry.groupLeadText || '').trim() || undefined,
                  distance: groupDistance > 0 ? groupDistance : undefined,
                }
                : undefined
              const fallbackAlternatives: ReaderLocateCandidate[] = (entry.alternatives || [])
                .filter((item) => Boolean(item) && item !== primary)
                .map((item) => ({
                  headingPath: String(item.headingPath || '').trim() || undefined,
                  snippet: String(item.focusSnippet || structuredSnippet).trim() || undefined,
                  highlightSnippet: structuredHighlight || String(item.focusSnippet || structuredSnippet).trim() || undefined,
                  blockId: String(item.blockId || '').trim() || undefined,
                  anchorId: String(item.anchorId || '').trim() || undefined,
                  anchorKind: String(item.anchorKind || structuredAnchorKind).trim() || undefined,
                  anchorNumber: Number.isFinite(Number(item.anchorNumber || structuredAnchorNumber || 0))
                    ? Math.floor(Number(item.anchorNumber || structuredAnchorNumber || 0))
                    : undefined,
                }))
              onOpenReader({
                sourcePath: primary.sourcePath,
                sourceName: primary.sourceName,
                headingPath: String(primary.headingPath || '').trim() || undefined,
                snippet: structuredSnippet || undefined,
                highlightSnippet: structuredHighlight || undefined,
                blockId: String(primary.blockId || '').trim() || undefined,
                anchorId: String(primary.anchorId || '').trim() || undefined,
                relatedBlockIds: locateTarget.relatedBlockIds,
                anchorKind: structuredAnchorKind || undefined,
                anchorNumber: structuredAnchorNumber,
                strictLocate: true,
                locateTarget,
                claimGroup,
                alternatives: fallbackAlternatives.length > 0 ? fallbackAlternatives : undefined,
                visibleAlternatives: undefined,
                evidenceAlternatives: undefined,
                initialAltIndex: 0,
              })
            }
            const bubbleClass = isUser
              ? (isImageOnlyUserMessage ? 'w-fit max-w-[22rem]' : 'w-fit max-w-[88%]')
              : 'w-full max-w-[88%]'
            const bubblePadClass = isImageOnlyUserMessage ? 'px-4 py-4' : 'px-6 py-4'

            return (
              <div
                key={message.id}
                data-msg-id={message.id}
                className={`flex gap-3 ${isUser ? 'justify-end' : ''}`}
              >
                {!isUser ? <AssistantAvatar /> : null}
                <div
                  className={`${bubbleClass} ${bubblePadClass} rounded-[28px] ${
                    isUser ? 'bg-[var(--msg-user-bg)]' : 'border border-[var(--border)] bg-[var(--msg-ai-bg)] shadow-[0_8px_24px_rgba(15,23,42,0.03)]'
                  }`}
                >
                  {isUser ? (
                    <>
                      {imageAttachments.length > 0 ? (
                        <div
                          className={`${
                            showUserText ? 'mb-3' : ''
                          } grid ${
                            imageAttachments.length === 1 ? 'max-w-[18rem] grid-cols-1' : 'max-w-[38rem] grid-cols-2 sm:grid-cols-3'
                          } gap-2`}
                        >
                          {imageAttachments.map((item) => {
                            const src = String(item.url || '').trim()
                            const key = `${item.sha1 || item.path}-${item.name}`
                            const frameClass = 'block overflow-hidden rounded-2xl border border-[var(--border)] bg-white/70'
                            if (src) {
                              return (
                                <a
                                  key={key}
                                  href={src}
                                  target="_blank"
                                  rel="noreferrer"
                                  className={frameClass}
                                >
                                  <img
                                    src={src}
                                    alt={item.name}
                                    className="block h-32 w-full object-cover"
                                    loading="lazy"
                                  />
                                </a>
                              )
                            }
                            return (
                              <div key={key} className={frameClass}>
                                <div className="flex h-32 items-center justify-center px-3 text-center text-xs text-black/45">
                                  {item.name}
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      ) : null}
                      {showUserText ? (
                        <Text className="whitespace-pre-wrap">{message.content}</Text>
                      ) : null}
                    </>
                  ) : (
                    <>
                      {message.notice ? (
                        <div className="mb-4 rounded-2xl border border-[var(--border)] bg-black/[0.03] px-4 py-3 text-sm text-black/70 dark:bg-white/[0.04] dark:text-white/70">
                          {message.notice}
                        </div>
                      ) : null}
                      {provenanceModeLabel ? (
                        <div className="mb-2">
                          <Text type="secondary" className="text-xs">{provenanceModeLabel}</Text>
                        </div>
                      ) : null}
                      <MarkdownRenderer
                        content={bodyContent}
                        citeDetails={citeDetails}
                        onCitationClick={openCitation}
                        inlineLocateTokenPolicy={enableLocateUi && guideSourcePath ? { quote: true, figure_ref: true } : undefined}
                        inlineTextLocateEnabled={enableLocateUi ? (!guideSourcePath || strictStructuredInlineLocate) : false}
                        inlineTextTailLocateEnabled={enableLocateUi ? guideInlineTextTailLocate : false}
                        locateSurfacePolicy={enableLocateUi && guideSourcePath
                          ? {
                            paragraph: guideInlineTextTailLocate,
                            list_item: guideInlineTextTailLocate,
                            quote: true,
                            blockquote: true,
                            equation: true,
                            figure: true,
                          }
                          : undefined}
                        canLocateSnippet={enableLocateUi ? ((snippet, meta) => {
                          if (strictStructuredLocateOnly) {
                            if (!strictStructuredInlineLocate) return false
                            const resolved = resolveExactStructuredInlineResolution(snippet, meta)
                            const entry = resolved?.entry || null
                            if (!entry) return false
                            const order = Number(resolved?.order || 0)
                            if (!allowedStructuredRenderOrders.has(order)) return false
                            const targetKind = normalizeStructuredLocateKind(String(meta?.kind || ''))
                            if (!['quote', 'blockquote', 'equation', 'figure'].includes(targetKind)) {
                              return false
                            }
                            if (!isStrictStructuredTargetCompatible(entry, targetKind)) {
                              return false
                            }
                            const claimType = String(entry.claimType || '').trim().toLowerCase()
                            const anchorKind = String(entry.anchorKind || '').trim().toLowerCase()
                            const formulaOrigin = String(entry.formulaOrigin || '').trim().toLowerCase()
                            const locateSurfacePolicy = String(entry.locateSurfacePolicy || '').trim().toLowerCase()
                            if ((anchorKind === 'quote' || claimType === 'quote_claim') && targetKind !== 'quote') {
                              return false
                            }
                            if ((anchorKind === 'blockquote' || claimType === 'blockquote_claim') && targetKind !== 'blockquote') {
                              return false
                            }
                            if ((anchorKind === 'figure' || claimType === 'figure_claim') && targetKind !== 'figure') {
                              return false
                            }
                            if (targetKind === 'equation') {
                              if (claimType !== 'formula_claim' || anchorKind !== 'equation') {
                                return false
                              }
                              if (formulaOrigin !== 'source' || locateSurfacePolicy !== 'primary') {
                                return false
                              }
                            }
                            if (targetKind === 'figure') {
                              return isPreferredStrictFigureRefSnippet(snippet)
                            }
                            return true
                          }
                          const raw = String(snippet || '').trim()
                          const formulaSnippet = hasFormulaSignal(raw)
                          if (!formulaSnippet && raw.length < 18) return false
                          const directPickedList = resolveProvenanceLocateCandidates(snippet, 1)
                          const directPicked = formulaSnippet
                            ? (directPickedList.find((item) => isEquationLocateCandidate(item)) || directPickedList[0] || null)
                            : (directPickedList[0] || null)
                          const pickedList = directPicked
                            ? directPickedList
                            : resolveLocateCandidates(snippet, 1)
                          const picked = formulaSnippet
                            ? (pickedList.find((item) => isEquationLocateCandidate(item)) || pickedList[0] || null)
                            : (pickedList[0] || null)
                          if (!picked) return false
                          const keyBase = locateCandidateKey(picked)
                          const snippetKey = normalizeLocateText(raw).slice(0, 96)
                          const key = keyBase
                            ? `${keyBase}::${snippetKey}`
                            : snippetKey
                          if (!key) return false
                          if (locateButtonShownKeys.has(key)) return false
                          if (!directPicked && optionalLocateButtonCount >= locateButtonCap) return false
                          locateButtonShownKeys.add(key)
                          if (!directPicked) optionalLocateButtonCount += 1
                          return true
                        }) : undefined}
                        onLocateSnippet={enableLocateUi && onOpenReader
                          ? (snippet, meta) => {
                            if (strictStructuredLocateOnly) {
                              if (!strictStructuredInlineLocate) return
                              const resolved = resolveExactStructuredInlineResolution(snippet, meta)
                              const entry = resolved?.entry || null
                              if (!entry) return
                              if (!allowedStructuredRenderOrders.has(Number(resolved?.order || 0))) return
                              openReaderByStructuredEntry(entry, snippet)
                              return
                            }
                            const raw = String(snippet || '').trim()
                            const formulaSnippet = hasFormulaSignal(raw)
                            const pickedListRaw = resolveLocateCandidates(snippet, 6)
                            const pickedList = formulaSnippet
                              ? [
                                ...pickedListRaw.filter((item) => isEquationLocateCandidate(item)),
                                ...pickedListRaw.filter((item) => !isEquationLocateCandidate(item)),
                              ]
                              : pickedListRaw
                            if (pickedList.length <= 0) return
                            openReaderByCandidates(pickedList, snippet)
                          }
                          : undefined}
                        locateTitleResolver={enableLocateUi ? ((snippet) => {
                          if (strictStructuredLocateOnly) {
                            const resolved = resolveExactStructuredInlineResolution(snippet)
                            const entry = resolved?.entry || null
                            if (entry) {
                              const heading = String(entry.primary.headingPath || '').trim()
                              return heading ? `\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e\uff1a${heading}` : '\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e'
                            }
                          }
                          const formulaSnippet = hasFormulaSignal(String(snippet || ''))
                          const pickedList = resolveLocateCandidates(snippet, formulaSnippet ? 2 : 1)
                          const picked = formulaSnippet
                            ? (pickedList.find((item) => isEquationLocateCandidate(item)) || pickedList[0] || null)
                            : (pickedList[0] || null)
                          if (!picked) return '\u5b9a\u4f4d\u5230\u539f\u6587'
                          const heading = String(picked.headingPath || '').trim()
                          return heading ? `\u5b9a\u4f4d\u5230\u539f\u6587\uff1a${heading}` : '\u5b9a\u4f4d\u5230\u539f\u6587'
                        }) : undefined}
                      />
                      {Boolean(onOpenReader) && provenanceLocateEntries.length > 0 ? (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {provenanceLocateEntries.map((entry, idx) => {
                            const heading = String(entry.primary?.headingPath || '').trim()
                            const label = String(entry.label || '').trim()
                            const snippet = shortSegmentLabel(
                              String(entry.anchorText || entry.evidenceQuote || entry.segmentText || label || ''),
                              72,
                            )
                            const headingLite = compactHeadingPath(heading, 56)
                            const text = snippet
                              || label
                              || headingLite
                              || '\u539f\u6587\u8bc1\u636e'
                            const seedSnippet = String(
                              entry.evidenceQuote
                              || entry.anchorText
                              || entry.segmentText
                              || entry.label
                              || '',
                            ).trim()
                            const focusSnippet = String(entry.primary?.focusSnippet || entry.primary?.matchText || seedSnippet || '').trim()
                            return (
                              <button
                                key={`${message.id}::prov::${String(entry.segmentId || idx)}::${idx}`}
                                type="button"
                                className="kb-prov-locate-chip"
                                aria-label={'\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e'}
                                title={heading
                                  ? `\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e\uff1a${heading}`
                                  : (headingLite ? `\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e\uff1a${headingLite}` : '\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e')}
                                data-kb-locate-focus={focusSnippet.slice(0, 220)}
                                data-kb-locate-block-id={String(entry.primary?.blockId || '').trim()}
                                data-kb-locate-anchor-id={String(entry.primary?.anchorId || '').trim()}
                                data-kb-locate-heading={String(entry.primary?.headingPath || '').trim()}
                                onClick={() => openReaderByStructuredEntry(entry, seedSnippet)}
                              >
                                <span className="kb-prov-locate-chip-num">{`\u8bc1\u636e${idx + 1}`}</span>
                                <span className="kb-prov-locate-chip-text">{text}</span>
                              </button>
                            )
                          })}
                        </div>
                      ) : null}
                      <CopyBar
                        text={message.copy_text || message.content}
                        markdown={message.copy_markdown}
                      />
                    </>
                  )}
                </div>
                {isUser ? (
                  <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-[var(--border)] bg-[var(--msg-user-bg)]">
                    <UserOutlined className="text-xs" />
                  </div>
                ) : null}
              </div>
            )
          })}

          {generationPartial !== undefined && generationPartial !== null ? (
            <div className="flex gap-3">
              <AssistantAvatar />
              <div className="w-full max-w-[88%] rounded-[28px] border border-[var(--border)] bg-[var(--msg-ai-bg)] px-6 py-4 shadow-[0_8px_24px_rgba(15,23,42,0.03)]">
                {generationStage ? (
                  <div className="mb-2 flex items-center gap-2">
                    <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--accent)]" />
                    <Text type="secondary" className="text-xs">
                      {generationStage}
                    </Text>
                  </div>
                ) : null}
                {generationPartial ? (
                  <MarkdownRenderer content={generationPartial} />
                ) : (
                  <div className="flex items-center gap-1 py-1">
                    <span className="typing-dot" />
                    <span className="typing-dot" style={{ animationDelay: '0.15s' }} />
                    <span className="typing-dot" style={{ animationDelay: '0.3s' }} />
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <CitationPopover
        detail={popoverDetail}
        position={popoverPos}
        loading={popoverLoading}
        guideLoading={popoverGuideLoading}
        inShelf={Boolean(popoverDetail && shelfItems.some((item) => item.key === toShelfItem(popoverDetail).key))}
        onClose={() => {
          setPopoverDetail(null)
          setPopoverPos(null)
          setPopoverLoading(false)
          setPopoverGuideLoading(false)
        }}
        onAddToShelf={addToShelf}
        onOpenShelf={() => setShelfOpen(true)}
        onOpenReader={openReaderFromDetail}
        onStartGuide={startPaperGuideFromDetail}
      />
      <CiteShelf
        open={shelfOpen}
        items={shelfItems}
        focusedKey={focusedShelfKey}
        summaryLoadingKey={shelfSummaryLoadingKey}
        repairLoadingKey={shelfRepairLoadingKey}
        snapshots={savedShelfSnapshots}
        selectedSnapshotId={selectedSavedSnapshotId}
        snapshotDiff={selectedSnapshotDiff}
        onToggle={() => setShelfOpen((value) => !value)}
        onSelect={(item) => {
          setFocusedShelfKey(item.key)
          fetchShelfSummaryForItem(item)
        }}
        onRemove={(key) => {
          setShelfItems((current) => current.filter((item) => item.key !== key))
          if (focusedShelfKey === key) setFocusedShelfKey('')
          if (shelfSummaryLoadingKey === key) setShelfSummaryLoadingKey('')
          if (shelfRepairLoadingKey === key) setShelfRepairLoadingKey('')
        }}
        onClear={() => {
          setShelfItems([])
          setFocusedShelfKey('')
          setShelfSummaryLoadingKey('')
          setShelfRepairLoadingKey('')
        }}
        onUpdateTags={(key, tags) => {
          const nextTags = normalizeShelfTags(tags)
          setShelfItems((current) => current.map((item) => (
            item.key === key ? { ...item, tags: nextTags } : item
          )))
        }}
        onUpdateNote={(key, note) => {
          const nextNote = normalizeShelfNote(note)
          setShelfItems((current) => current.map((item) => (
            item.key === key ? { ...item, note: nextNote } : item
          )))
        }}
        onRepair={(item) => {
          repairShelfItemMeta(item)
        }}
        onSelectSnapshot={setSelectedSavedSnapshotId}
        onSaveSnapshot={saveShelfSnapshot}
        onLoadSnapshot={loadShelfSnapshot}
        onDeleteSnapshot={deleteShelfSnapshot}
      />
    </>
  )
}

