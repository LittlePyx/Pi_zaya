import { expect, test, type Page } from '@playwright/test'
import {
  READER_REGRESSION_SOURCE_PATH,
} from '../../src/testing/readerRegressionFixtures'
import type { AgentTraceAuditResponse } from '../../src/api/chat'
import type { ArchivedAgentTraceSnapshotInput } from '../../src/components/chat/agentTraceArchiveState'
import type { AgentTraceHeaderSummaryInput } from '../../src/components/chat/agentTraceHeaderSummary'
import type { AgentTraceMetricCountInput } from '../../src/components/chat/agentTraceMetricCounts'
import type { AgentTracePanelStateInput } from '../../src/components/chat/agentTracePanelState'
import type { AgentTraceQualityGateTitleInput } from '../../src/components/chat/agentTraceQualityGate'
import type { AgentTraceScopeSummaryInput } from '../../src/components/chat/agentTraceScopeSummary'
import type { AgentTraceSourceRowsInput } from '../../src/components/chat/agentTraceSourceRows'
import type { AgentTraceSourceStatusInput } from '../../src/components/chat/agentTraceSourceStatus'
import type {
  AgentSourceSummaryViewModel,
  AgentTraceViewModel,
} from '../../src/components/chat/agentTraceViewModel'
import {
  installAppShellMocks,
  installEmptyCitationShelfMock,
  installIdleReferenceMocks,
} from './mockAppShell'
const LOCAL_KB_RE = /Local KB|本地文献库|鏈湴鏂囩尞搴?/

const SOURCE_PANEL_RE = /Sources & evidence|依据与来源/
const NEEDS_REVIEW_RE = /Needs review|需要核对/
const REVIEW_FRACTION_RE = /Review 1\/2|需核对 1\/2/
const CHECKED_FRACTION_RE = /Checked 1\/1|已核对 1\/1/
const GROUNDED_RE = /Evidence grounded|文献证据充分/
const ANSWER_QUALITY_RE = /Answer quality|回答质量/
const REPAIRED_RE = /Repaired|已修正/
const EVIDENCE_MAP_RE = /Evidence map|证据地图/
const MISMATCH_RE = /Citation does not match retrieved evidence|引用与检索证据不匹配/
const DIAGNOSTICS_RE = /Diagnostics|诊断信息/

const NOT_FROM_KB_RE = /Not from KB|非本地文献库|闈炴湰鍦版枃鐚簱/
const LOCAL_EXTERNAL_RE = /Local \+ external|文献库 \+ 外部补充|鏂囩尞搴?.*澶栭儴琛ュ厖/

function agentSummaryViewModel(overrides: Partial<AgentSourceSummaryViewModel> = {}): AgentSourceSummaryViewModel {
  return {
    evidenceLabel: 'Evidence grounded',
    evidenceStatus: 'grounded',
    totalClaims: 2,
    supportedClaims: 1,
    unsupportedClaims: 1,
    qualityGateStatus: 'repaired',
    qualityGateTitle: 'citation repair applied',
    taskLabel: 'Single paper',
    scopeSummary: 'library / 2 selected',
    hasErrors: true,
    researchRunStatus: 'done',
    evidenceMatrixRows: 3,
    sourcePolicy: 'local_only',
    evidenceMatrix: [],
    subtaskCount: 0,
    unsupportedClaimRows: [],
    references: [],
    ...overrides,
  }
}

type SummaryChipSnapshot = {
  id: string
  className?: string
  label: string | number
  testId?: string
  title?: string
  value: string | number
}

async function agentHeaderSummary(page: Page, input: AgentTraceHeaderSummaryInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (summaryInput) => {
    const { buildAgentTraceHeaderSummary } = await import('/src/components/chat/agentTraceHeaderSummary.ts')
    return buildAgentTraceHeaderSummary({}, summaryInput)
  }, input)
}

async function agentMetricCounts(page: Page, input: AgentTraceMetricCountInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (summaryInput) => {
    const { buildAgentTraceMetricCounts } = await import('/src/components/chat/agentTraceMetricCounts.ts')
    return buildAgentTraceMetricCounts(summaryInput)
  }, input)
}

async function agentScopeSummary(page: Page, input: AgentTraceScopeSummaryInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (summaryInput) => {
    const { buildAgentTraceScopeSummary } = await import('/src/components/chat/agentTraceScopeSummary.ts')
    return buildAgentTraceScopeSummary(summaryInput)
  }, input)
}

async function agentSourceStatus(page: Page, input: Omit<AgentTraceSourceStatusInput, 'labels'>) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (summaryInput) => {
    const { buildAgentTraceSourceStatus } = await import('/src/components/chat/agentTraceSourceStatus.ts')
    return buildAgentTraceSourceStatus({
      ...summaryInput,
      labels: {},
    })
  }, input)
}

async function agentSourceRows(page: Page, input: AgentTraceSourceRowsInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (rowsInput) => {
    const { buildAgentTraceSourceRows } = await import('/src/components/chat/agentTraceSourceRows.ts')
    return buildAgentTraceSourceRows(rowsInput)
  }, input)
}

async function agentTraceViewModel(page: Page, trace: Record<string, unknown>): Promise<AgentTraceViewModel> {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (inputTrace) => {
    const { buildAgentTraceViewModel } = await import('/src/components/chat/agentTraceViewModel.ts')
    return buildAgentTraceViewModel(inputTrace, {})
  }, trace)
}

async function agentTracePanelState(page: Page, input: AgentTracePanelStateInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (stateInput) => {
    const { buildAgentTracePanelState } = await import('/src/components/chat/agentTracePanelState.ts')
    return buildAgentTracePanelState(stateInput)
  }, input)
}

async function archivedTraceSnapshot(page: Page, input: ArchivedAgentTraceSnapshotInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (snapshotInput) => {
    const { buildArchivedAgentTraceSnapshot } = await import('/src/components/chat/agentTraceArchiveState.ts')
    return buildArchivedAgentTraceSnapshot(snapshotInput)
  }, input)
}

async function archivedTraceLoadedState(page: Page, messageId: number, response: AgentTraceAuditResponse) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async ({ inputMessageId, inputResponse }) => {
    const { buildArchivedAgentTraceLoadedState } = await import('/src/components/chat/agentTraceArchiveState.ts')
    return buildArchivedAgentTraceLoadedState(inputMessageId, inputResponse)
  }, {
    inputMessageId: messageId,
    inputResponse: response,
  })
}

async function answerSourceNoticeViewModel(page: Page, input: {
  answerContract?: unknown
  legacySourceSummary?: unknown
  fallbackNoticeText?: string
  allowFallbackNotice?: boolean
  S: Record<string, string>
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (noticeInput) => {
    const { buildAnswerSourceNoticeViewModel } = await import('/src/components/chat/answerSourceNoticeViewModel.ts')
    return buildAnswerSourceNoticeViewModel(noticeInput)
  }, input)
}

async function evidenceDrawerViewModel(page: Page, input: {
  sourceNotice: Record<string, unknown> | null
  citeDetails: Record<string, unknown>[]
  S: Record<string, string>
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (drawerInput) => {
    const { buildEvidenceDrawerViewModel } = await import('/src/components/chat/answerSourceNoticeViewModel.ts')
    return buildEvidenceDrawerViewModel({
      sourceNotice: drawerInput.sourceNotice as never,
      citeDetails: drawerInput.citeDetails as never,
      S: drawerInput.S,
    })
  }, input)
}

async function citationPopoverViewModel(page: Page, input: {
  detail: Record<string, unknown>
  S: Record<string, string>
  loading?: boolean
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (modelInput) => {
    const { buildCitationPopoverViewModel } = await import('/src/components/chat/citationPopoverViewModel.ts')
    return buildCitationPopoverViewModel({
      detail: modelInput.detail as never,
      S: modelInput.S as never,
      loading: Boolean(modelInput.loading),
    })
  }, input)
}

async function citationPopoverPositionSmoke(page: Page) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async () => {
    const {
      getCitationPopoverPositionStyle,
      getHiddenCitationPopoverStyle,
      isCitationPopoverDismissIgnoredTarget,
    } = await import('/src/components/chat/useCitationPopoverPosition.ts')
    const locateButton = document.createElement('button')
    locateButton.className = 'kb-md-locate-inline-btn'
    const locateButtonLabel = document.createElement('span')
    locateButton.append(locateButtonLabel)
    const locateBlock = document.createElement('button')
    locateBlock.setAttribute('data-kb-locate-block-id', 'block-1')
    const ordinaryButton = document.createElement('button')

    return {
      clamped: getCitationPopoverPositionStyle(
        { x: 480, y: 390 },
        { width: 120, height: 100 },
        { width: 500, height: 400 },
      ),
      hidden: getHiddenCitationPopoverStyle({ x: 7, y: 9 }),
      ignoredLocateBlock: isCitationPopoverDismissIgnoredTarget(locateBlock),
      ignoredLocateButtonChild: isCitationPopoverDismissIgnoredTarget(locateButtonLabel),
      ignoredOrdinaryButton: isCitationPopoverDismissIgnoredTarget(ordinaryButton),
      placed: getCitationPopoverPositionStyle(
        { x: 40, y: 50 },
        { width: 100, height: 80 },
        { width: 500, height: 400 },
      ),
    }
  })
}

async function citationPopoverStateSmoke(page: Page) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async () => {
    const { runCitationPopoverStateSmoke } = await import('/src/testing/citationPopoverStateSmoke.ts')
    return runCitationPopoverStateSmoke()
  })
}

async function citationPopoverPreviewSmoke(page: Page) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async () => {
    const { runCitationPopoverPreviewSmoke } = await import('/src/testing/citationPopoverPreviewSmoke.ts')
    return runCitationPopoverPreviewSmoke()
  })
}

async function readerCitationPopoverSmoke(page: Page) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async () => {
    const { runReaderCitationPopoverSmoke } = await import('/src/testing/readerCitationPopoverSmoke.ts')
    return runReaderCitationPopoverSmoke()
  })
}

async function readerCitationShelfSmoke(page: Page) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async () => {
    const { runReaderCitationShelfSmoke } = await import('/src/testing/readerCitationShelfSmoke.ts')
    return runReaderCitationShelfSmoke()
  })
}

async function readerBlockShelfSmoke(page: Page) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async () => {
    const { runReaderBlockShelfSmoke } = await import('/src/testing/readerBlockShelfSmoke.ts')
    return runReaderBlockShelfSmoke()
  })
}

async function citationPopoverMetadataSmoke(page: Page) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async () => {
    const {
      buildCitationPopoverMetadataPlan,
      loadCitationPopoverMetadata,
    } = await import('/src/components/chat/citationPopoverMetadata.ts')
    const systemADetail = {
      anchor: 'cite-a',
      bibliometricsChecked: false,
      citeFmt: '',
      doi: '',
      isInpaper: false,
      num: 1,
      raw: 'Raw citation A',
      sourceName: 'Paper A',
      sourcePath: '/tmp/paper-a.md',
      title: 'Paper A Title',
    }
    const systemBDetail = {
      anchor: 'cite-b',
      bibliometricsChecked: false,
      citeFmt: '',
      doi: '10.1000/system-b',
      isInpaper: true,
      num: 2,
      raw: '',
      sourceName: 'Current Paper',
      sourcePath: '/tmp/current.md',
      title: 'Upstream Paper',
    }
    const quietDetail = {
      anchor: 'cite-c',
      bibliometricsChecked: true,
      citeFmt: '',
      doi: '',
      isInpaper: true,
      num: 3,
      raw: '',
      sourceName: 'Current Paper',
      sourcePath: '',
      title: '',
    }
    const cachedMetricDetail = {
      anchor: 'cite-d',
      bibliometricsChecked: false,
      citeFmt: '',
      doi: '',
      isInpaper: false,
      num: 4,
      raw: '',
      sourceName: 'Cached Paper',
      sourcePath: '',
      summaryLine: 'A useful abstract summary.',
      summarySource: 'abstract',
      title: 'Cached Paper Title',
    }
    const calls: string[] = []
    const client = {
      bibliometrics: async (meta: Record<string, unknown>) => {
        calls.push(`fresh:${String(meta.target_locale || '')}`)
        return { kind: 'fresh' }
      },
      bibliometricsCached: async (meta: Record<string, unknown>) => {
        calls.push(`cached:${String(meta.target_locale || '')}`)
        return { kind: 'cached' }
      },
      citationMetaCached: async (sourcePath: string) => {
        calls.push(`citation:${sourcePath}`)
        return { sourcePath }
      },
    }
    const systemAPlan = buildCitationPopoverMetadataPlan(systemADetail as never, 'key-a')
    const systemAResult = await loadCitationPopoverMetadata(systemADetail as never, {
      client,
      plan: systemAPlan,
    })
    const systemBPlan = buildCitationPopoverMetadataPlan(systemBDetail as never, 'key-b')
    const systemBResult = await loadCitationPopoverMetadata(systemBDetail as never, {
      client,
      plan: systemBPlan,
    })
    const quietPlan = buildCitationPopoverMetadataPlan(quietDetail as never, 'key-c')
    const quietResult = await loadCitationPopoverMetadata(quietDetail as never, {
      client,
      plan: quietPlan,
    })
    const cachedMetricPlan = buildCitationPopoverMetadataPlan(cachedMetricDetail as never, 'key-d')
    const cachedMetricResult = await loadCitationPopoverMetadata(cachedMetricDetail as never, {
      client,
      plan: cachedMetricPlan,
    })

    return {
      cachedMetricMetas: cachedMetricResult.metas,
      cachedMetricPlan,
      calls,
      quietMetas: quietResult.metas,
      quietPlan,
      systemAMetas: systemAResult.metas,
      systemAPlan,
      systemBMetas: systemBResult.metas,
      systemBPlan,
    }
  })
}

async function readerCitationPopoverMetadataSmoke(page: Page) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async () => {
    const {
      buildReaderCitationPopoverMetadataPlan,
      loadReaderCitationPopoverMetadata,
      orderReaderCitationPopoverMetas,
    } = await import('/src/components/chat/readerCitationPopoverMetadata.ts')
    const freshDetail = {
      anchor: 'reader-a',
      bibliometricsChecked: false,
      citeFmt: '',
      doi: '10.1000/reader-a',
      isInpaper: true,
      num: 1,
      raw: '',
      sourceName: 'Current Paper',
      sourcePath: '/tmp/current.md',
      title: 'Reader Upstream A',
    }
    const cachedDetail = {
      anchor: 'reader-b',
      bibliometricsChecked: false,
      citeFmt: '',
      doi: '',
      isInpaper: false,
      num: 2,
      raw: '',
      sourceName: 'Reader Paper B',
      sourcePath: '',
      summaryLine: 'A usable article abstract.',
      summarySource: 'abstract',
      title: 'Reader Paper B',
    }
    const missingDetail = {
      anchor: 'reader-c',
      bibliometricsChecked: false,
      bindingStatus: 'missing_reference_entry',
      citeFmt: '',
      doi: '10.1000/missing',
      isInpaper: true,
      num: 3,
      raw: '',
      sourceName: 'Current Paper',
      sourcePath: '/tmp/current.md',
      title: 'Missing Reference',
    }
    const calls: string[] = []
    const client = {
      bibliometrics: async (meta: Record<string, unknown>) => {
        calls.push(`fresh:${String(meta.target_locale || '')}`)
        return {
          kind: 'fresh',
          summary_line: 'Fresh article abstract.',
          summary_source: 'abstract',
        }
      },
      bibliometricsCached: async (meta: Record<string, unknown>) => {
        calls.push(`cached:${String(meta.target_locale || '')}`)
        return { kind: 'cached' }
      },
      citationCardPolishCached: async (_detail: Record<string, unknown>, waitSeconds: number) => {
        calls.push(`polish:${waitSeconds}`)
        return { kind: 'polish' }
      },
    }
    const freshPlan = buildReaderCitationPopoverMetadataPlan(freshDetail as never, 'reader-key-a')
    const freshResult = await loadReaderCitationPopoverMetadata(freshDetail as never, {
      client,
      plan: freshPlan,
    })
    const cachedPlan = buildReaderCitationPopoverMetadataPlan(cachedDetail as never, 'reader-key-b')
    const cachedResult = await loadReaderCitationPopoverMetadata(cachedDetail as never, {
      client,
      plan: cachedPlan,
    })
    const missingPlan = buildReaderCitationPopoverMetadataPlan(missingDetail as never, 'reader-key-c')
    const missingResult = await loadReaderCitationPopoverMetadata(missingDetail as never, {
      client,
      plan: missingPlan,
    })
    const orderedKinds = orderReaderCitationPopoverMetas([
      { summary_line: 'Abstract', summary_source: 'abstract', kind: 'summary' },
      { kind: 'metadata' },
      {},
    ]).map((meta) => String(meta.kind || ''))

    return {
      cachedMetas: cachedResult.metas,
      cachedPlan,
      calls,
      freshMetas: freshResult.metas,
      freshPlan,
      missingMetas: missingResult.metas,
      missingPlan,
      orderedKinds,
    }
  })
}

async function citationPopoverFrameModel(page: Page, input: {
  detail: Record<string, unknown>
  S: Record<string, string>
  isSystemB?: boolean
  viewHeader?: Record<string, string>
  locatorSection?: Record<string, unknown>
  cardLocatorLabel?: string
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (modelInput) => {
    const { buildCitationPopoverFrameModel } = await import('/src/components/chat/citationPopoverFrameModel.ts')
    return buildCitationPopoverFrameModel({
      detail: modelInput.detail as never,
      S: modelInput.S as never,
      isSystemB: Boolean(modelInput.isSystemB),
      viewHeader: {
        kicker: modelInput.viewHeader?.kicker || '',
        title: modelInput.viewHeader?.title || '',
        subtitle: modelInput.viewHeader?.subtitle || '',
      },
      locatorSection: modelInput.locatorSection as never,
      cardLocatorLabel: modelInput.cardLocatorLabel || '',
      localizeKnownLabel: (value: string) => String(value || '').trim(),
    })
  }, input)
}

async function citationPopoverLocalization(page: Page, input: {
  labels: string[]
  bodies: string[]
  S: Record<string, string>
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (localizationInput) => {
    const { buildCitationPopoverLocalizers } = await import('/src/components/chat/citationPopoverLocalization.ts')
    const { localizeKnownBody, localizeKnownLabel } = buildCitationPopoverLocalizers(localizationInput.S as never)
    return {
      bodies: localizationInput.bodies.map((item) => localizeKnownBody(item)),
      labels: localizationInput.labels.map((item) => localizeKnownLabel(item)),
    }
  }, input)
}

async function citationPopoverStatusModel(page: Page, input: {
  detail: Record<string, unknown>
  S: Record<string, string>
  isSystemB?: boolean
  supportSection?: Record<string, unknown>
  warningSection?: Record<string, unknown>
  displayMain?: string
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (modelInput) => {
    const { buildCitationPopoverStatusModel } = await import('/src/components/chat/citationPopoverStatusModel.ts')
    return buildCitationPopoverStatusModel({
      detail: modelInput.detail as never,
      S: modelInput.S as never,
      isSystemB: Boolean(modelInput.isSystemB),
      supportSection: modelInput.supportSection as never,
      warningSection: modelInput.warningSection as never,
      displayMain: modelInput.displayMain || '',
      localizeKnownBody: (value: string) => String(value || '').trim(),
      localizeKnownLabel: (value: string) => String(value || '').trim(),
    })
  }, input)
}

async function systemBLiteratureCardModel(page: Page, input: {
  detail: Record<string, unknown>
  S: Record<string, string>
  isSystemB?: boolean
  loading?: boolean
  locatorSection?: Record<string, unknown>
  contextSummarySection?: Record<string, unknown>
  referenceSection?: Record<string, unknown>
  cardTakeaway?: string
  cardEvidenceLabel?: string
  cardReferenceLabel?: string
  cardSupportLabel?: string
  cardQualityFlags?: string[]
  sourcePaperText?: string
  headingPath?: string
  pageLabel?: string
  badgeLabel?: string
  doiLabel?: string
  systemBTitle?: string
  systemBTitleMissing?: boolean
  headerSubtitle?: string
  metrics?: string[]
  explicitSupportText?: string
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (modelInput) => {
    const { buildSystemBLiteratureCardModel } = await import('/src/components/chat/citationPopoverSystemB.ts')
    return buildSystemBLiteratureCardModel({
      detail: modelInput.detail as never,
      S: modelInput.S as never,
      isSystemB: modelInput.isSystemB ?? true,
      loading: Boolean(modelInput.loading),
      locatorSection: modelInput.locatorSection as never,
      contextSummarySection: modelInput.contextSummarySection as never,
      referenceSection: modelInput.referenceSection as never,
      cardTakeaway: modelInput.cardTakeaway || '',
      cardEvidenceLabel: modelInput.cardEvidenceLabel || '',
      cardReferenceLabel: modelInput.cardReferenceLabel || '',
      cardSupportLabel: modelInput.cardSupportLabel || '',
      cardQualityFlags: modelInput.cardQualityFlags || [],
      sourcePaperText: modelInput.sourcePaperText || '',
      headingPath: modelInput.headingPath || '',
      pageLabel: modelInput.pageLabel || '',
      badgeLabel: modelInput.badgeLabel || '',
      doiLabel: modelInput.doiLabel || '',
      systemBTitle: modelInput.systemBTitle || '',
      systemBTitleMissing: Boolean(modelInput.systemBTitleMissing),
      headerSubtitle: modelInput.headerSubtitle || '',
      metrics: modelInput.metrics || [],
      explicitSupportText: modelInput.explicitSupportText || '',
      displaySource: String(modelInput.detail.sourceName || ''),
      localizeKnownBody: (value: string) => String(value || '').trim(),
      localizeKnownLabel: (value: string) => String(value || '').trim(),
    })
  }, input)
}

async function systemBOverviewState(page: Page, input: {
  S: Record<string, string>
  isSystemB?: boolean
  loading?: boolean
  paperOverviewText?: string
  showReference?: boolean
  bibliometricsChecked?: boolean
  doiLabel?: string
  systemBTitle?: string
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (stateInput) => {
    const { buildSystemBOverviewState } = await import('/src/components/chat/citationPopoverSystemBOverviewState.ts')
    return buildSystemBOverviewState({
      S: stateInput.S as never,
      isSystemB: stateInput.isSystemB ?? true,
      loading: Boolean(stateInput.loading),
      paperOverviewText: stateInput.paperOverviewText || '',
      showReference: Boolean(stateInput.showReference),
      bibliometricsChecked: Boolean(stateInput.bibliometricsChecked),
      doiLabel: stateInput.doiLabel || '',
      systemBTitle: stateInput.systemBTitle || '',
    })
  }, input)
}

async function systemBTextPanelsModel(page: Page, input: {
  detail: Record<string, unknown>
  S: Record<string, string>
  isSystemB?: boolean
  contextSummarySection?: Record<string, unknown>
  referenceSection?: Record<string, unknown>
  cardTakeaway?: string
  cardEvidenceLabel?: string
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (modelInput) => {
    const { buildSystemBTextPanelsModel } = await import('/src/components/chat/citationPopoverSystemBTextPanels.ts')
    return buildSystemBTextPanelsModel({
      detail: modelInput.detail as never,
      S: modelInput.S as never,
      isSystemB: modelInput.isSystemB ?? true,
      contextSummarySection: modelInput.contextSummarySection as never,
      referenceSection: modelInput.referenceSection as never,
      cardTakeaway: modelInput.cardTakeaway || '',
      cardEvidenceLabel: modelInput.cardEvidenceLabel || '',
      localizeKnownBody: (value: string) => String(value || '').trim(),
      localizeKnownLabel: (value: string) => String(value || '').trim(),
    })
  }, input)
}

async function systemBTraceModel(page: Page, input: {
  detail: Record<string, unknown>
  S: Record<string, string>
  isSystemB?: boolean
  traceEnabled?: boolean
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (modelInput) => {
    const { buildSystemBTraceModel } = await import('/src/components/chat/citationPopoverSystemBTrace.ts')
    return buildSystemBTraceModel({
      detail: modelInput.detail as never,
      S: modelInput.S as never,
      isSystemB: modelInput.isSystemB ?? true,
      traceEnabled: modelInput.traceEnabled,
    })
  }, input)
}

async function systemBSourcePanelsModel(page: Page, input: {
  detail: Record<string, unknown>
  S: Record<string, string>
  isSystemB?: boolean
  locatorSection?: Record<string, unknown>
  cardReferenceLabel?: string
  cardSupportLabel?: string
  cardQualityFlags?: string[]
  sourcePaperText?: string
  headingPath?: string
  pageLabel?: string
  badgeLabel?: string
  doiLabel?: string
  systemBTitle?: string
  systemBTitleMissing?: boolean
  headerSubtitle?: string
  metrics?: string[]
  explicitSupportText?: string
  displaySource?: string
  systemBContextSource?: string
  systemBReferenceText?: string
  systemBExplicitReferenceText?: string
  paperOverviewText?: string
  citationContextText?: string
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (modelInput) => {
    const { buildSystemBSourcePanelsModel } = await import('/src/components/chat/citationPopoverSystemBSourcePanels.ts')
    return buildSystemBSourcePanelsModel({
      detail: modelInput.detail as never,
      S: modelInput.S as never,
      isSystemB: modelInput.isSystemB ?? true,
      locatorSection: modelInput.locatorSection as never,
      cardReferenceLabel: modelInput.cardReferenceLabel || '',
      cardSupportLabel: modelInput.cardSupportLabel || '',
      cardQualityFlags: modelInput.cardQualityFlags || [],
      sourcePaperText: modelInput.sourcePaperText || '',
      headingPath: modelInput.headingPath || '',
      pageLabel: modelInput.pageLabel || '',
      badgeLabel: modelInput.badgeLabel || '',
      doiLabel: modelInput.doiLabel || '',
      systemBTitle: modelInput.systemBTitle || '',
      systemBTitleMissing: Boolean(modelInput.systemBTitleMissing),
      headerSubtitle: modelInput.headerSubtitle || '',
      metrics: modelInput.metrics || [],
      explicitSupportText: modelInput.explicitSupportText || '',
      displaySource: modelInput.displaySource || String(modelInput.detail.sourceName || ''),
      systemBContextSource: modelInput.systemBContextSource || '',
      systemBReferenceText: modelInput.systemBReferenceText || '',
      systemBExplicitReferenceText: modelInput.systemBExplicitReferenceText || '',
      paperOverviewText: modelInput.paperOverviewText || '',
      citationContextText: modelInput.citationContextText || '',
    })
  }, input)
}

async function assistantMessageNoticeViewModel(page: Page, input: {
  message: Record<string, unknown>
  lowConfidenceMeta: Record<string, unknown> | null
  provenanceModeLabel?: string
  showProvenanceModeLabel?: boolean
  S: Record<string, string>
}) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (noticeInput) => {
    const { buildAssistantMessageNoticeViewModel } = await import('/src/components/chat/assistantMessageNoticeViewModel.ts')
    return buildAssistantMessageNoticeViewModel({
      message: noticeInput.message as never,
      lowConfidenceMeta: noticeInput.lowConfidenceMeta as never,
      provenanceModeLabel: noticeInput.provenanceModeLabel || '',
      showProvenanceModeLabel: Boolean(noticeInput.showProvenanceModeLabel),
      S: noticeInput.S,
    })
  }, input)
}

async function agentQualityGateTitle(page: Page, input: AgentTraceQualityGateTitleInput) {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (summaryInput) => {
    const { buildAgentTraceQualityGateTitle } = await import('/src/components/chat/agentTraceQualityGate.ts')
    return buildAgentTraceQualityGateTitle(summaryInput)
  }, input)
}

async function visibleSummaryChips(page: Page, viewModel: AgentSourceSummaryViewModel): Promise<SummaryChipSnapshot[]> {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')
  return page.evaluate(async (model) => {
    const { buildAgentTraceSummaryChips } = await import('/src/components/chat/agentTraceSummaryChips.ts')
    return buildAgentTraceSummaryChips({}, model)
      .filter((chip) => chip.visible !== false)
      .map((chip) => ({
        id: chip.id,
        className: chip.className,
        label: typeof chip.label === 'number' || typeof chip.label === 'string' ? chip.label : String(chip.label ?? ''),
        testId: chip.testId,
        title: chip.title,
        value: typeof chip.value === 'number' || typeof chip.value === 'string' ? chip.value : String(chip.value ?? ''),
      }))
  }, viewModel)
}

test.beforeEach(async ({ page }) => {
  await installAppShellMocks(page)
  await installIdleReferenceMocks(page)
  await installEmptyCitationShelfMock(page, {
    scopeId: 'message-list-regression-project',
    projectId: 'message-list-regression-project',
  })
})

test('agent trace metric counts prefer summary values when present', async ({ page }) => {
  const counts = await agentMetricCounts(page, {
    summary: {
      total_claims: 4,
      supported_claims: 3,
      unsupported_claims: 1,
      plan_step_count: 5,
      tool_call_count: 6,
      has_errors: true,
      evidence_matrix_rows: 7,
      subtask_count: 8,
    },
    verification: {
      total_claims: 1,
      supported_claims: 1,
      unsupported_claims: 0,
    },
    planCount: 1,
    stepCount: 2,
    errorCount: 0,
    evidenceMatrixCount: 3,
    researchSubtaskCount: 4,
  })

  expect(counts).toEqual({
    totalClaims: 4,
    supportedClaims: 3,
    unsupportedClaims: 1,
    planStepCount: 5,
    toolCallCount: 6,
    hasErrors: true,
    evidenceMatrixRows: 7,
    subtaskCount: 8,
  })
})

test('agent trace metric counts fall back to source records and lengths', async ({ page }) => {
  const counts = await agentMetricCounts(page, {
    summary: {},
    verification: {
      total_claims: 2,
      supported_claims: 1,
      unsupported_claims: 1,
    },
    planCount: 3,
    stepCount: 4,
    errorCount: 1,
    evidenceMatrixCount: 5,
    researchSubtaskCount: 6,
  })

  expect(counts).toEqual({
    totalClaims: 2,
    supportedClaims: 1,
    unsupportedClaims: 1,
    planStepCount: 3,
    toolCallCount: 4,
    hasErrors: true,
    evidenceMatrixRows: 5,
    subtaskCount: 6,
  })
})

test('agent trace source status prefers summary values and keeps not-applicable tasks general', async ({ page }) => {
  const status = await agentSourceStatus(page, {
    summary: {
      evidence_status: 'not_applicable',
      question_type: 'single_paper_qa',
      quality_gate_status: ' Repaired ',
      quality_gate_reasons: ['citation repair applied'],
      research_run_status: 'done',
      source_policy: 'external_allowed_with_notice',
    },
    verification: {
      evidence_status: 'grounded',
    },
    researchRun: {
      status: 'fallback',
      source_policy: 'local_only',
    },
    traceQuestionType: 'reading_guide',
  })

  expect(status).toEqual({
    evidenceStatus: 'not_applicable',
    evidenceLabel: 'Not from KB',
    qualityGateStatus: 'repaired',
    qualityGateTitle: 'citation repair applied',
    taskLabel: 'General',
    researchRunStatus: 'done',
    sourcePolicy: 'external_allowed_with_notice',
  })
})

test('agent trace source status falls back to verification, research run, and trace task type', async ({ page }) => {
  const status = await agentSourceStatus(page, {
    summary: {},
    verification: {
      evidence_status: 'grounded',
    },
    researchRun: {
      status: 'done',
      source_policy: 'local_only',
    },
    traceQuestionType: 'reading_guide',
  })

  expect(status).toEqual({
    evidenceStatus: 'grounded',
    evidenceLabel: 'Evidence grounded',
    qualityGateStatus: '',
    qualityGateTitle: '',
    taskLabel: 'Reading guide',
    researchRunStatus: 'done',
    sourcePolicy: 'local_only',
  })
})

test('agent trace source rows limit unsupported claims and dedupe step references', async ({ page }) => {
  const rows = await agentSourceRows(page, {
    verification: {
      claims: [
        { text: 'supported claim', supported: true },
        { text: 'unsupported explicit', supported: false },
        { text: 'unsupported reason only', unsupported_reason: 'missing_evidence_overlap' },
        { text: 'unsupported second', supported: false },
        { text: 'unsupported third omitted by limit', supported: false },
      ],
    },
    steps: [
      {
        output: {
          references: [
            { title: 'Ref A', source_path: '/tmp/a.md', ref_num: 1 },
            { title: 'Ref A', source_path: '/tmp/a.md', ref_num: 1 },
            { title: 'Ref B', source_path: '/tmp/b.md', ref_num: 2 },
          ],
        },
      },
      {
        output: {
          references: [
            { title: 'Ref C', source_path: '/tmp/c.md', ref_num: 3 },
            { title: 'Ref D omitted by limit', source_path: '/tmp/d.md', ref_num: 4 },
          ],
        },
      },
    ],
    unsupportedClaimLimit: 3,
    referenceLimit: 3,
  })

  expect(rows.unsupportedClaimRows.map((claim) => claim.text)).toEqual([
    'unsupported explicit',
    'unsupported reason only',
    'unsupported second',
  ])
  expect(rows.references.map((ref) => ref.title)).toEqual(['Ref A', 'Ref B', 'Ref C'])
})

test('agent trace view model assembles source summary and diagnostics', async ({ page }) => {
  const model = await agentTraceViewModel(page, {
    question_type: 'single_paper_qa',
    context: {
      query_scope: 'library',
      requested_query_scope: 'current_paper',
      selected_research_context_count: 2,
    },
    summary: {
      evidence_status: 'grounded',
      total_claims: 2,
      supported_claims: 1,
      unsupported_claims: 1,
      quality_gate_status: 'repaired',
      quality_gate_reasons: ['citation repair applied'],
      query_scope: 'library',
      requested_query_scope: 'current_paper',
      plan_step_count: 1,
      tool_call_count: 1,
    },
    verification: {
      claims: [
        { text: 'supported claim', supported: true },
        { text: 'unsupported claim', supported: false, unsupported_reason: 'missing_citation' },
      ],
    },
    plan: [
      { goal: 'Find evidence', status: 'done' },
    ],
    steps: [
      {
        tool: 'retrieve_evidence',
        output: {
          references: [
            { title: 'Ref A', source_path: '/tmp/a.md', ref_num: 1 },
          ],
        },
      },
    ],
    research_run: {
      status: 'done',
      source_policy: 'local_only',
      evidence_matrix: [
        { paper: 'Paper A' },
      ],
      subtasks: [
        { task: 'Subtask A' },
      ],
    },
  })

  expect(model.headerEvidence).toBe('Evidence grounded')
  expect(model.headerContext).toBe('Review 1/2')
  expect(model.sourceSummary.scopeSummary).toBe('library / requested current_paper / 2 selected')
  expect(model.sourceSummary.references.map((ref) => ref.title)).toEqual(['Ref A'])
  expect(model.sourceSummary.unsupportedClaimRows.map((claim) => claim.text)).toEqual(['unsupported claim'])
  expect(model.sourceSummary.researchRunStatus).toBe('done')
  expect(model.sourceSummary.sourcePolicy).toBe('local_only')
  expect(model.sourceSummary.evidenceMatrixRows).toBe(1)
  expect(model.sourceSummary.subtaskCount).toBe(1)
  expect(model.diagnostics.planStepCount).toBe(1)
  expect(model.diagnostics.toolCallCount).toBe(1)
})

test('agent trace panel state gates stored prompts and non-research traces', async ({ page }) => {
  await expect(agentTracePanelState(page, {
    traceRecord: {},
    hasTrace: false,
    canLazyLoad: false,
  })).resolves.toBe('hidden')
  await expect(agentTracePanelState(page, {
    traceRecord: {},
    hasTrace: false,
    canLazyLoad: true,
  })).resolves.toBe('stored_prompt')
  await expect(agentTracePanelState(page, {
    traceRecord: { mode: 'legacy_debug_trace' },
    hasTrace: true,
    canLazyLoad: false,
  })).resolves.toBe('hidden')
  await expect(agentTracePanelState(page, {
    traceRecord: { mode: 'research_agent' },
    hasTrace: true,
    canLazyLoad: false,
  })).resolves.toBe('trace')
})

test('archived trace helper prefers inline trace and isolates loaded state by message id', async ({ page }) => {
  const inlineSnapshot = await archivedTraceSnapshot(page, {
    trace: { mode: 'research_agent', marker: 'inline' },
    loadedState: {
      messageId: 9301,
      trace: { mode: 'research_agent', marker: 'loaded' },
      status: 'loaded',
    },
    messageId: 9301,
    canLoadTrace: true,
    hasLoadHandler: true,
  })
  const switchedSnapshot = await archivedTraceSnapshot(page, {
    trace: null,
    loadedState: {
      messageId: 9301,
      trace: { mode: 'research_agent', marker: 'loaded' },
      status: 'loaded',
    },
    messageId: 9302,
    canLoadTrace: true,
    hasLoadHandler: true,
  })

  expect(inlineSnapshot.traceRecord.marker).toBe('inline')
  expect(inlineSnapshot.hasInitialTrace).toBe(true)
  expect(inlineSnapshot.canLazyLoad).toBe(false)
  expect(switchedSnapshot.traceRecord).toEqual({})
  expect(switchedSnapshot.loadStatus).toBe('idle')
  expect(switchedSnapshot.canLazyLoad).toBe(true)
})

test('archived trace helper merges audit summary without overwriting trace summary', async ({ page }) => {
  const mergedState = await archivedTraceLoadedState(page, 9301, {
    message_id: 9301,
    conv_id: 'message-list-regression:agent-trace-lazy-audit',
    available: true,
    agent_trace: {
      mode: 'research_agent',
      verification: { total_claims: 9 },
    },
    summary: {
      total_claims: 1,
      query_scope: 'library',
    },
  })
  const preservedState = await archivedTraceLoadedState(page, 9301, {
    message_id: 9301,
    conv_id: 'message-list-regression:agent-trace-lazy-audit',
    available: true,
    agent_trace: {
      mode: 'research_agent',
      summary: { query_scope: 'raw-trace-context' },
    },
    summary: {
      query_scope: 'library',
    },
  })
  const emptyState = await archivedTraceLoadedState(page, 9301, {
    message_id: 9301,
    conv_id: 'message-list-regression:agent-trace-lazy-audit',
    available: false,
    agent_trace: {},
    summary: {
      query_scope: 'library',
    },
  })

  expect(mergedState.status).toBe('loaded')
  expect(mergedState.trace?.summary).toEqual({
    total_claims: 1,
    query_scope: 'library',
  })
  expect(mergedState.trace?.verification).toEqual({ total_claims: 9 })
  expect(preservedState.trace?.summary).toEqual({ query_scope: 'raw-trace-context' })
  expect(emptyState).toEqual({
    messageId: 9301,
    trace: null,
    status: 'empty',
  })
})

test('answer source notice view model prefers answer contract over legacy summary', async ({ page }) => {
  const model = await answerSourceNoticeViewModel(page, {
    answerContract: {
      source_summary: {
        kind: 'local_kb',
        label: 'Contract local source',
      },
      source_policy_payload: {
        kind: 'local_kb',
        uses_local_knowledge_base: true,
        uses_external_model: false,
        requires_user_notice: false,
        badge: {
          label_key: 'agent_trace_source_local_only',
          detail: 'Contract-backed evidence is available.',
        },
      },
    },
    legacySourceSummary: {
      kind: 'external_not_kb',
      label: 'Legacy external source',
      detail: 'Legacy summary should not win.',
    },
    S: {
      agent_trace_source_local_only: 'Local KB',
      agent_trace_evidence_not_from_kb: 'Not from KB',
      agent_trace_source_fallback: 'Source',
    },
  })

  expect(model).toEqual({
    label: 'Local KB',
    title: 'Contract-backed evidence is available.',
    kind: 'local_kb',
    usesLocalKnowledgeBase: true,
    usesExternalModel: false,
    requiresUserNotice: false,
  })
})

test('evidence drawer view model dedupes cards and derives compact source detail', async ({ page }) => {
  const drawer = await evidenceDrawerViewModel(page, {
    sourceNotice: {
      label: 'Local + external',
      title: 'Local citations [n] come from the knowledge base.',
      kind: 'local_plus_external',
      usesLocalKnowledgeBase: true,
      usesExternalModel: true,
      requiresUserNotice: true,
    },
    citeDetails: [
      {
        num: 1,
        displayNum: 1,
        sourcePath: '/tmp/a.md',
        sourceName: 'Paper A',
        cardEvidence: 'same evidence',
      },
      {
        num: 1,
        displayNum: 1,
        sourcePath: '/tmp/a.md',
        sourceName: 'Paper A',
        cardEvidence: 'same evidence',
      },
      {
        num: 2,
        displayNum: 2,
        sourcePath: '/tmp/b.md',
        sourceName: 'Paper B',
        cardEvidence: 'different evidence',
      },
    ],
    S: {
      msg_evidence_label: 'Evidence',
      agent_trace_label_evidence: 'Evidence',
      agent_trace_source_fallback: 'Source',
    },
  })

  expect(drawer.title).toBe('Evidence')
  expect(drawer.subtitle).toBe('Local + external')
  expect(drawer.sourceDetail).toBe('Local citations are grounded in the knowledge base; external context may supplement uncited background.')
  expect(drawer.visibleDetails.map((detail) => detail.sourcePath)).toEqual(['/tmp/a.md', '/tmp/b.md'])
})

test('citation popover position helpers clamp placement and preserve dismiss exemptions', async ({ page }) => {
  const position = await citationPopoverPositionSmoke(page)

  expect(position.placed).toEqual({ left: 50, top: 78 })
  expect(position.clamped).toEqual({ left: 368, top: 288 })
  expect(position.hidden).toEqual({ left: 17, top: 19, visibility: 'hidden' })
  expect(position.ignoredLocateButtonChild).toBe(true)
  expect(position.ignoredLocateBlock).toBe(true)
  expect(position.ignoredOrdinaryButton).toBe(false)
})

test('citation popover state hook opens, merges metadata, and closes predictably', async ({ page }) => {
  const state = await citationPopoverStateSmoke(page)

  expect(state.usableMetaCount).toBe(1)
  expect(state.renderedText).toBe('empty')
  expect(state.snapshots).toEqual([
    {
      doi: '',
      guideLoading: false,
      loading: false,
      pinned: false,
      requestKey: '',
      title: '',
      x: null,
      y: null,
    },
    {
      doi: '',
      guideLoading: false,
      loading: true,
      pinned: true,
      requestKey: 'cite-a|State Paper|4',
      title: 'State Paper Title',
      x: 12,
      y: 34,
    },
    {
      doi: '10.1000/state',
      guideLoading: false,
      loading: true,
      pinned: true,
      requestKey: 'cite-a|State Paper|4',
      title: 'State Paper Title',
      x: 12,
      y: 34,
    },
    {
      doi: '10.1000/state',
      guideLoading: true,
      loading: true,
      pinned: true,
      requestKey: 'cite-a|State Paper|4',
      title: 'State Paper Title',
      x: 12,
      y: 34,
    },
    {
      doi: '',
      guideLoading: false,
      loading: false,
      pinned: false,
      requestKey: '',
      title: '',
      x: null,
      y: null,
    },
  ])
})

test('citation popover preview hook manages hover timers and polish retries', async ({ page }) => {
  const preview = await citationPopoverPreviewSmoke(page)

  expect(preview.events).toEqual([
    'open',
    'close',
    'polish:preview-key:10.1000/preview',
  ])
  expect(preview.fetchCalls).toBe(2)
  expect(preview.polishWaitSeconds).toEqual([4, 4])
})

test('reader citation popover hook opens reader citations and ignores stale metadata', async ({ page }) => {
  const popover = await readerCitationPopoverSmoke(page)

  expect(popover.calls).toHaveLength(2)
  expect(popover.calls[0]).toContain('Slow Citation:')
  expect(popover.calls[1]).toContain('Fast Citation:')
  expect(popover.renderedText).toBe('ready')
  expect(popover.snapshots).toEqual([
    { loading: false, title: '', x: null, y: null },
    { loading: true, title: 'Slow Citation', x: 12, y: 34 },
    { loading: true, title: 'Fast Citation', x: 56, y: 78 },
    { loading: false, title: 'Fast Citation', x: 56, y: 78 },
  ])
})

test('reader citation shelf hook tracks local citation membership and preserves add callbacks', async ({ page }) => {
  const shelf = await readerCitationShelfSmoke(page)

  expect(shelf.events).toEqual([
    'add:Primary Citation',
    'add:Primary Citation',
    'add:Other Citation',
  ])
  expect(shelf.renderedText).toBe('true|true|2')
  expect(shelf.snapshots).toEqual([
    { otherInShelf: false, primaryInShelf: false, size: 0 },
    { otherInShelf: false, primaryInShelf: true, size: 1 },
    { otherInShelf: false, primaryInShelf: true, size: 1 },
    { otherInShelf: true, primaryInShelf: true, size: 2 },
  ])
})

test('reader block shelf hook builds selection payloads from reader block actions', async ({ page }) => {
  const shelf = await readerBlockShelfSmoke(page)

  expect(shelf.canAddBlockToShelf).toBe(true)
  expect(shelf.renderedText).toBe('true')
  expect(shelf.emptyPayload).toBeNull()
  expect(shelf.directPayload).toMatchObject({
    anchorId: 'anchor-direct',
    anchorKind: 'figure',
    blockId: 'block-direct',
    createdAt: 12345,
    headingPath: 'Intro / Figure',
    sourceName: 'Reader Paper',
    sourcePath: '/tmp/reader.md',
    text: 'Direct figure text',
  })
  expect(shelf.events).toEqual([
    {
      anchorId: 'anchor-a',
      anchorKind: 'table',
      blockId: 'block-a',
      createdAt: 67890,
      headingPath: 'Methods',
      sourceName: 'Reader Paper',
      sourcePath: '/tmp/reader.md',
      text: 'Table text',
    },
  ])
})

test('citation popover metadata helper plans route-specific citation and metric requests', async ({ page }) => {
  const metadata = await citationPopoverMetadataSmoke(page)

  expect(metadata.systemAPlan).toMatchObject({
    itemKey: 'key-a',
    needsSummaryBackfill: true,
    requestCount: 2,
    shouldFetchBibliometrics: true,
    shouldFetchCitationMeta: true,
    sourcePath: '/tmp/paper-a.md',
  })
  expect(metadata.systemAMetas).toEqual([
    { sourcePath: '/tmp/paper-a.md' },
    { kind: 'fresh' },
  ])
  expect(metadata.systemBPlan).toMatchObject({
    itemKey: 'key-b',
    needsSummaryBackfill: true,
    requestCount: 1,
    shouldFetchBibliometrics: true,
    shouldFetchCitationMeta: false,
    sourcePath: '/tmp/current.md',
  })
  expect(metadata.systemBMetas).toEqual([{ kind: 'fresh' }])
  expect(metadata.quietPlan).toMatchObject({
    itemKey: 'key-c',
    requestCount: 0,
    shouldFetchBibliometrics: false,
    shouldFetchCitationMeta: false,
    sourcePath: '',
  })
  expect(metadata.quietMetas).toEqual([])
  expect(metadata.cachedMetricPlan).toMatchObject({
    itemKey: 'key-d',
    needsSummaryBackfill: false,
    requestCount: 1,
    shouldFetchBibliometrics: true,
    shouldFetchCitationMeta: false,
    sourcePath: '',
  })
  expect(metadata.cachedMetricMetas).toEqual([{ kind: 'cached' }])
  expect(metadata.calls).toHaveLength(4)
  expect(metadata.calls[0]).toBe('citation:/tmp/paper-a.md')
  expect(metadata.calls[1]).toMatch(/^fresh:/)
  expect(metadata.calls[2]).toMatch(/^fresh:/)
  expect(metadata.calls[3]).toMatch(/^cached:/)
})

test('reader citation popover metadata helper plans reader-specific polish and metric requests', async ({ page }) => {
  const metadata = await readerCitationPopoverMetadataSmoke(page)

  expect(metadata.freshPlan).toMatchObject({
    itemKey: 'reader-key-a',
    missingReferenceEntry: false,
    needsSummaryBackfill: true,
    requestCount: 2,
    shouldFetchBibliometrics: true,
    shouldFetchPolish: true,
  })
  expect(metadata.freshMetas).toEqual([
    { kind: 'polish' },
    {
      kind: 'fresh',
      summary_line: 'Fresh article abstract.',
      summary_source: 'abstract',
    },
  ])
  expect(metadata.cachedPlan).toMatchObject({
    itemKey: 'reader-key-b',
    missingReferenceEntry: false,
    needsSummaryBackfill: false,
    requestCount: 2,
    shouldFetchBibliometrics: true,
    shouldFetchPolish: true,
  })
  expect(metadata.cachedMetas).toEqual([
    { kind: 'cached' },
    { kind: 'polish' },
  ])
  expect(metadata.missingPlan).toMatchObject({
    itemKey: 'reader-key-c',
    missingReferenceEntry: true,
    requestCount: 0,
    shouldFetchBibliometrics: false,
    shouldFetchPolish: false,
  })
  expect(metadata.missingMetas).toEqual([])
  expect(metadata.orderedKinds).toEqual(['metadata', 'summary'])
  expect(metadata.calls).toHaveLength(4)
  expect(metadata.calls[0]).toMatch(/^fresh:/)
  expect(metadata.calls[1]).toBe('polish:1.5')
  expect(metadata.calls[2]).toMatch(/^cached:/)
  expect(metadata.calls[3]).toBe('polish:1.5')
})

test('citation popover view model assembles route-specific frame, status, and cards', async ({ page }) => {
  const S = {
    cite_anchor_equation: 'Equation',
    cite_anchor_figure: 'Figure',
    cite_anchor_label: 'Anchor',
    cite_anchor_paragraph: 'Paragraph',
    cite_anchor_sentence: 'Sentence',
    cite_anchor_table: 'Table',
    cite_answer_point: 'Answer point',
    cite_binding_candidate: 'Candidate evidence',
    cite_binding_mismatch: 'Citation mismatch',
    cite_candidate_support_default: 'Candidate support fallback',
    cite_context: 'Context',
    cite_context_summary: 'Context summary',
    cite_current_paper_usage: 'Current paper usage',
    cite_evidence_chain: 'Evidence chain',
    cite_evidence_focus: 'Evidence focus',
    cite_external_metadata_warning: 'External metadata differs',
    cite_external_title: 'External title: {title}',
    cite_frontend_candidate_reason: 'Frontend candidate fallback',
    cite_kind_evidence: 'Answer evidence',
    cite_kind_upstream: 'Upstream citation',
    cite_loading: 'Loading',
    cite_loading_summary: 'Loading summary',
    cite_location_current: 'Current paper location',
    cite_location_paper: 'Source paper',
    cite_meta_author: 'Author',
    cite_meta_published: 'Published',
    cite_meta_source: 'Source',
    cite_missing_reference_entry: 'Missing entry',
    cite_missing_reference_entry_body: 'Reference {n} is missing from the converted bibliography.',
    cite_note: 'Note',
    cite_open_evidence: 'Open evidence',
    cite_original_evidence: 'Original evidence',
    cite_original_reference_entry: 'Original reference entry',
    cite_paper_overview: 'Article overview',
    cite_position: 'Position',
    cite_read_locate: 'Read location',
    cite_reference_entry: 'Reference entry',
    cite_reliability: 'Reliability',
    cite_summary_unavailable: 'Summary unavailable',
    cite_system_b_support_default: 'Bibliography link from current paper.',
    cite_trace_complete: 'Trace complete',
    cite_trace_review: 'Trace needs review',
    cite_upstream_reference: 'Upstream reference',
    cite_upstream_role: 'Upstream role',
  }
  const systemA = await citationPopoverViewModel(page, {
    detail: {
      num: 3,
      sourceName: 'Fixture Paper',
      sourcePath: '/tmp/fixture.md',
      title: 'Method',
      headingPath: 'Method',
      pageStart: 2,
      pageEnd: 2,
      anchorKind: 'sentence',
      cardClaim: 'The method improves imaging stability.',
      cardEvidence: 'The proposed calibration reduces drift across measurements.',
      cardTakeaway: 'Calibration reduces drift across measurements.',
      cardSupportExplanation: 'The retrieved sentence directly discusses calibration and drift.',
      bindingStatus: 'mismatch',
      bindingReason: 'retrieved evidence does not fully match the answer wording',
      bindingOverlapTerms: ['calibration', 'drift'],
      cardQualityFlags: ['binding_mismatch'],
      cardFlow: ['retrieve', 'verify'],
    },
    S,
  })
  const systemB = await citationPopoverViewModel(page, {
    detail: {
      isInpaper: true,
      num: 12,
      linkedNums: [12],
      sourceName: 'Current Paper',
      sourcePath: '/tmp/current.md',
      title: 'Upstream Work',
      authors: 'Doe J.',
      venue: 'Optics Letters',
      year: '2022',
      summarySource: 'abstract',
      summaryLine: 'This upstream work introduces a calibrated imaging pipeline.',
      citationContextSource: 'abstract',
      cardReferenceEntry: '[12] Doe, J. Upstream Work. Optics Letters, 2022. doi:10.1000/upstream.',
      cardContextSummary: 'The current paper cites this upstream work as calibration background.',
      cardQualityFlags: [],
    },
    S,
  })

  expect(systemA.isSystemB).toBe(false)
  expect(systemA.explainText).toBe('')
  expect(systemA.frame.kindLabel).toBe('Answer evidence')
  expect(systemA.frame.badgeLabel).toBe('#3')
  expect(systemA.frame.primaryActionLabel).toBe('Open evidence')
  expect(systemA.frame.canOpenReader).toBe(true)
  expect(systemA.status.bindingState).toEqual({ label: 'Citation mismatch', tone: 'mismatch' })
  expect(systemA.status.bindingOverlapText).toBe('calibration / drift')
  expect(systemA.systemA.showSupport).toBe(true)
  expect(systemA.systemA.contentCard.evidence).toContain('calibration reduces drift')

  expect(systemB.isSystemB).toBe(true)
  expect(systemB.frame.kindLabel).toBe('Upstream citation')
  expect(systemB.frame.badgeLabel).toBe('[R12]')
  expect(systemB.frame.primaryActionLabel).toBe('Read location')
  expect(systemB.frame.compactMetaItems.map((item) => item.key)).toEqual(['authors', 'published'])
  expect(systemB.systemB.paperOverviewText).toBe('This upstream work introduces a calibrated imaging pipeline.')
  expect(systemB.systemB.takeawayText).toBe('The current paper cites this upstream work as calibration background.')
  expect(systemB.systemB.showOverviewLoading).toBe(false)
})

test('citation popover localization maps known labels and body fallbacks', async ({ page }) => {
  const localized = await citationPopoverLocalization(page, {
    labels: [
      '答案依据',
      '链路需核对',
      'Missing reference entry',
      'Custom untouched label',
    ],
    bodies: [
      'Reference [42] is cited in the opened Reader document, but the converted References section does not contain a matching bibliography entry.',
      '前端缺少后端 cite_details，临时补齐候选依据。',
      '前端根据本轮 References 临时补齐。',
      '这条引用只能作为候选依据，需要人工核对。',
      'Custom body stays as-is.',
    ],
    S: {
      cite_answer_point: 'Answer point',
      cite_anchor_label: 'Anchor',
      cite_binding_candidate: 'Candidate evidence',
      cite_binding_mismatch: 'Citation mismatch',
      cite_candidate_support_default: 'Candidate support fallback',
      cite_context: 'Context',
      cite_context_summary: 'Context summary',
      cite_evidence_chain: 'Evidence chain',
      cite_evidence_focus: 'Evidence focus',
      cite_frontend_candidate_reason: 'Frontend candidate fallback',
      cite_kind_evidence: 'Answer evidence',
      cite_kind_upstream: 'Upstream citation',
      cite_location_current: 'Current location',
      cite_location_paper: 'Source paper',
      cite_meta_author: 'Author',
      cite_meta_published: 'Published',
      cite_meta_source: 'Source',
      cite_missing_reference_entry: 'Missing entry',
      cite_missing_reference_entry_body: 'Reference {n} is missing from the converted bibliography.',
      cite_note: 'Note',
      cite_original_evidence: 'Original evidence',
      cite_position: 'Position',
      cite_reference_entry: 'Reference entry',
      cite_reliability: 'Reliability',
      cite_trace_complete: 'Trace complete',
      cite_trace_review: 'Trace review',
      cite_upstream_reference: 'Upstream reference',
      cite_upstream_role: 'Upstream role',
    },
  })

  expect(localized.labels).toEqual([
    'Answer evidence',
    'Trace review',
    'Missing entry',
    'Custom untouched label',
  ])
  expect(localized.bodies).toEqual([
    'Reference 42 is missing from the converted bibliography.',
    'Frontend candidate fallback',
    'Candidate support fallback',
    'Candidate support fallback',
    'Custom body stays as-is.',
  ])
})

test('citation popover frame model derives route-specific badges, meta, and actions', async ({ page }) => {
  const S = {
    cite_anchor_equation: 'Equation',
    cite_anchor_figure: 'Figure',
    cite_anchor_label: 'Anchor',
    cite_anchor_paragraph: 'Paragraph',
    cite_anchor_sentence: 'Sentence',
    cite_anchor_table: 'Table',
    cite_kind_evidence: 'Answer evidence',
    cite_kind_upstream: 'Upstream citation',
    cite_meta_published: 'Published',
    cite_meta_source: 'Source',
    cite_open_evidence: 'Open evidence',
    cite_position: 'Position',
    cite_read_locate: 'Read location',
    cite_upstream_reference: 'Upstream reference',
  }
  const systemA = await citationPopoverFrameModel(page, {
    detail: {
      num: 9,
      displayNum: 5,
      displayNums: [7, 5],
      sourceName: 'Fixture Paper',
      sourcePath: '/tmp/fixture.md',
      title: 'Fallback title',
      headingPath: 'Method',
      pageStart: 2,
      pageEnd: 3,
      anchorKind: 'sentence',
      doi: '10.1000/frame',
      venue: 'TestConf',
      year: '2024',
      cardFlow: ['retrieve', 'verify'],
    },
    S,
    isSystemB: false,
    viewHeader: {
      title: 'Evidence claim title',
    },
    locatorSection: {
      text: 'Fixture Paper / Method',
    },
    cardLocatorLabel: 'Position',
  })
  const systemB = await citationPopoverFrameModel(page, {
    detail: {
      num: 9,
      linkedNums: [11, 3],
      sourceName: 'Current Paper',
      sourcePath: '/tmp/current.md',
      title: 'Upstream Work',
      authors: 'Doe J.',
      venue: 'Optics Letters',
      year: '2022',
      cardFlow: ['ignored'],
    },
    S,
    isSystemB: true,
  })

  expect(systemA.kindLabel).toBe('Answer evidence')
  expect(systemA.badgeLabel).toBe('#5/7')
  expect(systemA.systemATitle).toBe('Evidence claim title')
  expect(systemA.primaryActionLabel).toBe('Open evidence')
  expect(systemA.flowSteps).toEqual(['retrieve', 'verify'])
  expect(systemA.canOpenReader).toBe(true)
  expect(systemA.compactMetaItems.map((item) => item.key)).toEqual(['location', 'anchor', 'meta-Published', 'doi'])
  expect(systemA.compactMetaItems.find((item) => item.key === 'location')?.value).toBe('Method')

  expect(systemB.kindLabel).toBe('Upstream citation')
  expect(systemB.badgeLabel).toBe('[R3/9/11]')
  expect(systemB.systemBTitle).toBe('Upstream Work')
  expect(systemB.primaryActionLabel).toBe('Read location')
  expect(systemB.flowSteps).toEqual([])
  expect(systemB.compactMetaItems.map((item) => item.key)).toEqual(['authors', 'published'])
})

test('citation popover status model derives binding, support, and warning state', async ({ page }) => {
  const S = {
    cite_binding_candidate: 'Candidate evidence',
    cite_binding_mismatch: 'Citation mismatch',
    cite_candidate_support_default: 'Candidate support fallback',
    cite_external_metadata_warning: 'External metadata differs',
    cite_external_title: 'External title: {title}',
    cite_system_b_support_default: 'Bibliography link from current paper.',
  }
  const systemA = await citationPopoverStatusModel(page, {
    detail: {
      title: 'Known title',
      sourceName: 'Fixture Paper',
      bindingStatus: 'mismatch',
      bindingReason: 'retrieved evidence does not overlap the claim',
      bindingOverlapTerms: ['illumination', 'DMD'],
      cardQualityLabel: 'Quality',
      cardQualityScore: 0.42,
      cardQualityFlags: ['missing_reference_entry'],
      cardWarning: 'detail warning should lose to section',
      externalMetadataStatus: 'conflict',
      externalMetadataReason: 'External DOI conflicts with local metadata.',
      externalTitle: 'Different external title',
    },
    S,
    supportSection: {
      text: 'Human-readable support explanation.',
    },
    warningSection: {
      text: 'Reference [4] is missing from the converted bibliography.',
    },
    displayMain: 'Known title',
  })
  const systemB = await citationPopoverStatusModel(page, {
    detail: {
      bindingStatus: 'mismatch',
      cardQualityFlags: [],
    },
    S,
    isSystemB: true,
    displayMain: 'Upstream Work',
  })

  expect(systemA.bindingState).toEqual({
    label: 'Citation mismatch',
    tone: 'mismatch',
  })
  expect(systemA.bindingOverlapText).toBe('illumination / DMD')
  expect(systemA.supportText).toBe('Human-readable support explanation.')
  expect(systemA.showBindingReason).toBe(true)
  expect(systemA.showCardWarning).toBe(true)
  expect(systemA.cardWarning).toBe('Reference [4] is missing from the converted bibliography.')
  expect(systemA.showExternalMetadataWarning).toBe(true)
  expect(systemA.externalMetadataWarningText).toBe('External DOI conflicts with local metadata.')
  expect(systemA.externalMetadataTitleHint).toBe('External title: Different external title')

  expect(systemB.bindingState).toBeNull()
  expect(systemB.explicitSupportText).toBe('')
  expect(systemB.supportText).toBe('Bibliography link from current paper.')
})

test('citation popover System B overview state separates loading and unavailable states', async ({ page }) => {
  const S = {
    cite_loading: 'Loading',
    cite_loading_summary: 'Loading summary',
    cite_summary_unavailable: 'Summary unavailable',
  }

  const loading = await systemBOverviewState(page, {
    S,
    loading: true,
  })
  const unavailable = await systemBOverviewState(page, {
    S,
    bibliometricsChecked: true,
    doiLabel: 'doi:10.1000/example',
  })
  const hasReference = await systemBOverviewState(page, {
    S,
    bibliometricsChecked: true,
    showReference: true,
    systemBTitle: 'Upstream Work',
  })
  const hasOverview = await systemBOverviewState(page, {
    S,
    loading: true,
    bibliometricsChecked: true,
    paperOverviewText: 'Article overview is already available.',
    doiLabel: 'doi:10.1000/example',
  })

  expect(loading.showOverviewLoading).toBe(true)
  expect(loading.overviewLoadingLabel).toBe('Loading summary')
  expect(loading.showOverviewUnavailable).toBe(false)

  expect(unavailable.showOverviewLoading).toBe(false)
  expect(unavailable.showOverviewUnavailable).toBe(true)
  expect(unavailable.overviewUnavailableLabel).toBe('Summary unavailable')

  expect(hasReference.showOverviewUnavailable).toBe(false)
  expect(hasOverview.showOverviewLoading).toBe(false)
  expect(hasOverview.showOverviewUnavailable).toBe(false)
})

test('citation popover System B trace model gates trace display and normalizes steps', async ({ page }) => {
  const S = {
    cite_evidence_chain: 'Evidence chain',
    cite_trace_complete: 'Trace complete',
    cite_trace_review: 'Trace needs review',
  }
  const detail = {
    systemBTraceSteps: ['  locate reference  ', '', 'verify upstream work'],
    systemBTraceReason: '  Source chain verified.  ',
    systemBTraceScore: 0.76,
    systemBTraceComplete: true,
  }

  const hidden = await systemBTraceModel(page, {
    detail,
    S,
  })
  const visible = await systemBTraceModel(page, {
    detail,
    S,
    traceEnabled: true,
  })
  const nonSystemB = await systemBTraceModel(page, {
    detail,
    S,
    isSystemB: false,
    traceEnabled: true,
  })

  expect(hidden.showTrace).toBe(false)
  expect(hidden.traceSteps).toEqual(['locate reference', 'verify upstream work'])
  expect(hidden.traceReason).toBe('Source chain verified.')
  expect(hidden.traceScore).toBe(0.76)
  expect(hidden.traceStatus).toEqual({ label: 'Trace complete', tone: 'complete' })
  expect(hidden.traceLabel).toBe('Evidence chain')

  expect(visible.showTrace).toBe(true)
  expect(nonSystemB.showTrace).toBe(false)
  expect(nonSystemB.traceSteps).toEqual([])
  expect(nonSystemB.traceReason).toBe('')
})

test('citation popover System B text panels filter overview and takeaway candidates', async ({ page }) => {
  const referenceText = '[12] Doe, J. Upstream imaging method. IEEE Transactions on Imaging, 2024. doi:10.1000/example.'
  const model = await systemBTextPanelsModel(page, {
    detail: {
      summarySource: 'abstract',
      summaryLine: 'This article introduces a calibrated imaging pipeline for robust microscopy measurements.',
      citationContextSource: 'reader_references',
      cardContextSummary: 'links the answer back to an upstream reference',
      upstreamWorkRole: referenceText,
      cardSupportExplanation: 'Published in IEEE Transactions on Imaging, 2024.',
      supportRelation: 'The current paper cites this work to justify the upstream calibration method.',
      citationContext: 'Reader reference row text should not become the takeaway.',
    },
    S: {
      cite_context: 'Context',
      cite_context_summary: 'Context summary',
      cite_current_paper_usage: 'Current paper usage',
      cite_paper_overview: 'Article overview',
      cite_upstream_role: 'Upstream role',
    },
    contextSummarySection: {
      label: 'Reader usage',
      text: 'doi:10.1000/example, journal metadata, cited by 42.',
    },
    referenceSection: {
      text: referenceText,
    },
    cardTakeaway: referenceText,
    cardEvidenceLabel: 'Evidence focus',
  })

  expect(model.systemBReferenceText).toBe(referenceText)
  expect(model.systemBContextSource).toBe('reader_references')
  expect(model.paperOverviewText).toBe('This article introduces a calibrated imaging pipeline for robust microscopy measurements.')
  expect(model.paperOverviewLabel).toBe('Article overview')
  expect(model.takeawayText).toBe('The current paper cites this work to justify the upstream calibration method.')
  expect(model.takeawayLabel).toBe('Current paper usage')
  expect(model.contextSummaryText).toBe('')
  expect(model.contextSummaryLabel).toBe('Reader usage')
  expect(model.citationContextLabel).toBe('Evidence focus')
})

test('citation popover System B source panels derive clean locations and missing-title references', async ({ page }) => {
  const S = {
    cite_location_current: 'Current paper location',
    cite_note: 'Note',
    cite_original_reference_entry: 'Original reference entry',
    cite_reference_entry: 'Reference entry',
    cite_upstream_reference: 'Upstream reference',
  }
  const referenceText = '[12] Doe, J. Upstream imaging method. IEEE Transactions on Imaging, 2024. doi:10.1000/example.'

  const strong = await systemBSourcePanelsModel(page, {
    detail: {
      sourceName: 'Current Paper',
    },
    S,
    locatorSection: {
      text: 'Current Paper / Related Work / p. 2',
    },
    sourcePaperText: 'Current Paper',
    systemBContextSource: 'abstract',
    systemBReferenceText: referenceText,
    systemBExplicitReferenceText: referenceText,
    systemBTitle: 'Upstream imaging method',
    headerSubtitle: 'IEEE Transactions on Imaging, 2024',
    metrics: ['IF 5.0'],
    explicitSupportText: 'The current paper cites this method as related work.',
  })
  const weak = await systemBSourcePanelsModel(page, {
    detail: {
      sourceName: 'Current Paper',
      shelfOrigin: 'reader_references',
    },
    S,
    cardQualityFlags: ['weak_citation_context'],
    sourcePaperText: 'Current Paper',
    headingPath: 'Related Work',
    pageLabel: 'p. 2',
    badgeLabel: '[R12]',
    systemBContextSource: 'answer_context',
    systemBReferenceText: referenceText,
    systemBExplicitReferenceText: referenceText,
    systemBTitle: 'Upstream reference',
    systemBTitleMissing: true,
  })

  expect(strong.showLocation).toBe(true)
  expect(strong.locationText).toBe('Related Work / p. 2')
  expect(strong.showReference).toBe(true)
  expect(strong.supportText).toBe('The current paper cites this method as related work.')
  expect(strong.showSupport).toBe(false)

  expect(weak.showLocation).toBe(false)
  expect(weak.showReference).toBe(true)
  expect(weak.referenceLabel).toBe('Original reference entry')
  expect(weak.referencePreview).toContain('Upstream imaging method')
})

test('citation popover System B model suppresses weak locations while keeping missing-title references', async ({ page }) => {
  const model = await systemBLiteratureCardModel(page, {
    detail: {
      isInpaper: true,
      sourceName: 'Current Paper',
      citationContextSource: 'answer_context',
      cardReferenceEntry: '[12] Doe, J. Upstream imaging method. IEEE Transactions on Imaging, 2024. doi:10.1000/example.',
      bibliometricsChecked: true,
      cardQualityFlags: [],
    },
    S: {
      cite_context: 'Context',
      cite_context_summary: 'Context summary',
      cite_current_paper_usage: 'Current paper usage',
      cite_evidence_chain: 'Evidence chain',
      cite_loading: 'Loading',
      cite_loading_summary: 'Loading summary',
      cite_location_current: 'Current paper location',
      cite_note: 'Note',
      cite_original_reference_entry: 'Original reference entry',
      cite_paper_overview: 'Article overview',
      cite_reference_entry: 'Reference entry',
      cite_summary_unavailable: 'Summary unavailable',
      cite_system_b_support_default: 'Bibliography link from current paper.',
      cite_trace_complete: 'Trace complete',
      cite_trace_review: 'Trace needs review',
      cite_upstream_reference: 'Upstream reference',
      cite_upstream_role: 'Upstream role',
    },
    sourcePaperText: 'Current Paper',
    headingPath: 'Related Work',
    pageLabel: 'p. 2',
    badgeLabel: '[R12]',
    systemBTitle: 'Upstream reference',
    systemBTitleMissing: true,
  })

  expect(model.showLocation).toBe(false)
  expect(model.showReference).toBe(true)
  expect(model.referenceLabel).toBe('Original reference entry')
  expect(model.referencePreview).toContain('Upstream imaging method')
  expect(model.showOverviewUnavailable).toBe(false)
})

test('assistant message notice view model keeps contract source badge primary', async ({ page }) => {
  const model = await assistantMessageNoticeViewModel(page, {
    message: {
      role: 'assistant',
      notice: 'Note: no matching local knowledge-base evidence was found; this is an external model answer.',
      meta: {
        answer_contract: {
          source_summary: {
            kind: 'local_kb',
          },
          source_policy_payload: {
            kind: 'local_kb',
            uses_local_knowledge_base: true,
            uses_external_model: false,
            requires_user_notice: false,
            badge: {
              label_key: 'agent_trace_source_local_only',
              detail: 'Contract-backed evidence is available.',
            },
          },
        },
        agent_source_summary: {
          kind: 'external_not_kb',
          label: 'Legacy external source',
        },
      },
    },
    lowConfidenceMeta: null,
    S: {
      agent_trace_source_local_only: 'Local KB',
      agent_trace_evidence_not_from_kb: 'Not from KB',
      agent_trace_source_fallback: 'Source',
    },
  })

  expect(model.sourceNoticeViewModel).toMatchObject({
    label: 'Local KB',
    title: 'Contract-backed evidence is available.',
    kind: 'local_kb',
  })
  expect(model.legacySourceNoticeText).toBe('')
  expect(model.plainNoticeText).toBe('')
  expect(model.hasVisibleNotice).toBe(true)
})

test('assistant message notice view model separates fallback source and plain notices', async ({ page }) => {
  const fallbackSource = await assistantMessageNoticeViewModel(page, {
    message: {
      role: 'assistant',
      notice: 'Note: no matching local knowledge-base evidence was found; this is an external model answer.',
      meta: {},
    },
    lowConfidenceMeta: null,
    S: {
      agent_trace_evidence_not_from_kb: 'Not from KB',
      agent_trace_source_fallback: 'Source',
    },
  })
  const plainNotice = await assistantMessageNoticeViewModel(page, {
    message: {
      role: 'assistant',
      notice: 'Regular maintenance notice.',
      meta: {},
    },
    lowConfidenceMeta: null,
    S: {
      agent_trace_evidence_not_from_kb: 'Not from KB',
      agent_trace_source_fallback: 'Source',
    },
  })

  expect(fallbackSource.sourceNoticeViewModel).toMatchObject({
    label: 'Not from KB',
    title: 'Note: no matching local knowledge-base evidence was found; this is an external model answer.',
  })
  expect(fallbackSource.legacySourceNoticeText).toBe('')
  expect(fallbackSource.plainNoticeText).toBe('')
  expect(plainNotice.sourceNoticeViewModel).toBeNull()
  expect(plainNotice.legacySourceNoticeText).toBe('')
  expect(plainNotice.plainNoticeText).toBe('Regular maintenance notice.')
})

test('assistant message notice view model exposes low confidence and provenance flags', async ({ page }) => {
  const active = await assistantMessageNoticeViewModel(page, {
    message: {
      role: 'assistant',
      content: 'Evidence-backed answer.',
      meta: {},
    },
    lowConfidenceMeta: {
      isZh: false,
      reasonCode: 'weak_signal',
      reasonText: 'retrieval signal is weak',
      candidateRefs: [1, 2],
    },
    provenanceModeLabel: 'debug provenance',
    showProvenanceModeLabel: true,
    S: {},
  })
  const empty = await assistantMessageNoticeViewModel(page, {
    message: {
      role: 'assistant',
      content: 'Evidence-backed answer.',
      meta: {},
    },
    lowConfidenceMeta: null,
    provenanceModeLabel: 'debug provenance',
    showProvenanceModeLabel: false,
    S: {},
  })

  expect(active.showLowConfidence).toBe(true)
  expect(active.lowConfidenceMeta).toMatchObject({
    reasonText: 'retrieval signal is weak',
    candidateRefs: [1, 2],
  })
  expect(active.showProvenanceLabel).toBe(true)
  expect(active.provenanceModeLabel).toBe('debug provenance')
  expect(active.hasVisibleNotice).toBe(true)
  expect(empty.hasVisibleNotice).toBe(false)
})

test('agent trace quality gate title combines reasons before warnings', async ({ page }) => {
  const title = await agentQualityGateTitle(page, {
    reasons: ['missing citation repaired', 'claim overlap checked'],
    warnings: ['fallback citation used'],
  })

  expect(title).toBe('missing citation repaired / claim overlap checked / fallback citation used')
})

test('agent trace quality gate title trims and limits noisy lists', async ({ page }) => {
  const title = await agentQualityGateTitle(page, {
    reasons: [
      ' first reason ',
      'second   reason',
      'third reason',
      'fourth reason',
      'fifth reason omitted',
    ],
    warnings: 'not an array',
  })

  expect(title).toBe('first reason / second reason / third reason / fourth reason')
})

test('agent trace scope summary includes requested scope and selected count', async ({ page }) => {
  const summary = await agentScopeSummary(page, {
    queryScope: 'library',
    requestedScope: 'current_paper',
    selectedCount: 2,
    currentSource: 'Current source should not appear outside current_paper scope',
  })

  expect(summary).toBe('library / requested current_paper / 2 selected')
})

test('agent trace scope summary shows current paper source only for current-paper scope', async ({ page }) => {
  const currentPaperSummary = await agentScopeSummary(page, {
    queryScope: 'current_paper',
    requestedScope: 'current_paper',
    selectedCount: 1,
    currentSource: 'Fast hyperspectral single-pixel imaging fixture source',
  })
  const librarySummary = await agentScopeSummary(page, {
    queryScope: 'library',
    requestedScope: 'library',
    selectedCount: 0,
    currentSource: 'Fast hyperspectral single-pixel imaging fixture source',
  })

  expect(currentPaperSummary).toBe('current_paper / 1 selected / Fast hyperspectral single-pixel imaging fixture source')
  expect(librarySummary).toBe('library')
})

test('agent trace header summary keeps evidence status primary and claims as context', async ({ page }) => {
  const summary = await agentHeaderSummary(page, {
    evidenceLabel: 'Evidence grounded',
    totalClaims: 2,
    supportedClaims: 1,
    unsupportedClaims: 1,
    hasErrors: false,
    scopeSummary: 'library / 2 selected',
    taskLabel: 'Single paper',
  })

  expect(summary).toEqual({
    headerEvidence: 'Evidence grounded',
    headerContext: 'Review 1/2',
  })
})

test('agent trace header summary falls back from scope to task without evidence', async ({ page }) => {
  const scopedSummary = await agentHeaderSummary(page, {
    evidenceLabel: '',
    totalClaims: 0,
    supportedClaims: 0,
    unsupportedClaims: 0,
    hasErrors: false,
    scopeSummary: 'library / 2 selected',
    taskLabel: 'Single paper',
  })
  const taskSummary = await agentHeaderSummary(page, {
    evidenceLabel: '',
    totalClaims: 0,
    supportedClaims: 0,
    unsupportedClaims: 0,
    hasErrors: false,
    scopeSummary: '',
    taskLabel: 'Single paper',
  })

  expect(scopedSummary).toEqual({
    headerEvidence: 'Source check available',
    headerContext: 'library / 2 selected',
  })
  expect(taskSummary).toEqual({
    headerEvidence: 'Source check available',
    headerContext: 'Single paper',
  })
})

test('agent trace summary chip builder preserves compact chip order and test ids', async ({ page }) => {
  const chips = await visibleSummaryChips(page, agentSummaryViewModel())

  expect(chips.map((chip) => chip.id)).toEqual([
    'evidence',
    'claims',
    'unsupported-claims',
    'quality-gate',
    'task',
    'scope',
    'run-errors',
    'research-run',
    'source-policy',
  ])
  expect(chips.find((chip) => chip.id === 'evidence')).toMatchObject({
    className: 'kb-agent-trace-evidence-status is-grounded',
    testId: 'agent-trace-evidence-status',
    value: 'Evidence grounded',
  })
  expect(chips.find((chip) => chip.id === 'quality-gate')).toMatchObject({
    className: 'is-warning',
    testId: 'agent-trace-quality-gate',
    title: 'citation repair applied',
    value: 'Repaired',
  })
  expect(chips.find((chip) => chip.id === 'research-run')?.value).toBe('done / 3 rows')
  expect(chips.find((chip) => chip.id === 'source-policy')?.value).toBe('Local KB')
})

test('agent trace summary chip builder hides empty optional chips but keeps task', async ({ page }) => {
  const chips = await visibleSummaryChips(page, agentSummaryViewModel({
    evidenceLabel: '',
    evidenceStatus: '',
    totalClaims: 0,
    supportedClaims: 0,
    unsupportedClaims: 0,
    qualityGateStatus: '',
    qualityGateTitle: '',
    scopeSummary: '',
    hasErrors: false,
    researchRunStatus: '',
    evidenceMatrixRows: 0,
    sourcePolicy: '',
  }))

  expect(chips.map((chip) => chip.id)).toEqual(['task'])
  expect(chips[0]).toMatchObject({
    label: 'Task',
    value: 'Single paper',
  })
})

test('research agent trace references can open and enter the literature basket', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-reference-actions')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-reference-actions')
  const traceSummary = page.locator('.kb-agent-trace > summary').first()
  await expect(traceSummary).toContainText(SOURCE_PANEL_RE)
  await expect(traceSummary).toContainText(NEEDS_REVIEW_RE)
  await expect(traceSummary).toContainText(REVIEW_FRACTION_RE)
  await expect(traceSummary).not.toContainText('reference_followup')
  await expect(traceSummary).not.toContainText('done')
  await expect(traceSummary).not.toContainText(ANSWER_QUALITY_RE)
  await expect(page.getByText('Research Agent Trace')).toHaveCount(0)
  await expect(page.getByText('Tool Calls')).toHaveCount(0)
  await page.getByText(SOURCE_PANEL_RE).click()
  await expect(page.getByTestId('agent-trace-evidence-status')).toContainText(NEEDS_REVIEW_RE)
  await expect(page.getByTestId('agent-trace-quality-gate')).toContainText(ANSWER_QUALITY_RE)
  await expect(page.getByTestId('agent-trace-quality-gate')).toContainText(REPAIRED_RE)
  await expect(page.getByTestId('agent-evidence-matrix')).toContainText(EVIDENCE_MAP_RE)
  await expect(page.getByTestId('agent-evidence-matrix-row').first()).toContainText('Fast hyperspectral single-pixel imaging')
  await expect(page.getByTestId('agent-evidence-matrix-row').first()).toContainText('frequency-division multiplexed illumination')
  await expect(page.getByTestId('agent-trace-unsupported-claim')).toContainText('fully solves every downstream limitation')
  await expect(page.getByTestId('agent-trace-unsupported-claim')).toContainText(MISMATCH_RE)
  await expect(page.getByText('Resolved 1 upstream reference from 1 citing source paper.')).toBeHidden()
  await page.getByText(DIAGNOSTICS_RE).click()
  await expect(page.getByText('Resolved 1 upstream reference from 1 citing source paper.')).toBeVisible()

  const ref = page.getByTestId('agent-trace-reference').first()
  await expect(ref.getByTestId('agent-trace-ref-title')).toContainText('Fast hyperspectral single-pixel imaging')

  await ref.getByTestId('agent-trace-ref-open').click()
  const payload = page.getByTestId('message-list-open-payload')
  await expect(payload).toContainText(READER_REGRESSION_SOURCE_PATH)
  await expect(payload).toContainText('Fixture Paper / Related Work')
  await expect(payload).toContainText('frequency-division multiplexed illumination')

  await ref.getByTestId('agent-trace-ref-add').click()
  await expect(page.getByTestId('citation-shelf')).toHaveClass(/translate-x-0/)
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await expect(page.getByTestId('citation-shelf-item-title')).toContainText('Fast hyperspectral single-pixel imaging')
})

test('research agent trace can be loaded from stored audit endpoint on demand', async ({ page }) => {
  let requested = false
  await page.route('**/api/messages/9301/agent-trace**', async (route) => {
    requested = true
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        message_id: 9301,
        conv_id: 'message-list-regression:agent-trace-lazy-audit',
        available: true,
        agent_trace: {
          mode: 'research_agent',
          question_type: 'single_paper_qa',
          status: 'done',
          context: { query_scope: 'raw-trace-context' },
          plan: [
            { goal: 'Retrieve compact evidence.', tool: 'retrieve_evidence', status: 'done' },
          ],
          steps: [
            {
              tool: 'retrieve_evidence',
              status: 'done',
              observation: 'Stored audit trace was loaded on demand.',
              output: {},
            },
          ],
          verification: {
            total_claims: 9,
            supported_claims: 3,
            unsupported_claims: 0,
            evidence_status: 'grounded',
            evidence_hit_count: 3,
            evidence_status_reasons: [],
            claims: [],
          },
          errors: [],
        },
        summary: {
          available: true,
          question_type: 'single_paper_qa',
          total_claims: 1,
          supported_claims: 1,
          unsupported_claims: 0,
          evidence_status: 'grounded',
          evidence_hit_count: 3,
          evidence_status_reasons: [],
          tool_call_count: 1,
          has_errors: false,
          query_scope: 'library',
        },
      }),
    })
  })

  await page.goto('/__message_list_test__?scenario=agent-trace-lazy-audit')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-lazy-audit')
  await expect(page.getByText('Stored audit trace was loaded on demand.')).toBeHidden()
  await page.getByText(SOURCE_PANEL_RE).click()
  await expect(page.getByText('Stored audit trace was loaded on demand.')).toBeHidden()
  await expect(page.locator('.kb-agent-trace > summary').first()).toContainText(GROUNDED_RE)
  await expect(page.locator('.kb-agent-trace > summary').first()).toContainText(CHECKED_FRACTION_RE)
  await expect(page.getByTestId('agent-trace-evidence-status')).toContainText(GROUNDED_RE)
  await expect(page.locator('.kb-agent-trace-summary strong', { hasText: '1/1' })).toBeVisible()
  await expect(page.locator('.kb-agent-trace-summary strong', { hasText: 'library' })).toBeVisible()
  await expect(page.getByText('3/9')).toHaveCount(0)
  await expect(page.getByText('Research Agent Trace')).toHaveCount(0)
  await expect(page.getByText('Tool Calls')).toHaveCount(0)
  await page.getByText(DIAGNOSTICS_RE).click()
  await expect(page.getByText('Stored audit trace was loaded on demand.')).toBeVisible()
  expect(requested).toBe(true)
})

test('research agent debug sections stay out of the answer body', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-clean-answer')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-clean-answer')
  await expect(page.getByText('The answer should stay focused on the evidence-backed conclusion.')).toBeVisible()
  await expect(page.getByText('retrieve_evidence debug detail leaked')).toHaveCount(0)
  await expect(page.getByText('verify_answer_citations debug detail leaked')).toHaveCount(0)
  await expect(page.getByText('supported_claims: 1')).toHaveCount(0)
})

test('agent source summary shows a compact local source badge', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-source-summary-local')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-source-summary-local')
  await expect(page.getByText('The local paper reports that retrieval happens before generation')).toBeVisible()
  await expect(page.getByTestId('assistant-source-notice')).toContainText(LOCAL_KB_RE)
  await expect(page.getByText('agent_trace')).toHaveCount(0)
  await expect(page.getByText('tool calls')).toHaveCount(0)
})

test('answer contract source badge wins over legacy source summary', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-answer-contract-source-precedence')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-answer-contract-source-precedence')
  await expect(page.getByText('The contract-backed answer keeps the visible response focused on the paper evidence')).toBeVisible()
  await expect(page.getByTestId('assistant-source-notice')).toContainText(LOCAL_KB_RE)
  await expect(page.getByTestId('assistant-source-notice')).not.toContainText(NOT_FROM_KB_RE)
  await expect(page.getByText('Legacy external source')).toHaveCount(0)
  await expect(page.getByText('runtime_check')).toHaveCount(0)
  await expect(page.getByText('needs_review_count')).toHaveCount(0)
  await page.getByTestId('assistant-source-notice').click()
  await expect(page.getByTestId('evidence-drawer')).toBeVisible()
  await expect(page.getByTestId('evidence-source-summary')).toContainText(LOCAL_KB_RE)
  await expect(page.getByTestId('evidence-drawer-item')).toContainText('The answer is supported by the local fixture paper')
  await expect(page.getByTestId('evidence-drawer-item')).toContainText('frequency-division multiplexed illumination')
  await expect(page.getByTestId('evidence-drawer-item')).toContainText('Fixture Paper / Method')
  await expect(page.getByTestId('evidence-drawer')).not.toContainText('runtime_check')
  await expect(page.getByTestId('evidence-drawer')).not.toContainText('needs_review_count')
  await page.getByTestId('evidence-open-source').click()
  await expect(page.getByTestId('message-list-open-payload')).toContainText(READER_REGRESSION_SOURCE_PATH)
  await page.locator('.kb-evidence-drawer .ant-drawer-close').click()
  await page.locator('.kb-cite-chip').first().click()
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('frequency-division multiplexed illumination')
  await expect(page.locator('.kb-cite-pop')).toContainText('Fixture Paper')
  await expect(page.locator('.kb-cite-pop')).toContainText('Method')
})

test('external source notice is compacted outside the answer body', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-external-notice')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-external-notice')
  await expect(page.getByText('External context can still explain the concept clearly')).toBeVisible()
  await expect(page.getByTestId('assistant-source-notice')).toContainText(NOT_FROM_KB_RE)
  await expect(page.getByText('no matching local knowledge-base evidence was found')).toHaveCount(0)
  await expect(page.locator('.kb-markdown-chat')).not.toContainText('knowledge-base-grounded answer')
  await expect(page.locator('.kb-agent-trace > summary').first()).toContainText(NOT_FROM_KB_RE)
})

test('hybrid source notice is compacted outside the answer body', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-hybrid-notice')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-hybrid-notice')
  await expect(page.getByText('Local evidence: the paper uses retrieval before generation')).toBeVisible()
  await expect(page.getByText('External context: RAG often uses retrieved context')).toBeVisible()
  await expect(page.getByTestId('assistant-source-notice')).toContainText(LOCAL_EXTERNAL_RE)
  await expect(page.getByText('local citations [n] come from the knowledge base')).toHaveCount(0)
  await expect(page.locator('.kb-markdown-chat')).not.toContainText('uncited background may use external model context')
  await expect(page.locator('.kb-agent-trace > summary').first()).toContainText(NEEDS_REVIEW_RE)
})

test('streaming research agent partial hides appended trace json', async ({ page }) => {
  await page.goto('/__message_list_test__?scenario=agent-trace-streaming-clean')

  await expect(page.getByTestId('message-list-test-scenario')).toContainText('agent-trace-streaming-clean')
  await expect(page.getByText('Streaming answer stays focused on the conclusion.')).toBeVisible()
  await expect(page.getByTestId('assistant-source-notice')).toContainText(NOT_FROM_KB_RE)
  await expect(page.getByText('no matching local knowledge-base evidence was found')).toHaveCount(0)
  await expect(page.getByText('stream trace leaked')).toHaveCount(0)
  await expect(page.getByText('agent_trace')).toHaveCount(0)
})
