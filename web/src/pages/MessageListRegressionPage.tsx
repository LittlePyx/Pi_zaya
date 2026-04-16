import { useState } from 'react'
import { MessageList } from '../components/chat/MessageList'
import type { Message } from '../api/chat'
import type { ReaderOpenPayload } from '../components/chat/reader/readerTypes'
import {
  READER_REGRESSION_SOURCE_NAME,
  READER_REGRESSION_SOURCE_PATH,
} from '../testing/readerRegressionFixtures'

const EQUATION_TEXT = '$$\nC(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt\n$$'

const structuredPrimaryRerankMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'Equation (1) direct evidence.',
    rendered_body: 'Equation (1) direct evidence.',
    copy_text: 'Equation (1) direct evidence.',
    copy_markdown: 'Equation (1) direct evidence.',
    created_at: Date.now(),
    provenance: {
      source_path: READER_REGRESSION_SOURCE_PATH,
      source_name: READER_REGRESSION_SOURCE_NAME,
      strict_identity_ready: true,
      mapping_mode: 'fast',
      block_map: {
        'fig-1': {
          block_id: 'fig-1',
          anchor_id: 'a-fig-1',
          kind: 'figure',
          heading_path: 'Fixture Paper / 2. Method / Figure 1',
          text: 'Figure 1. SCI system pipeline.',
          number: 1,
        },
        'eq-1': {
          block_id: 'eq-1',
          anchor_id: 'a-eq-1',
          kind: 'equation',
          heading_path: 'Fixture Paper / 2. Method',
          text: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
          number: 1,
        },
      },
      segments: [
        {
          segment_id: 'seg-eq-primary-rerank',
          segment_index: 0,
          kind: 'equation',
          segment_type: 'equation',
          claim_type: 'formula_claim',
          must_locate: true,
          locate_policy: 'required',
          locate_surface_policy: 'primary',
          text: 'Equation (1) direct evidence.',
          raw_markdown: EQUATION_TEXT,
          display_markdown: EQUATION_TEXT,
          snippet_key: 'equation 1 direct evidence',
          evidence_mode: 'direct',
          evidence_block_ids: ['fig-1', 'eq-1'],
          primary_block_id: 'fig-1',
          primary_anchor_id: 'a-fig-1',
          primary_heading_path: 'Fixture Paper / 2. Method / Figure 1',
          evidence_quote: EQUATION_TEXT,
          evidence_confidence: 0.99,
          anchor_kind: 'equation',
          anchor_text: EQUATION_TEXT,
          equation_number: 1,
        },
      ],
    },
  },
]

const requiredFallbackAnchorMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'Single-pixel imaging needs coordinated modulation and reconstruction.',
    rendered_body: 'Single-pixel imaging needs coordinated modulation and reconstruction.',
    copy_text: 'Single-pixel imaging needs coordinated modulation and reconstruction.',
    copy_markdown: 'Single-pixel imaging needs coordinated modulation and reconstruction.',
    created_at: Date.now(),
    provenance: {
      source_path: READER_REGRESSION_SOURCE_PATH,
      source_name: READER_REGRESSION_SOURCE_NAME,
      strict_identity_ready: true,
      mapping_mode: 'fast',
      block_map: {
        'p-1': {
          block_id: 'p-1',
          anchor_id: 'a-p-1',
          kind: 'paragraph',
          heading_path: 'Fixture Paper / 1. Intro',
          text: 'Single-pixel imaging combines known modulation patterns with one detector and reconstruction.',
        },
        'p-2': {
          block_id: 'p-2',
          anchor_id: 'a-p-2',
          kind: 'paragraph',
          heading_path: 'Fixture Paper / 2. Method',
          text: 'Measurement reconstruction alternates between coded sensing and inverse recovery.',
        },
      },
      segments: [
        {
          segment_id: 'seg-required-fallback',
          segment_index: 0,
          kind: 'paragraph',
          segment_type: 'paragraph',
          claim_type: 'shell_sentence',
          must_locate: true,
          locate_policy: 'required',
          locate_surface_policy: 'primary',
          text: 'Single-pixel imaging combines known modulation patterns with one detector and reconstruction.',
          snippet_key: 'single pixel imaging modulation reconstruction',
          evidence_mode: 'direct',
          evidence_block_ids: ['p-1'],
          primary_block_id: '',
          primary_anchor_id: '',
          primary_heading_path: '',
          evidence_confidence: 0.9,
          anchor_kind: '',
          anchor_text: '',
          reader_open: {
            sourcePath: READER_REGRESSION_SOURCE_PATH,
            sourceName: READER_REGRESSION_SOURCE_NAME,
            headingPath: 'Fixture Paper / 1. Intro',
            snippet: 'Single-pixel imaging combines known modulation patterns with one detector and reconstruction.',
            highlightSnippet: 'Single-pixel imaging combines known modulation patterns with one detector and reconstruction.',
            blockId: 'p-1',
            anchorId: 'a-p-1',
            anchorKind: 'sentence',
            strictLocate: true,
            locateTarget: {
              segmentId: 'seg-required-fallback',
              sourceSegmentId: 'seg-required-fallback',
              headingPath: 'Fixture Paper / 1. Intro',
              snippet: 'Single-pixel imaging combines known modulation patterns with one detector and reconstruction.',
              highlightSnippet: 'Single-pixel imaging combines known modulation patterns with one detector and reconstruction.',
              blockId: 'p-1',
              anchorId: 'a-p-1',
              anchorKind: 'sentence',
              hitLevel: 'exact',
              claimType: 'shell_sentence',
              locatePolicy: 'required',
              locateSurfacePolicy: 'primary',
            },
            alternatives: [
              {
                headingPath: 'Fixture Paper / 2. Method',
                snippet: 'Measurement reconstruction alternates between coded sensing and inverse recovery.',
                highlightSnippet: 'Measurement reconstruction alternates between coded sensing and inverse recovery.',
                blockId: 'p-2',
                anchorId: 'a-p-2',
                anchorKind: 'paragraph',
              },
            ],
            visibleAlternatives: [
              {
                headingPath: 'Fixture Paper / 2. Method',
                snippet: 'Measurement reconstruction alternates between coded sensing and inverse recovery.',
                highlightSnippet: 'Measurement reconstruction alternates between coded sensing and inverse recovery.',
                blockId: 'p-2',
                anchorId: 'a-p-2',
                anchorKind: 'paragraph',
              },
            ],
            evidenceAlternatives: [
              {
                headingPath: 'Fixture Paper / 2. Method',
                snippet: 'Measurement reconstruction alternates between coded sensing and inverse recovery.',
                highlightSnippet: 'Measurement reconstruction alternates between coded sensing and inverse recovery.',
                blockId: 'p-2',
                anchorId: 'a-p-2',
                anchorKind: 'paragraph',
              },
            ],
          },
        },
      ],
    },
  },
]

const guideFigureRemapMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'Figure 1 shows the SCI system pipeline.',
    rendered_body: 'Figure 1 shows the SCI system pipeline.',
    copy_text: 'Figure 1 shows the SCI system pipeline.',
    copy_markdown: 'Figure 1 shows the SCI system pipeline.',
    created_at: Date.now(),
    provenance: {
      source_path: READER_REGRESSION_SOURCE_PATH,
      source_name: READER_REGRESSION_SOURCE_NAME,
      strict_identity_ready: true,
      mapping_mode: 'fast',
      block_map: {
        'p-figure-ref': {
          block_id: 'p-figure-ref',
          anchor_id: 'a-p-figure-ref',
          kind: 'paragraph',
          heading_path: 'Fixture Paper / 2. Method',
          text: 'Figure 1 shows the SCI system pipeline.',
        },
      },
      segments: [
        {
          segment_id: 'seg-guide-figure-remap',
          segment_index: 0,
          kind: 'paragraph',
          segment_type: 'paragraph',
          claim_type: 'figure_panel',
          must_locate: true,
          locate_policy: 'required',
          locate_surface_policy: 'primary',
          text: 'Figure 1 shows the SCI system pipeline.',
          snippet_key: 'figure 1 sci system pipeline',
          evidence_mode: 'direct',
          evidence_block_ids: ['p-figure-ref'],
          primary_block_id: 'p-figure-ref',
          primary_anchor_id: 'a-p-figure-ref',
          primary_heading_path: 'Fixture Paper / 2. Method',
          evidence_quote: 'Figure 1 shows the SCI system pipeline.',
          evidence_confidence: 0.92,
          anchor_kind: 'figure',
          anchor_text: 'Figure 1',
          support_slot_figure_number: 1,
        },
      ],
    },
  },
]

const guideFormulaRemapMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'Equation (1) gives the volume rendering integral.',
    rendered_body: 'Equation (1) gives the volume rendering integral.',
    copy_text: 'Equation (1) gives the volume rendering integral.',
    copy_markdown: 'Equation (1) gives the volume rendering integral.',
    created_at: Date.now(),
    provenance: {
      source_path: READER_REGRESSION_SOURCE_PATH,
      source_name: READER_REGRESSION_SOURCE_NAME,
      strict_identity_ready: true,
      mapping_mode: 'fast',
      block_map: {
        'p-method-1': {
          block_id: 'p-method-1',
          anchor_id: 'a-p-method-1',
          kind: 'paragraph',
          heading_path: 'Fixture Paper / 2. Method',
          text: 'Given a set of input multi-view images, NeRF transfers the pixels of the input images into rays.',
        },
      },
      segments: [
        {
          segment_id: 'seg-guide-formula-remap',
          segment_index: 0,
          kind: 'paragraph',
          segment_type: 'paragraph',
          claim_type: 'formula_claim',
          must_locate: true,
          locate_policy: 'required',
          locate_surface_policy: 'primary',
          text: 'Equation (1) gives the volume rendering integral.',
          raw_markdown: EQUATION_TEXT,
          display_markdown: EQUATION_TEXT,
          snippet_key: 'equation 1 volume rendering integral',
          evidence_mode: 'direct',
          evidence_block_ids: ['p-method-1'],
          primary_block_id: 'p-method-1',
          primary_anchor_id: 'a-p-method-1',
          primary_heading_path: 'Fixture Paper / 2. Method',
          evidence_quote: EQUATION_TEXT,
          evidence_confidence: 0.93,
          anchor_kind: 'equation',
          anchor_text: EQUATION_TEXT,
          equation_number: 1,
        },
      ],
    },
  },
]

const renderPacketContractMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'Equation (1) gives the volume rendering integral. [[CITE:s1234abcd:1]]',
    created_at: Date.now(),
    meta: {
      paper_guide_contracts: {
        version: 1,
        intent: { family: 'equation_lookup' },
        render_packet: {
          answer_markdown: 'Equation (1) gives the volume rendering integral. [[CITE:s1234abcd:1]]',
          notice: 'RenderPacket notice: this message should show notice without top-level fields.',
          rendered_body: 'Equation (1) gives the volume rendering integral. [1](#kb-cite-demo-1)',
          rendered_content: 'Equation (1) gives the volume rendering integral. [1](#kb-cite-demo-1)',
          copy_text: 'Equation (1) gives the volume rendering integral. [1]',
          copy_markdown: 'Equation (1) gives the volume rendering integral. [1](#kb-cite-demo-1)',
          cite_details: [
            {
              num: 1,
              anchor: 'kb-cite-demo-1',
              source_name: READER_REGRESSION_SOURCE_NAME,
              source_path: READER_REGRESSION_SOURCE_PATH,
              raw: 'Demo reference [1]',
            },
          ],
          locate_target: {
            segmentId: 'render-packet-seg-1',
            sourceSegmentId: 'render-packet-seg-1',
            headingPath: 'Fixture Paper / 2. Method',
            snippet: 'Equation (1) gives the volume rendering integral.',
            highlightSnippet: 'Equation (1) gives the volume rendering integral.',
            evidenceQuote: EQUATION_TEXT,
            anchorText: EQUATION_TEXT,
            blockId: 'eq-1',
            anchorId: 'a-eq-1',
            anchorKind: 'equation',
            anchorNumber: 1,
            claimType: 'formula_claim',
            locatePolicy: 'required',
            locateSurfacePolicy: 'primary',
          },
          reader_open: {
            sourcePath: READER_REGRESSION_SOURCE_PATH,
            sourceName: READER_REGRESSION_SOURCE_NAME,
            headingPath: 'Fixture Paper / 2. Method',
            snippet: 'Equation (1) gives the volume rendering integral.',
            highlightSnippet: 'Equation (1) gives the volume rendering integral.',
            blockId: 'eq-1',
            anchorId: 'a-eq-1',
            anchorKind: 'equation',
            anchorNumber: 1,
            strictLocate: true,
            locateTarget: {
              segmentId: 'render-packet-seg-1',
              sourceSegmentId: 'render-packet-seg-1',
              headingPath: 'Fixture Paper / 2. Method',
              snippet: 'Equation (1) gives the volume rendering integral.',
              highlightSnippet: 'Equation (1) gives the volume rendering integral.',
              evidenceQuote: EQUATION_TEXT,
              anchorText: EQUATION_TEXT,
              blockId: 'eq-1',
              anchorId: 'a-eq-1',
              anchorKind: 'equation',
              anchorNumber: 1,
              claimType: 'formula_claim',
              locatePolicy: 'required',
              locateSurfacePolicy: 'primary',
            },
          },
        },
      },
    },
  },
]

const renderPacketHiddenLocateMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'This answer should not expose a hidden locate target.',
    created_at: Date.now(),
    meta: {
      paper_guide_contracts: {
        version: 1,
        intent: { family: 'overview' },
        render_packet: {
          answer_markdown: 'This answer should not expose a hidden locate target.',
          rendered_body: 'This answer should not expose a hidden locate target.',
          rendered_content: 'This answer should not expose a hidden locate target.',
          copy_text: 'This answer should not expose a hidden locate target.',
          copy_markdown: 'This answer should not expose a hidden locate target.',
          locate_target: {
            segmentId: 'render-packet-hidden-seg',
            sourceSegmentId: 'render-packet-hidden-seg',
            headingPath: 'Fixture Paper / 2. Method',
            snippet: 'Hidden internal locate.',
            highlightSnippet: 'Hidden internal locate.',
            blockId: 'p-hidden-1',
            anchorId: 'a-p-hidden-1',
            anchorKind: 'paragraph',
            claimType: 'shell_sentence',
            locatePolicy: 'hidden',
            locateSurfacePolicy: 'hidden',
          },
          reader_open: {
            sourcePath: READER_REGRESSION_SOURCE_PATH,
            sourceName: READER_REGRESSION_SOURCE_NAME,
            headingPath: 'Fixture Paper / 2. Method',
            snippet: 'Hidden internal locate.',
            highlightSnippet: 'Hidden internal locate.',
            blockId: 'p-hidden-1',
            anchorId: 'a-p-hidden-1',
            anchorKind: 'paragraph',
            strictLocate: true,
            locateTarget: {
              segmentId: 'render-packet-hidden-seg',
              sourceSegmentId: 'render-packet-hidden-seg',
              headingPath: 'Fixture Paper / 2. Method',
              snippet: 'Hidden internal locate.',
              highlightSnippet: 'Hidden internal locate.',
              blockId: 'p-hidden-1',
              anchorId: 'a-p-hidden-1',
              anchorKind: 'paragraph',
              locatePolicy: 'hidden',
              locateSurfacePolicy: 'hidden',
            },
          },
        },
      },
    },
  },
]

const guideFilterOnlyMessages: Message[] = [
  {
    id: 1,
    role: 'user',
    content: 'Besides this paper, what other papers are relevant?',
    created_at: Date.now(),
  },
  {
    id: 2,
    role: 'assistant',
    content: 'No external paper matched strongly enough for this turn.',
    rendered_body: 'No external paper matched strongly enough for this turn.',
    copy_text: 'No external paper matched strongly enough for this turn.',
    copy_markdown: 'No external paper matched strongly enough for this turn.',
    created_at: Date.now(),
  },
]

const guideFilterOnlyRefs: Record<string, unknown> = {
  1: {
    hits: [],
    guide_filter: {
      active: true,
      hidden_self_source: true,
      filtered_hit_count: 1,
      guide_source_name: READER_REGRESSION_SOURCE_NAME,
    },
  },
}

const negativeEvidenceLocateMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'The paper does not mention ADMM in the retrieved context.',
    rendered_body: 'The paper does not mention ADMM in the retrieved context.',
    copy_text: 'The paper does not mention ADMM in the retrieved context.',
    copy_markdown: 'The paper does not mention ADMM in the retrieved context.',
    created_at: Date.now(),
    provenance: {
      source_path: READER_REGRESSION_SOURCE_PATH,
      source_name: READER_REGRESSION_SOURCE_NAME,
      strict_identity_ready: true,
      mapping_mode: 'fast',
      block_map: {
        'p-neg-1': {
          block_id: 'p-neg-1',
          anchor_id: 'a-p-neg-1',
          kind: 'paragraph',
          heading_path: 'Fixture Paper / 3. Discussion',
          text: 'The paper does not mention ADMM in the retrieved context.',
        },
      },
      segments: [
        {
          segment_id: 'seg-negative-note',
          segment_index: 0,
          kind: 'paragraph',
          segment_type: 'paragraph',
          claim_type: 'evidence_note_claim',
          must_locate: true,
          locate_policy: 'required',
          locate_surface_policy: 'primary',
          text: 'The paper does not mention ADMM in the retrieved context.',
          snippet_key: 'paper does not mention admm in retrieved context',
          evidence_mode: 'direct',
          evidence_block_ids: ['p-neg-1'],
          primary_block_id: 'p-neg-1',
          primary_anchor_id: 'a-p-neg-1',
          primary_heading_path: 'Fixture Paper / 3. Discussion',
          evidence_quote: 'The paper does not mention ADMM in the retrieved context.',
          evidence_confidence: 0.97,
          anchor_kind: 'sentence',
          anchor_text: 'The paper does not mention ADMM in the retrieved context.',
        },
      ],
    },
  },
]

type RegressionScenario =
  | 'structured-primary-rerank'
  | 'required-fallback-anchor'
  | 'guide-figure-remap'
  | 'guide-formula-remap'
  | 'render-packet-contract'
  | 'render-packet-hidden-locate'
  | 'guide-filter-empty-external'
  | 'negative-evidence-locate'

export default function MessageListRegressionPage() {
  const scenarioParam = (() => {
    if (typeof window === 'undefined') return ''
    return new URLSearchParams(window.location.search).get('scenario') || ''
  })().trim().toLowerCase()
  const scenario: RegressionScenario = (() => {
    if (scenarioParam === 'required-fallback-anchor') return 'required-fallback-anchor'
    if (scenarioParam === 'guide-figure-remap') return 'guide-figure-remap'
    if (scenarioParam === 'guide-formula-remap') return 'guide-formula-remap'
    if (scenarioParam === 'render-packet-contract') return 'render-packet-contract'
    if (scenarioParam === 'render-packet-hidden-locate') return 'render-packet-hidden-locate'
    if (scenarioParam === 'guide-filter-empty-external') return 'guide-filter-empty-external'
    if (scenarioParam === 'negative-evidence-locate') return 'negative-evidence-locate'
    return 'structured-primary-rerank'
  })()
  const regressionMessages: Message[] = (() => {
    if (scenario === 'required-fallback-anchor') return requiredFallbackAnchorMessages
    if (scenario === 'guide-figure-remap') return guideFigureRemapMessages
    if (scenario === 'guide-formula-remap') return guideFormulaRemapMessages
    if (scenario === 'render-packet-contract') return renderPacketContractMessages
    if (scenario === 'render-packet-hidden-locate') return renderPacketHiddenLocateMessages
    if (scenario === 'guide-filter-empty-external') return guideFilterOnlyMessages
    if (scenario === 'negative-evidence-locate') return negativeEvidenceLocateMessages
    return structuredPrimaryRerankMessages
  })()
  const regressionRefs: Record<string, unknown> = scenario === 'guide-filter-empty-external' ? guideFilterOnlyRefs : {}
  const [payload, setPayload] = useState<ReaderOpenPayload | null>(null)

  return (
    <div className="min-h-screen bg-[var(--bg)] px-6 py-6">
      <div className="mx-auto max-w-5xl space-y-4">
        <div>
          <div className="text-sm font-medium text-black/80 dark:text-white/80">
            MessageList locate regression harness
          </div>
          <div className="text-xs text-black/45 dark:text-white/45" data-testid="message-list-test-scenario">
            {scenario}
          </div>
        </div>

        <div className="rounded-3xl border border-[var(--border)] bg-[var(--panel)] p-4">
          <MessageList
            messages={regressionMessages}
            refs={regressionRefs}
            onOpenReader={(nextPayload) => setPayload(nextPayload)}
            paperGuideSourcePath={READER_REGRESSION_SOURCE_PATH}
            paperGuideSourceName={READER_REGRESSION_SOURCE_NAME}
          />
        </div>

        <div className="rounded-3xl border border-[var(--border)] bg-[var(--panel)] p-4">
          <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-black/40 dark:text-white/40">
            Last open payload
          </div>
          <pre
            className="min-h-24 whitespace-pre-wrap rounded-2xl border border-[var(--border)] bg-white/60 px-3 py-3 text-xs text-black/70 dark:bg-black/20 dark:text-white/70"
            data-testid="message-list-open-payload"
          >
            {payload ? JSON.stringify(payload, null, 2) : '(empty)'}
          </pre>
        </div>
      </div>
    </div>
  )
}
