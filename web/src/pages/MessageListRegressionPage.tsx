import { useCallback, useState } from 'react'
import { MessageList } from '../components/chat/MessageList'
import { PaperGuideReaderDrawer } from '../components/chat/PaperGuideReaderDrawer'
import type { Message } from '../api/chat'
import type { ReaderLocateResult, ReaderOpenPayload } from '../components/chat/reader/readerTypes'
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
          rendered_body: 'Equation (1) gives the volume rendering integral. [1]',
          rendered_content: 'Equation (1) gives the volume rendering integral. [1]',
          copy_text: 'Equation (1) gives the volume rendering integral. [1]',
          copy_markdown: 'Equation (1) gives the volume rendering integral. [1]',
          cite_details: [
            {
              num: 1,
              anchor: 'kb-cite-demo-1',
              is_inpaper: true,
              source_name: READER_REGRESSION_SOURCE_NAME,
              source_path: READER_REGRESSION_SOURCE_PATH,
              raw: '[1] Gehm M, Brady D. Single-shot compressive spectral imaging with a dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013',
              card_reference_entry: '[1] Gehm M, Brady D. Single-shot compressive spectral imaging with a dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013',
              title: 'Single-shot compressive spectral imaging with a dual-disperser architecture',
              authors: 'Gehm M, Brady D',
              venue: 'Optics Express',
              year: '2007',
              doi: '10.1364/OE.15.014013',
              doi_url: 'https://doi.org/10.1364/OE.15.014013',
              citation_count: 123,
              citation_source: 'OpenAlex',
              journal_if: '3.8',
              journal_quartile: 'Q2',
              answer_claim: 'Equation (1) uses this reference as the upstream source for single-shot compressive spectral imaging.',
              citation_context: 'The current paper cites this work when tracing the single-shot compressive spectral imaging background.',
              citation_context_source: 'answer_context',
              upstream_work_role: 'Cited prior work or background source used to trace the upstream origin of the answer.',
              user_question_relation: 'The user is asking about the evidence behind the answer; this reference is the upstream paper to open next.',
              evidence_quote: 'The answer points to this paper as the upstream source for single-shot compressive spectral imaging.',
              evidence_source: 'answer_context',
              summary_line: 'The answer points to this paper as the upstream source for single-shot compressive spectral imaging.',
              summary_source: 'answer_context',
              why_line: 'This reference is the cited prior work to open when tracing the concept behind the answer.',
              card_quality_flags: ['reference_entry_only'],
              system_b_trace_complete: false,
              system_b_trace_score: 0.42,
              system_b_trace_reason: '目前只拿到了答案句里的引用线索，还没有定位到当前论文正文中的引用语境。',
              system_b_trace_flags: ['answer_context_only', 'reference_entry_only'],
              system_b_trace_steps: ['答案句', '引用语境待核对', '上游文献'],
              system_b_trace_answer: 'Equation (1) uses this reference as the upstream source for single-shot compressive spectral imaging.',
              system_b_trace_context: 'The current paper cites this work when tracing the single-shot compressive spectral imaging background.',
              system_b_trace_reference: '[1] Gehm M, Brady D. Single-shot compressive spectral imaging with a dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013',
              system_b_trace_locator: READER_REGRESSION_SOURCE_NAME,
              system_b_trace_source: 'answer_context',
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

const citationHoverRaceMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'Compare two cited papers: [1](#kb-cite-race-a) and [2](#kb-cite-race-b).',
    created_at: Date.now(),
    meta: {
      paper_guide_contracts: {
        version: 1,
        intent: { family: 'citation_lookup' },
        render_packet: {
          answer_markdown: 'Compare two cited papers: [1](#kb-cite-race-a) and [2](#kb-cite-race-b).',
          rendered_body: 'Compare two cited papers: [1](#kb-cite-race-a) and [2](#kb-cite-race-b).',
          rendered_content: 'Compare two cited papers: [1](#kb-cite-race-a) and [2](#kb-cite-race-b).',
          copy_text: 'Compare two cited papers: [1] and [2].',
          copy_markdown: 'Compare two cited papers: [1](#kb-cite-race-a) and [2](#kb-cite-race-b).',
          cite_details: [
            {
              num: 1,
              anchor: 'kb-cite-race-a',
              source_name: 'Slow Paper A.pdf',
              source_path: '__citation_race__/slow-a.en.md',
              is_inpaper: true,
              raw: '[1] Slow Paper A.',
              doi: '10.0000/slow-a',
              doi_url: 'https://doi.org/10.0000/slow-a',
            },
            {
              num: 2,
              anchor: 'kb-cite-race-b',
              source_name: 'Fast Paper B.pdf',
              source_path: '__citation_race__/fast-b.en.md',
              is_inpaper: true,
              raw: '[2] Fast Paper B.',
              doi: '10.0000/fast-b',
              doi_url: 'https://doi.org/10.0000/fast-b',
            },
          ],
        },
      },
    },
  },
]

const weakSystemBPopoverMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'This discussion traces a cited upstream imaging paper [R3](#kb-cite-weak-system-b-r3).',
    created_at: Date.now(),
    meta: {
      paper_guide_contracts: {
        version: 1,
        intent: { family: 'citation_lookup' },
        render_packet: {
          answer_markdown: 'This discussion traces a cited upstream imaging paper [R3](#kb-cite-weak-system-b-r3).',
          rendered_body: 'This discussion traces a cited upstream imaging paper [R3](#kb-cite-weak-system-b-r3).',
          rendered_content: 'This discussion traces a cited upstream imaging paper [R3](#kb-cite-weak-system-b-r3).',
          copy_text: 'This discussion traces a cited upstream imaging paper [R3].',
          copy_markdown: 'This discussion traces a cited upstream imaging paper [R3](#kb-cite-weak-system-b-r3).',
          cite_details: [
            {
              num: 3,
              anchor: 'kb-cite-weak-system-b-r3',
              source_name: 'NatPhoton-2025-Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy.pdf',
              source_path: READER_REGRESSION_SOURCE_PATH,
              is_inpaper: true,
              raw: 'Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. & Aggarwal, J. K. The missing cone problem and low-pass distortion in optical serial sectioning microscopy. IEEE Trans. Acoust., Speech, Signal Process. 2, 890-893 (1988).',
              card_title: '上游参考文献',
              authors: 'Macias-Garza F',
              venue: 'IEEE Trans. Acoust., Speech, Signal Process.',
              year: '1988',
              doi: '10.1117/12.7976703',
              doi_url: 'https://doi.org/10.1117/12.7976703',
              citation_count: 22,
              citation_source: 'OpenAlex',
              journal_if: '1.2',
              journal_quartile: 'Q4',
              answer_claim: 'This citation should support the discussion of three-dimensional microscopic imaging limits.',
              citation_context: '## Authors\nAlessandro Zunino [1,4], Giacomo Garre [1,2,4], Eleonora Perego [1,3], Sabrina Zappone [1,2], Mattia Donato [1], Nadine Vastenhouw [3] & Giuseppe Vicidomini [1]',
              citation_context_source: 'source_markdown',
              evidence_quote: '**No useful citation context**',
              summary_line: '**No useful citation context**',
              location_label: 'Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy',
              card_locator: 'Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy',
              card_evidence: '',
              card_evidence_label: '引用语境',
              card_takeaway: '',
              card_quality_flags: ['weak_citation_context', 'missing_citation_context', 'missing_reference_title'],
              card_warning: '当前自动抽取的引用语境质量较弱，已隐藏低价值片段；建议打开原文核对。',
              system_b_trace_complete: false,
              system_b_trace_score: 0.25,
              system_b_trace_reason: '缺少当前论文里围绕该引用的正文语境，需要打开引用语境核对。',
              system_b_trace_flags: ['weak_citation_context', 'missing_citation_context', 'missing_reference_title'],
              system_b_trace_steps: ['答案句', '引用语境待核对', '上游文献'],
              system_b_trace_answer: 'This citation should support the discussion of three-dimensional microscopic imaging limits.',
              system_b_trace_reference: 'Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. & Aggarwal, J. K. The missing cone problem and low-pass distortion in optical serial sectioning microscopy. IEEE Trans. Acoust., Speech, Signal Process. 2, 890-893 (1988).',
              system_b_trace_locator: 'Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy',
              system_b_trace_source: 'source_markdown',
            },
          ],
        },
      },
    },
  },
]

const systemACitationPopoverMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'The method details are grounded in [1].',
    created_at: Date.now(),
    meta: {
      paper_guide_contracts: {
        version: 1,
        intent: { family: 'method' },
        render_packet: {
          answer_markdown: 'The method details are grounded in [1].',
          rendered_body: 'The method details are grounded in [1].',
          rendered_content: 'The method details are grounded in [1].',
          copy_text: 'The method details are grounded in [1].',
          copy_markdown: 'The method details are grounded in [1].',
          cite_details: [
            {
              num: 1,
              anchor: 'kb-cite-system-a-1',
              source_name: READER_REGRESSION_SOURCE_NAME,
              source_path: READER_REGRESSION_SOURCE_PATH,
              is_inpaper: false,
              title: 'Fixture Paper / 2. Method',
              raw: 'Given a set of input multi-view images, NeRF transfers the pixels of the input images into rays.',
              heading_path: 'Fixture Paper / 2. Method',
              answer_claim: 'The method details are grounded in this paper.',
              evidence_quote: 'Given a set of input multi-view images, NeRF transfers the pixels of the input images into rays.',
              evidence_source: 'retrieval_hit',
              location_label: 'Fixture Paper / 2. Method · p. 2 · sentence',
              summary_line: 'Given a set of input multi-view images, NeRF transfers the pixels of the input images into rays.',
              summary_source: 'retrieval_hit',
              support_relation: 'The Method section states the exact mechanism used by the answer.',
              why_line: 'The Method section states the exact mechanism used by the answer.',
              binding_status: 'grounded',
              binding_confidence: 0.84,
              binding_reason: 'The answer sentence and retrieved passage both mention NeRF and rays.',
              binding_overlap_terms: ['NeRF', 'rays'],
              block_id: 'p-method-1',
              anchor_id: 'a-p-method-1',
              anchor_kind: 'sentence',
              page_start: 2,
              page_end: 2,
              score: 8.9,
            },
          ],
        },
      },
    },
  },
]

const cardViewPriorityPopoverMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'This answer should render the polished card contract [1](#kb-cite-card-view-priority-1).',
    created_at: Date.now(),
    meta: {
      paper_guide_contracts: {
        version: 1,
        intent: { family: 'method' },
        render_packet: {
          answer_markdown: 'This answer should render the polished card contract [1](#kb-cite-card-view-priority-1).',
          rendered_body: 'This answer should render the polished card contract [1](#kb-cite-card-view-priority-1).',
          rendered_content: 'This answer should render the polished card contract [1](#kb-cite-card-view-priority-1).',
          copy_text: 'This answer should render the polished card contract [1].',
          copy_markdown: 'This answer should render the polished card contract [1](#kb-cite-card-view-priority-1).',
          cite_details: [
            {
              num: 1,
              anchor: 'kb-cite-card-view-priority-1',
              source_name: 'Legacy Repeated Source.pdf',
              source_path: READER_REGRESSION_SOURCE_PATH,
              is_inpaper: false,
              card_title: 'Legacy title should not win',
              card_subtitle: 'Legacy subtitle should not win',
              card_takeaway_label: 'Legacy label',
              card_takeaway: 'Legacy fallback takeaway should not render.',
              card_claim: 'Legacy claim should not render.',
              card_locator: 'Legacy Paper / Legacy Location',
              card_evidence_label: 'Legacy evidence',
              card_evidence: '## Legacy markdown evidence **should not render**',
              summary_line: '',
              card_view: {
                version: 1,
                route: 'system_a',
                kind: 'answer_evidence',
                header: {
                  kicker: 'Answer evidence',
                  title: 'Clean Card Title',
                  subtitle: 'Clean Method Section',
                },
                sections: [
                  {
                    id: 'takeaway',
                    label: 'Key point',
                    text: 'Polished card-view takeaway used by the popover and shelf.',
                    kind: 'insight',
                    hint: '',
                    tone: 'primary',
                  },
                  {
                    id: 'locator',
                    label: 'Location',
                    text: 'Clean Method Section · p. 4',
                    kind: 'locator',
                    hint: '',
                    tone: '',
                  },
                  {
                    id: 'evidence',
                    label: 'Source evidence',
                    text: 'The method uses calibrated measurements to reconstruct the scene in a verifiable way.',
                    kind: 'quote',
                    hint: '',
                    tone: '',
                  },
                ],
                summary: 'Polished card-view takeaway used by the popover and shelf.',
                quality: { label: 'polished', score: 0.92, flags: [], warning: '' },
              },
            },
          ],
        },
      },
    },
  },
]

const repeatedSystemAOldPacketMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: [
      '自适应采样策略：在感兴趣区域用高分辨率采样、边缘用低分辨率采样 [3](#kb-cite-repeated-system-a-3)。',
      '实际系统搭建：完整的单像素相机实验装置包含 DMD 投影和检测路径 [3](#kb-cite-repeated-system-a-3)。',
      '数据效率思维：采集阶段就做智能压缩，而不是等采集完再压缩 [3](#kb-cite-repeated-system-a-3)。',
    ].join('\n'),
    created_at: Date.now(),
    meta: {
      paper_guide_contracts: {
        version: 1,
        intent: { family: 'overview' },
        render_packet: {
          answer_markdown: [
            '自适应采样策略：在感兴趣区域用高分辨率采样、边缘用低分辨率采样 [3](#kb-cite-repeated-system-a-3)。',
            '实际系统搭建：完整的单像素相机实验装置包含 DMD 投影和检测路径 [3](#kb-cite-repeated-system-a-3)。',
            '数据效率思维：采集阶段就做智能压缩，而不是等采集完再压缩 [3](#kb-cite-repeated-system-a-3)。',
          ].join('\n'),
          rendered_body: [
            '自适应采样策略：在感兴趣区域用高分辨率采样、边缘用低分辨率采样 [3](#kb-cite-repeated-system-a-3)。',
            '实际系统搭建：完整的单像素相机实验装置包含 DMD 投影和检测路径 [3](#kb-cite-repeated-system-a-3)。',
            '数据效率思维：采集阶段就做智能压缩，而不是等采集完再压缩 [3](#kb-cite-repeated-system-a-3)。',
          ].join('\n'),
          rendered_content: '',
          copy_text: '',
          copy_markdown: '',
          cite_details: [
            {
              num: 3,
              anchor: 'kb-cite-repeated-system-a-3',
              source_name: 'SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf',
              source_path: READER_REGRESSION_SOURCE_PATH,
              is_inpaper: false,
              title: 'INTRODUCTION',
              heading_path: 'INTRODUCTION',
              evidence_quote: '## Foveated single-pixel imaging Single-pixel imaging is based on structured illumination and dynamic supersampling.',
              evidence_source: 'retrieval_hit',
              location_label: 'INTRODUCTION',
              anchor_kind: 'paragraph',
              page_start: 1,
              page_end: 1,
              score: 8.2,
            },
          ],
        },
      },
    },
  },
]

const lowQualitySystemAOldPacketMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: 'Deep learning review 1 [1](#kb-cite-low-quality-system-a-1)',
    created_at: Date.now(),
    meta: {
      paper_guide_contracts: {
        version: 1,
        intent: { family: 'overview' },
        render_packet: {
          answer_markdown: 'Deep learning review 1 [1](#kb-cite-low-quality-system-a-1)',
          rendered_body: 'Deep learning review 1 [1](#kb-cite-low-quality-system-a-1)',
          rendered_content: '',
          copy_text: '',
          copy_markdown: '',
          cite_details: [
            {
              num: 1,
              anchor: 'kb-cite-low-quality-system-a-1',
              source_name: 'LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf',
              source_path: READER_REGRESSION_SOURCE_PATH,
              is_inpaper: false,
              title: '5. Single-Pixel Imaging Realizations with Deep Learning',
              heading_path: '5. Single-Pixel Imaging Realizations with Deep Learning',
              answer_claim: 'Deep learning review 1',
              card_claim: 'Deep learning review 1',
              card_claim_label: '对应回答',
              evidence_quote: 'Advances and Challenges of Single-Pixel Imaging Based on Deep Learning Kai Song, Yaoxing Bian,\\ Dong Wang, Runrui Li, Ku Wu, Hongrui Liu, Chengbing Qin, Jianyong Hu,\\ and Liantuan Xiao* Single-pixel imaging technology can capture images at wavelengths outside the reach of conventional focal plane array detectors. However, limited image quality and long computation times still hinder practical application.',
              card_evidence: 'Advances and Challenges of Single-Pixel Imaging Based on Deep Learning Kai Song, Yaoxing Bian,\\ Dong Wang, Runrui Li, Ku Wu, Hongrui Liu, Chengbing Qin, Jianyong Hu,\\ and Liantuan Xiao* Single-pixel imaging technology can capture images at wavelengths outside the reach of conventional focal plane array detectors. However, limited image quality and long computation times still hinder practical application.',
              card_evidence_label: '原文证据',
              evidence_source: 'retrieval_hit',
              location_label: '5. Single-Pixel Imaging Realizations with Deep Learning',
              card_locator: '5. Single-Pixel Imaging Realizations with Deep Learning',
              anchor_kind: 'paragraph',
              page_start: 8,
              page_end: 8,
              score: 7.8,
            },
          ],
        },
      },
    },
  },
]

const fragmentarySystemAOldPacketMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: '这篇综述用 DMD 解释了单像素相机的基本配置 [2](#kb-cite-fragmentary-system-a-2)。',
    created_at: Date.now(),
    meta: {
      paper_guide_contracts: {
        version: 1,
        intent: { family: 'overview' },
        render_packet: {
          answer_markdown: '这篇综述用 DMD 解释了单像素相机的基本配置 [2](#kb-cite-fragmentary-system-a-2)。',
          rendered_body: '这篇综述用 DMD 解释了单像素相机的基本配置 [2](#kb-cite-fragmentary-system-a-2)。',
          rendered_content: '',
          copy_text: '',
          copy_markdown: '',
          cite_details: [
            {
              num: 2,
              anchor: 'kb-cite-fragmentary-system-a-2',
              source_name: 'NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf',
              source_path: READER_REGRESSION_SOURCE_PATH,
              is_inpaper: false,
              title: 'Abstract / Understanding compressed sensing',
              heading_path: 'Abstract / Understanding compressed sensing',
              answer_claim: '这篇综述用 DMD 解释了单像素相机的基本配置。',
              evidence_quote: 'rson can be described uniquely with a few targeted questions—a property closely related to sparsity that is key to many measurement problems and gives rise to the fields of both data compression and Figure 1. Computational imaging configurations. A DMD can be used to spatially filter light by selectively redirecting parts of an incident light beam at ±24° to the normal. a, Single-pixel camera configuration.',
              card_evidence: 'rson can be described uniquely with a few targeted questions—a property closely related to sparsity that is key to many measurement problems and gives rise to the fields of both data compression and Figure 1. Computational imaging configurations. A DMD can be used to spatially filter light by selectively redirecting parts of an incident light beam at ±24° to the normal. a, Single-pixel camera configuration.',
              card_evidence_label: '原文证据',
              evidence_source: 'retrieval_hit',
              location_label: 'Abstract / Understanding compressed sensing',
              card_locator: 'Abstract / Understanding compressed sensing',
              anchor_kind: 'paragraph',
              page_start: 2,
              page_end: 2,
              score: 8.0,
            },
          ],
        },
      },
    },
  },
]

const plainCitationRefsFallbackMessages: Message[] = [
  {
    id: 1,
    role: 'user',
    content: 'Explain two papers with citations.',
    created_at: Date.now(),
  },
  {
    id: 2,
    role: 'assistant',
    content: 'For a beginner, the broad point is that deep learning SPI improves reconstruction quality while also reducing iterative reconstruction cost and making low-sampling settings easier to use [1]. PILN uses a part-based image loop for self-supervised reconstruction [2].',
    rendered_body: 'For a beginner, the broad point is that deep learning SPI improves reconstruction quality while also reducing iterative reconstruction cost and making low-sampling settings easier to use [1]. PILN uses a part-based image loop for self-supervised reconstruction [2].',
    copy_text: 'For a beginner, the broad point is that deep learning SPI improves reconstruction quality while also reducing iterative reconstruction cost and making low-sampling settings easier to use [1]. PILN uses a part-based image loop for self-supervised reconstruction [2].',
    copy_markdown: 'For a beginner, the broad point is that deep learning SPI improves reconstruction quality while also reducing iterative reconstruction cost and making low-sampling settings easier to use [1]. PILN uses a part-based image loop for self-supervised reconstruction [2].',
    created_at: Date.now(),
  },
]

const plainCitationRefsFallbackRefs: Record<string, unknown> = {
  '1': {
    hits: [
      {
        score: 9.4,
        text: 'Single-pixel imaging based on deep learning improves reconstruction quality and computational speed.',
        ui_meta: {
          display_name: 'Deep Learning SPI Review.pdf',
          source_path: READER_REGRESSION_SOURCE_PATH,
          heading_path: 'Deep Learning SPI Review / 5. Realizations',
          summary_line: 'Single-pixel imaging based on deep learning improves reconstruction quality and computational speed.',
          why_line: 'This hit explains the broad deep-learning SPI advantage used by the answer.',
        },
        meta: {
          source_path: READER_REGRESSION_SOURCE_PATH,
          heading_path: 'Deep Learning SPI Review / 5. Realizations',
        },
      },
      {
        score: 8.7,
        text: 'PILN introduces a part-based image-loop network that uses the reconstructed image as the input of the next iteration.',
        ui_meta: {
          display_name: 'PILN Paper.pdf',
          source_path: '__fixtures__/piln-paper.en.md',
          heading_path: 'PILN Paper / Method',
          summary_line: 'PILN introduces a part-based image-loop network that uses the reconstructed image as the input of the next iteration.',
          why_line: 'This hit grounds the answer sentence about PILN method design.',
        },
        meta: {
          source_path: '__fixtures__/piln-paper.en.md',
          heading_path: 'PILN Paper / Method',
        },
      },
    ],
  },
}

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

const normalMultiDocAmbiguousInlineLocateMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: [
      '在提供的检索文档中，明确提到 **NeRF** 的文章有以下两篇：',
      '',
      '1. **DOC-1**：*CVPR-2024-SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image*',
      ' - 多次提及 NeRF 作为其底层场景表示方法，例如：“SCINeRF exploits neural radiance fields as its underlying scene representation”。',
      '',
      '2. **DOC-2**：*ICIP-2025-SCIGS: 3D Gaussians Splatting from A Snapshot Compressive Image*',
      ' - 明确对比了 NeRF-based 方法的局限性，如：“NeRF-based reconstruction methods still face limitations in handling dynamic scenes”。',
    ].join('\n'),
    rendered_body: [
      '在提供的检索文档中，明确提到 **NeRF** 的文章有以下两篇：',
      '',
      '1. **DOC-1**：*CVPR-2024-SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image*',
      ' - 多次提及 NeRF 作为其底层场景表示方法，例如：“SCINeRF exploits neural radiance fields as its underlying scene representation”。',
      '',
      '2. **DOC-2**：*ICIP-2025-SCIGS: 3D Gaussians Splatting from A Snapshot Compressive Image*',
      ' - 明确对比了 NeRF-based 方法的局限性，如：“NeRF-based reconstruction methods still face limitations in handling dynamic scenes”。',
    ].join('\n'),
    copy_text: 'normal multi-doc ambiguous inline locate',
    copy_markdown: 'normal multi-doc ambiguous inline locate',
    refs_user_msg_id: 9001,
    created_at: Date.now(),
  },
]

const normalMultiDocAmbiguousInlineLocateRefs: Record<string, unknown> = {
  '9001': {
    hits: [
      {
        ui_meta: {
          source_path: READER_REGRESSION_SOURCE_PATH,
          display_name: READER_REGRESSION_SOURCE_NAME,
          heading_path: 'Fixture Paper / Abstract',
          summary_line: 'Single-source fallback should not power multi-doc inline locate in normal mode.',
          why_line: 'This fixture intentionally leaves only one refs candidate.',
          reader_open: {
            sourcePath: READER_REGRESSION_SOURCE_PATH,
            sourceName: READER_REGRESSION_SOURCE_NAME,
            headingPath: 'Fixture Paper / Abstract',
            snippet: 'Single-source fallback should not power multi-doc inline locate in normal mode.',
            highlightSnippet: 'Single-source fallback should not power multi-doc inline locate in normal mode.',
          },
        },
      },
    ],
  },
}

const liveUserPendingRefsMessages: Message[] = [
  {
    id: 9101,
    role: 'user',
    content: 'Which paper compares Hadamard and Fourier single-pixel imaging?',
    created_at: Date.now(),
  },
]

const liveUserPendingRefs: Record<string, unknown> = {
  '9101': {
    display_state: 'pending',
    payload_mode: 'pending',
    enrichment_pending: true,
    hits: [
      {
        meta: {
          ref_pack_state: 'pending',
        },
        ui_meta: {
          source_path: READER_REGRESSION_SOURCE_PATH,
          display_name: READER_REGRESSION_SOURCE_NAME,
          heading_path: 'Fixture Paper / Theory Comparison',
          score_pending: true,
          summary_line: 'Provisional refs should render while the assistant answer is still streaming.',
          why_line: 'The refs payload is available before the assistant message is persisted.',
          reader_open: {
            sourcePath: READER_REGRESSION_SOURCE_PATH,
            sourceName: READER_REGRESSION_SOURCE_NAME,
            headingPath: 'Fixture Paper / Theory Comparison',
            snippet: 'Provisional refs should render while the assistant answer is still streaming.',
            highlightSnippet: 'Provisional refs should render while the assistant answer is still streaming.',
            strictLocate: false,
          },
        },
      },
    ],
  },
}

type RegressionScenario =
  | 'structured-primary-rerank'
  | 'required-fallback-anchor'
  | 'guide-figure-remap'
  | 'guide-formula-remap'
  | 'render-packet-contract'
  | 'render-packet-hidden-locate'
  | 'citation-hover-race'
  | 'weak-system-b-popover'
  | 'system-a-citation-popover'
  | 'card-view-priority-popover'
  | 'repeated-system-a-old-packet'
  | 'low-quality-system-a-old-packet'
  | 'fragmentary-system-a-old-packet'
  | 'plain-citation-refs-fallback'
  | 'guide-filter-empty-external'
  | 'negative-evidence-locate'
  | 'normal-multi-doc-ambiguous-inline-locate'
  | 'live-user-pending-refs'

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
    if (scenarioParam === 'citation-hover-race') return 'citation-hover-race'
    if (scenarioParam === 'weak-system-b-popover') return 'weak-system-b-popover'
    if (scenarioParam === 'system-a-citation-popover') return 'system-a-citation-popover'
    if (scenarioParam === 'card-view-priority-popover') return 'card-view-priority-popover'
    if (scenarioParam === 'repeated-system-a-old-packet') return 'repeated-system-a-old-packet'
    if (scenarioParam === 'low-quality-system-a-old-packet') return 'low-quality-system-a-old-packet'
    if (scenarioParam === 'fragmentary-system-a-old-packet') return 'fragmentary-system-a-old-packet'
    if (scenarioParam === 'plain-citation-refs-fallback') return 'plain-citation-refs-fallback'
    if (scenarioParam === 'guide-filter-empty-external') return 'guide-filter-empty-external'
    if (scenarioParam === 'negative-evidence-locate') return 'negative-evidence-locate'
    if (scenarioParam === 'normal-multi-doc-ambiguous-inline-locate') return 'normal-multi-doc-ambiguous-inline-locate'
    if (scenarioParam === 'live-user-pending-refs') return 'live-user-pending-refs'
    return 'structured-primary-rerank'
  })()
  const regressionMessages: Message[] = (() => {
    if (scenario === 'required-fallback-anchor') return requiredFallbackAnchorMessages
    if (scenario === 'guide-figure-remap') return guideFigureRemapMessages
    if (scenario === 'guide-formula-remap') return guideFormulaRemapMessages
    if (scenario === 'render-packet-contract') return renderPacketContractMessages
    if (scenario === 'render-packet-hidden-locate') return renderPacketHiddenLocateMessages
    if (scenario === 'citation-hover-race') return citationHoverRaceMessages
    if (scenario === 'weak-system-b-popover') return weakSystemBPopoverMessages
    if (scenario === 'system-a-citation-popover') return systemACitationPopoverMessages
    if (scenario === 'card-view-priority-popover') return cardViewPriorityPopoverMessages
    if (scenario === 'repeated-system-a-old-packet') return repeatedSystemAOldPacketMessages
    if (scenario === 'low-quality-system-a-old-packet') return lowQualitySystemAOldPacketMessages
    if (scenario === 'fragmentary-system-a-old-packet') return fragmentarySystemAOldPacketMessages
    if (scenario === 'plain-citation-refs-fallback') return plainCitationRefsFallbackMessages
    if (scenario === 'guide-filter-empty-external') return guideFilterOnlyMessages
    if (scenario === 'negative-evidence-locate') return negativeEvidenceLocateMessages
    if (scenario === 'normal-multi-doc-ambiguous-inline-locate') return normalMultiDocAmbiguousInlineLocateMessages
    if (scenario === 'live-user-pending-refs') return liveUserPendingRefsMessages
    return structuredPrimaryRerankMessages
  })()
  const regressionRefs: Record<string, unknown> = (() => {
    if (scenario === 'guide-filter-empty-external') return guideFilterOnlyRefs
    if (scenario === 'plain-citation-refs-fallback') return plainCitationRefsFallbackRefs
    if (scenario === 'normal-multi-doc-ambiguous-inline-locate') return normalMultiDocAmbiguousInlineLocateRefs
    if (scenario === 'live-user-pending-refs') return liveUserPendingRefs
    return {}
  })()
  const regressionGuideSourcePath = scenario === 'normal-multi-doc-ambiguous-inline-locate' || scenario === 'citation-hover-race'
    ? ''
    : READER_REGRESSION_SOURCE_PATH
  const regressionGuideSourceName = scenario === 'normal-multi-doc-ambiguous-inline-locate' || scenario === 'citation-hover-race'
    ? ''
    : READER_REGRESSION_SOURCE_NAME
  const [payload, setPayload] = useState<ReaderOpenPayload | null>(null)
  const [readerLocateResults, setReaderLocateResults] = useState<Record<string, ReaderLocateResult>>({})
  const readerEnabled = typeof window !== 'undefined'
    ? new URLSearchParams(window.location.search).get('reader') === '1'
    : false
  const recordLocateResult = useCallback((result: ReaderLocateResult) => {
    const key = String(result.locateFeedbackKey || '').trim()
    if (!key) return
    setReaderLocateResults((current) => {
      const prev = current[key]
      if (
        prev
        && prev.locateRequestId === result.locateRequestId
        && prev.status === result.status
        && prev.precision === result.precision
        && prev.hint === result.hint
      ) {
        return current
      }
      return { ...current, [key]: result }
    })
  }, [])

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
            readerLocateResults={readerLocateResults}
            paperGuideSourcePath={regressionGuideSourcePath}
            paperGuideSourceName={regressionGuideSourceName}
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

        {readerEnabled && payload ? (
          <div className="rounded-3xl border border-[var(--border)] bg-[var(--panel)] p-4">
            <PaperGuideReaderDrawer
              open
              payload={payload}
              onClose={() => {}}
              onAppendSelection={() => {}}
              presentation="inline"
              onLocateResult={recordLocateResult}
            />
          </div>
        ) : null}
      </div>
    </div>
  )
}
