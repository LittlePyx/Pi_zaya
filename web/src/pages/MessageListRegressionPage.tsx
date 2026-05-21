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
              title: 'Single-shot compressive spectral imaging with a dual-disperser architecture',
              authors: 'Gehm M, Brady D',
              venue: 'Optics Express',
              year: '2007',
              doi: '10.1364/OE.15.014013',
              doi_url: 'https://doi.org/10.1364/OE.15.014013',
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
    content: 'Deep learning SPI improves reconstruction quality [1]. PILN uses a part-based image loop for self-supervised reconstruction [2].',
    rendered_body: 'Deep learning SPI improves reconstruction quality [1]. PILN uses a part-based image loop for self-supervised reconstruction [2].',
    copy_text: 'Deep learning SPI improves reconstruction quality [1]. PILN uses a part-based image loop for self-supervised reconstruction [2].',
    copy_markdown: 'Deep learning SPI improves reconstruction quality [1]. PILN uses a part-based image loop for self-supervised reconstruction [2].',
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
  | 'system-a-citation-popover'
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
    if (scenarioParam === 'system-a-citation-popover') return 'system-a-citation-popover'
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
    if (scenario === 'system-a-citation-popover') return systemACitationPopoverMessages
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
      </div>
    </div>
  )
}
