import { useState } from 'react'
import { RefsPanel } from '../components/refs/RefsPanel'
import type { ReaderOpenPayload } from '../components/chat/reader/readerTypes'
import {
  READER_REGRESSION_SOURCE_NAME,
  READER_REGRESSION_SOURCE_PATH,
} from '../testing/readerRegressionFixtures'

const REFS_PANEL_SCENARIOS = new Set([
  'guide-filter-note',
  'negative-suppressed',
  'section-target',
  'auto-citation-meta',
  'citation-meta-reorder',
  'dedupe-active-source',
  'polish-status',
  'card-view-contract',
  'localized-relevance-fallback',
  'pending-with-hits',
  'research-basket-synthetic',
])

const REFS_PANEL_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: 'Where is Equation (1) introduced in the paper?',
    display_state: 'ready',
    hits: [
      {
        meta: {
          source_path: READER_REGRESSION_SOURCE_PATH,
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: READER_REGRESSION_SOURCE_NAME,
          source_path: READER_REGRESSION_SOURCE_PATH,
          heading_path: 'Fixture Paper / 2. Method',
          section_label: '2. Method',
          subsection_label: '2.1 Volume Rendering',
          summary_line: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
          summary_basis: '基于命中章节证据的 LLM 提炼',
          why_line: 'This hit answers where the equation-based rendering objective is introduced.',
          why_basis: '基于命中章节证据的 LLM 相关性说明',
          reader_open: {
            sourcePath: READER_REGRESSION_SOURCE_PATH,
            sourceName: READER_REGRESSION_SOURCE_NAME,
            headingPath: 'Fixture Paper / 2. Method',
            snippet: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
            highlightSnippet: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
            blockId: 'eq-1',
            anchorId: 'a-eq-1',
            relatedBlockIds: ['eq-1', 'p-method-1'],
            anchorKind: 'equation',
            anchorNumber: 1,
            strictLocate: true,
            locateTarget: {
              segmentId: 'refs-panel-seg-1',
              sourceSegmentId: 'refs-panel-seg-1',
              headingPath: 'Fixture Paper / 2. Method',
              snippet: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
              highlightSnippet: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
              blockId: 'eq-1',
              anchorId: 'a-eq-1',
              anchorKind: 'equation',
              anchorNumber: 1,
              hitLevel: 'block',
              locatePolicy: 'required',
              locateSurfacePolicy: 'primary',
              relatedBlockIds: ['eq-1', 'p-method-1'],
            },
            alternatives: [
              {
                headingPath: 'Fixture Paper / 2. Method / 2.2 Optimization',
                snippet: 'The optimization section explains how Equation (1) is minimized during training.',
                highlightSnippet: 'The optimization section explains how Equation (1) is minimized during training.',
                blockId: 'p-method-2',
                anchorId: 'a-p-method-2',
                anchorKind: 'equation',
                anchorNumber: 1,
              },
              {
                headingPath: 'Fixture Paper / 4. Experiments',
                snippet: 'Experimental analysis reuses the same rendering loss for ablation studies.',
                highlightSnippet: 'Experimental analysis reuses the same rendering loss for ablation studies.',
                blockId: 'p-exp-1',
                anchorId: 'a-p-exp-1',
              },
            ],
            visibleAlternatives: [
              {
                headingPath: 'Fixture Paper / 2. Method',
                snippet: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
                highlightSnippet: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
                blockId: 'eq-1',
                anchorId: 'a-eq-1',
                anchorKind: 'equation',
                anchorNumber: 1,
              },
              {
                headingPath: 'Fixture Paper / 2. Method / 2.2 Optimization',
                snippet: 'The optimization section explains how Equation (1) is minimized during training.',
                highlightSnippet: 'The optimization section explains how Equation (1) is minimized during training.',
                blockId: 'p-method-2',
                anchorId: 'a-p-method-2',
                anchorKind: 'equation',
                anchorNumber: 1,
              },
            ],
            evidenceAlternatives: [
              {
                headingPath: 'Fixture Paper / 2. Method',
                snippet: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
                highlightSnippet: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
                blockId: 'eq-1',
                anchorId: 'a-eq-1',
                anchorKind: 'equation',
                anchorNumber: 1,
              },
              {
                headingPath: 'Fixture Paper / 2. Method / 2.2 Optimization',
                snippet: 'The optimization section explains how Equation (1) is minimized during training.',
                highlightSnippet: 'The optimization section explains how Equation (1) is minimized during training.',
                blockId: 'p-method-2',
                anchorId: 'a-p-method-2',
                anchorKind: 'equation',
                anchorNumber: 1,
              },
              {
                headingPath: 'Fixture Paper / 4. Experiments',
                snippet: 'Experimental analysis reuses the same rendering loss for ablation studies.',
                highlightSnippet: 'Experimental analysis reuses the same rendering loss for ablation studies.',
                blockId: 'p-exp-1',
                anchorId: 'a-p-exp-1',
              },
            ],
            initialAltIndex: 0,
          },
        },
      },
    ],
  },
}

const REFS_PANEL_GUIDE_FILTER_ONLY_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: 'Besides this paper, what other papers in my library discuss ADMM?',
    display_state: 'hidden_by_guide',
    suppression_reason: 'guide_self_source_only',
    hits: [],
    guide_filter: {
      active: true,
      hidden_self_source: true,
      filtered_hit_count: 1,
      guide_source_name: READER_REGRESSION_SOURCE_NAME,
    },
  },
}

const REFS_PANEL_PENDING_WITH_HITS_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: 'Which papers mention NeRF?',
    display_state: 'pending',
    pending: true,
    payload_mode: 'pending',
    enrichment_pending: true,
    hits: [
      {
        text: 'SCINeRF uses neural radiance fields as the underlying scene representation.',
        meta: {
          source_path: READER_REGRESSION_SOURCE_PATH,
          ref_pack_state: 'pending',
          ref_best_heading_path: 'Fixture Paper / Abstract',
        },
        ui_meta: {
          display_name: READER_REGRESSION_SOURCE_NAME,
          source_path: READER_REGRESSION_SOURCE_PATH,
          heading_path: 'Fixture Paper / Abstract',
          section_label: 'Abstract',
          summary_line: 'SCINeRF uses neural radiance fields as the underlying scene representation.',
          summary_label: 'Guide',
          summary_title: 'Provisional Matched Section',
          why_line: 'This pending match directly mentions NeRF while the final card copy is still being refined.',
          score: null,
          score_pending: true,
          reader_open: {
            sourcePath: READER_REGRESSION_SOURCE_PATH,
            sourceName: READER_REGRESSION_SOURCE_NAME,
            headingPath: 'Fixture Paper / Abstract',
            snippet: 'SCINeRF uses neural radiance fields as the underlying scene representation.',
            highlightSnippet: 'SCINeRF uses neural radiance fields as the underlying scene representation.',
            strictLocate: false,
          },
        },
      },
    ],
  },
}

const REFS_PANEL_RESEARCH_BASKET_SYNTHETIC_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: 'Use the selected item from my literature basket.',
    display_state: 'ready',
    hits: [
      {
        text: 'Title: A hard to find preprint\nDOI: 10.1234/example.1\nSummary: selected metadata',
        score: 999,
        meta: {
          source_path: '__research_basket__/item_1_deadbeef',
          source_name: 'Research basket: A hard to find preprint',
          title: 'A hard to find preprint',
          doi: '10.1234/example.1',
          ref_pack_state: 'ready',
          research_basket_evidence: true,
          basket_source_role: 'synthetic_basket_item',
        },
        ui_meta: {
          display_name: 'Research basket: A hard to find preprint',
          source_path: '',
          heading_path: '',
          score: 9.2,
          score_pending: false,
          score_tier: 'high',
          summary_line: 'Title: A hard to find preprint DOI: 10.1234/example.1 Summary: selected metadata',
          summary_label: 'Research basket',
          summary_title: 'Selected Context',
          why_line: 'The user selected this literature-basket item for the current turn.',
          semantic_badges: [{ text: 'Research basket', score: 1 }],
          can_open: false,
          citation_meta: {
            title: 'A hard to find preprint',
            doi: '10.1234/example.1',
            source_name: 'Research basket: A hard to find preprint',
            source_path: '',
          },
          source_kind: 'research_basket',
          reader_open: {},
        },
      },
    ],
  },
}

const REFS_PANEL_NEGATIVE_SUPPRESSED_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: 'In the SCINeRF paper, where is ADMM discussed? Please point me to the source section.',
    display_state: 'suppressed',
    suppression_reason: 'focus_filter_removed_all',
    hits: [
      {
        text: 'Volume rendering equation used for neural field optimization.',
        meta: {
          source_path: 'F:\\library\\SCINeRF.pdf',
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: 'SCINeRF.pdf',
          source_path: 'F:\\library\\SCINeRF.pdf',
          heading_path: '2. Related Work',
          summary_line: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
          why_line: 'The paper does not mention ADMM and cannot point to a source section for it.',
          can_open: true,
          reader_open: {
            sourcePath: 'F:\\library\\SCINeRF.pdf',
            sourceName: 'SCINeRF.pdf',
            headingPath: '2. Related Work',
            snippet: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
            highlightSnippet: 'Equation (1) defines the volume rendering objective used for scene reconstruction.',
            blockId: 'eq-1',
            anchorId: 'a-eq-1',
            strictLocate: true,
          },
        },
      },
    ],
  },
}

const REFS_PANEL_SECTION_TARGET_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: 'Where does the paper discuss its limitations?',
    display_state: 'ready',
    hits: [
      {
        meta: {
          source_path: READER_REGRESSION_SOURCE_PATH,
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: READER_REGRESSION_SOURCE_NAME,
          source_path: READER_REGRESSION_SOURCE_PATH,
          heading_path: 'Fixture Paper / 5. Limitations',
          section_label: '5. Limitations',
          summary_line: 'The limitations section explains that temporal coverage is still traded against reconstruction stability.',
          summary_basis: '基于命中章节和小节标题的规则化摘要',
          why_line: 'This hit points directly to the section where the paper describes its current limitations.',
          why_basis: '基于章节级定位和问题关键词对齐',
          reader_open: {
            sourcePath: READER_REGRESSION_SOURCE_PATH,
            sourceName: READER_REGRESSION_SOURCE_NAME,
            headingPath: 'Fixture Paper / 5. Limitations',
            snippet: 'A current limitation is that the method still trades temporal coverage against reconstruction stability, especially when the scene departs from the static-scene assumption used by the reconstruction pipeline.',
            highlightSnippet: 'A current limitation is that the method still trades temporal coverage against reconstruction stability, especially when the scene departs from the static-scene assumption used by the reconstruction pipeline.',
            strictLocate: true,
            locateTarget: {
              segmentId: 'refs-panel-section-limitations',
              sourceSegmentId: 'refs-panel-section-limitations',
              headingPath: 'Fixture Paper / 5. Limitations',
              hitLevel: 'heading',
              claimType: 'section_claim',
              locatePolicy: 'required',
              locateSurfacePolicy: 'primary',
            },
            visibleAlternatives: [
              {
                headingPath: 'Fixture Paper / 5. Limitations',
                snippet: 'A current limitation is that the method still trades temporal coverage against reconstruction stability, especially when the scene departs from the static-scene assumption used by the reconstruction pipeline.',
                highlightSnippet: 'A current limitation is that the method still trades temporal coverage against reconstruction stability, especially when the scene departs from the static-scene assumption used by the reconstruction pipeline.',
              },
              {
                headingPath: 'Fixture Paper / 6. Future Work',
                snippet: 'Looking ahead, the most direct extension would be to combine the current pipeline with adaptive masking so dynamic scenes can be reconstructed more faithfully without increasing the hardware burden.',
                highlightSnippet: 'Looking ahead, the most direct extension would be to combine the current pipeline with adaptive masking so dynamic scenes can be reconstructed more faithfully without increasing the hardware burden.',
              },
            ],
            initialAltIndex: 0,
          },
        },
      },
    ],
  },
}

const REFS_PANEL_AUTO_CITATION_META_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: '哪几篇文章里提到了NeRF',
    display_state: 'ready',
    hits: [
      {
        meta: {
          source_path: '__refs_panel_regression__/CVPR-2024-SCINeRF.en.md',
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: 'CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf',
          source_path: '__refs_panel_regression__/CVPR-2024-SCINeRF.en.md',
          heading_path: 'Abstract',
          summary_line: '论文提出了 SCINeRF，并将 NeRF 作为底层 3D 场景表示。',
          why_line: 'Abstract 里说明该文把 NeRF 用作底层 3D 场景表示，可用来核对 NeRF 线索。',
          can_open: true,
          citation_meta: {},
        },
      },
    ],
  },
}

const REFS_PANEL_CITATION_META_REORDER_HITS = [
  {
    meta: {
      source_path: '__refs_panel_regression__/collection-a/Paper-A.en.md',
      ref_pack_state: 'ready',
    },
    ui_meta: {
      display_name: 'Paper A.pdf',
      source_path: '__refs_panel_regression__/collection-a/Paper-A.en.md',
      heading_path: 'Abstract',
      summary_line: 'Paper A evidence.',
      why_line: 'Paper A answers the first part.',
      citation_meta: {},
    },
  },
  {
    meta: {
      source_path: '__refs_panel_regression__/collection-b/Paper-B.en.md',
      ref_pack_state: 'ready',
    },
    ui_meta: {
      display_name: 'Paper B.pdf',
      source_path: '__refs_panel_regression__/collection-b/Paper-B.en.md',
      heading_path: 'Results',
      summary_line: 'Paper B evidence.',
      why_line: 'Paper B answers the second part.',
      citation_meta: {},
    },
  },
]

const REFS_PANEL_DEDUPE_ACTIVE_SOURCE_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: 'What should I read next after the active paper?',
    display_state: 'ready',
    hits: [
      {
        meta: {
          source_path: READER_REGRESSION_SOURCE_PATH,
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: READER_REGRESSION_SOURCE_NAME,
          source_path: READER_REGRESSION_SOURCE_PATH,
          heading_path: 'Fixture Paper / Introduction',
          score: 9.9,
          summary_line: 'This is the active reading-guide paper and should not appear as a reference card.',
          why_line: 'The user is already reading this paper.',
        },
      },
      {
        meta: {
          source_path: READER_REGRESSION_SOURCE_PATH,
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: READER_REGRESSION_SOURCE_NAME,
          source_path: READER_REGRESSION_SOURCE_PATH,
          heading_path: 'Fixture Paper / Results',
          score: 9.4,
          summary_line: 'A second hit from the active paper should also be hidden.',
          why_line: 'It duplicates the selected reading-guide source.',
        },
      },
      {
        meta: {
          source_path: 'F:\\library\\Related Work.en.md',
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: 'Related Work.pdf',
          source_path: 'F:\\library\\Related Work.en.md',
          heading_path: 'Related Work / Method',
          score: 8.2,
          summary_line: 'This related paper gives background context for the active paper.',
          why_line: 'It is an external source and should remain visible.',
        },
      },
      {
        meta: {
          source_path: 'F:\\library\\Related Work.pdf',
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: 'Related Work.pdf',
          source_path: 'F:\\library\\Related Work.pdf',
          heading_path: 'Related Work / Discussion',
          score: 7.8,
          summary_line: 'A second section from the same related paper should be merged into one card.',
          why_line: 'It has the same document identity as the previous related hit.',
        },
      },
    ],
  },
}

const REFS_PANEL_POLISH_STATUS_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: 'Which papers help me understand single-pixel imaging reconstruction quality?',
    display_state: 'ready',
    polish_status: 'heuristic',
    hits: [
      {
        text: 'A polished card grounded in matched evidence.',
        meta: {
          source_path: READER_REGRESSION_SOURCE_PATH,
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: READER_REGRESSION_SOURCE_NAME,
          source_path: READER_REGRESSION_SOURCE_PATH,
          heading_path: 'Fixture Paper / 3. Results',
          summary_line: 'The results section reports reconstruction quality improvements under the tested imaging setup.',
          summary_generation: 'llm_grounded',
          why_line: 'This card lets the user verify where reconstruction quality is evaluated and what evidence supports it.',
          why_generation: 'llm_grounded',
          polish_status: 'full',
          polish_detail: 'summary:llm_grounded->full;why:llm_grounded->full',
        },
      },
      {
        text: 'A deterministic fallback card grounded in section evidence.',
        meta: {
          source_path: 'F:\\library\\Fallback.en.md',
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: 'Fallback Paper.pdf',
          source_path: 'F:\\library\\Fallback.en.md',
          heading_path: 'Fallback Paper / 2. Method',
          summary_line: 'The method section describes the reconstruction pipeline used by the paper.',
          summary_generation: 'deterministic_grounded',
          why_line: 'Use this section to check the source wording for the reconstruction pipeline.',
          why_generation: 'deterministic_grounded',
          polish_status: 'heuristic',
          polish_detail: 'summary:deterministic_grounded->heuristic;why:deterministic_grounded->heuristic',
        },
      },
    ],
  },
}

const REFS_PANEL_CARD_VIEW_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: 'I am new to single-pixel imaging. Which source should I read first and why?',
    display_state: 'ready',
    hits: [
      {
        text: 'Fallback evidence text that should not define the visible card.',
        meta: {
          source_path: READER_REGRESSION_SOURCE_PATH,
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: READER_REGRESSION_SOURCE_NAME,
          source_path: READER_REGRESSION_SOURCE_PATH,
          heading_path: 'Fixture Paper / 2. Method',
          summary_label: 'Old label',
          summary_title: 'Old summary title',
          summary_line: 'Old fallback summary should not be rendered when card_view is present.',
          why_line: 'Old fallback reason should not be rendered when card_view is present.',
          card_view: {
            version: 1,
            route: 'references',
            kind: 'reference_locator',
            header: {
              kicker: 'References',
              title: READER_REGRESSION_SOURCE_NAME,
              subtitle: 'Fixture Paper / 2. Method',
            },
            sections: [
              {
                id: 'summary',
                label: 'Guide',
                title: 'What this section gives you',
                text: '\u8fd9\u4e00\u8282\u4ee5\u521d\u6b21\u9605\u8bfb\u6240\u9700\u7684\u7c92\u5ea6\u89e3\u91ca\u4e86\u8be5\u65b9\u6cd5\u3002',
              },
              {
                id: 'why',
                label: 'Relevance',
                title: 'Why it matches your question',
                text: '\u5b83\u628a\u8bba\u6587\u4e3b\u9898\u4e0e\u5177\u4f53\u65b9\u6cd5\u6b65\u9aa4\u76f4\u63a5\u8054\u7cfb\u8d77\u6765\uff0c\u9002\u5408\u7528\u4e8e\u56de\u7b54\u5f53\u524d\u95ee\u9898\u3002',
              },
              {
                id: 'location',
                label: 'Location',
                title: 'Original location',
                text: 'Fixture Paper / 2. Method',
              },
            ],
            summary: '\u8fd9\u4e00\u8282\u4ee5\u521d\u6b21\u9605\u8bfb\u6240\u9700\u7684\u7c92\u5ea6\u89e3\u91ca\u4e86\u8be5\u65b9\u6cd5\u3002',
            quality: {
              label: 'full',
              source: 'llm',
            },
          },
        },
      },
    ],
  },
}

const REFS_PANEL_LOCALIZED_RELEVANCE_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: '\u9891\u5206\u590d\u7528\u5355\u50cf\u7d20\u6210\u50cf\u4e3a\u4ec0\u4e48\u66f4\u5feb\uff1f',
    display_state: 'ready',
    hits: [
      {
        text: 'We propose and experimentally realize frequency-division-multiplexed single-pixel imaging.',
        meta: {
          source_path: 'F:\\library\\Localized-Support.en.md',
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: 'Localized Support.pdf',
          source_path: 'F:\\library\\Localized-Support.en.md',
          heading_path: 'Abstract',
          summary_line: '\u8be5\u6587\u5728\u6458\u8981\u4e2d\u7ed9\u51fa\u4e86\u9891\u5206\u590d\u7528\u7684\u5e76\u884c\u91c7\u96c6\u673a\u5236\u3002',
          card_support_explanation: '\u8fd9\u6761\u8bc1\u636e\u76f4\u63a5\u56de\u7b54\u4e86\u52a0\u901f\u6765\u6e90\uff0c\u5e76\u8bf4\u660e\u901f\u5ea6\u4e0e\u4fe1\u566a\u6bd4\u4e4b\u95f4\u7684\u6743\u8861\u3002',
          card_view: {
            sections: [
              {
                id: 'summary',
                text: 'English card guide that must not win in the Chinese locale.',
              },
              {
                id: 'why',
                text: 'We propose and experimentally realize frequency-division-multiplexed single-pixel imaging.',
              },
            ],
          },
        },
      },
      {
        text: 'Multiple carriers are measured in parallel without changing detector integration time.',
        meta: {
          source_path: 'F:\\library\\Legacy-Why.en.md',
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: 'Legacy Why.pdf',
          source_path: 'F:\\library\\Legacy-Why.en.md',
          heading_path: '2. Method',
          summary_line: '\u591a\u4e2a\u9891\u7387\u8f7d\u6ce2\u53ef\u4ee5\u5728\u540c\u4e00\u6b21\u79ef\u5206\u4e2d\u5e76\u884c\u6d4b\u91cf\u3002',
          why_line: '\u8be5\u6bb5\u540c\u65f6\u652f\u6491\u201c\u66f4\u5feb\u201d\u548c\u201c\u65e0\u9700\u6539\u53d8\u79ef\u5206\u65f6\u95f4\u201d\u4e24\u4e2a\u7ed3\u8bba\u3002',
          card_view: {
            sections: [
              {
                id: 'summary',
                text: '\u591a\u4e2a\u9891\u7387\u8f7d\u6ce2\u53ef\u4ee5\u5728\u540c\u4e00\u6b21\u79ef\u5206\u4e2d\u5e76\u884c\u6d4b\u91cf\u3002',
              },
            ],
          },
        },
      },
      {
        text: 'Raw evidence must not be relabeled as relevance copy.',
        meta: {
          source_path: 'F:\\library\\Evidence-Only.en.md',
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: 'Evidence Only.pdf',
          source_path: 'F:\\library\\Evidence-Only.en.md',
          heading_path: '3. Results',
          summary_line: '\u8be5\u6587\u62a5\u544a\u4e86\u591a\u9891\u7387\u5e76\u884c\u91c7\u96c6\u7ed3\u679c\u3002',
          card_view: {
            sections: [
              {
                id: 'summary',
                text: '\u8be5\u6587\u62a5\u544a\u4e86\u591a\u9891\u7387\u5e76\u884c\u91c7\u96c6\u7ed3\u679c\u3002',
              },
              {
                id: 'why',
                text: 'Raw evidence must not be relabeled as relevance copy.',
              },
            ],
          },
        },
      },
      {
        text: 'The source excerpt reports the measured acquisition result.',
        meta: {
          source_path: 'F:\\library\\English-Relevance.en.md',
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: 'English Relevance.pdf',
          source_path: 'F:\\library\\English-Relevance.en.md',
          heading_path: '4. Discussion',
          summary_line: '\u8be5\u6587\u5bf9\u9891\u5206\u590d\u7528\u7684\u91c7\u96c6\u7ed3\u679c\u8fdb\u884c\u4e86\u5b9e\u9a8c\u8ba8\u8bba\u3002',
          card_view: {
            sections: [
              {
                id: 'summary',
                text: '\u8be5\u6587\u5bf9\u9891\u5206\u590d\u7528\u7684\u91c7\u96c6\u7ed3\u679c\u8fdb\u884c\u4e86\u5b9e\u9a8c\u8ba8\u8bba\u3002',
              },
              {
                id: 'why',
                text: 'This discussion is relevant because it explains the acquisition-speed trade-off.',
              },
            ],
          },
        },
      },
      {
        text: 'The paper compares parallel acquisition speed and signal quality.',
        meta: {
          source_path: 'F:\\library\\English-Summary.en.md',
          ref_pack_state: 'ready',
        },
        ui_meta: {
          display_name: 'English Summary.pdf',
          source_path: 'F:\\library\\English-Summary.en.md',
          heading_path: '5. Conclusion',
          summary_line: 'This guide explains the measured speed and signal-quality trade-off.',
          why_line: '\u8fd9\u6761\u5b9a\u4f4d\u7528\u4e8e\u6838\u5bf9\u56de\u7b54\u4e2d\u7684\u901f\u5ea6\u4e0e\u4fe1\u53f7\u8d28\u91cf\u6743\u8861\u3002',
          card_view: {
            sections: [
              {
                id: 'summary',
                text: 'This guide explains the measured speed and signal-quality trade-off.',
              },
              {
                id: 'why',
                text: '\u8fd9\u6761\u5b9a\u4f4d\u7528\u4e8e\u6838\u5bf9\u56de\u7b54\u4e2d\u7684\u901f\u5ea6\u4e0e\u4fe1\u53f7\u8d28\u91cf\u6743\u8861\u3002',
              },
            ],
          },
        },
      },
    ],
  },
}

export default function RefsPanelRegressionPage() {
  const scenarioParam = (() => {
    if (typeof window === 'undefined') return ''
    return new URLSearchParams(window.location.search).get('scenario') || ''
  })().trim().toLowerCase()
  const scenario = REFS_PANEL_SCENARIOS.has(scenarioParam) ? scenarioParam : 'rich-reader-open'
  const [payload, setPayload] = useState<ReaderOpenPayload | null>(null)
  const [reverseCitationMetaHits, setReverseCitationMetaHits] = useState(false)

  const refsByScenario: Record<string, Record<string, unknown>> = {
    'guide-filter-note': REFS_PANEL_GUIDE_FILTER_ONLY_PAYLOAD,
    'negative-suppressed': REFS_PANEL_NEGATIVE_SUPPRESSED_PAYLOAD,
    'section-target': REFS_PANEL_SECTION_TARGET_PAYLOAD,
    'auto-citation-meta': REFS_PANEL_AUTO_CITATION_META_PAYLOAD,
    'citation-meta-reorder': {
      7: {
        prompt: 'Compare Paper A and Paper B.',
        display_state: 'ready',
        hits: reverseCitationMetaHits
          ? [...REFS_PANEL_CITATION_META_REORDER_HITS].reverse()
          : REFS_PANEL_CITATION_META_REORDER_HITS,
      },
    },
    'dedupe-active-source': REFS_PANEL_DEDUPE_ACTIVE_SOURCE_PAYLOAD,
    'polish-status': REFS_PANEL_POLISH_STATUS_PAYLOAD,
    'card-view-contract': REFS_PANEL_CARD_VIEW_PAYLOAD,
    'localized-relevance-fallback': REFS_PANEL_LOCALIZED_RELEVANCE_PAYLOAD,
    'pending-with-hits': REFS_PANEL_PENDING_WITH_HITS_PAYLOAD,
    'research-basket-synthetic': REFS_PANEL_RESEARCH_BASKET_SYNTHETIC_PAYLOAD,
  }
  const refs = refsByScenario[scenario] || REFS_PANEL_PAYLOAD

  return (
    <div className="min-h-screen bg-[var(--bg)] px-6 py-6">
      <div className="mx-auto max-w-5xl space-y-4">
        <div>
          <div className="text-sm font-medium text-black/80 dark:text-white/80">
            RefsPanel reader-open regression harness
          </div>
          <div className="text-xs text-black/45 dark:text-white/45" data-testid="refs-panel-test-scenario">
            {scenario}
          </div>
        </div>

        <div className="rounded-3xl border border-[var(--border)] bg-[var(--panel)] p-4">
          {scenario === 'citation-meta-reorder' ? (
            <button
              type="button"
              onClick={() => setReverseCitationMetaHits((current) => !current)}
            >
              Swap reference order
            </button>
          ) : null}
          <RefsPanel
            refs={refs}
            msgId={7}
            onOpenReader={(nextPayload) => setPayload(nextPayload)}
            activeSourcePath={scenario === 'dedupe-active-source' ? READER_REGRESSION_SOURCE_PATH : undefined}
            activeSourceName={scenario === 'dedupe-active-source' ? READER_REGRESSION_SOURCE_NAME : undefined}
          />
        </div>

        <div className="rounded-3xl border border-[var(--border)] bg-[var(--panel)] p-4">
          <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-black/40 dark:text-white/40">
            Last open payload
          </div>
          <pre
            className="min-h-24 whitespace-pre-wrap rounded-2xl border border-[var(--border)] bg-white/60 px-3 py-3 text-xs text-black/70 dark:bg-black/20 dark:text-white/70"
            data-testid="refs-panel-open-payload"
          >
            {payload ? JSON.stringify(payload, null, 2) : '(empty)'}
          </pre>
        </div>
      </div>
    </div>
  )
}
