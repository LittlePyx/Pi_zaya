import { useState } from 'react'
import { RefsPanel } from '../components/refs/RefsPanel'
import type { ReaderOpenPayload } from '../components/chat/reader/readerTypes'
import {
  READER_REGRESSION_SOURCE_NAME,
  READER_REGRESSION_SOURCE_PATH,
} from '../testing/readerRegressionFixtures'

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
                text: 'This section explains the method at the level a first reading needs.',
              },
              {
                id: 'why',
                label: 'Relevance',
                title: 'Why it matches your question',
                text: 'It is a good first stop because it connects the paper title to the concrete method steps.',
              },
              {
                id: 'location',
                label: 'Location',
                title: 'Original location',
                text: 'Fixture Paper / 2. Method',
              },
            ],
            summary: 'This section explains the method at the level a first reading needs.',
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

export default function RefsPanelRegressionPage() {
  const scenarioParam = (() => {
    if (typeof window === 'undefined') return ''
    return new URLSearchParams(window.location.search).get('scenario') || ''
  })().trim().toLowerCase()
  const scenario = scenarioParam === 'guide-filter-note'
    ? 'guide-filter-note'
    : scenarioParam === 'negative-suppressed'
      ? 'negative-suppressed'
      : scenarioParam === 'section-target'
        ? 'section-target'
        : scenarioParam === 'auto-citation-meta'
          ? 'auto-citation-meta'
          : scenarioParam === 'polish-status'
            ? 'polish-status'
            : scenarioParam === 'card-view-contract'
              ? 'card-view-contract'
              : scenarioParam === 'pending-with-hits'
                ? 'pending-with-hits'
        : 'rich-reader-open'
  const [payload, setPayload] = useState<ReaderOpenPayload | null>(null)

  const refs = scenario === 'guide-filter-note'
    ? REFS_PANEL_GUIDE_FILTER_ONLY_PAYLOAD
    : scenario === 'negative-suppressed'
      ? REFS_PANEL_NEGATIVE_SUPPRESSED_PAYLOAD
      : scenario === 'section-target'
        ? REFS_PANEL_SECTION_TARGET_PAYLOAD
        : scenario === 'auto-citation-meta'
          ? REFS_PANEL_AUTO_CITATION_META_PAYLOAD
          : scenario === 'polish-status'
            ? REFS_PANEL_POLISH_STATUS_PAYLOAD
            : scenario === 'card-view-contract'
              ? REFS_PANEL_CARD_VIEW_PAYLOAD
              : scenario === 'pending-with-hits'
                ? REFS_PANEL_PENDING_WITH_HITS_PAYLOAD
      : REFS_PANEL_PAYLOAD

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
          <RefsPanel
            refs={refs}
            msgId={7}
            onOpenReader={(nextPayload) => setPayload(nextPayload)}
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
