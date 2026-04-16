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
    hits: [],
    guide_filter: {
      active: true,
      hidden_self_source: true,
      filtered_hit_count: 1,
      guide_source_name: READER_REGRESSION_SOURCE_NAME,
    },
  },
}

const REFS_PANEL_NEGATIVE_SUPPRESSED_PAYLOAD: Record<string, unknown> = {
  7: {
    prompt: 'In the SCINeRF paper, where is ADMM discussed? Please point me to the source section.',
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

export default function RefsPanelRegressionPage() {
  const scenarioParam = (() => {
    if (typeof window === 'undefined') return ''
    return new URLSearchParams(window.location.search).get('scenario') || ''
  })().trim().toLowerCase()
  const scenario = scenarioParam === 'guide-filter-note'
    ? 'guide-filter-note'
    : scenarioParam === 'negative-suppressed'
      ? 'negative-suppressed'
      : 'rich-reader-open'
  const [payload, setPayload] = useState<ReaderOpenPayload | null>(null)

  const refs = scenario === 'guide-filter-note'
    ? REFS_PANEL_GUIDE_FILTER_ONLY_PAYLOAD
    : scenario === 'negative-suppressed'
      ? REFS_PANEL_NEGATIVE_SUPPRESSED_PAYLOAD
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
