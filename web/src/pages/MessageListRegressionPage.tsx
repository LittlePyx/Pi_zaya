import { useState } from 'react'
import { MessageList } from '../components/chat/MessageList'
import type { Message } from '../api/chat'
import type { ReaderOpenPayload } from '../components/chat/reader/readerTypes'
import {
  READER_REGRESSION_SOURCE_NAME,
  READER_REGRESSION_SOURCE_PATH,
} from '../testing/readerRegressionFixtures'

const EQUATION_TEXT = '$$\nC(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt\n$$'

const regressionMessages: Message[] = [
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

export default function MessageListRegressionPage() {
  const [payload, setPayload] = useState<ReaderOpenPayload | null>(null)

  return (
    <div className="min-h-screen bg-[var(--bg)] px-6 py-6">
      <div className="mx-auto max-w-5xl space-y-4">
        <div>
          <div className="text-sm font-medium text-black/80 dark:text-white/80">
            MessageList locate regression harness
          </div>
          <div className="text-xs text-black/45 dark:text-white/45" data-testid="message-list-test-scenario">
            structured-primary-rerank
          </div>
        </div>

        <div className="rounded-3xl border border-[var(--border)] bg-[var(--panel)] p-4">
          <MessageList
            messages={regressionMessages}
            refs={{}}
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
