import { useMemo, useState } from 'react'
import { MessageList } from '../components/chat/MessageList'
import type { ReaderOpenPayload } from '../components/chat/reader/readerTypes'
import {
  RESEARCH_LIBRARY_DOCS,
  RESEARCH_QA_CASES,
  RESEARCH_QA_MESSAGES,
  RESEARCH_QA_REFS,
} from '../testing/researchQaFixtures'

export default function ResearchQaReplayPage() {
  const [payload, setPayload] = useState<ReaderOpenPayload | null>(null)
  const coveredDocIds = useMemo(() => new Set(RESEARCH_QA_CASES.flatMap((item) => item.docIds)), [])
  const coveredDocs = useMemo(
    () => RESEARCH_LIBRARY_DOCS.filter((doc) => coveredDocIds.has(doc.id)),
    [coveredDocIds],
  )

  return (
    <div className="min-h-screen bg-[var(--bg)] px-5 py-5 text-[var(--text)]">
      <div className="mx-auto flex max-w-7xl flex-col gap-4">
        <section className="rounded-[8px] border border-[var(--border)] bg-[var(--panel)] p-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h1 className="m-0 text-[20px] font-semibold">Research QA replay</h1>
              <p className="mt-1 text-sm text-[var(--muted-text)]">
                基于当前本地文献库抽样构造的真实科研学习问题，用来检查回答质量、原文证据、文内参考和参考定位卡。
              </p>
            </div>
            <div className="flex gap-2 text-sm">
              <span className="rounded-full border border-[var(--border)] px-3 py-1" data-testid="research-qa-doc-count">
                文献 {RESEARCH_LIBRARY_DOCS.length}
              </span>
              <span className="rounded-full border border-[var(--border)] px-3 py-1" data-testid="research-qa-case-count">
                问题 {RESEARCH_QA_CASES.length}
              </span>
            </div>
          </div>

          <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {coveredDocs.map((doc) => (
              <div
                key={doc.id}
                className="rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] px-3 py-2"
                data-testid={`research-qa-doc-${doc.id}`}
              >
                <div className="text-sm font-medium">{doc.shortLabel}</div>
                <div className="mt-1 line-clamp-2 text-xs text-[var(--muted-text)]">{doc.title}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-[8px] border border-[var(--border)] bg-[var(--panel)] p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold">验收问题集</div>
              <div className="mt-1 text-xs text-[var(--muted-text)]">
                每个问题都按“普通研究生/科研学习者会怎么问”设计，避免直接说功能名。
              </div>
            </div>
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            {RESEARCH_QA_CASES.map((item, index) => (
              <div
                key={item.id}
                className="rounded-[8px] border border-[var(--border)] px-3 py-2"
                data-testid={`research-qa-case-${item.id}`}
              >
                <div className="text-xs text-[var(--muted-text)]">#{index + 1}</div>
                <div className="mt-1 text-sm font-medium">{item.question}</div>
                <div className="mt-2 text-xs text-[var(--muted-text)]">
                  {item.acceptance.join(' / ')}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="min-h-[720px] rounded-[8px] border border-[var(--border)] bg-[var(--panel)]">
          <MessageList
            messages={RESEARCH_QA_MESSAGES}
            refs={RESEARCH_QA_REFS}
            onOpenReader={(nextPayload) => setPayload(nextPayload)}
            paperGuideSourcePath=""
            paperGuideSourceName=""
          />
        </section>

        <section className="rounded-[8px] border border-[var(--border)] bg-[var(--panel)] p-4">
          <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-[var(--muted-text)]">
            Last open payload
          </div>
          <pre
            className="min-h-24 whitespace-pre-wrap rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] px-3 py-3 text-xs"
            data-testid="research-qa-open-payload"
          >
            {payload ? JSON.stringify(payload, null, 2) : '(empty)'}
          </pre>
        </section>
      </div>
    </div>
  )
}
