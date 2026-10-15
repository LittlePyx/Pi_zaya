import { useState } from 'react'
import { Button, Tag, message } from 'antd'
import { CheckCircleOutlined, DownloadOutlined, ExportOutlined, GlobalOutlined } from '@ant-design/icons'
import type { ExternalWebSource } from '../../api/chat'
import { libraryApi } from '../../api/library'

export function EvidenceDrawerWebSources({
  sources,
  S,
}: {
  sources: ExternalWebSource[]
  S: Record<string, string>
}) {
  const [importState, setImportState] = useState<Record<string, 'idle' | 'loading' | 'queued' | 'existing'>>({})
  if (sources.length <= 0) return null

  const importSource = async (source: ExternalWebSource) => {
    const pdfUrl = String(source.pdf_url || '').trim()
    if (!pdfUrl || importState[source.url] === 'loading') return
    setImportState((current) => ({ ...current, [source.url]: 'loading' }))
    try {
      const result = await libraryApi.importExternalPdf(pdfUrl, {
        title: source.title,
        convertNow: true,
        speedMode: 'balanced',
      })
      const nextState = result.candidate_status === 'already_imported' ? 'existing' : 'queued'
      setImportState((current) => ({ ...current, [source.url]: nextState }))
      message.success(
        nextState === 'existing'
          ? (S.evidence_web_import_existing || 'Already in the library.')
          : (S.evidence_web_import_queued || 'Imported. Conversion and indexing are now queued.'),
      )
    } catch (error) {
      setImportState((current) => ({ ...current, [source.url]: 'idle' }))
      message.error(
        error instanceof Error && error.message
          ? error.message
          : (S.evidence_web_import_failed || 'The PDF could not be imported safely.'),
      )
    }
  }

  return (
    <section className="kb-evidence-web-section" data-testid="evidence-web-sources">
      <div className="kb-evidence-section-heading">
        <GlobalOutlined />
        <span>{S.evidence_web_sources_heading || 'Web sources (external, not in library)'}</span>
      </div>
      <div className="kb-evidence-web-notice">{S.evidence_web_sources_notice}</div>
      <div className="kb-evidence-web-list">
        {sources.map((source, index) => {
          const state = importState[source.url] || 'idle'
          const identity = source.arxiv_id
            ? `arXiv ${source.arxiv_id}`
            : source.doi
              ? `DOI ${source.doi}`
              : ''
          return (
            <div key={`${source.url}-${index}`} className="kb-evidence-web-card">
              <div className="kb-evidence-web-badges">
                <Tag color="success" icon={<CheckCircleOutlined />}>
                  {source.verification_level === 'content_match'
                    ? (S.evidence_web_content_checked || 'Link and source title checked')
                    : (S.evidence_web_link_checked || 'Link checked')}
                </Tag>
                {identity ? <Tag>{identity}</Tag> : null}
              </div>
              <a
                className="kb-evidence-web-title"
                href={source.url}
                target="_blank"
                rel="noreferrer noopener"
                data-testid={`evidence-web-source-${index}`}
              >
                {source.title || source.domain || source.url}
              </a>
              {source.claim_text ? (
                <div className="kb-evidence-web-claim" data-testid={`evidence-web-claim-${index}`}>
                  <span>{S.evidence_web_claim_label || 'Supports this answer passage'}</span>
                  <q>{source.claim_text}</q>
                </div>
              ) : null}
              <div className="kb-evidence-web-meta">
                <span>{source.domain || new URL(source.url).hostname}</span>
                <a href={source.url} target="_blank" rel="noreferrer noopener" className="kb-evidence-web-open">
                  {S.evidence_web_open}<ExportOutlined />
                </a>
              </div>
              {source.import_eligible && source.pdf_url ? (
                <Button
                  size="small"
                  icon={<DownloadOutlined />}
                  loading={state === 'loading'}
                  disabled={state === 'queued' || state === 'existing'}
                  data-testid={`evidence-web-import-${index}`}
                  onClick={() => void importSource(source)}
                >
                  {state === 'queued'
                    ? (S.evidence_web_import_queued_short || 'Queued for processing')
                    : state === 'existing'
                      ? (S.evidence_web_import_existing_short || 'Already in library')
                      : (S.evidence_web_import_action || 'Import PDF and process')}
                </Button>
              ) : null}
            </div>
          )
        })}
      </div>
    </section>
  )
}
