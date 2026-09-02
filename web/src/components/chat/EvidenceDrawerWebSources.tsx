import { ExportOutlined, GlobalOutlined } from '@ant-design/icons'
import type { ExternalWebSource } from '../../api/chat'

export function EvidenceDrawerWebSources({
  sources,
  S,
}: {
  sources: ExternalWebSource[]
  S: Record<string, string>
}) {
  if (sources.length <= 0) return null
  return (
    <section className="kb-evidence-web-section" data-testid="evidence-web-sources">
      <div className="kb-evidence-section-heading">
        <GlobalOutlined />
        <span>{S.evidence_web_sources_heading || 'Web sources (external, not in library)'}</span>
      </div>
      <div className="kb-evidence-web-notice">{S.evidence_web_sources_notice}</div>
      <div className="kb-evidence-web-list">
        {sources.map((source, index) => (
          <a
            key={`${source.url}-${index}`}
            className="kb-evidence-web-card"
            href={source.url}
            target="_blank"
            rel="noreferrer noopener"
            data-testid={`evidence-web-source-${index}`}
          >
            <span className="kb-evidence-web-title">{source.title || source.domain || source.url}</span>
            <span className="kb-evidence-web-meta">
              <span>{source.domain || new URL(source.url).hostname}</span>
              <span className="kb-evidence-web-open">{S.evidence_web_open}<ExportOutlined /></span>
            </span>
          </a>
        ))}
      </div>
    </section>
  )
}
