interface CitationPopoverMetaRow {
  label: string
  value: string
}

interface CitationPopoverMetaPanelsProps {
  showMetaGrid: boolean
  metaRows: CitationPopoverMetaRow[]
  doiLabel: string
  doiHref: string
  loading: boolean
  isSystemB: boolean
  loadingLabel: string
  showMetrics: boolean
  metrics: string[]
}

export function CitationPopoverMetaPanels({
  showMetaGrid,
  metaRows,
  doiLabel,
  doiHref,
  loading,
  isSystemB,
  loadingLabel,
  showMetrics,
  metrics,
}: CitationPopoverMetaPanelsProps) {
  return (
    <>
      {showMetaGrid ? (
        <div className="kb-cite-pop-meta-grid">
          {metaRows.map((item) => (
            <div key={item.label} className="kb-cite-pop-meta-item">
              <span className="kb-cite-pop-meta-label">{item.label}</span>
              <span className="kb-cite-pop-meta-value">{item.value}</span>
            </div>
          ))}
          {doiLabel ? (
            <div className="kb-cite-pop-meta-item">
              <span className="kb-cite-pop-meta-label">DOI</span>
              {doiHref ? (
                <a className="kb-cite-pop-meta-value kb-cite-pop-link" href={doiHref} rel="noreferrer" target="_blank">
                  {doiLabel}
                </a>
              ) : (
                <span className="kb-cite-pop-meta-value">{doiLabel}</span>
              )}
            </div>
          ) : null}
        </div>
      ) : null}
      {loading && !isSystemB ? <div className="kb-cite-pop-sub">{loadingLabel}</div> : null}
      {!loading && showMetrics ? (
        <div className="kb-cite-pop-metrics">
          {isSystemB && doiLabel ? (
            doiHref ? (
              <a className="kb-cite-pop-metric kb-cite-pop-link" href={doiHref} rel="noreferrer" target="_blank">
                DOI {doiLabel}
              </a>
            ) : (
              <span className="kb-cite-pop-metric">DOI {doiLabel}</span>
            )
          ) : null}
          {metrics.map((item) => (
            <span key={item} className="kb-cite-pop-metric">{item}</span>
          ))}
        </div>
      ) : null}
    </>
  )
}
