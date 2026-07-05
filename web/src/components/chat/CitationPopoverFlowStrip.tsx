interface CitationPopoverFlowStripProps {
  explainText: string
  flowSteps: string[]
  flowAriaLabel: string
}

export function CitationPopoverFlowStrip({
  explainText,
  flowSteps,
  flowAriaLabel,
}: CitationPopoverFlowStripProps) {
  return (
    <>
      {explainText ? <div className="kb-cite-pop-explain" data-testid="citation-popover-explain">{explainText}</div> : null}
      {flowSteps.length > 0 ? (
        <div className="kb-cite-pop-flow" data-testid="citation-popover-flow" aria-label={flowAriaLabel}>
          {flowSteps.map((step, index) => (
            <div className="kb-cite-pop-flow-piece" key={step}>
              <span className="kb-cite-pop-flow-step">{step}</span>
              {index < flowSteps.length - 1 ? <span className="kb-cite-pop-flow-arrow">→</span> : null}
            </div>
          ))}
        </div>
      ) : null}
    </>
  )
}
