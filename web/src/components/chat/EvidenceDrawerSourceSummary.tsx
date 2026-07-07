export function EvidenceDrawerSourceSummary({
  label,
  detail,
}: {
  label: string
  detail: string
}) {
  return (
    <section className="kb-evidence-source-summary" data-testid="evidence-source-summary">
      <div className="kb-evidence-source-label">{label}</div>
      <div className="kb-evidence-source-detail">{detail}</div>
    </section>
  )
}
