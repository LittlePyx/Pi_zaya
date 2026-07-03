export type CompactMetaItem = {
  key: string
  label: string
  value: string
  href?: string
  tone: string
}

interface CitationPopoverHeaderProps {
  isSystemB: boolean
  kindLabel: string
  badgeLabel: string
  title: string
  subtitle: string
  compactMetaItems: CompactMetaItem[]
  onClose: () => void
}

export function CitationPopoverHeader({
  isSystemB,
  kindLabel,
  badgeLabel,
  title,
  subtitle,
  compactMetaItems,
  onClose,
}: CitationPopoverHeaderProps) {
  const showCompactMeta = compactMetaItems.length > 0

  return (
    <div className="kb-cite-pop-head">
      <div className="kb-cite-pop-head-copy">
        <div className="kb-cite-pop-kicker">
          <span className="kb-cite-pop-kind">{kindLabel}</span>
          <span className="kb-cite-pop-badge">{badgeLabel}</span>
        </div>
        <div className="kb-cite-pop-title">{title}</div>
        {subtitle ? <div className="kb-cite-pop-title-sub">{subtitle}</div> : null}
        {showCompactMeta ? (
          <div
            className="kb-cite-pop-compact-meta"
            data-testid={isSystemB ? 'citation-popover-system-b-compact-meta' : 'citation-popover-system-a-compact-meta'}
          >
            {compactMetaItems.map((item) => (
              item.href ? (
                <a
                  className={`kb-cite-pop-compact-pill kb-cite-pop-compact-${item.tone} kb-cite-pop-link`}
                  data-compact-key={item.key}
                  href={item.href}
                  key={item.key}
                  rel="noreferrer"
                  target="_blank"
                  title={item.value}
                >
                  {item.label ? <><span className="kb-cite-pop-compact-label">{item.label}</span>{' '}</> : null}
                  <span className="kb-cite-pop-compact-value">{item.value}</span>
                </a>
              ) : (
                <span
                  className={`kb-cite-pop-compact-pill kb-cite-pop-compact-${item.tone}`}
                  data-compact-key={item.key}
                  key={item.key}
                  title={item.value}
                >
                  {item.label ? <><span className="kb-cite-pop-compact-label">{item.label}</span>{' '}</> : null}
                  <span className="kb-cite-pop-compact-value">{item.value}</span>
                </span>
              )
            ))}
          </div>
        ) : null}
      </div>
      <button className="kb-cite-pop-close" onClick={onClose} type="button" aria-label="Close">
        ×
      </button>
    </div>
  )
}
