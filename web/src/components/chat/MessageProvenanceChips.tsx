import { compactHeadingPath } from './messageQuoteUtils'
import { shortSegmentLabel, type ProvenanceLocateEntry } from './reader/messageStructuredProvenance'

interface MessageProvenanceChipsProps {
  entries: ProvenanceLocateEntry[]
  messageId: number
  onOpenEntry: (entry: ProvenanceLocateEntry, snippet: string) => void
}

export function MessageProvenanceChips({
  entries,
  messageId,
  onOpenEntry,
}: MessageProvenanceChipsProps) {
  if (entries.length <= 0) return null

  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {entries.map((entry, idx) => {
        const heading = String(entry.primary?.headingPath || '').trim()
        const label = String(entry.label || '').trim()
        const snippet = shortSegmentLabel(
          String(entry.anchorText || entry.evidenceQuote || entry.segmentText || label || ''),
          72,
        )
        const headingLite = compactHeadingPath(heading, 56)
        const text = snippet
          || label
          || headingLite
          || '\u539f\u6587\u8bc1\u636e'
        const seedSnippet = String(
          entry.evidenceQuote
          || entry.anchorText
          || entry.segmentText
          || entry.label
          || '',
        ).trim()
        const focusSnippet = String(entry.primary?.focusSnippet || entry.primary?.matchText || seedSnippet || '').trim()
        return (
          <button
            key={`${messageId}::prov::${String(entry.segmentId || idx)}::${idx}`}
            type="button"
            className="kb-prov-locate-chip"
            aria-label={'\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e'}
            title={heading
              ? `\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e\uff1a${heading}`
              : (headingLite ? `\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e\uff1a${headingLite}` : '\u5b9a\u4f4d\u5230\u539f\u6587\u8bc1\u636e')}
            data-kb-locate-focus={focusSnippet.slice(0, 220)}
            data-kb-locate-block-id={String(entry.primary?.blockId || '').trim()}
            data-kb-locate-anchor-id={String(entry.primary?.anchorId || '').trim()}
            data-kb-locate-heading={String(entry.primary?.headingPath || '').trim()}
            onClick={() => onOpenEntry(entry, seedSnippet)}
          >
            <span className="kb-prov-locate-chip-num">{`\u8bc1\u636e${idx + 1}`}</span>
            <span className="kb-prov-locate-chip-text">{text}</span>
          </button>
        )
      })}
    </div>
  )
}
