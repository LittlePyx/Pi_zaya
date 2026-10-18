import { copyLiteralEvidenceSelection } from './literalEvidenceClipboard'

export function LiteralEvidenceQuote({ children }: { children: string }) {
  return <blockquote style={{ whiteSpace: 'pre-wrap' }} onCopy={copyLiteralEvidenceSelection}>{children}</blockquote>
}
