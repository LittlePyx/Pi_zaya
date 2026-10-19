import { createRoot } from 'react-dom/client'
import { MarkdownRenderer } from '../components/chat/MarkdownRenderer'
import type { ReaderDocBlock } from '../api/references'

export function renderMixedInlineMathSource(host: HTMLElement, source: string, block: ReaderDocBlock) {
  const root = createRoot(host)
  root.render(<MarkdownRenderer content={source} variant="reader" readerBlocks={[block]} />)
  return () => root.unmount()
}
