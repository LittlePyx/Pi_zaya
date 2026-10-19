import type { ReaderDocBlock } from '../../../api/references'
import { mixedEquationProse, mixedProseText } from './readerAstIdentity'

function sourceProseText(node: HTMLElement): string {
  const clone = node.cloneNode(true) as HTMLElement
  for (const math of clone.querySelectorAll('.katex')) {
    const source = math.querySelector('annotation[encoding="application/x-tex"]')
    if (!source) return ''
    math.replaceWith(document.createTextNode(source.textContent || ''))
  }
  return mixedProseText(String(clone.textContent || ''))
}

export function mixedProseHighlightSeed(target: HTMLElement, quote: string): string {
  return target.getAttribute('data-kb-source-prose') === '1'
    ? mixedProseText(quote.split('$$')[0])
    : quote
}

export function resolveMixedEquationProseTarget(root: HTMLElement, block: ReaderDocBlock | null, quote: string): HTMLElement | null {
  if (!block || block.kind !== 'equation' || !block.block_id || !block.anchor_id) return null
  const prefix = mixedEquationProse(String(block.raw_text || block.text || ''))
  const requested = mixedProseText(quote.split('$$')[0])
  if (prefix.length < 40 || requested.length < 40 || (!prefix.includes(requested) && !requested.includes(prefix))) return null
  const candidates = Array.from(root.querySelectorAll<HTMLElement>('[data-kb-source-prose="1"]')).filter((node) => (
    node.getAttribute('data-kb-block-id') === block.block_id
    && node.getAttribute('data-kb-anchor-id') === block.anchor_id
    && sourceProseText(node).startsWith(prefix)
  ))
  return candidates.length === 1 ? candidates[0] : null
}
