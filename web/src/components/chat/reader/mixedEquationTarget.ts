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

function sourceDisplayMathMatches(root: HTMLElement, blockId: string, anchorId: string, quote: string): boolean {
  const mathKey = (text: string) => text.replace(/\s+/g, ' ').trim()
  const requested = Array.from(quote.matchAll(/\$\$([\s\S]*?)\$\$/g), (match) => mathKey(match[1]))
  if (!requested.length) return !quote.includes('$$')
  const nodes = Array.from(root.querySelectorAll<HTMLElement>('.katex-display')).filter((node) => (
    node.getAttribute('data-kb-block-id') === blockId && node.getAttribute('data-kb-anchor-id') === anchorId
  ))
  const actual = nodes.map((node) => mathKey(node.querySelector('annotation[encoding="application/x-tex"]')?.textContent || ''))
  return actual.length === requested.length && actual.every((text, index) => Boolean(text) && text === requested[index])
}

export function mixedProseHighlightSeed(target: HTMLElement, quote: string, root: HTMLElement = target.parentElement || target): string {
  if (target.getAttribute('data-kb-source-prose') !== '1') return quote
  const requested = mixedProseText(quote.split('$$')[0])
  const source = sourceProseText(target)
  if (requested.length < 40 || (!source.startsWith(requested) && !requested.startsWith(source))) return ''
  if (!sourceDisplayMathMatches(root, target.getAttribute('data-kb-block-id') || '', target.getAttribute('data-kb-anchor-id') || '', quote)) return ''
  // The entire source prefix and supplied display math were checked above.
  // Highlight a literal prose run, rather than searching visible glyphs for TeX.
  return quote.split('$$')[0].split(/(?<!\\)\$[^$\n]+\$/g).map(mixedProseText)
    .find((text) => text.length >= 40 && source.includes(text)) || ''
}

export function resolveMixedEquationProseTarget(root: HTMLElement, block: ReaderDocBlock | null, quote: string): HTMLElement | null {
  if (!block || block.kind !== 'equation' || !block.block_id || !block.anchor_id) return null
  const prefix = mixedEquationProse(String(block.raw_text || block.text || ''))
  const requested = mixedProseText(quote.split('$$')[0])
  if (prefix.length < 40 || requested.length < 40 || (!prefix.includes(requested) && !requested.includes(prefix))) return null
  const mathKey = (text: string) => Array.from(text.matchAll(/\$\$([\s\S]*?)\$\$/g), (match) => match[1].replace(/\s+/g, ' ').trim())
  const supplied = mathKey(quote); const original = mathKey(String(block.raw_text || block.text || ''))
  if (supplied.length && (supplied.length !== original.length || supplied.some((text, index) => text !== original[index]))) return null
  if (!sourceDisplayMathMatches(root, block.block_id, block.anchor_id, quote)) return null
  const candidates = Array.from(root.querySelectorAll<HTMLElement>('[data-kb-source-prose="1"]')).filter((node) => (
    node.getAttribute('data-kb-block-id') === block.block_id
    && node.getAttribute('data-kb-anchor-id') === block.anchor_id
    && sourceProseText(node).startsWith(prefix)
  ))
  return candidates.length === 1 ? candidates[0] : null
}
