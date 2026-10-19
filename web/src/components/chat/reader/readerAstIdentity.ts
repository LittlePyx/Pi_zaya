export function readerNodeLineRange(node: unknown): { start: number; end: number } | null {
  const rec = (node || {}) as { position?: { start?: { line?: number }; end?: { line?: number } } }
  const start = Number(rec.position?.start?.line || 0)
  const endRaw = Number(rec.position?.end?.line || 0)
  if (!Number.isFinite(start) || start <= 0) return null
  const end = Number.isFinite(endRaw) && endRaw > 0 ? Math.max(start, endRaw) : start
  return { start: Math.floor(start), end: Math.floor(end) }
}

export function readerNodeText(node: unknown): string {
  if (!node || typeof node !== 'object') return ''
  const rec = node as { value?: unknown; children?: unknown[]; tagName?: string; properties?: { className?: string[]; encoding?: string } }
  if (rec.properties?.className?.includes('katex')) {
    const annotation = readerMathAnnotation(node)
    if (annotation !== null) return annotation
  }
  const own = typeof rec.value === 'string' ? rec.value : ''
  const childText = Array.isArray(rec.children)
    ? rec.children.map((child) => readerNodeText(child)).filter(Boolean).join(' ')
    : ''
  return `${own} ${childText}`.replace(/\s+/g, ' ').trim()
}

function readerMathAnnotation(node: unknown): string | null {
  if (!node || typeof node !== 'object') return null
  const rec = node as { tagName?: string; properties?: { encoding?: string }; children?: unknown[] }
  if (rec.tagName === 'annotation' && rec.properties?.encoding === 'application/x-tex') return readerNodeText({ children: rec.children })
  for (const child of rec.children || []) {
    const found = readerMathAnnotation(child)
    if (found !== null) return found
  }
  return null
}

export function mixedEquationProse(text: string): string {
  const boundary = text.indexOf('$$')
  if (boundary < 0) return ''
  return mixedProseText(text.slice(0, boundary))
}

export function mixedProseText(text: string): string {
  return text.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/(?<!\\)\$([^$\n]+)\$/g, '$1').replace(/\s+/g, ' ').replace(/\s+([,.;:])/g, '$1').trim()
}

export function mixedEquationProseMatches(
  node: unknown, item: { kind: string; token: { text?: string }; rawSource?: string; lineStart: number; lineEnd: number },
): boolean {
  if (item.kind !== 'equation') return false
  const range = readerNodeLineRange(node)
  if (!range || range.start < item.lineStart || range.end > item.lineEnd) return false
  const source = String(item.rawSource || item.token.text || '')
  const prefix = mixedEquationProse(source)
  const visible = mixedProseText(readerNodeText(node))
  const equation = source.match(/\$\$([\s\S]*?)\$\$/)?.[1].replace(/\s+/g, ' ').trim() || ''
  return prefix.length >= 40 && (visible === prefix || (visible.startsWith(prefix) && equation.length >= 8 && visible.includes(equation)))
}
