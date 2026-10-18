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
  const rec = node as { value?: unknown; children?: unknown[] }
  const own = typeof rec.value === 'string' ? rec.value : ''
  const childText = Array.isArray(rec.children)
    ? rec.children.map((child) => readerNodeText(child)).filter(Boolean).join(' ')
    : ''
  return `${own} ${childText}`.replace(/\s+/g, ' ').trim()
}

export function mixedEquationProse(text: string): string {
  const boundary = text.indexOf('$$')
  if (boundary < 0) return ''
  return text.slice(0, boundary).replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\s+/g, ' ').trim()
}

export function mixedProseText(text: string): string {
  return text.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\s+/g, ' ').trim()
}

export function mixedEquationProseMatches(
  node: unknown, item: { kind: string; token: { text?: string }; rawSource?: string; lineStart: number; lineEnd: number },
): boolean {
  if (item.kind !== 'equation') return false
  const range = readerNodeLineRange(node)
  if (!range || range.start < item.lineStart || range.end > item.lineEnd) return false
  const source = String(item.rawSource || item.token.text || '')
  const prefix = mixedEquationProse(source)
  const visible = readerNodeText(node)
  const equation = source.match(/\$\$([\s\S]*?)\$\$/)?.[1].replace(/\s+/g, ' ').trim() || ''
  return prefix.length >= 40 && (visible === prefix || (visible.startsWith(prefix) && equation.length >= 8 && visible.includes(equation)))
}
