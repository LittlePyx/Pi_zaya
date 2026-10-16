// Keep source notation literal while removing presentation Markdown. Match the
// backend evidence_math_display rules; no mathematical completion is performed.
const DISPLAY_MATH = /\$\$[^$]{1,1200}\$\$|(?<![\\$])\$(?!\$)[^$\n]{1,1200}\$(?!\$)|\\\([^\n]{1,1200}?\\\)|\\\[[^\n]{1,1200}?\\\]|\|\|[^|\n]{1,160}\|\||\b[A-Za-z]{1,3}\s*(?:\([^()\n]*\|[^()\n]*\)|\[[^[\]\n]*\|[^[\]\n]*\])|(?<![\w|])\|[^|\s]{1,120}\|(?![\w|])/g

export function cleanCitationDisplayText(value: string): string {
  const literals: string[] = []
  let prefix = 'KBMATHLITERAL'
  while (String(value || '').includes(prefix)) prefix += 'X'
  const text = String(value || '')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/(?:\$\s*)?\^\{\s*\[[\d,\-\s;]+\]\s*\}(?:\s*\$)?/g, ' ')
    .replace(/\\textsuperscript\{\s*\[[^\]\n]{1,80}\]\s*\}/gi, ' ')
    .replace(/\\(?:cite|citep|citet|citealp|upcite)\s*\{[^}\n]{1,200}\}/gi, ' ')
    .replace(/\[\[?\s*(?:CITE|SUPPORT|SID)\s*:[^\]\n]{1,200}\]?\]?/gi, ' ')
    .replace(DISPLAY_MATH, literal => {
      if (/^\$\s*[A-Za-z0-9]+\s*\$$/.test(literal)) return literal
      if (literal.startsWith('|') && !/[A-Za-z0-9\\]/.test(literal)) return literal
      if (literal.startsWith('||') || (literal.includes('|') && !/^[\u0024\\|]/.test(literal))) {
        const operands = literal.replace(/\\[A-Za-z]+|[_^]\{[^}]*\}/g, '')
        if (/[A-Za-z]{2,}/.test(operands)) return literal
      }
      literals.push(literal)
      return `${prefix}${literals.length - 1}END`
    })
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/^\s{0,3}[-*+]\s+/gm, '')
    .replace(/^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$/gm, ' ')
    .replace(/^\s*\|/gm, '')
    .replace(/\|\s*$/gm, '')
    .replace(/\s*\|\s*/g, ' ')
    .replace(/\$([^$\n]{1,160})\$/g, '$1')
    .replace(/!\[[^\]]*]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)]\([^)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/~~([^~]+)~~/g, '$1')
    .replace(/\\(?=\s|[,;])/g, ' ')
    .replace(/(^|\s)#{1,6}\s+/g, ' ')
    .replace(/\s*\|\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.;:!?，。；：！？])/g, '$1')
    .trim()
    .replace(/^(?:\.{2,}|\u2026)+\s*/, '')
  return text.replace(new RegExp(`${prefix}(\\d+)END`, 'g'), (_, index: string) => literals[Number(index)])
}
