export type RefCardCopyLocale = 'zh' | 'en'

interface SelectRefRelevanceTextOptions {
  cardText?: unknown
  explicitTexts?: unknown[]
  evidenceTexts?: unknown[]
  locale?: RefCardCopyLocale
}

interface SelectLocalizedRefCardTextOptions {
  cardText?: unknown
  explicitTexts?: unknown[]
  locale: RefCardCopyLocale
}

function compactText(value: unknown): string {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function comparisonText(value: unknown): string {
  return compactText(value)
    .toLowerCase()
    .replace(/[\u2018\u2019\u201c\u201d"'`]/g, '')
    .replace(/[\u2026.。!！?？,，;；:：()（）[\]{}]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function textMatchesLocale(value: string, locale: RefCardCopyLocale): boolean {
  const cjkCount = (value.match(/[\u3400-\u9fff]/g) || []).length
  const latinCount = (value.match(/[A-Za-z]/g) || []).length
  if (locale === 'zh') {
    return cjkCount >= 4 && (cjkCount >= 12 || cjkCount * 2 >= latinCount)
  }
  return latinCount >= 4 && (cjkCount === 0 || latinCount >= Math.max(8, cjkCount * 2))
}

export function selectLocalizedRefCardText({
  cardText,
  explicitTexts = [],
  locale,
}: SelectLocalizedRefCardTextOptions): string {
  const seen = new Set<string>()
  for (const raw of [cardText, ...explicitTexts]) {
    const text = compactText(raw)
    const key = comparisonText(text)
    if (!text || !key || seen.has(key)) continue
    seen.add(key)
    if (textMatchesLocale(text, locale)) return text
  }
  return ''
}

function duplicatesEvidence(value: string, evidenceTexts: unknown[]): boolean {
  const candidate = comparisonText(value)
  if (candidate.length < 12) return false
  for (const rawEvidence of evidenceTexts) {
    const evidence = comparisonText(rawEvidence)
    if (evidence.length < 12) continue
    if (candidate === evidence) return true
    const shorter = candidate.length <= evidence.length ? candidate : evidence
    const longer = candidate.length <= evidence.length ? evidence : candidate
    if (shorter.length >= 36 && longer.startsWith(shorter)) return true
    if (shorter.length >= 56 && longer.includes(shorter)) return true
  }
  return false
}

/**
 * Picks copy for the "why this is relevant" panel without ever treating a
 * source excerpt as a relevance explanation. A card-view section remains the
 * preferred contract, but localized explicit fields can repair a missing or
 * wrong-language section. A requested locale is a hard display boundary: if
 * no safe candidate matches it, render no panel instead of mixing languages.
 */
export function selectRefRelevanceText({
  cardText,
  explicitTexts = [],
  evidenceTexts = [],
  locale,
}: SelectRefRelevanceTextOptions): string {
  const candidates: string[] = []
  const seen = new Set<string>()
  for (const raw of [cardText, ...explicitTexts]) {
    const text = compactText(raw)
    const key = comparisonText(text)
    if (!text || !key || seen.has(key) || duplicatesEvidence(text, evidenceTexts)) continue
    seen.add(key)
    candidates.push(text)
  }
  if (!candidates.length) return ''
  if (locale) {
    const localized = candidates.find((text) => textMatchesLocale(text, locale))
    return localized || ''
  }
  return candidates[0]
}
