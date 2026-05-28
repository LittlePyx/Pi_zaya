import { expect, type Page } from '@playwright/test'

interface CitationShelfQualityOptions {
  minItems?: number
  maxReviewItems?: number
  minDoiLinks?: number
  minSourceOpenButtons?: number
  requireMetadataReady?: boolean
}

export async function expectCitationShelfQuality(
  page: Page,
  options: CitationShelfQualityOptions = {},
) {
  const minItems = options.minItems ?? 1
  const shelf = page.getByTestId('citation-shelf')
  await expect(shelf).toHaveClass(/translate-x-0/)

  const issues = await shelf.evaluate((node, qualityOptions) => {
    const root = node as HTMLElement
    const minItemsValue = qualityOptions.minItems ?? 1
    const badPhrases = [
      '[[CITE:',
      '```',
      '## ',
      'No summary available',
      'This hit is directly relevant',
      'The paper cites',
      'This is stated in',
      'has attrac',
    ]
    const reviewPhrases = [
      'Missing DOI',
      'Missing author',
      'Missing venue',
      '缺 DOI',
      '缺作者',
      '缺期刊',
      '元数据待',
      '待复查',
      '待核对',
    ]
    const out: string[] = []
    const textOf = (el: Element | null) => String(el?.textContent || '').replace(/\s+/g, ' ').trim()
    const hasBadPhrase = (value: string) => badPhrases.find((phrase) => value.includes(phrase)) || ''
    const hasReviewPhrase = (value: string) => reviewPhrases.find((phrase) => value.includes(phrase)) || ''
    const rect = root.getBoundingClientRect()
    if (rect.left < -1) out.push(`shelf overflows left: ${rect.left}`)
    if (rect.right > window.innerWidth + 1) out.push(`shelf overflows right: ${rect.right}/${window.innerWidth}`)
    if (rect.top < -1) out.push(`shelf overflows top: ${rect.top}`)
    if (rect.bottom > window.innerHeight + 1) out.push(`shelf overflows bottom: ${rect.bottom}/${window.innerHeight}`)

    const items = Array.from(root.querySelectorAll('[data-testid="citation-shelf-item"]')) as HTMLElement[]
    if (items.length < minItemsValue) out.push(`expected at least ${minItemsValue} shelf items, got ${items.length}`)

    const overflowSelectors = [
      '.kb-shelf-title',
      '.kb-shelf-count',
      '.kb-shelf-snapshot-row .ant-btn',
      '.kb-shelf-toolbar-main .ant-btn',
      '.kb-shelf-advanced-toggle',
      '.kb-shelf-item-title',
      '.kb-shelf-item-source',
      '.kb-shelf-source-open',
      '.kb-shelf-quality-chip',
      '.kb-shelf-repair-btn',
      '.kb-shelf-summary-text',
      '.kb-shelf-summary-source',
      '.kb-shelf-summary-quality',
      '.kb-shelf-doi',
      '.kb-shelf-readiness-status',
      '.kb-shelf-readiness-count',
      '.kb-shelf-readiness-source',
      '.kb-shelf-readiness-chip',
    ]
    for (const el of Array.from(root.querySelectorAll(overflowSelectors.join(','))) as HTMLElement[]) {
      if (!el.offsetParent && getComputedStyle(el).position !== 'fixed') continue
      if (el.scrollWidth > el.clientWidth + 2) {
        out.push(`inline overflow in .${String(el.className || '').replace(/\s+/g, '.')}: ${textOf(el).slice(0, 80)}`)
      }
    }

    items.forEach((item, index) => {
      const title = textOf(item.querySelector('[data-testid="citation-shelf-item-title"]'))
      const source = textOf(item.querySelector('[data-testid="citation-shelf-item-source"]'))
      const doi = textOf(item.querySelector('.kb-shelf-doi'))
      if (title.length < 8) out.push(`item ${index + 1} title too short: ${title}`)
      const badTitlePhrase = hasBadPhrase(title)
      if (badTitlePhrase) out.push(`item ${index + 1} title contains ${badTitlePhrase}`)
      if (!source && !doi) out.push(`item ${index + 1} lacks source or DOI line`)
      const reviewText = textOf(item.querySelector('.kb-shelf-quality-chips'))
      const reviewPhrase = hasReviewPhrase(reviewText)
      if (qualityOptions.maxReviewItems === 0 && reviewPhrase) {
        out.push(`item ${index + 1} still shows metadata review chip: ${reviewText}`)
      }
      const itemText = textOf(item)
      const badItemPhrase = hasBadPhrase(itemText)
      if (badItemPhrase) out.push(`item ${index + 1} contains ${badItemPhrase}`)
    })

    const doiLinks = Array.from(root.querySelectorAll('.kb-shelf-doi-link')) as HTMLElement[]
    const minDoiLinks = qualityOptions.minDoiLinks ?? 0
    if (doiLinks.length < minDoiLinks) out.push(`expected at least ${minDoiLinks} DOI links, got ${doiLinks.length}`)
    const sourceOpenButtons = Array.from(root.querySelectorAll('[data-testid="citation-shelf-open-source"]')) as HTMLElement[]
    const minSourceOpenButtons = qualityOptions.minSourceOpenButtons ?? 0
    if (sourceOpenButtons.length < minSourceOpenButtons) {
      out.push(`expected at least ${minSourceOpenButtons} source-open buttons, got ${sourceOpenButtons.length}`)
    }

    const readiness = root.querySelector('[data-testid="citation-shelf-readiness"]')
    const readinessText = textOf(readiness)
    if (items.length > 0 && !readiness) out.push('shelf readiness strip is missing')
    else if (items.length > 0 && readinessText.length < 12) out.push(`shelf readiness strip too thin: ${readinessText}`)
    if (qualityOptions.requireMetadataReady && items.length > 0 && !readinessText.includes(`${items.length}/${items.length}`)) {
      out.push(`shelf readiness is not fully metadata-ready: ${readinessText}`)
    }

    const summary = root.querySelector('[data-testid="citation-shelf-summary"]')
    const summaryText = textOf(summary)
    if (!summary) out.push('focused item does not show a summary panel')
    else if (summaryText.length < 24) out.push(`focused summary too short: ${summaryText}`)
    const summarySource = textOf(root.querySelector('.kb-shelf-summary-source'))
    if (summary && summarySource.length < 2) out.push('focused summary lacks source chip')

    return out
  }, { ...options, minItems })

  expect(issues).toEqual([])
}
