import type { CiteDetail, CitationCardViewSection } from './citationState'
import {
  citationDisplay,
  citationInlineLabel,
  citeMetricSummary,
} from './citationState'
import type { CompactMetaItem } from './CitationPopoverHeader'
import {
  anchorKindLabel,
  compact,
  isOnlyPaperLabel,
  pageRangeLabel,
  stripLocationIdentityPrefix,
  substantiallySame,
} from './citationPopoverUtils'

interface FrameStrings extends Record<string, string> {
  cite_anchor_equation: string
  cite_anchor_figure: string
  cite_anchor_label: string
  cite_anchor_paragraph: string
  cite_anchor_sentence: string
  cite_anchor_table: string
  cite_kind_evidence: string
  cite_kind_upstream: string
  cite_meta_published: string
  cite_meta_source: string
  cite_open_evidence: string
  cite_position: string
  cite_read_locate: string
  cite_upstream_reference: string
}

interface BuildCitationPopoverFrameModelOptions {
  detail: CiteDetail
  S: FrameStrings
  isSystemB: boolean
  viewHeader: {
    kicker: string
    title: string
    subtitle: string
  }
  locatorSection?: CitationCardViewSection
  cardLocatorLabel: string
  localizeKnownLabel: (value: string) => string
}

export interface CitationPopoverMetaRow {
  label: string
  value: string
}

export interface CitationPopoverFrameModel {
  canOpenReader: boolean
  compactMetaItems: CompactMetaItem[]
  displayMain: string
  displaySource: string
  doiHref: string
  doiLabel: string
  flowSteps: string[]
  headingPath: string
  kindLabel: string
  metaRows: CitationPopoverMetaRow[]
  metrics: string[]
  pageLabel: string
  primaryActionLabel: string
  showMetaGrid: boolean
  showMetrics: boolean
  sourcePaperText: string
  systemATitle: string
  systemBTitle: string
  systemBTitleMissing: boolean
  headerSubtitle: string
  badgeLabel: string
}

export function buildCitationPopoverFrameModel({
  detail,
  S,
  isSystemB,
  viewHeader,
  locatorSection,
  cardLocatorLabel,
  localizeKnownLabel,
}: BuildCitationPopoverFrameModelOptions): CitationPopoverFrameModel {
  const display = citationDisplay(detail)
  const doiLabel = compact(detail.doi) || compact(detail.doiUrl)
  const doiHref = compact(detail.doiUrl) || (doiLabel ? `https://doi.org/${doiLabel}` : '')
  const metrics = citeMetricSummary(detail)
  const inlineLabel = citationInlineLabel(detail, { includeSource: false })
  const canOpenReader = Boolean(compact(detail.sourcePath))
  const kindLabel = localizeKnownLabel(viewHeader.kicker) || (isSystemB ? S.cite_kind_upstream : S.cite_kind_evidence)
  const systemADisplayNumSeeds = [
    ...(Array.isArray(detail.displayNums) ? detail.displayNums : []),
    detail.displayNum,
  ].map((num) => Number(num || 0)).filter((num) => Number.isFinite(num) && num > 0)
  const displayNums = Array.from(new Set(
    (isSystemB
      ? [
          ...(Array.isArray(detail.linkedNums) ? detail.linkedNums : []),
          detail.num,
        ]
      : (systemADisplayNumSeeds.length > 0 ? systemADisplayNumSeeds : [detail.num]))
      .map((num) => Number(num || 0))
      .filter((num) => Number.isFinite(num) && num > 0),
  )).sort((a, b) => a - b)
  const badgeNumText = displayNums.length > 1 ? displayNums.join('/') : String(displayNums[0] || '')
  const badgeLabel = badgeNumText ? (isSystemB ? `[R${badgeNumText}]` : `#${badgeNumText}`) : inlineLabel
  const headingPath = compact(detail.headingPath) || (!isSystemB ? compact(detail.title) : '')
  const pageLabel = pageRangeLabel(detail.pageStart, detail.pageEnd)
  const sourcePaperText = compact(detail.sourceName) || compact(display.source)
  const cardTitle = compact(viewHeader.title) || compact(detail.cardTitle)
  const cardSubtitle = compact(viewHeader.subtitle) || compact(detail.cardSubtitle)
  const rawHeaderSubtitle = isSystemB ? cardSubtitle : ''
  const displayMain = compact(display.main)
  const systemATitle = cardTitle || ((displayMain && displayMain !== headingPath)
    ? displayMain
    : (compact(detail.sourceName) || compact(display.source) || displayMain))
  const systemBTitleMissing = !cardTitle && !compact(detail.title)
  const systemBTitle = cardTitle || compact(detail.title) || S.cite_upstream_reference
  const headerSubtitle = !isSystemB && rawHeaderSubtitle && !substantiallySame(rawHeaderSubtitle, systemBTitle)
    ? rawHeaderSubtitle
    : ''
  const systemASub = [headingPath, pageLabel].filter(Boolean).join(' · ')
  const rawSystemALocationText = compact(locatorSection?.text || '') || compact(detail.cardLocator) || compact(detail.locationLabel) || systemASub || systemATitle
  const systemALocationText = stripLocationIdentityPrefix(rawSystemALocationText, [
    systemATitle,
    sourcePaperText,
    detail.sourceName,
    display.source,
  ]) || rawSystemALocationText
  const systemAAnchorText = anchorKindLabel(detail.anchorKind, {
    sentence: S.cite_anchor_sentence,
    paragraph: S.cite_anchor_paragraph,
    equation: S.cite_anchor_equation,
    figure: S.cite_anchor_figure,
    table: S.cite_anchor_table,
  })
  const cardFlow = Array.isArray(detail.cardFlow)
    ? detail.cardFlow.map((item) => compact(item)).filter(Boolean)
    : []
  const primaryActionLabel = isSystemB ? S.cite_read_locate : S.cite_open_evidence
  const flowSteps = isSystemB ? [] : cardFlow
  const systemAMetaSource = display.source && !isOnlyPaperLabel(display.source, [systemATitle, sourcePaperText])
    ? display.source
    : ''
  const metaRows = [
    systemAMetaSource ? { label: S.cite_meta_source, value: systemAMetaSource } : null,
    display.venueYear ? { label: S.cite_meta_published, value: display.venueYear } : null,
  ].filter(Boolean) as CitationPopoverMetaRow[]
  const systemACompactMetaItems = !isSystemB
    ? ([
        systemALocationText ? {
          key: 'location',
          label: cardLocatorLabel || S.cite_position,
          value: systemALocationText,
          tone: 'location',
        } : null,
        systemAAnchorText ? {
          key: 'anchor',
          label: S.cite_anchor_label,
          value: systemAAnchorText,
          tone: 'muted',
        } : null,
        ...metaRows.map((item) => ({
          key: `meta-${item.label}`,
          label: item.label,
          value: item.value,
          tone: 'muted',
        })),
        doiLabel ? {
          key: 'doi',
          label: 'DOI',
          value: doiLabel,
          href: doiHref,
          tone: 'doi',
        } : null,
        ...metrics.map((item) => ({
          key: `metric-${item}`,
          label: '',
          value: item,
          tone: 'metric',
        })),
      ].filter(Boolean) as CompactMetaItem[])
    : []
  const systemBCompactMetaItems = isSystemB
    ? ([
        display.authors ? {
          key: 'authors',
          label: '',
          value: display.authors,
          tone: 'muted',
        } : null,
        display.venueYear ? {
          key: 'published',
          label: '',
          value: display.venueYear,
          tone: 'muted',
        } : null,
        doiLabel ? {
          key: 'doi',
          label: 'DOI',
          value: doiLabel,
          href: doiHref,
          tone: 'doi',
        } : null,
        ...metrics.map((item) => ({
          key: `metric-${item}`,
          label: '',
          value: item,
          tone: 'metric',
        })),
      ].filter(Boolean) as CompactMetaItem[])
    : []

  return {
    canOpenReader,
    compactMetaItems: isSystemB ? systemBCompactMetaItems : systemACompactMetaItems,
    displayMain,
    displaySource: display.source,
    doiHref,
    doiLabel,
    flowSteps,
    headingPath,
    kindLabel,
    metaRows,
    metrics,
    pageLabel,
    primaryActionLabel,
    showMetaGrid: false,
    showMetrics: false,
    sourcePaperText,
    systemATitle,
    systemBTitle,
    systemBTitleMissing,
    headerSubtitle,
    badgeLabel,
  }
}
