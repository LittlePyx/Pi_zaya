import type { Message, ResearchNoteSourceLink } from '../../api/chat'
import type { CiteShelfItem } from './citationState'
import {
  getMessageCiteDetailRecords,
  getMessageCopyMarkdownValue,
} from './messageRenderPacket'

export interface ResearchNoteLabels {
  question: string
  answer: string
  answerSources: string
  bibliography: string
  researchBasket: string
  authors: string
  venue: string
  year: string
  location: string
  evidence: string
  excerpt: string
  note: string
  summary: string
  source: string
  untitledSource: string
}

export interface ResearchNoteCitation {
  num: number
  title: string
  sourceName: string
  sourcePath: string
  authors: string
  venue: string
  year: string
  doi: string
  doiUrl: string
  citeFmt: string
  headingPath: string
  locationLabel: string
  evidenceQuote: string
  pageStart: number
  pageEnd: number
  blockId: string
  anchorId: string
  anchorKind: string
}

export interface ResearchNoteAnswer {
  messageId: number
  question: string
  answerMarkdown: string
  citations: ResearchNoteCitation[]
}

function text(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  return ''
}

function numberValue(value: unknown): number {
  const num = Number(value || 0)
  return Number.isFinite(num) && num > 0 ? Math.floor(num) : 0
}

function field(record: Record<string, unknown>, camel: string, snake?: string): string {
  return text(record[camel] ?? (snake ? record[snake] : undefined))
}

function sourceTitleFromPath(value: string): string {
  return text(value)
    .split(/[\\/]/)
    .pop()
    ?.replace(/\.(?:pdf|md)$/i, '')
    .replace(/\.en$/i, '')
    .replace(/[_-]+/g, ' ')
    .trim() || ''
}

function normalizeDoi(value: string): string {
  return text(value)
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
    .replace(/^doi\s*:\s*/i, '')
    .trim()
}

function preserveLinkedCitationBrackets(markdown: string): string {
  return String(markdown || '').replace(
    /\[(R?\d+)]\((https?:\/\/(?:dx\.)?doi\.org\/[^)\s]+|#[^)\s]+)\)/gi,
    '[[$1]]($2)',
  )
}

function citationFromRecord(record: Record<string, unknown>, fallbackNum: number): ResearchNoteCitation {
  const doi = normalizeDoi(field(record, 'doi'))
  const sourceName = field(record, 'source_name', 'sourceName') || field(record, 'sourceName', 'source_name')
  const sourcePath = field(record, 'source_path', 'sourcePath') || field(record, 'sourcePath', 'source_path')
  return {
    num: numberValue(record.num ?? record.displayNum) || fallbackNum,
    title: field(record, 'title') || field(record, 'card_title', 'cardTitle') || sourceTitleFromPath(sourceName || sourcePath),
    sourceName: sourceTitleFromPath(sourceName || sourcePath),
    sourcePath,
    authors: field(record, 'authors'),
    venue: field(record, 'venue'),
    year: field(record, 'year'),
    doi,
    doiUrl: field(record, 'doi_url', 'doiUrl') || (doi ? `https://doi.org/${doi}` : ''),
    citeFmt: field(record, 'cite_fmt', 'citeFmt'),
    headingPath: field(record, 'heading_path', 'headingPath'),
    locationLabel: field(record, 'location_label', 'locationLabel'),
    evidenceQuote: field(record, 'evidence_quote', 'evidenceQuote') || field(record, 'card_evidence', 'cardEvidence'),
    pageStart: numberValue(record.page_start ?? record.pageStart),
    pageEnd: numberValue(record.page_end ?? record.pageEnd),
    blockId: field(record, 'block_id', 'blockId') || field(record, 'blockId', 'block_id'),
    anchorId: field(record, 'anchor_id', 'anchorId') || field(record, 'anchorId', 'anchor_id'),
    anchorKind: field(record, 'anchor_kind', 'anchorKind') || field(record, 'anchorKind', 'anchor_kind'),
  }
}

function citationIdentity(citation: ResearchNoteCitation): string {
  if (citation.doi) return `doi:${citation.doi.toLowerCase()}`
  const title = citation.title.toLowerCase().replace(/\s+/g, ' ').trim()
  if (title) return `title:${title}`
  return `source:${citation.sourceName.toLowerCase()}|${citation.num}`
}

function uniqueCitations(citations: ResearchNoteCitation[]): ResearchNoteCitation[] {
  const seen = new Set<string>()
  return citations.filter((citation) => {
    const identity = `${citation.num}|${citationIdentity(citation)}`
    if (seen.has(identity)) return false
    seen.add(identity)
    return true
  })
}

export function buildResearchNoteAnswers(messages: Message[]): ResearchNoteAnswer[] {
  let latestQuestion = ''
  const answers: ResearchNoteAnswer[] = []
  for (const message of messages) {
    if (message.role === 'user') {
      latestQuestion = text(message.content)
      continue
    }
    if (message.role !== 'assistant') continue
    const answerMarkdown = preserveLinkedCitationBrackets(text(getMessageCopyMarkdownValue(message)))
    if (!answerMarkdown) continue
    const citations = uniqueCitations(
      getMessageCiteDetailRecords(message).map((record, index) => citationFromRecord(record, index + 1)),
    )
    answers.push({
      messageId: Number(message.id),
      question: latestQuestion,
      answerMarkdown,
      citations,
    })
  }
  return answers
}

function pageLabel(citation: ResearchNoteCitation): string {
  if (citation.pageStart <= 0) return ''
  return citation.pageEnd > citation.pageStart
    ? `p. ${citation.pageStart}–${citation.pageEnd}`
    : `p. ${citation.pageStart}`
}

function citationTitle(citation: ResearchNoteCitation, labels: ResearchNoteLabels): string {
  return citation.title || citation.sourceName || labels.untitledSource
}

function bibliographyLine(citation: ResearchNoteCitation, labels: ResearchNoteLabels): string {
  if (citation.citeFmt) {
    const withDoi = citation.doi && !citation.citeFmt.toLowerCase().includes(citation.doi.toLowerCase())
      ? `${citation.citeFmt} DOI: [${citation.doi}](${citation.doiUrl})`
      : citation.citeFmt
    return withDoi.replace(/^\s*\[?\d+\]?\s*[.)]?\s*/, '').trim()
  }
  const title = citationTitle(citation, labels)
  const authors = citation.authors ? `${citation.authors}. ` : ''
  const publication = [citation.venue, citation.year].filter(Boolean).join(', ')
  const doi = citation.doi ? ` DOI: [${citation.doi}](${citation.doiUrl}).` : ''
  return `${authors}${title}.${publication ? ` ${publication}.` : ''}${doi}`.trim()
}

function answerCitationMarkdown(citation: ResearchNoteCitation, labels: ResearchNoteLabels): string {
  const lines = [`#### [${citation.num}] ${citationTitle(citation, labels)}`]
  if (citation.authors) lines.push(`- ${labels.authors}: ${citation.authors}`)
  if (citation.venue) lines.push(`- ${labels.venue}: ${citation.venue}`)
  if (citation.year) lines.push(`- ${labels.year}: ${citation.year}`)
  if (citation.doi) lines.push(`- DOI: [${citation.doi}](${citation.doiUrl})`)
  const location = [citation.headingPath || citation.locationLabel, pageLabel(citation)].filter(Boolean).join(' · ')
  if (location) lines.push(`- ${labels.location}: ${location}`)
  if (citation.evidenceQuote) {
    lines.push('', `**${labels.evidence}**`, '', `> ${citation.evidenceQuote.replace(/\n+/g, '\n> ')}`)
  }
  return lines.join('\n')
}

function shelfIdentity(item: CiteShelfItem): string {
  const doi = normalizeDoi(item.doi)
  if (doi) return `doi:${doi.toLowerCase()}`
  const title = text(item.title || item.main).toLowerCase().replace(/\s+/g, ' ')
  return title ? `title:${title}` : `key:${item.key}`
}

function citationFromShelfItem(item: CiteShelfItem, fallbackNum: number): ResearchNoteCitation {
  const doi = normalizeDoi(item.doi)
  return {
    num: numberValue(item.num) || fallbackNum,
    title: text(item.title || item.main) || sourceTitleFromPath(item.sourceName || item.sourcePath),
    sourceName: sourceTitleFromPath(item.sourceName || item.sourcePath),
    sourcePath: text(item.sourcePath),
    authors: text(item.authors),
    venue: text(item.venue),
    year: text(item.year),
    doi,
    doiUrl: text(item.doiUrl) || (doi ? `https://doi.org/${doi}` : ''),
    citeFmt: text(item.citeFmt),
    headingPath: text(item.headingPath),
    locationLabel: text(item.locationLabel),
    evidenceQuote: text(item.evidenceQuote || item.cardEvidence),
    pageStart: numberValue(item.pageStart),
    pageEnd: numberValue(item.pageEnd),
    blockId: text(item.blockId),
    anchorId: text(item.anchorId),
    anchorKind: text(item.anchorKind),
  }
}

function shelfItemMarkdown(item: CiteShelfItem, index: number, labels: ResearchNoteLabels): string {
  const title = text(item.title || item.main) || labels.untitledSource
  const lines = [`### ${index + 1}. ${title}`]
  if (item.authors) lines.push(`- ${labels.authors}: ${item.authors}`)
  if (item.venue) lines.push(`- ${labels.venue}: ${item.venue}`)
  if (item.year) lines.push(`- ${labels.year}: ${item.year}`)
  const doi = normalizeDoi(item.doi)
  if (doi) lines.push(`- DOI: [${doi}](${item.doiUrl || `https://doi.org/${doi}`})`)
  const location = [item.headingPath || item.locationLabel, pageLabel({
    pageStart: item.pageStart,
    pageEnd: item.pageEnd,
  } as ResearchNoteCitation)].filter(Boolean).join(' · ')
  if (location) lines.push(`- ${labels.location}: ${location}`)
  if (item.summaryLine) lines.push('', `**${labels.summary}**`, '', item.summaryLine)
  const excerpt = text(item.shelfExcerpt || item.evidenceQuote || item.cardEvidence)
  if (excerpt) lines.push('', `**${labels.excerpt}**`, '', `> ${excerpt.replace(/\n+/g, '\n> ')}`)
  if (item.note) lines.push('', `**${labels.note}**`, '', item.note)
  return lines.join('\n')
}

export function buildResearchNoteBody(options: {
  answers: ResearchNoteAnswer[]
  selectedMessageIds: number[]
  includeShelf: boolean
  shelfItems: CiteShelfItem[]
  labels: ResearchNoteLabels
}): string {
  const selected = new Set(options.selectedMessageIds)
  const pickedAnswers = options.answers.filter((answer) => selected.has(answer.messageId))
  const sections: string[] = []
  const bibliography: ResearchNoteCitation[] = []
  const bibliographySeen = new Set<string>()

  pickedAnswers.forEach((answer, index) => {
    const question = answer.question || `${options.labels.question} ${index + 1}`
    const lines = [
      `## ${options.labels.question} ${index + 1}`,
      '',
      question,
      '',
      `### ${options.labels.answer}`,
      '',
      answer.answerMarkdown,
    ]
    if (answer.citations.length > 0) {
      lines.push('', `### ${options.labels.answerSources}`, '')
      lines.push(answer.citations.map((citation) => answerCitationMarkdown(citation, options.labels)).join('\n\n'))
    }
    sections.push(lines.join('\n').trim())
    for (const citation of answer.citations) {
      const identity = citationIdentity(citation)
      if (bibliographySeen.has(identity)) continue
      bibliographySeen.add(identity)
      bibliography.push(citation)
    }
  })

  if (options.includeShelf && options.shelfItems.length > 0) {
    const uniqueShelf: CiteShelfItem[] = []
    const seen = new Set<string>()
    for (const item of options.shelfItems) {
      const identity = shelfIdentity(item)
      if (seen.has(identity)) continue
      seen.add(identity)
      uniqueShelf.push(item)
    }
    sections.push([
      `## ${options.labels.researchBasket}`,
      '',
      uniqueShelf.map((item, index) => shelfItemMarkdown(item, index, options.labels)).join('\n\n'),
    ].join('\n').trim())
    uniqueShelf.forEach((item, index) => {
      const citation = citationFromShelfItem(item, index + 1)
      const identity = citationIdentity(citation)
      if (bibliographySeen.has(identity)) return
      bibliographySeen.add(identity)
      bibliography.push(citation)
    })
  }

  if (bibliography.length > 0) {
    sections.push([
      `## ${options.labels.bibliography}`,
      '',
      ...bibliography.map((citation, index) => `${index + 1}. ${bibliographyLine(citation, options.labels)}`),
    ].join('\n'))
  }
  return sections.filter(Boolean).join('\n\n').trim()
}

export function buildResearchNoteSourceLinks(options: {
  answers: ResearchNoteAnswer[]
  selectedMessageIds: number[]
  includeShelf: boolean
  shelfItems: CiteShelfItem[]
  conversationId: string
}): ResearchNoteSourceLink[] {
  const selected = new Set(options.selectedMessageIds)
  const links: ResearchNoteSourceLink[] = []
  const seen = new Set<string>()
  const add = (link: ResearchNoteSourceLink, identity: string) => {
    if (!identity || seen.has(identity)) return
    seen.add(identity)
    links.push(link)
  }
  for (const answer of options.answers) {
    if (!selected.has(answer.messageId)) continue
    add({
      kind: 'answer',
      label: answer.question || `#${answer.messageId}`,
      conversation_id: options.conversationId,
      message_id: answer.messageId,
    }, `answer:${options.conversationId}:${answer.messageId}`)
    for (const citation of answer.citations) {
      const identity = citation.sourcePath
        ? `source:${citation.sourcePath.toLowerCase()}|${citation.blockId || citation.anchorId || citation.headingPath}|${citation.num}`
        : `citation:${citationIdentity(citation)}`
      add({
        kind: 'source',
        label: citation.title || citation.sourceName,
        conversation_id: options.conversationId,
        message_id: answer.messageId,
        source_path: citation.sourcePath,
        source_name: citation.sourceName || citation.title,
        heading_path: citation.headingPath,
        location_label: citation.locationLabel,
        evidence_quote: citation.evidenceQuote,
        page_start: citation.pageStart,
        page_end: citation.pageEnd,
        block_id: citation.blockId,
        anchor_id: citation.anchorId,
        anchor_kind: citation.anchorKind,
      }, identity)
    }
  }
  if (options.includeShelf) {
    options.shelfItems.forEach((item, index) => {
      const citation = citationFromShelfItem(item, index + 1)
      const identity = citation.sourcePath
        ? `source:${citation.sourcePath.toLowerCase()}|${citation.blockId || citation.anchorId || citation.headingPath}|${citation.num}`
        : `citation:${citationIdentity(citation)}`
      add({
        kind: 'source',
        label: citation.title || citation.sourceName,
        conversation_id: options.conversationId,
        message_id: numberValue(item.traceAssistantMsgId || item.traceUserMsgId),
        source_path: citation.sourcePath,
        source_name: citation.sourceName || citation.title,
        heading_path: citation.headingPath,
        location_label: citation.locationLabel,
        evidence_quote: citation.evidenceQuote,
        page_start: citation.pageStart,
        page_end: citation.pageEnd,
        block_id: citation.blockId,
        anchor_id: citation.anchorId,
        anchor_kind: citation.anchorKind,
      }, identity)
    })
  }
  return links.slice(0, 240)
}

function splitBibliography(markdown: string): { main: string; bibliography: string[]; heading: string } {
  const value = String(markdown || '').trim()
  const pattern = /(?:^|\n)##\s+(参考文献|Bibliography)\s*\n/i
  const match = pattern.exec(value)
  if (!match || typeof match.index !== 'number') return { main: value, bibliography: [], heading: '' }
  const headingStart = value[match.index] === '\n' ? match.index + 1 : match.index
  const bodyStart = headingStart + String(match[0]).replace(/^\n/, '').length
  const bibliography = value.slice(bodyStart)
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^\d+[.)]\s*/, '').trim())
    .filter(Boolean)
  return {
    main: value.slice(0, headingStart).trim(),
    bibliography,
    heading: match[1],
  }
}

export function appendResearchNoteBody(existingBody: string, additionBody: string, labels: ResearchNoteLabels): string {
  const existing = splitBibliography(existingBody)
  const addition = splitBibliography(additionBody)
  let additionMain = addition.main
  const questionPattern = /^(##\s+)(研究问题|问题|Research question|Question)\s+\d+\s*$/gim
  const existingQuestionCount = (existing.main.match(questionPattern) || []).length
  let questionOffset = 0
  additionMain = additionMain.replace(questionPattern, (_match, prefix: string, word: string) => {
    questionOffset += 1
    return `${prefix}${word} ${existingQuestionCount + questionOffset}`
  })
  const bibliography: string[] = []
  const seen = new Set<string>()
  for (const line of [...existing.bibliography, ...addition.bibliography]) {
    const identity = line.toLowerCase().replace(/\s+/g, ' ').trim()
    if (!identity || seen.has(identity)) continue
    seen.add(identity)
    bibliography.push(line)
  }
  const sections = [existing.main, additionMain].filter(Boolean)
  if (bibliography.length > 0) {
    sections.push([
      `## ${existing.heading || addition.heading || labels.bibliography}`,
      '',
      ...bibliography.map((line, index) => `${index + 1}. ${line}`),
    ].join('\n'))
  }
  return sections.join('\n\n').trim()
}

export function researchNoteDefaultTitle(answer: ResearchNoteAnswer | undefined, prefix: string): string {
  const question = text(answer?.question).replace(/\s+/g, ' ')
  if (!question) return prefix
  const clipped = question.length > 54 ? `${question.slice(0, 54).trim()}…` : question
  return `${prefix}：${clipped}`
}
