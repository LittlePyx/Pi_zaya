import { expect, test } from '@playwright/test'
import {
  citationFormats,
  cleanCitationDisplayText,
  normalizeCiteDetail,
  toShelfItem,
  type CiteShelfItem,
} from '../../src/components/chat/citationState'
import {
  metadataQualityNeedsRepair,
  metadataQualityReady,
  sourceOpenQualityView,
  sourceQualityForItem,
} from '../../src/components/chat/citeShelfDisplay'
import {
  SHELF_MAX_ITEMS,
  articleSummaryPatchFromMeta,
  dedupeShelfItems,
  mergeShelfItemWithLive,
  shelfDiscoverySourceDetail,
  shelfItemHasDisplayableArticleSummary,
  shelfItemHasUsableLibraryFullText,
  shelfItemNeedsPersistedMetadataHydrate,
  shelfLibraryFullTextDetail,
} from '../../src/components/chat/citeShelfRuntime'
import { restoreShelfItems } from '../../src/components/chat/citeShelfStorage'
import type { ReaderLocateResult } from '../../src/components/chat/reader/readerTypes'
import { prepareRefsPanelHits } from '../../src/components/refs/refsPanelDisplay'

function shelfItem(meta: Record<string, unknown>): CiteShelfItem {
  const detail = normalizeCiteDetail({
    anchor: 'ref-1',
    source_path: 'paper.en.md',
    source_name: 'paper.pdf',
    title: 'Fast hyperspectral single-pixel imaging',
    authors: 'Jiang X, Li Z',
    venue: 'Optics Express',
    year: '2022',
    doi: '10.1364/oe.458742',
    ...meta,
  })
  if (!detail) throw new Error('fixture detail failed to normalize')
  const item = toShelfItem(detail)
  const rawSummaryQuality = meta.summaryQuality || meta.summary_quality
  return {
    ...item,
    key: typeof meta.key === 'string' ? meta.key : item.key,
    main: typeof meta.main === 'string' ? meta.main : item.main,
    tags: Array.isArray(meta.tags)
      ? meta.tags.map(tag => String(tag || '').trim()).filter(Boolean)
      : item.tags,
    note: typeof meta.note === 'string' ? meta.note : item.note,
    summaryLine: typeof meta.summaryLine === 'string'
      ? meta.summaryLine
      : typeof meta.summary_line === 'string'
        ? meta.summary_line
        : item.summaryLine,
    summarySource: typeof meta.summarySource === 'string'
      ? meta.summarySource
      : typeof meta.summary_source === 'string'
        ? meta.summary_source
        : item.summarySource,
    summaryProvider: typeof meta.summaryProvider === 'string'
      ? meta.summaryProvider
      : typeof meta.summary_provider === 'string'
        ? meta.summary_provider
        : item.summaryProvider,
    summaryQuality: rawSummaryQuality && typeof rawSummaryQuality === 'object'
      ? rawSummaryQuality as Record<string, unknown>
      : item.summaryQuality,
  }
}

test('citation display text removes internal grounding markers', () => {
  const cleaned = cleanCitationDisplayText(
    'Claim text [[SUPPORT:DOC-1-S2]] still readable [[CITE:ref-1:12]].',
  )

  expect(cleaned).toBe('Claim text still readable.')
  expect(cleaned).not.toContain('SUPPORT:')
  expect(cleaned).not.toContain('CITE:')
})

test('citation shelf metadata ready follows export acceptance when present', () => {
  const blocked = shelfItem({
    metadata_quality: { ok: true, status: 'ready', score: 100, missing_fields: [], issues: [] },
    metadata_export_acceptance: {
      export_ready: false,
      missing_fields: ['doi'],
      issue_codes: ['missing_doi'],
    },
  })

  expect(metadataQualityReady(blocked)).toBe(false)
  expect(metadataQualityNeedsRepair(blocked)).toBe(true)

  const ready = shelfItem({
    metadata_quality: { ok: false, status: 'warning', score: 84, missing_fields: ['authors'], issues: [] },
    metadata_export_acceptance: {
      export_ready: true,
      missing_fields: [],
      issue_codes: [],
    },
  })

  expect(metadataQualityReady(ready)).toBe(true)
  expect(metadataQualityNeedsRepair(ready)).toBe(false)
})

test('citation exports omit unknown bibliographic fields instead of fabricating placeholders', () => {
  const detail = normalizeCiteDetail({
    anchor: 'system-a-doi',
    citation_route: 'system_a',
    is_inpaper: false,
    source_path: 'db/NatPhoton-2025-Structured detection/Structured detection.en.md',
    source_name: 'NatPhoton-2025-Structured detection.pdf',
    title: 'Structured detection for simultaneous super-resolution',
    authors: '',
    venue: '',
    year: '',
    doi: '10.1038/example.structured',
  })
  if (!detail) throw new Error('minimal DOI fixture failed to normalize')

  const formats = citationFormats(detail)

  expect(formats.bibtex).toContain('@misc{ref_nd_')
  expect(formats.bibtex).toContain('title={Structured detection for simultaneous super-resolution}')
  expect(formats.bibtex).toContain('doi={10.1038/example.structured}')
  expect(formats.bibtex).toContain('file={NatPhoton-2025-Structured detection.pdf}')
  expect(formats.ris).toContain('TY  - GEN')
  expect(formats.ris).toContain('DO  - 10.1038/example.structured')
  for (const text of Object.values(formats)) {
    expect(text).not.toContain('[Unknown Authors]')
    expect(text).not.toContain('Unknown Venue')
    expect(text).not.toContain('20xx')
  }
  expect(formats.bibtex).not.toContain('author={')
  expect(formats.bibtex).not.toContain('journal={')
  expect(formats.bibtex).not.toContain('year={')
  expect(formats.ris).not.toContain('AU  -')
  expect(formats.ris).not.toContain('JO  -')
  expect(formats.ris).not.toContain('PY  -')
})

test('citation shelf live merge prefers export-ready metadata over quality-only metadata', () => {
  const current = shelfItem({
    metadata_quality: { ok: true, status: 'ready', score: 100, missing_fields: [], issues: [] },
    metadata_export_acceptance: {
      export_ready: false,
      missing_fields: ['doi'],
      issue_codes: ['missing_doi'],
    },
  })
  const live = shelfItem({
    doi: '10.1364/oe.458742',
    metadata_quality: { ok: true, status: 'ready', score: 100, missing_fields: [], issues: [] },
    metadata_export_acceptance: {
      export_ready: true,
      missing_fields: [],
      issue_codes: [],
    },
  })

  const merged = mergeShelfItemWithLive(current, live)

  expect(metadataQualityReady(merged)).toBe(true)
  expect(merged.metadataExportAcceptance?.export_ready).toBe(true)
})

test('citation shelf dedupe merges richer duplicate metadata instead of dropping it', () => {
  const local = shelfItem({
    key: 'local-weak-ref',
    main: 'Untitled',
    title: 'Untitled',
    doi: 'https://doi.org/10.1364/OE.458742',
    authors: '',
    venue: '',
    year: '',
    tags: ['kept-tag'],
    note: 'Keep my note.',
    metadata_quality: { status: 'pending', missing_fields: ['authors', 'venue', 'year'] },
  })
  const backend = shelfItem({
    key: 'backend-rich-ref',
    main: 'Fast hyperspectral single-pixel imaging',
    title: 'Fast hyperspectral single-pixel imaging',
    doi: '10.1364/oe.458742',
    authors: 'Jiang X, Li Z',
    venue: 'Optics Express',
    year: '2022',
    summaryLine: 'A compact article summary from the repaired bibliography metadata.',
    summarySource: 'abstract',
    metadata_quality: { ok: true, status: 'ready', missing_fields: [] },
    metadata_export_acceptance: { export_ready: true, missing_fields: [] },
  })

  const deduped = dedupeShelfItems([local, backend])

  expect(deduped).toHaveLength(1)
  expect(deduped[0].key).toBe('local-weak-ref')
  expect(deduped[0].title).toBe('Fast hyperspectral single-pixel imaging')
  expect(deduped[0].authors).toBe('Jiang X, Li Z')
  expect(deduped[0].venue).toBe('Optics Express')
  expect(deduped[0].year).toBe('2022')
  expect(deduped[0].summaryLine).toContain('compact article summary')
  expect(deduped[0].tags).toEqual(['kept-tag'])
  expect(deduped[0].note).toBe('Keep my note.')
  expect(metadataQualityReady(deduped[0])).toBe(true)
})

test('citation shelf dedupe still refreshes duplicates after the visible max', () => {
  const fullShelf = Array.from({ length: SHELF_MAX_ITEMS }, (_, idx) => shelfItem({
    anchor: `ref-${idx}`,
    source_name: `fixture-${idx}.pdf`,
    doi: `10.1000/${idx}`,
    title: `Untitled ${idx}`,
    authors: '',
    venue: '',
    year: '',
    metadata_quality: { status: 'pending', missing_fields: ['authors', 'venue', 'year'] },
  }))
  const richDuplicateAfterLimit = shelfItem({
    anchor: 'ref-rich-after-limit',
    source_name: 'fixture-0.pdf',
    doi: 'https://doi.org/10.1000/0',
    title: 'Complete metadata for the first paper',
    authors: 'A Researcher',
    venue: 'Journal of Test Fixtures',
    year: '2024',
    metadata_quality: { ok: true, status: 'ready', missing_fields: [] },
    metadata_export_acceptance: { export_ready: true, missing_fields: [] },
  })

  const deduped = dedupeShelfItems([...fullShelf, richDuplicateAfterLimit])

  expect(deduped).toHaveLength(SHELF_MAX_ITEMS)
  expect(deduped[0].title).toBe('Complete metadata for the first paper')
  expect(deduped[0].authors).toBe('A Researcher')
  expect(deduped[0].venue).toBe('Journal of Test Fixtures')
  expect(metadataQualityReady(deduped[0])).toBe(true)
})

test('citation shelf live merge upgrades context-only summary as a full article summary bundle', () => {
  const current = shelfItem({
    summaryLine: 'This citation is relevant to the surrounding answer context.',
    summarySource: 'citation_context',
    summaryQuality: { ok: true, status: 'grounded', source: 'citation_context' },
  })
  const live = shelfItem({
    summaryLine: 'The article proposes a fast single-pixel hyperspectral imaging method with compressed measurements.',
    summarySource: 'abstract',
    summaryProvider: 'crossref',
    summaryQuality: { ok: true, status: 'grounded', source: 'abstract' },
  })

  const merged = mergeShelfItemWithLive(current, live)

  expect(merged.summaryLine).toContain('fast single-pixel hyperspectral imaging')
  expect(merged.summarySource).toBe('abstract')
  expect(merged.summaryProvider).toBe('crossref')
  expect(merged.summaryQuality?.source).toBe('abstract')
  expect(shelfItemHasDisplayableArticleSummary(merged)).toBe(true)
})

test('legacy System B article summary remains displayable when its text aligns with the upstream title', () => {
  const detail = normalizeCiteDetail({
    anchor: 'legacy-cassi-r1',
    is_inpaper: true,
    citation_route: 'system_b',
    shelf_item_kind: 'reference',
    title: 'Single-shot compressive spectral imaging with a dual-disperser architecture',
    doi: '10.1364/OE.15.014013',
    raw: '[1] Gehm M, Brady D. Single-shot compressive spectral imaging with a dual-disperser architecture. Optics Express, 2007.',
    summary_line: 'This paper introduces a dual-disperser architecture for single-shot compressive spectral imaging.',
    summary_source: 'abstract',
  })
  if (!detail) throw new Error('legacy System B fixture failed to normalize')

  const item = toShelfItem(detail)

  expect(item.summaryLine).toContain('dual-disperser architecture')
  expect(shelfItemHasDisplayableArticleSummary(item)).toBe(true)
})

test('historical System B hydration and async summary patch reject a SCINeRF summary bound to Boyd ADMM', () => {
  const historical = restoreShelfItems([{
    key: 'historical-boyd-r4',
    anchor: 'historical-boyd-r4',
    isInpaper: true,
    citationRoute: 'system_b',
    shelfItemKind: 'reference',
    main: 'Distributed optimization and statistical learning via the alternating direction method of multipliers',
    title: 'Distributed optimization and statistical learning via the alternating direction method of multipliers',
    doi: '10.1561/2200000016',
    raw: '[4] Boyd et al. Distributed optimization and statistical learning via the alternating direction method of multipliers.',
    summaryLine: 'SCINeRF reconstructs a neural radiance field from one snapshot compressive image.',
    summarySource: 'abstract',
  }])

  expect(historical).toHaveLength(1)
  expect(historical[0].summaryLine).toBe('')
  expect(shelfItemHasDisplayableArticleSummary(historical[0])).toBe(false)

  const mismatchedPatch = articleSummaryPatchFromMeta(historical[0], {
    summary_line: 'SCINeRF reconstructs a neural radiance field from one snapshot compressive image.',
    summary_source: 'abstract',
    summary_provider: 'crossref',
    external_title: 'SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image',
    external_doi: '10.1109/cvpr52733.2024.01469',
    summary_quality: { ok: true, status: 'grounded', score: 96 },
  })

  expect(mismatchedPatch).toEqual({})
})

test('embedded System B summary rejects similar text when the external DOI identifies another article', () => {
  const detail = normalizeCiteDetail({
    anchor: 'legacy-cassi-external-conflict',
    is_inpaper: true,
    citation_route: 'system_b',
    shelf_item_kind: 'reference',
    title: 'Single-shot compressive spectral imaging with a dual-disperser architecture',
    doi: '10.1364/OE.15.014013',
    raw: '[1] Gehm M, Brady D. Single-shot compressive spectral imaging with a dual-disperser architecture.',
    summary_line: 'This paper introduces a dual-disperser architecture for single-shot compressive spectral imaging.',
    summary_source: 'abstract',
    external_title: 'Single-shot compressive spectral imaging with a dual-disperser architecture: an erratum',
    external_doi: '10.1364/OE.99.999999',
  })
  if (!detail) throw new Error('external identity conflict fixture failed to normalize')

  const item = toShelfItem(detail)

  expect(item.summaryLine).toBe('')
  expect(shelfItemHasDisplayableArticleSummary(item)).toBe(false)
})

test('historical System B shelf item opens matched library full text without losing discovery source', () => {
  const item = shelfItem({
    num: 17,
    source_path: 'db/NatPhoton-2019-Principles.en.md',
    source_name: 'NatPhoton-2019-Principles.en.md',
    is_inpaper: true,
    citation_route: 'system_b',
    shelf_item_kind: 'reference',
    raw: 'Zhang Y et al. 3D single-pixel video. J. Opt. 2016.',
    citation_context: 'The NatPhoton paper cites this work for 3D photometric stereo [17].',
    library_match_status: 'in_library',
    library_match_confidence: 0.99,
    library_match_reason: 'doi_exact',
    library_match_path: 'F:\\papers\\Journal of Optics-2016-3D single-pixel video.pdf',
    library_match_title: '3D single-pixel video',
  })

  const fullText = shelfLibraryFullTextDetail(item)
  const discovery = shelfDiscoverySourceDetail(item)

  expect(shelfItemHasUsableLibraryFullText(item)).toBe(true)
  expect(fullText).not.toBeNull()
  expect(fullText?.sourcePath).toBe('F:\\papers\\Journal of Optics-2016-3D single-pixel video.pdf')
  expect(fullText?.sourceName).toBe('Journal of Optics-2016-3D single-pixel video.pdf')
  expect(fullText?.num).toBe(0)
  expect(fullText?.isInpaper).toBe(false)
  expect(fullText?.citationRoute).toBe('system_a')
  expect(fullText?.headingPath).toBe('')
  expect(fullText?.raw).toBe('')
  expect(item.sourcePath).toBe('db/NatPhoton-2019-Principles.en.md')
  expect(item.num).toBe(17)
  expect(item.citationRoute).toBe('system_b')
  expect(discovery?.sourcePath).toBe('db/NatPhoton-2019-Principles.en.md')
  expect(discovery?.summaryLine).toBe('')
  expect(discovery?.evidenceQuote).toContain('photometric stereo [17]')
})

test('ready historical reference retries metadata hydration until library match is checked', () => {
  const unchecked = shelfItem({
    is_inpaper: true,
    shelf_item_kind: 'reference',
    bibliometrics_checked: true,
    metadata_quality: { ok: true, status: 'ready', score: 100, issues: [] },
    library_match_status: '',
  })
  const checkedMissing = shelfItem({
    ...unchecked,
    library_match_status: 'not_in_library',
  })

  expect(shelfItemNeedsPersistedMetadataHydrate(unchecked)).toBe(true)
  expect(shelfItemNeedsPersistedMetadataHydrate(checkedMissing)).toBe(false)
})

test('citation shelf source-open quality uses normalized source identity', () => {
  const item = shelfItem({
    source_path: 'DB\\\\Fixture//Paper.en.md/',
    source_name: 'paper.pdf',
  })
  const locateResult: ReaderLocateResult = {
    locateRequestId: 12,
    sourcePath: 'db/fixture/paper.en.md',
    sourceName: 'paper.pdf',
    status: 'exact',
    precision: 'phrase',
    ok: true,
    repairable: false,
    strictLocate: true,
    hint: 'phrase matched',
    reason: 'phrase matched',
  }
  const labels = {
    shelf_source_open_missing: 'Missing source',
    shelf_source_open_repaired_reopen: 'Repaired, verify again',
    shelf_source_open_failed: 'Auto-repairing',
    shelf_source_open_verified: 'Exact hit',
    shelf_source_open_verified_block: 'Block hit',
    shelf_source_open_fuzzy: 'Broad locate',
    shelf_source_open_section_verified: 'Section located',
    shelf_source_open_file_verified: 'Source opened',
    shelf_source_open_repair: 'Repair source',
    shelf_source_open_exact: 'Exact locate',
    shelf_source_open_page: 'Page locate',
    shelf_source_open_section: 'Section locate',
    shelf_source_open_file: 'Open source only',
  }

  const view = sourceOpenQualityView(item, null, labels, locateResult)

  expect(view.status).toBe('verified')
  expect(view.precision).toBe('phrase')
  expect(view.reason).toContain('phrase matched')
})

test('citation shelf source quality lookup tolerates normalized source paths', () => {
  const item = shelfItem({
    source_path: 'file:///F:/Research%20Papers/Fixture/Paper.en.md?download=1#reader',
    source_name: 'paper.pdf',
  })
  const qualityByPath = {
    'F:\\Research Papers\\Fixture\\Paper.en.md': {
      source_path: 'F:\\Research Papers\\Fixture\\Paper.en.md',
      conversion_quality: {
        status: 'warning',
        has_review_issue: true,
        issues: [{ label: 'table split', code: 'table_split' }],
      },
    },
  } as unknown as Parameters<typeof sourceQualityForItem>[1]

  expect(sourceQualityForItem(item, qualityByPath)?.status).toBe('warning')
})

test('refs panel dedupes and filters source-path URL variants as one document', () => {
  const entry = {
    hits: [
      {
        score: 2,
        text: 'weaker duplicate',
        meta: { source_path: 'F:\\Research Papers\\Fixture\\A#B.en.md' },
        ui_meta: { source_path: 'F:\\Research Papers\\Fixture\\A#B.en.md', display_name: 'A#B.pdf' },
      },
      {
        score: 9,
        text: 'richer duplicate',
        meta: { source_path: 'file:///F:/Research%20Papers/Fixture/A%23B.en.md?download=1#reader' },
        ui_meta: { source_path: 'file:///F:/Research%20Papers/Fixture/A%23B.en.md?download=1#reader' },
      },
      {
        score: 7,
        text: 'active guide source',
        meta: { source_path: 'file:///F:/Research%20Papers/Fixture/Guide.en.md#reader' },
        ui_meta: { source_path: 'file:///F:/Research%20Papers/Fixture/Guide.en.md#reader' },
      },
    ],
  }

  const prepared = prepareRefsPanelHits(entry, {
    activeSourcePath: 'F:\\Research Papers\\Fixture\\Guide.en.md',
  })

  expect(prepared.hits).toHaveLength(1)
  expect(prepared.hits[0].text).toBe('richer duplicate')
  expect(prepared.suppressedHitCount).toBe(0)
  expect(prepared.hiddenActiveSourceCount).toBe(1)
})

test('refs panel keeps same-basename papers from different directories distinct', () => {
  const entry = {
    hits: [
      {
        score: 9,
        text: 'collection A evidence',
        meta: { source_path: 'F:/library/collection-a/paper.en.md' },
        ui_meta: { source_path: 'F:/library/collection-a/paper.en.md', display_name: 'paper.pdf' },
      },
      {
        score: 8,
        text: 'collection B evidence',
        meta: { source_path: 'F:/library/collection-b/paper.en.md' },
        ui_meta: { source_path: 'F:/library/collection-b/paper.en.md', display_name: 'paper.pdf' },
      },
    ],
  }

  const prepared = prepareRefsPanelHits(entry, {
    activeSourcePath: 'F:/library/collection-c/paper.en.md',
  })

  expect(prepared.hits).toHaveLength(2)
  expect(prepared.hits.map((hit) => hit.text)).toEqual([
    'collection A evidence',
    'collection B evidence',
  ])
  expect(prepared.hiddenActiveSourceCount).toBe(0)
})

test('refs panel filters the markdown converted from the active bound PDF', () => {
  const entry = {
    hits: [
      {
        score: 9,
        text: 'current paper markdown evidence',
        meta: { source_path: 'F:/library/db/Paper/Paper.en.md' },
        ui_meta: { source_path: 'F:/library/db/Paper/Paper.en.md', display_name: 'Paper.pdf' },
      },
      {
        score: 8,
        text: 'another paper',
        meta: { source_path: 'F:/library/db/Other/Other.en.md' },
        ui_meta: { source_path: 'F:/library/db/Other/Other.en.md', display_name: 'Other.pdf' },
      },
    ],
  }

  const prepared = prepareRefsPanelHits(entry, {
    activeSourcePath: 'F:/library/pdfs/Paper.pdf',
    activeSourceName: 'Paper.pdf',
  })

  expect(prepared.hits.map((hit) => hit.text)).toEqual(['another paper'])
  expect(prepared.hiddenActiveSourceCount).toBe(1)
})

test('refs panel keeps cross-format namesakes from different collections distinct', () => {
  const entry = {
    hits: [
      {
        score: 9,
        text: 'collection A markdown evidence',
        meta: { source_path: 'F:/library/collection-a/Paper.en.md' },
        ui_meta: { source_path: 'F:/library/collection-a/Paper.en.md', display_name: 'Paper.pdf' },
      },
    ],
  }

  const prepared = prepareRefsPanelHits(entry, {
    activeSourcePath: 'F:/library/collection-b/Paper.pdf',
    activeSourceName: 'Paper.pdf',
  })

  expect(prepared.hits.map((hit) => hit.text)).toEqual(['collection A markdown evidence'])
  expect(prepared.hiddenActiveSourceCount).toBe(0)
})
