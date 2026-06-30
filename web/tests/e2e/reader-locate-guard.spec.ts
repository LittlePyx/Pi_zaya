import { expect, test } from '@playwright/test'
import {
  normalizeReaderSourcePathForMatch,
  readerLocateRepairRunMatchesActiveRequest,
  readerLocateResultMatchesActiveRequest,
  readerSourcePathsMatch,
} from '../../src/components/chat/reader/readerLocateGuard'
import { readerHighlightsSignature, stableReaderHighlightId } from '../../src/components/chat/reader/readerSessionState'
import {
  buildBasicReaderOpenPayload,
  inferReaderSourceNameFromPath,
} from '../../src/components/chat/reader/readerOpenPayloadUtils'
import {
  buildHighlightQueries,
  equationNumberMatchScore,
  extractEquationNumbers,
  extractFigureNumbers,
} from '../../src/components/chat/reader/readerDomUtils'
import { referenceSourcePathCacheKey } from '../../src/api/references'
import { basenameFromSourcePath, cleanFileSourcePathInput, normalizeSourcePathForMatch } from '../../src/utils/sourcePath'
import type { ReaderLocateResult, ReaderOpenPayload } from '../../src/components/chat/reader/readerTypes'

const payload: ReaderOpenPayload = {
  sourcePath: 'db/Fixture/Fixture.en.md',
  sourceName: 'Fixture.pdf',
  locateRequestId: 7,
  locateFeedbackKey: 'shelf-fixture-key',
}

const result: ReaderLocateResult = {
  locateRequestId: 7,
  sourcePath: 'db/Fixture/Fixture.en.md',
  sourceName: 'Fixture.pdf',
  locateFeedbackKey: 'shelf-fixture-key',
  status: 'failed',
  precision: 'failed',
  ok: false,
  repairable: true,
  strictLocate: true,
  hint: 'not found',
  reason: 'not found',
}

const guard = {
  locateRequestId: 7,
  sourcePath: 'db/Fixture/Fixture.en.md',
  conversationId: 'conv-a',
}

test('reader locate guard accepts only the active reader request', () => {
  expect(readerLocateResultMatchesActiveRequest({
    result,
    guard,
    currentPayload: payload,
    currentConversationId: 'conv-a',
    readerOpen: true,
  })).toBe(true)

  expect(readerLocateResultMatchesActiveRequest({
    result,
    guard,
    currentPayload: payload,
    currentConversationId: 'conv-b',
    readerOpen: true,
  })).toBe(false)

  expect(readerLocateResultMatchesActiveRequest({
    result: { ...result, locateRequestId: 8 },
    guard,
    currentPayload: payload,
    currentConversationId: 'conv-a',
    readerOpen: true,
  })).toBe(false)

  expect(readerLocateResultMatchesActiveRequest({
    result: { ...result, sourcePath: 'db/Other/Other.en.md' },
    guard,
    currentPayload: payload,
    currentConversationId: 'conv-a',
    readerOpen: true,
  })).toBe(false)

  expect(readerLocateResultMatchesActiveRequest({
    result,
    guard,
    currentPayload: { ...payload, locateFeedbackKey: 'other-key' },
    currentConversationId: 'conv-a',
    readerOpen: true,
  })).toBe(false)

  expect(readerLocateResultMatchesActiveRequest({
    result,
    guard,
    currentPayload: payload,
    currentConversationId: 'conv-a',
    readerOpen: false,
  })).toBe(false)

  expect(readerLocateResultMatchesActiveRequest({
    result: { ...result, locateRequestId: 0 },
    guard,
    currentPayload: payload,
    currentConversationId: 'conv-a',
    readerOpen: true,
  })).toBe(false)
})

test('reader anchor helpers understand Chinese equation and figure labels without mojibake', () => {
  expect(extractEquationNumbers('公式（３）给出了重建目标，第 5 式是约束项。')).toEqual(expect.arrayContaining([3, 5]))
  expect(extractFigureNumbers('图 ２ 展示系统结构，第 4 图给出消融结果。')).toEqual([2, 4])
  expect(equationNumberMatchScore('如公式 3 所示，目标函数保持一致。', [3])).toBeGreaterThan(0.9)

  const equationQueries = buildHighlightQueries('目标函数见公式 3。', { anchorKind: 'equation', anchorNumber: 3 })
  expect(equationQueries).toEqual(expect.arrayContaining(['公式 3', '公式(3)', '公式（3）', '第 3 式']))
  expect(equationQueries.join(' ')).not.toContain('\u95b8')

  const figureQueries = buildHighlightQueries('结构如图 2。', { anchorKind: 'figure', anchorNumber: 2 })
  expect(figureQueries).toEqual(expect.arrayContaining(['图 2', '图2', '第 2 图']))
  expect(figureQueries.join(' ')).not.toContain('\u95b8')
})

test('reader locate guard compares source paths by normalized identity', () => {
  expect(normalizeReaderSourcePathForMatch('DB\\\\Fixture//Fixture.en.md/')).toBe('db/fixture/fixture.en.md')
  expect(readerSourcePathsMatch('DB\\\\Fixture//Fixture.en.md/', 'db/fixture/fixture.en.md')).toBe(true)
  expect(readerSourcePathsMatch('db/fixture/fixture.en.md', 'db/other/fixture.en.md')).toBe(false)
  expect(readerSourcePathsMatch('', 'db/fixture/fixture.en.md')).toBe(false)
  expect(normalizeReaderSourcePathForMatch('file:///F:/Research%20Papers/./Fixture//Paper.en.md?download=1#anchor')).toBe('f:/research papers/fixture/paper.en.md')
  expect(normalizeReaderSourcePathForMatch('file:///F:/Research%20Papers/Fixture/A%23B.en.md?download=1#anchor')).toBe('f:/research papers/fixture/a#b.en.md')
  expect(normalizeReaderSourcePathForMatch('F:\\Research Papers\\Fixture\\A#B.en.md')).toBe('f:/research papers/fixture/a#b.en.md')
  expect(normalizeReaderSourcePathForMatch('F:\\Research Papers\\Fixture\\Paper.en.md#reader')).toBe('f:/research papers/fixture/paper.en.md')
  expect(readerSourcePathsMatch(
    'file:///F:/Research%20Papers/./Fixture//Paper.en.md?download=1#anchor',
    'F:\\Research Papers\\Fixture\\Paper.en.md',
  )).toBe(true)
  expect(readerSourcePathsMatch(
    'file:///F:/Research%20Papers/Fixture/A%23B.en.md?download=1#anchor',
    'F:\\Research Papers\\Fixture\\A#B.en.md',
  )).toBe(true)
  expect(readerSourcePathsMatch(
    './db/Fixture/../Fixture/Paper.en.md#locate',
    'db\\Fixture\\Paper.en.md',
  )).toBe(true)

  expect(readerLocateResultMatchesActiveRequest({
    result: {
      ...result,
      sourcePath: 'file:///F:/Research%20Papers/Fixture/Fixture.en.md#locate',
    },
    guard: {
      ...guard,
      sourcePath: 'F:\\Research Papers\\Fixture\\Fixture.en.md',
    },
    currentPayload: {
      ...payload,
      sourcePath: 'F:/Research Papers/./Fixture//Fixture.en.md?from=shelf',
    },
    currentConversationId: 'conv-a',
    readerOpen: true,
  })).toBe(true)
})

test('source path helpers keep display names and cache keys aligned', () => {
  const browserPath = 'file:///F:/Research%20Papers/Fixture/A%23B.en.md?download=1#reader'
  const localPath = 'F:\\Research Papers\\Fixture\\A#B.en.md'

  expect(cleanFileSourcePathInput(browserPath)).toBe('F:/Research Papers/Fixture/A#B.en.md')
  expect(basenameFromSourcePath(browserPath)).toBe('A#B.en.md')
  expect(inferReaderSourceNameFromPath(browserPath)).toBe('A#B.en.md')
  expect(normalizeSourcePathForMatch(browserPath)).toBe(normalizeReaderSourcePathForMatch(localPath))
  expect(referenceSourcePathCacheKey(browserPath)).toBe(referenceSourcePathCacheKey(localPath))
})

test('basic reader payload sanitizes locate candidates before opening reader', () => {
  const payload = buildBasicReaderOpenPayload({
    sourcePath: 'db/Fixture/Fixture.en.md',
    sourceName: ' Fixture.pdf ',
    headingPath: ' Fixture Paper / 2. Method ',
    snippet: ' Equation (1) defines the objective. ',
    highlightSnippet: ' Equation (1) defines the objective. ',
    blockId: ' eq-1 ',
    anchorId: ' a-eq-1 ',
    anchorKind: ' Equation ',
    anchorNumber: 1.8,
    relatedBlockIds: [' eq-1 ', 'eq-1', ' p-2 ', ''],
    locateTarget: {
      blockId: ' eq-1 ',
      anchorKind: ' Equation ',
      anchorNumber: 1.8,
      snippetAliases: [' Equation objective ', 'equation objective', ''],
      relatedBlockIds: [' p-2 ', 'p-2'],
    },
    alternatives: [
      {
        headingPath: 'Fixture Paper / 2. Method',
        snippet: 'Equation (1) defines the objective.',
        highlightSnippet: 'Equation (1) defines the objective.',
        blockId: 'eq-1',
        anchorId: 'a-eq-1',
        anchorKind: 'equation',
        anchorNumber: 1,
      },
      {
        headingPath: 'Fixture Paper / 2.2 Optimization',
        snippet: 'The optimization section minimizes Equation (1).',
        highlightSnippet: 'The optimization section minimizes Equation (1).',
        blockId: 'p-2',
        anchorId: 'a-p-2',
        anchorKind: ' Paragraph ',
      },
      {
        headingPath: 'Fixture Paper / 2.2 Optimization',
        snippet: 'The optimization section minimizes Equation (1).',
        highlightSnippet: 'The optimization section minimizes Equation (1).',
        blockId: 'p-2',
        anchorId: 'a-p-2',
        anchorKind: 'paragraph',
      },
      { anchorKind: 'paragraph' },
    ],
    visibleAlternatives: [
      {
        headingPath: 'Fixture Paper / 2. Method',
        snippet: 'Equation (1) defines the objective.',
        highlightSnippet: 'Equation (1) defines the objective.',
        blockId: 'eq-1',
        anchorId: 'a-eq-1',
        anchorKind: 'equation',
        anchorNumber: 1,
      },
      {
        headingPath: 'Fixture Paper / 2.2 Optimization',
        snippet: 'The optimization section minimizes Equation (1).',
        highlightSnippet: 'The optimization section minimizes Equation (1).',
        blockId: 'p-2',
        anchorId: 'a-p-2',
        anchorKind: 'paragraph',
      },
    ],
    evidenceAlternatives: [
      {
        headingPath: 'Fixture Paper / 2. Method',
        snippet: 'Equation (1) defines the objective.',
        highlightSnippet: 'Equation (1) defines the objective.',
        blockId: 'eq-1',
        anchorId: 'a-eq-1',
        anchorKind: 'equation',
        anchorNumber: 1,
      },
      {
        headingPath: 'Fixture Paper / 2.2 Optimization',
        snippet: 'The optimization section minimizes Equation (1).',
        highlightSnippet: 'The optimization section minimizes Equation (1).',
        blockId: 'p-2',
        anchorId: 'a-p-2',
        anchorKind: 'paragraph',
      },
      {
        headingPath: 'Fixture Paper / 4. Experiments',
        snippet: 'The experiments reuse the same objective for ablation.',
        highlightSnippet: 'The experiments reuse the same objective for ablation.',
        blockId: 'p-3',
        anchorId: 'a-p-3',
      },
    ],
    initialAltIndex: 99,
  })

  expect(payload).not.toBeNull()
  expect(payload?.sourceName).toBe('Fixture.pdf')
  expect(payload?.anchorKind).toBe('equation')
  expect(payload?.anchorNumber).toBe(1)
  expect(payload?.relatedBlockIds).toEqual(['eq-1', 'p-2'])
  expect(payload?.locateTarget?.anchorKind).toBe('equation')
  expect(payload?.locateTarget?.anchorNumber).toBe(1)
  expect(payload?.locateTarget?.snippetAliases).toEqual(['Equation objective'])
  expect(payload?.locateTarget?.relatedBlockIds).toEqual(['p-2'])
  expect(payload?.alternatives?.map((item) => item.blockId)).toEqual(['p-2'])
  expect(payload?.visibleAlternatives?.map((item) => item.blockId)).toEqual(['eq-1', 'p-2'])
  expect(payload?.visibleAlternatives?.map((item) => item.anchorKind)).toEqual(['equation', 'paragraph'])
  expect(payload?.evidenceAlternatives?.map((item) => item.blockId)).toEqual(['eq-1', 'p-2', 'p-3'])
  expect(payload?.initialAltIndex).toBe(2)
})

test('reader locate repair run guard rejects stale run callbacks', () => {
  expect(readerLocateRepairRunMatchesActiveRequest({
    expectedRunToken: 3,
    currentRunToken: 3,
    result,
    guard,
    currentPayload: payload,
    currentConversationId: 'conv-a',
    readerOpen: true,
  })).toBe(true)

  expect(readerLocateRepairRunMatchesActiveRequest({
    expectedRunToken: 3,
    currentRunToken: 4,
    result,
    guard,
    currentPayload: payload,
    currentConversationId: 'conv-a',
    readerOpen: true,
  })).toBe(false)

  expect(readerLocateRepairRunMatchesActiveRequest({
    expectedRunToken: 3,
    currentRunToken: 3,
    result,
    guard,
    currentPayload: payload,
    currentConversationId: 'conv-b',
    readerOpen: true,
  })).toBe(false)
})

test('reader highlight signature tracks content and stable location fields', () => {
  const base = [{
    id: 'h-1',
    text: 'same length aa',
    sourcePath: 'DB\\Fixture\\Paper.en.md',
    startOffset: 12,
    endOffset: 26,
    updatedAt: 100,
  }]

  expect(readerHighlightsSignature(base)).toBe(readerHighlightsSignature([{
    ...base[0],
    sourcePath: 'db//fixture//paper.en.md/',
  }]))

  expect(readerHighlightsSignature(base)).not.toBe(readerHighlightsSignature([{
    ...base[0],
    text: 'same length bb',
  }]))

  expect(readerHighlightsSignature(base)).not.toBe(readerHighlightsSignature([{
    ...base[0],
    startOffset: 13,
  }]))
})

test('reader fallback highlight ids are deterministic for legacy records', () => {
  const legacy = {
    text: 'important sentence',
    sourcePath: 'DB\\Fixture\\Paper.en.md',
    headingPath: 'Methods',
    blockId: 'p-12',
    startOffset: 10,
    endOffset: 28,
  }

  expect(stableReaderHighlightId(legacy)).toBe(stableReaderHighlightId({
    ...legacy,
    sourcePath: 'file:///db//fixture//paper.en.md?source=reader#selection',
  }))
  expect(stableReaderHighlightId(legacy)).not.toBe(stableReaderHighlightId({
    ...legacy,
    startOffset: 11,
  }))
})
