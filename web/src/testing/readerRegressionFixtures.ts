import type { ReaderDocAnchor, ReaderDocBlock } from '../api/references'
import type { ReaderOpenPayload } from '../components/chat/reader/readerTypes'

export type ReaderRegressionScenario =
  | 'strict-quote'
  | 'evidence-nav'
  | 'candidate-fallback'
  | 'equation'
  | 'figure'

export const READER_REGRESSION_SOURCE_PATH = '__reader_regression__/fixture.md'
export const READER_REGRESSION_SOURCE_NAME = 'Fixture Paper'

const INTRO_P1 = 'Our method exploits neural radiance fields (NeRF) for snapshot compressed imaging. This paragraph intentionally stays long for reader regression scrolling, revisiting geometry consistency, calibration demands, and multi-view reconstruction stability across repeated reading passes.'
const INTRO_P2 = 'Conventional high-speed imaging systems often face challenges such as high hardware cost and storage requirements. This paragraph intentionally stays long for reader regression scrolling, expanding on memory limits, hardware bottlenecks, and the practical tradeoffs that motivate compressive capture.'
const METHOD_P1 = 'Given a set of input multi-view images, NeRF transfers the pixels of the input images into rays. This paragraph intentionally stays long for reader regression scrolling, extending the explanation with sampling behavior, density estimation, and view-dependent color prediction in the same section.'
const CONCLUSION_P1 = 'Our method achieves stable reconstruction from a single snapshot. This paragraph intentionally stays long for reader regression scrolling, summarizing robustness, recovery quality, and the broader implications for efficient scene capture in constrained settings.'

const FIGURE_DATA_URI = `data:image/svg+xml;utf8,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="220" viewBox="0 0 480 220"><rect width="480" height="220" fill="#eef3f8"/><rect x="28" y="36" width="180" height="128" rx="18" fill="#9bb7d4"/><rect x="234" y="58" width="218" height="26" rx="13" fill="#cbd8e6"/><rect x="234" y="98" width="184" height="20" rx="10" fill="#d6e0ea"/><rect x="234" y="130" width="146" height="20" rx="10" fill="#d6e0ea"/><circle cx="128" cy="100" r="34" fill="#f7fafc"/><text x="28" y="196" font-family="Georgia, serif" font-size="18" fill="#425466">Figure 1. SCI system pipeline.</text></svg>',
)}`

export const readerRegressionMarkdown = [
  '# Fixture Paper',
  '',
  '## 1. Introduction',
  '',
  INTRO_P1,
  '',
  INTRO_P2,
  '',
  '> SCI compresses a short video into one coded measurement.',
  '',
  '## 2. Method',
  '',
  METHOD_P1,
  '',
  '$$',
  'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
  '$$',
  '',
  'Figure 1 shows the SCI system pipeline.',
  '',
  `![Figure 1](${FIGURE_DATA_URI})`,
  '',
  '*Figure 1. SCI system pipeline.*',
  '',
  '## 3. Conclusion',
  '',
  CONCLUSION_P1,
  '',
].join('\n')

export const readerRegressionBlocks: ReaderDocBlock[] = [
  { doc_id: 'fixture-doc', block_id: 'h-intro', anchor_id: 'a-h-intro', kind: 'heading', heading_path: 'Fixture Paper / 1. Introduction', text: '1. Introduction', line_start: 3, line_end: 3 },
  { doc_id: 'fixture-doc', block_id: 'p-intro-1', anchor_id: 'a-p-intro-1', kind: 'paragraph', heading_path: 'Fixture Paper / 1. Introduction', text: INTRO_P1, line_start: 5, line_end: 5 },
  { doc_id: 'fixture-doc', block_id: 'p-intro-2', anchor_id: 'a-p-intro-2', kind: 'paragraph', heading_path: 'Fixture Paper / 1. Introduction', text: INTRO_P2, line_start: 7, line_end: 7 },
  { doc_id: 'fixture-doc', block_id: 'quote-1', anchor_id: 'a-quote-1', kind: 'blockquote', heading_path: 'Fixture Paper / 1. Introduction', text: 'SCI compresses a short video into one coded measurement.', line_start: 9, line_end: 9 },
  { doc_id: 'fixture-doc', block_id: 'h-method', anchor_id: 'a-h-method', kind: 'heading', heading_path: 'Fixture Paper / 2. Method', text: '2. Method', line_start: 11, line_end: 11 },
  { doc_id: 'fixture-doc', block_id: 'p-method-1', anchor_id: 'a-p-method-1', kind: 'paragraph', heading_path: 'Fixture Paper / 2. Method', text: METHOD_P1, line_start: 13, line_end: 13 },
  { doc_id: 'fixture-doc', block_id: 'eq-1', anchor_id: 'a-eq-1', kind: 'equation', heading_path: 'Fixture Paper / 2. Method', text: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt', number: 1, line_start: 15, line_end: 17 },
  { doc_id: 'fixture-doc', block_id: 'p-figure-ref', anchor_id: 'a-p-figure-ref', kind: 'paragraph', heading_path: 'Fixture Paper / 2. Method', text: 'Figure 1 shows the SCI system pipeline.', line_start: 19, line_end: 19 },
  { doc_id: 'fixture-doc', block_id: 'fig-1', anchor_id: 'a-fig-1', kind: 'figure', heading_path: 'Fixture Paper / 2. Method', text: 'Figure 1. SCI system pipeline.', number: 1, line_start: 21, line_end: 21 },
  { doc_id: 'fixture-doc', block_id: 'h-conclusion', anchor_id: 'a-h-conclusion', kind: 'heading', heading_path: 'Fixture Paper / 3. Conclusion', text: '3. Conclusion', line_start: 25, line_end: 25 },
  { doc_id: 'fixture-doc', block_id: 'p-conclusion-1', anchor_id: 'a-p-conclusion-1', kind: 'paragraph', heading_path: 'Fixture Paper / 3. Conclusion', text: CONCLUSION_P1, line_start: 27, line_end: 27 },
]

export const readerRegressionAnchors: ReaderDocAnchor[] = readerRegressionBlocks.map((block) => ({
  anchor_id: block.anchor_id,
  block_id: block.block_id,
  kind: block.kind,
  heading_path: block.heading_path,
  text: block.text,
  line_start: block.line_start,
  line_end: block.line_end,
  number: block.number,
}))

export const readerRegressionDocResponse = {
  ok: true,
  source_path: READER_REGRESSION_SOURCE_PATH,
  source_name: READER_REGRESSION_SOURCE_NAME,
  md_path: 'fixture.md',
  markdown: readerRegressionMarkdown,
  anchors: readerRegressionAnchors,
  blocks: readerRegressionBlocks,
}

export function buildReaderRegressionPayload(scenario: ReaderRegressionScenario): ReaderOpenPayload {
  if (scenario === 'evidence-nav') {
    return {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 1. Introduction',
      snippet: 'SCI compresses a short video into one coded measurement.',
      highlightSnippet: 'SCI compresses a short video into one coded measurement.',
      blockId: 'quote-1',
      anchorId: 'a-quote-1',
      anchorKind: 'quote',
      strictLocate: true,
      locateTarget: {
        segmentId: 'seg-quote-1',
        sourceSegmentId: 'seg-quote-1',
        headingPath: 'Fixture Paper / 1. Introduction',
        snippet: 'SCI compresses a short video into one coded measurement.',
        highlightSnippet: 'SCI compresses a short video into one coded measurement.',
        evidenceQuote: 'SCI compresses a short video into one coded measurement.',
        anchorText: 'SCI compresses a short video into one coded measurement.',
        blockId: 'quote-1',
        anchorId: 'a-quote-1',
        anchorKind: 'quote',
        claimType: 'quote_claim',
        locatePolicy: 'required',
        locateSurfacePolicy: 'primary',
      },
      evidenceAlternatives: [
        {
          headingPath: 'Fixture Paper / 1. Introduction',
          snippet: 'SCI compresses a short video into one coded measurement.',
          highlightSnippet: 'SCI compresses a short video into one coded measurement.',
          blockId: 'quote-1',
          anchorId: 'a-quote-1',
          anchorKind: 'quote',
        },
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
          highlightSnippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
          blockId: 'eq-1',
          anchorId: 'a-eq-1',
          anchorKind: 'equation',
          anchorNumber: 1,
        },
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'Figure 1',
          highlightSnippet: 'Figure 1',
          blockId: 'fig-1',
          anchorId: 'a-fig-1',
          anchorKind: 'figure',
          anchorNumber: 1,
        },
      ],
      alternatives: [
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
          highlightSnippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
          blockId: 'eq-1',
          anchorId: 'a-eq-1',
          anchorKind: 'equation',
          anchorNumber: 1,
        },
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'Figure 1',
          highlightSnippet: 'Figure 1',
          blockId: 'fig-1',
          anchorId: 'a-fig-1',
          anchorKind: 'figure',
          anchorNumber: 1,
        },
      ],
      initialAltIndex: 0,
    }
  }

  if (scenario === 'candidate-fallback') {
    return {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 3. Conclusion',
      snippet: 'SCI compresses a short video into one coded measurement.',
      highlightSnippet: 'SCI compresses a short video into one coded measurement.',
      blockId: 'missing-quote',
      anchorId: 'missing-quote-anchor',
      anchorKind: 'quote',
      strictLocate: true,
      locateTarget: {
        segmentId: 'seg-missing-quote',
        sourceSegmentId: 'seg-missing-quote',
        headingPath: 'Fixture Paper / 3. Conclusion',
        snippet: 'SCI compresses a short video into one coded measurement.',
        highlightSnippet: 'SCI compresses a short video into one coded measurement.',
        evidenceQuote: 'SCI compresses a short video into one coded measurement.',
        anchorText: 'SCI compresses a short video into one coded measurement.',
        blockId: 'missing-quote',
        anchorId: 'missing-quote-anchor',
        anchorKind: 'quote',
        claimType: 'quote_claim',
        locatePolicy: 'required',
        locateSurfacePolicy: 'primary',
      },
      alternatives: [
        {
          headingPath: 'Fixture Paper / 1. Introduction',
          snippet: 'SCI compresses a short video into one coded measurement.',
          highlightSnippet: 'SCI compresses a short video into one coded measurement.',
          blockId: 'quote-1',
          anchorId: 'a-quote-1',
          anchorKind: 'quote',
        },
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
          highlightSnippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
          blockId: 'eq-1',
          anchorId: 'a-eq-1',
          anchorKind: 'equation',
          anchorNumber: 1,
        },
      ],
      visibleAlternatives: [
        {
          headingPath: 'Fixture Paper / 3. Conclusion',
          snippet: 'SCI compresses a short video into one coded measurement.',
          highlightSnippet: 'SCI compresses a short video into one coded measurement.',
          blockId: 'missing-quote',
          anchorId: 'missing-quote-anchor',
          anchorKind: 'quote',
        },
        {
          headingPath: 'Fixture Paper / 1. Introduction',
          snippet: 'SCI compresses a short video into one coded measurement.',
          highlightSnippet: 'SCI compresses a short video into one coded measurement.',
          blockId: 'quote-1',
          anchorId: 'a-quote-1',
          anchorKind: 'quote',
        },
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
          highlightSnippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
          blockId: 'eq-1',
          anchorId: 'a-eq-1',
          anchorKind: 'equation',
          anchorNumber: 1,
        },
      ],
      initialAltIndex: 0,
    }
  }

  if (scenario === 'equation') {
    return {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 2. Method',
      snippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
      highlightSnippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
      blockId: 'eq-1',
      anchorId: 'a-eq-1',
      anchorKind: 'equation',
      anchorNumber: 1,
      strictLocate: true,
      locateTarget: {
        segmentId: 'seg-eq-1',
        sourceSegmentId: 'seg-eq-1',
        headingPath: 'Fixture Paper / 2. Method',
        snippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
        highlightSnippet: 'C(r) = \\int_{t_n}^{t_f} T(t)\\sigma(r(t)) c(r(t), d) dt',
        blockId: 'eq-1',
        anchorId: 'a-eq-1',
        anchorKind: 'equation',
        anchorNumber: 1,
        claimType: 'formula_claim',
        locatePolicy: 'required',
        locateSurfacePolicy: 'primary',
      },
    }
  }

  if (scenario === 'figure') {
    return {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 2. Method',
      snippet: 'Figure 1',
      highlightSnippet: 'Figure 1',
      blockId: 'fig-1',
      anchorId: 'a-fig-1',
      anchorKind: 'figure',
      anchorNumber: 1,
      strictLocate: true,
      locateTarget: {
        segmentId: 'seg-fig-1',
        sourceSegmentId: 'seg-fig-1',
        headingPath: 'Fixture Paper / 2. Method',
        snippet: 'Figure 1',
        highlightSnippet: 'Figure 1',
        blockId: 'fig-1',
        anchorId: 'a-fig-1',
        anchorKind: 'figure',
        anchorNumber: 1,
        claimType: 'figure_claim',
        locatePolicy: 'required',
        locateSurfacePolicy: 'primary',
      },
    }
  }

  return {
    sourcePath: READER_REGRESSION_SOURCE_PATH,
    sourceName: READER_REGRESSION_SOURCE_NAME,
    headingPath: 'Fixture Paper / 1. Introduction',
    snippet: 'SCI compresses a short video into one coded measurement.',
    highlightSnippet: 'SCI compresses a short video into one coded measurement.',
    blockId: 'quote-1',
    anchorId: 'a-quote-1',
    anchorKind: 'quote',
    strictLocate: true,
    locateTarget: {
      segmentId: 'seg-quote-1',
      sourceSegmentId: 'seg-quote-1',
      headingPath: 'Fixture Paper / 1. Introduction',
      snippet: 'SCI compresses a short video into one coded measurement.',
      highlightSnippet: 'SCI compresses a short video into one coded measurement.',
      evidenceQuote: 'SCI compresses a short video into one coded measurement.',
      anchorText: 'SCI compresses a short video into one coded measurement.',
      blockId: 'quote-1',
      anchorId: 'a-quote-1',
      anchorKind: 'quote',
      claimType: 'quote_claim',
      locatePolicy: 'required',
      locateSurfacePolicy: 'primary',
    },
  }
}
