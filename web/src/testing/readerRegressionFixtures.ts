import type { ReaderDocAnchor, ReaderDocBlock } from '../api/references'
import type { ReaderOpenPayload } from '../components/chat/reader/readerTypes'

export type ReaderRegressionScenario =
  | 'strict-quote'
  | 'evidence-nav'
  | 'duplicate-sections'
  | 'duplicate-images'
  | 'candidate-fallback'
  | 'strict-missing-exact'
  | 'discussion-only'
  | 'limitations-only'
  | 'future-work-only'
  | 'equation'
  | 'figure'
  | 'multi-panel'

export const READER_REGRESSION_SOURCE_PATH = '__reader_regression__/fixture.md'
export const READER_REGRESSION_SOURCE_NAME = 'Fixture Paper'

const INTRO_P1 = 'Our method exploits neural radiance fields (NeRF) for snapshot compressed imaging. This paragraph intentionally stays long for reader regression scrolling, revisiting geometry consistency, calibration demands, and multi-view reconstruction stability across repeated reading passes.'
const INTRO_P2 = 'Conventional high-speed imaging systems often face challenges such as high hardware cost and storage requirements. This paragraph intentionally stays long for reader regression scrolling, expanding on memory limits, hardware bottlenecks, and the practical tradeoffs that motivate compressive capture.'
const METHOD_P1 = 'Given a set of input multi-view images, NeRF transfers the pixels of the input images into rays. This paragraph intentionally stays long for reader regression scrolling, extending the explanation with sampling behavior, density estimation, and view-dependent color prediction in the same section.'
const CONCLUSION_P1 = 'Our method achieves stable reconstruction from a single snapshot. This paragraph intentionally stays long for reader regression scrolling, summarizing robustness, recovery quality, and the broader implications for efficient scene capture in constrained settings.'
const DISCUSSION_P1 = 'The discussion emphasizes a practical tradeoff: faster capture is appealing, but reconstruction stability still depends on calibration quality, measurement design, and how much prior structure the pipeline assumes about the scene.'
const LIMITATIONS_P1 = 'A current limitation is that the method still trades temporal coverage against reconstruction stability, especially when the scene departs from the static-scene assumption used by the reconstruction pipeline.'
const FUTURE_WORK_P1 = 'Looking ahead, the most direct extension would be to combine the current pipeline with adaptive masking so dynamic scenes can be reconstructed more faithfully without increasing the hardware burden.'

const FIGURE_DATA_URI = '/reader-regression-figure.svg'

const MULTI_PANEL_SNIPPET = 'f Resulting iPSF from iISM after adaptive pixel-reassignment (APR), with same incident illumination power and number of detected photons. g Line profiles of the iPSF in the three configurations as indicated in d-f.'

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
  `**Figure 2.** ${MULTI_PANEL_SNIPPET}`,
  '',
  '## 4. Discussion',
  '',
  DISCUSSION_P1,
  '',
  '## 5. Limitations',
  '',
  LIMITATIONS_P1,
  '',
  '## 6. Future Work',
  '',
  FUTURE_WORK_P1,
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
  { doc_id: 'fixture-doc', block_id: 'fig-1', anchor_id: 'a-fig-1', kind: 'figure', heading_path: 'Fixture Paper / 2. Method / Figure 1', text: 'Figure 1. SCI system pipeline.', number: 1, line_start: 21, line_end: 21 },
  { doc_id: 'fixture-doc', block_id: 'h-conclusion', anchor_id: 'a-h-conclusion', kind: 'heading', heading_path: 'Fixture Paper / 3. Conclusion', text: '3. Conclusion', line_start: 25, line_end: 25 },
  { doc_id: 'fixture-doc', block_id: 'p-conclusion-1', anchor_id: 'a-p-conclusion-1', kind: 'paragraph', heading_path: 'Fixture Paper / 3. Conclusion', text: CONCLUSION_P1, line_start: 27, line_end: 27 },
  { doc_id: 'fixture-doc', block_id: 'p-fig-panels', anchor_id: 'a-p-fig-panels', kind: 'paragraph', heading_path: 'Fixture Paper / 3. Conclusion', text: `Figure 2. ${MULTI_PANEL_SNIPPET}`, line_start: 29, line_end: 29 },
  { doc_id: 'fixture-doc', block_id: 'h-discussion', anchor_id: 'a-h-discussion', kind: 'heading', heading_path: 'Fixture Paper / 4. Discussion', text: '4. Discussion', line_start: 31, line_end: 31 },
  { doc_id: 'fixture-doc', block_id: 'p-discussion-1', anchor_id: 'a-p-discussion-1', kind: 'paragraph', heading_path: 'Fixture Paper / 4. Discussion', text: DISCUSSION_P1, line_start: 33, line_end: 33 },
  { doc_id: 'fixture-doc', block_id: 'h-limitations', anchor_id: 'a-h-limitations', kind: 'heading', heading_path: 'Fixture Paper / 5. Limitations', text: '5. Limitations', line_start: 35, line_end: 35 },
  { doc_id: 'fixture-doc', block_id: 'p-limitations-1', anchor_id: 'a-p-limitations-1', kind: 'paragraph', heading_path: 'Fixture Paper / 5. Limitations', text: LIMITATIONS_P1, line_start: 37, line_end: 37 },
  { doc_id: 'fixture-doc', block_id: 'h-future-work', anchor_id: 'a-h-future-work', kind: 'heading', heading_path: 'Fixture Paper / 6. Future Work', text: '6. Future Work', line_start: 39, line_end: 39 },
  { doc_id: 'fixture-doc', block_id: 'p-future-work-1', anchor_id: 'a-p-future-work-1', kind: 'paragraph', heading_path: 'Fixture Paper / 6. Future Work', text: FUTURE_WORK_P1, line_start: 41, line_end: 41 },
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

const readerRegressionDuplicateImageMarkdown = [
  '# Fixture Paper',
  '',
  '## 2. Method',
  '',
  'Figure 1 shows the SCI system pipeline.',
  '',
  `![Figure 1](${FIGURE_DATA_URI})`,
  '',
  `![Figure 1](${FIGURE_DATA_URI})`,
  '',
  `![Figure 1](${FIGURE_DATA_URI})`,
  '',
  '*Figure 1. SCI system pipeline.*',
  '',
].join('\n')

export function buildReaderRegressionDocResponse(scenario: ReaderRegressionScenario) {
  if (scenario === 'duplicate-images') {
    return {
      ok: true,
      source_path: READER_REGRESSION_SOURCE_PATH,
      source_name: READER_REGRESSION_SOURCE_NAME,
      md_path: 'fixture-duplicate-images.md',
      markdown: readerRegressionDuplicateImageMarkdown,
      anchors: [] as ReaderDocAnchor[],
      blocks: [] as ReaderDocBlock[],
    }
  }
  return readerRegressionDocResponse
}

export function buildReaderRegressionPayload(scenario: ReaderRegressionScenario): ReaderOpenPayload {
  if (scenario === 'duplicate-images') {
    return {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 2. Method',
      snippet: 'Figure 1 shows the SCI system pipeline.',
      highlightSnippet: 'Figure 1 shows the SCI system pipeline.',
      strictLocate: false,
    }
  }

  if (scenario === 'multi-panel') {
    return {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 3. Conclusion',
      snippet: MULTI_PANEL_SNIPPET,
      highlightSnippet: MULTI_PANEL_SNIPPET,
      blockId: 'p-fig-panels',
      anchorId: 'a-p-fig-panels',
      anchorKind: 'paragraph',
      strictLocate: true,
      locateTarget: {
        segmentId: 'seg-fig-panels',
        sourceSegmentId: 'seg-fig-panels',
        headingPath: 'Fixture Paper / 3. Conclusion',
        snippet: MULTI_PANEL_SNIPPET,
        highlightSnippet: MULTI_PANEL_SNIPPET,
        evidenceQuote: MULTI_PANEL_SNIPPET,
        anchorText: MULTI_PANEL_SNIPPET,
        blockId: 'p-fig-panels',
        anchorId: 'a-p-fig-panels',
        anchorKind: 'paragraph',
        claimType: 'quote_claim',
        locatePolicy: 'required',
        locateSurfacePolicy: 'primary',
      },
    }
  }

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

  if (scenario === 'duplicate-sections') {
    return {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 2. Method',
      snippet: METHOD_P1,
      highlightSnippet: METHOD_P1,
      blockId: 'p-method-1',
      anchorId: 'a-p-method-1',
      anchorKind: 'paragraph',
      strictLocate: true,
      locateTarget: {
        segmentId: 'seg-method-1',
        sourceSegmentId: 'seg-method-1',
        headingPath: 'Fixture Paper / 2. Method',
        snippet: METHOD_P1,
        highlightSnippet: METHOD_P1,
        evidenceQuote: METHOD_P1,
        anchorText: METHOD_P1,
        blockId: 'p-method-1',
        anchorId: 'a-p-method-1',
        anchorKind: 'paragraph',
        claimType: 'quote_claim',
        locatePolicy: 'required',
        locateSurfacePolicy: 'primary',
      },
      evidenceAlternatives: [
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: METHOD_P1,
          highlightSnippet: METHOD_P1,
          blockId: 'p-method-1',
          anchorId: 'a-p-method-1',
          anchorKind: 'paragraph',
        },
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'The method section keeps extending the discussion with another supporting paragraph.',
          highlightSnippet: 'The method section keeps extending the discussion with another supporting paragraph.',
          blockId: 'p-method-dup-1',
          anchorId: 'a-p-method-dup-1',
          anchorKind: 'paragraph',
        },
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'A third method paragraph would otherwise create another identical-looking list item.',
          highlightSnippet: 'A third method paragraph would otherwise create another identical-looking list item.',
          blockId: 'p-method-dup-2',
          anchorId: 'a-p-method-dup-2',
          anchorKind: 'paragraph',
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
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'Figure 1 shows the SCI system pipeline.',
          highlightSnippet: 'Figure 1 shows the SCI system pipeline.',
          blockId: 'p-figure-ref',
          anchorId: 'a-p-figure-ref',
          anchorKind: 'paragraph',
        },
      ],
      alternatives: [
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'The method section keeps extending the discussion with another supporting paragraph.',
          highlightSnippet: 'The method section keeps extending the discussion with another supporting paragraph.',
          blockId: 'p-method-dup-1',
          anchorId: 'a-p-method-dup-1',
          anchorKind: 'paragraph',
        },
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'A third method paragraph would otherwise create another identical-looking list item.',
          highlightSnippet: 'A third method paragraph would otherwise create another identical-looking list item.',
          blockId: 'p-method-dup-2',
          anchorId: 'a-p-method-dup-2',
          anchorKind: 'paragraph',
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
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'Figure 1 shows the SCI system pipeline.',
          highlightSnippet: 'Figure 1 shows the SCI system pipeline.',
          blockId: 'p-figure-ref',
          anchorId: 'a-p-figure-ref',
          anchorKind: 'paragraph',
        },
      ],
      visibleAlternatives: [
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: METHOD_P1,
          highlightSnippet: METHOD_P1,
          blockId: 'p-method-1',
          anchorId: 'a-p-method-1',
          anchorKind: 'paragraph',
        },
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'The method section keeps extending the discussion with another supporting paragraph.',
          highlightSnippet: 'The method section keeps extending the discussion with another supporting paragraph.',
          blockId: 'p-method-dup-1',
          anchorId: 'a-p-method-dup-1',
          anchorKind: 'paragraph',
        },
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'A third method paragraph would otherwise create another identical-looking list item.',
          highlightSnippet: 'A third method paragraph would otherwise create another identical-looking list item.',
          blockId: 'p-method-dup-2',
          anchorId: 'a-p-method-dup-2',
          anchorKind: 'paragraph',
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
        {
          headingPath: 'Fixture Paper / 2. Method',
          snippet: 'Figure 1 shows the SCI system pipeline.',
          highlightSnippet: 'Figure 1 shows the SCI system pipeline.',
          blockId: 'p-figure-ref',
          anchorId: 'a-p-figure-ref',
          anchorKind: 'paragraph',
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

  if (scenario === 'strict-missing-exact') {
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
        segmentId: 'seg-missing-exact',
        sourceSegmentId: 'seg-missing-exact',
        headingPath: 'Fixture Paper / 3. Conclusion',
        snippet: 'SCI compresses a short video into one coded measurement.',
        highlightSnippet: 'SCI compresses a short video into one coded measurement.',
        evidenceQuote: 'SCI compresses a short video into one coded measurement.',
        anchorText: 'SCI compresses a short video into one coded measurement.',
        blockId: 'missing-quote',
        anchorId: 'missing-quote-anchor',
        anchorKind: 'quote',
        hitLevel: 'exact',
        claimType: 'quote_claim',
        locatePolicy: 'required',
        locateSurfacePolicy: 'primary',
      },
    }
  }

  if (scenario === 'discussion-only') {
    return {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 4. Discussion',
      strictLocate: true,
      locateTarget: {
        segmentId: 'seg-discussion-heading',
        sourceSegmentId: 'seg-discussion-heading',
        headingPath: 'Fixture Paper / 4. Discussion',
        hitLevel: 'heading',
        claimType: 'section_claim',
        locatePolicy: 'required',
        locateSurfacePolicy: 'primary',
      },
    }
  }

  if (scenario === 'limitations-only') {
    return {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 5. Limitations',
      strictLocate: true,
      locateTarget: {
        segmentId: 'seg-limitations-heading',
        sourceSegmentId: 'seg-limitations-heading',
        headingPath: 'Fixture Paper / 5. Limitations',
        hitLevel: 'heading',
        claimType: 'section_claim',
        locatePolicy: 'required',
        locateSurfacePolicy: 'primary',
      },
    }
  }

  if (scenario === 'future-work-only') {
    return {
      sourcePath: READER_REGRESSION_SOURCE_PATH,
      sourceName: READER_REGRESSION_SOURCE_NAME,
      headingPath: 'Fixture Paper / 6. Future Work',
      strictLocate: true,
      locateTarget: {
        segmentId: 'seg-future-work-heading',
        sourceSegmentId: 'seg-future-work-heading',
        headingPath: 'Fixture Paper / 6. Future Work',
        hitLevel: 'heading',
        claimType: 'section_claim',
        locatePolicy: 'required',
        locateSurfacePolicy: 'primary',
      },
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
      headingPath: 'Fixture Paper / 2. Method / Figure 1',
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
        headingPath: 'Fixture Paper / 2. Method / Figure 1',
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
