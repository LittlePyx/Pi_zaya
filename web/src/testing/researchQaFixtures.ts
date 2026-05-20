import type { Message } from '../api/chat'
import researchQaData from './researchQaData.json'

const DB_ROOT = 'F:/research-papers/2026/Jan/else/kb_chat/db'
const BASE_TIME = Date.parse('2026-05-20T10:00:00+08:00')

export interface ResearchLibraryDoc {
  id: string
  title: string
  sourcePath: string
  topic: string
  shortLabel: string
}

interface ResearchCitationOptions {
  num: number
  anchor: string
  doc: ResearchLibraryDoc
  headingPath: string
  evidenceQuote: string
  answerClaim: string
  supportRelation: string
  blockId: string
  anchorId: string
  anchorKind?: string
  pageStart?: number
  score?: number
}

interface ResearchSystemBOptions {
  num: number
  anchor: string
  citingDoc: ResearchLibraryDoc
  headingPath: string
  context: string
  answerClaim: string
  upstreamWorkRole: string
  userQuestionRelation: string
  raw: string
  title: string
  authors: string
  venue: string
  year: string
  doi?: string
  blockId: string
  anchorId: string
}

interface ResearchRefOptions {
  doc: ResearchLibraryDoc
  headingPath: string
  summaryLine: string
  whyLine: string
  snippet: string
  score: number
  year?: string
  venue?: string
  blockId?: string
  anchorId?: string
  anchorKind?: string
}

export interface ResearchQaCase {
  id: string
  docIds: string[]
  question: string
  answerMarkdown: string
  citeDetails: Array<Record<string, unknown>>
  refs: Array<Record<string, unknown>>
  acceptance: string[]
  userMessageId: number
  assistantMessageId: number
}

interface ResearchQaDataCase {
  id: string
  docIds: string[]
  question: string
  acceptance: string[]
}

const RESEARCH_QA_CASE_META_BY_ID = new Map(
  (researchQaData.cases as ResearchQaDataCase[]).map((item) => [item.id, item]),
)

function caseMeta(id: string): ResearchQaDataCase {
  const found = RESEARCH_QA_CASE_META_BY_ID.get(id)
  if (!found) throw new Error(`Unknown research QA case: ${id}`)
  return found
}

function mdPath(dir: string, file = dir) {
  return `${DB_ROOT}/${dir}/${file}.en.md`
}

export const RESEARCH_LIBRARY_DOCS: ResearchLibraryDoc[] = [
  {
    id: 'qclfm',
    title: 'Quantum correlation light-field microscope with extreme depth of field',
    sourcePath: mdPath('arXiv-Quantum correlation light-ﬁeld microscope with extreme depth of ﬁeld'),
    topic: 'quantum correlation microscopy',
    shortLabel: 'QCLFM',
  },
  {
    id: 'scinerf',
    title: 'SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image',
    sourcePath: mdPath('CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image'),
    topic: 'SCI plus NeRF 3D reconstruction',
    shortLabel: 'SCINeRF',
  },
  {
    id: 'spd-review',
    title: 'Emerging single-photon detection technique for high-performance photodetector',
    sourcePath: mdPath('Frontiers of Physics-2024-Emerging single-photon...performance photodetector'),
    topic: 'single-photon detectors',
    shortLabel: 'SPD review',
  },
  {
    id: 'scigs',
    title: 'SCIGS: 3D Gaussians Splatting from A Snapshot Compressive Image',
    sourcePath: mdPath('ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image'),
    topic: 'SCI plus 3D Gaussian splatting',
    shortLabel: 'SCIGS',
  },
  {
    id: '3d-sp-video',
    title: '3D single-pixel video',
    sourcePath: mdPath('Journal of Optics-2016-3D single-pixel video'),
    topic: 'video-rate 3D single-pixel imaging',
    shortLabel: '3D SPI video',
  },
  {
    id: 'dl-spi-review',
    title: 'Advances and Challenges of Single-Pixel Imaging Based on Deep Learning',
    sourcePath: mdPath('LPR-2025-Advances and Challenges of Single‐Pixel Imaging Based on Deep Learning'),
    topic: 'deep learning single-pixel imaging review',
    shortLabel: 'DL-SPI review',
  },
  {
    id: 'iism',
    title: 'Interferometric Image Scanning Microscopy with enhanced lateral resolution inside live cells',
    sourcePath: mdPath('LSA-2026-Interferometric Image Scanning...lateral resolution inside live cells'),
    topic: 'interferometric image scanning microscopy',
    shortLabel: 'iISM',
  },
  {
    id: 'sph-biological',
    title: 'Imaging biological tissue with high-throughput single-pixel compressive holography',
    sourcePath: mdPath('NatCommun-2021-Imaging biological tissue with...pixel compressive holography'),
    topic: 'single-pixel compressive holography',
    shortLabel: 'SPH biology',
  },
  {
    id: 'pidl-single-photon',
    title: 'High-resolution single-photon imaging with physics-informed deep learning',
    sourcePath: mdPath('NatCommun-2023-High-resolution single-photon imaging with physics-informed deep learning'),
    topic: 'physics-informed single-photon imaging',
    shortLabel: 'PI single-photon',
  },
  {
    id: 'spi-prospects',
    title: 'Principles and prospects for single-pixel imaging',
    sourcePath: mdPath('NatPhoton-2019-Principles and prospects for single-pixel imaging'),
    topic: 'single-pixel imaging foundations',
    shortLabel: 'SPI prospects',
  },
  {
    id: 's2ism',
    title: 'Structured detection for high-SNR image scanning microscopy in thick samples',
    sourcePath: mdPath('NatPhoton-2025-Structured detection for...in laser scanning microscopy'),
    topic: 'structured detection microscopy',
    shortLabel: 's2ISM',
  },
  {
    id: 'perovskite-laser',
    title: 'Electrically driven lasing from a dual-cavity perovskite device',
    sourcePath: mdPath('Nature-2025-Electrically driven lasing from a dual-cavity perovskite device'),
    topic: 'perovskite optoelectronic device',
    shortLabel: 'Perovskite laser',
  },
  {
    id: 'cassi',
    title: 'Single-shot compressive spectral imaging with a dual-disperser architecture',
    sourcePath: mdPath('OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture'),
    topic: 'compressive spectral imaging',
    shortLabel: 'CASSI',
  },
  {
    id: 'hsi-fsi',
    title: 'Hadamard single-pixel imaging versus Fourier single-pixel imaging',
    sourcePath: mdPath('OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging'),
    topic: 'Hadamard versus Fourier SPI',
    shortLabel: 'HSI vs FSI',
  },
  {
    id: 'fdm-metamaterials',
    title: 'Frequency-division-multiplexed single-pixel imaging with metamaterials',
    sourcePath: mdPath('Optica-2016-Frequency-division-multiplexed single-pixel imaging with metamaterials'),
    topic: 'metamaterial single-pixel imaging',
    shortLabel: 'FDM metamaterials',
  },
  {
    id: 'rt-spi-supp',
    title: 'Robust real-time single-pixel imaging with differential detection - supplement',
    sourcePath: mdPath('Optica-2024-Robust real-time single-pixel...differential detection- supplement'),
    topic: 'real-time SPI supplement',
    shortLabel: 'RT-SPI supp',
  },
  {
    id: 'piln',
    title: 'Part-based image-loop network for single-pixel imaging',
    sourcePath: mdPath('Optics & Laser Technology-2024-Part-based image-loop network for single-pixel imaging'),
    topic: 'single-pixel image reconstruction network',
    shortLabel: 'PILN',
  },
  {
    id: 'visual-perception',
    title: 'Some informational aspects of visual perception',
    sourcePath: mdPath('Psychological Review-1954-Some informational aspects of visual perception'),
    topic: 'visual perception and information',
    shortLabel: 'Visual perception',
  },
  {
    id: 'foveated-spi',
    title: 'Adaptive foveated single-pixel imaging with dynamic supersampling',
    sourcePath: mdPath('SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling'),
    topic: 'adaptive foveated single-pixel imaging',
    shortLabel: 'Foveated SPI',
  },
  {
    id: 'seq-cs',
    title: 'Sequentially designed compressed sensing',
    sourcePath: mdPath('SSP-2012-Sequentially designed compressed sensing'),
    topic: 'adaptive compressed sensing',
    shortLabel: 'Sequential CS',
  },
  {
    id: 'denoising-review',
    title: 'Brief review of image denoising techniques',
    sourcePath: mdPath('Visual Computing for Industry, Biomedicine, and Art-2019-Brief review...techniques'),
    topic: 'image denoising review',
    shortLabel: 'Denoising review',
  },
]

function doc(id: string) {
  const found = RESEARCH_LIBRARY_DOCS.find((item) => item.id === id)
  if (!found) throw new Error(`Unknown research library doc: ${id}`)
  return found
}

function systemA(opts: ResearchCitationOptions): Record<string, unknown> {
  return {
    num: opts.num,
    anchor: opts.anchor,
    source_name: opts.doc.title,
    source_path: opts.doc.sourcePath,
    title: opts.doc.title,
    heading_path: opts.headingPath,
    evidence_quote: opts.evidenceQuote,
    answer_claim: opts.answerClaim,
    support_relation: opts.supportRelation,
    why_line: opts.supportRelation,
    block_id: opts.blockId,
    anchor_id: opts.anchorId,
    anchor_kind: opts.anchorKind || 'sentence',
    page_start: opts.pageStart || 1,
    score: opts.score || 9.5,
  }
}

function systemB(opts: ResearchSystemBOptions): Record<string, unknown> {
  return {
    num: opts.num,
    anchor: opts.anchor,
    source_name: opts.citingDoc.title,
    source_path: opts.citingDoc.sourcePath,
    is_inpaper: true,
    title: opts.title,
    authors: opts.authors,
    venue: opts.venue,
    year: opts.year,
    doi: opts.doi || '',
    raw: opts.raw,
    cite_fmt: opts.raw,
    heading_path: opts.headingPath,
    evidence_quote: opts.context,
    citation_context: opts.context,
    citation_context_source: 'citing_paper_related_work',
    answer_claim: opts.answerClaim,
    upstream_work_role: opts.upstreamWorkRole,
    user_question_relation: opts.userQuestionRelation,
    support_relation: opts.userQuestionRelation,
    why_line: opts.upstreamWorkRole,
    block_id: opts.blockId,
    anchor_id: opts.anchorId,
    anchor_kind: 'sentence',
    score: 9.2,
  }
}

function refHit(opts: ResearchRefOptions): Record<string, unknown> {
  return {
    text: opts.snippet,
    meta: {
      source_path: opts.doc.sourcePath,
      ref_pack_state: 'ready',
      heading_path: opts.headingPath,
      ref_show_snippets: [opts.snippet],
      ref_locs: [
        {
          heading_path: opts.headingPath,
          text: opts.snippet,
          block_id: opts.blockId || '',
          anchor_id: opts.anchorId || '',
          anchor_kind: opts.anchorKind || 'sentence',
        },
      ],
    },
    ui_meta: {
      source_path: opts.doc.sourcePath,
      display_name: opts.doc.title,
      heading_path: opts.headingPath,
      score: opts.score,
      summary_label: '导读',
      summary_title: '这条证据说明什么',
      summary_line: opts.summaryLine,
      why_line: opts.whyLine,
      can_open: true,
      citation_meta: {
        title: opts.doc.title,
        venue: opts.venue || opts.doc.topic,
        year: opts.year || '',
        source_name: opts.doc.title,
        source_path: opts.doc.sourcePath,
      },
      reader_open: {
        sourcePath: opts.doc.sourcePath,
        sourceName: opts.doc.title,
        headingPath: opts.headingPath,
        snippet: opts.snippet,
        highlightSnippet: opts.snippet,
        blockId: opts.blockId || undefined,
        anchorId: opts.anchorId || undefined,
        anchorKind: opts.anchorKind || 'sentence',
        strictLocate: Boolean(opts.blockId || opts.anchorId),
      },
    },
  }
}

type ResearchQaCaseInput = Omit<
  ResearchQaCase,
  'userMessageId' | 'assistantMessageId' | 'docIds' | 'question' | 'acceptance'
> &
  Partial<Pick<ResearchQaCase, 'docIds' | 'question' | 'acceptance'>>

function makeCase(index: number, value: ResearchQaCaseInput): ResearchQaCase {
  const meta = caseMeta(value.id)
  const userMessageId = 12000 + index * 2
  return {
    ...value,
    docIds: meta.docIds,
    question: meta.question,
    acceptance: meta.acceptance,
    userMessageId,
    assistantMessageId: userMessageId + 1,
  }
}

const scigs = doc('scigs')
const scinerf = doc('scinerf')
const hsiFsi = doc('hsi-fsi')
const foveated = doc('foveated-spi')
const dlSpiReview = doc('dl-spi-review')
const qclfm = doc('qclfm')
const s2ism = doc('s2ism')
const pidlSinglePhoton = doc('pidl-single-photon')
const perovskiteLaser = doc('perovskite-laser')

export const RESEARCH_QA_CASES: ResearchQaCase[] = [
  makeCase(0, {
    id: 'scigs-dynamic-3d',
    docIds: ['scigs', 'scinerf'],
    question: 'SCIGS 这篇到底想解决什么问题？它和 SCINeRF 的区别在哪里？',
    answerMarkdown: [
      'SCIGS 的目标不是再做一个普通 SCI 帧解码器，而是把单张 snapshot compressed image 里的高速动态信息恢复成一个可渲染的显式 3D 场景。它指出深度学习解码器容易缺少 3D 结构一致性，而 NeRF 路线在动态场景上还有压力；因此 SCIGS 用 3DGS 变体、primitive-level transformation network 和高频滤波来处理动态 3D 重建。[1](#qa-scigs-a1)',
      '',
      '和 SCINeRF 相比，两者都从 SCI 出发，但 SCINeRF 主要把 SCI 成像过程写进 NeRF 的 test-time optimization，用隐式场景表示来保证多视角一致性；SCIGS 则把核心表示换成显式 Gaussian primitives，更强调动态场景和效率。[2](#qa-scigs-a2)',
    ].join('\n'),
    citeDetails: [
      systemA({
        num: 1,
        anchor: 'qa-scigs-a1',
        doc: scigs,
        headingPath: 'SCIGS / Abstract',
        evidenceQuote: 'SCIGS is proposed as a variant of 3DGS and the first method to reconstruct a 3D explicit scene from a single compressed image, extending to dynamic 3D scenes.',
        answerClaim: 'SCIGS 面向从单张 SCI 压缩图恢复动态显式 3D 场景，而不是只恢复视频帧。',
        supportRelation: '摘要同时交代了深度学习、NeRF 方法的痛点，以及 SCIGS 的 3DGS 变体和动态场景目标。',
        blockId: 'scigs-abstract',
        anchorId: 'scigs-abstract-a',
        pageStart: 1,
      }),
      systemA({
        num: 2,
        anchor: 'qa-scigs-a2',
        doc: scinerf,
        headingPath: 'SCINeRF / 1. Introduction',
        evidenceQuote: 'SCINeRF recovers the underlying 3D scene representation from a single compressed image and jointly optimizes camera poses and NeRF.',
        answerClaim: 'SCINeRF 的核心是把 SCI 物理成像过程放进 NeRF 训练，并联合优化相机位姿。',
        supportRelation: '这段能解释 SCINeRF 的路线，因此适合与 SCIGS 的 3DGS 显式表示做对照。',
        blockId: 'scinerf-intro-nerf',
        anchorId: 'scinerf-intro-nerf-a',
        pageStart: 2,
      }),
    ],
    refs: [
      refHit({
        doc: scigs,
        headingPath: 'SCIGS / Abstract',
        summaryLine: '摘要直接把问题界定为：SCI 能捕获高速动态信息，但现有深度学习和 NeRF 解码都不足以稳定处理动态 3D 场景。',
        whyLine: '用户问“想解决什么”和“与 SCINeRF 的区别”，这条命中同时给出痛点、方法选择和动态场景目标。',
        snippet: 'current deep learning-based and NeRF-based reconstruction methods face challenges... we propose SCIGS, a variant of 3DGS...',
        score: 9.78,
        year: '2025',
        venue: 'ICIP',
        blockId: 'scigs-abstract',
        anchorId: 'scigs-abstract-a',
      }),
      refHit({
        doc: scinerf,
        headingPath: 'SCINeRF / 1. Introduction',
        summaryLine: 'SCINeRF 说明它通过 NeRF 和相机位姿联合优化，从单张压缩图恢复 3D 场景表示。',
        whyLine: '这条证据用于建立对照：SCINeRF 是 NeRF 隐式表示路线，SCIGS 是 3DGS 显式表示路线。',
        snippet: 'we conduct a joint optimization on both the camera poses and NeRF, via minimizing the difference between the synthesized compressed image and real measurement',
        score: 9.31,
        year: '2024',
        venue: 'CVPR',
        blockId: 'scinerf-intro-nerf',
        anchorId: 'scinerf-intro-nerf-a',
      }),
    ],
    acceptance: [
      '回答先解释研究问题，再比较两篇论文路线。',
      '至少出现一个 SCIGS 原文证据和一个 SCINeRF 对照证据。',
      '参考卡文字不能只说“直接相关”，要说明命中段落承担的判断。',
    ],
  }),
  makeCase(1, {
    id: 'scinerf-admm-origin',
    docIds: ['scinerf'],
    question: 'ADMM 是作者自己发明的吗？我应该把它当成这篇论文的新东西吗？',
    answerMarkdown: [
      '不是。SCINeRF 把 ADMM 放在 Related Work 里讲的是已有 SCI 优化方法常用的求解框架，不是这篇论文的原创贡献。这里真正的新意是把 SCI 物理成像约束和 NeRF 的场景表示结合起来，做单张压缩图到 3D 场景的 test-time optimization。[1](#qa-admm-a1)',
      '',
      '如果你沿着“ADMM 从哪来”继续读，优先点开文内编号 [R4](#qa-admm-b4)：它是 SCINeRF 引用的 ADMM 基础文献，作用是解释 Related Work 里“existing methods employ ADMM”的上游来源。',
    ].join('\n'),
    citeDetails: [
      systemA({
        num: 1,
        anchor: 'qa-admm-a1',
        doc: scinerf,
        headingPath: 'SCINeRF / 2. Related Work / Snapshot Compressive Imaging',
        evidenceQuote: 'most of the existing methods employ alternating direction method of multipliers (ADMM) [4], which leads to good results and easier to adapt to different systems.',
        answerClaim: 'ADMM 在这篇里是背景方法，不是 SCINeRF 的原创贡献。',
        supportRelation: 'Related Work 明确说 existing methods employ ADMM，并把它接到编号 [4]。',
        blockId: 'scinerf-rw-admm',
        anchorId: 'scinerf-rw-admm-a',
        pageStart: 3,
      }),
      systemB({
        num: 4,
        anchor: 'qa-admm-b4',
        citingDoc: scinerf,
        headingPath: 'SCINeRF / 2. Related Work / Snapshot Compressive Imaging',
        context: 'most of the existing methods employ alternating direction method of multipliers (ADMM) [4]',
        answerClaim: 'ADMM 是 SCINeRF 借用来描述既有优化路线的上游方法。',
        upstreamWorkRole: '这篇参考文献提供 ADMM 的通用优化框架，是 Related Work 中“已有 SCI 方法常用 ADMM”的来源。',
        userQuestionRelation: '用户问“是不是作者发明的”，这条文内参考把 ADMM 指向上游基础方法，因此回答应明确“不是原创贡献”。',
        raw: '[4] Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1):1-122, 2011.',
        title: 'Distributed optimization and statistical learning via the alternating direction method of multipliers',
        authors: 'Stephen Boyd; Neal Parikh; Eric Chu; Borja Peleato; Jonathan Eckstein',
        venue: 'Foundations and Trends in Machine Learning',
        year: '2011',
        doi: '10.1561/2200000016',
        blockId: 'scinerf-rw-admm',
        anchorId: 'scinerf-rw-admm-a',
      }),
    ],
    refs: [
      refHit({
        doc: scinerf,
        headingPath: 'SCINeRF / 2. Related Work / Snapshot Compressive Imaging',
        summaryLine: '这一段把 ADMM 放在“existing methods”里，并把编号 [4] 接到 ADMM 基础文献。',
        whyLine: '用户关心是不是本文原创；命中段落能直接区分“背景方法”和“本文贡献”。',
        snippet: 'most of the existing methods employ alternating direction method of multipliers (ADMM) [4]',
        score: 9.69,
        year: '2024',
        venue: 'CVPR',
        blockId: 'scinerf-rw-admm',
        anchorId: 'scinerf-rw-admm-a',
      }),
    ],
    acceptance: [
      '回答必须先说清楚“不是本文原创”。',
      'System B 卡片要解释上游文献角色，而不是只列参考文献信息。',
      'System A 卡片打开后应定位到 Related Work 中 ADMM 的语境。',
    ],
  }),
  makeCase(2, {
    id: 'hadamard-fourier-choice',
    docIds: ['hsi-fsi'],
    question: '我做单像素实验，Hadamard 和 Fourier 到底该怎么选？',
    answerMarkdown: [
      '这篇比较给出的取舍很实用：如果测量次数很少、你更在意欠采样下尽快出结构，FSI 往往更占优，因为傅里叶基对很多自然图像有更强的能量集中；如果测量次数比较充足、你更在意背景干净和低噪声，HSI 会更稳。[1](#qa-hsi-fsi-a1)',
      '',
      '所以可以把它当成实验设计问题：快速预览或极低采样先试 FSI；追求充分采样后的均匀重建质量，Hadamard 更值得作为基线。[2](#qa-hsi-fsi-a2)',
    ].join('\n'),
    citeDetails: [
      systemA({
        num: 1,
        anchor: 'qa-hsi-fsi-a1',
        doc: hsiFsi,
        headingPath: 'Hadamard versus Fourier SPI / 2.4 Efficiency',
        evidenceQuote: 'efficient single-pixel imaging reconstructs a sharp image with a small number of measurements, and efficiency depends on how well the transformation concentrates image energy.',
        answerClaim: '少量测量时的选择核心是能量集中能力。',
        supportRelation: '效率章节直接把测量次数、清晰重建和变换能量集中联系起来。',
        blockId: 'hsi-fsi-efficiency',
        anchorId: 'hsi-fsi-efficiency-a',
        pageStart: 6,
      }),
      systemA({
        num: 2,
        anchor: 'qa-hsi-fsi-a2',
        doc: hsiFsi,
        headingPath: 'Hadamard versus Fourier SPI / Comparison Results',
        evidenceQuote: 'In the case of a small number of measurements, FSI outperforms HSI... In the case of a large number of measurements, HSI has better reconstruction quality than binary FSI.',
        answerClaim: 'FSI 更适合极少测量，HSI 更适合测量充足时的稳健质量。',
        supportRelation: '比较结果直接给出两种基在不同测量规模下的结论。',
        blockId: 'hsi-fsi-results',
        anchorId: 'hsi-fsi-results-a',
        pageStart: 11,
      }),
    ],
    refs: [
      refHit({
        doc: hsiFsi,
        headingPath: 'Hadamard versus Fourier SPI / Comparison Results',
        summaryLine: '实验比较明确区分了少量测量和大量测量两种场景下 HSI/FSI 的优势。',
        whyLine: '用户问“怎么选”，这条证据能把选择条件转换成测量预算和质量偏好的判断。',
        snippet: 'in the case of a small number of measurements, FSI outperforms HSI... in the case of a large number of measurements, HSI has better reconstruction quality',
        score: 9.62,
        year: '2017',
        venue: 'Optics Express',
        blockId: 'hsi-fsi-results',
        anchorId: 'hsi-fsi-results-a',
      }),
    ],
    acceptance: [
      '回答应给出条件化建议，不要一句话武断说某个一定更好。',
      '参考卡应说明“测量次数/能量集中/噪声质量”的关系。',
    ],
  }),
  makeCase(3, {
    id: 'foveated-dynamic-supersampling',
    docIds: ['foveated-spi', 'visual-perception'],
    question: 'dynamic supersampling 是不是就是只盯着画面重要的地方多拍一点？',
    answerMarkdown: [
      '可以这么理解，但它比“只放大重要区域”更细。它用 foveated 的空间变分采样：中心或被运动触发的区域给高分辨率，外围保持低分辨率监视；同时把多个低分辨率子帧融合，给相对静止的区域逐步补细节。[1](#qa-foveated-a1)',
      '',
      '这套策略借鉴了视觉系统的 fovea/saccade 思想：不是把其他区域完全丢掉，而是在整幅视野持续监控的同时，把有限测量资源动态分给更值得看的地方。[2](#qa-foveated-a2)',
    ].join('\n'),
    citeDetails: [
      systemA({
        num: 1,
        anchor: 'qa-foveated-a1',
        doc: foveated,
        headingPath: 'Adaptive foveated SPI / Introduction',
        evidenceQuote: 'The position of the fovea can be guided by visual stimuli detected in previous images, while multiple low-resolution frames are fused to synthesize higher-resolution detail.',
        answerClaim: 'dynamic supersampling 是“动态选重点 + 多帧补细节”的组合。',
        supportRelation: '引言明确同时描述 fovea 指导和低分辨率帧融合，能支撑这个解释。',
        blockId: 'foveated-intro',
        anchorId: 'foveated-intro-a',
        pageStart: 1,
      }),
      systemA({
        num: 2,
        anchor: 'qa-foveated-a2',
        doc: foveated,
        headingPath: 'Adaptive foveated SPI / Discussion and Conclusions',
        evidenceQuote: 'The framework rapidly records detail of fast-changing features while accumulating enhanced detail of more slowly changing regions over several consecutive frames.',
        answerClaim: '它不是简单 zoom，而是利用动态场景的时空冗余分配测量资源。',
        supportRelation: '结论段落概括了快变区域和慢变区域的不同处理方式。',
        blockId: 'foveated-discussion',
        anchorId: 'foveated-discussion-a',
        pageStart: 8,
      }),
    ],
    refs: [
      refHit({
        doc: foveated,
        headingPath: 'Adaptive foveated SPI / Introduction',
        summaryLine: '引言把 foveated 采样、运动触发和 supersampling 放在同一条机制链里。',
        whyLine: '用户用直觉说“重要地方多拍一点”，这条证据能把直觉校正成可实现的采样策略。',
        snippet: 'The position of the fovea within the field of view can then be guided... fusion of multiple low-resolution frames to synthesize a higher-resolution image',
        score: 9.55,
        year: '2017',
        venue: 'Science Advances',
        blockId: 'foveated-intro',
        anchorId: 'foveated-intro-a',
      }),
      refHit({
        doc: doc('visual-perception'),
        headingPath: 'Some informational aspects of visual perception / visual information framing',
        summaryLine: '这篇较早的视觉感知文献适合作为“为什么视觉系统会选择性分配信息处理资源”的概念背景。',
        whyLine: '它不是 SPI 技术论文，但能帮助解释 foveated 思路背后的信息处理动机。',
        snippet: 'visual perception can be discussed as an information selection and processing problem',
        score: 7.84,
        year: '1954',
        venue: 'Psychological Review',
        blockId: 'visual-info',
        anchorId: 'visual-info-a',
      }),
    ],
    acceptance: [
      '回答应把用户朴素表述翻译成技术机制。',
      '参考定位卡应能区分技术证据和背景性启发文献。',
    ],
  }),
  makeCase(4, {
    id: 'dl-spi-benefits-risks',
    docIds: ['dl-spi-review'],
    question: '深度学习给单像素成像带来的好处和坑分别是什么？',
    answerMarkdown: [
      '好处主要有三类：更快的重建、更好的图像质量、以及把 SPI 从“成像”扩展到分类、检测、分割等 image-free sensing 任务。综述把这看成 DL 给 SPI 带来的范式变化。[1](#qa-dl-spi-a1)',
      '',
      '坑也很明确：数据依赖、可解释性不足、过拟合和泛化有限；此外高保真和实时速度之间仍有天然权衡，抗干扰和信息利用也没有彻底解决。[2](#qa-dl-spi-a2) 读这篇时最好把“数据驱动很快”和“物理模型更稳但可能更慢”一起看。',
    ].join('\n'),
    citeDetails: [
      systemA({
        num: 1,
        anchor: 'qa-dl-spi-a1',
        doc: dlSpiReview,
        headingPath: 'DL-SPI Review / 6. Challenges and Outlooks',
        evidenceQuote: 'DL technology has ushered in a paradigm shift in SPI, leading to advancements in imaging speed, image quality, and information analysis capabilities.',
        answerClaim: '深度学习提升 SPI 的速度、质量和信息分析能力。',
        supportRelation: '综述结论段直接列出了 DL-SPI 的主要收益。',
        blockId: 'dl-spi-outlook-benefits',
        anchorId: 'dl-spi-outlook-benefits-a',
        pageStart: 16,
      }),
      systemA({
        num: 2,
        anchor: 'qa-dl-spi-a2',
        doc: dlSpiReview,
        headingPath: 'DL-SPI Review / 6. Challenges and Outlooks',
        evidenceQuote: 'limitations include reliance on extensive datasets, limited interpretability, susceptibility to overfitting, and limited generalization.',
        answerClaim: 'DL-SPI 的主要风险是数据依赖、解释性和泛化问题。',
        supportRelation: '同一节明确列出风险，并继续讨论速度质量权衡与抗干扰问题。',
        blockId: 'dl-spi-outlook-risks',
        anchorId: 'dl-spi-outlook-risks-a',
        pageStart: 16,
      }),
    ],
    refs: [
      refHit({
        doc: dlSpiReview,
        headingPath: 'DL-SPI Review / 6. Challenges and Outlooks',
        summaryLine: '这一节同时总结了 DL-SPI 的收益、瓶颈和未来解决路径。',
        whyLine: '用户问“好处和坑”，这条命中天然对应优缺点对照，不需要额外拼接模板。',
        snippet: 'advancements in imaging speed, image quality, and information analysis capabilities... reliance on extensive datasets, limited interpretability, overfitting, and limited generalization',
        score: 9.71,
        year: '2025',
        venue: 'Laser & Photonics Reviews',
        blockId: 'dl-spi-outlook-risks',
        anchorId: 'dl-spi-outlook-risks-a',
      }),
    ],
    acceptance: [
      '回答必须同时覆盖收益和风险。',
      '不能把综述回答成宣传口径，要指出数据、解释性、泛化和速度质量权衡。',
    ],
  }),
  makeCase(5, {
    id: 'qclfm-refocus',
    docIds: ['qclfm'],
    question: '这个 quantum correlation light-field microscope 是怎么把离焦样品重新对焦的？',
    answerMarkdown: [
      '它的关键是把一个光子的位置信息和纠缠伙伴的角度/动量信息配对起来，等价于一次性得到 light field。重聚焦分两步：先用位置和角度信息做 ray tracing，把离焦距离对应的光线路径反推回来；再在微观样品上补上波动光学传播的反向校正，因为显微尺度下衍射不能忽略。[1](#qa-qclfm-a1)',
      '',
      '所以它不是靠机械扫描找焦面，而是用相关光子给出的角度信息在后处理里重建不同深度。[2](#qa-qclfm-a2)',
    ].join('\n'),
    citeDetails: [
      systemA({
        num: 1,
        anchor: 'qa-qclfm-a1',
        doc: qclfm,
        headingPath: 'QCLFM / A. Concept',
        evidenceQuote: 'Digital refocusing is achieved using two steps: ray tracing from position and angular information, then reversing diffraction by wave propagation of distance -z.',
        answerClaim: 'QCLFM 通过 ray tracing 加反向波传播实现数字重聚焦。',
        supportRelation: 'Concept 节直接列出重聚焦的两个计算步骤。',
        blockId: 'qclfm-concept-refocus',
        anchorId: 'qclfm-concept-refocus-a',
        pageStart: 3,
      }),
      systemA({
        num: 2,
        anchor: 'qa-qclfm-a2',
        doc: qclfm,
        headingPath: 'QCLFM / Experimental Results',
        evidenceQuote: 'A single-shot measurement was taken on the lens tissue stack with digital refocusing performed in post-processing.',
        answerClaim: '实验展示是在单次测量后通过后处理把不同深度纤维带回焦内。',
        supportRelation: '实验结果说明这种重聚焦不是扫描得到，而是单次采集后的计算处理。',
        blockId: 'qclfm-results-tissue',
        anchorId: 'qclfm-results-tissue-a',
        pageStart: 6,
      }),
    ],
    refs: [
      refHit({
        doc: qclfm,
        headingPath: 'QCLFM / A. Concept',
        summaryLine: 'Concept 节把重聚焦拆成 ray tracing 和 wave propagation 两步，是回答“怎么重新对焦”的核心证据。',
        whyLine: '用户问机制，不是问性能指标；这段命中方法步骤而不是只命中标题。',
        snippet: 'First... ray tracing operation... the second step is to reverse this diffraction by applying a wave propagation of distance -z',
        score: 9.64,
        year: '2024',
        venue: 'arXiv',
        blockId: 'qclfm-concept-refocus',
        anchorId: 'qclfm-concept-refocus-a',
      }),
    ],
    acceptance: [
      '回答应解释两个步骤，而不是只复述“extreme depth of field”。',
      '定位卡应命中 Concept/Experimental Results，不应漂到 references。',
    ],
  }),
  makeCase(6, {
    id: 's2ism-thick-samples',
    docIds: ['s2ism', 'iism'],
    question: 's2ISM 这篇说的 trade-off 是什么？为什么厚样本会麻烦？',
    answerMarkdown: [
      '这篇关心的是 image scanning microscopy 里分辨率、信噪比和 optical sectioning 的互相牵制。厚样本会带来更多离焦背景和散射信号，单纯提高分辨率或收更多光不一定同时保住轴向切片能力和 SNR。[1](#qa-s2ism-a1)',
      '',
      's2ISM 的思路是用 SPAD detector array 做 structured detection，并结合图像形成模型反演，让检测端保留更多空间/轴向信息，而不是只靠传统针孔式取舍。[2](#qa-s2ism-a2)',
    ].join('\n'),
    citeDetails: [
      systemA({
        num: 1,
        anchor: 'qa-s2ism-a1',
        doc: s2ism,
        headingPath: 'Structured detection microscopy / Abstract',
        evidenceQuote: 's2ISM is presented to overcome the trade-off between resolution, signal-to-noise ratio, and optical sectioning in image scanning microscopy.',
        answerClaim: 's2ISM 面向分辨率、SNR 和 optical sectioning 之间的取舍。',
        supportRelation: '摘要直接概括了论文要打破的显微成像 trade-off。',
        blockId: 's2ism-abstract-tradeoff',
        anchorId: 's2ism-abstract-tradeoff-a',
        pageStart: 1,
      }),
      systemA({
        num: 2,
        anchor: 'qa-s2ism-a2',
        doc: s2ism,
        headingPath: 'Structured detection microscopy / Results',
        evidenceQuote: 'The method uses a SPAD detector array and inversion of an image formation model to improve imaging in thick samples.',
        answerClaim: 's2ISM 通过结构化检测和模型反演保留更多有用检测信息。',
        supportRelation: '结果部分说明了 SPAD 阵列和模型反演怎样服务于厚样本显微。',
        blockId: 's2ism-results-spad',
        anchorId: 's2ism-results-spad-a',
        pageStart: 2,
      }),
    ],
    refs: [
      refHit({
        doc: s2ism,
        headingPath: 'Structured detection microscopy / Abstract',
        summaryLine: '摘要直接提出 resolution、SNR、optical sectioning 三者之间的取舍，是问题里的 trade-off 来源。',
        whyLine: '用户问“trade-off 是什么”，这条证据给出三项变量；后续卡片再解释厚样本为什么放大矛盾。',
        snippet: 'overcome the trade-off between resolution, signal-to-noise ratio, and optical sectioning',
        score: 9.44,
        year: '2025',
        venue: 'Nature Photonics',
        blockId: 's2ism-abstract-tradeoff',
        anchorId: 's2ism-abstract-tradeoff-a',
      }),
    ],
    acceptance: [
      '回答应把 trade-off 的三个量说全。',
      '厚样本解释应关联离焦背景/散射/SNR，而不是泛泛说“更难”。',
    ],
  }),
  makeCase(7, {
    id: 'single-photon-pidl',
    docIds: ['pidl-single-photon', 'spd-review'],
    question: 'physics-informed deep learning 在单光子成像里到底帮了什么？',
    answerMarkdown: [
      '它主要帮两件事：第一，把 SPAD 阵列的多源噪声和成像物理写进训练数据/模型约束，避免网络只靠黑箱拟合；第二，用大规模合成数据训练增强网络，把低分辨率、低 bit-depth 的单光子图像恢复到更高分辨率和更平滑背景。[1](#qa-pidl-a1)',
      '',
      '这对单光子成像特别重要，因为弱光下 shot noise、dark count、afterpulsing、crosstalk 等噪声会直接决定图像质量。[2](#qa-pidl-a2)',
    ].join('\n'),
    citeDetails: [
      systemA({
        num: 1,
        anchor: 'qa-pidl-a1',
        doc: pidlSinglePhoton,
        headingPath: 'Physics-informed single-photon imaging / Figure 1',
        evidenceQuote: 'The work models multi-source SPAD noise and uses large-scale synthetic data to train neural networks for single-photon enhancement.',
        answerClaim: 'physics-informed DL 把 SPAD 噪声物理和大规模合成训练结合起来做增强。',
        supportRelation: 'Figure 1 说明了噪声模型、校准数据、合成数据和神经网络增强的整体链路。',
        blockId: 'pidl-fig1-noise-model',
        anchorId: 'pidl-fig1-noise-model-a',
        pageStart: 2,
      }),
      systemA({
        num: 2,
        anchor: 'qa-pidl-a2',
        doc: pidlSinglePhoton,
        headingPath: 'Physics-informed single-photon imaging / Figure 1',
        evidenceQuote: 'The physical noise model consists of shot noise, fixed-pattern noise, dark count rate, afterpulsing, crosstalk, and deadtime noise.',
        answerClaim: '单光子图像质量受多源 SPAD 噪声共同影响。',
        supportRelation: '噪声组成解释了为什么必须引入物理先验，而不是只做普通超分辨率。',
        blockId: 'pidl-noise-components',
        anchorId: 'pidl-noise-components-a',
        pageStart: 2,
      }),
    ],
    refs: [
      refHit({
        doc: pidlSinglePhoton,
        headingPath: 'Physics-informed single-photon imaging / Figure 1',
        summaryLine: 'Figure 1 把 SPAD 噪声模型、真实校准数据、合成训练集和增强网络串成一条方法链。',
        whyLine: '用户问“帮了什么”，这条证据能说明物理信息具体进入了噪声建模和训练数据生成。',
        snippet: 'multi-source physical noise model of SPAD arrays... large-scale synthetic data was applied to train neural networks for single-photon enhancement',
        score: 9.53,
        year: '2023',
        venue: 'Nature Communications',
        blockId: 'pidl-fig1-noise-model',
        anchorId: 'pidl-fig1-noise-model-a',
      }),
      refHit({
        doc: doc('spd-review'),
        headingPath: 'SPD review / Abstract',
        summaryLine: '单光子探测综述可补充 detector 端的背景：高灵敏度伴随成本、低温、噪声等工程限制。',
        whyLine: '它帮助用户把算法论文里的 SPAD 噪声问题放回探测器技术背景中理解。',
        snippet: 'Single-photon detections are highly sensitive but face complexity, manufacturing cost, and special operating conditions.',
        score: 8.18,
        year: '2024',
        venue: 'Frontiers of Physics',
        blockId: 'spd-review-abstract',
        anchorId: 'spd-review-abstract-a',
      }),
    ],
    acceptance: [
      '回答应说清物理先验具体进入哪里。',
      '参考卡应能区分算法主文献和探测器背景综述。',
    ],
  }),
  makeCase(8, {
    id: 'perovskite-boundary',
    docIds: ['perovskite-laser'],
    question: '这篇 perovskite laser 和我的单像素成像主线关系大吗？值得一起读吗？',
    answerMarkdown: [
      '它和单像素成像主线不是同一条技术链：这篇重点是 dual-cavity perovskite device 的 electrically driven lasing，更偏光电器件和发光/激光机制。[1](#qa-perovskite-a1)',
      '',
      '但它不是完全没用。如果你的综述要覆盖“新型光源、探测器、材料器件怎样支撑计算成像系统”，它可以作为外围材料；如果你的当前目标是 SPI/SCI 重建算法、编码策略或参考定位，它优先级应低于 SCIGS、Hadamard/Fourier、DL-SPI review 这些主线文献。',
    ].join('\n'),
    citeDetails: [
      systemA({
        num: 1,
        anchor: 'qa-perovskite-a1',
        doc: perovskiteLaser,
        headingPath: 'Perovskite laser / Abstract',
        evidenceQuote: 'The paper reports electrically driven lasing from a dual-cavity perovskite device.',
        answerClaim: '这篇主线是 perovskite 发光/激光器件，不是 SPI/SCI 重建。',
        supportRelation: '摘要主题足以判断它与单像素成像库的关系更偏外围器件。',
        blockId: 'perovskite-abstract',
        anchorId: 'perovskite-abstract-a',
        pageStart: 1,
      }),
    ],
    refs: [
      refHit({
        doc: perovskiteLaser,
        headingPath: 'Perovskite laser / Abstract',
        summaryLine: '摘要主题是 electrically driven lasing 和 dual-cavity perovskite device，说明它属于光电器件论文。',
        whyLine: '用户问“和单像素成像主线关系大不大”，这条证据用于判断主题边界，而不是强行拉入 SPI 主线。',
        snippet: 'electrically driven lasing from a dual-cavity perovskite device',
        score: 8.74,
        year: '2025',
        venue: 'Nature',
        blockId: 'perovskite-abstract',
        anchorId: 'perovskite-abstract-a',
      }),
    ],
    acceptance: [
      '回答应能识别边界文献，不应为了命中而夸大相关性。',
      '参考定位卡应说明“为什么这是外围而非主线”。',
    ],
  }),
]

export const RESEARCH_QA_MESSAGES: Message[] = RESEARCH_QA_CASES.flatMap((item, index) => [
  {
    id: item.userMessageId,
    role: 'user',
    content: item.question,
    created_at: BASE_TIME + index * 90_000,
  },
  {
    id: item.assistantMessageId,
    role: 'assistant',
    content: item.answerMarkdown,
    rendered_body: item.answerMarkdown,
    copy_text: item.answerMarkdown,
    copy_markdown: item.answerMarkdown,
    cite_details: item.citeDetails,
    refs_user_msg_id: item.userMessageId,
    created_at: BASE_TIME + index * 90_000 + 30_000,
  },
])

export const RESEARCH_QA_REFS: Record<string, unknown> = Object.fromEntries(
  RESEARCH_QA_CASES.map((item) => [
    String(item.userMessageId),
    {
      prompt: item.question,
      display_state: 'ready',
      hits: item.refs,
    },
  ]),
)
