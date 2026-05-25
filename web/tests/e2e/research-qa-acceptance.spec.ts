import { expect, test, type Locator, type Page, type Route } from '@playwright/test'

const CONV_ID = 'conv-research-qa-acceptance'
const BASE_TIME = 1_780_000_000

const LPR_MD = 'db/LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning/LPR-2025.en.md'
const OPTICS_MD = 'db/Optics-2024-Part-based image-loop network for single-pixel imaging/Optics-2024.en.md'
const NATPHOTON_MD = 'db/NatPhoton-2019-Principles and prospects for single-pixel imaging/NatPhoton-2019.en.md'
const SCINERF_MD = 'db/CVPR-2024-SCINeRF/CVPR-2024-SCINeRF.en.md'

const conversation = {
  id: CONV_ID,
  title: '科研验收：SPI 与 SCINeRF',
  created_at: BASE_TIME,
  updated_at: BASE_TIME + 60,
  project_id: null,
  mode: 'normal',
  bound_source_path: '',
  bound_source_name: '',
  bound_source_ready: false,
}

const messages = [
  {
    id: 201,
    role: 'user',
    content: '深度学习给单像素成像带来的好处和坑分别是什么？',
    created_at: BASE_TIME + 1,
  },
  {
    id: 202,
    role: 'assistant',
    refs_user_msg_id: 201,
    content: [
      '## 好处',
      '',
      '**成像质量提升**：深度模型能学习复杂的非线性重建映射，在低采样测量下改善单像素成像的重建质量 [1](#spi-a1)。',
      '',
      '**降低采样率**：ILNet 这类方法说明，网络结构可以把更少的测量转成可用图像，但代价是泛化要谨慎检查 [2](#spi-a2)。',
      '',
      '## 坑',
      '',
      '**物理约束仍然存在**：深度学习不能消除探测器动态范围、噪声和量化电子学之间的基本折衷 [3](#spi-a3)。',
    ].join('\n'),
    rendered_body: [
      '## 好处',
      '',
      '**成像质量提升**：深度模型能学习复杂的非线性重建映射，在低采样测量下改善单像素成像的重建质量 [1](#spi-a1)。',
      '',
      '**降低采样率**：ILNet 这类方法说明，网络结构可以把更少的测量转成可用图像，但代价是泛化要谨慎检查 [2](#spi-a2)。',
      '',
      '## 坑',
      '',
      '**物理约束仍然存在**：深度学习不能消除探测器动态范围、噪声和量化电子学之间的基本折衷 [3](#spi-a3)。',
    ].join('\n'),
    cite_details: [
      {
        num: 1,
        anchor: 'spi-a1',
        source_name: 'LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf',
        source_path: LPR_MD,
        title: '4. Strategy and Advantages of Single-Pixel Imaging via Deep Learning',
        heading_path: '4. Strategy and Advantages / Data-driven strategy',
        is_inpaper: false,
        answer_claim: '深度模型能在低采样测量下改善单像素成像的重建质量。',
        evidence_quote: 'The encoder samples the image into low-dimensional measurements, while the decoder maps these measurements onto the target image.',
        location_label: '4. Strategy and Advantages / Data-driven strategy · p. 7 · paragraph',
        support_relation: '这条命中直接解释了端到端编码器/解码器怎样把少量测量映射回目标图像。',
        block_id: 'lpr-strategy-41',
        anchor_id: 'p-lpr-41',
        anchor_kind: 'paragraph',
        page_start: 7,
      },
      {
        num: 2,
        anchor: 'spi-a2',
        source_name: 'Optics & Laser Technology-2024-Part-based image-loop network for single-pixel imaging.pdf',
        source_path: OPTICS_MD,
        title: 'ILNet reconstruction under low sampling ratio',
        heading_path: '3. Experiments / Low sampling ratio',
        is_inpaper: false,
        answer_claim: 'ILNet 说明网络结构可以把更少测量转成可用图像，但泛化需要检查。',
        evidence_quote: 'ILNet reconstructs target images from detector signals under a lower sampling ratio and fewer iterations.',
        location_label: '3. Experiments / Low sampling ratio · p. 5 · paragraph',
        support_relation: '这条证据支撑“采样率下降”和“低迭代重建”的具体说法。',
        block_id: 'ilnet-low-sampling',
        anchor_id: 'p-ilnet-low-sampling',
        anchor_kind: 'paragraph',
        page_start: 5,
      },
      {
        num: 3,
        anchor: 'spi-a3',
        source_name: 'NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf',
        source_path: NATPHOTON_MD,
        title: 'Principles and detector trade-offs',
        heading_path: 'Principles and prospects / Outlook',
        is_inpaper: false,
        answer_claim: '深度学习不能消除探测器动态范围、噪声和量化电子学之间的基本折衷。',
        evidence_quote: 'Single-pixel imaging involves a balance between detector dynamic range and quantization electronics.',
        location_label: 'Principles and prospects / Outlook · p. 9 · paragraph',
        support_relation: '这条证据提醒用户算法收益仍受成像硬件和物理噪声限制。',
        block_id: 'natphoton-outlook',
        anchor_id: 'p-natphoton-outlook',
        anchor_kind: 'paragraph',
        page_start: 9,
      },
    ],
    copy_markdown: '',
    copy_text: '',
    created_at: BASE_TIME + 2,
  },
  {
    id: 203,
    role: 'user',
    content: 'ADMM 是怎么来的？作者这里是不是借鉴了别人以前的想法？',
    created_at: BASE_TIME + 3,
  },
  {
    id: 204,
    role: 'assistant',
    refs_user_msg_id: 203,
    content: '不是作者在 SCINeRF 里新发明的。论文把 ADMM 放在已有压缩成像重建方法脉络里，意思是借用一个成熟优化框架作为相关工作背景 [4](#scinerf-r4)。ADMM-Net 则是把这种迭代优化思想展开成网络结构的代表性前作 [21](#scinerf-r21)。',
    rendered_body: '不是作者在 SCINeRF 里新发明的。论文把 ADMM 放在已有压缩成像重建方法脉络里，意思是借用一个成熟优化框架作为相关工作背景 [4](#scinerf-r4)。ADMM-Net 则是把这种迭代优化思想展开成网络结构的代表性前作 [21](#scinerf-r21)。',
    cite_details: [
      {
        num: 4,
        anchor: 'scinerf-r4',
        source_name: 'CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf',
        source_path: SCINERF_MD,
        is_inpaper: true,
        title: 'Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers',
        authors: 'Boyd, Parikh, Chu, Peleato, Eckstein',
        venue: 'Foundations and Trends in Machine Learning',
        year: '2011',
        doi: '10.1561/2200000016',
        doi_url: 'https://doi.org/10.1561/2200000016',
        raw: '[4] Boyd S., Parikh N., Chu E., Peleato B., Eckstein J. Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Foundations and Trends in Machine Learning, 2011.',
        answer_claim: 'ADMM 不是 SCINeRF 作者新发明的，而是相关工作里借用的成熟优化框架。',
        heading_path: 'SCINeRF / 2. Related Work / Snapshot Compressive Imaging',
        location_label: 'SCINeRF / 2. Related Work / Snapshot Compressive Imaging',
        card_locator_label: '引用出现位置',
        card_locator: 'SCINeRF / 2. Related Work / Snapshot Compressive Imaging',
        citation_context: 'Most existing methods employ ADMM-based optimization for snapshot compressive imaging reconstruction.',
        upstream_work_role: 'ADMM 提供的是通用优化框架，帮助理解早期 SCI 重建方法的来源。',
        user_question_relation: '用户问“是不是借鉴了别人以前的想法”，这条参考正好说明它是上游方法背景，而不是原创贡献。',
      },
      {
        num: 21,
        anchor: 'scinerf-r21',
        source_name: 'CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf',
        source_path: SCINERF_MD,
        is_inpaper: true,
        title: 'ADMM-Net: A Deep Learning Approach for Compressive Sensing MRI',
        authors: 'Yang, Sun, Li, Xu',
        venue: 'NeurIPS',
        year: '2016',
        raw: '[21] Yang Y., Sun J., Li H., Xu Z. ADMM-Net: A Deep Learning Approach for Compressive Sensing MRI. NeurIPS, 2016.',
        answer_claim: 'ADMM-Net 是把迭代优化思想展开成网络结构的代表性前作。',
        heading_path: 'SCINeRF / 2. Related Work / Snapshot Compressive Imaging',
        location_label: 'SCINeRF / 2. Related Work / Snapshot Compressive Imaging',
        card_locator_label: '引用出现位置',
        card_locator: 'SCINeRF / 2. Related Work / Snapshot Compressive Imaging',
        citation_context: 'ADMM-Net unfolds ADMM iterations into a trainable deep network.',
        upstream_work_role: '它说明“优化算法展开成网络”的思想在 SCINeRF 之前已经存在。',
        user_question_relation: '这能帮助用户沿着 Related Work 追溯方法线索。',
      },
    ],
    copy_markdown: '',
    copy_text: '',
    created_at: BASE_TIME + 4,
  },
]

const refsPayload = {
  '201': {
    prompt: '深度学习给单像素成像带来的好处和坑分别是什么？',
    display_state: 'ready',
    payload_mode: 'stored_full',
    hits: [
      {
        text: 'The encoder samples the image into low-dimensional measurements while the decoder maps them onto the target image.',
        meta: { source_path: LPR_MD, ref_pack_state: 'ready' },
        ui_meta: {
          display_name: 'LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf',
          source_path: LPR_MD,
          heading_path: '4. Strategy and Advantages / Data-driven strategy',
          summary_line: '这一节说明深度学习怎样把低维单像素测量映射回目标图像，是“成像质量提升”的主要证据。',
          why_line: '用户问好处和坑，这条卡片负责解释“收益来自哪里”：端到端模型利用训练数据学习非线性重建映射。',
          summary_generation: 'llm_grounded',
          why_generation: 'llm_grounded',
          score: 9.36,
          reader_open: {
            sourcePath: LPR_MD,
            sourceName: 'LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf',
            headingPath: '4. Strategy and Advantages / Data-driven strategy',
            snippet: 'The encoder samples the image into low-dimensional measurements.',
            blockId: 'lpr-strategy-41',
            anchorId: 'p-lpr-41',
            anchorKind: 'paragraph',
            strictLocate: true,
          },
        },
      },
      {
        text: 'ILNet reconstructs target images under a lower sampling ratio.',
        meta: { source_path: OPTICS_MD, ref_pack_state: 'ready' },
        ui_meta: {
          display_name: 'Optics & Laser Technology-2024-Part-based image-loop network for single-pixel imaging.pdf',
          source_path: OPTICS_MD,
          heading_path: '3. Experiments / Low sampling ratio',
          summary_line: '这条证据给出低采样率重建例子，可用来说明深度网络降低采样需求。',
          why_line: '它和问题里的“好处”直接相关，但同时提醒泛化能力需要用不同场景验证。',
          summary_generation: 'llm_grounded',
          why_generation: 'llm_grounded',
          score: 8.92,
        },
      },
      {
        text: 'Single-pixel imaging involves a balance between detector dynamic range and quantization electronics.',
        meta: { source_path: NATPHOTON_MD, ref_pack_state: 'ready' },
        ui_meta: {
          display_name: 'NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf',
          source_path: NATPHOTON_MD,
          heading_path: 'Principles and prospects / Outlook',
          summary_line: '这篇综述指出单像素成像仍受探测器动态范围和量化电子学制约。',
          why_line: '它用于回答“坑”：深度模型能改善重建，但不能把硬件和物理约束直接抹掉。',
          summary_generation: 'llm_grounded',
          why_generation: 'llm_grounded',
          score: 8.67,
        },
      },
    ],
  },
  '203': {
    prompt: 'ADMM 是怎么来的？作者这里是不是借鉴了别人以前的想法？',
    display_state: 'ready',
    payload_mode: 'stored_full',
    hits: [
      {
        text: 'SCINeRF related work cites ADMM and ADMM-Net as earlier methods.',
        meta: { source_path: SCINERF_MD, ref_pack_state: 'ready' },
        ui_meta: {
          display_name: 'CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf',
          source_path: SCINERF_MD,
          heading_path: '2. Related Work / Snapshot compressive imaging',
          summary_line: 'Related Work 把 ADMM 和 ADMM-Net 放在已有压缩成像重建脉络下，而不是作为 SCINeRF 的原创点。',
          why_line: '用户问“是不是借鉴以前的想法”，这张卡片明确把答案定位到论文的相关工作段落和上游参考。',
          summary_generation: 'llm_grounded',
          why_generation: 'llm_grounded',
          score: 9.74,
        },
      },
    ],
  },
}

async function fulfillJson(route: Route, body: unknown, headers?: Record<string, string>) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    headers,
    body: JSON.stringify(body),
  })
}

async function installResearchQaBackend(page: Page) {
  await page.route('**/api/settings', async (route) => {
    if (route.request().method() === 'PATCH') {
      await fulfillJson(route, { ok: true })
      return
    }
    await fulfillJson(route, {
      model: 'test-model',
      base_url: '',
      has_api_key: true,
      db_dir: '',
      prefs: {
        ui_locale: 'zh',
        theme: 'light',
        top_k: 6,
        temperature: 0.2,
        max_tokens: 1216,
        deep_read: false,
      },
    })
  })

  await page.route('**/api/projects', async (route) => {
    await fulfillJson(route, [])
  })

  await page.route(/\/api\/conversations(?:\?.*)?$/, async (route) => {
    if (route.request().method() === 'POST') {
      await fulfillJson(route, { id: CONV_ID })
      return
    }
    await fulfillJson(route, [conversation])
  })

  await page.route(`**/api/conversations/${CONV_ID}`, async (route) => {
    await fulfillJson(route, conversation)
  })

  await page.route(new RegExp(`/api/conversations/${CONV_ID}/messages(?:\\?.*)?$`), async (route) => {
    await fulfillJson(route, messages)
  })

  await page.route(`**/api/conversations/${CONV_ID}/messages_page*`, async (route) => {
    await fulfillJson(route, {
      messages,
      has_more_before: false,
      oldest_loaded_id: 201,
      newest_loaded_id: 204,
    })
  })

  await page.route(`**/api/references/conversation/${CONV_ID}`, async (route) => {
    await fulfillJson(route, refsPayload, {
      'server-timing': 'total;dur=12, stored_full;dur=4',
      'x-kb-refs-mode': 'stored_full',
      'x-kb-refs-counts': 'packs=2,hits=4,pending=0',
    })
  })

  await page.route('**/api/references/citation-meta', async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/references/citation-card-polish', async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/references/bibliometrics', async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/references/reader/doc', async (route) => {
    await fulfillJson(route, {
      source_path: LPR_MD,
      source_name: 'Fixture reader document',
      blocks: [],
      anchors: [],
    })
  })
}

async function expectCitationChipsAreClickableLinks(scope: Locator, selector: string) {
  const chips = scope.locator(selector)
  const count = await chips.count()
  expect(count).toBeGreaterThan(0)
  const hrefs = await chips.evaluateAll((items) =>
    items.map((item) => ({
      tag: item.tagName.toLowerCase(),
      href: item.getAttribute('href') || '',
      text: item.textContent || '',
    })),
  )
  for (const item of hrefs) {
    expect(item.tag, `${item.text} should render as an anchor`).toBe('a')
    expect(item.href, `${item.text} should point at a citation anchor`).toMatch(/^#.+/)
  }
  expect(new Set(hrefs.map((item) => item.href)).size).toBe(hrefs.length)
}

async function expectCitationPopoverClean(page: Page) {
  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await expect(popover).not.toContainText('[[CITE:')
  await expect(popover).not.toContainText('```')
  await expect(popover).not.toContainText('## ')
  await expect(popover).not.toContainText('No summary available')
  await expect(popover).not.toContainText('The paper cites')
  await expect(popover).not.toContainText('This is stated in')
  await expect(popover).not.toContainText('This hit is directly relevant')
  await expect(popover).not.toContainText('has attrac')
}

test('research QA acceptance: polished refs and both citation systems stay clickable', async ({ page }) => {
  await installResearchQaBackend(page)

  await page.goto('/')
  const conversationRow = page.locator('.kb-conv-row', { hasText: '科研验收：SPI 与 SCINeRF' })
  await expect(conversationRow).toHaveCount(1)
  await conversationRow.click()

  await expect(page.locator('body')).toContainText('深度学习给单像素成像带来的好处和坑分别是什么？')
  await expect(page.locator('body')).toContainText('ADMM 是怎么来的？作者这里是不是借鉴了别人以前的想法？')
  await expect(page.locator('body')).not.toContainText('[[CITE:')
  await expect(page.locator('body')).not.toContainText('The paper cites')
  await expect(page.locator('body')).not.toContainText('This hit is directly relevant')
  await expect(page.locator('body')).not.toContainText('适合作为定位入口')

  const refsPanels = page.locator('.kb-refs-panel')
  await expect(refsPanels).toHaveCount(2)

  const firstRefs = refsPanels.nth(0)
  await firstRefs.locator('.ant-collapse-header').click()
  await expect(firstRefs.locator('.kb-ref-title').first()).toContainText('LPR-2025')
  await expect(firstRefs.locator('.kb-ref-card-text').filter({ hasText: '成像质量提升' })).toBeVisible()
  await expect(firstRefs.locator('.kb-ref-card-text').filter({ hasText: '收益来自哪里' })).toBeVisible()
  await expect(firstRefs).not.toContainText('The paper cites')
  await expect(firstRefs).not.toContainText('This hit is directly relevant')

  const firstAssistant = page.locator('div[data-msg-id="202"]')
  await expect(firstAssistant.locator('.kb-cite-chip')).toHaveCount(3)
  await expectCitationChipsAreClickableLinks(firstAssistant, '.kb-cite-chip')
  await expect(firstAssistant).not.toContainText('[1]')
  await expect(firstAssistant).not.toContainText('[2]')
  await firstAssistant.locator('.kb-cite-chip').first().hover()
  await firstAssistant.locator('.kb-cite-chip').first().click()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-a/)
  await expectCitationPopoverClean(page)
  await expect(page.getByTestId('citation-popover-system-a-location')).toContainText('Data-driven strategy')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('low-dimensional measurements')
  await expect(page.getByTestId('citation-popover-system-a-claim')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-a-support')).toHaveCount(0)

  await page.locator('.kb-cite-pop-close').click()
  const secondRefs = refsPanels.nth(1)
  await secondRefs.locator('.ant-collapse-header').click()
  await expect(secondRefs.locator('.kb-ref-title')).toContainText('SCINeRF')
  await expect(secondRefs.locator('.kb-ref-card-text').filter({ hasText: '不是作为 SCINeRF 的原创点' })).toBeVisible()
  await expect(secondRefs.locator('.kb-ref-card-text').filter({ hasText: '上游参考' })).toBeVisible()

  const secondAssistant = page.locator('div[data-msg-id="204"]')
  const systemBChips = secondAssistant.locator('.kb-cite-chip-sysb')
  await expect(systemBChips).toHaveCount(2)
  await expectCitationChipsAreClickableLinks(secondAssistant, '.kb-cite-chip-sysb')
  await expect(systemBChips.first()).toHaveText('[R4]')
  await systemBChips.first().hover()
  await systemBChips.first().click()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-b/)
  await expectCitationPopoverClean(page)
  await expect(page.getByTestId('citation-popover-system-b-claim')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-role')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-relation')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-takeaway')).toContainText('ADMM')
  await expect(page.getByTestId('citation-popover-system-b-takeaway')).toContainText('上游')
  await expect(page.getByTestId('citation-popover-system-b-context')).toContainText('ADMM-based optimization')
  await expect(page.getByTestId('citation-popover-system-b-location')).toContainText('引用出现位置')
  await expect(page.getByTestId('citation-popover-system-b-location')).toContainText('SCINeRF')
  await expect(page.getByTestId('citation-popover-system-b-location')).not.toContainText('尚未定位到具体章节或页码')
  await expect(page.getByTestId('citation-popover-system-b-reference')).toHaveCount(0)
  await expect(page.locator('.kb-cite-pop')).toContainText('10.1561/2200000016')
})
