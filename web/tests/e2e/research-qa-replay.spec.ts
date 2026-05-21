import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/references/citation-meta', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
  await page.route('**/api/references/bibliometrics', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
})

test('research QA replay covers multiple real library documents and citation card modes', async ({ page }) => {
  await page.goto('/__research_qa_replay__')

  await expect(page.getByTestId('research-qa-doc-count')).toContainText('文献 21')
  await expect(page.getByTestId('research-qa-case-count')).toContainText('问题 14')

  await expect(page.getByTestId('research-qa-doc-scigs')).toContainText('SCIGS')
  await expect(page.getByTestId('research-qa-doc-hsi-fsi')).toContainText('HSI vs FSI')
  await expect(page.getByTestId('research-qa-doc-foveated-spi')).toContainText('Foveated SPI')
  await expect(page.getByTestId('research-qa-doc-qclfm')).toContainText('QCLFM')
  await expect(page.getByTestId('research-qa-doc-pidl-single-photon')).toContainText('PI single-photon')
  await expect(page.getByTestId('research-qa-doc-perovskite-laser')).toContainText('Perovskite laser')
  await expect(page.getByTestId('research-qa-doc-spi-prospects')).toContainText('SPI prospects')
  await expect(page.getByTestId('research-qa-doc-cassi')).toContainText('CASSI')
  await expect(page.getByTestId('research-qa-doc-piln')).toContainText('PILN')
  await expect(page.getByTestId('research-qa-doc-spd-review')).toContainText('SPD review')

  await expect(page.getByTestId('research-qa-case-scigs-dynamic-3d')).toContainText('SCIGS 这篇到底想解决什么问题')
  await expect(page.getByTestId('research-qa-case-hadamard-fourier-choice')).toContainText('Hadamard 和 Fourier')
  await expect(page.getByTestId('research-qa-case-single-photon-pidl')).toContainText('physics-informed deep learning')
  await expect(page.getByTestId('research-qa-case-spi-roadmap-beginner')).toContainText('刚开始看单像素成像')
  await expect(page.getByTestId('research-qa-case-cassi-to-3d-sci-lineage')).toContainText('压缩快照成像')
  await expect(page.getByTestId('research-qa-case-piln-dl-spi-position')).toContainText('PILN')

  await expect(page.locator('body')).toContainText('SCIGS 的目标不是再做一个普通 SCI 帧解码器')
  await expect(page.locator('body')).toContainText('不是。SCINeRF 把 ADMM 放在 Related Work')
  await expect(page.locator('body')).toContainText('可以这么理解，但它比“只放大重要区域”更细')
  await expect(page.locator('body')).toContainText('它和单像素成像主线不是同一条技术链')

  await expect(page.locator('body')).not.toContainText('The paper cites')
  await expect(page.locator('body')).not.toContainText('This hit is directly relevant')
  await expect(page.locator('body')).not.toContainText('适合作为定位入口')

  const firstRefsHeader = page.locator('.kb-refs-panel .ant-collapse-header').first()
  await expect(firstRefsHeader).toBeVisible()
  await firstRefsHeader.click()
  await expect(page.locator('.kb-ref-title').first()).toContainText('SCIGS')
  await expect(page.getByText('这条命中同时给出痛点、方法选择和动态场景目标')).toBeVisible()

  const systemAChip = page.locator('.kb-cite-chip:not(.kb-cite-chip-sysb)').first()
  await expect(systemAChip).toBeVisible()
  await systemAChip.click()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-a/)
  await expect(page.getByTestId('citation-popover-system-a-claim')).toContainText('SCIGS 面向')
  await expect(page.getByTestId('citation-popover-system-a-source')).toContainText('SCIGS')
  await expect(page.getByTestId('citation-popover-system-a-location')).toContainText('SCIGS / Abstract')
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('variant of 3DGS')
  await page.locator('.kb-cite-pop-action-primary').click()
  await expect(page.getByTestId('research-qa-open-payload')).toContainText('"blockId": "scigs-abstract"')
  await expect(page.getByTestId('research-qa-open-payload')).toContainText('"strictLocate": true')

  const systemBChip = page.locator('.kb-cite-chip-sysb').first()
  await expect(systemBChip).toBeVisible()
  await systemBChip.scrollIntoViewIfNeeded()
  await systemBChip.click()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-b/)
  await expect(page.getByTestId('citation-popover-explain')).toContainText('上游工作')
  await expect(page.getByTestId('citation-popover-flow')).toContainText('当前论文引用处')
  await expect(page.getByTestId('citation-popover-system-b-claim')).toContainText('ADMM 是 SCINeRF 借用')
  await expect(page.getByTestId('citation-popover-system-b-citing-source')).toContainText('SCINeRF')
  await expect(page.getByTestId('citation-popover-system-b-context')).toContainText('existing methods employ')
  await expect(page.getByTestId('citation-popover-system-b-role')).toContainText('ADMM 的通用优化框架')
  await expect(page.getByTestId('citation-popover-system-b-relation')).toContainText('不是原创贡献')
  await expect(page.getByTestId('citation-popover-system-b-reference')).toContainText('Distributed optimization')
})
