import { readFile } from 'node:fs/promises'
import { expect, test } from '@playwright/test'
import { expectCitationShelfQuality } from './cite-shelf-quality'

async function expectNoHorizontalOverflow(page: import('@playwright/test').Page) {
  const metrics = await page.evaluate(() => ({
    bodyScrollWidth: document.body.scrollWidth,
    bodyClientWidth: document.body.clientWidth,
    docScrollWidth: document.documentElement.scrollWidth,
    docClientWidth: document.documentElement.clientWidth,
  }))
  expect(metrics.bodyScrollWidth, 'body should not create horizontal scroll').toBeLessThanOrEqual(metrics.bodyClientWidth + 2)
  expect(metrics.docScrollWidth, 'document should not create horizontal scroll').toBeLessThanOrEqual(metrics.docClientWidth + 2)
}

async function expectElementInsideViewport(page: import('@playwright/test').Page, selector: string) {
  const box = await page.locator(selector).first().boundingBox()
  expect(box, `${selector} should have a layout box`).not.toBeNull()
  const viewport = page.viewportSize()
  expect(viewport, 'viewport is required for layout checks').not.toBeNull()
  if (!box || !viewport) return
  expect(box.x, `${selector} should not overflow left`).toBeGreaterThanOrEqual(0)
  expect(box.y, `${selector} should not overflow top`).toBeGreaterThanOrEqual(0)
  expect(box.x + box.width, `${selector} should not overflow right`).toBeLessThanOrEqual(viewport.width + 1)
  expect(box.y + box.height, `${selector} should not overflow bottom`).toBeLessThanOrEqual(viewport.height + 1)
}

async function expectNoLocalInlineOverflow(page: import('@playwright/test').Page, selector: string) {
  const offenders = await page.locator(selector).evaluateAll((nodes) =>
    nodes
      .map((node, index) => {
        const el = node as HTMLElement
        return {
          index,
          className: el.className,
          text: (el.textContent || '').slice(0, 80),
          scrollWidth: el.scrollWidth,
          clientWidth: el.clientWidth,
        }
      })
      .filter((item) => item.scrollWidth > item.clientWidth + 2),
  )
  expect(offenders, `${selector} should not have inline overflow`).toEqual([])
}

async function addCitationToShelf(page: import('@playwright/test').Page, chip: import('@playwright/test').Locator) {
  await chip.scrollIntoViewIfNeeded()
  await chip.click()
  const popover = page.locator('.kb-cite-pop')
  await expect(popover).toBeVisible()
  await popover.locator('.kb-cite-pop-add').click()
  await expect(page.getByTestId('citation-shelf')).toHaveClass(/translate-x-0/)
  const closePopover = page.locator('.kb-cite-pop-close')
  if (await closePopover.count()) {
    await closePopover.click()
  }
}

test.beforeEach(async ({ page }) => {
  let sourceRepairRequested = false
  let sourceRepairCompleted = false
  await page.route('**/api/references/citation-meta', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    })
  })
  await page.route('**/api/references/citation-card-polish', async (route) => {
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
      body: JSON.stringify({ bibliometrics_checked: true }),
    })
  })
  await page.route('**/api/library/quality/sources', async (route) => {
    const payload = route.request().postDataJSON() as { sources?: Array<{ source_path?: string; source_name?: string }> }
    const sources = Array.isArray(payload.sources) ? payload.sources : []
    const needsReviewSource = (item: { source_path?: string }) =>
      String(item.source_path || '').includes('SCINeRF') && !sourceRepairCompleted
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        review_count: sources.filter(needsReviewSource).length,
        items: sources.map((item) => {
          const sourcePath = String(item.source_path || '')
          const needsReview = needsReviewSource(item)
          return {
            source_path: sourcePath,
            source_name: String(item.source_name || ''),
            pdf_path: '',
            md_path: sourcePath,
            md_exists: true,
            conversion_quality: {
              status: needsReview ? 'warning' : 'good',
              label: needsReview ? 'Needs review' : 'Ready',
              score: needsReview ? 74 : 96,
              summary: needsReview ? 'Needs review | Q74 | 4 pages | 12 refs | 2 figures | 3 math' : 'Ready | Q96 | 9 pages | 35 refs | 4 figures | 8 math',
              has_review_issue: needsReview,
              issues: needsReview ? [
                { code: 'missing_page_markers', label: 'Missing page anchors', severity: 'warning', count: 1 },
              ] : [],
              metrics: {
                page_markers: needsReview ? 4 : 9,
                references: needsReview ? 12 : 35,
                reference_lines: needsReview ? 12 : 35,
                figures: needsReview ? 2 : 4,
                display_math: needsReview ? 2 : 5,
                inline_math: needsReview ? 1 : 3,
              },
            },
          }
        }),
      }),
    })
  })
  await page.route('**/api/library/quality/repair', async (route) => {
    const payload = route.request().postDataJSON() as { sources?: Array<{ source_path?: string; source_name?: string }> }
    const sources = Array.isArray(payload.sources) ? payload.sources : []
    sourceRepairRequested = true
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        requested: sources.length,
        enqueued: sources.length,
        skipped_busy: 0,
        failed: 0,
        items: sources.map((item, idx) => ({
          source_path: String(item.source_path || ''),
          source_name: String(item.source_name || ''),
          pdf_name: `source-${idx}.pdf`,
          pdf_path: `F:\\kb\\pdfs\\source-${idx}.pdf`,
          ok: true,
          enqueued: true,
          skipped_busy: false,
          error: '',
          task_id: `repair-${idx}`,
        })),
      }),
    })
  })
  await page.route('**/api/library/convert/status', async (route) => {
    if (sourceRepairRequested) sourceRepairCompleted = true
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'data: {"running":false,"done":true,"total":1,"completed":1,"current":"","active_count":0,"active_tasks":[],"cur_page_done":0,"cur_page_total":0,"cur_page_msg":"","last":""}\n\n',
    })
  })
})

test('research QA replay opens a single diagnostic case from quality center', async ({ page }) => {
  await page.addInitScript(() => {
    window.sessionStorage.setItem('kb.researchQaReplay.failureCase.v1', JSON.stringify({
      id: 'scinerf-admm-origin',
      question: 'ADMM 是作者自己发明的吗？',
      failures: [
        { name: 'citation_card_quality', domain: 'citation_cards', detail: 'missing title' },
      ],
      doc_ids: ['scinerf', 'scigs'],
      expected_doc_ids: ['scinerf', 'scigs'],
      ref_doc_ids: ['scinerf'],
      citation_doc_ids: [],
      missing_expected_doc_ids: ['scigs'],
      citation_count: 1,
      system_b_count: 0,
      ref_hit_count: 2,
      diagnostic_summary: {
        citation_routes: { system_a: 1, system_b: 0 },
        missing_expected_doc_count: 1,
        citation_diagnostic_count: 1,
        ref_diagnostic_count: 1,
        citation_card_failure_count: 1,
        shelf_failure_count: 1,
        ref_card_failure_count: 1,
        shelf_metadata_ready_count: 0,
        shelf_export_ready_count: 0,
        shelf_summary_export_ready_count: 1,
        shelf_doi_count: 0,
        shelf_source_clickable_count: 1,
        shelf_review_count: 1,
        system_b_needs_review_count: 0,
      },
      citation_diagnostics: [
        {
          route: 'system_a',
          num: 1,
          anchor: 'scinerf-a1',
          title: 'SCINeRF citation',
          source_name: 'SCINeRF',
          source_path: 'db/scinerf/scinerf.en.md',
          heading_path: 'Method / ADMM',
          evidence_quote: 'ADMM is used as an optimization component.',
          quality_issues: [
            { name: 'system_a_missing_evidence', field: 'evidence_quote', severity: 'error' },
          ],
          shelf_quality_issues: [
            { name: 'shelf_doi_not_promoted', field: 'doi', severity: 'error' },
          ],
        },
      ],
      ref_diagnostics: [
        {
          title: 'SCINeRF ref card',
          source_name: 'SCINeRF',
          source_path: 'db/scinerf/scinerf.en.md',
          heading_path: 'Method / ADMM',
          score: 8.8,
          summary_line: 'SCINeRF explains how ADMM appears in the pipeline.',
          why_line: 'Relevant for the origin question.',
          polish_status: 'full',
          ref_pack_state: 'ready',
          evidence_quote: 'ADMM solver context.',
          quality_issues: [
            { name: 'ref_card_summary_too_short', field: 'summary_line', severity: 'error' },
          ],
        },
      ],
    }))
  })
  await page.goto('/__research_qa_replay__?case=scinerf-admm-origin&source=quality')

  await expect(page.getByTestId('research-qa-case-count')).toContainText('1')
  await expect(page.getByTestId('research-qa-diagnostic-case')).toContainText('scinerf-admm-origin')
  await expect(page.getByTestId('research-qa-diagnostic-failures')).toContainText('citation_card_quality')
  await expect(page.getByTestId('research-qa-diagnostic-docs')).toContainText('scinerf')
  await expect(page.getByTestId('research-qa-diagnostic-missing-docs')).toContainText('scigs')
  await expect(page.getByTestId('research-qa-diagnostic-quality-gates')).toContainText('citation failures 1')
  await expect(page.getByTestId('research-qa-diagnostic-quality-gates')).toContainText('shelf failures 1')
  await expect(page.getByTestId('research-qa-diagnostic-quality-gates')).toContainText('ref failures 1')
  await expect(page.getByTestId('research-qa-diagnostic-quality-gates')).toContainText('export ready 0')
  await expect(page.getByTestId('research-qa-diagnostic-quality-gates')).toContainText('summary export 1')
  await expect(page.getByTestId('research-qa-diagnostic-quality-gates')).toContainText('shelf review 1')
  await expect(page.getByTestId('research-qa-diagnostic-citations')).toContainText('SCINeRF citation')
  await expect(page.getByTestId('research-qa-diagnostic-citations')).toContainText('system_a_missing_evidence')
  await expect(page.getByTestId('research-qa-diagnostic-citations')).toContainText('shelf_doi_not_promoted')
  await expect(page.getByTestId('research-qa-diagnostic-refs')).toContainText('SCINeRF ref card')
  await expect(page.getByTestId('research-qa-diagnostic-refs')).toContainText('ref_card_summary_too_short')
  await expect(page.getByTestId('research-qa-case-scinerf-admm-origin')).toBeVisible()
  await expect(page.getByTestId('research-qa-case-scigs-dynamic-3d')).toHaveCount(0)
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
  await expect(page.locator('body')).not.toContainText('[[CITE:')
  await expect(page.locator('body')).not.toContainText('No summary available')
  await expect(page.locator('body')).not.toContainText('## Foveated')
  await expect(page.locator('body')).not.toContainText('适合作为定位入口')

  const firstRefsHeader = page.locator('.kb-refs-panel .ant-collapse-header').first()
  await expect(firstRefsHeader).toBeVisible()
  await firstRefsHeader.click()
  await expect(page.locator('.kb-ref-title').first()).toContainText('SCIGS')
  await expect(page.getByText('这条命中同时给出痛点、方法选择和动态场景目标')).toBeVisible()

  const systemAChip = page.locator('.kb-cite-chip:not(.kb-cite-chip-sysb)').first()
  await expect(systemAChip).toBeVisible()
  await expect(systemAChip).toHaveAttribute('href', /^#.+/)
  await systemAChip.click()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-a/)
  await expect(page.locator('.kb-cite-pop')).not.toContainText('[[CITE:')
  await expect(page.locator('.kb-cite-pop')).not.toContainText('## ')
  await expect(page.locator('.kb-cite-pop')).not.toContainText('has attrac')
  await expect(page.locator('.kb-cite-pop')).not.toContainText('No summary available')
  await expect(page.locator('.kb-cite-pop')).toContainText('SCIGS')
  await expect(page.getByTestId('citation-popover-system-a-compact-meta')).toContainText('SCIGS / Abstract')
  await expect(page.getByTestId('citation-popover-system-a-location')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-a-evidence')).toContainText('variant of 3DGS')
  await expect(page.getByTestId('citation-popover-system-a-claim')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-a-source')).toHaveCount(0)
  await page.locator('.kb-cite-pop-action-primary').click()
  await expect(page.getByTestId('research-qa-open-payload')).toContainText('"blockId": "scigs-abstract"')
  await expect(page.getByTestId('research-qa-open-payload')).toContainText('"strictLocate": true')

  const systemBChip = page.locator('.kb-cite-chip-sysb').first()
  await expect(systemBChip).toBeVisible()
  await expect(systemBChip).toHaveAttribute('href', /^#.+/)
  await systemBChip.scrollIntoViewIfNeeded()
  await systemBChip.click()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-b/)
  await expect(page.locator('.kb-cite-pop')).not.toContainText('[[CITE:')
  await expect(page.locator('.kb-cite-pop')).not.toContainText('```')
  await expect(page.locator('.kb-cite-pop')).not.toContainText('No summary available')
  await expect(page.locator('.kb-cite-pop')).not.toContainText('The paper cites')
  await expect(page.getByTestId('citation-popover-explain')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-flow')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-claim')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-role')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-relation')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-citing-source')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-takeaway')).toContainText('ADMM')
  await expect(page.getByTestId('citation-popover-system-b-takeaway')).toContainText('来源')
  await expect(page.getByTestId('citation-popover-system-b-context')).toHaveCount(0)
  await expect(page.getByTestId('citation-popover-system-b-location')).toContainText('当前论文引用处')
  await expect(page.getByTestId('citation-popover-system-b-location')).toContainText('SCINeRF')
  await expect(page.getByTestId('citation-popover-system-b-location')).not.toContainText('尚未定位到具体章节或页码')
  await expect(page.getByTestId('citation-popover-system-b-reference')).toContainText('alternating direction')
  await page.locator('.kb-cite-pop-add').click()
  await expect(page.getByTestId('citation-shelf')).toBeVisible()
  await expect(page.locator('.kb-shelf-kind')).toContainText('参考文献')
  await expect(page.locator('.kb-shelf-summary')).not.toContainText('No summary available')
})

test('research QA replay visual acceptance: refs and citation surfaces stay readable', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/__research_qa_replay__')
  await expectNoHorizontalOverflow(page)

  const firstRefsHeader = page.locator('.kb-refs-panel .ant-collapse-header').first()
  await expect(firstRefsHeader).toBeVisible()
  await firstRefsHeader.click()
  await expect(page.locator('.kb-ref-item').first()).toBeVisible()
  await expectNoLocalInlineOverflow(page, '.kb-ref-title, .kb-ref-card-title, .kb-ref-card-text, .kb-ref-action')

  const systemAChip = page.locator('.kb-cite-chip:not(.kb-cite-chip-sysb)').first()
  await systemAChip.scrollIntoViewIfNeeded()
  await systemAChip.click()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-a/)
  await expectElementInsideViewport(page, '.kb-cite-pop')
  await expectNoLocalInlineOverflow(page, '.kb-cite-pop-title, .kb-cite-pop-main, .kb-cite-pop-locator-text, .kb-cite-pop-quote blockquote, .kb-cite-pop-open-shelf, .kb-cite-pop-add')

  await page.locator('.kb-cite-pop-close').click()
  const systemBChip = page.locator('.kb-cite-chip-sysb').first()
  await systemBChip.scrollIntoViewIfNeeded()
  await systemBChip.click()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-b/)
  await expectElementInsideViewport(page, '.kb-cite-pop')
  await expectNoLocalInlineOverflow(page, '.kb-cite-pop-title, .kb-cite-pop-main, .kb-cite-pop-locator-text, .kb-cite-pop-quote blockquote, .kb-cite-pop-open-shelf, .kb-cite-pop-add')
  await expectNoHorizontalOverflow(page)
})

test('research QA replay citation shelf acceptance: saved literature stays useful', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/__research_qa_replay__')

  await addCitationToShelf(page, page.locator('.kb-cite-chip-sysb').filter({ hasText: 'R4' }).first())
  await page.getByTestId('citation-shelf-close').click()
  await addCitationToShelf(page, page.locator('.kb-cite-chip:not(.kb-cite-chip-sysb)').first())

  await expectCitationShelfQuality(page, {
    minItems: 2,
    requireMetadataReady: true,
    maxReviewItems: 0,
    minDoiLinks: 1,
    minSourceOpenButtons: 2,
  })
  const readiness = page.getByTestId('citation-shelf-readiness')
  await expect(readiness).toContainText(/2\/2/)
  await expect(readiness).not.toContainText(/0\/2/)
  await expectNoHorizontalOverflow(page)

  await page.getByTestId('citation-shelf-search').fill('SCINeRF')
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  await expect(page.getByTestId('citation-shelf-item').first()).toContainText('SCINeRF')
  await expectNoLocalInlineOverflow(page, '.kb-shelf-toolbar-main .ant-btn, .kb-shelf-item-title, .kb-shelf-item-source, .kb-shelf-summary-text')
})

test('research QA replay citation shelf workflow: snapshots and CSV export work', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/__research_qa_replay__')

  await addCitationToShelf(page, page.locator('.kb-cite-chip-sysb').filter({ hasText: 'R4' }).first())
  await page.getByTestId('citation-shelf-close').click()
  await addCitationToShelf(page, page.locator('.kb-cite-chip:not(.kb-cite-chip-sysb)').first())
  await expectCitationShelfQuality(page, {
    minItems: 2,
    requireMetadataReady: true,
    maxReviewItems: 0,
    minDoiLinks: 1,
    minSourceOpenButtons: 2,
  })

  await page.getByTestId('citation-shelf-save-snapshot').click()
  await expect(page.getByTestId('citation-shelf-load-snapshot')).toBeEnabled()
  await expect(page.getByTestId('citation-shelf-delete-snapshot')).toBeEnabled()

  await page.getByTestId('citation-shelf-clear').click()
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(0)

  await page.getByTestId('citation-shelf-load-snapshot').click()
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(2)
  await page.getByTestId('citation-shelf-item').first().click()
  await expectCitationShelfQuality(page, {
    minItems: 2,
    requireMetadataReady: true,
    maxReviewItems: 0,
    minDoiLinks: 1,
    minSourceOpenButtons: 2,
  })
  await expect(page.getByTestId('citation-shelf-source-quality-strip')).toHaveCount(0)
  await expect(page.getByTestId('citation-shelf')).not.toContainText('Missing page anchors')
  await expect(page.getByTestId('citation-shelf')).not.toContainText('定位校准中')

  await page.getByTestId('citation-shelf-search').fill('SCINeRF')
  await expect(page.getByTestId('citation-shelf-item')).toHaveCount(1)
  const autoRepairRequest = page.waitForRequest('**/api/library/quality/repair')
  const repairedSourceQuality = page.waitForResponse(async (response) => {
    if (!response.url().includes('/api/library/quality/sources') || response.request().method() !== 'POST') return false
    try {
      const body = await response.json() as { items?: Array<{ source_path?: string, conversion_quality?: { status?: string } }> }
      return Boolean((body.items || []).some((item) =>
        String(item.source_path || '').includes('SCINeRF')
        && String(item.conversion_quality?.status || '') === 'good',
      ))
    } catch {
      return false
    }
  })
  await page.locator('.kb-shelf-advanced-toggle').click()
  await page.getByTestId('citation-shelf-add-visible').click()
  const autoRepairPayload = autoRepairRequest.then((request) => request.postDataJSON() as { sources?: Array<{ source_path?: string }> })
  await expect.poll(async () => (await autoRepairPayload).sources?.[0]?.source_path || '').toContain('SCINeRF')
  await repairedSourceQuality
  await expect(page.getByTestId('citation-shelf-batch-count')).toContainText('1')
  await expect(page.getByTestId('citation-shelf-readiness')).toContainText(/2\/2/)
  await expect(page.getByTestId('citation-shelf-export-preflight')).toHaveCount(0)

  const downloadPromise = page.waitForEvent('download')
  await page.getByTestId('citation-shelf-export-csv').click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toMatch(/^cite_shelf_selected_\d{8}_\d{4}\.csv$/)
  const downloadPath = await download.path()
  expect(downloadPath, 'CSV export should produce a downloadable file').not.toBeNull()
  if (downloadPath) {
    const csv = await readFile(downloadPath, 'utf8')
    expect(csv).toContain('title,authors,year,venue,doi,source,source_quality_status,source_quality_issues')
    expect(csv).toContain('SCINeRF')
    expect(csv).toContain('good')
    expect(csv).not.toContain('Missing page anchors')
    expect(csv).not.toContain('No summary available')
    expect(csv).not.toContain('[[CITE:')
  }

  await page.getByTestId('citation-shelf-remove-visible').click()
  await expect(page.getByTestId('citation-shelf-batch-count')).toHaveCount(0)

  await page.getByTestId('citation-shelf-delete-snapshot').click()
  await expect(page.getByTestId('citation-shelf-load-snapshot')).toHaveCount(0)
  await expectNoHorizontalOverflow(page)
})

test('research QA replay mobile visual acceptance: cards stack without clipping', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/__research_qa_replay__')
  await expectNoHorizontalOverflow(page)

  const firstRefsHeader = page.locator('.kb-refs-panel .ant-collapse-header').first()
  await firstRefsHeader.scrollIntoViewIfNeeded()
  await firstRefsHeader.click()
  const firstCard = page.locator('.kb-ref-card').nth(0)
  const secondCard = page.locator('.kb-ref-card').nth(1)
  await expect(firstCard).toBeVisible()
  await expect(secondCard).toBeVisible()
  const [firstBox, secondBox] = await Promise.all([firstCard.boundingBox(), secondCard.boundingBox()])
  expect(firstBox).not.toBeNull()
  expect(secondBox).not.toBeNull()
  if (firstBox && secondBox) {
    expect(secondBox.y, 'mobile refs evidence cards should stack vertically').toBeGreaterThan(firstBox.y + firstBox.height - 1)
  }
  await expectNoLocalInlineOverflow(page, '.kb-ref-title, .kb-ref-card-title, .kb-ref-card-text, .kb-ref-action')

  const systemAChip = page.locator('.kb-cite-chip:not(.kb-cite-chip-sysb)').first()
  await systemAChip.scrollIntoViewIfNeeded()
  await systemAChip.click()
  await expect(page.locator('.kb-cite-pop')).toHaveClass(/kb-cite-pop-system-a/)
  await expectElementInsideViewport(page, '.kb-cite-pop')
  await expectNoLocalInlineOverflow(page, '.kb-cite-pop-title, .kb-cite-pop-main, .kb-cite-pop-locator-text, .kb-cite-pop-quote blockquote, .kb-cite-pop-open-shelf, .kb-cite-pop-add')
  await page.locator('.kb-cite-pop-add').click()
  await expectCitationShelfQuality(page, { minItems: 1 })
})
