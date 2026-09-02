import { expect, test } from '@playwright/test'

test('library auto recovery selects safe repair and reindex actions only', async ({ page }) => {
  await page.goto('/__message_list_test__')

  const result = await page.evaluate(async () => {
    const { libraryAutoRecoveryKind } = await import('/src/pages/library/libraryAutoRecovery.ts')
    const base = {
      name: 'paper.pdf',
      path: 'F:/pdfs/paper.pdf',
      md_exists: true,
      md_path: 'F:/markdown/paper.md',
      md_folder: 'F:/markdown',
      category: 'converted',
      task_state: 'idle' as const,
      status: 'converted',
      replace_task: false,
      queue_pos: 0,
      cur_page_done: 0,
      cur_page_total: 0,
      cur_page_msg: '',
      conversion_quality: null,
    }
    return {
      repair: libraryAutoRecoveryKind({
        ...base,
        index_state: 'quality_blocked',
        conversion_quality: {
          status: 'warning',
          label: 'Needs repair',
          score: 71,
          summary: 'repairable',
          has_review_issue: true,
          issues: [],
          metrics: {},
          conversion_report: null,
        },
      }),
      reindex: libraryAutoRecoveryKind({ ...base, index_state: 'index_stale' }),
      ready: libraryAutoRecoveryKind({ ...base, index_state: 'ready' }),
      missingMarkdown: libraryAutoRecoveryKind({ ...base, md_exists: false, index_state: 'not_converted' }),
      busy: libraryAutoRecoveryKind({ ...base, task_state: 'running', index_state: 'index_stale' }),
    }
  })

  expect(result).toEqual({
    repair: 'repair',
    reindex: 'reindex',
    ready: '',
    missingMarkdown: '',
    busy: '',
  })
})
