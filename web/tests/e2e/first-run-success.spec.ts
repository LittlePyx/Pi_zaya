import { expect, test, type Page, type Route } from '@playwright/test'

type Phase = 'connect_model' | 'prepare_document' | 'ask_question' | 'completed'

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function installCleanProfileBackend(page: Page, getPhase: () => Phase) {
  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const requestUrl = new URL(route.request().url())
    const path = requestUrl.pathname
    const phase = getPhase()
    const hasKey = phase !== 'connect_model'

    if (path === '/api/auth/status') {
      await fulfillJson(route, { required: false, configured: false, authenticated: true, env: 'test', production: false })
      return
    }
    if (path === '/api/app/onboarding-status') {
      await fulfillJson(route, {
        text_model_ready: hasKey,
        imported_document_count: phase === 'connect_model' || phase === 'prepare_document' ? 0 : 1,
        ready_document_count: phase === 'ask_question' || phase === 'completed' ? 1 : 0,
        grounded_answer_count: phase === 'completed' ? 1 : 0,
        current_step: phase,
        completed: phase === 'completed',
      })
      return
    }
    if (path === '/api/app/update-check') {
      await fulfillJson(route, { enabled: false, status: 'disabled', current: { name: 'Pi_zaya', version: 'test' }, latest: null, update_available: false, instructions: [] })
      return
    }
    if (path === '/api/settings' || path === '/api/settings/readiness') {
      const text = {
        target: 'text', has_api_key: hasKey, base_url: '', model: hasKey ? 'test-model' : '',
        status: hasKey ? 'ok' : 'missing', severity: hasKey ? 'ok' : 'error', reason: hasKey ? 'Ready' : 'API key missing',
      }
      const vision = { ...text, target: 'vision', status: 'fallback', severity: 'warning', uses_text_fallback: true }
      const readiness = { providers: { text, vision }, overall: hasKey ? { status: 'warning', reason: 'Vision is optional', target: 'vision' } : { status: 'error', reason: 'API key missing', target: 'text' } }
      if (path === '/api/settings/readiness') {
        await fulfillJson(route, readiness)
      } else {
        await fulfillJson(route, {
          model: text.model,
          base_url: '',
          has_api_key: hasKey,
          connection: { text, vision, auto_route: false },
          readiness,
          app_readiness: { status: hasKey ? 'warning' : 'error', env: 'test', production: false, auth_required: false, items: [], llm: readiness },
          db_dir: 'C:/Pi_zaya/data/db',
          library_paths: {
            pdf_dir: 'C:/Pi_zaya/data/pdfs',
            md_dir: 'C:/Pi_zaya/data/markdown',
            pdf_source: 'environment',
            md_source: 'environment',
            uses_managed_defaults: true,
          },
          prefs: { ui_locale: 'en', theme: 'light', top_k: 6, max_tokens: 1216 },
        })
      }
      return
    }
    if (path === '/api/readiness') {
      await fulfillJson(route, { status: 'ok', env: 'test', production: false, auth_required: false, items: [] })
      return
    }
    if (path === '/api/sidebar') {
      await fulfillJson(route, { projects: [], root_conversations: [], project_conversations: {} })
      return
    }
    if (path === '/api/projects' || path === '/api/conversations') {
      await fulfillJson(route, [])
      return
    }
    if (path === '/api/library/files') {
      await fulfillJson(route, {
        items: [],
        counts: { total_view: 0, total_all: 0, pending: 0, converted: 0, queued: 0, running: 0, reconverting: 0, quality_review: 0, quality_ready: 0 },
        truncated: false,
        scope: requestUrl.searchParams.get('scope') || '200',
        queue: { running: false, active_count: 0, active_tasks: [], current: '', done: 0, total: 0 },
      })
      return
    }
    if (path === '/api/references/sync/status' || path === '/api/library/convert/status') {
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: {"running":false,"done":true,"status":"idle"}\n\n' })
      return
    }
    if (path.startsWith('/api/references/conversation/')) {
      await fulfillJson(route, {})
      return
    }
    if (path.startsWith('/api/chat/citation-shelf')) {
      await fulfillJson(route, { version: 1, scope: 'project', scope_id: 'root', project_id: null, items: [], open: false, revision: 0, updated_at: 0 })
      return
    }
    await fulfillJson(route, {})
  })
}

test('clean profile progresses from model setup to the first cited answer', async ({ page }) => {
  let phase: Phase = 'connect_model'
  await installCleanProfileBackend(page, () => phase)

  await page.goto('/')
  const guide = page.getByTestId('first-run-api-guide')
  await expect(guide).toHaveAttribute('data-current-step', 'connect_model')
  await expect(page.locator('.kb-chat-connection-alert')).toHaveCount(0)
  await guide.getByRole('button', { name: 'Configure text API key' }).click()
  await expect(page.locator('[data-api-target="text"]')).toHaveClass(/is-targeted/)

  phase = 'prepare_document'
  await page.goto('/library')
  await expect(guide).toHaveAttribute('data-current-step', 'prepare_document')
  await expect(page.getByText('C:/Pi_zaya/data/pdfs', { exact: true })).toBeVisible()
  await expect(page.getByText('Using application-managed default directories')).toBeVisible()
  await expect(page.getByTestId('library-show-advanced')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Update KB' })).toHaveCount(0)

  phase = 'ask_question'
  await page.reload()
  await expect(guide).toHaveAttribute('data-current-step', 'ask_question')
  await guide.getByRole('button', { name: 'Ask a question' }).click()
  await expect(page).toHaveURL(/\/$/)

  phase = 'completed'
  await page.reload()
  await expect(page.getByTestId('first-run-api-guide')).toHaveCount(0)
})
