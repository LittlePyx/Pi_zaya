import { expect, test, type Page, type Route } from '@playwright/test'

type ReportedIssue = {
  source?: string
  domain?: string
  severity?: string
  summary?: string
  detail?: string
  route?: string
  context?: Record<string, unknown>
  payload?: Record<string, unknown>
  fingerprint?: string
}

type ReporterTestWindow = Window & {
  __kbReporterUnhandledRejections?: number
}

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function installReporterBackend(page: Page, reports: ReportedIssue[]) {
  await page.route('**/*', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (!url.pathname.startsWith('/api/')) {
      await route.continue()
      return
    }

    if (url.pathname === '/api/user-issues' && request.method() === 'POST') {
      reports.push(request.postDataJSON() as ReportedIssue)
      await fulfillJson(route, { ok: true, issue: { id: 'issue-test' } })
      return
    }

    if (url.pathname === '/api/settings') {
      await fulfillJson(route, {
        model: 'test-model',
        base_url: '',
        has_api_key: true,
        db_dir: '',
        connection: {
          text: { has_api_key: true, base_url: '', model: 'test-model' },
          vision: { has_api_key: true, base_url: '', model: 'test-vision-model', uses_text_fallback: false },
          auto_route: true,
        },
        readiness: {
          providers: {
            text: {
              target: 'text',
              has_api_key: true,
              base_url: '',
              model: 'test-model',
              uses_text_fallback: false,
              status: 'ok',
              severity: 'ok',
              reason: 'ready',
            },
            vision: {
              target: 'vision',
              has_api_key: true,
              base_url: '',
              model: 'test-vision-model',
              uses_text_fallback: false,
              status: 'ok',
              severity: 'ok',
              reason: 'ready',
            },
          },
          overall: { status: 'ok', reason: 'ready', target: '' },
        },
        app_readiness: {
          status: 'ok',
          env: 'test',
          production: false,
          auth_required: false,
          items: [],
        },
        prefs: {
          ui_locale: 'en',
          theme: 'light',
          quality_data_sharing_enabled: false,
        },
      })
      return
    }

    if (url.pathname === '/api/auth/status') {
      await fulfillJson(route, {
        required: false,
        configured: false,
        authenticated: true,
        env: 'test',
        production: false,
      })
      return
    }

    if (url.pathname === '/api/sidebar') {
      await fulfillJson(route, {
        projects: [],
        root_conversations: [],
        project_conversations: {},
      })
      return
    }

    if (url.pathname === '/api/projects' || url.pathname === '/api/conversations') {
      await fulfillJson(route, [])
      return
    }

    if (url.pathname === '/api/readiness') {
      await fulfillJson(route, {
        status: 'ok',
        env: 'test',
        production: false,
        auth_required: false,
        items: [],
      })
      return
    }

    if (url.pathname === '/api/app/update-check') {
      await fulfillJson(route, {
        enabled: false,
        status: 'disabled',
        checked_at: 1780748500,
        current: { name: 'Pi_zaya', version: 'test' },
        latest: null,
        update_available: false,
        instructions: [],
      })
      return
    }

    await fulfillJson(route, {})
  })
}

async function installFailingReporterBackend(page: Page) {
  let issuePostCalls = 0
  await page.route('**/*', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (!url.pathname.startsWith('/api/')) {
      await route.continue()
      return
    }

    if (url.pathname === '/api/user-issues' && request.method() === 'POST') {
      issuePostCalls += 1
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'issue database temporarily unavailable' }),
      })
      return
    }

    if (url.pathname === '/api/settings') {
      await fulfillJson(route, {
        model: 'test-model',
        base_url: '',
        has_api_key: true,
        db_dir: '',
        connection: {
          text: { has_api_key: true, base_url: '', model: 'test-model' },
          vision: { has_api_key: true, base_url: '', model: 'test-vision-model', uses_text_fallback: false },
          auto_route: true,
        },
        readiness: {
          providers: {},
          overall: { status: 'ok', reason: 'ready', target: '' },
        },
        app_readiness: {
          status: 'ok',
          env: 'test',
          production: false,
          auth_required: false,
          items: [],
        },
        prefs: {
          ui_locale: 'en',
          theme: 'light',
          quality_data_sharing_enabled: false,
        },
      })
      return
    }

    if (url.pathname === '/api/auth/status') {
      await fulfillJson(route, {
        required: false,
        configured: false,
        authenticated: true,
        env: 'test',
        production: false,
      })
      return
    }

    if (url.pathname === '/api/sidebar') {
      await fulfillJson(route, {
        projects: [],
        root_conversations: [],
        project_conversations: {},
      })
      return
    }

    if (url.pathname === '/api/projects' || url.pathname === '/api/conversations') {
      await fulfillJson(route, [])
      return
    }

    if (url.pathname === '/api/readiness') {
      await fulfillJson(route, {
        status: 'ok',
        env: 'test',
        production: false,
        auth_required: false,
        items: [],
      })
      return
    }

    if (url.pathname === '/api/app/update-check') {
      await fulfillJson(route, {
        enabled: false,
        status: 'disabled',
        checked_at: 1780748500,
        current: { name: 'Pi_zaya', version: 'test' },
        latest: null,
        update_available: false,
        instructions: [],
      })
      return
    }

    await fulfillJson(route, {})
  })
  return {
    issuePostCalls: () => issuePostCalls,
  }
}

test('frontend issue reporter redacts sensitive runtime error material before truncation', async ({ page }) => {
  const reports: ReportedIssue[] = []
  await installReporterBackend(page, reports)
  await page.goto('/')
  await page.waitForFunction(() => Boolean(document.querySelector('#root')?.childElementCount))
  await expect.poll(() => reports.length).toBe(0)

  await page.evaluate(async () => {
    const mod = await import('/src/userIssueReporter.ts')
    mod.installUserIssueReporter()
    mod.installUserIssueReporter()

    const message = `${'x'.repeat(420)} sk-secretsecretsecret Authorization: Bearer abcdefghijklmnopqrstuvwxyz https://proxy.example/callback#access_token=fragmentsecret`
    window.dispatchEvent(new ErrorEvent('error', {
      message,
      filename: 'file:///C:/Users/Alice/private-paper.pdf',
      lineno: 12,
      colno: 34,
      error: new Error(message),
    }))
  })

  await expect.poll(() => reports.length).toBe(1)
  const report = reports[0]
  const body = JSON.stringify(report)

  expect(report.summary).toContain('[token]')
  expect(report.summary).not.toContain('sk-')
  expect(report.summary).toContain('Authorization: [token]')
  expect(report.summary).toContain('https://proxy.example/callback')
  expect(body).not.toContain('secretsecret')
  expect(body).not.toContain('abcdefghijklmnopqrstuvwxyz')
  expect(body).not.toContain('#access_token=')
  expect(body).not.toContain('fragmentsecret')
  expect(body).not.toContain('Alice')
  expect(body).not.toContain('private-paper')
  expect(report.payload?.source).toBe('[local-path]')
  expect(report.context?.url).toBe('/')
  expect(Object.prototype.hasOwnProperty.call(report.context || {}, 'user_agent')).toBe(false)
})

test('frontend issue reporter redacts UNC network paths before reporting', async ({ page }) => {
  const reports: ReportedIssue[] = []
  await installReporterBackend(page, reports)
  await page.goto('/')
  await page.waitForFunction(() => Boolean(document.querySelector('#root')?.childElementCount))
  await expect.poll(() => reports.length).toBe(0)

  await page.evaluate(async () => {
    const mod = await import('/src/userIssueReporter.ts')
    mod.installUserIssueReporter()

    const message = String.raw`Failed to open \\lab-server\private-share\secret-paper.pdf for alice@example.com with token sk-secretsecretsecret`
    window.dispatchEvent(new ErrorEvent('error', {
      message,
      filename: String.raw`\\lab-server\private-share\secret-paper.pdf`,
      lineno: 5,
      colno: 9,
      error: new Error(message),
    }))
  })

  await expect.poll(() => reports.length).toBe(1)
  const body = JSON.stringify(reports[0])

  expect(reports[0].summary).toContain('[local-path]')
  expect(reports[0].summary).toContain('[email]')
  expect(reports[0].summary).toContain('[token]')
  expect(reports[0].payload?.source).toBe('[local-path]')
  expect(body).not.toContain('lab-server')
  expect(body).not.toContain('private-share')
  expect(body).not.toContain('secret-paper')
  expect(body).not.toContain('alice@example.com')
  expect(body).not.toContain('sk-secret')
})

test('frontend issue reporter redacts external script URL paths before reporting', async ({ page }) => {
  const reports: ReportedIssue[] = []
  await installReporterBackend(page, reports)
  await page.goto('/')
  await page.waitForFunction(() => Boolean(document.querySelector('#root')?.childElementCount))
  await expect.poll(() => reports.length).toBe(0)

  await page.evaluate(async () => {
    const mod = await import('/src/userIssueReporter.ts')
    mod.installUserIssueReporter()

    window.dispatchEvent(new ErrorEvent('error', {
      message: 'External script failed to load',
      filename: 'https://private.example/workspaces/Alice/Secret-Lab-Draft.pdf?access_token=fragmentsecret#section',
      lineno: 7,
      colno: 11,
      error: new Error('External script failed to load'),
    }))
  })

  await expect.poll(() => reports.length).toBe(1)
  const report = reports[0]
  const body = JSON.stringify(report)

  expect(report.payload?.source).toBe('[source-redacted]')
  expect(report.context?.url).toBe('/')
  expect(body).not.toContain('private.example')
  expect(body).not.toContain('Secret-Lab-Draft')
  expect(body).not.toContain('Alice')
  expect(body).not.toContain('access_token')
  expect(body).not.toContain('fragmentsecret')
})

test('manual issue reporter redacts nested diagnostics before posting', async ({ page }) => {
  const reports: ReportedIssue[] = []
  await installReporterBackend(page, reports)
  await page.goto('/')
  await page.waitForFunction(() => Boolean(document.querySelector('#root')?.childElementCount))
  await expect.poll(() => reports.length).toBe(0)

  await page.evaluate(async () => {
    const mod = await import('/src/userIssueReporter.ts')
    mod.reportUserIssue({
      source: 'frontend C:/Users/Alice/private-paper.pdf',
      domain: 'chat_generation',
      severity: 'error',
      summary: 'Chat send failed',
      detail: 'Stream failed with Authorization: Bearer abcdefghijklmnopqrstuvwxyz',
      route: '/chat#token=fragmentsecret',
      context: {
        user_agent: 'Mozilla/5.0 Secret Lab Browser alice@example.com',
        prompt_text: 'Explain Secret Lab Draft in detail',
        question: 'Why does Secret Lab Draft mention Alice?',
        source_path: 'C:/Users/Alice/private-paper.pdf',
        selected_count: 3,
        samples: ['Private converted paragraph from Secret Lab Draft'],
        nested: {
          answer_text: 'The private prototype failed.',
          example_text: 'Another private example from the paper.',
          code: 'citation_missing',
        },
      },
      payload: {
        error_kind: 'generation_stream_failed',
        question_text: 'What does Secret Lab Draft say?',
        filename: 'Secret Lab Draft.pdf',
        documents: ['Secret Lab Draft.pdf'],
        paper_count: 1,
        items: [
          {
            quote_text: 'A private quote from the converted paper.',
            code: 'missing_images',
          },
        ],
      },
      fingerprint: 'chat-send|C:/Users/Alice/private-paper.pdf|sk-secretsecretsecret',
    })
  })

  await expect.poll(() => reports.length).toBe(1)
  const report = reports[0]
  const body = JSON.stringify(report)
  const items = report.payload?.items as Array<Record<string, unknown>> | undefined

  expect(report.source).toBe('frontend [local-path]')
  expect(report.domain).toBe('chat_generation')
  expect(report.detail).toContain('Authorization: [token]')
  expect(report.route).toBe('/chat')
  expect(report.context?.prompt_text).toBe('[redacted]')
  expect(report.context?.question).toBe('[redacted]')
  expect(report.context?.source_path).toBe('[redacted]')
  expect(report.context?.user_agent).toBe('[redacted]')
  expect(report.context?.selected_count).toBe(3)
  expect(report.context?.samples).toBe('[redacted]')
  expect((report.context?.nested as Record<string, unknown> | undefined)?.answer_text).toBe('[redacted]')
  expect((report.context?.nested as Record<string, unknown> | undefined)?.example_text).toBe('[redacted]')
  expect((report.context?.nested as Record<string, unknown> | undefined)?.code).toBe('citation_missing')
  expect(report.payload?.error_kind).toBe('generation_stream_failed')
  expect(report.payload?.question_text).toBe('[redacted]')
  expect(report.payload?.filename).toBe('[redacted]')
  expect(report.payload?.documents).toBe('[redacted]')
  expect(report.payload?.paper_count).toBe(1)
  expect(items?.[0]?.quote_text).toBe('[redacted]')
  expect(items?.[0]?.code).toBe('missing_images')
  expect(report.fingerprint).toMatch(/^frontend-[a-f0-9]{8}-[a-z0-9]+$/)
  expect(body).not.toContain('Secret Lab')
  expect(body).not.toContain('Alice')
  expect(body).not.toContain('Mozilla/5.0')
  expect(body).not.toContain('private-paper')
  expect(body).not.toContain('private prototype')
  expect(body).not.toContain('abcdefghijklmnopqrstuvwxyz')
  expect(body).not.toContain('sk-secret')
  expect(body).not.toContain('#token=')
  expect(body).not.toContain('fragmentsecret')
})

test('frontend issue reporter stays silent when local issue storage fails', async ({ page }) => {
  const backend = await installFailingReporterBackend(page)
  const pageErrors: string[] = []
  const consoleErrors: string[] = []
  page.on('pageerror', (error) => pageErrors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text())
  })

  await page.addInitScript(() => {
    const target = window as ReporterTestWindow
    target.__kbReporterUnhandledRejections = 0
    window.addEventListener('unhandledrejection', () => {
      target.__kbReporterUnhandledRejections = (target.__kbReporterUnhandledRejections || 0) + 1
    })
  })
  await page.goto('/')
  await page.waitForFunction(() => Boolean(document.querySelector('#root')?.childElementCount))

  await page.evaluate(async () => {
    const mod = await import('/src/userIssueReporter.ts')
    mod.reportUserIssue({
      source: 'frontend',
      domain: 'runtime',
      severity: 'error',
      summary: 'Synthetic reporter failure test',
      detail: 'This issue write is expected to fail server-side.',
      fingerprint: 'synthetic-reporter-failure',
    })
  })

  await expect.poll(() => backend.issuePostCalls()).toBe(1)
  await page.waitForTimeout(250)

  expect(await page.evaluate(() => (window as ReporterTestWindow).__kbReporterUnhandledRejections || 0)).toBe(0)
  expect(pageErrors).toEqual([])
  expect(consoleErrors.join('\n')).not.toContain('issue database temporarily unavailable')
})
