import { expect, test, type Page, type Route } from '@playwright/test'

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

function settingsPayload(model: string, theme: 'light' | 'dark') {
  return {
    model,
    base_url: 'https://api.example.test/v1',
    has_api_key: true,
    db_dir: '',
    connection: {
      text: { has_api_key: true, model, base_url: 'https://api.example.test/v1' },
      vision: { has_api_key: true, model: `${model}-vision`, base_url: 'https://vision.example.test/v1' },
      auto_route: false,
    },
    readiness: {
      overall: { status: 'ok', severity: 'ok', reason: 'Ready' },
      providers: {
        text: { has_api_key: true, model, base_url: 'https://api.example.test/v1', status: 'ok', severity: 'ok', reason: 'Ready' },
        vision: { has_api_key: true, model: `${model}-vision`, base_url: 'https://vision.example.test/v1', status: 'ok', severity: 'ok', reason: 'Ready' },
      },
    },
    app_readiness: {
      status: 'ok',
      env: 'development',
      production: false,
      auth_required: false,
      items: [],
    },
    prefs: {
      ui_locale: 'zh',
      theme,
      top_k: 6,
      temperature: 0.2,
      max_tokens: 1216,
      deep_read: false,
    },
  }
}

async function openRegressionRoute(page: Page) {
  await page.route('**/api/sidebar**', async (route) => {
    await fulfillJson(route, { projects: [], root_conversations: [], project_conversations: {} })
  })
  await page.goto('/__message_list_test__')
}

test('settings load keeps the newest response when concurrent loads finish out of order', async ({ page }) => {
  let settingsCalls = 0
  await page.route('**/api/settings', async (route) => {
    if (route.request().method() !== 'GET') {
      await fulfillJson(route, { ok: true })
      return
    }
    settingsCalls += 1
    if (settingsCalls === 1) {
      await new Promise((resolve) => setTimeout(resolve, 250))
      await fulfillJson(route, settingsPayload('old-model', 'dark'))
      return
    }
    await fulfillJson(route, settingsPayload('new-model', 'light'))
  })
  await openRegressionRoute(page)

  const state = await page.evaluate(async () => {
    const { useSettingsStore } = await import('/src/stores/settingsStore.ts')
    useSettingsStore.setState({
      model: '',
      textModel: '',
      theme: 'dark',
      loaded: false,
      loadPending: false,
      loadError: '',
    })
    const first = useSettingsStore.getState().load()
    await new Promise((resolve) => setTimeout(resolve, 20))
    const second = useSettingsStore.getState().load()
    await Promise.all([first, second])
    const next = useSettingsStore.getState()
    return {
      model: next.model,
      textModel: next.textModel,
      theme: next.theme,
      loaded: next.loaded,
      loadPending: next.loadPending,
      loadError: next.loadError,
    }
  })

  expect(settingsCalls).toBe(2)
  expect(state.model).toBe('new-model')
  expect(state.textModel).toBe('new-model')
  expect(state.theme).toBe('light')
  expect(state.loaded).toBe(true)
  expect(state.loadPending).toBe(false)
  expect(state.loadError).toBe('')
})

test('settings load records failure without clearing the last usable settings', async ({ page }) => {
  await page.route('**/api/settings', async (route) => {
    if (route.request().method() !== 'GET') {
      await fulfillJson(route, { ok: true })
      return
    }
    await fulfillJson(route, { detail: 'settings unavailable' }, 503)
  })
  await openRegressionRoute(page)

  const state = await page.evaluate(async () => {
    const { useSettingsStore } = await import('/src/stores/settingsStore.ts')
    useSettingsStore.setState({
      model: 'previous-model',
      textModel: 'previous-model',
      theme: 'dark',
      loaded: true,
      loadPending: false,
      loadError: '',
    })
    await useSettingsStore.getState().load()
    const next = useSettingsStore.getState()
    return {
      model: next.model,
      textModel: next.textModel,
      theme: next.theme,
      loaded: next.loaded,
      loadPending: next.loadPending,
      loadError: next.loadError,
    }
  })

  expect(state.model).toBe('previous-model')
  expect(state.textModel).toBe('previous-model')
  expect(state.theme).toBe('dark')
  expect(state.loaded).toBe(true)
  expect(state.loadPending).toBe(false)
  expect(state.loadError).toContain('settings unavailable')
})

test('settings readiness refresh records failure and clears it after recovery', async ({ page }) => {
  let readinessCalls = 0
  await page.route('**/api/settings/readiness', async (route) => {
    readinessCalls += 1
    if (readinessCalls === 1) {
      await fulfillJson(route, { detail: 'readiness unavailable' }, 503)
      return
    }
    await fulfillJson(route, settingsPayload('ready-model', 'light').readiness)
  })
  await openRegressionRoute(page)

  const state = await page.evaluate(async () => {
    const { useSettingsStore } = await import('/src/stores/settingsStore.ts')
    useSettingsStore.setState({
      model: 'previous-model',
      textModel: 'previous-model',
      llmReadiness: null,
      readinessPending: false,
      readinessError: '',
    })
    let firstError = ''
    try {
      await useSettingsStore.getState().refreshReadiness()
    } catch (err) {
      firstError = err instanceof Error ? err.message : String(err || '')
    }
    const failed = useSettingsStore.getState()
    await useSettingsStore.getState().refreshReadiness()
    const recovered = useSettingsStore.getState()
    return {
      firstError,
      failedError: failed.readinessError,
      failedPending: failed.readinessPending,
      recoveredError: recovered.readinessError,
      recoveredPending: recovered.readinessPending,
      model: recovered.model,
      textModel: recovered.textModel,
      overallStatus: recovered.llmReadiness?.overall?.status,
    }
  })

  expect(readinessCalls).toBe(2)
  expect(state.firstError).toContain('readiness unavailable')
  expect(state.failedError).toContain('readiness unavailable')
  expect(state.failedPending).toBe(false)
  expect(state.recoveredError).toBe('')
  expect(state.recoveredPending).toBe(false)
  expect(state.model).toBe('ready-model')
  expect(state.textModel).toBe('ready-model')
  expect(state.overallStatus).toBe('ok')
})

test('app readiness refresh records failure and clears it after recovery', async ({ page }) => {
  let readinessCalls = 0
  await page.route('**/api/readiness', async (route) => {
    readinessCalls += 1
    if (readinessCalls === 1) {
      await fulfillJson(route, { detail: 'release readiness unavailable' }, 500)
      return
    }
    await fulfillJson(route, settingsPayload('ready-model', 'light').app_readiness)
  })
  await openRegressionRoute(page)

  const state = await page.evaluate(async () => {
    const { useSettingsStore } = await import('/src/stores/settingsStore.ts')
    useSettingsStore.setState({
      appReadiness: null,
      appReadinessPending: false,
      appReadinessError: '',
    })
    let firstError = ''
    try {
      await useSettingsStore.getState().refreshAppReadiness()
    } catch (err) {
      firstError = err instanceof Error ? err.message : String(err || '')
    }
    const failed = useSettingsStore.getState()
    await useSettingsStore.getState().refreshAppReadiness()
    const recovered = useSettingsStore.getState()
    return {
      firstError,
      failedError: failed.appReadinessError,
      failedPending: failed.appReadinessPending,
      recoveredError: recovered.appReadinessError,
      recoveredPending: recovered.appReadinessPending,
      status: recovered.appReadiness?.status,
      authRequired: recovered.appReadiness?.auth_required,
    }
  })

  expect(readinessCalls).toBe(2)
  expect(state.firstError).toContain('release readiness unavailable')
  expect(state.failedError).toContain('release readiness unavailable')
  expect(state.failedPending).toBe(false)
  expect(state.recoveredError).toBe('')
  expect(state.recoveredPending).toBe(false)
  expect(state.status).toBe('ok')
  expect(state.authRequired).toBe(false)
})
