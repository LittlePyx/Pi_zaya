import { expect, test, type Page, type Route } from '@playwright/test'

type ProviderStatus = {
  has_api_key: boolean
  status: 'missing' | 'fallback' | 'configured' | 'ok' | 'failed'
  severity: 'ok' | 'warning' | 'error'
  reason?: string
  model?: string
  base_url?: string
  uses_text_fallback?: boolean
}

type AppReadinessMock = {
  status: 'ok' | 'warning' | 'error'
  env: string
  production: boolean
  auth_required: boolean
  items: Array<{
    key: string
    status: 'ok' | 'warning' | 'error'
    severity: 'ok' | 'warning' | 'error'
    label: string
    detail: string
    action: string
  }>
  llm: unknown
  restore: {
    latest: Record<string, unknown> | null
    acknowledgement: Record<string, unknown> | null
    acknowledged: boolean
  }
}

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

async function enableInternalSettingsTools(page: Page) {
  await page.addInitScript(() => {
    window.sessionStorage.setItem('kb.internal.showSettingsDiagnostics', '1')
  })
}

function provider(target: 'text' | 'vision', overrides: Partial<ProviderStatus> = {}) {
  const hasKey = overrides.has_api_key ?? true
  const status = overrides.status || (hasKey ? 'ok' : 'missing')
  const severity = overrides.severity || (hasKey ? 'ok' : 'error')
  return {
    target,
    has_api_key: hasKey,
    base_url: overrides.base_url || '',
    model: overrides.model || (target === 'text' ? 'test-text-model' : 'test-vision-model'),
    uses_text_fallback: Boolean(overrides.uses_text_fallback),
    status,
    severity,
    reason: overrides.reason || (hasKey ? 'Ready' : 'API key missing'),
  }
}

async function installMinimalBackend(
  page: Page,
  options: {
    text?: Partial<ProviderStatus>
    vision?: Partial<ProviderStatus>
    restoreReview?: boolean
    authRequired?: boolean
    failAnswerDepthPatch?: boolean
    failQualityDataSharingPatch?: boolean
    failQualityDataCleanup?: boolean
    failSettingsReloadAfterQualityPatch?: boolean
    qualityCollectorLocal?: boolean
    qualityCollectorInsecure?: boolean
    qualityCollectorCredentials?: boolean
    qualityCollectorInvalidPort?: boolean
    qualityCollectorMissingToken?: boolean
    qualityCollectorStatusFails?: boolean
  } = {},
) {
  const text = provider('text', options.text)
  const vision = provider('vision', options.vision)
  let autoBackupEnabled = true
  let qualityDataSharingEnabled = false
  let authenticated = Boolean(options.authRequired)
  let logoutCalls = 0
  let qualityCollectorPending = 2
  let qualityCollectorTests = 0
  let qualityCollectorFlushes = 0
  let failNextSettingsGet = false
  const qualityCollectorHost = options.qualityCollectorLocal
    ? '127.0.0.1:9000'
    : options.qualityCollectorInvalidPort
      ? 'collector.example:bad'
      : 'collector.example'
  const qualityCollectorBlocked = Boolean(
    options.qualityCollectorLocal
    || options.qualityCollectorInsecure
    || options.qualityCollectorCredentials
    || options.qualityCollectorInvalidPort
    || options.qualityCollectorMissingToken,
  )
  const settingsPatches: Array<Record<string, unknown>> = []
  const readiness = {
    providers: { text, vision },
    overall: text.severity === 'error'
      ? { status: 'error', reason: text.reason, target: 'text' }
      : vision.severity === 'error' || vision.status === 'fallback'
        ? { status: vision.severity === 'error' ? 'error' : 'warning', reason: vision.reason, target: 'vision' }
        : { status: 'ok', reason: 'Ready', target: '' },
  }
  const appReadiness: AppReadinessMock = {
    status: text.severity === 'error' ? 'error' : 'warning',
    env: 'production',
    production: true,
    auth_required: true,
    items: [
      {
        key: 'text_llm',
        status: text.severity,
        severity: text.severity,
        label: 'Text model',
        detail: text.reason,
        action: text.severity === 'error' ? 'configure_text_api_key' : '',
      },
      {
        key: 'api_auth',
        status: 'ok',
        severity: 'ok',
        label: 'API access protection',
        detail: 'Enabled',
        action: '',
      },
      {
        key: 'cors',
        status: 'warning',
        severity: 'warning',
        label: 'CORS origins',
        detail: 'Review allowed origins',
        action: 'set_allowed_origins',
      },
    ],
    llm: readiness,
    restore: {
      latest: null,
      acknowledgement: null,
      acknowledged: false,
    },
  }
  const auditEvents: Array<Record<string, unknown>> = []
  if (options.restoreReview) {
    appReadiness.items.push({
      key: 'recent_restore',
      status: 'warning',
      severity: 'warning',
      label: 'Recent restore',
      detail: 'Backup backup-20260606-120000.zip was restored recently.',
      action: 'restart_and_check',
    })
    appReadiness.restore = {
      latest: {
        event: 'restore',
        status: 'restored',
        backup: 'backup-20260606-120000.zip',
        created_at: 1780747200,
        ok: true,
        restart_required: true,
      },
      acknowledgement: null,
      acknowledged: false,
    }
    auditEvents.unshift({
      event: 'restore',
      status: 'restored',
      ok: true,
      backup: 'backup-20260606-120000.zip',
      created_at: 1780747200,
      restart_required: true,
      restored_count: 3,
      components: { chat: true, library: true, db: true },
      checks: {},
      errors: [],
      warnings: [],
    })
  }
  const backups = [
    {
      name: 'backup-20260606-120000.zip',
      created_at: 1780747200,
      label: 'manual',
      size_bytes: 2048,
      path: 'F:/tmp/backup-20260606-120000.zip',
    },
  ]

  await page.route('**/api/maintenance/status', async (route) => {
    await fulfillJson(route, {
      data_protection: {
        enabled: autoBackupEnabled,
        status: 'enabled',
        can_toggle: true,
        manual_backup_available: true,
        backup_count: backups.length,
        latest_backup: backups[0] || null,
      },
      auto_backup: {
        enabled: autoBackupEnabled,
        strict: false,
        min_interval_s: 30,
        source: 'user',
        locked: false,
      },
      backups: {
        count: backups.length,
        latest: backups[0] || null,
        keep: 30,
        directory: 'F:/tmp/backups',
      },
    })
  })
  await page.route('**/api/auth/status', async (route) => {
    await fulfillJson(route, {
      required: Boolean(options.authRequired),
      configured: Boolean(options.authRequired),
      authenticated,
      env: 'test',
      production: false,
    })
  })
  await page.route('**/api/auth/logout', async (route) => {
    logoutCalls += 1
    authenticated = false
    await fulfillJson(route, { ok: true })
  })
  await page.route('**/api/sidebar**', async (route) => {
    await fulfillJson(route, {
      projects: [],
      root_conversations: [],
      project_conversations: {},
    })
  })
  await page.route('**/api/app/update-check**', async (route) => {
    await fulfillJson(route, {
      enabled: false,
      status: 'disabled',
      checked_at: 1780748500,
      current: { name: 'Pi_zaya', version: 'test' },
      latest: null,
      update_available: false,
      instructions: [],
    })
  })
  await page.route('**/api/settings', async (route) => {
    if (route.request().method() === 'PATCH') {
      const raw = route.request().postData() || '{}'
      const patch = JSON.parse(raw) as {
        answer_depth_auto?: boolean
        auto_backup_enabled?: boolean
        quality_data_sharing_enabled?: boolean
      }
      settingsPatches.push(patch)
      if (typeof patch.answer_depth_auto === 'boolean' && options.failAnswerDepthPatch) {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'preference save failed' }),
        })
        return
      }
      if (typeof patch.quality_data_sharing_enabled === 'boolean' && options.failQualityDataSharingPatch) {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'quality sharing save failed' }),
        })
        return
      }
      if (typeof patch.auto_backup_enabled === 'boolean') {
        autoBackupEnabled = patch.auto_backup_enabled
      }
      if (typeof patch.quality_data_sharing_enabled === 'boolean') {
        qualityDataSharingEnabled = patch.quality_data_sharing_enabled
        if (options.failSettingsReloadAfterQualityPatch) {
          failNextSettingsGet = true
        }
      }
      await fulfillJson(route, {
        ok: true,
        ...(patch.quality_data_sharing_enabled === false
          ? {
              quality_data_cleanup: options.failQualityDataCleanup
                ? { ok: false, removed: 0, error: 'database locked' }
                : { ok: true, removed: qualityCollectorPending, error: '' },
            }
          : {}),
      })
      return
    }
    if (failNextSettingsGet) {
      failNextSettingsGet = false
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'settings reload failed after quality sharing save' }),
      })
      return
    }
    await fulfillJson(route, {
      model: text.model,
      base_url: text.base_url,
      has_api_key: text.has_api_key,
      connection: {
        text: { has_api_key: text.has_api_key, model: text.model, base_url: text.base_url },
        vision: {
          has_api_key: vision.has_api_key,
          model: vision.model,
          base_url: vision.base_url,
          uses_text_fallback: vision.uses_text_fallback,
        },
        auto_route: false,
      },
      readiness,
      app_readiness: appReadiness,
      prefs: {
        ui_locale: 'en',
        theme: 'light',
        top_k: 6,
        temperature: 0.2,
        max_tokens: 1216,
        deep_read: false,
        auto_backup_enabled: autoBackupEnabled,
        quality_data_sharing_enabled: qualityDataSharingEnabled,
      },
    })
  })
  await page.route('**/api/settings/readiness', async (route) => {
    await fulfillJson(route, readiness)
  })
  await page.route('**/api/readiness', async (route) => {
    await fulfillJson(route, appReadiness)
  })
  await page.route('**/api/maintenance/backups', async (route) => {
    if (route.request().method() === 'POST') {
      const created = {
        name: 'backup-20260606-121500-manual.zip',
        created_at: 1780748100,
        label: 'manual',
        size_bytes: 4096,
        path: 'F:/tmp/backup-20260606-121500-manual.zip',
      }
      backups.unshift(created)
      await fulfillJson(route, created)
      return
    }
    await fulfillJson(route, { items: backups })
  })
  await page.route('**/api/maintenance/backups/cleanup', async (route) => {
    const removed = backups.splice(1)
    await fulfillJson(route, {
      ok: true,
      keep: 1,
      before: backups.length + removed.length,
      deleted: removed.length,
      failed: 0,
      dry_run: false,
      items: removed,
      errors: [],
    })
  })
  await page.route('**/api/maintenance/restore-audit**', async (route) => {
    await fulfillJson(route, { items: auditEvents })
  })
  await page.route('**/api/maintenance/restore-review/acknowledge', async (route) => {
    auditEvents.unshift({
      event: 'restore_review_acknowledged',
      status: 'acknowledged',
      ok: true,
      backup: 'backup-20260606-120000.zip',
      created_at: 1780748500,
      restore_created_at: 1780747200,
      checks: {
        api_restarted: true,
        api_keys_checked: true,
        knowledge_base_checked: true,
        chat_history_checked: true,
        library_data_checked: true,
      },
      errors: [],
      warnings: [],
    })
    appReadiness.items = appReadiness.items.filter((item) => item.key !== 'recent_restore')
    appReadiness.restore = {
      ...appReadiness.restore,
      acknowledgement: {
        event: 'restore_review_acknowledged',
        status: 'acknowledged',
        backup: 'backup-20260606-120000.zip',
        created_at: 1780748500,
        ok: true,
      },
      acknowledged: true,
    }
    await fulfillJson(route, {
      ok: true,
      status: 'acknowledged',
      backup: 'backup-20260606-120000.zip',
      restore_created_at: 1780747200,
      acknowledged_at: 1780748500,
    })
  })
  await page.route('**/api/maintenance/backups/*/verify', async (route) => {
    const name = decodeURIComponent(route.request().url().split('/backups/')[1]?.split('/verify')[0] || '')
    await fulfillJson(route, {
      ok: true,
      name,
      errors: [],
      warnings: [],
      checks: {
        zip: { ok: true },
        sqlite: { 'chat.sqlite3': { ok: true }, 'library.sqlite3': { ok: true } },
      },
      verified_at: 1780748200,
    })
  })
  await page.route('**/api/maintenance/backups/*/restore-dry-run', async (route) => {
    const name = decodeURIComponent(route.request().url().split('/backups/')[1]?.split('/restore-dry-run')[0] || '')
    await fulfillJson(route, {
      ok: true,
      can_restore: true,
      name,
      extracted_file_count: 4,
      destinations: [
        { archive: 'chat.sqlite3', source_exists: true, target: 'F:/tmp/chat.sqlite3' },
        { archive: 'library.sqlite3', source_exists: true, target: 'F:/tmp/library.sqlite3' },
        { archive: 'db/', source_exists: true, target: 'F:/tmp/db', source_file_count: 2 },
      ],
      errors: [],
      warnings: [],
      restore_steps: ['Stop server', 'Copy files', 'Restart server'],
      checked_at: 1780748300,
    })
  })
  await page.route('**/api/maintenance/backups/*/restore', async (route) => {
    const name = decodeURIComponent(route.request().url().split('/backups/')[1]?.split('/restore')[0] || '')
    auditEvents.unshift({
      event: 'restore',
      status: 'restored',
      ok: true,
      backup: name,
      created_at: 1780748400,
      restart_required: true,
      restored_count: 3,
      components: { chat: true, library: true, db: true },
      checks: {},
      errors: [],
      warnings: [],
    })
    await fulfillJson(route, {
      ok: true,
      name,
      status: 'restored',
      pre_restore_backup: {
        name: 'backup-20260606-122000-pre-restore.zip',
        created_at: 1780748400,
        label: 'pre-restore',
        size_bytes: 8192,
      },
      restored: [
        { kind: 'directory', target: 'F:/tmp/db', file_count: 2 },
        { kind: 'file', target: 'F:/tmp/chat.sqlite3', size_bytes: 1024 },
        { kind: 'file', target: 'F:/tmp/library.sqlite3', size_bytes: 1024 },
      ],
      errors: [],
      warnings: [],
      restart_required: true,
    })
  })
  await page.route('**/api/maintenance/diagnostics/export', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/zip',
      headers: { 'content-disposition': 'attachment; filename="diagnostics-test.zip"' },
      body: 'zip',
    })
  })
  await page.route('**/api/user-issues/remote/status', async (route) => {
    if (options.qualityCollectorStatusFails) {
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'collector status failed' }),
      })
      return
    }
    await fulfillJson(route, {
      ok: true,
      enabled: qualityDataSharingEnabled && !qualityCollectorBlocked,
      remote_enabled: true,
      remote_url_configured: true,
      remote_url_host: qualityCollectorHost,
      remote_url_scheme: options.qualityCollectorInsecure ? 'http' : 'https',
      remote_url_has_valid_scheme: true,
      remote_url_has_valid_port: !options.qualityCollectorInvalidPort,
      remote_url_has_credentials: Boolean(options.qualityCollectorCredentials),
      remote_url_is_local: Boolean(options.qualityCollectorLocal),
      remote_url_local_allowed: false,
      remote_url_secure: !options.qualityCollectorInsecure,
      remote_url_allowed: !qualityCollectorBlocked,
      remote_block_reason: options.qualityCollectorLocal
        ? 'local_remote_url'
        : options.qualityCollectorInsecure
          ? 'insecure_remote_url'
          : options.qualityCollectorCredentials
            ? 'remote_url_credentials'
            : options.qualityCollectorInvalidPort
              ? 'invalid_remote_url'
              : options.qualityCollectorMissingToken
                ? 'missing_remote_token'
              : '',
      remote_token_configured: !options.qualityCollectorMissingToken,
      remote_token_required: true,
      remote_unauthenticated_allowed: false,
      quality_data_sharing_enabled: qualityDataSharingEnabled,
      outbox: {
        total: qualityDataSharingEnabled ? qualityCollectorPending : 0,
        pending: qualityDataSharingEnabled ? qualityCollectorPending : 0,
        retryable: qualityDataSharingEnabled ? qualityCollectorPending : 0,
        sent: 0,
        latest_error: '',
        latest_attempts: 0,
        next_attempt_at: 0,
      },
    })
  })
  await page.route('**/api/user-issues/remote/test', async (route) => {
    qualityCollectorTests += 1
    await fulfillJson(route, {
      ok: qualityDataSharingEnabled && !options.qualityCollectorLocal,
      enabled: qualityDataSharingEnabled && !options.qualityCollectorLocal,
      status_code: qualityDataSharingEnabled && !options.qualityCollectorLocal ? 200 : 0,
      error: qualityDataSharingEnabled && !options.qualityCollectorLocal ? '' : 'remote reporting is disabled',
      outbox: {
        total: qualityCollectorPending,
        pending: qualityCollectorPending,
        retryable: qualityCollectorPending,
        sent: 0,
      },
    })
  })
  await page.route('**/api/user-issues/outbox/flush**', async (route) => {
    qualityCollectorFlushes += 1
    const sent = qualityDataSharingEnabled ? qualityCollectorPending : 0
    qualityCollectorPending = 0
    await fulfillJson(route, {
      ok: qualityDataSharingEnabled,
      enabled: qualityDataSharingEnabled,
      sent,
      failed: 0,
      summary: {
        total: 0,
        pending: 0,
        retryable: 0,
        sent,
      },
    })
  })
  await page.route('**/api/projects', async (route) => {
    await fulfillJson(route, [])
  })
  await page.route(/\/api\/conversations(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, [])
  })
  await page.route('**/api/references/conversation/**', async (route) => {
    await fulfillJson(route, {})
  })
  await page.route('**/api/chat/citation-shelf**', async (route) => {
    await fulfillJson(route, { version: 1, scope: 'project', items: [], revision: 1, updated_at: 0 })
  })
  return {
    settingsPatches,
    getLogoutCalls: () => logoutCalls,
    getQualityCollectorTests: () => qualityCollectorTests,
    getQualityCollectorFlushes: () => qualityCollectorFlushes,
  }
}

test('text API block opens settings focused on the text provider', async ({ page }) => {
  await installMinimalBackend(page, {
    text: { has_api_key: false, status: 'missing', severity: 'error' },
  })
  await page.goto('/')

  await expect(page.getByTestId('research-context-state')).toHaveAttribute('data-research-api-block-target', 'text')
  const alert = page.locator('.kb-chat-connection-alert')
  await expect(alert).toBeVisible()

  await alert.getByRole('button', { name: 'Open settings', exact: true }).click()

  await expect(page.locator('[data-api-target="text"]')).toHaveClass(/is-targeted/)
  await expect(page.locator('[data-api-target="vision"]')).not.toHaveClass(/is-targeted/)
  await expect.poll(async () => (
    page.evaluate(() => document.activeElement?.closest('[data-api-target]')?.getAttribute('data-api-target') || '')
  )).toBe('text')
})

test('settings event can focus the vision provider', async ({ page }) => {
  await installMinimalBackend(page)
  await page.goto('/')
  await expect(page.getByTestId('research-context-state')).toHaveAttribute('data-research-api-text', 'ok')

  await page.evaluate(() => {
    window.dispatchEvent(new CustomEvent('kb:open-settings', { detail: { target: 'vision' } }))
  })

  await expect(page.locator('[data-api-target="vision"]')).toHaveClass(/is-targeted/)
  await expect(page.locator('[data-api-target="text"]')).not.toHaveClass(/is-targeted/)
  await expect.poll(async () => (
    page.evaluate(() => document.activeElement?.closest('[data-api-target]')?.getAttribute('data-api-target') || '')
  )).toBe('vision')
})

test('settings shows compact connection status without admin maintenance tools', async ({ page }) => {
  await installMinimalBackend(page, {
    text: { has_api_key: false, status: 'missing', severity: 'error' },
  })
  await page.goto('/')

  await expect(page.getByTestId('release-readiness-banner')).toHaveCount(0)
  await page.locator('button[aria-label="Open settings"]').click()

  const readiness = page.getByTestId('settings-release-readiness')
  await expect(readiness).toContainText('Connection status')
  await expect(readiness).toContainText('API availability')
  await expect(readiness).toContainText('Blocked')
  await expect(page.locator('.kb-settings-drawer')).not.toContainText('Release readiness')
  await expect(readiness).not.toContainText('Data protection')
  await expect(readiness).not.toContainText('Restore review')
  await expect(readiness).not.toContainText('CORS origins')
  await expect(page.getByTestId('settings-admin-tools-summary')).toHaveCount(0)
  await expect(page.getByTestId('settings-release-readiness-details')).toHaveCount(0)
  await expect(page.getByTestId('settings-quality-collector-status')).toHaveCount(0)
  await page.locator('.kb-settings-advanced > summary').click()
  await expect(page.getByTestId('settings-auto-backup-switch')).toHaveCount(0)
})

test('internal release readiness banner opens the settings readiness panel', async ({ page }) => {
  await enableInternalSettingsTools(page)
  await installMinimalBackend(page, {
    text: { has_api_key: false, status: 'missing', severity: 'error' },
  })
  await page.goto('/')

  const banner = page.getByTestId('release-readiness-banner')
  await expect(banner).toContainText('Release readiness is blocked')
  await expect(banner).toContainText('1 blockers')

  await banner.getByRole('button', { name: 'Review' }).click()

  await expect(page.getByTestId('settings-release-readiness')).toContainText('Connection status')
})

test('restore review notice stays hidden from ordinary users', async ({ page }) => {
  await installMinimalBackend(page, { restoreReview: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const readiness = page.getByTestId('settings-release-readiness')
  await expect(readiness).not.toContainText('Restore review needed')
  await expect(readiness).not.toContainText('A restore was recorded recently')
  await expect(readiness.getByRole('button', { name: 'Confirm restarted and reviewed' })).toHaveCount(0)
  await expect(page.getByTestId('settings-admin-tools-summary')).toHaveCount(0)
})

test('internal settings can acknowledge a recent restore review', async ({ page }) => {
  await enableInternalSettingsTools(page)
  await installMinimalBackend(page, { restoreReview: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const readiness = page.getByTestId('settings-release-readiness')
  await expect(readiness).toContainText('Restore review needed')
  await readiness.getByRole('button', { name: 'Confirm restarted and reviewed' }).click()
  await page.getByRole('button', { name: 'OK' }).click()

  await expect(readiness).not.toContainText('Restore review needed')
  await expect(readiness).not.toContainText('Confirm restarted and reviewed')
})

test('settings can toggle automatic backup preference', async ({ page }) => {
  await enableInternalSettingsTools(page)
  await installMinimalBackend(page)
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  await page.locator('.kb-settings-advanced > summary').click()
  const toggle = page.getByTestId('settings-auto-backup-switch')
  await expect(toggle).toBeChecked()

  await toggle.click()
  await expect(toggle).not.toBeChecked()

  await toggle.click()
  await expect(toggle).toBeChecked()
})

test('settings can opt in to developer quality data sharing', async ({ page }) => {
  const backend = await installMinimalBackend(page)
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await expect(toggle).not.toBeChecked()
  await expect(page.locator('.kb-settings-drawer')).toContainText('Share anonymous quality data with the developer')
  await expect(page.getByTestId('settings-quality-collector-status')).toHaveCount(0)

  await toggle.click()
  await expect(toggle).toBeChecked()
  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)
  await expect(page.getByTestId('settings-quality-collector-status')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Test collector' })).toHaveCount(0)
  await expect.poll(() => backend.getQualityCollectorTests()).toBe(0)
})

test('settings keeps quality data sharing state after opt-in save even if reload fails', async ({ page }) => {
  const backend = await installMinimalBackend(page, { failSettingsReloadAfterQualityPatch: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await expect(toggle).not.toBeChecked()
  await toggle.click()

  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)
  await expect(toggle).toBeChecked()
  await expect(page.getByText('settings reload failed after quality sharing save')).toHaveCount(0)
  await expect(page.getByTestId('settings-quality-collector-status')).toHaveCount(0)
})

test('settings warns if opt-out cannot clear pending quality reports', async ({ page }) => {
  const backend = await installMinimalBackend(page, { failQualityDataCleanup: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await toggle.click()
  await expect(toggle).toBeChecked()
  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)

  await toggle.click()
  await expect(toggle).not.toBeChecked()
  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === false)).toBe(true)
  await expect(page.getByText('Quality data sharing is off, but clearing unsent remote reports failed: database locked')).toBeVisible()
})

test('settings rolls back lightweight preferences when saving fails', async ({ page }) => {
  const backend = await installMinimalBackend(page, { failAnswerDepthPatch: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-answer-depth-auto-switch')
  await expect(toggle).toBeChecked()

  await toggle.click()
  await expect.poll(() => backend.settingsPatches.some((patch) => patch.answer_depth_auto === false)).toBe(true)
  await expect(toggle).toBeChecked()
  await expect(page.getByText('500 Internal Server Error: preference save failed')).toBeVisible()
  await expect(page.getByText('{"detail":"preference save failed"}')).toHaveCount(0)
})

test('internal settings tools can inspect and flush quality collector status', async ({ page }) => {
  await enableInternalSettingsTools(page)
  const backend = await installMinimalBackend(page)
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await expect(toggle).not.toBeChecked()
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('Off')

  await toggle.click()
  await expect(toggle).toBeChecked()
  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('Ready')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('collector.example')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('2')

  await page.getByRole('button', { name: 'Test collector' }).click()
  await expect.poll(() => backend.getQualityCollectorTests()).toBe(1)

  await page.getByRole('button', { name: 'Send pending' }).click()
  await expect.poll(() => backend.getQualityCollectorFlushes()).toBe(1)
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('0')
})

test('settings warns when quality collector is not HTTPS', async ({ page }) => {
  await enableInternalSettingsTools(page)
  const backend = await installMinimalBackend(page, { qualityCollectorInsecure: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await toggle.click()

  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('Needs setup')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('collector.example')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('not using HTTPS')
  await expect(page.getByRole('button', { name: 'Test collector' })).toBeDisabled()
})

test('settings warns when quality collector points at localhost', async ({ page }) => {
  await enableInternalSettingsTools(page)
  const backend = await installMinimalBackend(page, { qualityCollectorLocal: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await toggle.click()

  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('Needs setup')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('127.0.0.1:9000')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('public HTTPS collector')
  await expect(page.getByRole('button', { name: 'Test collector' })).toBeDisabled()
})

test('settings warns when quality collector URL contains credentials', async ({ page }) => {
  await enableInternalSettingsTools(page)
  const backend = await installMinimalBackend(page, { qualityCollectorCredentials: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await toggle.click()

  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('Needs setup')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('credentials in the URL')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('KB_USER_ISSUES_REMOTE_TOKEN')
  await expect(page.getByRole('button', { name: 'Test collector' })).toBeDisabled()
})

test('settings warns when quality collector sender token is missing', async ({ page }) => {
  await enableInternalSettingsTools(page)
  const backend = await installMinimalBackend(page, { qualityCollectorMissingToken: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await toggle.click()

  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('Needs setup')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('KB_USER_ISSUES_REMOTE_TOKEN')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('collector.example')
  await expect(page.getByRole('button', { name: 'Test collector' })).toBeDisabled()
})

test('settings warns when quality collector URL has an invalid port', async ({ page }) => {
  await enableInternalSettingsTools(page)
  const backend = await installMinimalBackend(page, { qualityCollectorInvalidPort: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await toggle.click()

  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('Needs setup')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('collector.example:bad')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('not a valid HTTP(S) endpoint')
  await expect(page.getByRole('button', { name: 'Test collector' })).toBeDisabled()
})

test('settings reports quality collector status check failures separately', async ({ page }) => {
  await enableInternalSettingsTools(page)
  const backend = await installMinimalBackend(page, { qualityCollectorStatusFails: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await toggle.click()

  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('Check failed')
  await expect(page.getByTestId('settings-quality-collector-status')).toContainText('collector status failed')
  await expect(page.getByTestId('settings-quality-collector-status')).not.toContainText('Remote collection is disabled')
  await expect(page.getByRole('button', { name: 'Test collector' })).toBeDisabled()
})

test('settings keeps quality data sharing off when saving the opt-in fails', async ({ page }) => {
  const backend = await installMinimalBackend(page, { failQualityDataSharingPatch: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const toggle = page.getByTestId('settings-quality-data-sharing-switch')
  await expect(toggle).not.toBeChecked()
  await toggle.click()

  await expect.poll(() => backend.settingsPatches.some((patch) => patch.quality_data_sharing_enabled === true)).toBe(true)
  await expect(toggle).not.toBeChecked()
  await expect(page.getByTestId('settings-quality-collector-status')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Test collector' })).toHaveCount(0)
  await expect.poll(() => backend.getQualityCollectorTests()).toBe(0)
})

test('user-facing settings hides the access-token session controls by default', async ({ page }) => {
  test.skip(
    process.env.VITE_ENABLE_AUTH_GATE === '1'
      && process.env.VITE_PRIVATE_INSTANCE_AUTH === '1'
      && process.env.VITE_ALLOW_LOCAL_AUTH_GATE === '1',
    'this assertion covers the default user-facing build',
  )

  const backend = await installMinimalBackend(page, { authRequired: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const lockButton = page.getByTestId('settings-auth-lock-button')
  await expect(lockButton).toHaveCount(0)
  await expect(page.locator('.kb-settings-drawer')).not.toContainText('Access token session')
  await expect.poll(() => backend.getLogoutCalls()).toBe(0)
})

test('settings can lock the current access-token session when the auth gate is enabled', async ({ page }) => {
  test.skip(
    process.env.VITE_ENABLE_AUTH_GATE !== '1'
      || process.env.VITE_PRIVATE_INSTANCE_AUTH !== '1'
      || process.env.VITE_ALLOW_LOCAL_AUTH_GATE !== '1',
    'access-token session controls are hidden in the user-facing build',
  )

  const backend = await installMinimalBackend(page, { authRequired: true })
  await page.goto('/')

  await page.locator('button[aria-label="Open settings"]').click()

  const lockButton = page.getByTestId('settings-auth-lock-button')
  await expect(lockButton).toBeVisible()
  await expect(page.locator('.kb-settings-drawer')).toContainText('Access token session')

  await lockButton.click()
  await page.getByRole('button', { name: 'OK' }).click()

  await expect.poll(() => backend.getLogoutCalls()).toBe(1)
  await expect(page.locator('.kb-auth-gate')).toContainText('Access Token Required')
})
