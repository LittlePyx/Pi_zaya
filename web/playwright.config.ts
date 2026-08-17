import { defineConfig, devices } from '@playwright/test'

const DEFAULT_PORT = 4173
const DEFAULT_HOST = '127.0.0.1'
const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8005'

function envTruthy(value: string | undefined) {
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').trim().toLowerCase())
}

function parsePort(value: string | undefined) {
  const port = Number(value || '')
  return Number.isInteger(port) && port > 0 && port < 65_536 ? port : null
}

function parseBaseUrlPort(value: string | undefined) {
  if (!value) return null
  try {
    return parsePort(new URL(value).port)
  } catch {
    return null
  }
}

function parseBaseUrlHost(value: string | undefined) {
  if (!value) return null
  try {
    return new URL(value).hostname || null
  } catch {
    return null
  }
}

function shouldUseExternalServer() {
  const raw = process.env.PW_EXTERNAL_SERVER
  if (raw && raw.trim()) return envTruthy(raw)
  return Boolean(process.env.PW_BASE_URL)
}

const externalServer = shouldUseExternalServer()
const host = process.env.PW_HOST || parseBaseUrlHost(process.env.PW_BASE_URL) || DEFAULT_HOST
const port = parsePort(process.env.PW_PORT)
  || parsePort(process.env.PLAYWRIGHT_PORT)
  || parseBaseUrlPort(process.env.PW_BASE_URL)
  || DEFAULT_PORT
const baseURL = process.env.PW_BASE_URL || `http://${host}:${port}`
const configuredWorkers = Number(process.env.PW_WORKERS || '')
const workers = Number.isFinite(configuredWorkers) && configuredWorkers > 0
  ? Math.floor(configuredWorkers)
  : 4

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  expect: {
    timeout: 8_000,
  },
  fullyParallel: true,
  workers,
  use: {
    baseURL,
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: externalServer
    ? undefined
    : {
        command: `npm run dev -- --host ${host} --port ${port} --strictPort`,
        url: baseURL,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
        env: {
          ...process.env,
          VITE_BACKEND_URL: process.env.VITE_BACKEND_URL || DEFAULT_BACKEND_URL,
          VITE_ENABLE_INTERNAL_DEBUG: process.env.VITE_ENABLE_INTERNAL_DEBUG || '1',
          VITE_ENABLE_INTERNAL_ROUTES: process.env.VITE_ENABLE_INTERNAL_ROUTES || '1',
          VITE_SHOW_USER_QUALITY_DIAGNOSTICS: process.env.VITE_SHOW_USER_QUALITY_DIAGNOSTICS || '0',
        },
      },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
