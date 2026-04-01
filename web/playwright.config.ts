import { defineConfig, devices } from '@playwright/test'

const DEFAULT_PORT = 4173
const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8005'
const baseURL = process.env.PW_BASE_URL || `http://127.0.0.1:${DEFAULT_PORT}`
const externalServer = process.env.PW_EXTERNAL_SERVER === '1'

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  expect: {
    timeout: 8_000,
  },
  fullyParallel: true,
  use: {
    baseURL,
    trace: 'on-first-retry',
  },
  webServer: externalServer
    ? undefined
    : {
        command: `npm run dev -- --host 127.0.0.1 --port ${DEFAULT_PORT} --strictPort`,
        port: DEFAULT_PORT,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
        env: {
          ...process.env,
          VITE_BACKEND_URL: process.env.VITE_BACKEND_URL || DEFAULT_BACKEND_URL,
        },
      },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
