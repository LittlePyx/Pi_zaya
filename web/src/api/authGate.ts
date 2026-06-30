type ViteEnv = Record<string, unknown>

const VITE_ENV = ((import.meta as ImportMeta & { env?: ViteEnv }).env || {}) as ViteEnv

function envTruthy(value: unknown): boolean {
  return ['1', 'true', 'yes', 'on'].includes(String(value ?? '').trim().toLowerCase())
}

function isLocalDevEnv(env: ViteEnv): boolean {
  const mode = String(env.MODE ?? '').trim().toLowerCase()
  return mode === 'development' || envTruthy(env.DEV)
}

export function authGateBuildEnabled(env: ViteEnv = VITE_ENV): boolean {
  if (!envTruthy(env.VITE_ENABLE_AUTH_GATE)) return false
  if (!envTruthy(env.VITE_PRIVATE_INSTANCE_AUTH)) return false
  if (isLocalDevEnv(env) && !envTruthy(env.VITE_ALLOW_LOCAL_AUTH_GATE)) return false
  return true
}
