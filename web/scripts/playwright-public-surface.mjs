process.env.PW_PUBLIC_SURFACE = '1'
process.env.PW_WORKERS = process.env.PW_WORKERS || '1'
process.env.VITE_ENABLE_INTERNAL_DEBUG = '0'
process.env.VITE_ENABLE_INTERNAL_ROUTES = '0'
process.env.VITE_SHOW_USER_QUALITY_DIAGNOSTICS = '0'
process.env.VITE_SHOW_INTERNAL_SETTINGS = '0'
process.env.VITE_ENABLE_AUTH_GATE = '0'
process.env.VITE_ALLOW_LOCAL_AUTH_GATE = '0'
process.env.VITE_PRIVATE_INSTANCE_AUTH = '0'
process.env.VITE_ENABLE_EVIDENCE_MATRIX_WORKSPACE = '0'

const requestedArgs = process.argv.slice(2)
process.argv = [
  process.argv[0],
  process.argv[1],
  'public-surface.spec.ts',
  'library-quality-hidden.spec.ts',
  ...requestedArgs,
]

await import('./playwright-with-port.mjs')
