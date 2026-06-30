import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import { createRequire } from 'node:module'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'

const DEFAULT_PORT = 4173
const LOCAL_PORT_SPAN = 1000
const MAX_PORT_ATTEMPTS = 120
const MAX_EPHEMERAL_ATTEMPTS = 20
const STALE_LOCK_MS = 10 * 60 * 1000

function hasServerOverride(env) {
  return Boolean(env.PW_BASE_URL || env.PW_PORT || env.PLAYWRIGHT_PORT)
}

function portLockDir() {
  const key = createHash('sha1').update(process.cwd()).digest('hex').slice(0, 12)
  return path.join(os.tmpdir(), 'kb-chat-playwright-ports', key)
}

async function removeStaleLock(file) {
  try {
    const stat = await fs.stat(file)
    if (Date.now() - stat.mtimeMs > STALE_LOCK_MS) {
      await fs.unlink(file)
    }
  } catch {
    // Missing or unreadable locks are handled by the next create attempt.
  }
}

async function lockPort(port) {
  const dir = portLockDir()
  const file = path.join(dir, `${port}.lock`)
  await fs.mkdir(dir, { recursive: true })
  await removeStaleLock(file)
  try {
    const handle = await fs.open(file, 'wx')
    await handle.writeFile(JSON.stringify({ pid: process.pid, createdAt: Date.now() }))
    await handle.close()
    return async () => {
      try {
        await fs.unlink(file)
      } catch {
        // Another process may have already cleaned up a stale lock.
      }
    }
  } catch (error) {
    if (error?.code === 'EEXIST') return null
    throw error
  }
}

function portIsFree(port, host = '127.0.0.1') {
  return new Promise(resolve => {
    const server = net.createServer()
    server.unref()
    server.once('error', () => resolve(false))
    server.listen({ host, port }, () => {
      server.close(() => resolve(true))
    })
  })
}

function getEphemeralPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port: 0 }, () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : null
      server.close(() => {
        if (port) resolve(port)
        else reject(new Error('Unable to allocate an ephemeral Playwright port'))
      })
    })
  })
}

async function reservePort() {
  for (let i = 0; i < MAX_PORT_ATTEMPTS; i += 1) {
    const port = DEFAULT_PORT + ((process.pid + i) % LOCAL_PORT_SPAN)
    const release = await lockPort(port)
    if (!release) continue
    if (await portIsFree(port)) return { port, release }
    await release()
  }

  for (let i = 0; i < MAX_EPHEMERAL_ATTEMPTS; i += 1) {
    const port = await getEphemeralPort()
    const release = await lockPort(port)
    if (!release) continue
    if (await portIsFree(port)) return { port, release }
    await release()
  }

  throw new Error('Unable to reserve a free Playwright dev-server port')
}

const reservation = !hasServerOverride(process.env) && !process.env.CI
  ? await reservePort()
  : null
if (reservation) process.env.PW_PORT = String(reservation.port)

const require = createRequire(import.meta.url)
const playwrightCli = require.resolve('@playwright/test/cli')
const child = spawn(process.execPath, [playwrightCli, 'test', ...process.argv.slice(2)], {
  env: process.env,
  stdio: 'inherit',
})

let exiting = false
async function exitWith(code) {
  if (exiting) return
  exiting = true
  if (reservation) await reservation.release()
  process.exit(code)
}

child.on('error', error => {
  console.error(error)
  void exitWith(1)
})

child.on('exit', code => {
  void exitWith(code ?? 1)
})

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, () => {
    child.kill(signal)
    void exitWith(signal === 'SIGINT' ? 130 : 143)
  })
}
