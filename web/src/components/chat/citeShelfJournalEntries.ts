import { readShelfStorage, writeShelfStorage } from './citeShelfStorage'

const knownKeys = new Set<string>()
const suffix = ':operation:'

export function storeShelfOperation(key: string, id: string, value: unknown): void {
  const entry = `${key}${suffix}${id}`
  knownKeys.add(entry)
  writeShelfStorage(entry, JSON.stringify(value))
}

export function readShelfOperations(key: string): unknown[] {
  const keys = new Set(knownKeys)
  for (const storageName of ['localStorage', 'sessionStorage'] as const) {
    try {
      const storage = window[storageName]
      for (let index = 0; index < storage.length; index += 1) {
        const entry = storage.key(index)
        if (entry?.startsWith(`${key}${suffix}`)) keys.add(entry)
      }
    } catch { /* Retain the in-memory fallback for this document. */ }
  }
  return [...keys].filter((entry) => entry.startsWith(`${key}${suffix}`)).flatMap((entry) => {
    try { const value = JSON.parse(readShelfStorage(entry) || 'null'); return value ? [value] : [] }
    catch { return [] }
  })
}

export function acknowledgeShelfOperations(key: string, ids: string[]): void {
  for (const id of ids) {
    const entry = `${key}${suffix}${id}`
    writeShelfStorage(entry, '')
    knownKeys.delete(entry)
  }
}
