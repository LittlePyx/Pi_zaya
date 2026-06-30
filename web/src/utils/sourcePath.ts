function decodeSourcePathText(value: string): string {
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

function looksLikeUrlSuffix(suffix: string, separator: '?' | '#'): boolean {
  const tail = String(suffix || '').trim()
  if (!tail || tail.includes('/') || tail.includes('\\')) return false
  const low = tail.toLowerCase()
  if (separator === '?') {
    return tail.includes('=') || ['download', 'reader', 'viewer', 'locate', 'selection'].includes(low)
  }
  if (low.startsWith('page=') || low.startsWith('p=')) return true
  return !tail.includes('.') && tail.length <= 80
}

function stripSourcePathUrlSuffix(value: string): string {
  const text = String(value || '')
  if (!text) return ''
  if (/^file:\/\//i.test(text)) {
    const queryAt = text.indexOf('?')
    const hashAt = text.indexOf('#')
    const candidates = [queryAt, hashAt].filter((idx) => idx >= 0)
    if (!candidates.length) return text
    return text.slice(0, Math.min(...candidates))
  }
  let cutAt = text.length
  for (const separator of ['?', '#'] as const) {
    const idx = text.indexOf(separator)
    if (idx >= 0 && looksLikeUrlSuffix(text.slice(idx + 1), separator)) {
      cutAt = Math.min(cutAt, idx)
    }
  }
  return text.slice(0, cutAt)
}

function unwrapFileSourceUrl(value: string): string {
  if (!/^file:\/\//i.test(value)) return value
  if (/^file:\/\/\//i.test(value)) return value.replace(/^file:\/\/\//i, '')
  return `//${value.replace(/^file:\/\//i, '')}`
}

function normalizePathSegments(value: string): string {
  const slashPath = value.replace(/\\/g, '/')
  const uncPrefix = slashPath.startsWith('//')
  const absolutePrefix = !uncPrefix && slashPath.startsWith('/')
  const parts: string[] = []
  for (const rawPart of slashPath.split('/')) {
    const part = rawPart.trim()
    if (!part || part === '.') continue
    if (part === '..') {
      const prev = parts[parts.length - 1]
      if (prev && prev !== '..' && !/^[A-Za-z]:$/.test(prev)) {
        parts.pop()
      } else {
        parts.push(part)
      }
      continue
    }
    parts.push(part)
  }
  let out = parts.join('/')
  if (uncPrefix && out) out = `//${out}`
  else if (absolutePrefix && out) out = `/${out}`
  out = out.replace(/^\/([A-Za-z]:)(\/|$)/, '$1$2')
  return out.replace(/\/$/, '')
}

export function cleanFileSourcePathInput(value: unknown): string {
  const raw = String(value || '')
    .trim()
    .replace(/\0/g, ' ')
  if (!raw) return ''
  return decodeSourcePathText(unwrapFileSourceUrl(stripSourcePathUrlSuffix(raw))).trim()
}

export function normalizeSourcePathForMatch(value: unknown): string {
  const clean = cleanFileSourcePathInput(value)
  if (!clean) return ''
  return normalizePathSegments(clean).toLowerCase()
}

export function basenameFromSourcePath(value: unknown): string {
  const clean = cleanFileSourcePathInput(value)
  if (!clean) return ''
  return clean.split(/[\\/]/).filter(Boolean).pop() || clean
}
