import { normalizeSourcePathForMatch } from '../../utils/sourcePath'
import type { LocateCandidate } from './reader/messageLocateCandidates'

export function sourceDocumentIdentityKey(input: string): string {
  const normalized = normalizeSourcePathForMatch(input)
  if (!normalized) return ''
  const parts = normalized.split('/').map((item) => item.trim()).filter(Boolean)
  const file = parts[parts.length - 1] || normalized
  const stem = file
    .replace(/\.en\.md$/i, '')
    .replace(/\.md$/i, '')
    .replace(/\.pdf$/i, '')
    .replace(/\s+/g, ' ')
    .trim()
  return stem || file
}

export function sourcePathsReferToSameDocument(left: string, right: string): boolean {
  const leftNorm = normalizeSourcePathForMatch(left)
  const rightNorm = normalizeSourcePathForMatch(right)
  if (!leftNorm || !rightNorm) return false
  if (leftNorm === rightNorm) return true
  const leftId = sourceDocumentIdentityKey(leftNorm)
  const rightId = sourceDocumentIdentityKey(rightNorm)
  return Boolean(leftId && rightId && leftId === rightId)
}

export function sourcePathLookupKeys(input: string): string[] {
  const exact = normalizeSourcePathForMatch(input)
  const identity = sourceDocumentIdentityKey(exact)
  return Array.from(new Set([exact, identity].filter(Boolean)))
}

export function lookupGuideCandidatesBySourcePath(
  map: Map<string, LocateCandidate[]>,
  sourcePath: string,
): LocateCandidate[] {
  for (const key of sourcePathLookupKeys(sourcePath)) {
    const hit = map.get(key)
    if (hit && hit.length > 0) return hit
  }
  return []
}
