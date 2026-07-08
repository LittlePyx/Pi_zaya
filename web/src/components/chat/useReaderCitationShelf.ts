import { useCallback, useState } from 'react'

import {
  toShelfItem,
  type CiteDetail,
} from './citationState'

export type ReaderCitationShelfAddHandler = (detail: CiteDetail) => void

export interface UseReaderCitationShelfOptions {
  onAddCitationToShelf?: ReaderCitationShelfAddHandler
}

export interface ReaderCitationShelfController {
  addCitationToShelf: (detail: CiteDetail) => void
  hasCitation: (detail: CiteDetail | null | undefined) => boolean
  shelfKeys: ReadonlySet<string>
}

export function readerCitationShelfKey(detail: CiteDetail | null | undefined): string {
  if (!detail) return ''
  return toShelfItem(detail).key
}

export function readerCitationShelfHas(
  shelfKeys: ReadonlySet<string>,
  detail: CiteDetail | null | undefined,
): boolean {
  const key = readerCitationShelfKey(detail)
  return Boolean(key && shelfKeys.has(key))
}

export function addReaderCitationShelfKey(
  shelfKeys: ReadonlySet<string>,
  detail: CiteDetail,
): Set<string> {
  const key = readerCitationShelfKey(detail)
  const next = new Set(shelfKeys)
  if (key) next.add(key)
  return next
}

export function useReaderCitationShelf(
  options: UseReaderCitationShelfOptions = {},
): ReaderCitationShelfController {
  const { onAddCitationToShelf } = options
  const [shelfKeys, setShelfKeys] = useState<Set<string>>(() => new Set())

  const addCitationToShelf = useCallback((detail: CiteDetail) => {
    onAddCitationToShelf?.(detail)
    setShelfKeys((current) => addReaderCitationShelfKey(current, detail))
  }, [onAddCitationToShelf])

  const hasCitation = useCallback((detail: CiteDetail | null | undefined) => (
    readerCitationShelfHas(shelfKeys, detail)
  ), [shelfKeys])

  return {
    addCitationToShelf,
    hasCitation,
    shelfKeys,
  }
}
