import { useCallback, useRef } from 'react'

export type LibraryQualityOperationToken = {
  id: number
  key: string
  scope: string
}

type UseLibraryQualityOperationGuardParams = {
  scope: string
  onBegin: () => void
}

export function useLibraryQualityOperationGuard({
  scope,
  onBegin,
}: UseLibraryQualityOperationGuardParams) {
  const qualityOperationSeqRef = useRef(0)
  const activeQualityOperationRef = useRef<LibraryQualityOperationToken | null>(null)

  const beginQualityOperation = useCallback((key: string): LibraryQualityOperationToken => {
    const token = {
      id: qualityOperationSeqRef.current + 1,
      key,
      scope,
    }
    qualityOperationSeqRef.current = token.id
    activeQualityOperationRef.current = token
    onBegin()
    return token
  }, [onBegin, scope])

  const qualityOperationIsCurrent = useCallback((token?: LibraryQualityOperationToken | null): boolean => {
    if (!token) return true
    const active = activeQualityOperationRef.current
    return Boolean(active && active.id === token.id && active.key === token.key && active.scope === token.scope && scope === token.scope)
  }, [scope])

  const qualityOperationIsActive = useCallback((token?: LibraryQualityOperationToken | null): boolean => {
    if (!token) return true
    const active = activeQualityOperationRef.current
    return Boolean(active && active.id === token.id && active.key === token.key && active.scope === token.scope)
  }, [])

  const clearQualityOperation = useCallback((token?: LibraryQualityOperationToken | null) => {
    if (!token) {
      activeQualityOperationRef.current = null
      return
    }
    const active = activeQualityOperationRef.current
    if (active && active.id === token.id && active.key === token.key) {
      activeQualityOperationRef.current = null
    }
  }, [])

  return {
    beginQualityOperation,
    qualityOperationIsCurrent,
    qualityOperationIsActive,
    clearQualityOperation,
  }
}
