/* eslint-disable react-hooks/set-state-in-effect */

import { useCallback, useEffect, useState } from 'react'

export type ReaderCandidatePickerChangeSource = 'system' | 'manual'

export interface UseReaderCandidatePickerStateOptions {
  locateRequestId: number
  open: boolean
  sourcePath: string
}

export interface ReaderCandidatePickerState {
  activeAltIndex: number
  altChangeSource: ReaderCandidatePickerChangeSource
  candidatePickerExpanded: boolean
  setActiveAltIndex: (idx: number, source?: ReaderCandidatePickerChangeSource) => void
  setCandidatePickerExpanded: (expanded: boolean) => void
  toggleCandidatePicker: () => void
}

export interface UseReaderCandidatePickerSyncOptions {
  payloadKey: unknown
  requestedAltIndex: number
  setActiveAltIndex: ReaderCandidatePickerState['setActiveAltIndex']
  setCandidatePickerExpanded: ReaderCandidatePickerState['setCandidatePickerExpanded']
  shouldAutoExpandCandidatePicker: boolean
}

export function useReaderCandidatePickerState({
  locateRequestId,
  open,
  sourcePath,
}: UseReaderCandidatePickerStateOptions): ReaderCandidatePickerState {
  const [activeAltIndex, setActiveAltIndexState] = useState(0)
  const [altChangeSource, setAltChangeSource] = useState<ReaderCandidatePickerChangeSource>('system')
  const [candidatePickerExpanded, updateCandidatePickerExpanded] = useState(false)

  const setActiveAltIndex = useCallback((
    idx: number,
    source: ReaderCandidatePickerChangeSource = 'system',
  ) => {
    setAltChangeSource(source)
    setActiveAltIndexState(idx)
  }, [])

  const setCandidatePickerExpanded = useCallback((expanded: boolean) => {
    updateCandidatePickerExpanded(expanded)
  }, [])

  const toggleCandidatePicker = useCallback(() => {
    updateCandidatePickerExpanded((prev) => !prev)
  }, [])

  useEffect(() => {
    updateCandidatePickerExpanded(false)
  }, [open, locateRequestId, sourcePath])

  return {
    activeAltIndex,
    altChangeSource,
    candidatePickerExpanded,
    setActiveAltIndex,
    setCandidatePickerExpanded,
    toggleCandidatePicker,
  }
}

export function useReaderCandidatePickerSync({
  payloadKey,
  requestedAltIndex,
  setActiveAltIndex,
  setCandidatePickerExpanded,
  shouldAutoExpandCandidatePicker,
}: UseReaderCandidatePickerSyncOptions) {
  useEffect(() => {
    setActiveAltIndex(requestedAltIndex, 'system')
  }, [payloadKey, requestedAltIndex, setActiveAltIndex])

  useEffect(() => {
    if (!shouldAutoExpandCandidatePicker) return
    setCandidatePickerExpanded(true)
  }, [setCandidatePickerExpanded, shouldAutoExpandCandidatePicker])
}
