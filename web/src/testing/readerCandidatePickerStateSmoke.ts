import { createElement, useLayoutEffect, useState } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import {
  useReaderCandidatePickerState,
  useReaderCandidatePickerSync,
  type ReaderCandidatePickerState,
} from '../components/chat/useReaderCandidatePickerState'

export interface ReaderCandidatePickerStateSnapshot {
  activeAltIndex: number
  altChangeSource: string
  candidatePickerExpanded: boolean
}

export interface ReaderCandidatePickerStateSmokeResult {
  afterAutoExpand: ReaderCandidatePickerStateSnapshot
  afterManualSelect: ReaderCandidatePickerStateSnapshot
  afterPayloadReset: ReaderCandidatePickerStateSnapshot
  afterRequestedSync: ReaderCandidatePickerStateSnapshot
  renderedText: string
}

interface HarnessProps {
  locateRequestId: number
  payloadKey: string
  requestedAltIndex: number
  shouldAutoExpandCandidatePicker: boolean
  sourcePath: string
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve())
  })
}

function snapshot(controller: ReaderCandidatePickerState): ReaderCandidatePickerStateSnapshot {
  return {
    activeAltIndex: controller.activeAltIndex,
    altChangeSource: controller.altChangeSource,
    candidatePickerExpanded: controller.candidatePickerExpanded,
  }
}

async function waitForController(
  readController: () => ReaderCandidatePickerState,
  predicate: (state: ReaderCandidatePickerStateSnapshot) => boolean,
): Promise<ReaderCandidatePickerStateSnapshot> {
  for (let i = 0; i < 8; i += 1) {
    await nextFrame()
    const current = snapshot(readController())
    if (predicate(current)) return current
  }
  return snapshot(readController())
}

export async function runReaderCandidatePickerStateSmoke(): Promise<ReaderCandidatePickerStateSmokeResult> {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  let controller: ReaderCandidatePickerState | null = null
  let setProps: ((next: HarnessProps) => void) | null = null

  const readController = () => {
    if (!controller) throw new Error('reader candidate picker state smoke did not mount')
    return controller
  }
  const readSetProps = () => {
    if (!setProps) throw new Error('reader candidate picker state props setter did not mount')
    return setProps
  }

  function Harness() {
    const [props, updateProps] = useState<HarnessProps>({
      locateRequestId: 1,
      payloadKey: 'payload-a',
      requestedAltIndex: 2,
      shouldAutoExpandCandidatePicker: false,
      sourcePath: '/tmp/a.md',
    })
    const picker = useReaderCandidatePickerState({
      locateRequestId: props.locateRequestId,
      open: true,
      sourcePath: props.sourcePath,
    })
    useReaderCandidatePickerSync({
      payloadKey: props.payloadKey,
      requestedAltIndex: props.requestedAltIndex,
      setActiveAltIndex: picker.setActiveAltIndex,
      setCandidatePickerExpanded: picker.setCandidatePickerExpanded,
      shouldAutoExpandCandidatePicker: props.shouldAutoExpandCandidatePicker,
    })
    useLayoutEffect(() => {
      controller = picker
      setProps = updateProps
    }, [picker])
    return createElement(
      'div',
      { id: 'reader-candidate-picker-smoke' },
      `${picker.activeAltIndex}|${picker.altChangeSource}|${picker.candidatePickerExpanded}`,
    )
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  const afterRequestedSync = await waitForController(
    readController,
    (state) => state.activeAltIndex === 2 && state.altChangeSource === 'system',
  )

  flushSync(() => {
    readController().toggleCandidatePicker()
    readController().setActiveAltIndex(1, 'manual')
  })
  const afterManualSelect = snapshot(readController())

  flushSync(() => {
    readSetProps()({
      locateRequestId: 2,
      payloadKey: 'payload-b',
      requestedAltIndex: 0,
      shouldAutoExpandCandidatePicker: false,
      sourcePath: '/tmp/b.md',
    })
  })
  const afterPayloadReset = await waitForController(
    readController,
    (state) => (
      state.activeAltIndex === 0
      && state.altChangeSource === 'system'
      && !state.candidatePickerExpanded
    ),
  )

  flushSync(() => {
    readSetProps()({
      locateRequestId: 2,
      payloadKey: 'payload-b',
      requestedAltIndex: 0,
      shouldAutoExpandCandidatePicker: true,
      sourcePath: '/tmp/b.md',
    })
  })
  const afterAutoExpand = await waitForController(
    readController,
    (state) => state.candidatePickerExpanded,
  )

  const renderedText = host.textContent || ''
  root.unmount()
  host.remove()

  return {
    afterAutoExpand,
    afterManualSelect,
    afterPayloadReset,
    afterRequestedSync,
    renderedText,
  }
}
