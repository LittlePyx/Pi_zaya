import { createElement } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import {
  handleReaderHighlightUndoShortcut,
  isEditableUndoTarget,
  isReaderHighlightUndoShortcutEvent,
  useReaderHighlightUndoShortcut,
  type ReaderHighlightUndoShortcutMessenger,
} from '../components/chat/useReaderHighlightUndoShortcut'

export interface ReaderHighlightUndoShortcutSmokeResult {
  disabledUndoCount: number
  editableIgnored: boolean
  handlerHandled: boolean
  hookDefaultPrevented: boolean
  hookUndoCount: number
  inputIsEditable: boolean
  missedEmptyUndo: boolean
  missedShiftUndo: boolean
  normalIsEditable: boolean
  pureEventCounts: {
    prevented: number
    stopped: number
  }
  renderedText: string
  shortcutDetection: {
    ctrl: boolean
    meta: boolean
    ordinary: boolean
  }
  successMessages: string[]
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve())
  })
}

export async function runReaderHighlightUndoShortcutSmoke(): Promise<ReaderHighlightUndoShortcutSmokeResult> {
  const host = document.createElement('div')
  const input = document.createElement('input')
  const normal = document.createElement('button')
  document.body.append(host, input, normal)

  const successMessages: string[] = []
  const messenger: ReaderHighlightUndoShortcutMessenger = {
    success: (content) => {
      successMessages.push(String(content))
    },
  }
  let prevented = 0
  let stopped = 0
  const handlerHandled = handleReaderHighlightUndoShortcut({
    key: 'z',
    ctrlKey: true,
    metaKey: false,
    shiftKey: false,
    target: normal,
    preventDefault: () => {
      prevented += 1
    },
    stopPropagation: () => {
      stopped += 1
    },
  }, {
    messenger,
    onUndo: () => true,
    successLabel: 'Undo complete',
  })
  const missedEmptyUndo = handleReaderHighlightUndoShortcut({
    key: 'z',
    ctrlKey: true,
    metaKey: false,
    shiftKey: false,
    target: normal,
    preventDefault: () => {
      prevented += 1
    },
    stopPropagation: () => {
      stopped += 1
    },
  }, {
    messenger,
    onUndo: () => false,
    successLabel: 'Undo complete',
  })
  const missedShiftUndo = handleReaderHighlightUndoShortcut({
    key: 'z',
    ctrlKey: true,
    metaKey: false,
    shiftKey: true,
    target: normal,
    preventDefault: () => {
      prevented += 1
    },
    stopPropagation: () => {
      stopped += 1
    },
  }, {
    messenger,
    onUndo: () => true,
    successLabel: 'Undo complete',
  })
  const editableIgnored = handleReaderHighlightUndoShortcut({
    key: 'z',
    ctrlKey: true,
    metaKey: false,
    shiftKey: false,
    target: input,
    preventDefault: () => {
      prevented += 1
    },
    stopPropagation: () => {
      stopped += 1
    },
  }, {
    messenger,
    onUndo: () => true,
    successLabel: 'Undo complete',
  })
  const shortcutDetection = {
    ctrl: isReaderHighlightUndoShortcutEvent({
      key: 'z',
      ctrlKey: true,
      target: normal,
      preventDefault: () => {},
      stopPropagation: () => {},
    }),
    meta: isReaderHighlightUndoShortcutEvent({
      key: 'Z',
      metaKey: true,
      target: normal,
      preventDefault: () => {},
      stopPropagation: () => {},
    }),
    ordinary: isReaderHighlightUndoShortcutEvent({
      key: 'z',
      target: normal,
      preventDefault: () => {},
      stopPropagation: () => {},
    }),
  }

  const root = createRoot(host)
  let hookUndoCount = 0
  let disabledUndoCount = 0
  function EnabledHarness() {
    useReaderHighlightUndoShortcut({
      enabled: true,
      messenger,
      onUndo: () => {
        hookUndoCount += 1
        return true
      },
      successLabel: 'Hook undone',
    })
    return createElement('div', { id: 'reader-highlight-undo-shortcut-smoke' }, 'enabled')
  }
  function DisabledHarness() {
    useReaderHighlightUndoShortcut({
      enabled: false,
      messenger,
      onUndo: () => {
        disabledUndoCount += 1
        return true
      },
      successLabel: 'Disabled undone',
    })
    return createElement('div', { id: 'reader-highlight-undo-shortcut-smoke' }, 'disabled')
  }

  flushSync(() => {
    root.render(createElement(EnabledHarness))
  })
  const hookEvent = new KeyboardEvent('keydown', {
    bubbles: true,
    cancelable: true,
    ctrlKey: true,
    key: 'z',
  })
  window.dispatchEvent(hookEvent)
  await nextFrame()
  const hookDefaultPrevented = hookEvent.defaultPrevented

  flushSync(() => {
    root.render(createElement(DisabledHarness))
  })
  window.dispatchEvent(new KeyboardEvent('keydown', {
    bubbles: true,
    cancelable: true,
    ctrlKey: true,
    key: 'z',
  }))
  await nextFrame()

  const renderedText = host.textContent || ''
  root.unmount()
  host.remove()
  input.remove()
  normal.remove()

  return {
    disabledUndoCount,
    editableIgnored,
    handlerHandled,
    hookDefaultPrevented,
    hookUndoCount,
    inputIsEditable: isEditableUndoTarget(input),
    missedEmptyUndo,
    missedShiftUndo,
    normalIsEditable: isEditableUndoTarget(normal),
    pureEventCounts: {
      prevented,
      stopped,
    },
    renderedText,
    shortcutDetection,
    successMessages,
  }
}
