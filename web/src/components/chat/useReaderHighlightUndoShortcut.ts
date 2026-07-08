import { useEffect, type ReactNode } from 'react'
import { message } from 'antd'

export interface ReaderHighlightUndoShortcutMessenger {
  success: (content: ReactNode) => void
}

export interface ReaderHighlightUndoKeyboardEvent {
  key?: string
  ctrlKey?: boolean
  metaKey?: boolean
  shiftKey?: boolean
  target: EventTarget | null
  preventDefault: () => void
  stopPropagation: () => void
}

export interface HandleReaderHighlightUndoShortcutOptions {
  messenger?: ReaderHighlightUndoShortcutMessenger
  onUndo: () => boolean
  successLabel: string
}

export interface UseReaderHighlightUndoShortcutOptions extends HandleReaderHighlightUndoShortcutOptions {
  enabled: boolean
}

const defaultUndoShortcutMessenger: ReaderHighlightUndoShortcutMessenger = {
  success: (content) => {
    message.success(content)
  },
}

export function isEditableUndoTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  return Boolean(target.closest('input, textarea, select, [contenteditable="true"], .ant-input'))
}

export function isReaderHighlightUndoShortcutEvent(event: ReaderHighlightUndoKeyboardEvent): boolean {
  const key = String(event.key || '').toLowerCase()
  return (Boolean(event.ctrlKey) || Boolean(event.metaKey))
    && !event.shiftKey
    && key === 'z'
    && !isEditableUndoTarget(event.target)
}

export function handleReaderHighlightUndoShortcut(
  event: ReaderHighlightUndoKeyboardEvent,
  {
    messenger = defaultUndoShortcutMessenger,
    onUndo,
    successLabel,
  }: HandleReaderHighlightUndoShortcutOptions,
): boolean {
  if (!isReaderHighlightUndoShortcutEvent(event)) return false
  if (!onUndo()) return false
  event.preventDefault()
  event.stopPropagation()
  messenger.success(successLabel || 'Undone')
  return true
}

export function useReaderHighlightUndoShortcut({
  enabled,
  messenger,
  onUndo,
  successLabel,
}: UseReaderHighlightUndoShortcutOptions) {
  useEffect(() => {
    if (!enabled) return undefined
    const handleKeyDown = (event: KeyboardEvent) => {
      handleReaderHighlightUndoShortcut(event, {
        messenger,
        onUndo,
        successLabel,
      })
    }
    window.addEventListener('keydown', handleKeyDown, true)
    return () => {
      window.removeEventListener('keydown', handleKeyDown, true)
    }
  }, [enabled, messenger, onUndo, successLabel])
}
