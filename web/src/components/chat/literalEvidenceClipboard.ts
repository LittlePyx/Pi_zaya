import type { ClipboardEvent } from 'react'

export function copyLiteralEvidenceSelection(event: ClipboardEvent<HTMLQuoteElement>) {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount !== 1) return
  if (!event.currentTarget.contains(selection.getRangeAt(0).commonAncestorContainer)) return
  const text = selection.toString()
  if (!text) return
  event.clipboardData.setData('text/plain', text)
  event.preventDefault()
}
