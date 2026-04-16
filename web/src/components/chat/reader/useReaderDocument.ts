/* eslint-disable react-hooks/set-state-in-effect */

import { useEffect, useRef, useState } from 'react'
import { referencesApi, type ReaderDocAnchor, type ReaderDocBlock } from '../../../api/references'

interface UseReaderDocumentOptions {
  open: boolean
  sourcePath: string
  sourceName: string
  onBeforeLoad?: () => void
}

interface ReaderDocumentState {
  loading: boolean
  error: string
  markdown: string
  readerAnchors: ReaderDocAnchor[]
  readerBlocks: ReaderDocBlock[]
  resolvedName: string
}

export function useReaderDocument({
  open,
  sourcePath,
  sourceName,
  onBeforeLoad,
}: UseReaderDocumentOptions): ReaderDocumentState {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [markdown, setMarkdown] = useState('')
  const [readerAnchors, setReaderAnchors] = useState<ReaderDocAnchor[]>([])
  const [readerBlocks, setReaderBlocks] = useState<ReaderDocBlock[]>([])
  const [resolvedName, setResolvedName] = useState('')
  const beforeLoadRef = useRef<UseReaderDocumentOptions['onBeforeLoad']>(onBeforeLoad)

  useEffect(() => {
    beforeLoadRef.current = onBeforeLoad
  }, [onBeforeLoad])

  useEffect(() => {
    if (!open || !sourcePath) return
    let cancelled = false
    beforeLoadRef.current?.()
    setLoading(true)
    setError('')
    setMarkdown('')
    setReaderAnchors([])
    setReaderBlocks([])
    referencesApi.readerDoc(sourcePath)
      .then((res) => {
        if (cancelled) return
        setMarkdown(String(res.markdown || ''))
        setReaderAnchors(Array.isArray(res.anchors) ? res.anchors : [])
        setReaderBlocks(Array.isArray(res.blocks) ? res.blocks : [])
        setResolvedName(String(res.source_name || sourceName || '').trim())
      })
      .catch((err) => {
        if (cancelled) return
        setMarkdown('')
        setReaderAnchors([])
        setReaderBlocks([])
        setError(err instanceof Error ? err.message : 'Failed to load reader document')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, sourceName, sourcePath])

  return {
    loading,
    error,
    markdown,
    readerAnchors,
    readerBlocks,
    resolvedName,
  }
}
