/* eslint-disable react-hooks/set-state-in-effect */

import { useEffect, useRef, useState } from 'react'
import { referencesApi, type ReaderDocAnchor, type ReaderDocBlock, type ReaderDocResponse } from '../../../api/references'
import { normalizeCiteDetail, type CiteDetail } from '../citationState'

interface UseReaderDocumentOptions {
  open: boolean
  sourcePath: string
  sourceName: string
  documentOverride?: ReaderDocResponse | null
  onBeforeLoad?: () => void
}

interface ReaderDocumentState {
  loading: boolean
  error: string
  markdown: string
  readerAnchors: ReaderDocAnchor[]
  readerBlocks: ReaderDocBlock[]
  citeDetails: CiteDetail[]
  resolvedName: string
}

function normalizeReaderCiteDetails(value: unknown): CiteDetail[] {
  if (!Array.isArray(value)) return []
  const out: CiteDetail[] = []
  const seen = new Set<string>()
  for (const raw of value) {
    const detail = normalizeCiteDetail(raw)
    if (!detail) continue
    const key = String(detail.anchor || '').trim()
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push(detail)
  }
  return out
}

export function useReaderDocument({
  open,
  sourcePath,
  sourceName,
  documentOverride,
  onBeforeLoad,
}: UseReaderDocumentOptions): ReaderDocumentState {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [markdown, setMarkdown] = useState('')
  const [readerAnchors, setReaderAnchors] = useState<ReaderDocAnchor[]>([])
  const [readerBlocks, setReaderBlocks] = useState<ReaderDocBlock[]>([])
  const [citeDetails, setCiteDetails] = useState<CiteDetail[]>([])
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
    setCiteDetails([])
    if (documentOverride) {
      setMarkdown(String(documentOverride.markdown || ''))
      setReaderAnchors(Array.isArray(documentOverride.anchors) ? documentOverride.anchors : [])
      setReaderBlocks(Array.isArray(documentOverride.blocks) ? documentOverride.blocks : [])
      setCiteDetails(normalizeReaderCiteDetails(documentOverride.cite_details || documentOverride.reference_cite_details))
      setResolvedName(String(documentOverride.source_name || sourceName || '').trim())
      setLoading(false)
      return () => {
        cancelled = true
      }
    }
    referencesApi.readerDoc(sourcePath)
      .then((res) => {
        if (cancelled) return
        setMarkdown(String(res.markdown || ''))
        setReaderAnchors(Array.isArray(res.anchors) ? res.anchors : [])
        setReaderBlocks(Array.isArray(res.blocks) ? res.blocks : [])
        setCiteDetails(normalizeReaderCiteDetails(res.cite_details || res.reference_cite_details))
        setResolvedName(String(res.source_name || sourceName || '').trim())
      })
      .catch((err) => {
        if (cancelled) return
        setMarkdown('')
        setReaderAnchors([])
        setReaderBlocks([])
        setCiteDetails([])
        setError(err instanceof Error ? err.message : 'Failed to load reader document')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [documentOverride, open, sourceName, sourcePath])

  return {
    loading,
    error,
    markdown,
    readerAnchors,
    readerBlocks,
    citeDetails,
    resolvedName,
  }
}
