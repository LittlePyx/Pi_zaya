import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, Input, Select, message } from 'antd'
import type { CiteShelfItem } from './citationState'
import { citationDisplay, citationFormats, citeMetricSummary, summarySourceLabel } from './citationState'

interface Props {
  open: boolean
  items: CiteShelfItem[]
  focusedKey: string
  summaryLoadingKey: string
  onToggle: () => void
  onClear: () => void
  onSelect: (item: CiteShelfItem) => void
  onRemove: (key: string) => void
}

export function CiteShelf({ open, items, focusedKey, summaryLoadingKey, onToggle, onClear, onSelect, onRemove }: Props) {
  const [expandedSummaryKeys, setExpandedSummaryKeys] = useState<Record<string, boolean>>({})
  const [selectedKeys, setSelectedKeys] = useState<Record<string, boolean>>({})
  const [searchText, setSearchText] = useState('')
  const [sortKey, setSortKey] = useState<'recent' | 'cited' | 'year' | 'impact'>('recent')
  const [copyState, setCopyState] = useState<'idle' | 'gbt' | 'bibtex' | 'error'>('idle')
  const copyStateTimerRef = useRef<number | null>(null)

  const splitSummary = (text: string): string[] => {
    const normalized = String(text || '').replace(/\s+/g, ' ').trim()
    if (!normalized) return []
    const sentences = normalized
      .split(/(?<=[\u3002\uff01\uff1f!?；;])\s*/)
      .map((item) => item.trim())
      .filter(Boolean)
    if (sentences.length >= 2) return sentences.slice(0, 4)

    if (normalized.length <= 100) return [normalized]
    const chunks: string[] = []
    let current = ''
    const parts = normalized
      .split(/(?<=[\uff0c,])\s*/)
      .map((item) => item.trim())
      .filter(Boolean)
    for (const part of parts) {
      if ((current + part).length > 56 && current) {
        chunks.push(current.trim())
        current = part
      } else {
        current += part
      }
    }
    if (current.trim()) chunks.push(current.trim())
    return (chunks.length > 0 ? chunks : [normalized]).slice(0, 4)
  }

  const normalizeDoiLike = (value: string): string =>
    String(value || '')
      .trim()
      .toLowerCase()
      .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, '')
      .replace(/^[\s"'`([{<]+|[\s"'`)\]}>.,;:]+$/g, '')
      .trim()

  const normalizeTitle = (value: string): string =>
    String(value || '')
      .toLowerCase()
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()

  const paperIdentity = (item: CiteShelfItem): string => {
    const doiKey = normalizeDoiLike(item.doi || item.doiUrl)
    if (doiKey) return `doi:${doiKey}`
    const titleKey = normalizeTitle(item.title || item.main)
    const year = /^\d{4}$/.test(String(item.year || '').trim()) ? String(item.year).trim() : ''
    if (titleKey) return `title:${titleKey}|${year}`
    return `key:${item.key}`
  }

  const impactScore = (item: CiteShelfItem): number => {
    const ifValue = Number.parseFloat(String(item.journalIf || '').replace(/[^\d.]/g, ''))
    const ifScore = Number.isFinite(ifValue) ? ifValue : 0
    const quartile = String(item.journalQuartile || '').toUpperCase().trim()
    const quartileScore = quartile === 'Q1' ? 4 : quartile === 'Q2' ? 3 : quartile === 'Q3' ? 2 : quartile === 'Q4' ? 1 : 0
    const core = String(item.conferenceTier || '').toUpperCase().trim()
    const coreScore = core === 'A*' ? 4 : core === 'A' ? 3 : core === 'B' ? 2 : core === 'C' ? 1 : 0
    const ccf = String(item.conferenceCcf || '').toUpperCase().trim()
    const ccfScore = ccf === 'A' ? 3 : ccf === 'B' ? 2 : ccf === 'C' ? 1 : 0
    return ifScore * 10 + quartileScore + coreScore + ccfScore
  }

  const sourceTraceLabel = (item: CiteShelfItem): { label: string; debugTitle: string } => {
    const num = Number(item.num || 0)
    const label = Number.isFinite(num) && num > 0 ? `参考 #${num}` : ''
    const anchor = String(item.anchor || '').trim()
    const debugTitle = anchor ? `锚点: ${anchor}` : ''
    return { label, debugTitle }
  }

  const duplicateCountByIdentity = useMemo(() => {
    const counter: Record<string, number> = {}
    for (const item of items) {
      const key = paperIdentity(item)
      counter[key] = (counter[key] || 0) + 1
    }
    return counter
  }, [items])

  const visibleItems = useMemo(() => {
    const keyword = searchText.trim().toLowerCase()
    const matched = items.filter((item) => {
      if (!keyword) return true
      const text = [
        item.title,
        item.main,
        item.authors,
        item.venue,
        item.doi,
        item.doiUrl,
        item.sourceName,
      ]
        .map((part) => String(part || '').toLowerCase())
        .join(' ')
      return text.includes(keyword)
    })
    const sorted = [...matched]
    if (sortKey === 'cited') {
      sorted.sort((a, b) => (b.citationCount || 0) - (a.citationCount || 0))
    } else if (sortKey === 'year') {
      sorted.sort((a, b) => Number(String(b.year || 0)) - Number(String(a.year || 0)))
    } else if (sortKey === 'impact') {
      sorted.sort((a, b) => impactScore(b) - impactScore(a))
    }
    return sorted
  }, [impactScore, items, searchText, sortKey])

  const selectedCount = Object.values(selectedKeys).filter(Boolean).length
  const selectedItems = useMemo(
    () => items.filter((item) => Boolean(selectedKeys[item.key])),
    [items, selectedKeys],
  )

  const setTransientCopyState = (next: 'gbt' | 'bibtex' | 'error') => {
    setCopyState(next)
    if (copyStateTimerRef.current !== null) {
      window.clearTimeout(copyStateTimerRef.current)
      copyStateTimerRef.current = null
    }
    copyStateTimerRef.current = window.setTimeout(() => {
      setCopyState('idle')
      copyStateTimerRef.current = null
    }, 1800)
  }

  const writeClipboard = async (text: string) => {
    const payload = String(text || '').trim()
    if (!payload) return
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(payload)
      return
    }
    const el = document.createElement('textarea')
    el.value = payload
    el.setAttribute('readonly', 'true')
    el.style.position = 'fixed'
    el.style.left = '-9999px'
    document.body.appendChild(el)
    el.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(el)
    if (!ok) throw new Error('clipboard-copy-failed')
  }

  const copySelectedAs = async (kind: 'gbt' | 'bibtex') => {
    if (selectedItems.length <= 0) return
    const text = selectedItems.map((item) => citationFormats(item)[kind]).join('\n\n')
    try {
      await writeClipboard(text)
      setTransientCopyState(kind)
    } catch {
      setTransientCopyState('error')
    }
  }

  const nowStamp = () => {
    const d = new Date()
    const pad = (v: number) => String(v).padStart(2, '0')
    return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`
  }

  const downloadTextFile = (filename: string, text: string, mime = 'text/plain;charset=utf-8') => {
    const blob = new Blob([text], { type: mime })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const csvEscape = (value: unknown): string => {
    const text = String(value ?? '')
    if (!text) return ''
    if (!/[",\n]/.test(text)) return text
    return `"${text.replace(/"/g, '""')}"`
  }

  const exportSelectedAs = (kind: 'bib' | 'csv') => {
    if (selectedItems.length <= 0) return
    try {
      const base = `cite_shelf_selected_${nowStamp()}`
      if (kind === 'bib') {
        const bib = selectedItems.map((item) => citationFormats(item).bibtex).join('\n\n').trim()
        if (!bib) return
        downloadTextFile(`${base}.bib`, bib, 'application/x-bibtex;charset=utf-8')
        message.success(`已导出 BibTeX（${selectedItems.length} 条）`)
        return
      }
      const headers = [
        'title',
        'authors',
        'year',
        'venue',
        'doi',
        'source',
        'reference_num',
        'citation_count',
        'journal_if',
        'journal_quartile',
        'conference_tier',
        'conference_ccf',
        'summary',
      ]
      const rows = selectedItems.map((item) => ([
        item.title || item.main,
        item.authors,
        item.year,
        item.venue,
        item.doi || item.doiUrl,
        item.sourceName || item.sourcePath,
        item.num || '',
        item.citationCount || 0,
        item.journalIf,
        item.journalQuartile,
        item.conferenceTier,
        item.conferenceCcf,
        item.summaryLine,
      ].map((field) => csvEscape(field)).join(',')))
      const csv = `${headers.join(',')}\n${rows.join('\n')}`
      downloadTextFile(`${base}.csv`, csv, 'text/csv;charset=utf-8')
      message.success(`已导出 CSV（${selectedItems.length} 条）`)
    } catch {
      message.error('导出失败，请稍后重试')
    }
  }

  const toggleSelect = (key: string, checked: boolean) => {
    setSelectedKeys((prev) => {
      const next = { ...prev }
      if (checked) next[key] = true
      else delete next[key]
      return next
    })
  }
  const removeSelected = () => {
    const keys = Object.keys(selectedKeys).filter((key) => selectedKeys[key])
    for (const key of keys) onRemove(key)
    setSelectedKeys({})
  }
  const clearSelected = () => setSelectedKeys({})

  useEffect(() => {
    const validKeys = new Set(items.map((item) => item.key))
    setSelectedKeys((prev) => {
      const next: Record<string, boolean> = {}
      let changed = false
      for (const [key, checked] of Object.entries(prev)) {
        if (!checked) continue
        if (validKeys.has(key)) next[key] = true
        else changed = true
      }
      if (!changed && Object.keys(next).length !== Object.keys(prev).length) changed = true
      return changed ? next : prev
    })
    setExpandedSummaryKeys((prev) => {
      const next: Record<string, boolean> = {}
      let changed = false
      for (const [key, expanded] of Object.entries(prev)) {
        if (!expanded) continue
        if (validKeys.has(key)) next[key] = true
        else changed = true
      }
      if (!changed && Object.keys(next).length !== Object.keys(prev).length) changed = true
      return changed ? next : prev
    })
  }, [items])

  useEffect(() => {
    return () => {
      if (copyStateTimerRef.current !== null) {
        window.clearTimeout(copyStateTimerRef.current)
        copyStateTimerRef.current = null
      }
    }
  }, [])

  return (
    <>
      <button
        className={`fixed right-4 top-1/2 z-30 -translate-y-1/2 rounded-full border border-[var(--border)] bg-[var(--panel)] px-4 py-3 text-sm shadow-[0_10px_30px_rgba(15,23,42,0.12)] transition ${open ? 'pointer-events-none opacity-0' : ''}`}
        onClick={onToggle}
        type="button"
      >
        文献篮
      </button>
      <aside
        className={`fixed right-0 top-0 z-40 h-full w-[380px] max-w-[92vw] border-l border-[var(--border)] bg-[var(--panel)] shadow-[0_24px_64px_rgba(15,23,42,0.18)] transition-transform duration-300 ${open ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="flex h-full flex-col">
          <div className="kb-shelf-head border-b border-[var(--border)] px-4 py-4">
            <div className="kb-shelf-head-top">
              <div className="kb-shelf-head-meta">
                <div className="kb-shelf-title">文献篮</div>
                <div className="kb-shelf-count">
                  已收藏 {items.length} 条{searchText.trim() ? ` · 匹配 ${visibleItems.length} 条` : ''}
                </div>
              </div>
              <div className="kb-shelf-head-actions">
                <Button size="small" onClick={onClear} disabled={items.length === 0}>
                  清空
                </Button>
                <Button size="small" onClick={onToggle}>
                  关闭
                </Button>
              </div>
            </div>
            {selectedCount > 0 ? (
              <div className="kb-shelf-batch-row">
                <span className="kb-shelf-batch-count">已勾选 {selectedCount} 条</span>
                <Button size="small" onClick={removeSelected}>
                  批量移除
                </Button>
                <Button size="small" onClick={() => void copySelectedAs('gbt')}>
                  {copyState === 'gbt' ? '已复制 GB/T' : '复制 GB/T'}
                </Button>
                <Button size="small" onClick={() => void copySelectedAs('bibtex')}>
                  {copyState === 'bibtex' ? '已复制 BibTeX' : '复制 BibTeX'}
                </Button>
                <Button size="small" onClick={() => exportSelectedAs('bib')}>
                  导出 BibTeX
                </Button>
                <Button size="small" onClick={() => exportSelectedAs('csv')}>
                  导出 CSV
                </Button>
                <button type="button" className="kb-shelf-clear-select" onClick={clearSelected}>
                  清除勾选
                </button>
              </div>
            ) : null}
          </div>
          <div className="kb-shelf-scroll flex-1 overflow-y-auto px-4 py-4">
            {items.length > 0 ? (
              <>
                <div className="kb-shelf-toolbar">
                  <Input
                    allowClear
                    placeholder="搜索标题 / 作者 / DOI"
                    value={searchText}
                    onChange={(event) => setSearchText(event.target.value)}
                    className="kb-shelf-search"
                  />
                  <Select
                    value={sortKey}
                    onChange={(value) => setSortKey(value)}
                    className="kb-shelf-sort"
                    options={[
                      { value: 'recent', label: '最近加入' },
                      { value: 'cited', label: '被引次数' },
                      { value: 'year', label: '年份' },
                      { value: 'impact', label: 'IF/评级' },
                    ]}
                  />
                </div>
                {copyState === 'error' ? (
                  <div className="kb-shelf-copy-hint">复制失败，请检查剪贴板权限</div>
                ) : null}
              </>
            ) : null}
            {items.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-[var(--border)] px-4 py-5 text-sm text-black/45 dark:text-white/45">
                从文内参考弹窗点击“加入文献篮”，这里会保存标题、作者、来源、DOI 与文献指标。
              </div>
                ) : (
              <div className="space-y-3">
                {visibleItems.map((item) => {
                  const display = citationDisplay(item)
                  const subtitle = display.source
                  const duplicateCount = duplicateCountByIdentity[paperIdentity(item)] || 0
                  const trace = sourceTraceLabel(item)

                  return (
                    <div
                      key={item.key}
                      className={`rounded-2xl border px-4 py-3 transition ${
                        item.key === focusedKey
                          ? 'border-[var(--accent)] bg-[var(--msg-user-bg)]'
                          : 'border-[var(--border)] bg-[var(--panel)]'
                      }`}
                      onClick={() => onSelect(item)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault()
                          onSelect(item)
                        }
                      }}
                      role="button"
                      tabIndex={0}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <input
                          aria-label={`select-${item.key}`}
                          type="checkbox"
                          className="kb-shelf-check"
                          checked={Boolean(selectedKeys[item.key])}
                          onChange={(event) => {
                            event.stopPropagation()
                            toggleSelect(item.key, event.target.checked)
                          }}
                          onClick={(event) => event.stopPropagation()}
                        />
                        <div className="min-w-0 flex-1">
                          <div className="text-sm font-medium leading-6">{display.main}</div>
                          {display.authors ? (
                            <div className="mt-1 text-xs text-black/50 dark:text-white/50">{display.authors}</div>
                          ) : null}
                        </div>
                        <button
                          type="button"
                          className="text-xs text-black/35 transition hover:text-black/70 dark:text-white/35 dark:hover:text-white/70"
                          onClick={(event) => {
                            event.stopPropagation()
                            onRemove(item.key)
                          }}
                        >
                          移除
                        </button>
                      </div>
                      {subtitle ? (
                        <div className="mt-2 text-xs text-black/45 dark:text-white/45">{subtitle}</div>
                      ) : null}
                      {trace.label ? (
                        <div className="mt-2">
                          <span className="kb-shelf-origin" title={trace.debugTitle || undefined}>
                            {trace.label}
                          </span>
                        </div>
                      ) : null}
                      {duplicateCount > 1 ? (
                        <div className="mt-2">
                          <span className="kb-shelf-dup">可能重复 ×{duplicateCount}</span>
                        </div>
                      ) : null}
                      {citeMetricSummary(item).length > 0 ? (
                        <div className="mt-2 flex flex-wrap gap-1.5">
                          {citeMetricSummary(item).map((metric) => (
                            <span key={metric} className="kb-shelf-metric">
                              {metric}
                            </span>
                          ))}
                        </div>
                      ) : null}
                      <div className="mt-2 text-xs">
                        {item.doiUrl ? (
                          <a className="text-[var(--accent)]" href={item.doiUrl} rel="noreferrer" target="_blank">
                            {item.doi || item.doiUrl}
                          </a>
                        ) : (
                          <span className="text-black/35 dark:text-white/35">暂无 DOI 链接</span>
                        )}
                      </div>
                      {item.key === focusedKey ? (
                        <div className="kb-shelf-summary">
                          {summaryLoadingKey === item.key ? (
                            <div className="kb-shelf-summary-text">正在生成学术概括...</div>
                          ) : item.summaryLine ? (
                            <>
                              <div className="kb-shelf-summary-meta">
                                <span className="kb-shelf-summary-head">学术概括</span>
                                <span className="kb-shelf-summary-source">{summarySourceLabel(item.summarySource)}</span>
                              </div>
                              {(() => {
                                const lines = splitSummary(item.summaryLine)
                                const expanded = Boolean(expandedSummaryKeys[item.key])
                                const visibleLines = expanded ? lines : lines.slice(0, 2)
                                const canExpand = lines.length > 2
                                return (
                                  <>
                                    <ol className="kb-shelf-summary-list">
                                      {visibleLines.map((line) => (
                                        <li key={line} className="kb-shelf-summary-text">{line}</li>
                                      ))}
                                    </ol>
                                    {canExpand ? (
                                      <button
                                        type="button"
                                        className="kb-shelf-summary-toggle"
                                        onClick={(event) => {
                                          event.stopPropagation()
                                          setExpandedSummaryKeys((prev) => ({ ...prev, [item.key]: !expanded }))
                                        }}
                                      >
                                        {expanded ? '收起概括' : `展开剩余 ${lines.length - visibleLines.length} 条`}
                                      </button>
                                    ) : null}
                                  </>
                                )
                              })()}
                            </>
                          ) : (
                            <div className="kb-shelf-summary-empty">暂无可用学术概括</div>
                          )}
                        </div>
                      ) : null}
                    </div>
                  )
                })}
                {visibleItems.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-[var(--border)] px-3 py-4 text-xs text-black/45 dark:text-white/45">
                    未匹配到文献，请调整搜索词或排序方式。
                  </div>
                ) : null}
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  )
}
