import { Button } from 'antd'
import type { CiteShelfItem } from './citationState'
import { citationDisplay, citeMetricSummary } from './citationState'

interface Props {
  open: boolean
  items: CiteShelfItem[]
  focusedKey: string
  onToggle: () => void
  onClear: () => void
  onRemove: (key: string) => void
}

export function CiteShelf({ open, items, focusedKey, onToggle, onClear, onRemove }: Props) {
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
          <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-4">
            <div>
              <div className="text-base font-semibold">文献篮</div>
              <div className="text-xs text-black/45 dark:text-white/45">
                已收藏 {items.length} 条
              </div>
            </div>
            <div className="flex gap-2">
              <Button size="small" onClick={onClear} disabled={items.length === 0}>
                清空
              </Button>
              <Button size="small" onClick={onToggle}>
                关闭
              </Button>
            </div>
          </div>
          <div className="kb-shelf-scroll flex-1 overflow-y-auto px-4 py-4">
            {items.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-[var(--border)] px-4 py-5 text-sm text-black/45 dark:text-white/45">
                从文内参考弹窗点击“加入文献篮”，这里会保存题名、作者、来源、DOI 和文献指标。
              </div>
                ) : (
              <div className="space-y-3">
                {items.map((item) => {
                  const display = citationDisplay(item)
                  const subtitle = display.source

                  return (
                    <div
                      key={item.key}
                      className={`rounded-2xl border px-4 py-3 transition ${
                        item.key === focusedKey
                          ? 'border-[var(--accent)] bg-[var(--msg-user-bg)]'
                          : 'border-[var(--border)] bg-[var(--panel)]'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <div className="text-sm font-medium leading-6">{display.main}</div>
                          {display.authors ? (
                            <div className="mt-1 text-xs text-black/50 dark:text-white/50">{display.authors}</div>
                          ) : null}
                        </div>
                        <button
                          type="button"
                          className="text-xs text-black/35 transition hover:text-black/70 dark:text-white/35 dark:hover:text-white/70"
                          onClick={() => onRemove(item.key)}
                        >
                          移除
                        </button>
                      </div>
                      {subtitle ? (
                        <div className="mt-2 text-xs text-black/45 dark:text-white/45">{subtitle}</div>
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
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  )
}
