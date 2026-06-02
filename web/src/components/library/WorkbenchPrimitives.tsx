import { type ReactNode } from 'react'

export type WorkbenchTone = 'good' | 'warn' | 'info' | 'danger' | 'processing' | 'neutral'

export interface WorkbenchMetricItem {
  key: string
  label: ReactNode
  value: ReactNode
  tone?: WorkbenchTone
}

export function WorkbenchPanel({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return <section className={`kb-workbench-panel ${className}`.trim()}>{children}</section>
}

export function WorkbenchStatusPill({
  children,
  tone = 'neutral',
}: {
  children: ReactNode
  tone?: WorkbenchTone
}) {
  return <span className={`kb-workbench-pill is-${tone}`}>{children}</span>
}

export function WorkbenchMetricStrip({
  items,
  className = '',
}: {
  items: WorkbenchMetricItem[]
  className?: string
}) {
  return (
    <div className={`kb-workbench-metric-strip ${className}`.trim()}>
      {items.map((item) => (
        <span key={item.key} className={`kb-workbench-metric is-${item.tone || 'neutral'}`}>
          <em>{item.label}</em>
          <strong>{item.value}</strong>
        </span>
      ))}
    </div>
  )
}
