import type { ReactNode } from 'react'

export function AgentTraceSummaryChip({
  label,
  value,
  className,
  title,
  testId,
}: {
  label: ReactNode
  value: ReactNode
  className?: string
  title?: string
  testId?: string
}) {
  return (
    <div className={className} data-testid={testId}>
      <span>{label}</span>
      <strong title={title}>{value}</strong>
    </div>
  )
}
