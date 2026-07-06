import type { ReactNode } from 'react'

export type AgentTraceSummaryChipProps = {
  label: ReactNode
  value: ReactNode
  className?: string
  title?: string
  testId?: string
}

export function AgentTraceSummaryChip({
  label,
  value,
  className,
  title,
  testId,
}: AgentTraceSummaryChipProps) {
  return (
    <div className={className} data-testid={testId}>
      <span>{label}</span>
      <strong title={title}>{value}</strong>
    </div>
  )
}
