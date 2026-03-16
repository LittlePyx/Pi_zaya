import type { ReactNode } from 'react'
import { Button, Drawer } from 'antd'

interface PaperGuideReaderShellProps {
  open: boolean
  isInlinePresentation: boolean
  title: string
  titleTooltip?: string
  onClose: () => void
  onCollapse?: () => void
  onAfterOpenChange?: (nextOpen: boolean) => void
  children: ReactNode
}

export function PaperGuideReaderShell({
  open,
  isInlinePresentation,
  title,
  titleTooltip,
  onClose,
  onCollapse,
  onAfterOpenChange,
  children,
}: PaperGuideReaderShellProps) {
  if (isInlinePresentation) {
    if (!open) return null
    return (
      <div className="flex h-full min-h-0 min-w-0 w-full flex-col overflow-hidden bg-[var(--panel)]">
        <div className="kb-reader-shell-head">
          <div className="min-w-0 flex-1">
            <div
              className="kb-reader-shell-title"
              title={titleTooltip || title}
            >
              {title}
            </div>
          </div>
          <div className="kb-reader-shell-actions">
            {onCollapse ? (
              <Button size="small" type="text" className="kb-reader-shell-btn" onClick={onCollapse}>
                Fold
              </Button>
            ) : null}
            <Button size="small" type="text" className="kb-reader-shell-btn" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden px-3 py-2">
          {children}
        </div>
      </div>
    )
  }

  return (
    <Drawer
      open={open}
      size={560}
      mask={false}
      title={title}
      onClose={onClose}
      afterOpenChange={onAfterOpenChange}
      destroyOnClose={false}
    >
      {children}
    </Drawer>
  )
}
