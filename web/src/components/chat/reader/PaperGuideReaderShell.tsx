import type { ReactNode } from 'react'
import { Button, Drawer } from 'antd'
import { ExportOutlined } from '@ant-design/icons'

interface PaperGuideReaderShellProps {
  open: boolean
  isInlinePresentation: boolean
  surface?: 'dock' | 'page'
  title: string
  titleTooltip?: string
  onClose: () => void
  onCollapse?: () => void
  onOpenStandalone?: () => void
  openStandaloneLabel?: string
  collapseLabel?: string
  closeLabel?: string
  onAfterOpenChange?: (nextOpen: boolean) => void
  children: ReactNode
}

export function PaperGuideReaderShell({
  open,
  isInlinePresentation,
  surface = 'dock',
  title,
  titleTooltip,
  onClose,
  onCollapse,
  onOpenStandalone,
  openStandaloneLabel = 'Open window',
  collapseLabel = 'Fold',
  closeLabel = 'Close',
  onAfterOpenChange,
  children,
}: PaperGuideReaderShellProps) {
  if (isInlinePresentation) {
    if (!open) return null
    const isPageSurface = surface === 'page'
    return (
      <div className={`kb-reader-shell ${isPageSurface ? 'is-page' : 'is-dock'}`}>
        {!isPageSurface ? (
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
              {onOpenStandalone ? (
                <Button
                  size="small"
                  type="text"
                  className="kb-reader-shell-btn"
                  icon={<ExportOutlined />}
                  aria-label={openStandaloneLabel}
                  title={openStandaloneLabel}
                  onClick={onOpenStandalone}
                />
              ) : null}
              {onCollapse ? (
                <Button size="small" type="text" className="kb-reader-shell-btn" onClick={onCollapse}>
                  {collapseLabel}
                </Button>
              ) : null}
              <Button size="small" type="text" className="kb-reader-shell-btn" onClick={onClose}>
                {closeLabel}
              </Button>
            </div>
          </div>
        ) : null}
        <div className={`kb-reader-shell-content ${isPageSurface ? 'is-page' : 'is-dock'}`}>
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
      extra={onOpenStandalone ? (
        <Button
          size="small"
          type="text"
          className="kb-reader-shell-btn"
          icon={<ExportOutlined />}
          aria-label={openStandaloneLabel}
          title={openStandaloneLabel}
          onClick={onOpenStandalone}
        />
      ) : undefined}
      onClose={onClose}
      afterOpenChange={onAfterOpenChange}
      destroyOnClose={false}
    >
      {children}
    </Drawer>
  )
}
