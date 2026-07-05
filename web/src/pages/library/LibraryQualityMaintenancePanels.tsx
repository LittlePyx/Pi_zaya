import { useMemo } from 'react'
import { Button, Progress, Tag, Typography } from 'antd'
import type {
  LibraryFigureAssetRefreshResponse,
  LibraryFigureAssetScanItem,
  LibraryFigureAssetScanResponse,
} from '../../api/library'
import type {
  ShelfMetadataBackfillJobState,
  ShelfMetadataBackfillResponse,
  ShelfMetadataBackfillScanResponse,
} from '../../api/references'
import {
  normalizeTextValue,
  qualityStatusText,
  qualityVerificationText,
} from './libraryPageUtils'
import './LibraryQualityMaintenancePanels.css'

const { Text } = Typography

type LibraryQualityMetadataBackfillPanelProps = {
  S: Record<string, string>
  state: ShelfMetadataBackfillJobState | null
  scan: ShelfMetadataBackfillScanResponse | null
  result: ShelfMetadataBackfillResponse | null
  tone: string
  running: boolean
  phase: string
  progress: number
  refreshing: boolean
  onStart: () => void
  onRefresh: () => void
}

export function LibraryQualityMetadataBackfillPanel({
  S,
  state,
  scan,
  result,
  tone,
  running,
  phase,
  progress,
  refreshing,
  onStart,
  onRefresh,
}: LibraryQualityMetadataBackfillPanelProps) {
  return (
    <div
      className={`kb-lib-quality-metadata-backfill is-${tone}`}
      data-testid="library-metadata-backfill-health"
    >
      <div className="kb-lib-quality-metadata-backfill-head">
        <div>
          <Text className="kb-lib-quality-report-section-title">Literature metadata</Text>
          <strong>{scan ? `${Number(scan.export_ready || 0)}/${Number(scan.scanned || 0)}` : 'Not scanned'}</strong>
        </div>
        <Tag color={running ? 'processing' : tone === 'good' ? 'success' : tone === 'error' ? 'error' : tone === 'warning' ? 'warning' : 'default'}>
          {running ? phase : (tone === 'unknown' ? 'Idle' : qualityStatusText(tone, S))}
        </Tag>
      </div>
      <div className="kb-lib-quality-metadata-backfill-grid">
        <span>
          <strong>{Number(scan?.scanned || 0)}</strong>
          <em>refs scanned</em>
        </span>
        <span>
          <strong>{Number(scan?.target_count || 0)}</strong>
          <em>repairable</em>
        </span>
        <span>
          <strong>{Number(scan?.needs_repair || 0)}</strong>
          <em>remaining</em>
        </span>
        <span>
          <strong>{Number(result?.preheated || result?.persisted || result?.changed || 0)}</strong>
          <em>preheated</em>
        </span>
      </div>
      {running ? (
        <Progress percent={progress} size="small" showInfo={false} />
      ) : null}
      {(scan?.missing_fields || []).length > 0 ? (
        <div className="kb-lib-quality-metadata-backfill-fields">
          {(scan?.missing_fields || []).slice(0, 5).map((field) => (
            <em key={field.name}>{field.name} x{field.count}</em>
          ))}
        </div>
      ) : null}
      {state?.error_detail ? (
        <p>{state.error_detail}</p>
      ) : result?.verification ? (
        <p>{qualityVerificationText(result.verification as unknown as Record<string, unknown>)}</p>
      ) : null}
      <div className="kb-lib-quality-metadata-backfill-actions">
        <Button
          size="small"
          loading={refreshing || running}
          disabled={running}
          onClick={onStart}
        >
          {running ? 'Running' : 'Preheat'}
        </Button>
        <Button
          size="small"
          type="text"
          loading={refreshing && !running}
          onClick={onRefresh}
        >
          Refresh
        </Button>
      </div>
    </div>
  )
}

type LibraryQualityFigureAssetsPanelProps = {
  S: Record<string, string>
  scan: LibraryFigureAssetScanResponse | null
  scanRunning: boolean
  refreshResult: LibraryFigureAssetRefreshResponse | null
  refreshRunning: boolean
  onScan: (includeAll: boolean) => void
  onRefresh: () => void
}

export function LibraryQualityFigureAssetsPanel({
  S,
  scan,
  scanRunning,
  refreshResult,
  refreshRunning,
  onScan,
  onRefresh,
}: LibraryQualityFigureAssetsPanelProps) {
  const tone = scanRunning || refreshRunning
    ? 'warning'
    : scan
      ? (
          normalizeTextValue(scan.status).toLowerCase() === 'error'
            ? 'error'
            : Number(scan.docs_with_issues || 0) > 0 || Number(scan.refresh_recommended || 0) > 0
              ? 'warning'
              : 'good'
        )
      : 'unknown'
  const issueStats = useMemo(
    () => Object.entries(scan?.issue_counts || {})
      .map(([name, count]) => ({ name, count: Number(count || 0) }))
      .filter((item) => item.name && item.count > 0)
      .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
      .slice(0, 6),
    [scan],
  )
  const previewItems = useMemo<LibraryFigureAssetScanItem[]>(
    () => (Array.isArray(scan?.items) ? scan.items : [])
      .filter((item) => item && (Number(item.issue_count || 0) > 0 || Boolean(item.refresh_recommended)))
      .slice(0, 5),
    [scan],
  )
  const refreshableCount = Number(scan?.refresh_recommended || 0)

  return (
    <div
      className={`kb-lib-quality-figure-assets is-${tone}`}
      data-testid="library-figure-assets-health"
    >
      <div className="kb-lib-quality-figure-assets-head">
        <div>
          <Text className="kb-lib-quality-report-section-title">Figure assets</Text>
          <strong>
            {scan
              ? `${Number(scan.refresh_recommended || 0)}/${Number(scan.scanned || 0)} refresh`
              : 'Not scanned'}
          </strong>
        </div>
        <Tag color={scanRunning || refreshRunning ? 'processing' : tone === 'good' ? 'success' : tone === 'error' ? 'error' : tone === 'warning' ? 'warning' : 'default'}>
          {scanRunning
            ? 'Scanning'
            : refreshRunning
              ? 'Queueing'
              : tone === 'unknown'
                ? 'Idle'
                : qualityStatusText(tone, S)}
        </Tag>
      </div>
      <div className="kb-lib-quality-figure-assets-grid">
        <span>
          <strong>{Number(scan?.scanned || 0)}</strong>
          <em>sources scanned</em>
        </span>
        <span>
          <strong>{Number(scan?.figures || 0)}</strong>
          <em>figures</em>
        </span>
        <span>
          <strong>{Number(scan?.docs_with_issues || 0)}</strong>
          <em>issue docs</em>
        </span>
        <span>
          <strong>{refreshableCount}</strong>
          <em>refresh queue</em>
        </span>
      </div>
      {issueStats.length > 0 ? (
        <div className="kb-lib-quality-figure-assets-fields" data-testid="library-figure-assets-issues">
          {issueStats.map((item) => (
            <em key={item.name}>{item.name} x{item.count}</em>
          ))}
        </div>
      ) : null}
      {previewItems.length > 0 ? (
        <div className="kb-lib-quality-figure-assets-list" data-testid="library-figure-assets-list">
          {previewItems.map((item) => {
            const issueCodes = Object.keys(item.issue_counts || {}).filter((code) => Number(item.issue_counts?.[code] || 0) > 0)
            const firstIssue = item.issues?.[0]
            return (
              <div key={item.md_path || item.pdf_name || item.source_name} className="kb-lib-quality-figure-assets-row">
                <span title={item.source_name || item.pdf_name || item.md_path}>{item.source_name || item.pdf_name || 'Converted source'}</span>
                <strong>{issueCodes.slice(0, 3).join(' / ') || firstIssue?.code || 'issue'}</strong>
                <em>{firstIssue?.message || `${item.issue_count} issue(s)`}</em>
              </div>
            )
          })}
        </div>
      ) : null}
      {refreshResult ? (
        <div className="kb-lib-quality-figure-assets-result" data-testid="library-figure-assets-refresh-result">
          <span>queued <strong>{Number(refreshResult.enqueued || 0)}</strong></span>
          <span>busy <strong>{Number(refreshResult.skipped_busy || 0)}</strong></span>
          <span>failed <strong>{Number(refreshResult.failed || 0)}</strong></span>
        </div>
      ) : null}
      <div className="kb-lib-quality-figure-assets-actions">
        <Button
          size="small"
          loading={scanRunning}
          disabled={refreshRunning}
          onClick={() => { onScan(false) }}
        >
          Scan
        </Button>
        <Button
          size="small"
          type="text"
          loading={scanRunning}
          disabled={refreshRunning}
          onClick={() => { onScan(true) }}
        >
          Show all
        </Button>
        <Button
          size="small"
          className="kb-lib-quality-domain-action"
          loading={refreshRunning}
          disabled={scanRunning || scan === null || refreshableCount <= 0}
          onClick={onRefresh}
        >
          Refresh flagged
        </Button>
      </div>
    </div>
  )
}
