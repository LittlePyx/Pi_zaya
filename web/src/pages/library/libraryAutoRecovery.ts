import type { LibraryFileItem } from '../../api/library'
import { hasConversionQualityIssue } from './libraryPageUtils'

export type LibraryAutoRecoveryKind = 'repair' | 'reindex' | ''

export function libraryAutoRecoveryKind(item: LibraryFileItem): LibraryAutoRecoveryKind {
  if (item.task_state !== 'idle' || !item.md_exists) return ''
  if (hasConversionQualityIssue(item)) return 'repair'
  const indexState = String(item.index_state || '').trim().toLowerCase()
  if (['not_indexed', 'index_stale', 'not_ready'].includes(indexState)) return 'reindex'
  return ''
}

export function libraryAutoRecoverySignature(item: LibraryFileItem, kind: LibraryAutoRecoveryKind): string {
  const quality = item.conversion_quality
  const report = quality?.conversion_report
  return [
    kind,
    item.name,
    item.task_state,
    item.index_state || '',
    item.index_status || '',
    quality?.status || '',
    quality?.score || 0,
    report?.repair_attempt_count || 0,
    report?.latest_repair_attempt?.status || '',
    item.last_conversion?.outcome || '',
    item.last_conversion?.retry_action || '',
  ].join('|')
}
