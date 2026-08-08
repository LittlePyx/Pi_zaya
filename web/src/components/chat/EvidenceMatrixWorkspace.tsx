import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Checkbox,
  Input,
  Modal,
  Popconfirm,
  Select,
  Spin,
  Tabs,
  Tag,
  message,
} from 'antd'
import {
  DeleteOutlined,
  DownloadOutlined,
  FileAddOutlined,
  FileTextOutlined,
  ReloadOutlined,
  SaveOutlined,
} from '@ant-design/icons'
import {
  chatApi,
  type EvidenceComparisonAudit,
  type EvidenceComparisonDimensionInput,
  type EvidenceComparisonDimensionName,
  type EvidenceComparisonMode,
  type EvidenceMatrixCellField,
  type EvidenceMatrixExportFormat,
  type EvidenceMatrixRecord,
  type EvidenceWatchEvent,
  type EvidenceWatchKind,
  type ProjectEvidenceMatrixRow,
} from '../../api/chat'
import { useT } from '../../i18n'
import type { CiteShelfItem } from './citationState'


interface Props {
  open: boolean
  projectId: string
  activeConvId?: string | null
  seedItems: CiteShelfItem[]
  onClose: () => void
  onOpenEvidence?: (evidence: Record<string, unknown>) => void
  onUseForBrief?: (matrix: EvidenceMatrixRecord) => void
}

const CELL_FIELDS: EvidenceMatrixCellField[] = [
  'method',
  'dataset_or_experiment',
  'metric',
  'key_result',
  'limitation',
]
const COMPARISON_DIMENSIONS: EvidenceComparisonDimensionName[] = [
  'task',
  'dataset',
  'evaluation_protocol',
  'metric',
]

function emptyComparisonDimensions(): Record<EvidenceComparisonDimensionName, EvidenceComparisonDimensionInput> {
  return Object.fromEntries(COMPARISON_DIMENSIONS.map((dimension) => [
    dimension,
    { dimension, left_value: '', right_value: '', mapping_confirmed: false },
  ])) as Record<EvidenceComparisonDimensionName, EvidenceComparisonDimensionInput>
}

function numeric(value: unknown): number {
  const number = Number(value)
  return Number.isFinite(number) ? number : 0
}

function cloneRows(rows: ProjectEvidenceMatrixRow[]): ProjectEvidenceMatrixRow[] {
  return rows.map((row) => ({
    ...row,
    cells: Object.fromEntries(
      Object.entries(row.cells || {}).map(([key, cell]) => [
        key,
        cell ? { ...cell, evidence_ids: [...(cell.evidence_ids || [])] } : cell,
      ]),
    ),
  }))
}

function cellValue(row: ProjectEvidenceMatrixRow, field: EvidenceMatrixCellField): string {
  return String(row.cells?.[field]?.value || '')
}

function rowIncomplete(row: ProjectEvidenceMatrixRow): boolean {
  return CELL_FIELDS.some((field) => !cellValue(row, field).trim())
}

export function EvidenceMatrixWorkspace({
  open,
  projectId,
  activeConvId,
  seedItems,
  onClose,
  onOpenEvidence,
  onUseForBrief,
}: Props) {
  const S = useT()
  const [matrices, setMatrices] = useState<EvidenceMatrixRecord[]>([])
  const [active, setActive] = useState<EvidenceMatrixRecord | null>(null)
  const [revisions, setRevisions] = useState<EvidenceMatrixRecord[]>([])
  const [selectedRevision, setSelectedRevision] = useState<number | null>(null)
  const [title, setTitle] = useState('')
  const [objective, setObjective] = useState('')
  const [rows, setRows] = useState<ProjectEvidenceMatrixRow[]>([])
  const [search, setSearch] = useState('')
  const [rowFilter, setRowFilter] = useState<'all' | 'incomplete'>('all')
  const [tab, setTab] = useState('matrix')
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [saving, setSaving] = useState(false)
  const [scanningChanges, setScanningChanges] = useState(false)
  const [applyingChanges, setApplyingChanges] = useState(false)
  const [watchEvents, setWatchEvents] = useState<EvidenceWatchEvent[]>([])
  const [exporting, setExporting] = useState<EvidenceMatrixExportFormat | ''>('')
  const [auditingComparison, setAuditingComparison] = useState(false)
  const [comparisonMode, setComparisonMode] = useState<EvidenceComparisonMode>('ranking')
  const [leftComparisonRowId, setLeftComparisonRowId] = useState('')
  const [rightComparisonRowId, setRightComparisonRowId] = useState('')
  const [comparisonDimensions, setComparisonDimensions] = useState(emptyComparisonDimensions)
  const [leftTarget, setLeftTarget] = useState('')
  const [rightTarget, setRightTarget] = useState('')
  const [targetMappingConfirmed, setTargetMappingConfirmed] = useState(false)
  const [leftResult, setLeftResult] = useState('')
  const [rightResult, setRightResult] = useState('')

  const applyRecord = useCallback((record: EvidenceMatrixRecord | null) => {
    setActive(record)
    setTitle(String(record?.title || ''))
    setObjective(String(record?.objective || ''))
    setRows(cloneRows(record?.rows || []))
    setSelectedRevision(record ? Number(record.revision || 1) : null)
    setLeftComparisonRowId(String(record?.rows?.[0]?.id || ''))
    setRightComparisonRowId(String(record?.rows?.[1]?.id || ''))
    setComparisonDimensions(emptyComparisonDimensions())
    setLeftTarget('')
    setRightTarget('')
    setTargetMappingConfirmed(false)
    setLeftResult('')
    setRightResult('')
  }, [])

  const loadRevisions = useCallback(async (matrixId: string) => {
    setRevisions(await chatApi.listEvidenceMatrixRevisions(matrixId))
  }, [])

  const scanChanges = useCallback(async (quiet = false) => {
    if (!projectId) return
    setScanningChanges(true)
    try {
      const result = await chatApi.scanEvidenceChanges(projectId)
      setWatchEvents(result.items || [])
      if (!quiet) {
        message.success(result.items.length > 0
          ? S.evidence_watch_scan_found.replace('{n}', String(result.items.length))
          : S.evidence_watch_scan_clear)
      }
    } catch (error) {
      if (!quiet) message.error(error instanceof Error ? error.message : S.evidence_watch_scan_failed)
    } finally {
      setScanningChanges(false)
    }
  }, [S.evidence_watch_scan_clear, S.evidence_watch_scan_failed, S.evidence_watch_scan_found, projectId])

  const selectMatrix = useCallback(async (matrixId: string) => {
    setLoading(true)
    try {
      const record = await chatApi.getEvidenceMatrix(matrixId)
      applyRecord(record)
      await loadRevisions(matrixId)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_load_failed)
    } finally {
      setLoading(false)
    }
  }, [S.evidence_matrix_load_failed, applyRecord, loadRevisions])

  const loadMatrices = useCallback(async () => {
    if (!projectId) return
    setLoading(true)
    try {
      const records = await chatApi.listEvidenceMatrices(projectId)
      setMatrices(records)
      if (records.length > 0) {
        const record = await chatApi.getEvidenceMatrix(records[0].id)
        applyRecord(record)
        await loadRevisions(record.id)
      } else {
        applyRecord(null)
        setRevisions([])
      }
      await scanChanges(true)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_load_failed)
    } finally {
      setLoading(false)
    }
  }, [S.evidence_matrix_load_failed, applyRecord, loadRevisions, projectId, scanChanges])

  useEffect(() => {
    if (!open || !projectId) return
    void loadMatrices()
  }, [loadMatrices, open, projectId])

  const dirty = useMemo(() => {
    if (!active) return Boolean(title.trim() || objective.trim() || rows.length)
    return title !== active.title
      || objective !== active.objective
      || JSON.stringify(rows) !== JSON.stringify(active.rows || [])
  }, [active, objective, rows, title])

  const activeWatchEvents = useMemo(
    () => watchEvents.filter((event) => event.matrix_id === active?.id),
    [active?.id, watchEvents],
  )
  const applicableWatchEvents = useMemo(
    () => activeWatchEvents.filter((event) => event.actionable && event.kind !== 'source_unavailable'),
    [activeWatchEvents],
  )
  const hasUnavailableWatchEvent = useMemo(
    () => activeWatchEvents.some((event) => event.kind === 'source_unavailable'),
    [activeWatchEvents],
  )
  const hasMetadataWatchEvent = useMemo(
    () => activeWatchEvents.some((event) => event.kind === 'source_metadata_changed'),
    [activeWatchEvents],
  )

  const visibleRows = useMemo(() => {
    const query = search.trim().toLowerCase()
    return rows
      .filter((row) => rowFilter === 'all' || rowIncomplete(row))
      .filter((row) => !query || [row.paper, row.source_name, row.notes, ...CELL_FIELDS.map((field) => cellValue(row, field))]
        .some((value) => String(value || '').toLowerCase().includes(query)))
      .sort((left, right) => String(left.paper || left.source_name).localeCompare(String(right.paper || right.source_name)))
  }, [rowFilter, rows, search])

  const beginNew = () => {
    applyRecord(null)
    setRevisions([])
    setTitle(S.evidence_matrix_default_title)
    setObjective('')
    setRows([])
    setTab('matrix')
  }

  const refreshListsAfterRecord = async (record: EvidenceMatrixRecord) => {
    setMatrices(await chatApi.listEvidenceMatrices(projectId))
    applyRecord(record)
    await loadRevisions(record.id)
    await scanChanges(true)
  }

  const applyWatchChanges = async () => {
    if (!active || applicableWatchEvents.length <= 0) return
    if (dirty) {
      message.warning(S.evidence_watch_save_first)
      return
    }
    setApplyingChanges(true)
    try {
      const result = await chatApi.applyEvidenceChanges(
        active.id,
        active.revision,
        applicableWatchEvents.map((event) => event.id),
      )
      await refreshListsAfterRecord(result.record)
      message.success(result.record.quality_status === 'verified'
        ? S.evidence_watch_applied_verified.replace('{n}', String(result.refreshed_source_count))
        : S.evidence_watch_applied_review.replace('{n}', String(result.refreshed_source_count)))
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_watch_apply_failed)
    } finally {
      setApplyingChanges(false)
    }
  }

  const ignoreWatchChange = async (event: EvidenceWatchEvent) => {
    try {
      await chatApi.ignoreEvidenceChange(projectId, event.id)
      setWatchEvents((current) => current.filter((item) => item.id !== event.id))
      message.success(S.evidence_watch_acknowledged)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_watch_acknowledge_failed)
    }
  }

  const generate = async () => {
    if (!title.trim()) {
      message.warning(S.evidence_matrix_title_required)
      return
    }
    if (seedItems.length <= 0) {
      message.warning(S.research_brief_sources_required)
      return
    }
    setGenerating(true)
    try {
      const record = await chatApi.generateEvidenceMatrix(projectId, {
        title: title.trim(),
        objective: objective.trim(),
        item_keys: seedItems.map((item) => String(item.key || '').trim()).filter(Boolean),
        source_conv_id: activeConvId || null,
        matrix_id: active?.id || null,
        expected_revision: active?.revision || null,
      })
      await refreshListsAfterRecord(record)
      setTab('matrix')
      message.success(
        record.quality_status === 'verified'
          ? S.evidence_matrix_generated_verified
          : S.evidence_matrix_generated_review,
      )
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_generate_failed)
    } finally {
      setGenerating(false)
    }
  }

  const save = async () => {
    if (!title.trim()) {
      message.warning(S.evidence_matrix_title_required)
      return
    }
    setSaving(true)
    try {
      if (!active) {
        const record = await chatApi.createEvidenceMatrix(projectId, {
          title: title.trim(),
          objective: objective.trim(),
          source_conv_id: activeConvId || null,
        })
        await refreshListsAfterRecord(record)
      } else {
        const record = await chatApi.updateEvidenceMatrix(active.id, {
          expected_revision: active.revision,
          title: title.trim(),
          objective,
          row_updates: rows.map((row) => ({
            row_id: row.id,
            notes: row.notes || '',
            cells: CELL_FIELDS.map((field) => ({ field, value: cellValue(row, field) })),
          })),
        })
        await refreshListsAfterRecord(record)
      }
      message.success(S.evidence_matrix_saved)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_save_failed)
    } finally {
      setSaving(false)
    }
  }

  const updateCell = (rowId: string, field: EvidenceMatrixCellField, value: string) => {
    setRows((current) => current.map((row) => {
      if (row.id !== rowId) return row
      return {
        ...row,
        cells: {
          ...row.cells,
          [field]: {
            field,
            value,
            support_status: value.trim() ? 'needs_review' : 'missing',
            evidence_ids: [...(row.cells?.[field]?.evidence_ids || [])],
            manual_override: value !== cellValue(row, field) || Boolean(row.cells?.[field]?.manual_override),
          },
        },
      }
    }))
  }

  const updateNotes = (rowId: string, notes: string) => {
    setRows((current) => current.map((row) => row.id === rowId ? { ...row, notes } : row))
  }

  const restore = async () => {
    if (!active || !selectedRevision) return
    try {
      const record = await chatApi.restoreEvidenceMatrix(active.id, selectedRevision, active.revision)
      await refreshListsAfterRecord(record)
      message.success(S.evidence_matrix_restored)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_restore_failed)
    }
  }

  const remove = async () => {
    if (!active) return
    try {
      await chatApi.deleteEvidenceMatrix(active.id)
      await loadMatrices()
      message.success(S.evidence_matrix_deleted)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_delete_failed)
    }
  }

  const exportMatrix = async (format: EvidenceMatrixExportFormat) => {
    if (!active) return
    setExporting(format)
    try {
      await chatApi.downloadEvidenceMatrix(active.id, format)
      message.success(S.evidence_matrix_exported.replace('{format}', format.toUpperCase()))
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_export_failed)
    } finally {
      setExporting('')
    }
  }

  const updateComparisonDimension = (
    dimension: EvidenceComparisonDimensionName,
    patch: Partial<EvidenceComparisonDimensionInput>,
  ) => {
    setComparisonDimensions((current) => ({
      ...current,
      [dimension]: { ...current[dimension], ...patch, dimension },
    }))
  }

  const auditComparison = async () => {
    if (!active || active.quality_status !== 'verified') {
      message.warning(S.evidence_matrix_comparison_requires_verified)
      return
    }
    if (dirty) {
      message.warning(S.evidence_matrix_comparison_save_first)
      return
    }
    const dimensions = COMPARISON_DIMENSIONS.map((dimension) => comparisonDimensions[dimension])
    if (
      !leftComparisonRowId
      || !rightComparisonRowId
      || leftComparisonRowId === rightComparisonRowId
      || dimensions.some((item) => !item.left_value.trim() || !item.right_value.trim())
      || !leftTarget.trim()
      || !rightTarget.trim()
      || !leftResult.trim()
      || !rightResult.trim()
    ) {
      message.warning(S.evidence_matrix_comparison_complete_contract)
      return
    }
    setAuditingComparison(true)
    try {
      const record = await chatApi.auditEvidenceComparison(active.id, {
        expected_revision: active.revision,
        mode: comparisonMode,
        left_row_id: leftComparisonRowId,
        right_row_id: rightComparisonRowId,
        dimensions,
        left_target: leftTarget.trim(),
        right_target: rightTarget.trim(),
        target_mapping_confirmed: targetMappingConfirmed,
        left_result: leftResult.trim(),
        right_result: rightResult.trim(),
      })
      const latest = record.comparison_audits?.[record.comparison_audits.length - 1]
      await refreshListsAfterRecord(record)
      setTab('comparisons')
      message.success(
        latest?.status === 'verified'
          ? S.evidence_matrix_comparison_verified
          : S.evidence_matrix_comparison_not_comparable,
      )
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_comparison_failed)
    } finally {
      setAuditingComparison(false)
    }
  }

  const deleteComparison = async (comparison: EvidenceComparisonAudit) => {
    if (!active) return
    try {
      const record = await chatApi.deleteEvidenceComparison(active.id, comparison.id, active.revision)
      await refreshListsAfterRecord(record)
      setTab('comparisons')
      message.success(S.evidence_matrix_comparison_deleted)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_comparison_delete_failed)
    }
  }

  const evidence = active?.evidence || []
  const quality = active?.quality || {}
  const comparisonFlags = active?.comparison_flags || []
  const comparisonAudits = active?.comparison_audits || []
  const fieldLabels: Record<EvidenceMatrixCellField, string> = {
    method: S.evidence_matrix_col_method,
    dataset_or_experiment: S.evidence_matrix_col_experiment,
    metric: S.evidence_matrix_col_metric,
    key_result: S.evidence_matrix_col_result,
    limitation: S.evidence_matrix_col_limitation,
  }
  const comparisonDimensionLabels: Record<EvidenceComparisonDimensionName, string> = {
    task: S.evidence_matrix_comparison_dimension_task,
    dataset: S.evidence_matrix_comparison_dimension_dataset,
    evaluation_protocol: S.evidence_matrix_comparison_dimension_protocol,
    metric: S.evidence_matrix_comparison_dimension_metric,
  }
  const comparisonRowOptions = rows.map((row) => ({
    value: row.id,
    label: row.paper || row.source_name,
  }))
  const watchKindLabels: Record<EvidenceWatchKind, string> = {
    source_added: S.evidence_watch_kind_added,
    source_removed: S.evidence_watch_kind_removed,
    source_unavailable: S.evidence_watch_kind_unavailable,
    source_content_changed: S.evidence_watch_kind_content,
    source_metadata_changed: S.evidence_watch_kind_metadata,
  }

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width="min(1480px, 96vw)"
      title={S.evidence_matrix_workspace_title}
      destroyOnHidden
      className="kb-evidence-matrix-modal"
    >
      <Spin spinning={loading}>
        <div className="kb-evidence-matrix-workspace">
          <aside className="kb-evidence-matrix-list">
            <Button icon={<FileAddOutlined />} onClick={beginNew} block>{S.evidence_matrix_new}</Button>
            {matrices.map((matrix) => (
              <button
                key={matrix.id}
                type="button"
                className={active?.id === matrix.id ? 'is-active' : ''}
                onClick={() => { void selectMatrix(matrix.id) }}
              >
                <strong>{matrix.title}</strong>
                <span>r{matrix.revision} · {matrix.quality_status}</span>
              </button>
            ))}
            {matrices.length <= 0 ? <div className="kb-evidence-matrix-empty">{S.evidence_matrix_empty}</div> : null}
          </aside>

          <section className="kb-evidence-matrix-main">
            <div className="kb-evidence-matrix-toolbar">
              <div className="kb-evidence-matrix-status">
                <Tag color={active?.quality_status === 'verified' ? 'green' : active?.quality_status === 'needs_review' ? 'orange' : 'default'}>
                  {active?.quality_status || 'draft'}
                </Tag>
                <span>{S.evidence_matrix_source_count.replace('{n}', String(seedItems.length))}</span>
                {dirty ? <span className="is-dirty">{S.research_brief_unsaved}</span> : null}
              </div>
              <div className="kb-evidence-matrix-actions">
                <Button icon={<ReloadOutlined />} loading={scanningChanges} onClick={() => { void scanChanges() }} data-testid="evidence-watch-scan">
                  {S.evidence_watch_scan}
                </Button>
                <Button icon={<SaveOutlined />} loading={saving} onClick={() => { void save() }} data-testid="evidence-matrix-save">{S.evidence_matrix_save}</Button>
                <Button type="primary" icon={<ReloadOutlined />} loading={generating} onClick={() => { void generate() }} data-testid="evidence-matrix-generate">
                  {active ? S.evidence_matrix_refresh : S.evidence_matrix_generate}
                </Button>
                {active?.quality_status === 'verified' && onUseForBrief ? (
                  <Button icon={<FileTextOutlined />} onClick={() => onUseForBrief(active)} data-testid="evidence-matrix-use-for-brief">{S.evidence_matrix_use_for_brief}</Button>
                ) : null}
              </div>
            </div>

            <Input value={title} onChange={(event) => setTitle(event.target.value)} placeholder={S.evidence_matrix_title_placeholder} data-testid="evidence-matrix-title" />
            <Input.TextArea
              value={objective}
              onChange={(event) => setObjective(event.target.value)}
              autoSize={{ minRows: 2, maxRows: 4 }}
              placeholder={S.evidence_matrix_objective_placeholder}
              data-testid="evidence-matrix-objective"
            />

            {active && activeWatchEvents.length > 0 ? (
              <section className="kb-evidence-watch" data-testid="evidence-watch-inbox">
                <div className="kb-evidence-watch-heading">
                  <div>
                    <strong>{S.evidence_watch_title}</strong>
                    <span>{S.evidence_watch_body.replace('{n}', String(activeWatchEvents.length))}</span>
                  </div>
                  {applicableWatchEvents.length > 0 && !hasUnavailableWatchEvent && !hasMetadataWatchEvent ? (
                    <Popconfirm
                      title={S.evidence_watch_apply_confirm.replace('{n}', String(applicableWatchEvents.length))}
                      onConfirm={() => { void applyWatchChanges() }}
                    >
                      <Button type="primary" loading={applyingChanges} data-testid="evidence-watch-apply">
                        {S.evidence_watch_apply.replace('{n}', String(applicableWatchEvents.length))}
                      </Button>
                    </Popconfirm>
                  ) : null}
                </div>
                <div className="kb-evidence-watch-list">
                  {activeWatchEvents.map((event) => {
                    const impact = event.impact || {} as EvidenceWatchEvent['impact']
                    const impactText = S.evidence_watch_impact
                      .replace('{rows}', String(impact.affected_row_ids?.length || 0))
                      .replace('{fields}', String(impact.affected_fields?.length || 0))
                      .replace('{comparisons}', String(impact.affected_comparison_ids?.length || 0))
                      .replace('{briefs}', String(impact.affected_brief_count || 0))
                      .replace('{citations}', String(impact.affected_citation_count || 0))
                    return (
                      <article key={event.id} data-testid="evidence-watch-event">
                        <div className="kb-evidence-watch-event-main">
                          <div>
                            <Tag color={event.severity === 'error' ? 'red' : event.severity === 'warning' ? 'orange' : 'blue'}>
                              {watchKindLabels[event.kind] || event.kind}
                            </Tag>
                            <strong>{event.source_name || event.source_path}</strong>
                          </div>
                          <span>{impactText}</span>
                          {event.kind === 'source_unavailable' ? <small>{S.evidence_watch_unavailable_help}</small> : null}
                          {event.kind === 'source_metadata_changed' ? <small>{S.evidence_watch_metadata_help}</small> : null}
                        </div>
                        {event.kind !== 'source_unavailable' ? (
                          <Button size="small" onClick={() => { void ignoreWatchChange(event) }}>
                            {S.evidence_watch_acknowledge}
                          </Button>
                        ) : null}
                      </article>
                    )
                  })}
                </div>
              </section>
            ) : null}

            {active?.quality_status === 'verified' ? (
              <Alert
                type="success"
                showIcon
                message={S.evidence_matrix_verified_title}
                description={S.evidence_matrix_verified_body
                  .replace('{supported}', String(numeric(quality.supported_cell_count)))
                  .replace('{total}', String(numeric(quality.populated_cell_count)))}
              />
            ) : active ? (
              <Alert
                type="warning"
                showIcon
                message={S.evidence_matrix_review_title}
                description={(quality.reasons as string[] | undefined)?.join(', ') || S.evidence_matrix_review_body}
              />
            ) : null}

            {comparisonFlags.map((flag, index) => (
              <Alert
                key={`${String(flag.code || 'flag')}-${index}`}
                type="info"
                showIcon
                message={S.evidence_matrix_comparison_notice}
                description={flag.code === 'experimental_conditions_differ'
                  ? S.evidence_matrix_flag_conditions
                  : flag.code === 'metrics_differ'
                    ? S.evidence_matrix_flag_metrics
                    : String(flag.message || '')}
              />
            ))}

            <Tabs
              activeKey={tab}
              onChange={setTab}
              items={[
                {
                  key: 'matrix',
                  label: `${S.evidence_matrix_matrix_tab} (${rows.length})`,
                  children: (
                    <>
                      <div className="kb-evidence-matrix-filters">
                        <Input.Search value={search} onChange={(event) => setSearch(event.target.value)} placeholder={S.evidence_matrix_search} allowClear />
                        <Select
                          value={rowFilter}
                          onChange={setRowFilter}
                          options={[
                            { value: 'all', label: S.evidence_matrix_filter_all },
                            { value: 'incomplete', label: S.evidence_matrix_filter_incomplete },
                          ]}
                        />
                      </div>
                      <div className="kb-evidence-matrix-table-wrap">
                        <table data-testid="project-evidence-matrix">
                          <thead>
                            <tr>
                              <th>{S.evidence_matrix_col_paper}</th>
                              {CELL_FIELDS.map((field) => <th key={field}>{fieldLabels[field]}</th>)}
                              <th>{S.evidence_matrix_col_notes}</th>
                            </tr>
                          </thead>
                          <tbody>
                            {visibleRows.map((row) => (
                              <tr key={row.id} data-testid="project-evidence-matrix-row">
                                <td>
                                  <strong>{row.paper || row.source_name}</strong>
                                  <span>{[row.year, row.source_name].filter(Boolean).join(' · ')}</span>
                                </td>
                                {CELL_FIELDS.map((field) => {
                                  const cell = row.cells?.[field]
                                  return (
                                    <td key={field} className={!cellValue(row, field).trim() ? 'is-missing' : cell?.manual_override ? 'needs-review' : ''}>
                                      <Input.TextArea
                                        value={cellValue(row, field)}
                                        onChange={(event) => updateCell(row.id, field, event.target.value)}
                                        autoSize={{ minRows: 3, maxRows: 8 }}
                                        placeholder={S.evidence_matrix_cell_missing}
                                      />
                                      <small>{cell?.support_status || 'missing'}</small>
                                    </td>
                                  )
                                })}
                                <td>
                                  <Input.TextArea
                                    value={row.notes || ''}
                                    onChange={(event) => updateNotes(row.id, event.target.value)}
                                    autoSize={{ minRows: 3, maxRows: 8 }}
                                    placeholder={S.evidence_matrix_notes_placeholder}
                                  />
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {visibleRows.length <= 0 ? <div className="kb-evidence-matrix-empty">{S.evidence_matrix_no_rows}</div> : null}
                      </div>
                    </>
                  ),
                },
                {
                  key: 'comparisons',
                  label: `${S.evidence_matrix_comparison_tab} (${comparisonAudits.length})`,
                  children: (
                    <div className="kb-evidence-comparison-workspace">
                      <Alert
                        type="info"
                        showIcon
                        message={S.evidence_matrix_comparison_contract_title}
                        description={S.evidence_matrix_comparison_contract_body}
                      />
                      <div className="kb-evidence-comparison-form">
                        <div className="kb-evidence-comparison-pair-row">
                          <label>
                            <span>{S.evidence_matrix_comparison_mode}</span>
                            <Select<EvidenceComparisonMode>
                              value={comparisonMode}
                              onChange={setComparisonMode}
                              options={[
                                { value: 'ranking', label: S.evidence_matrix_comparison_mode_ranking },
                                { value: 'replication', label: S.evidence_matrix_comparison_mode_replication },
                              ]}
                            />
                          </label>
                          <label>
                            <span>{S.evidence_matrix_comparison_left_source}</span>
                            <Select value={leftComparisonRowId || undefined} onChange={setLeftComparisonRowId} options={comparisonRowOptions} />
                          </label>
                          <label>
                            <span>{S.evidence_matrix_comparison_right_source}</span>
                            <Select value={rightComparisonRowId || undefined} onChange={setRightComparisonRowId} options={comparisonRowOptions} />
                          </label>
                        </div>
                        <div className="kb-evidence-comparison-contract-table">
                          <div className="is-header">
                            <span>{S.evidence_matrix_comparison_dimension}</span>
                            <span>{S.evidence_matrix_comparison_left_value}</span>
                            <span>{S.evidence_matrix_comparison_right_value}</span>
                            <span>{S.evidence_matrix_comparison_mapping}</span>
                          </div>
                          {COMPARISON_DIMENSIONS.map((dimension) => (
                            <div key={dimension}>
                              <strong>{comparisonDimensionLabels[dimension]}</strong>
                              <Input
                                value={comparisonDimensions[dimension].left_value}
                                onChange={(event) => updateComparisonDimension(dimension, { left_value: event.target.value })}
                                placeholder={S.evidence_matrix_comparison_exact_phrase}
                              />
                              <Input
                                value={comparisonDimensions[dimension].right_value}
                                onChange={(event) => updateComparisonDimension(dimension, { right_value: event.target.value })}
                                placeholder={S.evidence_matrix_comparison_exact_phrase}
                              />
                              {dimension === 'metric' ? (
                                <small>{S.evidence_matrix_comparison_controlled_metric}</small>
                              ) : (
                                <Checkbox
                                  checked={Boolean(comparisonDimensions[dimension].mapping_confirmed)}
                                  onChange={(event) => updateComparisonDimension(dimension, { mapping_confirmed: event.target.checked })}
                                >
                                  {S.evidence_matrix_comparison_confirm_mapping}
                                </Checkbox>
                              )}
                            </div>
                          ))}
                        </div>
                        <div className="kb-evidence-comparison-pair-row">
                          <label>
                            <span>{S.evidence_matrix_comparison_left_target}</span>
                            <Input value={leftTarget} onChange={(event) => setLeftTarget(event.target.value)} placeholder={S.evidence_matrix_comparison_exact_phrase} />
                          </label>
                          <label>
                            <span>{S.evidence_matrix_comparison_right_target}</span>
                            <Input value={rightTarget} onChange={(event) => setRightTarget(event.target.value)} placeholder={S.evidence_matrix_comparison_exact_phrase} />
                          </label>
                          {comparisonMode === 'replication' ? (
                            <Checkbox checked={targetMappingConfirmed} onChange={(event) => setTargetMappingConfirmed(event.target.checked)}>
                              {S.evidence_matrix_comparison_confirm_target}
                            </Checkbox>
                          ) : <span />}
                        </div>
                        <div className="kb-evidence-comparison-pair-row">
                          <label>
                            <span>{S.evidence_matrix_comparison_left_result}</span>
                            <Input value={leftResult} onChange={(event) => setLeftResult(event.target.value)} placeholder=".0423 / 31 dB" />
                          </label>
                          <label>
                            <span>{S.evidence_matrix_comparison_right_result}</span>
                            <Input value={rightResult} onChange={(event) => setRightResult(event.target.value)} placeholder=".0445 / 30 dB" />
                          </label>
                          <Button
                            type="primary"
                            loading={auditingComparison}
                            disabled={!active || active.quality_status !== 'verified'}
                            onClick={() => { void auditComparison() }}
                            data-testid="evidence-comparison-audit"
                          >
                            {S.evidence_matrix_comparison_audit}
                          </Button>
                        </div>
                      </div>
                      <div className="kb-evidence-comparison-list">
                        {comparisonAudits.map((comparison) => (
                          <article key={comparison.id} data-testid="evidence-comparison-result">
                            <div className="kb-evidence-comparison-card-head">
                              <div>
                                <Tag color={comparison.status === 'verified' ? 'green' : 'orange'}>{comparison.status}</Tag>
                                {comparison.confirmed_conflict ? <Tag color="red">{S.evidence_matrix_comparison_conflict}</Tag> : null}
                                <strong>{comparison.left_source_name} / {comparison.right_source_name}</strong>
                              </div>
                              <Popconfirm title={S.evidence_matrix_comparison_delete_confirm} onConfirm={() => { void deleteComparison(comparison) }}>
                                <Button danger size="small" icon={<DeleteOutlined />}>{S.evidence_matrix_comparison_delete}</Button>
                              </Popconfirm>
                            </div>
                            <p>{comparison.conclusion}</p>
                            {comparison.reasons.length > 0 ? (
                              <Alert type="warning" showIcon message={S.evidence_matrix_comparison_boundaries} description={comparison.reasons.join(', ')} />
                            ) : null}
                            <small>
                              {S.evidence_matrix_comparison_timing.replace('{ms}', String(numeric(comparison.phase_timings_ms?.total).toFixed(1)))}
                            </small>
                            <div className="kb-evidence-comparison-evidence">
                              {comparison.evidence.map((item, index) => (
                                <button
                                  key={String(item.id || index)}
                                  type="button"
                                  onClick={() => onOpenEvidence?.(item)}
                                  disabled={!onOpenEvidence}
                                >
                                  <strong>{String(item.source_name || item.source_path || S.default_source_fallback)}</strong>
                                  <span>{[item.supports, item.heading_path || item.location_label].filter(Boolean).join(' · ')}</span>
                                  <p>{String(item.evidence_quote || '')}</p>
                                </button>
                              ))}
                            </div>
                          </article>
                        ))}
                        {comparisonAudits.length <= 0 ? <div className="kb-evidence-matrix-empty">{S.evidence_matrix_comparison_empty}</div> : null}
                      </div>
                    </div>
                  ),
                },
                {
                  key: 'evidence',
                  label: `${S.evidence_matrix_evidence_tab} (${evidence.length})`,
                  children: (
                    <div className="kb-evidence-matrix-evidence-list">
                      {evidence.map((item, index) => (
                        <button
                          key={String(item.id || index)}
                          type="button"
                          onClick={() => onOpenEvidence?.(item)}
                          disabled={!onOpenEvidence}
                          data-testid="project-evidence-matrix-evidence"
                        >
                          <strong>{String(item.source_name || item.source_path || S.default_source_fallback)}</strong>
                          <span>{[item.field, item.heading_path || item.location_label].filter(Boolean).join(' · ')}</span>
                          <p>{String(item.evidence_quote || '')}</p>
                        </button>
                      ))}
                      {evidence.length <= 0 ? <div className="kb-evidence-matrix-empty">{S.evidence_matrix_no_evidence}</div> : null}
                    </div>
                  ),
                },
              ]}
            />

            {active ? (
              <div className="kb-evidence-matrix-footer">
                <Select
                  value={selectedRevision || undefined}
                  onChange={(value) => setSelectedRevision(Number(value))}
                  placeholder={S.evidence_matrix_versions}
                  options={revisions.map((record) => ({
                    value: record.revision,
                    label: `r${record.revision} · ${new Date(record.updated_at * 1000).toLocaleString()}`,
                  }))}
                />
                <Button onClick={() => { void restore() }} disabled={!selectedRevision || selectedRevision === active.revision}>
                  {S.evidence_matrix_restore}
                </Button>
                <Select<EvidenceMatrixExportFormat>
                  value={undefined}
                  placeholder={S.evidence_matrix_export}
                  loading={Boolean(exporting)}
                  onChange={(format) => { void exportMatrix(format) }}
                  suffixIcon={<DownloadOutlined />}
                  options={[
                    { value: 'markdown', label: 'Markdown' },
                    { value: 'csv', label: 'CSV' },
                    { value: 'xlsx', label: 'XLSX' },
                  ]}
                />
                <Popconfirm title={S.evidence_matrix_delete_confirm} onConfirm={() => { void remove() }}>
                  <Button danger icon={<DeleteOutlined />}>{S.evidence_matrix_delete}</Button>
                </Popconfirm>
              </div>
            ) : null}
          </section>
        </div>
      </Spin>
    </Modal>
  )
}
