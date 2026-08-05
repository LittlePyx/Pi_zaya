import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
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
  type EvidenceMatrixCellField,
  type EvidenceMatrixExportFormat,
  type EvidenceMatrixRecord,
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
  const [exporting, setExporting] = useState<EvidenceMatrixExportFormat | ''>('')

  const applyRecord = useCallback((record: EvidenceMatrixRecord | null) => {
    setActive(record)
    setTitle(String(record?.title || ''))
    setObjective(String(record?.objective || ''))
    setRows(cloneRows(record?.rows || []))
    setSelectedRevision(record ? Number(record.revision || 1) : null)
  }, [])

  const loadRevisions = useCallback(async (matrixId: string) => {
    setRevisions(await chatApi.listEvidenceMatrixRevisions(matrixId))
  }, [])

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
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_load_failed)
    } finally {
      setLoading(false)
    }
  }, [S.evidence_matrix_load_failed, applyRecord, loadRevisions, projectId])

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

  const evidence = active?.evidence || []
  const quality = active?.quality || {}
  const comparisonFlags = active?.comparison_flags || []
  const fieldLabels: Record<EvidenceMatrixCellField, string> = {
    method: S.evidence_matrix_col_method,
    dataset_or_experiment: S.evidence_matrix_col_experiment,
    metric: S.evidence_matrix_col_metric,
    key_result: S.evidence_matrix_col_result,
    limitation: S.evidence_matrix_col_limitation,
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
