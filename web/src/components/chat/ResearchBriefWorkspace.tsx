import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Input,
  Modal,
  Popconfirm,
  Radio,
  Select,
  Spin,
  Tabs,
  Tag,
  message,
} from 'antd'
import {
  DeleteOutlined,
  DiffOutlined,
  DownloadOutlined,
  FileAddOutlined,
  ReloadOutlined,
  SaveOutlined,
} from '@ant-design/icons'
import {
  chatApi,
  type EvidenceMatrixRecord,
  type ResearchBriefExportFormat,
  type ResearchBriefRecord,
  type ResearchBriefUpdateDecision,
  type ResearchBriefUpdatePlan,
} from '../../api/chat'
import { useT } from '../../i18n'
import type { CiteShelfItem } from './citationState'
import { MarkdownRenderer } from './MarkdownRenderer'


interface Props {
  open: boolean
  projectId: string
  activeConvId?: string | null
  seedItems: CiteShelfItem[]
  sourceMatrixId?: string
  initialBriefId?: string
  onClose: () => void
  onOpenEvidence?: (evidence: Record<string, unknown>) => void
}

function numeric(value: unknown): number {
  const number = Number(value)
  return Number.isFinite(number) ? number : 0
}

function qualityTone(status: string): string {
  if (status === 'verified') return 'green'
  if (status === 'needs_review') return 'orange'
  return 'default'
}

function lineageTone(status: string): string {
  if (status === 'current' || status === 'current_equivalent') return 'green'
  if (status === 'matrix_updated' || status === 'matrix_updated_unverified' || status === 'matrix_unverified') return 'orange'
  if (status === 'untracked') return 'default'
  return 'red'
}

function localeKey(): string {
  if (typeof document === 'undefined') return 'zh'
  return String(document.documentElement.lang || navigator.language || 'zh').toLowerCase().startsWith('en')
    ? 'en'
    : 'zh'
}

export function ResearchBriefWorkspace({
  open,
  projectId,
  activeConvId,
  seedItems,
  sourceMatrixId = '',
  initialBriefId = '',
  onClose,
  onOpenEvidence,
}: Props) {
  const S = useT()
  const [briefs, setBriefs] = useState<ResearchBriefRecord[]>([])
  const [verifiedMatrices, setVerifiedMatrices] = useState<EvidenceMatrixRecord[]>([])
  const [selectedMatrixId, setSelectedMatrixId] = useState('')
  const [active, setActive] = useState<ResearchBriefRecord | null>(null)
  const [revisions, setRevisions] = useState<ResearchBriefRecord[]>([])
  const [selectedRevision, setSelectedRevision] = useState<number | null>(null)
  const [title, setTitle] = useState('')
  const [objective, setObjective] = useState('')
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [saving, setSaving] = useState(false)
  const [exporting, setExporting] = useState<ResearchBriefExportFormat | ''>('')
  const [tab, setTab] = useState('edit')
  const [updatePlan, setUpdatePlan] = useState<ResearchBriefUpdatePlan | null>(null)
  const [updateDecisions, setUpdateDecisions] = useState<Record<string, ResearchBriefUpdateDecision>>({})
  const [planningUpdate, setPlanningUpdate] = useState(false)
  const [applyingUpdate, setApplyingUpdate] = useState(false)

  const applyRecord = useCallback((record: ResearchBriefRecord | null) => {
    setActive(record)
    setTitle(String(record?.title || ''))
    setObjective(String(record?.objective || ''))
    setContent(String(record?.content_markdown || ''))
    setSelectedRevision(record ? Number(record.revision || 1) : null)
    const matrixId = String(record?.lineage?.source_matrix_id || record?.quality?.source_matrix_id || '').trim()
    if (matrixId) setSelectedMatrixId(matrixId)
  }, [])

  const loadRevisions = useCallback(async (briefId: string) => {
    const rows = await chatApi.listResearchBriefRevisions(briefId)
    setRevisions(rows)
  }, [])

  const configureUpdatePlan = useCallback((plan: ResearchBriefUpdatePlan | null) => {
    setUpdatePlan(plan)
    setUpdateDecisions(Object.fromEntries(
      (plan?.items || []).map((item) => [item.id, item.recommended === 'reject' ? 'reject' : 'accept']),
    ))
  }, [])

  const loadOpenUpdatePlan = useCallback(async (record: ResearchBriefRecord) => {
    if (record.lineage?.status !== 'matrix_updated') {
      configureUpdatePlan(null)
      return
    }
    try {
      const plan = await chatApi.getCurrentResearchBriefUpdatePlan(record.id)
      configureUpdatePlan(plan.base_brief_revision === record.revision ? plan : null)
    } catch {
      configureUpdatePlan(null)
    }
  }, [configureUpdatePlan])

  const selectBrief = useCallback(async (briefId: string) => {
    setLoading(true)
    try {
      const record = await chatApi.getResearchBrief(briefId)
      applyRecord(record)
      await Promise.all([loadRevisions(briefId), loadOpenUpdatePlan(record)])
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_load_failed)
    } finally {
      setLoading(false)
    }
  }, [S.research_brief_load_failed, applyRecord, loadOpenUpdatePlan, loadRevisions])

  const loadBriefs = useCallback(async () => {
    if (!projectId) return
    setLoading(true)
    try {
      const rows = await chatApi.listResearchBriefs(projectId)
      setBriefs(rows)
      if (rows.length > 0) {
        const requestedId = String(initialBriefId || '').trim()
        const selected = rows.find((item) => item.id === requestedId) || rows[0]
        const record = await chatApi.getResearchBrief(selected.id)
        applyRecord(record)
        await Promise.all([loadRevisions(record.id), loadOpenUpdatePlan(record)])
      } else {
        applyRecord(null)
        setRevisions([])
        configureUpdatePlan(null)
      }
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_load_failed)
    } finally {
      setLoading(false)
    }
  }, [S.research_brief_load_failed, applyRecord, configureUpdatePlan, initialBriefId, loadOpenUpdatePlan, loadRevisions, projectId])

  const loadMatrices = useCallback(async () => {
    if (!projectId) return
    try {
      const rows = (await chatApi.listEvidenceMatrices(projectId))
        .filter((record) => record.quality_status === 'verified')
      setVerifiedMatrices(rows)
      setSelectedMatrixId((current) => {
        const requested = String(sourceMatrixId || '').trim()
        if (requested && rows.some((row) => row.id === requested)) return requested
        if (current && rows.some((row) => row.id === current)) return current
        if (current) return current
        return rows[0]?.id || ''
      })
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.evidence_matrix_load_failed)
    }
  }, [S.evidence_matrix_load_failed, projectId, sourceMatrixId])

  useEffect(() => {
    if (!open || !projectId) return
    void loadBriefs()
    void loadMatrices()
  }, [loadBriefs, loadMatrices, open, projectId])

  const dirty = useMemo(() => {
    if (!active) return Boolean(title.trim() || objective.trim() || content.trim())
    return title !== active.title || objective !== active.objective || content !== active.content_markdown
  }, [active, content, objective, title])

  const beginNewBrief = () => {
    applyRecord(null)
    configureUpdatePlan(null)
    setRevisions([])
    setTitle(S.research_brief_default_title)
    setObjective('')
    setContent('')
    setTab('edit')
  }

  const refreshListsAfterRecord = async (record: ResearchBriefRecord) => {
    const rows = await chatApi.listResearchBriefs(projectId)
    setBriefs(rows)
    applyRecord(record)
    configureUpdatePlan(null)
    await loadRevisions(record.id)
  }

  const generate = async () => {
    if (!title.trim()) {
      message.warning(S.research_brief_title_required)
      return
    }
    const boundMatrixId = String(
      active?.lineage?.source_matrix_id || active?.quality?.source_matrix_id || '',
    ).trim()
    const matrixId = boundMatrixId || selectedMatrixId
    if (!matrixId) {
      message.warning(S.research_brief_matrix_required)
      return
    }
    if (!verifiedMatrices.some((matrix) => matrix.id === matrixId)) {
      message.warning(S.research_brief_matrix_not_verified)
      return
    }
    setGenerating(true)
    try {
      const record = await chatApi.generateResearchBrief(projectId, {
        title: title.trim(),
        objective: objective.trim(),
        item_keys: seedItems.map((item) => String(item.key || '').trim()).filter(Boolean),
        matrix_id: matrixId,
        source_conv_id: activeConvId || null,
        brief_id: active?.id || null,
        expected_revision: active?.revision || null,
        locale: localeKey(),
      })
      await refreshListsAfterRecord(record)
      setTab('preview')
      message.success(
        record.quality_status === 'verified'
          ? S.research_brief_generated_verified
          : S.research_brief_generated_review,
      )
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_generate_failed)
    } finally {
      setGenerating(false)
    }
  }

  const planIncrementalUpdate = async () => {
    if (!active || active.lineage?.status !== 'matrix_updated') return
    if (dirty) {
      message.warning(S.research_brief_update_save_first)
      return
    }
    setPlanningUpdate(true)
    try {
      const plan = await chatApi.createResearchBriefUpdatePlan(active.id, {
        expected_revision: active.revision,
        locale: localeKey(),
      })
      configureUpdatePlan(plan)
      message.success(S.research_brief_update_plan_ready)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_update_plan_failed)
    } finally {
      setPlanningUpdate(false)
    }
  }

  const applyIncrementalUpdate = async () => {
    if (!active || !updatePlan) return
    setApplyingUpdate(true)
    try {
      const record = await chatApi.applyResearchBriefUpdatePlan(active.id, updatePlan.id, {
        expected_revision: active.revision,
        decisions: updatePlan.items.map((item) => ({
          item_id: item.id,
          decision: updateDecisions[item.id] || 'reject',
        })),
      })
      await refreshListsAfterRecord(record)
      setTab('preview')
      message.success(
        record.quality_status === 'verified'
          ? S.research_brief_update_applied_verified
          : S.research_brief_update_applied_review,
      )
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_update_apply_failed)
    } finally {
      setApplyingUpdate(false)
    }
  }

  const discardIncrementalUpdate = async () => {
    if (!active || !updatePlan) return
    setApplyingUpdate(true)
    try {
      await chatApi.discardResearchBriefUpdatePlan(active.id, updatePlan.id)
      configureUpdatePlan(null)
      message.success(S.research_brief_update_plan_discarded)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_update_plan_failed)
    } finally {
      setApplyingUpdate(false)
    }
  }

  const save = async () => {
    if (!title.trim()) {
      message.warning(S.research_brief_title_required)
      return
    }
    setSaving(true)
    try {
      const record = active
        ? await chatApi.updateResearchBrief(active.id, {
            expected_revision: active.revision,
            title: title.trim(),
            objective,
            content_markdown: content,
          })
        : await chatApi.createResearchBrief(projectId, {
            title: title.trim(),
            objective,
            content_markdown: content,
            source_conv_id: activeConvId || null,
          })
      await refreshListsAfterRecord(record)
      message.success(S.research_brief_saved)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_save_failed)
    } finally {
      setSaving(false)
    }
  }

  const restore = async () => {
    if (!active || !selectedRevision || selectedRevision === active.revision) return
    setSaving(true)
    try {
      const record = await chatApi.restoreResearchBrief(active.id, selectedRevision, active.revision)
      await refreshListsAfterRecord(record)
      message.success(S.research_brief_restored)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_restore_failed)
    } finally {
      setSaving(false)
    }
  }

  const remove = async () => {
    if (!active) return
    setSaving(true)
    try {
      await chatApi.deleteResearchBrief(active.id)
      message.success(S.research_brief_deleted)
      await loadBriefs()
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_delete_failed)
    } finally {
      setSaving(false)
    }
  }

  const download = async (format: ResearchBriefExportFormat) => {
    if (!active) return
    setExporting(format)
    try {
      await chatApi.downloadResearchBrief(active.id, format)
      message.success(S.research_brief_exported.replace('{format}', format.toUpperCase()))
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_export_failed)
    } finally {
      setExporting('')
    }
  }

  const quality = active?.quality || {}
  const qualityReasons = Array.isArray(quality.reasons) ? quality.reasons.map(String) : []
  const claimRepair = quality.claim_repair && typeof quality.claim_repair === 'object'
    ? quality.claim_repair as Record<string, unknown>
    : {}
  const evidence = Array.isArray(active?.evidence)
    ? active.evidence.filter((item) => !item.citation_slot_filler)
    : []
  const lineage = active?.lineage
  const lineageStatus = String(lineage?.status || 'untracked')
  const lineageImpact = lineage?.impact || {}
  const lineageRows = Array.isArray(lineageImpact.rows) ? lineageImpact.rows : []
  const affectedCitations = Array.isArray(lineageImpact.affected_citation_numbers)
    ? lineageImpact.affected_citation_numbers
    : []
  const boundMatrixId = String(lineage?.source_matrix_id || quality.source_matrix_id || '').trim()
  const matrixOptions = useMemo(() => {
    const options = verifiedMatrices.map((matrix) => ({
      value: matrix.id,
      label: `${matrix.title} · r${matrix.revision}`,
    }))
    if (boundMatrixId && !options.some((option) => option.value === boundMatrixId)) {
      options.push({
        value: boundMatrixId,
        label: `${String(lineage?.source_matrix_title || boundMatrixId)} · ${S.research_brief_matrix_not_verified}`,
      })
    }
    return options
  }, [S.research_brief_matrix_not_verified, boundMatrixId, lineage?.source_matrix_title, verifiedMatrices])
  const lineageUpdated = lineageStatus === 'matrix_updated' || lineageStatus === 'matrix_updated_unverified'
  const lineageBlocked = Boolean(lineage && lineage.export_allowed === false)
  const historicalSnapshot = Boolean(
    lineage?.historical_verified && !lineage?.latest_verified,
  )
  const targetMatrixId = boundMatrixId || selectedMatrixId
  const canGenerate = Boolean(
    targetMatrixId && verifiedMatrices.some((matrix) => matrix.id === targetMatrixId),
  )

  const lineageStatusLabel = lineageStatus === 'current'
    ? S.research_brief_lineage_current
    : lineageStatus === 'current_equivalent'
      ? S.research_brief_lineage_equivalent
      : lineageUpdated
        ? S.research_brief_lineage_updated_title
        : lineageStatus === 'untracked'
          ? ''
          : lineageBlocked
            ? S.research_brief_lineage_blocked_title
            : S.research_brief_lineage_unverified_title
  const acceptedUpdateCount = updatePlan?.items.filter(
    (item) => (updateDecisions[item.id] || 'reject') === 'accept',
  ).length || 0
  const preservationPercent = Math.round(
    numeric(updatePlan?.preservation?.unaffected_preservation_ratio) * 100,
  )

  return (
    <>
      <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={1100}
      destroyOnHidden
      title={S.research_brief_workspace_title}
      className="kb-research-brief-modal"
      data-testid="research-brief-workspace"
    >
      <div className="kb-research-brief-layout">
        <aside className="kb-research-brief-list">
          <Button
            block
            icon={<FileAddOutlined />}
            onClick={beginNewBrief}
            data-testid="research-brief-new"
          >
            {S.research_brief_new}
          </Button>
          <div className="kb-research-brief-list-scroll">
            {briefs.map((item) => (
              <button
                type="button"
                key={item.id}
                className={`kb-research-brief-list-item ${active?.id === item.id ? 'is-active' : ''}`}
                onClick={() => void selectBrief(item.id)}
              >
                <span>{item.title}</span>
                <small>
                  r{item.revision} · {item.quality_status}
                  {item.lineage && item.lineage.status !== 'current' && item.lineage.status !== 'untracked'
                    ? ` · ${item.lineage.status}`
                    : ''}
                </small>
              </button>
            ))}
            {briefs.length <= 0 && !loading ? (
              <div className="kb-research-brief-empty">{S.research_brief_empty}</div>
            ) : null}
          </div>
        </aside>
        <section className="kb-research-brief-editor">
          {loading ? (
            <div className="kb-research-brief-loading"><Spin /></div>
          ) : (
            <>
              <div className="kb-research-brief-toolbar">
                <div className="kb-research-brief-status">
                  <Tag color={qualityTone(String(active?.quality_status || 'draft'))}>
                    {String(active?.quality_status || 'draft')}
                  </Tag>
                  {active && lineageStatusLabel ? (
                    <Tag color={lineageTone(lineageStatus)} data-testid="research-brief-lineage-tag">
                      {lineageStatusLabel}
                    </Tag>
                  ) : null}
                  <span>{S.research_brief_source_count.replace('{n}', String(seedItems.length))}</span>
                  {active ? <span>r{active.revision}</span> : null}
                  {dirty ? <span className="is-dirty">{S.research_brief_unsaved}</span> : null}
                </div>
                <div className="kb-research-brief-actions">
                  <Button icon={<SaveOutlined />} loading={saving} disabled={!dirty} onClick={() => void save()} data-testid="research-brief-save">
                    {S.research_brief_save}
                  </Button>
                  <Button
                    type="primary"
                    icon={active && lineageStatus === 'matrix_updated' ? <DiffOutlined /> : <ReloadOutlined />}
                    loading={generating || planningUpdate}
                    disabled={!canGenerate || Boolean(active && lineageUpdated && lineageStatus !== 'matrix_updated')}
                    onClick={() => void (
                      active && lineageStatus === 'matrix_updated'
                        ? planIncrementalUpdate()
                        : generate()
                    )}
                    data-testid="research-brief-generate"
                  >
                    {active && lineageUpdated
                      ? S.research_brief_update_latest
                      : active
                        ? S.research_brief_regenerate
                        : S.research_brief_generate}
                  </Button>
                </div>
              </div>
              <Input
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                placeholder={S.research_brief_title_placeholder}
                maxLength={240}
                data-testid="research-brief-title"
              />
              <Input.TextArea
                value={objective}
                onChange={(event) => setObjective(event.target.value)}
                placeholder={S.research_brief_objective_placeholder}
                autoSize={{ minRows: 2, maxRows: 4 }}
                maxLength={4000}
                data-testid="research-brief-objective"
              />
              <Select
                value={selectedMatrixId || undefined}
                onChange={setSelectedMatrixId}
                placeholder={S.research_brief_matrix_placeholder}
                options={matrixOptions}
                status={verifiedMatrices.length <= 0 ? 'warning' : undefined}
                disabled={Boolean(active && boundMatrixId)}
                data-testid="research-brief-source-matrix"
              />
              {active && boundMatrixId ? (
                <div className="kb-research-brief-matrix-lock">{S.research_brief_matrix_locked}</div>
              ) : null}
              {verifiedMatrices.length <= 0 ? (
                <Alert type="warning" showIcon message={S.research_brief_matrix_required} />
              ) : null}
              {active?.quality_status === 'needs_review' ? (
                <Alert
                  type="warning"
                  showIcon
                  message={S.research_brief_review_title}
                  description={qualityReasons.length > 0 ? qualityReasons.join(', ') : S.research_brief_review_body}
                />
              ) : null}
              {active && lineageUpdated ? (
                <Alert
                  type={lineageStatus === 'matrix_updated' ? 'warning' : 'error'}
                  showIcon
                  message={lineageStatus === 'matrix_updated'
                    ? S.research_brief_lineage_updated_title
                    : S.research_brief_lineage_unverified_title}
                  description={(
                    <div className="kb-research-brief-lineage-impact" data-testid="research-brief-lineage-impact">
                      <p>
                        {(lineageStatus === 'matrix_updated'
                          ? S.research_brief_lineage_updated_body
                          : S.research_brief_lineage_unverified_body)
                          .replace('{saved}', String(lineage?.source_matrix_revision || 0))
                          .replace('{current}', String(lineage?.current_matrix_revision || 0))}
                      </p>
                      <p>
                        {S.research_brief_lineage_impact
                          .replace('{rows}', String(numeric(lineageImpact.changed_row_count)))
                          .replace('{fields}', String(numeric(lineageImpact.changed_field_count)))
                          .replace('{comparisons}', String(numeric(lineageImpact.changed_comparison_count)))
                          .replace('{sources}', String(numeric(lineageImpact.changed_source_count)))}
                      </p>
                      {lineageRows.length > 0 ? (
                        <ul>
                          {lineageRows.slice(0, 6).map((row) => (
                            <li key={row.row_id}>
                              {row.source_name}: {row.fields.join(', ')} ({row.change})
                            </li>
                          ))}
                        </ul>
                      ) : null}
                      <p>
                        {affectedCitations.length > 0
                          ? S.research_brief_lineage_affected_citations.replace(
                              '{citations}',
                              affectedCitations.map((number) => `[${number}]`).join(', '),
                            )
                          : S.research_brief_lineage_no_citation_impact}
                      </p>
                    </div>
                  )}
                />
              ) : null}
              {active && lineageStatus === 'current_equivalent' ? (
                <Alert type="info" showIcon message={S.research_brief_lineage_equivalent} />
              ) : null}
              {active && lineageBlocked ? (
                <Alert
                  type="error"
                  showIcon
                  message={S.research_brief_lineage_blocked_title}
                  description={S.research_brief_lineage_blocked_body}
                  data-testid="research-brief-lineage-blocked"
                />
              ) : null}
              {active && lineageStatus === 'matrix_unverified' ? (
                <Alert
                  type="warning"
                  showIcon
                  message={S.research_brief_lineage_unverified_title}
                  description={S.research_brief_lineage_unverified_body.replace(
                    '{current}',
                    String(lineage?.current_matrix_revision || 0),
                  )}
                />
              ) : null}
              {active?.quality_status === 'verified' ? (
                <Alert
                  type="success"
                  showIcon
                  message={historicalSnapshot
                    ? S.research_brief_historical_verified_title
                    : S.research_brief_verified_title}
                  description={(historicalSnapshot
                    ? S.research_brief_historical_verified_body
                    : S.research_brief_verified_body)
                    .replace('{supported}', String(numeric(quality.supported_claims)))
                    .replace('{total}', String(numeric(quality.total_claims)))
                    .replace('{revision}', String(lineage?.source_matrix_revision || 0))}
                />
              ) : null}
              {String(quality.generation_mode || '') === 'extractive_fallback' ? (
                <Alert
                  type="info"
                  showIcon
                  message={S.research_brief_extract_fallback_title}
                  description={S.research_brief_extract_fallback_body}
                />
              ) : null}
              {String(quality.generation_mode || '') === 'model_synthesis_repaired' ? (
                <Alert
                  type="info"
                  showIcon
                  message={S.research_brief_claim_repair_title}
                  description={S.research_brief_claim_repair_body
                    .replace('{preserved}', String(numeric(claimRepair.preserved_model_claims)))
                    .replace('{removed}', String(numeric(
                      claimRepair.removed_claims_total ?? claimRepair.removed_unsupported_claims,
                    )))
                    .replace('{supplemented}', String(numeric(claimRepair.supplemented_source_claims)))}
                />
              ) : null}
              <Tabs
                activeKey={tab}
                onChange={setTab}
                items={[
                  {
                    key: 'edit',
                    label: S.research_brief_edit_tab,
                    children: (
                      <Input.TextArea
                        value={content}
                        onChange={(event) => setContent(event.target.value)}
                        placeholder={S.research_brief_content_placeholder}
                        className="kb-research-brief-textarea"
                        data-testid="research-brief-content"
                      />
                    ),
                  },
                  {
                    key: 'preview',
                    label: S.research_brief_preview_tab,
                    children: (
                      <div className="kb-research-brief-preview" data-testid="research-brief-preview">
                        <MarkdownRenderer content={content} citeDetails={[]} linkifyPlainCitations={false} />
                      </div>
                    ),
                  },
                  {
                    key: 'evidence',
                    label: `${S.research_brief_evidence_tab} (${evidence.length})`,
                    children: (
                      <div className="kb-research-brief-evidence" data-testid="research-brief-evidence">
                        {evidence.map((item, index) => (
                          <button
                            type="button"
                            key={`${String(item.source_path || item.source_name || '')}-${index}`}
                            className="kb-research-brief-evidence-item"
                            onClick={() => onOpenEvidence?.(item)}
                            disabled={!onOpenEvidence}
                          >
                            <strong>[{String(item.citation_number || index + 1)}] {String(item.source_name || item.source_path || S.default_source_fallback)}</strong>
                            <span>{String(item.heading_path || item.location_label || '')}</span>
                            <small>{String(item.evidence_quote || '')}</small>
                          </button>
                        ))}
                        {evidence.length <= 0 ? <div className="kb-research-brief-empty">{S.research_brief_no_evidence}</div> : null}
                      </div>
                    ),
                  },
                ]}
              />
              <div className="kb-research-brief-footer">
                <div className="kb-research-brief-revisions">
                  <Select
                    value={selectedRevision || undefined}
                    placeholder={S.research_brief_versions}
                    options={revisions.map((item) => ({
                      value: item.revision,
                      label: `r${item.revision} · ${item.quality_status}`,
                    }))}
                    onChange={(value) => setSelectedRevision(Number(value))}
                    disabled={!active}
                    data-testid="research-brief-revision-select"
                  />
                  <Button onClick={() => void restore()} disabled={!active || !selectedRevision || selectedRevision === active.revision}>
                    {S.research_brief_restore}
                  </Button>
                </div>
                <div className="kb-research-brief-export-actions">
                  {(['markdown', 'docx', 'bibtex', 'ris'] as ResearchBriefExportFormat[]).map((format) => (
                    <Button
                      key={format}
                      icon={<DownloadOutlined />}
                      loading={exporting === format}
                      disabled={!active || lineageBlocked || Boolean(exporting && exporting !== format)}
                      onClick={() => void download(format)}
                      data-testid={`research-brief-export-${format}`}
                    >
                      {format === 'markdown' ? 'MD' : format.toUpperCase()}
                    </Button>
                  ))}
                  {lineageBlocked ? (
                    <span className="kb-research-brief-export-blocked">{S.research_brief_export_blocked}</span>
                  ) : null}
                  <Popconfirm
                    title={S.research_brief_delete_confirm}
                    onConfirm={() => void remove()}
                    disabled={!active}
                  >
                    <Button danger icon={<DeleteOutlined />} disabled={!active} data-testid="research-brief-delete" />
                  </Popconfirm>
                </div>
              </div>
            </>
          )}
        </section>
      </div>
      </Modal>
      <Modal
        open={Boolean(updatePlan)}
        onCancel={() => configureUpdatePlan(null)}
        width={1040}
        destroyOnHidden
        title={S.research_brief_update_review_title}
        className="kb-research-brief-update-modal"
        footer={[
          <Button key="close" onClick={() => configureUpdatePlan(null)}>
            {S.research_brief_update_close}
          </Button>,
          <Button
            key="discard"
            danger
            loading={applyingUpdate}
            onClick={() => void discardIncrementalUpdate()}
            data-testid="research-brief-update-discard"
          >
            {S.research_brief_update_discard}
          </Button>,
          <Button
            key="apply"
            type="primary"
            loading={applyingUpdate}
            onClick={() => void applyIncrementalUpdate()}
            data-testid="research-brief-update-apply"
          >
            {S.research_brief_update_apply
              .replace('{accepted}', String(acceptedUpdateCount))
              .replace('{total}', String(updatePlan?.items.length || 0))}
          </Button>,
        ]}
      >
        {updatePlan ? (
          <div className="kb-research-brief-update-review" data-testid="research-brief-update-plan">
            <Alert
              type="info"
              showIcon
              message={S.research_brief_update_summary
                .replace('{source}', String(updatePlan.source_matrix_revision))
                .replace('{target}', String(updatePlan.target_matrix_revision))
                .replace('{count}', String(updatePlan.items.length))}
              description={S.research_brief_update_preservation
                .replace('{percent}', String(preservationPercent))
                .replace('{chars}', String(numeric(updatePlan.preservation.unaffected_character_count)))}
            />
            {updatePlan.items.length <= 0 ? (
              <Alert
                type="success"
                showIcon
                message={S.research_brief_update_audit_only}
              />
            ) : null}
            {updatePlan.items.map((item, index) => (
              <section
                key={item.id}
                className="kb-research-brief-update-item"
                data-testid={`research-brief-update-item-${index}`}
              >
                <div className="kb-research-brief-update-item-header">
                  <div>
                    <strong>{item.heading || S.research_brief_update_affected_claim}</strong>
                    <small>
                      {S.research_brief_update_citations
                        .replace(
                          '{before}',
                          item.citation_numbers_before.map((number) => `[${number}]`).join(', ') || '—',
                        )
                        .replace(
                          '{after}',
                          item.citation_numbers_after.map((number) => `[${number}]`).join(', ') || S.research_brief_update_deleted,
                        )}
                    </small>
                  </div>
                  <Radio.Group
                    value={updateDecisions[item.id] || 'reject'}
                    onChange={(event) => setUpdateDecisions((current) => ({
                      ...current,
                      [item.id]: event.target.value as ResearchBriefUpdateDecision,
                    }))}
                    optionType="button"
                    buttonStyle="solid"
                    options={[
                      { label: S.research_brief_update_accept, value: 'accept' },
                      { label: S.research_brief_update_reject, value: 'reject' },
                    ]}
                    data-testid={`research-brief-update-decision-${index}`}
                  />
                </div>
                <div className="kb-research-brief-update-diff">
                  <div>
                    <span>{S.research_brief_update_before}</span>
                    <pre>{item.old_markdown}</pre>
                  </div>
                  <div>
                    <span>{S.research_brief_update_after}</span>
                    <pre>{item.proposed_markdown || S.research_brief_update_deleted}</pre>
                  </div>
                </div>
              </section>
            ))}
            <div className="kb-research-brief-update-preview">
              <strong>{S.research_brief_update_preview}</strong>
              <MarkdownRenderer
                content={updatePlan.preview_content_markdown}
                citeDetails={[]}
                linkifyPlainCitations={false}
              />
            </div>
          </div>
        ) : null}
      </Modal>
    </>
  )
}
