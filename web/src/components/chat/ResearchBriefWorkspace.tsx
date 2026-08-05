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
  ReloadOutlined,
  SaveOutlined,
} from '@ant-design/icons'
import {
  chatApi,
  type EvidenceMatrixRecord,
  type ResearchBriefExportFormat,
  type ResearchBriefRecord,
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

  const applyRecord = useCallback((record: ResearchBriefRecord | null) => {
    setActive(record)
    setTitle(String(record?.title || ''))
    setObjective(String(record?.objective || ''))
    setContent(String(record?.content_markdown || ''))
    setSelectedRevision(record ? Number(record.revision || 1) : null)
  }, [])

  const loadRevisions = useCallback(async (briefId: string) => {
    const rows = await chatApi.listResearchBriefRevisions(briefId)
    setRevisions(rows)
  }, [])

  const selectBrief = useCallback(async (briefId: string) => {
    setLoading(true)
    try {
      const record = await chatApi.getResearchBrief(briefId)
      applyRecord(record)
      await loadRevisions(briefId)
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_load_failed)
    } finally {
      setLoading(false)
    }
  }, [S.research_brief_load_failed, applyRecord, loadRevisions])

  const loadBriefs = useCallback(async () => {
    if (!projectId) return
    setLoading(true)
    try {
      const rows = await chatApi.listResearchBriefs(projectId)
      setBriefs(rows)
      if (rows.length > 0) {
        const record = await chatApi.getResearchBrief(rows[0].id)
        applyRecord(record)
        await loadRevisions(record.id)
      } else {
        applyRecord(null)
        setRevisions([])
      }
    } catch (error) {
      message.error(error instanceof Error ? error.message : S.research_brief_load_failed)
    } finally {
      setLoading(false)
    }
  }, [S.research_brief_load_failed, applyRecord, loadRevisions, projectId])

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
    await loadRevisions(record.id)
  }

  const generate = async () => {
    if (!title.trim()) {
      message.warning(S.research_brief_title_required)
      return
    }
    if (!selectedMatrixId) {
      message.warning(S.research_brief_matrix_required)
      return
    }
    setGenerating(true)
    try {
      const record = await chatApi.generateResearchBrief(projectId, {
        title: title.trim(),
        objective: objective.trim(),
        item_keys: seedItems.map((item) => String(item.key || '').trim()).filter(Boolean),
        matrix_id: selectedMatrixId,
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
  const evidence = Array.isArray(active?.evidence) ? active.evidence : []

  return (
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
                <small>r{item.revision} · {item.quality_status}</small>
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
                  <span>{S.research_brief_source_count.replace('{n}', String(seedItems.length))}</span>
                  {active ? <span>r{active.revision}</span> : null}
                  {dirty ? <span className="is-dirty">{S.research_brief_unsaved}</span> : null}
                </div>
                <div className="kb-research-brief-actions">
                  <Button icon={<SaveOutlined />} loading={saving} disabled={!dirty} onClick={() => void save()} data-testid="research-brief-save">
                    {S.research_brief_save}
                  </Button>
                  <Button type="primary" icon={<ReloadOutlined />} loading={generating} onClick={() => void generate()} data-testid="research-brief-generate">
                    {active ? S.research_brief_regenerate : S.research_brief_generate}
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
                options={verifiedMatrices.map((matrix) => ({
                  value: matrix.id,
                  label: `${matrix.title} · r${matrix.revision}`,
                }))}
                status={verifiedMatrices.length <= 0 ? 'warning' : undefined}
                data-testid="research-brief-source-matrix"
              />
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
              {active?.quality_status === 'verified' ? (
                <Alert
                  type="success"
                  showIcon
                  message={S.research_brief_verified_title}
                  description={S.research_brief_verified_body
                    .replace('{supported}', String(numeric(quality.supported_claims)))
                    .replace('{total}', String(numeric(quality.total_claims)))}
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
                      disabled={!active || Boolean(exporting && exporting !== format)}
                      onClick={() => void download(format)}
                      data-testid={`research-brief-export-${format}`}
                    >
                      {format === 'markdown' ? 'MD' : format.toUpperCase()}
                    </Button>
                  ))}
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
  )
}
