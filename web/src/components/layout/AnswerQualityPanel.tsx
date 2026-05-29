import { useCallback, useEffect, useMemo, useState } from 'react'
import { Button, Select, Switch, Typography } from 'antd'
import { useT } from '../../i18n'
import { generateApi, type AnswerQualitySummary } from '../../api/generate'

const { Text } = Typography

export function AnswerQualityPanel({ open }: { open: boolean }) {
  const S = useT()
  const [qualityLoading, setQualityLoading] = useState(false)
  const [qualitySummary, setQualitySummary] = useState<AnswerQualitySummary | null>(null)
  const [qualityError, setQualityError] = useState('')
  const [intentFilter, setIntentFilter] = useState('')
  const [depthFilter, setDepthFilter] = useState('')
  const [onlyFailed, setOnlyFailed] = useState(false)

  const loadQualitySummary = useCallback(async () => {
    setQualityLoading(true)
    setQualityError('')
    try {
      const res = await generateApi.qualitySummary({
        limit: 200,
        intent: intentFilter,
        depth: depthFilter,
        onlyFailed,
      })
      setQualitySummary(res)
    } catch (err) {
      setQualityError(err instanceof Error ? err.message : S.quality_loading_failed)
    } finally {
      setQualityLoading(false)
    }
  }, [S.quality_loading_failed, depthFilter, intentFilter, onlyFailed])

  useEffect(() => {
    if (!open) return
    void loadQualitySummary()
  }, [open, loadQualitySummary])

  const qualityIntentRows = useMemo(
    () =>
      Object.entries(qualitySummary?.by_intent || {})
        .sort((a, b) => Number((b[1] || {}).count || 0) - Number((a[1] || {}).count || 0))
        .slice(0, 4),
    [qualitySummary],
  )
  const qualityDepthRows = useMemo(
    () =>
      Object.entries(qualitySummary?.by_depth || {})
        .sort((a, b) => Number((b[1] || {}).count || 0) - Number((a[1] || {}).count || 0))
        .slice(0, 4),
    [qualitySummary],
  )
  const qualityFailReasons = useMemo(
    () => Object.entries(qualitySummary?.fail_reasons || {}).slice(0, 3),
    [qualitySummary],
  )

  const fmtRate = (v: number) => `${Math.round(Math.max(0, Math.min(1, Number(v || 0))) * 100)}%`

  return (
    <div className="kb-settings-quality-panel">
      <div className="kb-settings-quality-head">
        <Text className="kb-settings-quality-title">{S.quality_title}</Text>
        <Button size="small" loading={qualityLoading} onClick={() => { void loadQualitySummary() }}>
          {S.quality_refresh}
        </Button>
      </div>

      <div className="kb-settings-quality-filters">
        <Select
          size="small"
          className="kb-settings-quality-filter"
          value={intentFilter}
          onChange={(v) => setIntentFilter(String(v || ''))}
          options={[
            { label: S.quality_all_intents, value: '' },
            { label: 'reading', value: 'reading' },
            { label: 'compare', value: 'compare' },
            { label: 'idea', value: 'idea' },
            { label: 'experiment', value: 'experiment' },
            { label: 'troubleshoot', value: 'troubleshoot' },
            { label: 'writing', value: 'writing' },
          ]}
        />
        <Select
          size="small"
          className="kb-settings-quality-filter"
          value={depthFilter}
          onChange={(v) => setDepthFilter(String(v || ''))}
          options={[
            { label: S.quality_all_depths, value: '' },
            { label: 'L1', value: 'L1' },
            { label: 'L2', value: 'L2' },
            { label: 'L3', value: 'L3' },
          ]}
        />
        <div className="kb-settings-quality-failed-toggle">
          <Switch size="small" checked={onlyFailed} onChange={setOnlyFailed} />
          <Text type="secondary" className="text-xs">{S.quality_only_failed}</Text>
        </div>
      </div>

      {qualityError ? (
        <Text type="danger" className="text-xs">{qualityError}</Text>
      ) : (
        <div className="kb-settings-quality-body">
          <div className="kb-settings-quality-grid">
            <div className="kb-settings-quality-metric"><span>{S.quality_sample_count}</span><strong>{qualitySummary?.total || 0}</strong></div>
            <div className="kb-settings-quality-metric"><span>{S.quality_min_ok}</span><strong>{fmtRate(Number(qualitySummary?.minimum_ok_rate || 0))}</strong></div>
            <div className="kb-settings-quality-metric"><span>{S.quality_failed_rate}</span><strong>{fmtRate(Number(qualitySummary?.failed_rate || 0))}</strong></div>
            <div className="kb-settings-quality-metric"><span>{S.quality_structure}</span><strong>{fmtRate(Number(qualitySummary?.structure_complete_rate || 0))}</strong></div>
            <div className="kb-settings-quality-metric"><span>{S.quality_evidence}</span><strong>{fmtRate(Number(qualitySummary?.evidence_coverage_rate || 0))}</strong></div>
            <div className="kb-settings-quality-metric"><span>{S.quality_next_steps}</span><strong>{fmtRate(Number(qualitySummary?.next_steps_coverage_rate || 0))}</strong></div>
            <div className="kb-settings-quality-metric"><span>{S.quality_core}</span><strong>{fmtRate(Number(qualitySummary?.avg_core_section_coverage || 0))}</strong></div>
          </div>

          {qualityIntentRows.length > 0 ? (
            <div className="kb-settings-quality-intents">
              {qualityIntentRows.map(([intent, rec]) => (
                <div key={intent} className="kb-settings-quality-intent-row">
                  <span>{intent}</span>
                  <span>{rec.count}</span>
                  <span>{fmtRate(Number(rec.minimum_ok_rate || 0))}</span>
                </div>
              ))}
            </div>
          ) : (
            <Text type="secondary" className="text-xs">{S.quality_no_samples}</Text>
          )}

          {qualityDepthRows.length > 0 ? (
            <div className="kb-settings-quality-intents">
              {qualityDepthRows.map(([depth, rec]) => (
                <div key={depth} className="kb-settings-quality-intent-row">
                  <span>{depth}</span>
                  <span>{rec.count}</span>
                  <span>{fmtRate(Number(rec.minimum_ok_rate || 0))}</span>
                </div>
              ))}
            </div>
          ) : null}

          {qualityFailReasons.length > 0 ? (
            <Text type="secondary" className="text-xs">
              {S.quality_fail_reasons}{qualityFailReasons.map(([k, v]) => `${k}(${v})`).join(' / ')}
            </Text>
          ) : null}
        </div>
      )}
    </div>
  )
}
