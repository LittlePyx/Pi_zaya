import { useEffect, useMemo, useState } from 'react'
import { Button, Select, Switch, Typography } from 'antd'
import { generateApi, type AnswerQualitySummary } from '../../api/generate'

const { Text } = Typography

export function AnswerQualityPanel({ open }: { open: boolean }) {
  const [qualityLoading, setQualityLoading] = useState(false)
  const [qualitySummary, setQualitySummary] = useState<AnswerQualitySummary | null>(null)
  const [qualityError, setQualityError] = useState('')
  const [intentFilter, setIntentFilter] = useState('')
  const [depthFilter, setDepthFilter] = useState('')
  const [onlyFailed, setOnlyFailed] = useState(false)

  const loadQualitySummary = async () => {
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
      setQualityError(err instanceof Error ? err.message : '加载失败')
    } finally {
      setQualityLoading(false)
    }
  }

  useEffect(() => {
    if (!open) return
    void loadQualitySummary()
  }, [open, intentFilter, depthFilter, onlyFailed])

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
        <Text className="kb-settings-quality-title">回答质量（最近样本）</Text>
        <Button size="small" loading={qualityLoading} onClick={() => { void loadQualitySummary() }}>
          刷新
        </Button>
      </div>

      <div className="kb-settings-quality-filters">
        <Select
          size="small"
          className="kb-settings-quality-filter"
          value={intentFilter}
          onChange={(v) => setIntentFilter(String(v || ''))}
          options={[
            { label: '全部意图', value: '' },
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
            { label: '全部深度', value: '' },
            { label: 'L1', value: 'L1' },
            { label: 'L2', value: 'L2' },
            { label: 'L3', value: 'L3' },
          ]}
        />
        <div className="kb-settings-quality-failed-toggle">
          <Switch size="small" checked={onlyFailed} onChange={setOnlyFailed} />
          <Text type="secondary" className="text-xs">仅未达标</Text>
        </div>
      </div>

      {qualityError ? (
        <Text type="danger" className="text-xs">{qualityError}</Text>
      ) : (
        <div className="kb-settings-quality-body">
          <div className="kb-settings-quality-grid">
            <div className="kb-settings-quality-metric"><span>样本数</span><strong>{qualitySummary?.total || 0}</strong></div>
            <div className="kb-settings-quality-metric"><span>最低达标</span><strong>{fmtRate(Number(qualitySummary?.minimum_ok_rate || 0))}</strong></div>
            <div className="kb-settings-quality-metric"><span>未达标占比</span><strong>{fmtRate(Number(qualitySummary?.failed_rate || 0))}</strong></div>
            <div className="kb-settings-quality-metric"><span>结构完整</span><strong>{fmtRate(Number(qualitySummary?.structure_complete_rate || 0))}</strong></div>
            <div className="kb-settings-quality-metric"><span>证据覆盖</span><strong>{fmtRate(Number(qualitySummary?.evidence_coverage_rate || 0))}</strong></div>
            <div className="kb-settings-quality-metric"><span>下一步覆盖</span><strong>{fmtRate(Number(qualitySummary?.next_steps_coverage_rate || 0))}</strong></div>
            <div className="kb-settings-quality-metric"><span>核心覆盖</span><strong>{fmtRate(Number(qualitySummary?.avg_core_section_coverage || 0))}</strong></div>
          </div>

          {qualityIntentRows.length > 0 ? (
            <div className="kb-settings-quality-intents">
              {qualityIntentRows.map(([intent, rec]) => (
                <div key={intent} className="kb-settings-quality-intent-row">
                  <span>{intent}</span>
                  <span>{rec.count} 条</span>
                  <span>达标 {fmtRate(Number(rec.minimum_ok_rate || 0))}</span>
                </div>
              ))}
            </div>
          ) : (
            <Text type="secondary" className="text-xs">暂无已完成回答样本。</Text>
          )}

          {qualityDepthRows.length > 0 ? (
            <div className="kb-settings-quality-intents">
              {qualityDepthRows.map(([depth, rec]) => (
                <div key={depth} className="kb-settings-quality-intent-row">
                  <span>{depth}</span>
                  <span>{rec.count} 条</span>
                  <span>达标 {fmtRate(Number(rec.minimum_ok_rate || 0))}</span>
                </div>
              ))}
            </div>
          ) : null}

          {qualityFailReasons.length > 0 ? (
            <Text type="secondary" className="text-xs">
              常见未达标原因：{qualityFailReasons.map(([k, v]) => `${k}(${v})`).join(' / ')}
            </Text>
          ) : null}
        </div>
      )}
    </div>
  )
}

