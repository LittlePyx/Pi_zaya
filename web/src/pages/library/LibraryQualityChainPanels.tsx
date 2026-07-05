import { Tag, Typography } from 'antd'
import type {
  LibraryQualityActionHistoryItem,
  LibraryQualityFeatureHealth,
  LibraryQualityFeatureHealthItem,
  LibraryQualityFullChain,
  LibraryQualityFullChainRootCause,
  LibraryQualityFullChainStage,
} from '../../api/library'
import {
  formatQualityRepairHistoryTime,
  normalizeTextValue,
  qualityActionDeltaText,
  qualityActionHistoryActionText,
  qualityFeatureActionText,
  qualityFullChainActionText,
  qualityStatusText,
} from './libraryPageUtils'
import './LibraryQualityChainPanels.css'

const { Text } = Typography

type QualityFullChainActionResult = {
  status: 'success' | 'warning' | 'error' | 'info'
  summary: string
  detail?: string
  deltaText?: string
  verificationText?: string
  improved?: boolean | null
  updatedAt: number
}

type LibraryQualityChainPanelsProps = {
  S: Record<string, string>
  featureHealth: LibraryQualityFeatureHealth | null
  featureItems: LibraryQualityFeatureHealthItem[]
  fullChain: LibraryQualityFullChain | null
  fullChainStages: LibraryQualityFullChainStage[]
  fullChainRootCauses: LibraryQualityFullChainRootCause[]
  fullChainActionHistory: LibraryQualityActionHistoryItem[]
  actionKey: string
  liveResults: Record<string, QualityFullChainActionResult>
  persistedResults: Record<string, QualityFullChainActionResult>
  onFeatureAction: (item: LibraryQualityFeatureHealthItem) => void
  onStageAction: (stage: LibraryQualityFullChainStage) => void
  onHistoryOpen: (item: LibraryQualityActionHistoryItem) => void
}

export function LibraryQualityChainPanels({
  S,
  featureHealth,
  featureItems,
  fullChain,
  fullChainStages,
  fullChainRootCauses,
  fullChainActionHistory,
  actionKey,
  liveResults,
  persistedResults,
  onFeatureAction,
  onStageAction,
  onHistoryOpen,
}: LibraryQualityChainPanelsProps) {
  return (
    <>
      {featureHealth && featureItems.length > 0 ? (
        <div
          className={`kb-lib-quality-feature-health is-${normalizeTextValue(featureHealth.status).toLowerCase() || 'unknown'}`}
          data-testid="library-quality-feature-health"
        >
          <div className="kb-lib-quality-feature-health-head">
            <div>
              <Text className="kb-lib-quality-report-section-title">Feature health</Text>
              <strong>Q{Math.max(0, Math.min(100, Math.round(Number(featureHealth.score || 0))))}</strong>
            </div>
            <Tag color={featureHealth.status === 'good' ? 'success' : featureHealth.status === 'error' ? 'error' : featureHealth.status === 'warning' ? 'warning' : 'default'}>
              {qualityStatusText(normalizeTextValue(featureHealth.status).toLowerCase(), S)}
            </Tag>
          </div>
          {featureHealth.summary ? <p>{featureHealth.summary}</p> : null}
          <div className="kb-lib-quality-feature-health-grid">
            {featureItems.map((item) => {
              const featureStatus = normalizeTextValue(item.status).toLowerCase() || 'unknown'
              const featureAction = qualityFeatureActionText(item)
              const featureStageKey = normalizeTextValue(item.target_stage || item.key).toLowerCase()
              const featureStageResult = liveResults[featureStageKey] || persistedResults[featureStageKey]
              return (
                <div
                  key={item.key}
                  className={`kb-lib-quality-feature-card is-${featureStatus}${item.blocking ? ' is-blocking' : ''}`}
                  data-quality-feature={item.key}
                  data-testid="library-quality-feature-card"
                >
                  <div className="kb-lib-quality-feature-card-head">
                    <span>{item.label}</span>
                    <Tag color={featureStatus === 'good' ? 'success' : featureStatus === 'error' ? 'error' : featureStatus === 'warning' ? 'warning' : 'default'}>
                      {qualityStatusText(featureStatus, S)}
                    </Tag>
                  </div>
                  <strong>{item.summary || item.detail}</strong>
                  {item.detail ? <span>{item.detail}</span> : null}
                  {featureStageResult?.deltaText || featureStageResult?.verificationText ? (
                    <div
                      className={`kb-lib-quality-feature-result is-${featureStageResult.status}`}
                      data-testid="library-quality-feature-result"
                    >
                      {featureStageResult.deltaText ? <span>{featureStageResult.deltaText}</span> : null}
                      {featureStageResult.verificationText ? <em>{featureStageResult.verificationText}</em> : null}
                    </div>
                  ) : null}
                  <div className="kb-lib-quality-feature-card-foot">
                    <em>Q{Math.max(0, Math.min(100, Math.round(Number(item.score || 0))))}</em>
                    <button
                      type="button"
                      className="kb-lib-quality-feature-action"
                      data-testid="library-quality-feature-action"
                      onClick={() => { onFeatureAction(item) }}
                    >
                      {featureAction}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      ) : null}
      {fullChain ? (
        <div
          className={`kb-lib-quality-full-chain is-${normalizeTextValue(fullChain.status).toLowerCase() || 'unknown'}`}
          data-testid="library-quality-full-chain"
        >
          <div className="kb-lib-quality-full-chain-head">
            <div>
              <Text className="kb-lib-quality-report-section-title">Full-chain health</Text>
              <strong>Q{Math.max(0, Math.min(100, Math.round(Number(fullChain.score || 0))))}</strong>
            </div>
            <Tag color={fullChain.status === 'good' ? 'success' : fullChain.status === 'error' ? 'error' : fullChain.status === 'warning' ? 'warning' : 'default'}>
              {qualityStatusText(normalizeTextValue(fullChain.status).toLowerCase(), S)}
            </Tag>
          </div>
          {fullChain.summary ? (
            <p>{fullChain.summary}</p>
          ) : null}
          {fullChainStages.length > 0 ? (
            <div className="kb-lib-quality-full-chain-stages">
              {fullChainStages.map((stage) => {
                const stageStatus = normalizeTextValue(stage.status).toLowerCase() || 'unknown'
                const stageCount = Number(stage.count || 0)
                const stageActionText = qualityFullChainActionText(stage)
                const cleanStageKey = normalizeTextValue(stage.key).toLowerCase()
                const stageBusy = actionKey === cleanStageKey
                const stageResult = liveResults[cleanStageKey] || persistedResults[cleanStageKey]
                return (
                  <div
                    key={stage.key}
                    className={`kb-lib-quality-full-chain-stage is-${stageStatus}${stage.blocking ? ' is-blocking' : ''}`}
                    data-quality-stage={stage.key}
                    data-testid="library-quality-full-chain-stage"
                  >
                    <div>
                      <span>{stage.label}</span>
                      <Tag color={stageStatus === 'good' ? 'success' : stageStatus === 'error' ? 'error' : stageStatus === 'warning' ? 'warning' : 'default'}>
                        {qualityStatusText(stageStatus, S)}
                      </Tag>
                    </div>
                    <strong>{stage.detail || normalizeTextValue(stage.action).replace(/_/g, ' ')}</strong>
                    <div className="kb-lib-quality-full-chain-stage-foot">
                      <em>{stageCount > 0 ? `${stageCount}` : normalizeTextValue(stage.action).replace(/_/g, ' ')}</em>
                      <button
                        type="button"
                        className="kb-lib-quality-full-chain-stage-action"
                        disabled={Boolean(actionKey) && !stageBusy}
                        data-testid="library-quality-full-chain-stage-action"
                        onClick={() => { onStageAction(stage) }}
                      >
                        {stageBusy ? 'Working' : stageActionText}
                      </button>
                    </div>
                    {stageResult ? (
                      <div
                        className={`kb-lib-quality-full-chain-stage-result is-${stageResult.status}`}
                        data-testid="library-quality-full-chain-stage-result"
                      >
                        <span>{stageResult.summary}</span>
                        {stageResult.detail ? <em>{stageResult.detail}</em> : null}
                        {stageResult.deltaText ? <em>{stageResult.deltaText}</em> : null}
                        {stageResult.verificationText ? <em>{stageResult.verificationText}</em> : null}
                      </div>
                    ) : null}
                  </div>
                )
              })}
            </div>
          ) : null}
          {fullChainRootCauses.length > 0 ? (
            <div className="kb-lib-quality-full-chain-roots">
              {fullChainRootCauses.map((cause) => {
                const causeSeverity = normalizeTextValue(cause.severity).toLowerCase() || 'warning'
                return (
                  <em
                    key={`${cause.domain}-${cause.code}`}
                    className={`is-${causeSeverity}`}
                    data-testid="library-quality-full-chain-root-cause"
                  >
                    {cause.label || cause.code}
                    <span>{cause.code} x{Number(cause.count || 0)}</span>
                  </em>
                )
              })}
            </div>
          ) : null}
          {fullChainActionHistory.length > 0 ? (
            <div className="kb-lib-quality-full-chain-history" data-testid="library-quality-full-chain-history">
              <Text className="kb-lib-quality-report-section-title">Recent actions</Text>
              <div className="kb-lib-quality-full-chain-history-list">
                {fullChainActionHistory.slice(0, 4).map((item) => {
                  const actionStatus = normalizeTextValue(item.status).toLowerCase() || 'info'
                  const createdAt = Number(item.created_at || 0) * 1000
                  const actionText = qualityActionHistoryActionText(item)
                  const deltaText = qualityActionDeltaText(item)
                  return (
                    <div
                      key={item.id || `${item.stage_key}-${item.created_at}-${item.summary}`}
                      className={`kb-lib-quality-full-chain-history-row is-${actionStatus}`}
                      data-testid="library-quality-full-chain-history-row"
                    >
                      <span>{item.stage_label || item.stage_key}</span>
                      <strong>{item.summary}</strong>
                      <em>{deltaText || formatQualityRepairHistoryTime(createdAt)}</em>
                      <button
                        type="button"
                        className="kb-lib-quality-full-chain-history-open"
                        data-testid="library-quality-full-chain-history-open"
                        onClick={() => { onHistoryOpen(item) }}
                      >
                        {actionText}
                      </button>
                    </div>
                  )
                })}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </>
  )
}
