import { useCallback } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { message } from 'antd'
import type { LibraryFileItem } from '../../api/library'

export type LibraryQualityFocusBrowseMode = 'list' | 'categories' | 'tags'
export type LibraryQualityFocusTabKey = 'pending' | 'converted' | 'all'

export type LibraryQualityHistoryFocusPlanInput = {
  files: Pick<LibraryFileItem, 'name'>[]
  names: string[]
  scope: string
}

export function buildLibraryQualityHistoryFocusPlan({
  files,
  names,
  scope,
}: LibraryQualityHistoryFocusPlanInput) {
  const availableNames = new Set(files.map((item) => item.name))
  const rawTargets = Array.from(new Set(names.map((name) => String(name || '').trim()).filter(Boolean)))
  const targets = rawTargets.filter((name) => availableNames.has(name))

  return {
    rawTargets,
    targets,
    shouldLoadAllScope: targets.length <= 0 && rawTargets.length > 0 && scope !== 'all',
  }
}

export type LibraryQualityFocusActionsInput = {
  S: Record<string, string>
  files: LibraryFileItem[]
  scope: string
  qualityReviewCount: number
  qualityHistoryRemainingNames: string[]
  loadFiles: (scope: string) => Promise<unknown> | unknown
  setBrowseMode: Dispatch<SetStateAction<LibraryQualityFocusBrowseMode>>
  setFileKeyword: Dispatch<SetStateAction<string>>
  setOnlyQualityIssues: Dispatch<SetStateAction<boolean>>
  setQualityCenterOpen: Dispatch<SetStateAction<boolean>>
  setQualityHistoryFocusNames: Dispatch<SetStateAction<string[]>>
  setScope: Dispatch<SetStateAction<string>>
  setTabKey: Dispatch<SetStateAction<LibraryQualityFocusTabKey>>
}

export function useLibraryQualityFocusActions({
  S,
  files,
  scope,
  qualityReviewCount,
  qualityHistoryRemainingNames,
  loadFiles,
  setBrowseMode,
  setFileKeyword,
  setOnlyQualityIssues,
  setQualityCenterOpen,
  setQualityHistoryFocusNames,
  setScope,
  setTabKey,
}: LibraryQualityFocusActionsInput) {
  const handleFocusQualityReview = useCallback(() => {
    if (qualityReviewCount <= 0) {
      message.info(S.lib_quality_report_no_issues)
      return
    }
    setQualityCenterOpen(true)
    setQualityHistoryFocusNames([])
    setOnlyQualityIssues(true)
    setBrowseMode('list')
    setTabKey('all')
  }, [
    S.lib_quality_report_no_issues,
    qualityReviewCount,
    setBrowseMode,
    setOnlyQualityIssues,
    setQualityCenterOpen,
    setQualityHistoryFocusNames,
    setTabKey,
  ])

  const handleFocusQualityIssue = useCallback((label: string) => {
    const keyword = String(label || '').trim()
    if (!keyword) return
    setQualityCenterOpen(true)
    setFileKeyword(keyword)
    setQualityHistoryFocusNames([])
    setOnlyQualityIssues(true)
    setBrowseMode('list')
    setTabKey('all')
  }, [
    setBrowseMode,
    setFileKeyword,
    setOnlyQualityIssues,
    setQualityCenterOpen,
    setQualityHistoryFocusNames,
    setTabKey,
  ])

  const focusQualityHistoryNames = useCallback((names: string[]) => {
    const plan = buildLibraryQualityHistoryFocusPlan({
      files,
      names,
      scope,
    })
    setQualityCenterOpen(true)
    if (!plan.targets.length) {
      if (plan.rawTargets.length > 0) {
        setQualityHistoryFocusNames(plan.rawTargets)
        setBrowseMode('list')
        setTabKey('all')
        if (plan.shouldLoadAllScope) {
          setScope('all')
          void loadFiles('all')
        }
        return
      }
      message.info(S.lib_quality_history_no_remaining)
      return
    }
    setQualityHistoryFocusNames(plan.targets)
    setBrowseMode('list')
    setTabKey('all')
  }, [
    S.lib_quality_history_no_remaining,
    files,
    loadFiles,
    scope,
    setBrowseMode,
    setQualityCenterOpen,
    setQualityHistoryFocusNames,
    setScope,
    setTabKey,
  ])

  const handleFocusQualityHistoryRemaining = useCallback(() => {
    focusQualityHistoryNames(qualityHistoryRemainingNames)
  }, [focusQualityHistoryNames, qualityHistoryRemainingNames])

  return {
    focusQualityHistoryNames,
    handleFocusQualityHistoryRemaining,
    handleFocusQualityIssue,
    handleFocusQualityReview,
  }
}
