interface SystemBOverviewStateStrings extends Record<string, string> {
  cite_loading: string
  cite_loading_summary: string
  cite_summary_unavailable: string
}

export interface BuildSystemBOverviewStateOptions {
  S: SystemBOverviewStateStrings
  isSystemB: boolean
  loading: boolean
  paperOverviewText: string
  showReference: boolean
  bibliometricsChecked: boolean
  doiLabel: string
  systemBTitle: string
}

export interface SystemBOverviewState {
  showOverviewLoading: boolean
  overviewLoadingLabel: string
  showOverviewUnavailable: boolean
  overviewUnavailableLabel: string
}

export function buildSystemBOverviewState({
  S,
  isSystemB,
  loading,
  paperOverviewText,
  showReference,
  bibliometricsChecked,
  doiLabel,
  systemBTitle,
}: BuildSystemBOverviewStateOptions): SystemBOverviewState {
  const showOverviewLoading = Boolean(isSystemB && loading && !paperOverviewText)
  const showOverviewUnavailable = Boolean(
    isSystemB
    && !loading
    && bibliometricsChecked
    && !paperOverviewText
    && !showReference
    && (doiLabel || systemBTitle),
  )

  return {
    showOverviewLoading,
    overviewLoadingLabel: S.cite_loading_summary || S.cite_loading,
    showOverviewUnavailable,
    overviewUnavailableLabel: S.cite_summary_unavailable,
  }
}
