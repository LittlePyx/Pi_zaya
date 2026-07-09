import { useCallback, useState } from 'react'
import { message } from 'antd'
import type { LibraryFileItem } from '../../api/library'
import { useLibraryStore } from '../../stores/libraryStore'

export type LibrarySuggestionRefreshActionsInput = {
  S: Record<string, string>
  items: LibraryFileItem[]
}

export function useLibrarySuggestionRefreshActions({
  S,
  items,
}: LibrarySuggestionRefreshActionsInput) {
  const regenerateSuggestions = useLibraryStore((s) => s.regenerateSuggestions)
  const [suggestionsRefreshing, setSuggestionsRefreshing] = useState(false)

  const refreshSuggestionsForVisible = useCallback(async () => {
    const targets = items.map((item) => item.name).filter(Boolean)
    if (!targets.length) {
      message.info(S.lib_msg_no_suggestion_candidates)
      return
    }
    setSuggestionsRefreshing(true)
    try {
      const updated = await regenerateSuggestions({ pdf_names: targets, auto_apply_empty: true })
      message.success(S.lib_msg_suggestions_refreshed_count.replace('{n}', String(updated)))
    } catch (err) {
      message.error(err instanceof Error ? err.message : S.lib_msg_refresh_suggestion_fail)
    } finally {
      setSuggestionsRefreshing(false)
    }
  }, [S, items, regenerateSuggestions])

  return {
    refreshSuggestionsForVisible,
    suggestionsRefreshing,
  }
}
