import { useEffect, useRef } from 'react'

interface LocalePreferences {
  loaded: boolean
  uiLocale: string
  refsCardLocale: string
  localePreferencesRevision: number
}

export function useConversationLocaleRefresh(
  activeConvId: string | null,
  conversationLoading: boolean,
  preferences: LocalePreferences,
  refresh: () => Promise<void>,
) {
  const previousKey = useRef<string | null>(null)
  const { loaded, uiLocale, refsCardLocale, localePreferencesRevision } = preferences
  useEffect(() => {
    if (!loaded) return
    const nextKey = `${uiLocale}:${refsCardLocale}:${localePreferencesRevision}`
    if (previousKey.current === null) {
      // The initial message request already uses the server's saved preferences.
      previousKey.current = nextKey
      return
    }
    if (previousKey.current === nextKey || conversationLoading) return
    previousKey.current = nextKey
    if (activeConvId) void refresh()
  }, [activeConvId, conversationLoading, loaded, uiLocale, refsCardLocale, localePreferencesRevision, refresh])
}
