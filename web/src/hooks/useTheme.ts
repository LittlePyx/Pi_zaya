import { useEffect } from 'react'
import { useSettingsStore } from '../stores/settingsStore'

export function useTheme() {
  const theme = useSettingsStore(s => s.theme)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  return theme
}
