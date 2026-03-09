import { useLayoutEffect } from 'react'
import { useSettingsStore } from '../stores/settingsStore'

export function useTheme() {
  const theme = useSettingsStore(s => s.theme)

  useLayoutEffect(() => {
    const root = document.documentElement
    const isDark = theme === 'dark'

    // Freeze transitions for one paint cycle to avoid mixed-theme flicker.
    root.classList.add('theme-switching')
    root.classList.toggle('dark', isDark)
    root.style.colorScheme = isDark ? 'dark' : 'light'

    let raf1 = 0
    let raf2 = 0
    raf1 = window.requestAnimationFrame(() => {
      raf2 = window.requestAnimationFrame(() => {
        root.classList.remove('theme-switching')
      })
    })

    return () => {
      if (raf1) window.cancelAnimationFrame(raf1)
      if (raf2) window.cancelAnimationFrame(raf2)
      root.classList.remove('theme-switching')
    }
  }, [theme])

  return theme
}
