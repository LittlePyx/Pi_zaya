import { theme, type ThemeConfig } from 'antd'

const fontFamily = '-apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif'

export const uiTokens = {
  radius: 8,
  controlHeight: 34,
  fontFamily,
}

export const lightPalette = {
  bg: '#fcfcfd',
  panel: '#ffffff',
  text: '#1f2329',
  mutedText: '#667085',
  border: '#e8e8e8',
  accent: '#1677ff',
}

export const darkPalette = {
  bg: '#141414',
  panel: '#1f1f1f',
  text: '#e7eaef',
  mutedText: '#98a2b3',
  border: '#303543',
  accent: '#4daafc',
}

export function buildAppTheme(mode: 'light' | 'dark'): ThemeConfig {
  const palette = mode === 'dark' ? darkPalette : lightPalette
  return {
    algorithm: mode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
    token: {
      borderRadius: uiTokens.radius,
      colorBgContainer: palette.panel,
      colorBgLayout: palette.bg,
      colorBorder: palette.border,
      colorPrimary: palette.accent,
      colorText: palette.text,
      colorTextSecondary: palette.mutedText,
      controlHeight: uiTokens.controlHeight,
      fontFamily: uiTokens.fontFamily,
      wireframe: false,
    },
  }
}
