import { theme, type ThemeConfig } from 'antd'

const fontFamily = '-apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif'

export const uiTokens = {
  radius: 8,
  radiusSmall: 6,
  controlHeight: 34,
  fontFamily,
  shadowSoft: '0 1px 2px rgba(15, 23, 42, 0.04)',
  shadowSoftDark: '0 1px 2px rgba(0, 0, 0, 0.2)',
  shadowPopover: '0 14px 32px rgba(15, 23, 42, 0.11)',
  shadowPopoverDark: '0 18px 38px rgba(0, 0, 0, 0.34)',
}

export const lightPalette = {
  bg: '#fcfcfd',
  panel: '#ffffff',
  elevated: '#ffffff',
  text: '#1f2329',
  mutedText: '#667085',
  border: '#e8e8e8',
  borderSubtle: 'rgba(31, 35, 41, 0.08)',
  accent: '#1677ff',
}

export const darkPalette = {
  bg: '#141414',
  panel: '#1f1f1f',
  elevated: '#24272f',
  text: '#e7eaef',
  mutedText: '#98a2b3',
  border: '#303543',
  borderSubtle: 'rgba(231, 234, 239, 0.12)',
  accent: '#4daafc',
}

export function buildAppTheme(mode: 'light' | 'dark'): ThemeConfig {
  const palette = mode === 'dark' ? darkPalette : lightPalette
  return {
    algorithm: mode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
    token: {
      borderRadius: uiTokens.radius,
      borderRadiusSM: uiTokens.radiusSmall,
      boxShadow: mode === 'dark' ? uiTokens.shadowPopoverDark : uiTokens.shadowPopover,
      boxShadowSecondary: mode === 'dark' ? uiTokens.shadowSoftDark : uiTokens.shadowSoft,
      colorBgContainer: palette.panel,
      colorBgElevated: palette.elevated,
      colorBgLayout: palette.bg,
      colorBorder: palette.border,
      colorBorderSecondary: palette.borderSubtle,
      colorPrimary: palette.accent,
      colorText: palette.text,
      colorTextSecondary: palette.mutedText,
      controlHeight: uiTokens.controlHeight,
      fontFamily: uiTokens.fontFamily,
      wireframe: false,
    },
  }
}
