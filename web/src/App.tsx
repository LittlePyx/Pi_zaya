import { Suspense, lazy, useEffect } from 'react'
import { ConfigProvider, Skeleton } from 'antd'
import enUS from 'antd/locale/en_US'
import zhCN from 'antd/locale/zh_CN'
import { BrowserRouter, Navigate, Routes, Route } from 'react-router-dom'
import { lightTheme, darkTheme } from './styles/theme'
import { useTheme } from './hooks/useTheme'
import { useSettingsStore } from './stores/settingsStore'
import { AppLayout } from './components/layout/AppSider'
import { authGateBuildEnabled } from './api/authGate'

const ChatPage = lazy(() => import('./pages/ChatPage'))
const LibraryPage = lazy(() => import('./pages/LibraryPage'))
const ResearchNotesPage = lazy(() => import('./pages/ResearchNotesPage'))
const ReaderPage = lazy(() => import('./pages/ReaderPage'))

function internalRoutesEnabled() {
  return import.meta.env.VITE_ENABLE_INTERNAL_ROUTES === '1'
}

function authGateEnabled() {
  return authGateBuildEnabled()
}

const ENABLE_INTERNAL_ROUTES = internalRoutesEnabled()
const ENABLE_AUTH_GATE = authGateEnabled()
const AuthGateComponent = ENABLE_AUTH_GATE
  ? lazy(() => import('./components/layout/AuthGate').then(module => ({ default: module.AuthGate })))
  : null
const MessageListRegressionPage = ENABLE_INTERNAL_ROUTES ? lazy(() => import('./pages/MessageListRegressionPage')) : null
const RefsPanelRegressionPage = ENABLE_INTERNAL_ROUTES ? lazy(() => import('./pages/RefsPanelRegressionPage')) : null
const ReaderRegressionPage = ENABLE_INTERNAL_ROUTES ? lazy(() => import('./pages/ReaderRegressionPage')) : null
const ReaderSplitRegressionPage = ENABLE_INTERNAL_ROUTES ? lazy(() => import('./pages/ReaderSplitRegressionPage')) : null
const ResearchQaReplayPage = ENABLE_INTERNAL_ROUTES ? lazy(() => import('./pages/ResearchQaReplayPage')) : null

function RouteFallback() {
  return (
    <div className="kb-route-fallback" aria-busy="true">
      <Skeleton active title paragraph={{ rows: 6 }} />
    </div>
  )
}

function App() {
  const theme = useTheme()
  const load = useSettingsStore(s => s.load)
  const uiLocale = useSettingsStore(s => s.uiLocale)
  const isReaderRegressionRoute = ENABLE_INTERNAL_ROUTES
    && typeof window !== 'undefined'
    && window.location.pathname.startsWith('/__')
  const antdLocale = uiLocale === 'en' ? enUS : zhCN

  useEffect(() => {
    if (isReaderRegressionRoute) return
    void load()
  }, [load, isReaderRegressionRoute])

  return (
    <ConfigProvider locale={antdLocale} theme={theme === 'dark' ? darkTheme : lightTheme}>
      {AuthGateComponent ? (
        <Suspense fallback={null}>
          <AuthGateComponent />
        </Suspense>
      ) : null}
      <BrowserRouter>
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            {ENABLE_INTERNAL_ROUTES
              && MessageListRegressionPage
              && RefsPanelRegressionPage
              && ReaderRegressionPage
              && ReaderSplitRegressionPage
              && ResearchQaReplayPage ? (
              <>
                <Route path="/__message_list_test__" element={<MessageListRegressionPage />} />
                <Route path="/__refs_panel_test__" element={<RefsPanelRegressionPage />} />
                <Route path="/__reader_test__" element={<ReaderRegressionPage />} />
                <Route path="/__reader_split_test__" element={<ReaderSplitRegressionPage />} />
                <Route path="/__research_qa_replay__" element={<ResearchQaReplayPage />} />
              </>
            ) : null}
            <Route path="/reader/session/:sessionId" element={<ReaderPage />} />
            <Route path="/" element={<AppLayout><ChatPage /></AppLayout>} />
            <Route path="/library" element={<AppLayout><LibraryPage /></AppLayout>} />
            <Route path="/notes" element={<AppLayout><ResearchNotesPage /></AppLayout>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </ConfigProvider>
  )
}

export default App
