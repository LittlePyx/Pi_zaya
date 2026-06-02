import { Suspense, lazy, useEffect } from 'react'
import { ConfigProvider } from 'antd'
import enUS from 'antd/locale/en_US'
import zhCN from 'antd/locale/zh_CN'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { lightTheme, darkTheme } from './styles/theme'
import { useTheme } from './hooks/useTheme'
import { useSettingsStore } from './stores/settingsStore'
import { AppLayout } from './components/layout/AppSider'

const ChatPage = lazy(() => import('./pages/ChatPage'))
const LibraryPage = lazy(() => import('./pages/LibraryPage'))
const MessageListRegressionPage = lazy(() => import('./pages/MessageListRegressionPage'))
const RefsPanelRegressionPage = lazy(() => import('./pages/RefsPanelRegressionPage'))
const ReaderRegressionPage = lazy(() => import('./pages/ReaderRegressionPage'))
const ReaderSplitRegressionPage = lazy(() => import('./pages/ReaderSplitRegressionPage'))
const ResearchQaReplayPage = lazy(() => import('./pages/ResearchQaReplayPage'))

function App() {
  const theme = useTheme()
  const load = useSettingsStore(s => s.load)
  const uiLocale = useSettingsStore(s => s.uiLocale)
  const isReaderRegressionRoute = typeof window !== 'undefined'
    && window.location.pathname.startsWith('/__')
  const antdLocale = uiLocale === 'en' ? enUS : zhCN

  useEffect(() => {
    if (isReaderRegressionRoute) return
    void load()
  }, [load, isReaderRegressionRoute])

  return (
    <ConfigProvider locale={antdLocale} theme={theme === 'dark' ? darkTheme : lightTheme}>
      <BrowserRouter>
        <Suspense fallback={null}>
          <Routes>
            <Route path="/__message_list_test__" element={<MessageListRegressionPage />} />
            <Route path="/__refs_panel_test__" element={<RefsPanelRegressionPage />} />
            <Route path="/__reader_test__" element={<ReaderRegressionPage />} />
            <Route path="/__reader_split_test__" element={<ReaderSplitRegressionPage />} />
            <Route path="/__research_qa_replay__" element={<ResearchQaReplayPage />} />
            <Route path="/" element={<AppLayout><ChatPage /></AppLayout>} />
            <Route path="/library" element={<AppLayout><LibraryPage /></AppLayout>} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </ConfigProvider>
  )
}

export default App
