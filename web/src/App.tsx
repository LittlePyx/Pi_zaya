import { useEffect } from 'react'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { lightTheme, darkTheme } from './styles/theme'
import { useTheme } from './hooks/useTheme'
import { useSettingsStore } from './stores/settingsStore'
import { AppLayout } from './components/layout/AppSider'
import ChatPage from './pages/ChatPage'
import LibraryPage from './pages/LibraryPage'
import MessageListRegressionPage from './pages/MessageListRegressionPage'
import RefsPanelRegressionPage from './pages/RefsPanelRegressionPage'
import ReaderRegressionPage from './pages/ReaderRegressionPage'
import ReaderSplitRegressionPage from './pages/ReaderSplitRegressionPage'

function App() {
  const theme = useTheme()
  const load = useSettingsStore(s => s.load)
  const isReaderRegressionRoute = typeof window !== 'undefined'
    && window.location.pathname.startsWith('/__')

  useEffect(() => {
    if (isReaderRegressionRoute) return
    void load()
  }, [load, isReaderRegressionRoute])

  return (
    <ConfigProvider locale={zhCN} theme={theme === 'dark' ? darkTheme : lightTheme}>
      <BrowserRouter>
        <Routes>
          <Route path="/__message_list_test__" element={<MessageListRegressionPage />} />
          <Route path="/__refs_panel_test__" element={<RefsPanelRegressionPage />} />
          <Route path="/__reader_test__" element={<ReaderRegressionPage />} />
          <Route path="/__reader_split_test__" element={<ReaderSplitRegressionPage />} />
          <Route path="/" element={<AppLayout><ChatPage /></AppLayout>} />
          <Route path="/library" element={<AppLayout><LibraryPage /></AppLayout>} />
        </Routes>
      </BrowserRouter>
    </ConfigProvider>
  )
}

export default App
