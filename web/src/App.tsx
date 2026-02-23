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

function App() {
  const theme = useTheme()
  const load = useSettingsStore(s => s.load)

  useEffect(() => { load() }, [load])

  return (
    <ConfigProvider locale={zhCN} theme={theme === 'dark' ? darkTheme : lightTheme}>
      <BrowserRouter>
        <AppLayout>
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/library" element={<LibraryPage />} />
          </Routes>
        </AppLayout>
      </BrowserRouter>
    </ConfigProvider>
  )
}

export default App
