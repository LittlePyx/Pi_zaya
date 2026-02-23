import { type ReactNode, useEffect } from 'react'
import { Layout, Menu, Button, List, Typography, Popconfirm } from 'antd'
import {
  MessageOutlined, BookOutlined, PlusOutlined,
  DeleteOutlined, SunOutlined, MoonOutlined, SettingOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { useChatStore } from '../../stores/chatStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { S } from '../../i18n/zh'
import { useState } from 'react'
import { SettingsDrawer } from './SettingsDrawer'

const { Sider, Content } = Layout
const { Text } = Typography

export function AppLayout({ children }: { children: ReactNode }) {
  const nav = useNavigate()
  const loc = useLocation()
  const conversations = useChatStore(s => s.conversations)
  const activeConvId = useChatStore(s => s.activeConvId)
  const loadConvs = useChatStore(s => s.loadConversations)
  const selectConv = useChatStore(s => s.selectConversation)
  const createConv = useChatStore(s => s.createConversation)
  const deleteConv = useChatStore(s => s.deleteConversation)
  const theme = useSettingsStore(s => s.theme)
  const toggleTheme = useSettingsStore(s => s.toggleTheme)
  const [drawerOpen, setDrawerOpen] = useState(false)

  useEffect(() => { loadConvs() }, [loadConvs])

  const menuKey = loc.pathname === '/library' ? 'library' : 'chat'

  return (
    <Layout className="h-screen">
      <Sider width={260} className="!bg-[var(--panel)] border-r border-[var(--border)] flex flex-col overflow-hidden">
        <div className="p-4 text-lg font-bold truncate" style={{ color: 'var(--accent)' }}>
          {S.title}
        </div>

        <Menu
          mode="inline"
          selectedKeys={[menuKey]}
          className="!bg-transparent !border-none"
          items={[
            { key: 'chat', icon: <MessageOutlined />, label: S.chat, onClick: () => nav('/') },
            { key: 'library', icon: <BookOutlined />, label: S.page_library, onClick: () => nav('/library') },
          ]}
        />

        <div className="flex items-center gap-1 px-3 py-2">
          <Button size="small" icon={theme === 'dark' ? <SunOutlined /> : <MoonOutlined />} onClick={toggleTheme} />
          <Button size="small" icon={<SettingOutlined />} onClick={() => setDrawerOpen(true)} />
          <Button size="small" icon={<PlusOutlined />} onClick={() => { createConv(); nav('/') }}>
            {S.new_chat}
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto px-2">
          <List
            size="small"
            dataSource={conversations}
            renderItem={c => (
              <List.Item
                className={`!px-2 !py-1 cursor-pointer rounded ${c.id === activeConvId ? '!bg-[var(--msg-user-bg)]' : ''}`}
                onClick={() => { selectConv(c.id); nav('/') }}
                extra={
                  c.id === activeConvId ? (
                    <Popconfirm title="删除此对话？" onConfirm={() => deleteConv(c.id)}>
                      <DeleteOutlined className="text-xs opacity-50 hover:opacity-100" />
                    </Popconfirm>
                  ) : null
                }
              >
                <Text ellipsis className="text-xs">{c.title}</Text>
              </List.Item>
            )}
          />
        </div>

        <SettingsDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
      </Sider>

      <Content className="overflow-auto bg-[var(--bg)]">
        {children}
      </Content>
    </Layout>
  )
}
