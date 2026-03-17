import { type ReactNode, useEffect, useMemo, useState } from 'react'
import {
  Layout,
  Menu,
  Button,
  Typography,
  Modal,
  Input,
  Empty,
  Tooltip,
  message,
} from 'antd'
import { Dropdown } from 'antd'
import type { MenuProps } from 'antd'
import {
  MessageOutlined,
  BookOutlined,
  PlusOutlined,
  DeleteOutlined,
  SunOutlined,
  MoonOutlined,
  SettingOutlined,
  FolderOpenOutlined,
  EditOutlined,
  MoreOutlined,
  SearchOutlined,
  CaretRightOutlined,
  CaretDownOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { useChatStore } from '../../stores/chatStore'
import { useSettingsStore } from '../../stores/settingsStore'
import type { Conversation, Project } from '../../api/chat'
import { SettingsDrawer } from './SettingsDrawer'

const { Sider, Content } = Layout
const { Text } = Typography

function formatRelativeTime(ts?: number) {
  if (!ts) return ''
  const now = Date.now() / 1000
  const diff = Math.max(0, Math.floor(now - ts))
  if (diff < 60) return '刚刚'
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)} 天前`
  const d = new Date(ts * 1000)
  const mm = `${d.getMonth() + 1}`.padStart(2, '0')
  const dd = `${d.getDate()}`.padStart(2, '0')
  return `${mm}-${dd}`
}

function matchesKeyword(text: string, keyword: string) {
  if (!keyword) return true
  return String(text || '').toLowerCase().includes(keyword)
}

interface SwitchStressOptions {
  rounds?: number
  delayMs?: number
  includeLibrary?: boolean
  awaitSelect?: boolean
  convIds?: string[]
}

interface KbSwitchPerfSummary {
  total: number
  success: number
  stale: number
  error: number
  sameConv: number
  avgSuccessMs: number
}

interface SwitchStressResult {
  rounds: number
  delayMs: number
  includeLibrary: boolean
  awaitSelect: boolean
  elapsedMs: number
  summary: KbSwitchPerfSummary | null
}

interface KbSwitchPerfEvent {
  ts: number
  convId: string
  token: number
  status: string
  durationMs: number
  usedCache: boolean
  messageCount: number
  note: string
}

interface KbSwitchPerfApi {
  getLogs: () => KbSwitchPerfEvent[]
  clear: () => void
  summary: () => KbSwitchPerfSummary
}

interface KbDebugApi {
  runSwitchStress?: (opts?: SwitchStressOptions) => Promise<SwitchStressResult>
  getSwitchPerf?: () => KbSwitchPerfEvent[]
  clearSwitchPerf?: () => void
}

interface DebugWindow extends Window {
  __kbDebug?: KbDebugApi
  __kbSwitchPerf?: KbSwitchPerfApi
}

function ConversationRow({
  conversation,
  active,
  onOpen,
  onRename,
  onDelete,
  onMove,
  moveMenuItems,
}: {
  conversation: Conversation
  active: boolean
  onOpen: () => void
  onRename: () => void
  onDelete: () => void
  onMove?: (targetProjectId: string) => void
  moveMenuItems?: MenuProps['items']
}) {
  const menuItems: MenuProps['items'] = [
    { key: 'rename', icon: <EditOutlined />, label: '重命名' },
    ...(moveMenuItems && moveMenuItems.length > 0
      ? [{ key: 'move', icon: <FolderOpenOutlined />, label: '移动到', children: moveMenuItems }]
      : []),
    { type: 'divider' },
    { key: 'delete', icon: <DeleteOutlined />, label: '删除', danger: true },
  ]

  return (
    <div
      className={`kb-conv-row flex items-center gap-1 rounded-lg px-1.5 py-0.5 text-xs cursor-pointer ${
        active ? 'is-active' : ''
      }`}
      onClick={onOpen}
    >
      <MessageOutlined className="shrink-0 opacity-60" />
      <div className="kb-conv-meta min-w-0 flex-1">
        <div className="kb-conv-title" title={conversation.title}>
          <span className="kb-conv-title-text">{conversation.title}</span>
        </div>
        <span className="kb-conv-time">{formatRelativeTime(conversation.updated_at) || ' '}</span>
      </div>
      <Dropdown
        trigger={['click']}
        menu={{
          items: menuItems,
          onClick: ({ key, domEvent }) => {
            domEvent.stopPropagation()
            if (key === 'rename') {
              onRename()
              return
            }
            if (String(key).startsWith('move:')) {
              onMove?.(String(key).slice(5))
              return
            }
            Modal.confirm({
              title: '确认删除这个对话吗？',
              onOk: async () => {
                await onDelete()
              },
            })
          },
        }}
      >
        <Button
          type="text"
          size="small"
          icon={<MoreOutlined />}
          className="kb-side-menu-trigger"
          onClick={(e) => e.stopPropagation()}
        />
      </Dropdown>
    </div>
  )
}

function ProjectSection({
  project,
  selected,
  conversations,
  activeConvId,
  collapsed,
  onToggleCollapsed,
  onSelect,
  onOpenConversation,
  onRenameConversation,
  onDeleteConversation,
  onRename,
  onDelete,
}: {
  project: Project
  selected: boolean
  conversations: Conversation[]
  activeConvId: string | null
  collapsed: boolean
  onToggleCollapsed: () => void
  onSelect: () => void
  onOpenConversation: (id: string) => void
  onRenameConversation: (conversation: Conversation) => void
  onDeleteConversation: (id: string) => void
  onRename: () => void
  onDelete: () => void
}) {
  return (
    <div className={`kb-project-card rounded-lg overflow-hidden ${selected ? 'is-active' : ''}`}>
      <div className="kb-project-head flex items-center gap-1 px-2 py-1">
        <Button
          type="text"
          size="small"
          className="!w-6 !h-6 !min-w-0"
          icon={collapsed ? <CaretRightOutlined /> : <CaretDownOutlined />}
          onClick={(e) => {
            e.stopPropagation()
            onToggleCollapsed()
          }}
        />
        <div className="min-w-0 flex-1 cursor-pointer" onClick={onSelect}>
          <Text ellipsis className="text-[13px] font-medium">
            {project.name}
          </Text>
        </div>
        <div className="ml-auto flex items-center gap-1">
          <Text type="secondary" className="kb-count-text">{conversations.length}</Text>
          <Dropdown
            trigger={['click']}
            menu={{
              items: [
                { key: 'rename', icon: <EditOutlined />, label: '重命名项目' },
                { key: 'delete', icon: <DeleteOutlined />, label: '删除项目', danger: true },
              ],
              onClick: ({ key, domEvent }) => {
                domEvent.stopPropagation()
                if (key === 'rename') {
                  onRename()
                  return
                }
                Modal.confirm({
                  title: '确认删除这个项目吗？',
                  content: '项目下会话会保留，并自动移动到未分组。',
                  onOk: async () => {
                    await onDelete()
                  },
                })
              },
            }}
          >
            <Button
              type="text"
              size="small"
              icon={<MoreOutlined />}
              className="kb-side-menu-trigger"
              onClick={(e) => e.stopPropagation()}
            />
          </Dropdown>
        </div>
      </div>

      {!collapsed ? (
        <div className="kb-project-body px-1 pb-1 space-y-0.5">
          {conversations.length > 0 ? (
            conversations.map((conversation) => (
              <ConversationRow
                key={conversation.id}
                conversation={conversation}
                active={conversation.id === activeConvId}
                onOpen={() => onOpenConversation(conversation.id)}
                onRename={() => onRenameConversation(conversation)}
                onDelete={() => onDeleteConversation(conversation.id)}
              />
            ))
          ) : (
            <div className="px-2 py-1">
              <Text type="secondary" className="!text-xs">这个项目下暂无会话</Text>
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}

export function AppLayout({ children }: { children: ReactNode }) {
  const nav = useNavigate()
  const loc = useLocation()
  const projects = useChatStore((s) => s.projects)
  const activeProjectId = useChatStore((s) => s.activeProjectId)
  const projectConversations = useChatStore((s) => s.projectConversations)
  const rootConversations = useChatStore((s) => s.rootConversations)
  const activeConvId = useChatStore((s) => s.activeConvId)
  const loadSidebarData = useChatStore((s) => s.loadSidebarData)
  const selectProject = useChatStore((s) => s.selectProject)
  const createProject = useChatStore((s) => s.createProject)
  const renameProject = useChatStore((s) => s.renameProject)
  const deleteProject = useChatStore((s) => s.deleteProject)
  const selectConv = useChatStore((s) => s.selectConversation)
  const createConv = useChatStore((s) => s.createConversation)
  const renameConv = useChatStore((s) => s.renameConversation)
  const deleteConv = useChatStore((s) => s.deleteConversation)
  const moveConversation = useChatStore((s) => s.moveConversation)
  const theme = useSettingsStore((s) => s.theme)
  const toggleTheme = useSettingsStore((s) => s.toggleTheme)

  const [drawerOpen, setDrawerOpen] = useState(false)
  const [projectModalOpen, setProjectModalOpen] = useState(false)
  const [projectModalMode, setProjectModalMode] = useState<'create' | 'rename'>('create')
  const [editingProject, setEditingProject] = useState<Project | null>(null)
  const [projectName, setProjectName] = useState('')
  const [conversationModalOpen, setConversationModalOpen] = useState(false)
  const [editingConversation, setEditingConversation] = useState<Conversation | null>(null)
  const [conversationTitle, setConversationTitle] = useState('')
  const [keyword, setKeyword] = useState('')
  const [collapsedProjects, setCollapsedProjects] = useState<Record<string, boolean>>({})

  useEffect(() => {
    void loadSidebarData().catch((err: unknown) => {
      message.error(err instanceof Error ? err.message : '加载历史会话失败')
    })
  }, [loadSidebarData])

  const menuKey = loc.pathname === '/library' ? 'library' : 'chat'
  const normalizedKeyword = keyword.trim().toLowerCase()
  const projectMoveMenuItems = useMemo<MenuProps['items']>(
    () => projects.map((project) => ({ key: `move:${project.id}`, label: project.name })),
    [projects],
  )

  const sortedRootConversations = useMemo(
    () => [...rootConversations].sort((a, b) => b.updated_at - a.updated_at),
    [rootConversations],
  )

  const filteredRootConversations = useMemo(
    () => sortedRootConversations.filter((conversation) => matchesKeyword(conversation.title, normalizedKeyword)),
    [sortedRootConversations, normalizedKeyword],
  )

  const visibleProjects = useMemo(() => {
    return projects
      .map((project) => {
        const allConversations = [...(projectConversations[project.id] || [])].sort((a, b) => b.updated_at - a.updated_at)
        const filteredConversations = allConversations.filter((conversation) => matchesKeyword(conversation.title, normalizedKeyword))
        const show = !normalizedKeyword
          || matchesKeyword(project.name, normalizedKeyword)
          || filteredConversations.length > 0
        return {
          project,
          conversations: filteredConversations,
          show,
        }
      })
      .filter((item) => item.show)
  }, [projects, projectConversations, normalizedKeyword])

  const totalConversationCount = useMemo(
    () => rootConversations.length + Object.values(projectConversations).reduce((sum, items) => sum + items.length, 0),
    [rootConversations, projectConversations],
  )

  const visibleConversationCount = useMemo(
    () => filteredRootConversations.length + visibleProjects.reduce((sum, item) => sum + item.conversations.length, 0),
    [filteredRootConversations, visibleProjects],
  )
  const allConversationIds = useMemo(() => {
    const ids = new Set<string>()
    for (const item of rootConversations) ids.add(item.id)
    for (const group of Object.values(projectConversations)) {
      for (const item of group) ids.add(item.id)
    }
    return Array.from(ids)
  }, [projectConversations, rootConversations])

  useEffect(() => {
    const w = window as DebugWindow
    const base = w.__kbDebug || {}
    let running = false

    const runSwitchStress = async (opts: SwitchStressOptions = {}): Promise<SwitchStressResult> => {
      if (running) throw new Error('switch stress is already running')
      const rounds = Math.min(500, Math.max(1, Math.floor(Number(opts.rounds ?? 50))))
      const delayMs = Math.max(0, Math.floor(Number(opts.delayMs ?? 40)))
      const includeLibrary = opts.includeLibrary !== false
      const awaitSelect = opts.awaitSelect !== false
      const inputIds = Array.isArray(opts.convIds)
        ? opts.convIds.map((id) => String(id || '').trim()).filter(Boolean)
        : []
      const idPool = inputIds.length > 0 ? inputIds : allConversationIds
      if (idPool.length === 0) {
        throw new Error('no conversations available for stress run')
      }
      const sleep = (ms: number) => new Promise<void>((resolve) => {
        window.setTimeout(resolve, ms)
      })
      w.__kbSwitchPerf?.clear()
      running = true
      const startedAt = performance.now()
      try {
        for (let i = 0; i < rounds; i += 1) {
          const convId = idPool[i % idPool.length]
          if (awaitSelect) {
            await selectConv(convId)
          } else {
            void selectConv(convId)
          }
          if (includeLibrary) {
            nav('/library')
            if (delayMs > 0) await sleep(delayMs)
            nav('/')
          }
          if (delayMs > 0) await sleep(delayMs)
        }
      } finally {
        running = false
      }
      const elapsedMs = Number((performance.now() - startedAt).toFixed(2))
      return {
        rounds,
        delayMs,
        includeLibrary,
        awaitSelect,
        elapsedMs,
        summary: w.__kbSwitchPerf?.summary ? w.__kbSwitchPerf.summary() : null,
      }
    }

    const getSwitchPerf = () => (w.__kbSwitchPerf?.getLogs ? w.__kbSwitchPerf.getLogs() : [])
    const clearSwitchPerf = () => {
      w.__kbSwitchPerf?.clear()
    }

    w.__kbDebug = {
      ...base,
      runSwitchStress,
      getSwitchPerf,
      clearSwitchPerf,
    }

    return () => {
      const current = w.__kbDebug
      if (!current || current.runSwitchStress !== runSwitchStress) return
      const next: KbDebugApi = { ...current }
      delete next.runSwitchStress
      delete next.getSwitchPerf
      delete next.clearSwitchPerf
      if (Object.keys(next).length === 0) {
        delete w.__kbDebug
      } else {
        w.__kbDebug = next
      }
    }
  }, [allConversationIds, nav, selectConv])

  const toggleProjectCollapsed = (projectId: string) => {
    setCollapsedProjects((cur) => ({ ...cur, [projectId]: !cur[projectId] }))
  }

  const openCreateProject = () => {
    setProjectModalMode('create')
    setEditingProject(null)
    setProjectName('')
    setProjectModalOpen(true)
  }

  const openRenameProject = (project: Project) => {
    setProjectModalMode('rename')
    setEditingProject(project)
    setProjectName(project.name)
    setProjectModalOpen(true)
  }

  const submitProjectModal = async () => {
    const name = projectName.trim()
    if (!name) return
    if (projectModalMode === 'create') {
      await createProject(name)
    } else if (editingProject) {
      await renameProject(editingProject.id, name)
    }
    setProjectModalOpen(false)
    setEditingProject(null)
    setProjectName('')
  }

  const openRenameConversation = (conversation: Conversation) => {
    setEditingConversation(conversation)
    setConversationTitle(conversation.title)
    setConversationModalOpen(true)
  }

  const submitConversationModal = async () => {
    const title = conversationTitle.trim()
    if (!title || !editingConversation) return
    await renameConv(editingConversation.id, title)
    setConversationModalOpen(false)
    setEditingConversation(null)
    setConversationTitle('')
  }

  return (
    <Layout className="h-screen min-h-0 overflow-hidden">
      <Sider width={320} className="kb-sider flex flex-col overflow-hidden">
        <div className="kb-sider-brand px-2.5 pt-1.5 pb-1.5">
          <div className="kb-sider-team-logo-wrap">
            <img src="/team_logo.png" alt="Team logo" className="kb-sider-team-logo" />
          </div>
          <Text className="block text-[14px] font-semibold leading-tight tracking-tight">
            π-zaya · 你的知识库助理
          </Text>
          <Text type="secondary" className="!text-[11px]">
            当前会话 {normalizedKeyword ? `${visibleConversationCount}/${totalConversationCount}` : totalConversationCount} 条
          </Text>
        </div>

        <Menu
          mode="inline"
          selectedKeys={[menuKey]}
          className="kb-sider-menu !bg-transparent !border-none"
          items={[
            { key: 'chat', icon: <MessageOutlined />, label: '对话', onClick: () => nav('/') },
            { key: 'library', icon: <BookOutlined />, label: '文献管理', onClick: () => nav('/library') },
          ]}
        />

        <div className="kb-sider-toolbar px-2 pb-1 pt-0.5">
          <div className="kb-sider-main-actions flex gap-2">
            <Button
              type="primary"
              size="small"
              icon={<PlusOutlined />}
              className="kb-sider-main-action flex-1"
              onClick={async () => {
                await createConv()
                nav('/')
              }}
            >
              新建对话
            </Button>
            <Button
              size="small"
              icon={<FolderOpenOutlined />}
              className="kb-sider-main-action flex-1"
              onClick={openCreateProject}
            >
              新建项目
            </Button>
          </div>
          <div className="kb-sider-tool-buttons mt-1 flex items-center gap-1">
            <Tooltip title={theme === 'dark' ? '切换浅色模式' : '切换深色模式'}>
              <Button
                className="kb-sider-icon-btn"
                size="small"
                icon={theme === 'dark' ? <SunOutlined /> : <MoonOutlined />}
                onClick={toggleTheme}
              />
            </Tooltip>
            <Tooltip title="打开设置">
              <Button className="kb-sider-icon-btn" size="small" icon={<SettingOutlined />} onClick={() => setDrawerOpen(true)} />
            </Tooltip>
          </div>
          <div className="kb-sider-search-row mt-1">
            <Input
              className="kb-sider-search-input"
              allowClear
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              placeholder="搜索项目或会话"
              prefix={<SearchOutlined className="opacity-50" />}
            />
          </div>
        </div>

        <div className="kb-sider-scroll flex-1 overflow-y-auto px-1.5 pb-1.5 space-y-0.5">
          {visibleProjects.length > 0 ? (
            visibleProjects.map(({ project, conversations }) => (
              <ProjectSection
                key={project.id}
                project={project}
                selected={project.id === activeProjectId}
                conversations={conversations}
                activeConvId={activeConvId}
                collapsed={Boolean(collapsedProjects[project.id])}
                onToggleCollapsed={() => toggleProjectCollapsed(project.id)}
                onSelect={() => selectProject(project.id)}
                onOpenConversation={async (id) => {
                  nav('/')
                  void selectConv(id)
                }}
                onRenameConversation={openRenameConversation}
                onDeleteConversation={async (id) => {
                  await deleteConv(id)
                }}
                onRename={() => openRenameProject(project)}
                onDelete={async () => {
                  await deleteProject(project.id)
                }}
              />
            ))
          ) : (
            <div className="kb-sider-empty">
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<Text type="secondary">没有匹配的项目或会话</Text>} />
            </div>
          )}

          <div className={`kb-ungrouped-panel rounded-lg overflow-hidden ${activeProjectId === null ? 'is-active' : ''}`}>
            <button
              type="button"
              className="kb-ungrouped-head w-full flex items-center justify-between gap-2 text-left"
              onClick={() => selectProject(null)}
            >
              <Text className="text-[13px] font-medium">未分组对话</Text>
              <Text type="secondary" className="kb-count-text">{filteredRootConversations.length}</Text>
            </button>
            <div className="kb-root-conversations px-1 py-1 space-y-0.5">
              {filteredRootConversations.length > 0 ? (
                filteredRootConversations.map((conversation) => (
                  <ConversationRow
                    key={conversation.id}
                    conversation={conversation}
                    active={conversation.id === activeConvId}
                    onOpen={async () => {
                      nav('/')
                      void selectConv(conversation.id)
                    }}
                    onRename={() => openRenameConversation(conversation)}
                    onDelete={async () => {
                      await deleteConv(conversation.id)
                    }}
                    onMove={async (targetProjectId) => {
                      await moveConversation(conversation.id, targetProjectId)
                    }}
                    moveMenuItems={projectMoveMenuItems}
                  />
                ))
              ) : (
                <div className="px-2 py-1">
                  <Text type="secondary" className="!text-xs">暂无未分组会话</Text>
                </div>
              )}
            </div>
          </div>
        </div>

        <SettingsDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />

        <Modal
          title={projectModalMode === 'create' ? '新建项目' : '重命名项目'}
          open={projectModalOpen}
          onOk={() => { void submitProjectModal() }}
          onCancel={() => {
            setProjectModalOpen(false)
            setEditingProject(null)
            setProjectName('')
          }}
          okButtonProps={{ disabled: !projectName.trim() }}
        >
          <Input
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder="输入项目名称"
            onPressEnter={() => { void submitProjectModal() }}
          />
        </Modal>

        <Modal
          title="重命名会话"
          open={conversationModalOpen}
          onOk={() => { void submitConversationModal() }}
          onCancel={() => {
            setConversationModalOpen(false)
            setEditingConversation(null)
            setConversationTitle('')
          }}
          okButtonProps={{ disabled: !conversationTitle.trim() }}
        >
          <Input
            value={conversationTitle}
            onChange={(e) => setConversationTitle(e.target.value)}
            placeholder="输入会话标题"
            onPressEnter={() => { void submitConversationModal() }}
          />
        </Modal>
      </Sider>

      <Content className={`${loc.pathname === '/' ? 'overflow-hidden' : 'overflow-auto'} min-h-0 bg-[var(--bg)]`}>
        {children}
      </Content>
    </Layout>
  )
}
