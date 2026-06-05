import { type KeyboardEvent, type ReactNode, useCallback, useEffect, useMemo, useState } from 'react'
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
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  ApiOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { useT } from '../../i18n'
import { useChatStore } from '../../stores/chatStore'
import { useSettingsStore } from '../../stores/settingsStore'
import type { Conversation, Project } from '../../api/chat'
import { READER_SESSION_NAV_CHANNEL } from '../chat/reader/readerTypes'
import { SettingsDrawer } from './SettingsDrawer'

const { Sider, Content } = Layout
const { Text } = Typography

const SIDEBAR_WIDTH = 296
const SIDEBAR_COLLAPSED_WIDTH = 68
const OPEN_SETTINGS_EVENT = 'kb:open-settings'
const LINKED_CONVERSATION_QUERY_KEYS = ['conversation', 'conversation_id', 'conv'] as const

function linkedConversationIdFromSearch(search: string) {
  const params = new URLSearchParams(search)
  for (const key of LINKED_CONVERSATION_QUERY_KEYS) {
    const value = String(params.get(key) || '').trim()
    if (value) return value
  }
  return ''
}

function clearLinkedConversationSearch(search: string) {
  const params = new URLSearchParams(search)
  for (const key of LINKED_CONVERSATION_QUERY_KEYS) params.delete(key)
  const next = params.toString()
  return next ? `?${next}` : ''
}

function formatRelativeTime(ts: number | undefined, txt: { just_now: string; minutes_ago: string; hours_ago: string; days_ago: string }) {
  if (!ts) return ''
  const now = Date.now() / 1000
  const diff = Math.max(0, Math.floor(now - ts))
  if (diff < 60) return txt.just_now
  if (diff < 3600) return txt.minutes_ago.replace('{n}', String(Math.floor(diff / 60)))
  if (diff < 86400) return txt.hours_ago.replace('{n}', String(Math.floor(diff / 3600)))
  if (diff < 86400 * 7) return txt.days_ago.replace('{n}', String(Math.floor(diff / 86400)))
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
  const S = useT()
  const openFromKeyboard = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    onOpen()
  }
  const menuItems: MenuProps['items'] = [
    { key: 'rename', icon: <EditOutlined />, label: S.rename },
    ...(moveMenuItems && moveMenuItems.length > 0
      ? [{ key: 'move', icon: <FolderOpenOutlined />, label: S.move_to, children: moveMenuItems }]
      : []),
    { type: 'divider' },
    { key: 'delete', icon: <DeleteOutlined />, label: S.delete, danger: true },
  ]

  return (
    <div
      className={`kb-conv-row flex items-center gap-1 rounded-lg px-1.5 py-0.5 text-xs cursor-pointer ${
        active ? 'is-active' : ''
      }`}
      onClick={onOpen}
      onKeyDown={openFromKeyboard}
      role="button"
      tabIndex={0}
      aria-current={active ? 'page' : undefined}
    >
      <MessageOutlined className="shrink-0 opacity-60" />
      <div className="kb-conv-meta min-w-0 flex-1">
        <div className="kb-conv-title" title={conversation.title}>
          <span className="kb-conv-title-text">{conversation.title}</span>
        </div>
        <span className="kb-conv-time">{formatRelativeTime(conversation.updated_at, S) || ' '}</span>
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
              title: S.confirm_delete_conversation,
              okText: S.confirm_ok,
              cancelText: S.confirm_cancel,
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
          aria-label={S.conversation_actions || 'Conversation actions'}
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
  const S = useT()
  return (
    <div className={`kb-project-card rounded-lg overflow-hidden ${selected ? 'is-active' : ''}`}>
      <div className="kb-project-head flex items-center gap-1 px-2 py-1">
        <Button
          type="text"
          size="small"
          className="!w-6 !h-6 !min-w-0"
          icon={collapsed ? <CaretRightOutlined /> : <CaretDownOutlined />}
          aria-label={collapsed ? (S.expand_project || 'Expand project') : (S.collapse_project || 'Collapse project')}
          onClick={(e) => {
            e.stopPropagation()
            onToggleCollapsed()
          }}
        />
        <button type="button" className="kb-project-title-btn min-w-0 flex-1" onClick={onSelect}>
          <Text ellipsis className="text-[13px] font-medium">
            {project.name}
          </Text>
        </button>
        <div className="ml-auto flex items-center gap-1">
          <Text type="secondary" className="kb-count-text">{conversations.length}</Text>
          <Dropdown
            trigger={['click']}
            menu={{
              items: [
                { key: 'rename', icon: <EditOutlined />, label: S.rename_project },
                { key: 'delete', icon: <DeleteOutlined />, label: S.delete_project, danger: true },
              ],
              onClick: ({ key, domEvent }) => {
                domEvent.stopPropagation()
                if (key === 'rename') {
                  onRename()
                  return
                }
                Modal.confirm({
                  title: S.confirm_delete_project,
                  content: S.delete_project_warning,
                  okText: S.confirm_ok,
                  cancelText: S.confirm_cancel,
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
              aria-label={S.project_actions || 'Project actions'}
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
              <Text type="secondary" className="!text-xs">{S.no_conversations_in_project}</Text>
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}

export function AppLayout({ children }: { children: ReactNode }) {
  const S = useT()
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
  const sidebarCollapsed = useSettingsStore((s) => s.sidebarCollapsed)
  const updateSettings = useSettingsStore((s) => s.update)
  const llmReadiness = useSettingsStore((s) => s.llmReadiness)
  const hasTextApiKey = useSettingsStore((s) => s.hasTextApiKey)
  const visionUsesTextFallback = useSettingsStore((s) => s.visionUsesTextFallback)

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
    const openSettings = () => setDrawerOpen(true)
    window.addEventListener(OPEN_SETTINGS_EVENT, openSettings)
    return () => window.removeEventListener(OPEN_SETTINGS_EVENT, openSettings)
  }, [])

  useEffect(() => {
    void loadSidebarData().catch((err: unknown) => {
      message.error(err instanceof Error ? err.message : S.sidebar_load_failed)
    })
  }, [S.sidebar_load_failed, loadSidebarData])

  useEffect(() => {
    if (loc.pathname !== '/') return
    const linkedConversationId = linkedConversationIdFromSearch(loc.search)
    if (!linkedConversationId) return
    void selectConv(linkedConversationId)
    const nextSearch = clearLinkedConversationSearch(loc.search)
    if (nextSearch !== loc.search) {
      nav({ pathname: '/', search: nextSearch }, { replace: true })
    }
  }, [loc.pathname, loc.search, nav, selectConv])

  const openLinkedConversation = useCallback((conversationId: string) => {
    const cid = String(conversationId || '').trim()
    if (!cid) return
    if (loc.pathname !== '/') {
      nav('/', { replace: false })
    }
    void selectConv(cid)
    try {
      window.focus()
    } catch {
      // Browser focus can be denied; selecting the conversation is enough.
    }
  }, [loc.pathname, nav, selectConv])

  useEffect(() => {
    const handlePayload = (raw: unknown) => {
      const data = (raw && typeof raw === 'object') ? raw as Record<string, unknown> : {}
      if (String(data.type || '') !== 'reader-return-to-conversation') return
      openLinkedConversation(String(data.conversationId || ''))
    }
    const handleWindowMessage = (event: MessageEvent) => {
      if (event.origin && event.origin !== window.location.origin) return
      handlePayload(event.data)
    }
    window.addEventListener('message', handleWindowMessage)
    let channel: BroadcastChannel | null = null
    if (typeof BroadcastChannel !== 'undefined') {
      channel = new BroadcastChannel(READER_SESSION_NAV_CHANNEL)
      channel.onmessage = (event) => handlePayload(event.data)
    }
    return () => {
      window.removeEventListener('message', handleWindowMessage)
      channel?.close()
    }
  }, [openLinkedConversation])

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

  const toggleProjectCollapsed = useCallback((projectId: string) => {
    setCollapsedProjects((cur) => ({ ...cur, [projectId]: !cur[projectId] }))
  }, [])

  const toggleSidebarCollapsed = useCallback(() => {
    void updateSettings({ sidebarCollapsed: !sidebarCollapsed })
  }, [sidebarCollapsed, updateSettings])

  const openConversation = useCallback((conversationId: string) => {
    nav('/')
    void selectConv(conversationId)
  }, [nav, selectConv])

  const startNewConversation = useCallback(async () => {
    await createConv()
    nav('/')
  }, [createConv, nav])

  const removeConversation = useCallback(async (conversationId: string) => {
    await deleteConv(conversationId)
  }, [deleteConv])

  const chooseProject = useCallback((projectId: string | null) => {
    selectProject(projectId)
  }, [selectProject])

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

  const sidebarToggleLabel = sidebarCollapsed
    ? (S.expand_sidebar || 'Expand sidebar')
    : (S.collapse_sidebar || 'Collapse sidebar')
  const connectionStatus = llmReadiness?.overall.status || (!hasTextApiKey ? 'error' : (visionUsesTextFallback ? 'warning' : 'ok'))
  const connectionLabel = connectionStatus === 'error'
    ? S.connection_status_error
    : connectionStatus === 'warning'
      ? S.connection_status_warning
      : S.connection_status_ok

  return (
    <Layout className="h-screen min-h-0 overflow-hidden">
      <Sider
        width={SIDEBAR_WIDTH}
        collapsedWidth={SIDEBAR_COLLAPSED_WIDTH}
        collapsed={sidebarCollapsed}
        trigger={null}
        className={`kb-sider flex flex-col overflow-hidden ${sidebarCollapsed ? 'is-collapsed' : ''}`}
      >
        <div className="kb-sider-brand px-2.5 pt-1.5 pb-1.5">
          <div className="kb-sider-brand-row">
            <div className="kb-sider-team-logo-wrap">
              <img src="/team_logo.png" alt="Team logo" className="kb-sider-team-logo" />
            </div>
            <Tooltip title={sidebarToggleLabel} placement={sidebarCollapsed ? 'right' : 'bottom'}>
              <Button
                className="kb-sider-collapse-btn"
                size="small"
                type="text"
                icon={sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                aria-label={sidebarToggleLabel}
                onClick={toggleSidebarCollapsed}
              />
            </Tooltip>
          </div>
          <div className="kb-sider-brand-copy">
            <Text className="block text-[14px] font-semibold leading-tight tracking-tight">
              {S.brand_subtitle}
            </Text>
            <Text type="secondary" className="!text-[11px]">
              {S.conversation_count.replace('{n}', String(normalizedKeyword ? `${visibleConversationCount}/${totalConversationCount}` : totalConversationCount))}
            </Text>
          </div>
        </div>

        <Menu
          mode="inline"
          inlineCollapsed={sidebarCollapsed}
          selectedKeys={[menuKey]}
          className="kb-sider-menu !bg-transparent !border-none"
          items={[
            { key: 'chat', icon: <MessageOutlined />, label: S.chat, onClick: () => nav('/') },
            { key: 'library', icon: <BookOutlined />, label: S.page_library, onClick: () => nav('/library') },
          ]}
        />

        <div className="kb-sider-toolbar px-2 pb-1 pt-0.5">
          <div className="kb-sider-main-actions flex gap-2">
            <Tooltip title={sidebarCollapsed ? S.new_chat : ''} placement="right">
              <Button
                type="primary"
                size="small"
                icon={<PlusOutlined />}
                aria-label={S.new_chat}
                className="kb-sider-main-action flex-1"
                onClick={() => { void startNewConversation() }}
              >
                {S.new_chat}
              </Button>
            </Tooltip>
            <Tooltip title={sidebarCollapsed ? S.new_project : ''} placement="right">
              <Button
                size="small"
                icon={<FolderOpenOutlined />}
                aria-label={S.new_project}
                className="kb-sider-main-action flex-1"
                onClick={openCreateProject}
              >
                {S.new_project}
              </Button>
            </Tooltip>
          </div>
          <div className="kb-sider-tool-buttons mt-1 flex items-center gap-1">
            <Tooltip title={theme === 'dark' ? S.switch_light_mode : S.switch_dark_mode} placement={sidebarCollapsed ? 'right' : 'top'}>
              <Button
                className="kb-sider-icon-btn"
                size="small"
                icon={theme === 'dark' ? <SunOutlined /> : <MoonOutlined />}
                aria-label={theme === 'dark' ? S.switch_light_mode : S.switch_dark_mode}
                onClick={toggleTheme}
              />
            </Tooltip>
            <Tooltip title={S.open_settings} placement={sidebarCollapsed ? 'right' : 'top'}>
              <Button
                className="kb-sider-icon-btn"
                size="small"
                icon={<SettingOutlined />}
                aria-label={S.open_settings}
                onClick={() => setDrawerOpen(true)}
              />
            </Tooltip>
            <Tooltip title={connectionLabel} placement={sidebarCollapsed ? 'right' : 'top'}>
              <Button
                className={`kb-sider-icon-btn kb-sider-connection-btn is-${connectionStatus}`}
                size="small"
                icon={<ApiOutlined />}
                aria-label={connectionLabel}
                onClick={() => setDrawerOpen(true)}
              />
            </Tooltip>
          </div>
          <div className="kb-sider-search-row mt-1">
            <Input
              className="kb-sider-search-input"
              allowClear
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              placeholder={S.search_project_or_conversation}
              aria-label={S.search_project_or_conversation}
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
                onSelect={() => chooseProject(project.id)}
                onOpenConversation={openConversation}
                onRenameConversation={openRenameConversation}
                onDeleteConversation={removeConversation}
                onRename={() => openRenameProject(project)}
                onDelete={async () => {
                  await deleteProject(project.id)
                }}
              />
            ))
          ) : (
            <div className="kb-sider-empty">
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<Text type="secondary">{S.no_matching_items}</Text>} />
            </div>
          )}

          <div className={`kb-ungrouped-panel rounded-lg overflow-hidden ${activeProjectId === null ? 'is-active' : ''}`}>
            <button
              type="button"
              className="kb-ungrouped-head w-full flex items-center justify-between gap-2 text-left"
              onClick={() => chooseProject(null)}
            >
              <Text className="text-[13px] font-medium">{S.ungrouped_conversations}</Text>
              <Text type="secondary" className="kb-count-text">{filteredRootConversations.length}</Text>
            </button>
            <div className="kb-root-conversations px-1 py-1 space-y-0.5">
              {filteredRootConversations.length > 0 ? (
                filteredRootConversations.map((conversation) => (
                  <ConversationRow
                    key={conversation.id}
                    conversation={conversation}
                    active={conversation.id === activeConvId}
                    onOpen={() => openConversation(conversation.id)}
                    onRename={() => openRenameConversation(conversation)}
                    onDelete={() => removeConversation(conversation.id)}
                    onMove={async (targetProjectId) => {
                      await moveConversation(conversation.id, targetProjectId)
                    }}
                    moveMenuItems={projectMoveMenuItems}
                  />
                ))
              ) : (
                <div className="px-2 py-1">
                  <Text type="secondary" className="!text-xs">{S.no_ungrouped_conversations}</Text>
                </div>
              )}
            </div>
          </div>
        </div>

        <SettingsDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />

        <Modal
          title={projectModalMode === 'create' ? S.new_project : S.rename_project}
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
            placeholder={S.input_project_name_placeholder}
            onPressEnter={() => { void submitProjectModal() }}
          />
        </Modal>

        <Modal
          title={S.rename_conversation_title}
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
            placeholder={S.input_conversation_title_placeholder}
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
