import { type ReactNode, useEffect, useState } from 'react'
import { Layout, Menu, Button, Typography, Popconfirm, Modal, Input, Empty } from 'antd'
import { Dropdown } from 'antd'
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
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { useChatStore } from '../../stores/chatStore'
import { useSettingsStore } from '../../stores/settingsStore'
import type { Conversation, Project } from '../../api/chat'
import { S } from '../../i18n/zh'
import { SettingsDrawer } from './SettingsDrawer'

const { Sider, Content } = Layout
const { Text } = Typography

function ConversationRow({
  conversation,
  active,
  moveMenuItems,
  onOpen,
  onRename,
  onMove,
  onDelete,
}: {
  conversation: Conversation
  active: boolean
  moveMenuItems: { key: string; label: string }[]
  onOpen: () => void
  onRename: () => void
  onMove: (target: string | null) => void
  onDelete: () => void
}) {
  return (
    <div
      className={`group flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs cursor-pointer ${
        active ? 'bg-[var(--msg-user-bg)]' : 'hover:bg-black/5 dark:hover:bg-white/5'
      }`}
      onClick={onOpen}
    >
      <MessageOutlined className="shrink-0 opacity-60" />
      <Text ellipsis className="!text-xs flex-1 min-w-0">
        {conversation.title}
      </Text>
      <Button
        type="text"
        size="small"
        icon={<EditOutlined />}
        className="!w-6 !h-6 shrink-0 opacity-0 group-hover:opacity-100"
        onClick={(e) => {
          e.stopPropagation()
          onRename()
        }}
      />
      <Dropdown
        menu={{
          items: moveMenuItems,
          onClick: ({ key, domEvent }) => {
            domEvent.stopPropagation()
            onMove(key === '__ungrouped__' ? null : key)
          },
        }}
        trigger={['click']}
      >
        <Button
          type="text"
          size="small"
          icon={<MoreOutlined />}
          className="!w-6 !h-6 shrink-0 opacity-0 group-hover:opacity-100"
          onClick={(e) => e.stopPropagation()}
        />
      </Dropdown>
      <Popconfirm
        title="删除这个会话？"
        onConfirm={(e) => {
          e?.stopPropagation()
          onDelete()
        }}
      >
        <Button
          type="text"
          size="small"
          icon={<DeleteOutlined />}
          className="!w-6 !h-6 shrink-0 opacity-0 group-hover:opacity-100"
          onClick={(e) => e.stopPropagation()}
        />
      </Popconfirm>
    </div>
  )
}

function ProjectSection({
  project,
  selected,
  conversations,
  activeConvId,
  moveMenuItems,
  onSelect,
  onOpenConversation,
  onRenameConversation,
  onMoveConversation,
  onDeleteConversation,
  onRename,
  onDelete,
}: {
  project: Project
  selected: boolean
  conversations: Conversation[]
  activeConvId: string | null
  moveMenuItems: { key: string; label: string }[]
  onSelect: () => void
  onOpenConversation: (id: string) => void
  onRenameConversation: (conversation: Conversation) => void
  onMoveConversation: (id: string, target: string | null) => void
  onDeleteConversation: (id: string) => void
  onRename: () => void
  onDelete: () => void
}) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--panel)]/50 overflow-hidden">
      <div
        className={`group flex items-center gap-2 px-3 py-2 cursor-pointer ${
          selected ? 'bg-[var(--msg-user-bg)]' : 'hover:bg-black/5 dark:hover:bg-white/5'
        }`}
        onClick={onSelect}
      >
        <FolderOpenOutlined className="opacity-70" />
        <Text ellipsis className="flex-1 min-w-0 text-sm">
          {project.name}
        </Text>
        <Text type="secondary" className="!text-[11px]">
          {conversations.length}
        </Text>
        <Button
          type="text"
          size="small"
          icon={<EditOutlined />}
          className="!w-6 !h-6 opacity-0 group-hover:opacity-100"
          onClick={(e) => {
            e.stopPropagation()
            onRename()
          }}
        />
        <Popconfirm
          title="删除这个项目？"
          description="项目下会话会保留，并移到未分组。"
          onConfirm={(e) => {
            e?.stopPropagation()
            onDelete()
          }}
        >
          <Button
            type="text"
            size="small"
            icon={<DeleteOutlined />}
            className="!w-6 !h-6 opacity-0 group-hover:opacity-100"
            onClick={(e) => e.stopPropagation()}
          />
        </Popconfirm>
      </div>
      <div className="px-2 pb-2 space-y-1">
        {conversations.length > 0 ? (
          conversations.map((conversation) => (
            <ConversationRow
              key={conversation.id}
              conversation={conversation}
              active={conversation.id === activeConvId}
              moveMenuItems={moveMenuItems}
              onOpen={() => onOpenConversation(conversation.id)}
              onRename={() => onRenameConversation(conversation)}
              onMove={(target) => onMoveConversation(conversation.id, target)}
              onDelete={() => onDeleteConversation(conversation.id)}
            />
          ))
        ) : (
          <div className="px-2 py-2">
            <Text type="secondary" className="!text-xs">
              这个项目里还没有会话
            </Text>
          </div>
        )}
      </div>
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

  useEffect(() => {
    loadSidebarData()
  }, [loadSidebarData])

  const menuKey = loc.pathname === '/library' ? 'library' : 'chat'
  const moveMenuItems = [
    { key: '__ungrouped__', label: '移到未分组' },
    ...projects.map((project) => ({ key: project.id, label: `移到 ${project.name}` })),
  ]

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
      <Sider width={300} className="!bg-[var(--panel)] border-r border-[var(--border)] flex flex-col overflow-hidden">
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
          <Button size="small" icon={<FolderOpenOutlined />} onClick={openCreateProject}>
            新建项目
          </Button>
          <Button
            size="small"
            type="primary"
            icon={<PlusOutlined />}
            onClick={async () => {
              await createConv()
              nav('/')
            }}
          >
            {S.new_chat}
          </Button>
        </div>

        <div className="px-3 pt-1 pb-2">
          <div
            className={`rounded-lg px-3 py-2 cursor-pointer border ${
              activeProjectId === null
                ? 'border-[var(--accent)] bg-[var(--msg-user-bg)]'
                : 'border-[var(--border)] hover:bg-black/5 dark:hover:bg-white/5'
            }`}
            onClick={() => selectProject(null)}
          >
            <Text className="text-sm">未分组会话</Text>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-3">
          {projects.length > 0 ? (
            projects.map((project) => (
              <ProjectSection
                key={project.id}
                project={project}
                selected={project.id === activeProjectId}
                conversations={projectConversations[project.id] || []}
                activeConvId={activeConvId}
                moveMenuItems={moveMenuItems.filter((item) => item.key !== project.id)}
                onSelect={() => selectProject(project.id)}
                onOpenConversation={async (id) => {
                  await selectConv(id)
                  nav('/')
                }}
                onRenameConversation={openRenameConversation}
                onMoveConversation={async (id, target) => {
                  await moveConversation(id, target)
                }}
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
            <div className="rounded-xl border border-dashed border-[var(--border)] px-4 py-5">
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={<Text type="secondary">还没有项目，先创建一个项目也可以继续用未分组会话</Text>}
              />
            </div>
          )}

          <div className="rounded-xl border border-[var(--border)] bg-[var(--panel)]/50 overflow-hidden">
            <div className="px-3 py-2 border-b border-[var(--border)]">
              <Text className="text-sm">未分组会话</Text>
            </div>
            <div className="px-2 py-2 space-y-1">
              {rootConversations.length > 0 ? (
                rootConversations.map((conversation) => (
                  <ConversationRow
                    key={conversation.id}
                    conversation={conversation}
                    active={conversation.id === activeConvId}
                    moveMenuItems={moveMenuItems.filter((item) => item.key !== '__ungrouped__')}
                    onOpen={async () => {
                      await selectConv(conversation.id)
                      nav('/')
                    }}
                    onRename={() => openRenameConversation(conversation)}
                    onMove={async (target) => {
                      await moveConversation(conversation.id, target)
                    }}
                    onDelete={async () => {
                      await deleteConv(conversation.id)
                    }}
                  />
                ))
              ) : (
                <div className="px-2 py-2">
                  <Text type="secondary" className="!text-xs">
                    暂时没有未分组会话
                  </Text>
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
