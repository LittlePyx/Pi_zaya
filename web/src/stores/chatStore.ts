import { create } from 'zustand'
import { chatApi, type Conversation, type Message } from '../api/chat'
import { api } from '../api/client'

interface GenerationState {
  sessionId: string
  taskId: string
  stage: string
  partial: string
  done: boolean
}

interface ChatState {
  conversations: Conversation[]
  activeConvId: string | null
  messages: Message[]
  refs: Record<string, unknown>
  generation: GenerationState | null
  sseController: AbortController | null

  loadConversations: () => Promise<void>
  selectConversation: (id: string) => Promise<void>
  createConversation: () => Promise<string>
  deleteConversation: (id: string) => Promise<void>
  sendMessage: (prompt: string, opts: {
    topK: number; temperature: number; maxTokens: number; deepRead: boolean
  }) => Promise<void>
  cancelGeneration: () => void
  clearGeneration: () => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  activeConvId: null,
  messages: [],
  refs: {},
  generation: null,
  sseController: null,

  loadConversations: async () => {
    const list = await chatApi.listConversations()
    set({ conversations: list })
  },

  selectConversation: async (id) => {
    set({ activeConvId: id, generation: null })
    const [msgs, refs] = await Promise.all([
      chatApi.getMessages(id),
      chatApi.getRefs(id),
    ])
    set({ messages: msgs, refs })
  },

  createConversation: async () => {
    const { id } = await chatApi.createConversation()
    await get().loadConversations()
    set({ activeConvId: id, messages: [], refs: {}, generation: null })
    return id
  },

  deleteConversation: async (id) => {
    await chatApi.deleteConversation(id)
    const s = get()
    if (s.activeConvId === id) {
      set({ activeConvId: null, messages: [], refs: {} })
    }
    await get().loadConversations()
  },

  sendMessage: async (prompt, opts) => {
    let convId = get().activeConvId
    if (!convId) {
      convId = await get().createConversation()
    }

    const res = await api.post<{
      session_id: string; task_id: string;
      user_msg_id: number; assistant_msg_id: number
    }>('/api/generate', {
      conv_id: convId, prompt,
      top_k: opts.topK, temperature: opts.temperature,
      max_tokens: opts.maxTokens, deep_read: opts.deepRead,
    })

    // Add user message to local state immediately
    set(s => ({
      messages: [...s.messages, {
        id: res.user_msg_id, role: 'user', content: prompt, created_at: Date.now() / 1000,
      }],
      generation: {
        sessionId: res.session_id, taskId: res.task_id,
        stage: 'starting', partial: '', done: false,
      },
    }))

    // Start SSE
    const ctrl = new AbortController()
    set({ sseController: ctrl })

    try {
      const sseRes = await fetch(`/api/generate/${res.session_id}/stream`, {
        signal: ctrl.signal,
      })
      const reader = sseRes.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            set({
              generation: {
                sessionId: res.session_id, taskId: res.task_id,
                stage: data.stage || '', partial: data.partial || '',
                done: !!data.done,
              },
            })
            if (data.done) {
              // Reload messages from server
              const msgs = await chatApi.getMessages(convId!)
              const refs = await chatApi.getRefs(convId!)
              set({ messages: msgs, refs, generation: null })
              await get().loadConversations()
              return
            }
          } catch { /* skip bad JSON */ }
        }
      }
    } catch {
      // aborted or network error
    } finally {
      set({ sseController: null })
    }
  },

  cancelGeneration: () => {
    const s = get()
    if (s.generation && s.sseController) {
      s.sseController.abort()
      api.post(`/api/generate/${s.generation.sessionId}/cancel?task_id=${s.generation.taskId}`)
        .catch(() => {})
    }
    set({ generation: null, sseController: null })
  },

  clearGeneration: () => set({ generation: null }),
}))
