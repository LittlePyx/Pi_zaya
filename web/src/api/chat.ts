import { api } from './client'

export interface Conversation {
  id: string
  title: string
  created_at: number
  updated_at: number
}

export interface Message {
  id: number
  role: string
  content: string
  created_at: number
}

export const chatApi = {
  listConversations: (limit = 50) =>
    api.get<Conversation[]>(`/api/conversations?limit=${limit}`),
  createConversation: (title = '新对话') =>
    api.post<{ id: string }>('/api/conversations', { title }),
  deleteConversation: (id: string) =>
    api.delete(`/api/conversations/${id}`),
  getMessages: (convId: string) =>
    api.get<Message[]>(`/api/conversations/${convId}/messages`),
  appendMessage: (convId: string, role: string, content: string) =>
    api.post<{ id: number }>(`/api/conversations/${convId}/messages`, { role, content }),
  getRefs: (convId: string) =>
    api.get<Record<string, unknown>>(`/api/conversations/${convId}/refs`),
  updateTitle: (convId: string, title: string) =>
    api.patch(`/api/conversations/${convId}/title`, { title }),
}
