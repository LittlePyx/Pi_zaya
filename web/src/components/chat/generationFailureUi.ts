import type { Message } from '../../api/chat'

const GENERATION_FAILURE_PATTERNS = [
  /调用模型失败/i,
  /回答任务未能启动/i,
  /回答连接(?:中断|失败)/i,
  /generation (?:could not be started|stream (?:failed|ended before completion))/i,
  /answer stream failed/i,
]

export function isGenerationFailureAnswer(text: string) {
  const clean = String(text || '').trim()
  if (!clean) return false
  if (GENERATION_FAILURE_PATTERNS.some((pattern) => pattern.test(clean))) return true
  return clean.length <= 500
    && /(?:connection|network|api key|authentication|unauthorized|forbidden|timeout|base_url|model)/i.test(clean)
    && /(?:error|failed|failure|invalid|missing|失败|错误|超时|未配置)/i.test(clean)
}

export function generationRetryPrompt(
  messages: Message[],
  assistantMessageId: number,
  linkedUserMessageId?: number,
) {
  const linkedId = Number(linkedUserMessageId || 0)
  if (linkedId > 0) {
    const linked = messages.find((item) => item.id === linkedId && item.role === 'user')
    if (linked?.content.trim()) return linked.content.trim()
  }

  const assistantIndex = messages.findIndex((item) => item.id === assistantMessageId)
  if (assistantIndex <= 0) return ''
  for (let index = assistantIndex - 1; index >= 0; index -= 1) {
    const candidate = messages[index]
    if (candidate.role === 'user' && candidate.content.trim()) return candidate.content.trim()
  }
  return ''
}
