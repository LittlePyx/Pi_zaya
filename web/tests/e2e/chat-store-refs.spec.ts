import { expect, test } from '@playwright/test'

test('chat refs enrichment polling cannot overwrite refs after active conversation changes', async ({ page }) => {
  await page.goto('/__message_list_test__')

  const result = await page.evaluate(async () => {
    const chatMod = await import('/src/api/chat.ts')
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    const chatApi = chatMod.chatApi as unknown as {
      getRefsWithMeta: (convId: string) => Promise<{
        data: Record<string, unknown>
        meta: Record<string, unknown>
      }>
    }

    let callCount = 0
    const callConvIds: string[] = []
    let resolveSecond: (() => void) | null = null

    chatApi.getRefsWithMeta = async (convId: string) => {
      callCount += 1
      callConvIds.push(convId)
      if (callCount === 1) {
        return {
          data: {
            cite_a: {
              payload_mode: 'fast',
              enrichment_pending: true,
            },
          },
          meta: { mode: 'initial' },
        }
      }
      if (callCount === 2) {
        await new Promise<void>((resolve) => {
          resolveSecond = resolve
        })
        return {
          data: {
            cite_a: {
              payload_mode: 'ready',
              enrichment_pending: false,
              hits: [{ meta: { title: 'Late refs from conversation A' } }],
            },
          },
          meta: { mode: 'poll' },
        }
      }
      return { data: {}, meta: { mode: 'extra' } }
    }

    useChatStore.setState({
      activeConvId: 'conv-a',
      refs: {},
      conversationCacheById: {},
    })

    await useChatStore.getState().selectConversation('conv-a')
    const deadline = Date.now() + 3000
    while (!resolveSecond && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 20))
    }
    if (!resolveSecond) throw new Error('second refs poll did not start')

    useChatStore.setState({
      activeConvId: 'conv-b',
      refs: {
        cite_b: {
          payload_mode: 'ready',
          hits: [{ meta: { title: 'Current refs from conversation B' } }],
        },
      },
    })
    resolveSecond()
    await new Promise((resolve) => setTimeout(resolve, 80))

    const state = useChatStore.getState()
    const refsPerf = (window as Window & {
      __kbRefsPerf?: { getLogs: () => Array<Record<string, unknown>> }
    }).__kbRefsPerf?.getLogs() || []
    return {
      activeConvId: state.activeConvId,
      refs: state.refs,
      callConvIds,
      phases: refsPerf.map((event) => String(event.phase || '')),
      lastAEvent: refsPerf.filter((event) => event.convId === 'conv-a').at(-1) || null,
    }
  })

  expect(result.activeConvId).toBe('conv-b')
  expect(Object.keys(result.refs)).toEqual(['cite_b'])
  expect(JSON.stringify(result.refs)).toContain('Current refs from conversation B')
  expect(JSON.stringify(result.refs)).not.toContain('Late refs from conversation A')
  expect(result.callConvIds).toEqual(['conv-a', 'conv-a'])
  expect(result.phases).toContain('poll_stale')
  expect(result.lastAEvent?.phase).toBe('poll_stale')
})

test('chat refs polling aborts the in-flight refs request when switching conversations', async ({ page }) => {
  await page.goto('/__message_list_test__')

  const result = await page.evaluate(async () => {
    const chatMod = await import('/src/api/chat.ts')
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    type RefsLoader = (
      convId: string,
      init?: RequestInit,
    ) => Promise<{ data: Record<string, unknown>; meta: Record<string, unknown> }>
    const chatApi = chatMod.chatApi as unknown as {
      getRefsWithMeta: RefsLoader
      getMessagesPage: (convId: string) => Promise<Record<string, unknown>>
    }

    let callCount = 0
    let secondSignalAborted = false
    let resolveSecond: (() => void) | null = null

    chatApi.getMessagesPage = async (convId: string) => ({
      messages: [{
        id: convId === 'conv-b' ? 202 : 101,
        role: 'assistant',
        content: `Loaded ${convId}`,
      }],
      has_more_before: false,
      oldest_loaded_id: convId === 'conv-b' ? 202 : 101,
    })

    chatApi.getRefsWithMeta = async (convId: string, init?: RequestInit) => {
      callCount += 1
      if (callCount === 1) {
        return {
          data: {
            cite_a: {
              payload_mode: 'fast',
              enrichment_pending: true,
            },
          },
          meta: { mode: 'initial' },
        }
      }
      if (callCount === 2) {
        return new Promise((resolve, reject) => {
          const signal = init?.signal
          const abort = () => {
            secondSignalAborted = true
            reject(new DOMException('Aborted', 'AbortError'))
          }
          if (signal?.aborted) {
            abort()
            return
          }
          signal?.addEventListener('abort', abort, { once: true })
          resolveSecond = () => {
            resolve({
              data: {
                cite_a: {
                  payload_mode: 'ready',
                  enrichment_pending: false,
                  hits: [{ meta: { title: 'Should not arrive after switch' } }],
                },
              },
              meta: { mode: 'late' },
            })
          }
        })
      }
      return { data: {}, meta: { mode: 'extra' } }
    }

    useChatStore.setState({
      activeConvId: 'conv-a',
      activeConversation: { id: 'conv-a', title: 'Conversation A', project_id: null, created_at: 1, updated_at: 1 },
      rootConversations: [
        { id: 'conv-a', title: 'Conversation A', project_id: null, created_at: 1, updated_at: 1 },
        { id: 'conv-b', title: 'Conversation B', project_id: null, created_at: 2, updated_at: 2 },
      ],
      refs: {},
      messages: [],
      conversationCacheById: {
        'conv-b': {
          messages: [],
          refs: {
            cite_b: {
              payload_mode: 'ready',
              hits: [{ meta: { title: 'Current refs from conversation B' } }],
            },
          },
          messagesHasMoreBefore: false,
          oldestLoadedMessageId: null,
          generation: null,
          sseController: null,
          uploadItems: [],
          pendingImages: [],
          cachedAt: Date.now(),
        },
      },
    })

    await useChatStore.getState().selectConversation('conv-a')
    const deadline = Date.now() + 3000
    while (!resolveSecond && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 20))
    }
    if (!resolveSecond) throw new Error('second refs poll did not start')

    await useChatStore.getState().selectConversation('conv-b')
    await new Promise((resolve) => setTimeout(resolve, 80))

    const state = useChatStore.getState()
    return {
      activeConvId: state.activeConvId,
      secondSignalAborted,
      refs: state.refs,
      callCount,
    }
  })

  expect(result.activeConvId).toBe('conv-b')
  expect(result.secondSignalAborted).toBe(true)
  expect(Object.keys(result.refs)).toEqual(['cite_b'])
  expect(JSON.stringify(result.refs)).toContain('Current refs from conversation B')
  expect(JSON.stringify(result.refs)).not.toContain('Should not arrive after switch')
  expect(result.callCount).toBe(2)
})
