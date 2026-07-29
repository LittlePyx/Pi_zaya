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
            '101': {
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
            '101': {
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
        '202': {
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
  expect(Object.keys(result.refs)).toEqual(['202'])
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
            '101': {
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
                '101': {
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
            '202': {
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
  expect(Object.keys(result.refs)).toEqual(['202'])
  expect(JSON.stringify(result.refs)).toContain('Current refs from conversation B')
  expect(JSON.stringify(result.refs)).not.toContain('Should not arrive after switch')
  expect(result.callCount).toBe(2)
})

test('newer same-conversation locale refs load supersedes an older pending response', async ({ page }) => {
  await page.goto('/__message_list_test__')

  const result = await page.evaluate(async () => {
    const chatMod = await import('/src/api/chat.ts')
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    const { useSettingsStore } = await import('/src/stores/settingsStore.ts')
    type RefsResult = { data: Record<string, unknown>; meta: Record<string, unknown> }
    const chatApi = chatMod.chatApi as unknown as {
      getMessagesPage: (convId: string) => Promise<Record<string, unknown>>
      getRefsWithMeta: (convId: string, init?: RequestInit) => Promise<RefsResult>
    }

    let messageCall = 0
    let refsCall = 0
    let firstRefsSignalAborted = false
    let resolveFirstRefs: ((value: RefsResult) => void) | null = null
    chatApi.getMessagesPage = async () => {
      messageCall += 1
      return {
        messages: [{ id: 10, role: 'assistant', content: messageCall === 1 ? 'English message' : '\u4e2d\u6587\u6d88\u606f' }],
        has_more_before: false,
        oldest_loaded_id: 10,
      }
    }
    chatApi.getRefsWithMeta = async (_convId: string, init?: RequestInit) => {
      refsCall += 1
      if (refsCall === 1) {
        init?.signal?.addEventListener('abort', () => { firstRefsSignalAborted = true }, { once: true })
        return new Promise<RefsResult>((resolve) => { resolveFirstRefs = resolve })
      }
      return {
        data: {
          '9': {
            payload_mode: 'ready',
            enrichment_pending: false,
            hits: [{ ui_meta: { summary_line: '\u4e2d\u6587\u65b0\u6458\u8981' } }],
          },
        },
        meta: { mode: 'zh-full' },
      }
    }

    useSettingsStore.setState({ uiLocale: 'en', refsCardLocale: 'en', localePreferencesRevision: 1 })
    useChatStore.setState({
      activeConvId: 'locale-conv',
      activeConversation: { id: 'locale-conv', title: 'Locale', project_id: null, created_at: 1, updated_at: 1 },
      messages: [],
      refs: {},
      conversationCacheById: {},
    })

    const firstRefresh = useChatStore.getState().refreshActiveConversationLocale()
    const deadline = Date.now() + 2_000
    while (!resolveFirstRefs && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 10))
    }
    if (!resolveFirstRefs) throw new Error('first refs request did not start')

    useSettingsStore.setState({ uiLocale: 'zh', refsCardLocale: 'zh', localePreferencesRevision: 2 })
    await useChatStore.getState().refreshActiveConversationLocale()
    resolveFirstRefs({
      data: {
        '9': {
          payload_mode: 'pending',
          enrichment_pending: true,
          hits: [{ ui_meta: { summary_line: 'Late English summary' } }],
        },
      },
      meta: { mode: 'en-late' },
    })
    await firstRefresh
    await new Promise((resolve) => setTimeout(resolve, 30))

    const state = useChatStore.getState()
    return {
      messages: state.messages.map((message) => message.content),
      refs: state.refs,
      messageCall,
      refsCall,
      firstRefsSignalAborted,
    }
  })

  expect(result.messages).toEqual(['\u4e2d\u6587\u6d88\u606f'])
  expect(JSON.stringify(result.refs)).toContain('\u4e2d\u6587\u65b0\u6458\u8981')
  expect(JSON.stringify(result.refs)).not.toContain('Late English summary')
  expect(result.messageCall).toBe(2)
  expect(result.refsCall).toBe(2)
  expect(result.firstRefsSignalAborted).toBe(true)
})

test('older locale message refresh cannot overwrite a newer revision', async ({ page }) => {
  await page.goto('/__message_list_test__')

  const result = await page.evaluate(async () => {
    const chatMod = await import('/src/api/chat.ts')
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    const { useSettingsStore } = await import('/src/stores/settingsStore.ts')
    type MessagePageResult = Record<string, unknown>
    const chatApi = chatMod.chatApi as unknown as {
      getMessagesPage: (convId: string) => Promise<MessagePageResult>
      getRefsWithMeta: () => Promise<{ data: Record<string, unknown>; meta: Record<string, unknown> }>
    }
    let pageCall = 0
    let resolveOldPage: ((value: MessagePageResult) => void) | null = null
    chatApi.getMessagesPage = async () => {
      pageCall += 1
      if (pageCall === 1) {
        return new Promise<MessagePageResult>((resolve) => { resolveOldPage = resolve })
      }
      return {
        messages: [{ id: 20, role: 'assistant', content: '\u6700\u65b0\u4e2d\u6587\u6e32\u67d3' }],
        has_more_before: false,
        oldest_loaded_id: 20,
      }
    }
    chatApi.getRefsWithMeta = async () => ({ data: {}, meta: { mode: 'empty' } })

    useSettingsStore.setState({ uiLocale: 'en', refsCardLocale: 'en', localePreferencesRevision: 10 })
    useChatStore.setState({
      activeConvId: 'message-locale-conv',
      activeConversation: { id: 'message-locale-conv', title: 'Locale', project_id: null, created_at: 1, updated_at: 1 },
      messages: [],
      refs: {},
      conversationCacheById: {},
    })
    const oldRefresh = useChatStore.getState().refreshActiveConversationLocale()
    const deadline = Date.now() + 2_000
    while (!resolveOldPage && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 10))
    }
    if (!resolveOldPage) throw new Error('old locale page request did not start')

    useSettingsStore.setState({ uiLocale: 'zh', refsCardLocale: 'zh', localePreferencesRevision: 11 })
    await useChatStore.getState().refreshActiveConversationLocale()
    resolveOldPage({
      messages: [{ id: 20, role: 'assistant', content: 'Late old English rendering' }],
      has_more_before: false,
      oldest_loaded_id: 20,
    })
    await oldRefresh
    await new Promise((resolve) => setTimeout(resolve, 20))

    const state = useChatStore.getState()
    return {
      messages: state.messages.map((message) => message.content),
      cachedMessages: state.conversationCacheById['message-locale-conv']?.messages.map((message) => message.content) || [],
      pageCall,
    }
  })

  expect(result.messages).toEqual(['\u6700\u65b0\u4e2d\u6587\u6e32\u67d3'])
  expect(result.cachedMessages).toEqual(['\u6700\u65b0\u4e2d\u6587\u6e32\u67d3'])
  expect(result.pageCall).toBe(2)
})
