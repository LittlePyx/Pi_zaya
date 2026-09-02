import { expect, test } from '@playwright/test'

test('paper guide reuses the newest unarchived conversation for the same normalized source', async ({ page }) => {
  await page.goto('/__message_list_test__')

  const result = await page.evaluate(async () => {
    const { findReusablePaperGuideConversation } = await import('/src/stores/chatStore.ts')
    const conversations = [
      {
        id: 'older-same-project',
        title: 'Older',
        mode: 'paper_guide' as const,
        bound_source_path: 'F:\\papers\\same-paper.md',
        project_id: 'project-a',
        created_at: 1,
        updated_at: 20,
      },
      {
        id: 'newer-other-project',
        title: 'Newer elsewhere',
        mode: 'paper_guide' as const,
        bound_source_path: 'f:/papers/same-paper.md',
        project_id: 'project-b',
        created_at: 2,
        updated_at: 40,
      },
      {
        id: 'archived-same-project',
        title: 'Archived',
        mode: 'paper_guide' as const,
        bound_source_path: 'F:/papers/same-paper.md',
        project_id: 'project-a',
        archived: true,
        created_at: 3,
        updated_at: 60,
      },
      {
        id: 'normal-chat',
        title: 'Normal',
        mode: 'normal' as const,
        bound_source_path: 'F:/papers/same-paper.md',
        project_id: 'project-a',
        created_at: 4,
        updated_at: 80,
      },
    ]
    return {
      preferredProject: findReusablePaperGuideConversation(
        conversations,
        'file:///F:/papers/./same-paper.md?reader',
        'project-a',
      )?.id,
      noProjectMatch: findReusablePaperGuideConversation(
        conversations,
        'F:/papers/same-paper.md',
        'project-c',
      )?.id,
      missing: findReusablePaperGuideConversation(
        conversations,
        'F:/papers/other-paper.md',
        'project-a',
      )?.id || null,
    }
  })

  expect(result).toEqual({
    preferredProject: 'older-same-project',
    noProjectMatch: 'newer-other-project',
    missing: null,
  })
})

test('create paper guide selects an existing session without creating another conversation', async ({ page }) => {
  await page.goto('/__message_list_test__')

  const result = await page.evaluate(async () => {
    const chatMod = await import('/src/api/chat.ts')
    const { useChatStore } = await import('/src/stores/chatStore.ts')
    const chatApi = chatMod.chatApi as unknown as {
      createConversation: (...args: unknown[]) => Promise<{ id: string }>
      getMessagesPage: (convId: string) => Promise<Record<string, unknown>>
      getConversation: (convId: string) => Promise<Record<string, unknown>>
      getRefsWithMeta: () => Promise<{ data: Record<string, unknown>; meta: Record<string, unknown> }>
    }
    let createCalls = 0
    chatApi.createConversation = async () => {
      createCalls += 1
      return { id: 'unexpected-new-session' }
    }
    chatApi.getMessagesPage = async () => ({
      messages: [],
      has_more_before: false,
      oldest_loaded_id: null,
    })
    chatApi.getConversation = async () => ({
      id: 'existing-guide',
      title: 'Existing guide',
      mode: 'paper_guide',
      bound_source_path: 'F:/papers/same-paper.md',
      created_at: 1,
      updated_at: 2,
    })
    chatApi.getRefsWithMeta = async () => ({ data: {}, meta: {} })

    useChatStore.setState({
      activeConvId: null,
      activeConversation: null,
      rootConversations: [{
        id: 'existing-guide',
        title: 'Existing guide',
        mode: 'paper_guide',
        bound_source_path: 'F:/papers/same-paper.md',
        project_id: null,
        created_at: 1,
        updated_at: 2,
      }],
      projectConversations: {},
      conversationCacheById: {},
      messages: [],
      refs: {},
    })

    const selectedId = await useChatStore.getState().createPaperGuideConversation({
      sourcePath: 'f:\\papers\\same-paper.md',
      sourceName: 'same-paper',
    })
    return {
      selectedId,
      createCalls,
      activeConvId: useChatStore.getState().activeConvId,
    }
  })

  expect(result).toEqual({
    selectedId: 'existing-guide',
    createCalls: 0,
    activeConvId: 'existing-guide',
  })
})
