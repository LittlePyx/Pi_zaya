import { expect, test } from '@playwright/test'

test('library ref-sync stream ignores stale events from a superseded stream', async ({ page }) => {
  await page.goto('/__message_list_test__')

  const result = await page.evaluate(async () => {
    const refsMod = await import('/src/api/references.ts')
    const { useLibraryStore } = await import('/src/stores/libraryStore.ts')
    const refsApi = refsMod.referencesApi as unknown as {
      streamSyncStatus: (
        onData: (data: Record<string, unknown>) => void,
        onDone: () => void,
        onError?: (err: unknown) => void,
      ) => AbortController
    }
    const streams: Array<{
      ctrl: AbortController
      onData: (data: Record<string, unknown>) => void
      onDone: () => void
      onError?: (err: unknown) => void
    }> = []

    refsApi.streamSyncStatus = (onData, onDone, onError) => {
      const ctrl = new AbortController()
      streams.push({ ctrl, onData, onDone, onError })
      return ctrl
    }

    useLibraryStore.setState({
      refSync: null,
      refSyncController: null,
    })

    useLibraryStore.getState().startRefSyncStream()
    useLibraryStore.getState().startRefSyncStream()

    const first = streams[0]
    const second = streams[1]
    second.onData({
      running: true,
      status: 'running',
      stage: 'new-stream',
      message: 'new stream is current',
      docs_done: 2,
      docs_total: 4,
      run_id: 22,
      refs_metadata_ready: 22,
    })

    first.onData({
      running: true,
      status: 'running',
      stage: 'old-stream',
      message: 'old stream arrived late',
      docs_done: 1,
      docs_total: 9,
      run_id: 11,
      refs_metadata_ready: 11,
    })
    first.onDone()

    const afterStale = useLibraryStore.getState()
    const controllerStillCurrent = afterStale.refSyncController === second.ctrl

    second.onDone()
    const afterCurrentDone = useLibraryStore.getState()

    return {
      streamCount: streams.length,
      firstAborted: first.ctrl.signal.aborted,
      afterStale: {
        controllerStillCurrent,
        running: afterStale.refSync?.running,
        stage: afterStale.refSync?.stage,
        message: afterStale.refSync?.message,
        runId: afterStale.refSync?.runId,
        docsDone: afterStale.refSync?.docsDone,
        statsReady: afterStale.refSync?.stats?.refs_metadata_ready,
      },
      afterCurrentDone: {
        controllerCleared: afterCurrentDone.refSyncController === null,
        running: afterCurrentDone.refSync?.running,
        stage: afterCurrentDone.refSync?.stage,
        runId: afterCurrentDone.refSync?.runId,
      },
    }
  })

  expect(result.streamCount).toBe(2)
  expect(result.firstAborted).toBe(true)
  expect(result.afterStale.controllerStillCurrent).toBe(true)
  expect(result.afterStale.running).toBe(true)
  expect(result.afterStale.stage).toBe('new-stream')
  expect(result.afterStale.message).toBe('new stream is current')
  expect(result.afterStale.runId).toBe(22)
  expect(result.afterStale.docsDone).toBe(2)
  expect(result.afterStale.statsReady).toBe(22)
  expect(result.afterCurrentDone.controllerCleared).toBe(true)
  expect(result.afterCurrentDone.running).toBe(false)
  expect(result.afterCurrentDone.stage).toBe('new-stream')
  expect(result.afterCurrentDone.runId).toBe(22)
})
