import { createElement, useLayoutEffect } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import type { CiteDetail } from '../components/chat/citationState'
import {
  useReaderCitationShelf,
  type ReaderCitationShelfController,
} from '../components/chat/useReaderCitationShelf'

export interface ReaderCitationShelfSmokeSnapshot {
  otherInShelf: boolean
  primaryInShelf: boolean
  size: number
}

export interface ReaderCitationShelfSmokeResult {
  events: string[]
  renderedText: string
  snapshots: ReaderCitationShelfSmokeSnapshot[]
}

function snapshotReaderCitationShelf(
  controller: ReaderCitationShelfController,
  primaryDetail: CiteDetail,
  otherDetail: CiteDetail,
): ReaderCitationShelfSmokeSnapshot {
  return {
    otherInShelf: controller.hasCitation(otherDetail),
    primaryInShelf: controller.hasCitation(primaryDetail),
    size: controller.shelfKeys.size,
  }
}

export function runReaderCitationShelfSmoke(): ReaderCitationShelfSmokeResult {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const events: string[] = []
  const snapshots: ReaderCitationShelfSmokeSnapshot[] = []
  let controller: ReaderCitationShelfController | null = null

  const primaryDetail = {
    anchor: 'reader-shelf-a',
    citeFmt: '',
    doi: '',
    isInpaper: true,
    num: 1,
    raw: '',
    sourceName: 'Current Paper',
    sourcePath: '/tmp/current.md',
    title: 'Primary Citation',
  } as CiteDetail
  const otherDetail = {
    anchor: 'reader-shelf-b',
    citeFmt: '',
    doi: '',
    isInpaper: true,
    num: 2,
    raw: '',
    sourceName: 'Current Paper',
    sourcePath: '/tmp/current.md',
    title: 'Other Citation',
  } as CiteDetail

  const readController = () => {
    if (!controller) throw new Error('reader citation shelf smoke did not mount')
    return controller
  }

  function Harness() {
    const shelf = useReaderCitationShelf({
      onAddCitationToShelf: (detail) => {
        events.push(`add:${detail.title || ''}`)
      },
    })
    useLayoutEffect(() => {
      controller = shelf
      snapshots.push(snapshotReaderCitationShelf(shelf, primaryDetail, otherDetail))
    })
    return createElement(
      'div',
      { id: 'reader-citation-shelf-smoke' },
      [
        String(shelf.hasCitation(primaryDetail)),
        String(shelf.hasCitation(otherDetail)),
        String(shelf.shelfKeys.size),
      ].join('|'),
    )
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  flushSync(() => {
    readController().addCitationToShelf(primaryDetail)
  })
  flushSync(() => {
    readController().addCitationToShelf(primaryDetail)
  })
  flushSync(() => {
    readController().addCitationToShelf(otherDetail)
  })

  const renderedText = host.textContent || ''
  root.unmount()
  host.remove()

  return {
    events,
    renderedText,
    snapshots,
  }
}
