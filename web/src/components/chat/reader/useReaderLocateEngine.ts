import { useEffect, useRef, useState, type RefObject } from 'react'
import type { ReaderDocBlock } from '../../../api/references'
import type { ReaderLocateCandidate } from './readerTypes'
import {
  bindVisibleEquationAnchors,
  buildHighlightQueries,
  clearReaderFocusClasses,
  clearReaderInlineHits,
  closestReadableBlock,
  equationNumberMatchScore,
  extractEquationNumbers,
  extractFigureNumbers,
  formulaOverlapScore,
  hasFormulaSignal,
  headingCandidates,
  headingMatchScore,
  highlightExactTextInContainer,
  nearbyReadableBlocks,
  normalizeText,
  orderedEquationReaderBlocks,
  resolveDirectTargetNode,
  resolveInlineFormulaTarget,
  resolveRelatedTargetNodes,
  resolveStickyHighlightTarget,
  scrollReaderTargetIntoView,
  snippetMatchScore,
  snippetProbeText,
  tokenizeText,
  visibleEquationBlocks,
} from './readerDomUtils'

interface StickyLocateHighlight {
  blockId: string
  anchorId: string
  anchorKind: string
  anchorNumber: number
  headingPath: string
  highlightSeed: string
  highlightQueries: string[]
  relatedBlockIds: string[]
  strictLocate: boolean
}

interface UseReaderLocateEngineArgs {
  open: boolean
  drawerReady: boolean
  markdown: string
  locateRequestId: number
  sourcePath: string
  strictLocate: boolean
  contentRef: RefObject<HTMLDivElement | null>
  readerBlocks: ReaderDocBlock[]
  alternatives: ReaderLocateCandidate[]
  relatedBlockIds: string[]
  activeAltIndex: number
  setActiveAltIndex: (idx: number) => void
  activeHeadingPath: string
  activeFocusSnippet: string
  activeHighlightSnippet: string
  activeAnchorId: string
  activeBlockId: string
  activeAnchorKind: string
  activeAnchorNumber: number
  expectsEquationBinding: boolean
}

export function useReaderLocateEngine({
  open,
  drawerReady,
  markdown,
  locateRequestId,
  sourcePath,
  strictLocate,
  contentRef,
  readerBlocks,
  alternatives,
  relatedBlockIds,
  activeAltIndex,
  setActiveAltIndex,
  activeHeadingPath,
  activeFocusSnippet,
  activeHighlightSnippet,
  activeAnchorId,
  activeBlockId,
  activeAnchorKind,
  activeAnchorNumber,
  expectsEquationBinding,
}: UseReaderLocateEngineArgs) {
  const stickyLocateHighlightRef = useRef<StickyLocateHighlight | null>(null)
  const lastAutoScrollKeyRef = useRef('')
  const [locateHint, setLocateHint] = useState('')
  const [equationBindingReady, setEquationBindingReady] = useState(false)
  const [equationBindingBoundCount, setEquationBindingBoundCount] = useState(0)

  useEffect(() => {
    if (!open) return
    stickyLocateHighlightRef.current = null
    lastAutoScrollKeyRef.current = ''
    setLocateHint('')
    const root = contentRef.current
    if (!root) return
    clearReaderFocusClasses(root)
    clearReaderInlineHits(root)
  }, [open, locateRequestId, sourcePath, clearReaderFocusClasses, clearReaderInlineHits, contentRef])

  useEffect(() => {
    if (!open || !drawerReady || !markdown) return
    const sticky = stickyLocateHighlightRef.current
    if (!sticky) return
    const root = contentRef.current
    if (!root) return

    clearReaderFocusClasses(root)
    const target = resolveStickyHighlightTarget(root, readerBlocks, sticky)
    if (!target) return
    const focusedBlock = closestReadableBlock(target) || target
    focusedBlock.classList.add('kb-reader-focus')
    const inlineFormulaTarget = sticky.anchorKind === 'inline_formula'
      ? resolveInlineFormulaTarget(focusedBlock, sticky.highlightSeed)
      : null
    if (inlineFormulaTarget && inlineFormulaTarget !== focusedBlock) {
      inlineFormulaTarget.classList.add('kb-reader-focus-secondary')
    }
    resolveRelatedTargetNodes(root, readerBlocks, sticky.relatedBlockIds || [], focusedBlock)
      .forEach((node) => node.classList.add('kb-reader-focus-secondary'))
    if (sticky.highlightQueries.length > 0 && !root.querySelector('.kb-reader-inline-hit')) {
      for (const query of sticky.highlightQueries) {
        const hit = highlightExactTextInContainer(focusedBlock, query)
        if (hit) break
      }
    }
  }, [
    open,
    drawerReady,
    markdown,
    readerBlocks,
    equationBindingReady,
    contentRef,
    clearReaderFocusClasses,
    resolveStickyHighlightTarget,
    closestReadableBlock,
    resolveInlineFormulaTarget,
    resolveRelatedTargetNodes,
    highlightExactTextInContainer,
  ])

  useEffect(() => {
    if (!open || !drawerReady) {
      setEquationBindingReady(false)
      setEquationBindingBoundCount(0)
      return
    }
    if (!markdown) {
      setEquationBindingReady(false)
      setEquationBindingBoundCount(0)
      return
    }
    if (!expectsEquationBinding) {
      setEquationBindingReady(true)
      setEquationBindingBoundCount(0)
      return
    }
    setEquationBindingReady(false)
    setEquationBindingBoundCount(0)
  }, [open, drawerReady, markdown, expectsEquationBinding, locateRequestId, sourcePath])

  useEffect(() => {
    if (!open || !drawerReady || !markdown) return
    if (!expectsEquationBinding) return
    const equationBlockCount = orderedEquationReaderBlocks(readerBlocks).length
    if (equationBlockCount <= 0) {
      setEquationBindingReady(true)
      setEquationBindingBoundCount(0)
      return
    }
    let cancelled = false
    let raf = 0
    let timer = 0
    let observer: MutationObserver | null = null
    let lastVisibleCount = -1
    let stablePasses = 0
    // Strict locate may need more time for KaTeX and large markdown to fully render/bind anchors.
    const deadline = Date.now() + (strictLocate ? 6500 : 1600)
    const finalize = (boundCount: number) => {
      if (cancelled) return
      observer?.disconnect()
      window.cancelAnimationFrame(raf)
      window.clearTimeout(timer)
      setEquationBindingBoundCount(boundCount)
      setEquationBindingReady(true)
    }
    const scheduleBind = (delayMs = 0) => {
      if (cancelled) return
      window.cancelAnimationFrame(raf)
      window.clearTimeout(timer)
      const trigger = () => {
        if (cancelled) return
        raf = window.requestAnimationFrame(bind)
      }
      if (delayMs > 0) {
        timer = window.setTimeout(trigger, delayMs)
      } else {
        trigger()
      }
    }
    const bind = () => {
      if (cancelled) return
      const root = contentRef.current
      if (!root) {
        if (Date.now() < deadline) {
          scheduleBind(80)
          return
        }
        finalize(0)
        return
      }
      const boundCount = bindVisibleEquationAnchors(root, readerBlocks)
      const visibleCount = root.querySelectorAll('.katex-display').length
      const targetBindCount = Math.min(Math.max(0, visibleCount), Math.max(0, equationBlockCount))
      const bindingSatisfied = targetBindCount <= 0 || boundCount >= targetBindCount
      if (visibleCount === lastVisibleCount && bindingSatisfied) {
        stablePasses += 1
      } else {
        stablePasses = 0
      }
      lastVisibleCount = visibleCount
      if (bindingSatisfied && stablePasses >= 1) {
        finalize(boundCount)
        return
      }
      if (Date.now() < deadline) {
        scheduleBind(bindingSatisfied ? 40 : 90)
        return
      }
      finalize(boundCount)
    }
    const root = contentRef.current
    if (root) {
      observer = new MutationObserver(() => {
        if (cancelled) return
        scheduleBind(20)
      })
      observer.observe(root, { childList: true, subtree: true })
    }
    scheduleBind(0)
    return () => {
      cancelled = true
      observer?.disconnect()
      window.cancelAnimationFrame(raf)
      window.clearTimeout(timer)
    }
  }, [
    open,
    drawerReady,
    markdown,
    readerBlocks,
    expectsEquationBinding,
    strictLocate,
    locateRequestId,
    contentRef,
    orderedEquationReaderBlocks,
    bindVisibleEquationAnchors,
  ])

  useEffect(() => {
    if (!open || !drawerReady || !markdown) return
    if (expectsEquationBinding && !equationBindingReady) return
    let cancelled = false
    let attempts = 0
    let locateRaf = 0
    let scrollRaf = 0
    let retryTimer = 0
    let observer: MutationObserver | null = null
    // Strict locate may need more time for KaTeX and large markdown to fully render/bind anchors.
    const deadline = Date.now() + (strictLocate ? 6500 : 1800)
    const scheduleLocate = (delayMs = 0) => {
      if (cancelled) return
      window.cancelAnimationFrame(locateRaf)
      window.clearTimeout(retryTimer)
      const trigger = () => {
        if (cancelled) return
        locateRaf = window.requestAnimationFrame(runLocate)
      }
      if (delayMs > 0) {
        retryTimer = window.setTimeout(trigger, delayMs)
      } else {
        trigger()
      }
    }
    const finishLocate = () => {
      observer?.disconnect()
      window.cancelAnimationFrame(locateRaf)
      window.clearTimeout(retryTimer)
    }
    const retryLocate = () => {
      if (Date.now() >= deadline) return false
      attempts += 1
      scheduleLocate(Math.min(60 + attempts * 35, 220))
      return true
    }
    const runLocate = () => {
      if (cancelled) return
      const root = contentRef.current
      if (!root || root.clientHeight <= 0 || root.scrollHeight <= 0) {
        retryLocate()
        return
      }
      if (expectsEquationBinding) bindVisibleEquationAnchors(root, readerBlocks)
      clearReaderFocusClasses(root)
      clearReaderInlineHits(root)

      const directResolved = resolveDirectTargetNode(root, readerBlocks, {
        blockId: activeBlockId,
        anchorId: activeAnchorId,
        anchorKind: activeAnchorKind,
      })
      let target: HTMLElement | null = directResolved.target
      let headingTarget: HTMLElement | null = null
      const readerBlockHint = directResolved.hintBlock
      const hasDirectIdentityHint = Boolean(
        activeBlockId
        || activeAnchorId
        || String(readerBlockHint?.block_id || '').trim()
        || String(readerBlockHint?.anchor_id || '').trim(),
      )
      if (!target && hasDirectIdentityHint && alternatives.length > 1) {
        let resolvedAltIndex = -1
        for (let idx = 0; idx < alternatives.length; idx += 1) {
          if (idx === activeAltIndex) continue
          const alt = alternatives[idx]
          if (!alt || typeof alt !== 'object') continue
          const altBlockId = String(alt.blockId || '').trim()
          const altAnchorId = String(alt.anchorId || '').trim()
          if (!altBlockId && !altAnchorId) continue

          const altResolved = resolveDirectTargetNode(root, readerBlocks, {
            blockId: altBlockId,
            anchorId: altAnchorId,
          })
          if (altResolved.target) {
            resolvedAltIndex = idx
            break
          }
        }
        if (resolvedAltIndex >= 0) {
          setActiveAltIndex(resolvedAltIndex)
          return
        }
      }
      if (!target && hasDirectIdentityHint) {
        if (retryLocate()) return
        setLocateHint('Exact evidence block not found. Falling back to fuzzy locate.')
      }
      if (!target && strictLocate) {
        if (retryLocate()) return
        setLocateHint(hasDirectIdentityHint
          ? 'Exact evidence block not found. Strict locate stopped before fuzzy fallback.'
          : 'Strict locate could not resolve an exact evidence block.')
        finishLocate()
        return
      }
      if (!target) {
        const hintHeadingPath = String(readerBlockHint?.heading_path || '').trim()
        const headingNeedles = headingCandidates(activeHeadingPath || hintHeadingPath).map(normalizeText).filter(Boolean)
        if (headingNeedles.length > 0) {
          const headings = Array.from(root.querySelectorAll<HTMLElement>('h1,h2,h3,h4,h5,h6'))
          let bestHeading: HTMLElement | null = null
          let bestHeadingScore = 0
          for (const heading of headings) {
            const text = String(heading.textContent || '').trim()
            for (const needle of headingNeedles) {
              const score = headingMatchScore(needle, text)
              if (score > bestHeadingScore) {
                bestHeading = heading
                bestHeadingScore = score
              }
            }
          }
          if (bestHeading && bestHeadingScore >= 0.18) {
            headingTarget = bestHeading
          }
        }
        const hintBlockText = String(readerBlockHint?.text || '').trim()
        const focusSeed = String(activeHighlightSnippet || activeFocusSnippet || hintBlockText).trim()
        if (focusSeed) {
          const probe = snippetProbeText(focusSeed)
          const hintKind = String(readerBlockHint?.kind || '').trim().toLowerCase()
          const eqNumbersAll = [
            ...extractEquationNumbers(`${activeFocusSnippet} ${activeHeadingPath}`),
            ...extractEquationNumbers(`${hintBlockText} ${hintHeadingPath}`),
          ]
          const hintNumber = Number(readerBlockHint?.number || 0)
          if (Number.isFinite(hintNumber) && hintNumber > 0) {
            eqNumbersAll.push(Math.floor(hintNumber))
          }
          const eqNumbers = Array.from(new Set(eqNumbersAll.filter((item) => Number.isFinite(item) && item > 0)))
          const figNumbersAll = [
            ...extractFigureNumbers(`${activeFocusSnippet} ${activeHeadingPath}`),
            ...extractFigureNumbers(`${hintBlockText} ${hintHeadingPath}`),
          ]
          if (Number.isFinite(hintNumber) && hintNumber > 0 && (activeAnchorKind === 'figure' || hintKind === 'figure')) {
            figNumbersAll.push(Math.floor(hintNumber))
          }
          const figNumbers = Array.from(new Set(figNumbersAll.filter((item) => Number.isFinite(item) && item > 0)))
          const preferFormula = Boolean(
            hasFormulaSignal(probe)
            || hasFormulaSignal(hintBlockText)
            || hintKind === 'equation'
            || activeAnchorKind === 'equation'
          )
          const preferFigure = Boolean(activeAnchorKind === 'figure' || hintKind === 'figure' || figNumbers.length > 0)
          if (!target && preferFigure && Array.isArray(readerBlocks) && readerBlocks.length > 0) {
            const figBlock = readerBlocks.find((item) => {
              if (String(item?.kind || '').trim().toLowerCase() !== 'figure') return false
              const blockNumber = Number(item?.number || 0)
              return figNumbers.length <= 0 || (Number.isFinite(blockNumber) && figNumbers.includes(Math.floor(blockNumber)))
            }) || null
            if (figBlock) {
              target = root.querySelector<HTMLElement>(`[data-kb-block-id="${CSS.escape(String(figBlock.block_id || '').trim())}"]`)
              if (!target) {
                target = Array.from(root.querySelectorAll<HTMLElement>('[data-kb-anchor-id]'))
                  .find((node) => String(node.getAttribute('data-kb-anchor-id') || '') === String(figBlock.anchor_id || '').trim()) || null
              }
            }
          }
          const equationBlocks = visibleEquationBlocks(root)
          const blocks = preferFormula
            ? equationBlocks
            : Array.from(root.querySelectorAll<HTMLElement>('p,li,blockquote,pre,code,figcaption,td,th,.katex-display,[data-kb-anchor-kind="figure"]'))
          const allNodes = Array.from(root.querySelectorAll<HTMLElement>('h1,h2,h3,h4,h5,h6,p,li,blockquote,pre,code,figcaption,td,th,.katex-display,[data-kb-anchor-kind="equation"],[data-kb-anchor-kind="figure"]'))
          const nodeIndex = new Map<HTMLElement, number>()
          allNodes.forEach((node, idx) => nodeIndex.set(node, idx))
          const headingIndex = headingTarget ? Number(nodeIndex.get(headingTarget) ?? -1) : -1

          if (!target && eqNumbers.length > 0) {
            let eqNumBest: HTMLElement | null = null
            let eqNumBestScore = 0
            for (const block of blocks) {
              const text = String(block.textContent || '')
              let score = equationNumberMatchScore(text, eqNumbers)
              if (score <= 0) continue
              score += 0.45 * formulaOverlapScore(probe, text)
              score += 0.35 * snippetMatchScore(probe, text)
              if (headingIndex >= 0) {
                const blockIndex = Number(nodeIndex.get(block) ?? -1)
                if (blockIndex >= 0) {
                  const distance = Math.abs(blockIndex - headingIndex)
                  score += Math.max(0, 0.1 - distance * 0.002)
                }
              }
              if (score > eqNumBestScore) {
                eqNumBest = block
                eqNumBestScore = score
              }
            }
            if (eqNumBest && eqNumBestScore >= 0.18) {
              target = eqNumBest
            }
          }

          if (!target) {
            let best: HTMLElement | null = null
            let bestScore = 0
            for (const block of blocks) {
              let score = snippetMatchScore(probe, block.textContent || '')
              if (preferFormula) {
                score += 0.6 * formulaOverlapScore(probe, block.textContent || '')
              }
              if (eqNumbers.length > 0) {
                score += 0.55 * equationNumberMatchScore(block.textContent || '', eqNumbers)
              }
              const blockAnchor = String(block.getAttribute('data-kb-anchor-id') || '').trim()
              if (activeAnchorId && blockAnchor === activeAnchorId) {
                score += 0.9
              }
              if (headingIndex >= 0) {
                const blockIndex = Number(nodeIndex.get(block) ?? -1)
                if (blockIndex >= 0) {
                  const distance = Math.abs(blockIndex - headingIndex)
                  score += Math.max(0, 0.18 - distance * 0.004)
                }
              }
              if (score > bestScore) {
                best = block
                bestScore = score
              }
            }
            const dynamicThreshold = preferFormula
              ? 0.13
              : (tokenizeText(probe).length >= 8 ? 0.12 : 0.09)
            if (best && bestScore >= dynamicThreshold) {
              target = best
            }
          }
        }
      }
      if (!target && headingTarget) target = headingTarget
      if (!target) {
        const anyReadable = root.querySelector<HTMLElement>('h1,h2,h3,p,li,blockquote,.katex-display,[data-kb-anchor-kind="equation"],[data-kb-anchor-kind="figure"]')
        if (anyReadable) {
          target = anyReadable
          setLocateHint((prev) => prev || 'Fuzzy locate fallback used.')
        }
      }
      if (!target) {
        if (retryLocate()) return
        if (activeFocusSnippet || activeHeadingPath) {
          setLocateHint('Exact snippet not found. Ask again to generate a finer mapping.')
        }
        finishLocate()
        return
      }

      const anchorKindForLocate = String(activeAnchorKind || readerBlockHint?.kind || '').trim().toLowerCase()
      const anchorNumberForLocate = Number.isFinite(Number(activeAnchorNumber || readerBlockHint?.number || 0))
        ? Math.floor(Number(activeAnchorNumber || readerBlockHint?.number || 0))
        : 0
      let nextLocateHint = ''
      const highlightSeed = String(activeHighlightSnippet || activeFocusSnippet || readerBlockHint?.text || '').trim()
      const highlightQueries = buildHighlightQueries(highlightSeed, {
        anchorKind: anchorKindForLocate,
        anchorNumber: anchorNumberForLocate,
      })
      const tryExactHighlight = (container: HTMLElement | null): HTMLElement | null => {
        if (!container || highlightQueries.length <= 0) return null
        for (const query of highlightQueries) {
          const hit = highlightExactTextInContainer(container, query)
          if (hit) return hit
        }
        return null
      }

      let focusedBlock = closestReadableBlock(target) || target
      let exactHit: HTMLElement | null = null
      let inlineFormulaHit: HTMLElement | null = null
      let usedNeighbor = false
      if (anchorKindForLocate === 'inline_formula') {
        inlineFormulaHit = resolveInlineFormulaTarget(focusedBlock, highlightSeed)
        if (!inlineFormulaHit && strictLocate) {
          for (const neighbor of nearbyReadableBlocks(root, focusedBlock, 1)) {
            const candidate = resolveInlineFormulaTarget(neighbor, highlightSeed)
            if (!candidate) continue
            focusedBlock = closestReadableBlock(neighbor) || neighbor
            inlineFormulaHit = candidate
            usedNeighbor = true
            break
          }
        }
      } else if (anchorKindForLocate !== 'figure' && anchorKindForLocate !== 'equation' && highlightQueries.length > 0) {
        exactHit = tryExactHighlight(focusedBlock)
        if (!exactHit && strictLocate) {
          const maxDistance = anchorKindForLocate === 'equation' ? 1 : 2
          for (const neighbor of nearbyReadableBlocks(root, focusedBlock, maxDistance)) {
            const hit = tryExactHighlight(neighbor)
            if (!hit) continue
            focusedBlock = closestReadableBlock(neighbor) || neighbor
            exactHit = hit
            usedNeighbor = true
            break
          }
        }
      }

      focusedBlock.classList.add('kb-reader-focus')
      if (inlineFormulaHit && inlineFormulaHit !== focusedBlock) {
        inlineFormulaHit.classList.add('kb-reader-focus-secondary')
      }
      const relatedTargets = resolveRelatedTargetNodes(root, readerBlocks, relatedBlockIds, focusedBlock)
      relatedTargets.forEach((node) => node.classList.add('kb-reader-focus-secondary'))
      const focusNode = exactHit || inlineFormulaHit || focusedBlock
      stickyLocateHighlightRef.current = {
        blockId: String(focusedBlock.getAttribute('data-kb-block-id') || target.getAttribute('data-kb-block-id') || activeBlockId || readerBlockHint?.block_id || '').trim(),
        anchorId: String(focusedBlock.getAttribute('data-kb-anchor-id') || target.getAttribute('data-kb-anchor-id') || activeAnchorId || readerBlockHint?.anchor_id || '').trim(),
        anchorKind: anchorKindForLocate,
        anchorNumber: anchorNumberForLocate,
        headingPath: String(activeHeadingPath || readerBlockHint?.heading_path || '').trim(),
        highlightSeed,
        highlightQueries: anchorKindForLocate === 'equation' || anchorKindForLocate === 'figure'
          ? []
          : [...highlightQueries],
        relatedBlockIds: [...relatedBlockIds],
        strictLocate,
      }
      const autoScrollKey = `${locateRequestId}::${activeAltIndex}`
      if (lastAutoScrollKeyRef.current !== autoScrollKey) {
        lastAutoScrollKeyRef.current = autoScrollKey
        scrollRaf = window.requestAnimationFrame(() => {
          if (cancelled) return
          scrollReaderTargetIntoView(root, focusNode)
        })
      }

      if (strictLocate) {
        if (exactHit) {
          if (anchorKindForLocate === 'figure') {
            nextLocateHint = 'Exact figure block match.'
          } else {
            nextLocateHint = 'Exact source phrase match.'
          }
        } else if (anchorKindForLocate === 'inline_formula') {
          nextLocateHint = inlineFormulaHit
            ? (usedNeighbor ? 'Neighbor inline formula match.' : 'Inline formula match.')
            : 'Explanation block matched, but inline formula was not found.'
        } else if (anchorKindForLocate === 'figure') {
          nextLocateHint = 'Figure block matched.'
        } else if (anchorKindForLocate === 'equation') {
          nextLocateHint = 'Equation block matched.'
        } else if (highlightQueries.length > 0) {
          nextLocateHint = usedNeighbor ? 'Neighbor evidence block matched, but exact inline phrase was not found.' : 'Evidence block matched, but exact inline phrase was not found.'
        } else {
          nextLocateHint = 'Evidence block matched.'
        }
      } else if (!activeFocusSnippet && activeHeadingPath) {
        nextLocateHint = 'Located by heading.'
      }
      if (nextLocateHint) {
        setLocateHint(nextLocateHint)
      }
      finishLocate()
    }
    const root = contentRef.current
    if (root) {
      observer = new MutationObserver(() => {
        if (cancelled) return
        scheduleLocate(20)
      })
      observer.observe(root, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['data-kb-block-id', 'data-kb-anchor-id'],
      })
    }
    scheduleLocate(0)
    return () => {
      cancelled = true
      finishLocate()
      window.cancelAnimationFrame(scrollRaf)
    }
  }, [
    open,
    drawerReady,
    markdown,
    locateRequestId,
    activeHeadingPath,
    activeFocusSnippet,
    activeHighlightSnippet,
    activeAltIndex,
    activeAnchorId,
    activeBlockId,
    activeAnchorKind,
    activeAnchorNumber,
    readerBlocks,
    alternatives,
    relatedBlockIds,
    strictLocate,
    expectsEquationBinding,
    equationBindingReady,
    contentRef,
    setActiveAltIndex,
    bindVisibleEquationAnchors,
    clearReaderFocusClasses,
    clearReaderInlineHits,
    resolveDirectTargetNode,
    headingCandidates,
    normalizeText,
    headingMatchScore,
    snippetProbeText,
    extractEquationNumbers,
    extractFigureNumbers,
    hasFormulaSignal,
    equationNumberMatchScore,
    visibleEquationBlocks,
    snippetMatchScore,
    formulaOverlapScore,
    tokenizeText,
    nearbyReadableBlocks,
    buildHighlightQueries,
    highlightExactTextInContainer,
    closestReadableBlock,
    resolveInlineFormulaTarget,
    resolveRelatedTargetNodes,
    scrollReaderTargetIntoView,
  ])

  return {
    locateHint,
    equationBindingReady,
    equationBindingBoundCount,
  }
}
