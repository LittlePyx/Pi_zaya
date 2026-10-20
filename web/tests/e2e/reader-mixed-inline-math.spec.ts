import { expect, test } from '@playwright/test'

test('mixed source prose with inline mathematics retains its identity and quote target', async ({ page }, testInfo) => {
  await page.goto('/__message_list_test__?scenario=agent-answer-contract-source-precedence')
  const result = await page.evaluate(async () => {
    const { renderMixedInlineMathSource } = await import('/src/testing/readerMixedInlineMathSmoke.tsx')
    const { resolveDirectTargetNode, bindVisibleEquationAnchors } = await import('/src/components/chat/reader/readerDomUtils.ts')
    const quote = String.raw`where $\odot$ denotes the element-wise multiplication and $\boldsymbol{b}$ denotes one instance of binary Bernoulli vector whose entries are independently sampled from a Bernoulli distribution with probability $p \in (0, 1)$.`
    const prose = quote + String.raw` Then, a set of image pairs $\{(\widehat{\boldsymbol{y}}_m, \overline{\boldsymbol{y}}_m)\}_{m=1}^M$ is defined as: for each $m$,`
    const equation = String.raw`\widehat{\boldsymbol{y}}_m := \boldsymbol{b}_m \odot \boldsymbol{y}; \quad \overline{\boldsymbol{y}}_m := (\mathbf{1} - \boldsymbol{b}_m) \odot \boldsymbol{y}. \tag{6}`
    const source = `${prose}\n\n$$\n${equation}\n$$`
    const block = { block_id: 'mixed-inline', anchor_id: 'eq-inline', kind: 'equation', raw_text: source, text: source, line_start: 1, line_end: 5 }
    const host = document.createElement('article')
    document.body.append(host)
    renderMixedInlineMathSource(host, source, block)
    for (let i = 0; i < 100 && !host.querySelector('.katex-display'); i++) await new Promise((r) => setTimeout(r, 10))
    bindVisibleEquationAnchors(host, [block])
    const opts = { blockId: block.block_id, anchorId: block.anchor_id, anchorKind: 'equation', evidenceQuote: quote }
    const target = resolveDirectTargetNode(host, [block], opts).target
    const math = resolveDirectTargetNode(host, [block], { ...opts, evidenceQuote: `$$${equation}$$` }).target
    const changed = resolveDirectTargetNode(host, [block], { ...opts, evidenceQuote: quote.replace('independently sampled', 'jointly sampled') }).target
    const duplicate = target?.cloneNode(true)
    if (duplicate) host.append(duplicate)
    const ambiguous = resolveDirectTargetNode(host, [block], opts).target
    return { html: host.innerHTML, checks: {
      prose: target?.getAttribute('data-kb-source-prose') === '1',
      phrase: target?.textContent?.includes('denotes the element-wise multiplication and'),
      id: target?.getAttribute('data-kb-block-id'),
      mathPreserved: math?.classList.contains('katex-display'),
      changedNotProse: changed?.getAttribute('data-kb-source-prose') !== '1',
      ambiguousNotProse: ambiguous?.getAttribute('data-kb-source-prose') !== '1',
    } }
  })
  await testInfo.attach('rendered-source.html', { body: result.html, contentType: 'text/html' })
  expect(result.checks).toEqual({ prose: true, phrase: true, id: 'mixed-inline', mathPreserved: true, changedNotProse: true, ambiguousNotProse: true })
})

test('a complete mixed quote highlights real prose only after its math and identity match', async ({ page }, testInfo) => {
  await page.goto('/__message_list_test__?scenario=agent-answer-contract-source-precedence')
  const result = await page.evaluate(async () => {
    const { renderMixedInlineMathSource } = await import('/src/testing/readerMixedInlineMathSmoke.tsx')
    const { resolveDirectTargetNode, bindVisibleEquationAnchors, buildHighlightQueries, highlightExactTextInContainer } = await import('/src/components/chat/reader/readerDomUtils.ts')
    const { mixedProseHighlightSeed } = await import('/src/components/chat/reader/mixedEquationTarget.ts')
    const prose = String.raw`where $\odot$ denotes the element-wise multiplication and $\boldsymbol{b}$ denotes one instance of binary Bernoulli vector whose entries are independently sampled from a Bernoulli distribution with probability $p \in (0, 1)$. Then, a set of image pairs $\{(\widehat{\boldsymbol{y}}_m, \overline{\boldsymbol{y}}_m)\}_{m=1}^M$ is defined as: for each $m$,`
    const equation = String.raw`\widehat{\boldsymbol{y}}_m := \boldsymbol{b}_m \odot \boldsymbol{y}; \quad \overline{\boldsymbol{y}}_m := (\mathbf{1} - \boldsymbol{b}_m) \odot \boldsymbol{y}. \tag{6}`
    const quote = `${prose}\n$$\n${equation}\n$$`
    const block = { block_id: 'mixed-complete', anchor_id: 'eq-complete', kind: 'equation', raw_text: quote, text: quote, line_start: 1, line_end: 4 }
    const host = document.createElement('article'); document.body.append(host)
    renderMixedInlineMathSource(host, quote, block)
    for (let i = 0; i < 100 && !host.querySelector('.katex-display'); i++) await new Promise((r) => setTimeout(r, 10))
    bindVisibleEquationAnchors(host, [block])
    const opts = { blockId: block.block_id, anchorId: block.anchor_id, anchorKind: 'equation', evidenceQuote: quote }
    const target = resolveDirectTargetNode(host, [block], opts).target
    const seed = target ? mixedProseHighlightSeed(target, quote, host) : ''
    const hit = target ? buildHighlightQueries(seed, { anchorKind: 'paragraph' }).some((query) => Boolean(highlightExactTextInContainer(target, query))) : false
    const wrong = quote.replace('(\\mathbf{1} - \\boldsymbol{b}_m)', '(\\mathbf{1} + \\boldsymbol{b}_m)')
    const wrongTarget = resolveDirectTargetNode(host, [block], { ...opts, evidenceQuote: wrong }).target
    const wrongSeed = target ? mixedProseHighlightSeed(target, wrong, host) : ''
    const changedSeed = target ? mixedProseHighlightSeed(target, quote.replace('independently sampled', 'jointly sampled'), host) : ''
    const math = host.querySelector('.katex-display')
    if (math) host.append(math.cloneNode(true))
    const ambiguous = resolveDirectTargetNode(host, [block], opts).target
    return { html: host.innerHTML, checks: { prose: target?.getAttribute('data-kb-source-prose') === '1', seed, hit, wrongNotProse: wrongTarget?.getAttribute('data-kb-source-prose') !== '1', wrongSeed, changedSeed, ambiguousNotProse: ambiguous?.getAttribute('data-kb-source-prose') !== '1' } }
  })
  await testInfo.attach('complete-mixed-source.html', { body: result.html, contentType: 'text/html' })
  expect(result.checks).toEqual({ prose: true, seed: 'denotes the element-wise multiplication and', hit: true, wrongNotProse: true, wrongSeed: '', changedSeed: '', ambiguousNotProse: true })
})
