# Paper Guide Figure Identity And Locate Plan

## 1. Goal

Fix figure/image jump drift by making figure identity stable at conversion time and authoritative across:

1. converter output
2. markdown/image mapping
3. paper-guide provenance
4. frontend locate and reader navigation

This plan is intentionally narrower than the full Paper Guide architecture work.
It focuses on one repeated user-visible failure:

- the system knows the question is about `Figure N` or panel `(a)/(b)`, but the final jump opens the wrong image, the wrong caption, or a nearby paragraph

## 2. Current Diagnosis

### 2.1 What is already true

The current pipeline already contains partial figure-number understanding:

1. extracted image assets are saved as page-local files such as `page_7_fig_1.png`
2. per-page metadata already stores `fig_no`, `fig_ident`, caption text, and bounding boxes
3. markdown repair/remap passes already try to fix image links by nearby caption number
4. paper-guide provenance and reader locate already have figure-aware branches

### 2.2 What is still wrong

The current primary identity of an extracted image is still effectively:

- `page_<page>_fig_<page-local-order>.png`

That is not the same thing as:

- `Figure 4`
- `Figure 4b`
- `Figure 4 caption block`

This creates a structural mismatch:

1. extraction names are page-local and order-sensitive
2. user questions are figure-number or panel oriented
3. provenance wants block/anchor identity
4. frontend wants one stable primary locate target

### 2.3 Why this causes jump drift

There are currently several layers trying to repair the mismatch:

1. converter-side image remap by caption
2. figure caption binding in provenance
3. frontend figure-aware candidate picking
4. reader fallback by figure number or fuzzy signals

These repairs help, but they still operate on top of an unstable base identity.
As long as raw asset order can drift, the system keeps needing recovery logic.

## 3. Core Decision

Do not treat the raw extracted filename as the canonical identity of a figure.

Instead, introduce a canonical `FigureIdentity` record at conversion time and use it as the source of truth everywhere else.

In short:

1. raw asset name is storage detail
2. `figure_id` and `paper_figure_number` are product identity
3. locate should target figure/caption blocks derived from that identity, not rediscover identity later

## 4. Design Principles

### 4.1 Identity first

Every figure-related output should be anchored to a stable figure record before markdown or paper-guide runtime uses it.

### 4.2 Exact locate and figure jump are different surfaces

The system should distinguish:

1. exact evidence locate
2. figure/caption jump
3. nearby fallback

Figure/image jump must not masquerade as exact text locate.

### 4.3 Caption matching is stronger than page-local asset order

If conversion confidently knows that `page_7_fig_1.png` corresponds to `Figure 4`, all downstream logic should inherit that decision.

### 4.4 Renaming helps, but metadata is the true contract

It is acceptable to add stable aliases such as `fig_004.png`, but correctness must come from structured metadata, not from parsing filenames alone.

## 5. Canonical FigureIdentity Model

Each figure record should represent one paper-level figure identity.

Recommended fields:

```json
{
  "figure_id": "fig_004",
  "paper_figure_number": 4,
  "figure_ident": "4",
  "panel_letters": ["a", "b", "c"],
  "page_index": 6,
  "asset_name_raw": "page_7_fig_1.png",
  "asset_name_alias": "fig_004.png",
  "figure_block_id": "blk_fg_...",
  "caption_block_id": "blk_cap_...",
  "anchor_id": "fg_004",
  "caption_text": "Figure 4. ...",
  "locate_anchor": "Figure 4. ...",
  "bbox": [0, 0, 0, 0],
  "crop_bbox": [0, 0, 0, 0],
  "caption_bbox": [0, 0, 0, 0],
  "binding_confidence": 0.97,
  "binding_source": "caption_match"
}
```

Notes:

1. `figure_id` should be stable and paper-scoped
2. `asset_name_raw` keeps backward compatibility with existing extraction files
3. `asset_name_alias` is optional but useful for human readability and later tooling
4. `figure_block_id` and `caption_block_id` let provenance and reader land on structured blocks, not only on image files
5. `binding_confidence` is important for controlled fallback when caption extraction is weak

## 6. Converter Changes

### 6.1 Keep raw assets, add canonical aliases

Do not remove current raw filenames.

Recommended output pattern:

1. raw extracted asset:
   - `page_7_fig_1.png`
2. optional alias:
   - `fig_004.png`
3. sidecar metadata:
   - `page_7_fig_1.meta.json`
   - or document-level figure index

Reason:

1. raw files remain reproducible from extraction
2. aliases make inspection and debugging easier
3. old references do not break

### 6.2 Add a document-level figure index

Current per-page `page_<n>_fig_index.json` is useful but not enough as the main contract.

Add a document-level index such as:

- `assets/figure_index.json`

Suggested structure:

```json
{
  "paper_id": "...",
  "figures": [
    {
      "figure_id": "fig_004",
      "paper_figure_number": 4,
      "asset_name_raw": "page_7_fig_1.png",
      "asset_name_alias": "fig_004.png",
      "page": 7,
      "caption_text": "Figure 4. ...",
      "figure_block_id": "",
      "caption_block_id": "",
      "anchor_id": "fg_004"
    }
  ]
}
```

### 6.3 Promote caption binding into first-class conversion output

The current converter already extracts:

1. `fig_no`
2. `fig_ident`
3. `caption`
4. `bbox` and `caption_bbox`

That work should stop being only a markdown-repair helper and become the official figure identity binding step.

### 6.4 Annotate source blocks with figure identity

When figure blocks and caption blocks are created, attach:

1. `figure_id`
2. `paper_figure_number`
3. `figure_ident`
4. `panel_letters` when available
5. `anchor_id`

This is the bridge from conversion output to runtime provenance.

## 7. Markdown And Asset Reference Rules

### 7.1 Markdown should expose figure number explicitly

For image markdown, prefer stable alt text:

- `![Figure 4](./assets/page_7_fig_1.png)`

or, once aliasing is stable:

- `![Figure 4](./assets/fig_004.png)`

The important part is not only the visible path.
The important part is that the alt text, asset metadata, and source block metadata all agree on the same figure number.

### 7.2 Do not depend on filename parsing alone

Downstream code should never assume that:

- `page_7_fig_1.png == Figure 1`

That assumption is incorrect by design.

### 7.3 Use aliases only when confidence is high

If figure binding is ambiguous:

1. keep only the raw asset name
2. store ambiguity in metadata
3. avoid inventing a confident `fig_004.png` alias

## 8. Provenance Contract Changes

For figure-related answer segments, provenance should carry figure identity explicitly.

Recommended additions:

```json
{
  "claim_type": "figure_claim",
  "anchor_kind": "figure",
  "figure_id": "fig_004",
  "anchor_target_number": 4,
  "primary_block_id": "blk_fg_...",
  "primary_anchor_id": "fg_004",
  "caption_block_id": "blk_cap_...",
  "asset_name_raw": "page_7_fig_1.png",
  "asset_name_alias": "fig_004.png",
  "jump_surface": "figure_caption"
}
```

`jump_surface` should be explicit:

1. `exact_text`
2. `figure_caption`
3. `figure_asset`
4. `nearby_fallback`

This prevents the UI from presenting every figure jump as if it were sentence-level exact locate.

## 9. Frontend Locate And Reader Rules

### 9.1 Primary resolve order for figure jumps

For `claim_type=figure_claim`, the reader should resolve in this order:

1. `primary_block_id`
2. `primary_anchor_id`
3. `figure_id`
4. `anchor_target_number`
5. approved alternatives from the same structured claim group

It should not jump directly from a failed exact figure binding into whole-document fuzzy matching under the same exact-locate UI.

### 9.2 Separate user surfaces

Recommended UI split:

1. `Locate exact evidence`
   - only when a strict text/block target exists
2. `Open figure/caption`
   - valid for image/caption support even when no sentence-level exact target exists
3. `Nearby evidence`
   - optional approximate recovery surface, clearly labeled

### 9.3 Figure panel questions

For questions such as `Figure 1 (f)/(g)`:

1. primary locate should prefer the caption clause or figure-linked explanatory block
2. broader nearby result paragraphs stay secondary context
3. panel letters should be stored in provenance when extraction can resolve them

## 10. Why Naming During Conversion Will Help

If conversion emits stable figure identity early, downstream improvements become easier:

1. markdown remap becomes verification, not rescue
2. provenance can bind to `Figure 4` directly instead of inferring it from answer text
3. frontend locate has fewer ambiguous candidates
4. regression cases become easier to reason about because the same figure keeps the same identity across runs

However, naming alone is not enough.

If the only change is:

- rename `page_7_fig_1.png` to `figure_4.png`

without also updating structured metadata and block bindings, the system will still drift in cases such as:

1. multi-panel figures
2. multiple figures on the same page
3. caption extraction ambiguity
4. figure asset present but caption block missing

## 11. Implementation Phases

### Phase A. Converter Identity Hardening

Goal:
Make figure identity stable at conversion time.

Primary files:

1. `kb/converter/page_figure_metadata.py`
2. `kb/converter/page_text_blocks.py`
3. `kb/converter/page_vision_direct_page.py`
4. `kb/converter/pipeline.py`

Required outputs:

1. document-level `figure_index.json`
2. `figure_id` and `paper_figure_number` on figure metadata
3. optional stable asset aliases when confidence is high

### Phase B. Source Block And Provenance Binding

Goal:
Make runtime figure support use conversion identity rather than rediscovery.

Primary files:

1. `kb/references.py`
2. `kb/paper_guide_provenance.py`
3. `kb/retrieval_engine.py`
4. `kb/paper_guide_answer_post_runtime.py`

Required outputs:

1. figure and caption blocks expose `figure_id` and `paper_figure_number`
2. figure claims in provenance carry explicit figure identity and jump surface

### Phase C. Frontend Locate Surface Tightening

Goal:
Make figure jumps deterministic and honestly labeled.

Primary files:

1. `web/src/components/chat/MessageList.tsx`
2. `web/src/components/chat/reader/useReaderLocateEngine.ts`
3. `web/src/components/chat/MarkdownRenderer.tsx`
4. `web/src/testing/readerRegressionFixtures.ts`

Required outcomes:

1. figure locate prefers structured identity over fuzzy recovery
2. exact locate and figure/caption jump are visually distinct
3. panel-level cases can be regression tested

## 12. Regression And Acceptance

### 12.1 Must-pass regression shapes

At minimum, keep a focused figure set covering:

1. one page with two different figures
2. one paper where figure asset order and figure number differ
3. one explicit panel question such as `(b)` or `(f)/(g)`
4. one figure summary question
5. one caption-first locate question

### 12.2 Acceptance criteria

This slice is complete only when:

1. the same paper reconverted twice keeps the same `figure_id -> figure_number` mapping
2. figure jumps no longer depend on page-local extraction order
3. `Figure N` questions land on the same structured figure/caption target on repeated clicks
4. explicit panel questions prefer caption-panel support over nearby generic result paragraphs
5. figure asset jumps are labeled as figure/caption support, not exact quote locate

## 13. Non-Goals

This slice should not try to solve everything at once.

Out of scope:

1. perfect OCR for all panel text inside images
2. full multimodal understanding of chart semantics
3. arbitrary figure reasoning without a bound paper
4. replacing the broader Paper Guide provenance architecture

## 14. Recommended Next Coding Order

When implementation starts, the safest order is:

1. harden converter figure identity first
2. push `figure_id` into source blocks and provenance
3. update frontend locate to consume figure identity explicitly
4. only then reduce remaining figure-specific heuristics

This order keeps the stack moving from stable source identity toward UI behavior, rather than adding more frontend recovery on top of unstable conversion output.
