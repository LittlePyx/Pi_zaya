# Literature Classification Tagging Handoff

## 1. Current Status

The library taxonomy feature currently has these pieces implemented:

- persistent library-level fields:
  - `paper_category`
  - `reading_status`
  - `note`
  - `user_tags`
- browse surfaces:
  - `List`
  - `Categories`
  - `Tags`
- editing flows:
  - single-paper edit drawer
  - batch edit drawer
- suggestion flows:
  - regenerate suggestions
  - accept / dismiss suggestion category
  - accept / dismiss suggestion tags
  - `only suggested` filter
  - per-row suggestion count badge

## 2. Why Suggestions Currently Feel Weak

The current suggestion layer is intentionally minimal and heuristic-based.

It is implemented in:

- `kb/library_store.py`
  - `_generate_suggestions_for_row(...)`
  - `regenerate_paper_suggestions(...)`
  - `apply_paper_suggestion_actions(...)`

### Main limitations

1. It only uses shallow local signals.

Current inputs are mainly:

- PDF path / filename
- citation metadata stored in `citation_meta`
- title / venue / optional abstract if present

It does **not** yet use:

- markdown full text
- paper-guide summaries
- reader highlights
- conversation history
- project context

2. The rule vocabulary is still very small.

Current built-in category and tag rules only cover a narrow set such as:

- `NeRF`
- `3DGS`
- `SCI`
- `Diffusion`
- `Survey`
- `Dataset`
- `Benchmark`

and a small tag set like:

- `single-image`
- `pose-free`
- `real-time`
- `camera-pose`
- `compressive-sensing`

So many papers naturally get:

- no category suggestion
- only 1-3 tags

3. Category suggestion is conservative by design.

Right now a category suggestion is only surfaced when:

- the paper does not already have a user category
- the dismissed category does not match the new guess

This is correct for ownership, but it also makes category suggestions appear less often.

4. Tag suggestions are filtered aggressively.

Suggested tags are removed if they are already:

- accepted by the user
- dismissed by the user
- duplicated by normalization

They are also capped to a small list.

So after a few accept / dismiss actions, a paper may show very few remaining suggestions.

5. Suggestions are local and user-private, but not yet user-adaptive enough.

The current engine already avoids a global one-size-fits-all result, but it only lightly uses the user's existing library taxonomy.

It does **not** yet learn a richer user-specific vocabulary or clustering pattern.

## 3. Recommended Next Steps

The next window should focus on improving suggestion quality, not adding more surface UI first.

### Priority A. Better Inputs

Upgrade suggestion generation to consume more useful signals:

1. markdown-derived title / headings / keywords
2. abstract or intro snippet if available
3. existing reader highlights or saved notes
4. optional paper-guide summary text

This is the highest-value improvement.

### Priority B. Better User-Specific Vocabulary

Build suggestion vocab from the current user's library:

1. mine frequent accepted categories
2. mine frequent accepted tags
3. mine co-occurring tags per category
4. rank suggestions using user-local frequency + text match

Goal:

- not one fixed global rule table
- but a user-adaptive taxonomy recommender

### Priority C. Better Suggestion Model

Move from hardcoded phrase rules to a hybrid:

1. rule-based seed candidates
2. text-match scoring over title / md summary / keywords
3. optional lightweight LLM refinement later

Do **not** jump straight to fully automatic write-back.

Keep:

- system suggests
- user confirms

### Priority D. Batch Suggestion Actions

Once suggestion quality improves, add:

- batch accept suggestions
- batch dismiss suggestions
- `only suggested + selected papers + apply`

This will make the suggestion layer operationally useful.

### Priority E. Conversation Bridge

After suggestion quality is good enough, implement:

- CiteShelf -> `Save to library tags`
- CiteShelf -> `Save note to library`
- optional `Set library category`

This should remain explicit, never silent.

## 4. Concrete Implementation Order

Recommended order for the next window:

1. improve backend suggestion inputs in `kb/library_store.py`
2. keep current routes and UI contract stable
3. rerun:
   - `pytest tests/sanity/test_library_phase1_api.py -q`
   - `npm run build`
4. then add batch suggestion actions
5. only after that start CiteShelf bridge

## 5. Key Files

### Backend

- `kb/library_store.py`
- `api/routers/library.py`

### Frontend

- `web/src/api/library.ts`
- `web/src/stores/libraryStore.ts`
- `web/src/pages/LibraryPage.tsx`
- `web/src/styles/index.css`

### Plans

- `docs/LITERATURE_CLASSIFICATION_TAGGING_PLAN.md`
- `docs/LITERATURE_CLASSIFICATION_TAGGING_UX_PLAN.md`

## 6. Current Validation Baseline

Last known checks passed:

- `pytest tests/sanity/test_library_phase1_api.py -q`
- `npm run build`

The running backend was restarted after suggestion routes were added, and `openapi.json` includes:

- `/api/library/meta/suggestions/regenerate`
- `/api/library/meta/suggestions/apply`

## 7. Short Continuation Prompt

If continuing in a new window, use this:

> Continue improving literature classification suggestions. Read `docs/LITERATURE_CLASSIFICATION_TAGGING_HANDOFF.md` first, then improve suggestion quality in `kb/library_store.py` by using richer document-derived signals instead of only shallow title/path heuristics.
