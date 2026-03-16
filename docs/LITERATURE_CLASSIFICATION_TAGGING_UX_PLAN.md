# Literature Classification And Tagging UX Plan

## 1. Purpose

This document defines the interaction and information-architecture plan for the
literature classification and tagging feature.

It is a companion to:

- `docs/LITERATURE_CLASSIFICATION_TAGGING_PLAN.md`

That document focuses on product rules, data ownership, and storage model.
This document focuses on how the feature should look and behave in the product.

## 2. Product Goal

The goal is not to add a few extra metadata fields beside each paper.

The goal is to turn the current library page from:

- a file processing page

into:

- a research library workspace

That workspace should support three tasks well:

1. find a known paper quickly
2. browse papers by topic
3. turn temporary conversation insights into reusable library structure

## 3. Core UX Decision

The library should support three complementary browsing modes:

1. `List`
2. `Categories`
3. `Tags`

These are not three independent products.
They are three views over the same persistent paper metadata.

### Role of each view

- `List` is the operational default view
- `Categories` is the top-down thematic view
- `Tags` is the cross-cutting index view

## 4. Information Architecture

The design must separate four concepts clearly.

### A. File Processing State

Examples:

- `pending`
- `converted`
- `queued`
- `running`

This already exists on the current library page and should remain visible, but
it is not part of the paper taxonomy model.

### B. Primary Category

One paper has one main category.

Purpose:

- gives the paper a stable home
- supports top-level browsing

### C. Tags

One paper may have many tags.

Purpose:

- supports horizontal slicing
- supports more than one semantic angle

### D. Reading Status

Recommended values:

- `unread`
- `reading`
- `done`
- `revisit`

Purpose:

- workflow tracking
- not semantic classification

## 5. Main Entry Surface

The primary entry should remain the existing library page:

- `web/src/pages/LibraryPage.tsx`

But the top of the page should grow from a file-processing header into a small
library control bar.

### Recommended top-level layout

1. page header
2. mode switcher
3. shared filter bar
4. active view content
5. right-side or drawer-based paper details editor

## 6. Mode Switcher

The mode switcher should be compact and always visible on desktop.

Recommended tabs:

- `List`
- `Categories`
- `Tags`

Default:

- `List`

Reason:

- it remains the best default for mixed operational and research tasks

## 7. Shared Filter Bar

All views should use the same filter state.

Recommended filters:

- search
- category
- tag
- reading status
- `only unclassified`
- `only suggested`
- `only unread`

Recommended behavior:

- selecting a category in `Categories` view updates the shared filters
- selecting a tag in `Tags` view updates the shared filters
- switching back to `List` preserves current filter context

This is important because the category and tag views should behave as fast
entry points, not isolated subpages.

## 8. List View

This should be the main working surface.

### Row content

Each row should show:

1. title or file name
2. file processing state
3. primary category pill
4. first few user tags
5. reading status badge
6. suggestion indicator if pending
7. actions:
   - `Guide`
   - `Open PDF`
   - `Edit`
   - `More`

### Suggested interaction rule

The row itself should remain mostly a navigation surface.

Inline editing should stay light.
Heavy editing should happen in a dedicated details panel.

### Why not full inline editing

Because the current library page is already busy with:

- conversion actions
- upload flows
- rename flows
- reindex flows

Adding rich inline taxonomy editing to every row would make the page harder to
scan and maintain.

## 9. Paper Details Panel

The feature needs a dedicated edit surface.

Recommended shape:

- right-side drawer on desktop
- full-screen panel on narrow widths

### Sections in the panel

#### A. Basic Info

- title
- authors
- venue
- year
- DOI

#### B. My Organization

- primary category
- user tags
- reading status
- note

#### C. Suggestions

- suggested category
- suggested tags
- accept / dismiss actions

#### D. Quick Actions

- open PDF
- open MD
- open guide conversation
- bind to reading workflow

### Why a dedicated panel

Because the classification feature is fundamentally an editing workflow, not
just a filter workflow.

Without a dedicated panel, the list will either become cluttered or the editing
flow will feel weak.

## 10. Categories View

This view should work like a thematic dashboard, not a second table.

### Structure

Each category appears as a compact card showing:

- category label
- paper count
- unread count
- recent papers
- common tags

### Interaction

Clicking a category card should:

1. set the shared category filter
2. optionally switch to `List`

This creates a clean flow:

- browse by category
- then inspect actual papers in the list

### Why cards instead of a table

Because categories are organizational buckets.
Cards communicate buckets better than rows.

## 11. Tags View

This view should be an index, not a decorative tag cloud.

### Structure

Each tag row should show:

- tag label
- usage count
- optional unread count
- a short sample of recent papers

### Interaction

Clicking a tag should:

1. set the shared tag filter
2. optionally switch back to `List`

### Why not a tag cloud

Because clouds look expressive but are poor tools for serious library
navigation.
You need sorting, counts, and predictable scanning.

## 12. Suggestion UX

The suggestion system should feel helpful, not intrusive.

### Row-level suggestion signal

If a paper has unapplied suggestions, the row should show a small status like:

- `2 suggestions`

### Details-level suggestion actions

Inside the details panel:

- accept one suggestion
- dismiss one suggestion
- accept all
- dismiss all

### Default rule

Suggestions are visible but not auto-applied by default.

This is consistent with the ownership rules from
`LITERATURE_CLASSIFICATION_TAGGING_PLAN.md`.

## 13. Reading Status UX

Reading status should be visible in the list and editable in the panel.

Recommended UI:

- small badge in the row
- segmented control or select inside the details panel

Why explicit:

- status is likely to be filtered often
- status is a first-class workflow field, not a hidden tag

## 14. Batch Editing

This feature will not scale without batch actions.

### Batch edit bar should support

- set category
- add tag
- remove tag
- set reading status
- clear category

### Activation

Only show the batch bar when one or more rows are selected.

This avoids permanent UI clutter.

## 15. CiteShelf Integration

This is the most important cross-surface rule.

The current CiteShelf should remain conversation-local.

### Keep in CiteShelf

- temporary tags
- temporary notes
- research-session grouping

### Add from CiteShelf

- `Save selected tags to library`
- `Save note to library`
- `Set library category`

### Do not do

- automatic synchronization of shelf tags into library tags

Reason:

- shelf tags often represent temporary task context
- library tags should stay long-term and curated

## 16. Reader And Citation Integration

The reader and citation surfaces should expose lightweight entry points only.

Recommended actions:

- `Add tag`
- `Save to library`
- `Open library details`

Do not turn the reader into the primary metadata editor.

## 17. Delivery Phases

### UX-A. Library Metadata In List View

Goal:

Add visible category/tag/status signals into the existing list.

Work:

1. add category pill
2. add first tags
3. add reading status badge
4. add suggestion count badge

### UX-B. Details Panel

Goal:

Provide a proper editing surface.

Work:

1. build details drawer
2. wire user fields
3. wire suggestion accept/dismiss

### UX-C. Shared Filters

Goal:

Make taxonomy useful for browsing.

Work:

1. add search + category + tag + status filters
2. keep filter state across views

### UX-D. Categories View

Goal:

Support thematic browsing.

Work:

1. category cards
2. counts
3. jump into filtered list

### UX-E. Tags View

Goal:

Support horizontal discovery.

Work:

1. sortable tag index
2. counts
3. jump into filtered list

### UX-F. Batch Actions

Goal:

Make the feature practical for larger libraries.

Work:

1. multi-select
2. batch edit bar
3. batch category/tag/status updates

### UX-G. CiteShelf Bridge

Goal:

Connect conversation work to persistent library structure.

Work:

1. `Save to library tags`
2. `Save note to library`
3. optional `Set library category`

## 18. Conflict Check Against Existing Plans

### A. `LITERATURE_CLASSIFICATION_TAGGING_PLAN.md`

Status:

- aligned

Reason:

- that document defines data ownership and storage rules
- this document defines view structure and interaction rules
- they are complementary, not competing

### B. `CITE_SHELF_OPTIMIZATION_IDEAS.md`

Status:

- mostly aligned, but one boundary must be enforced

Important rule:

- CiteShelf remains a temporary conversation workspace
- library taxonomy remains the persistent system of record

Potential risk:

- if future CiteShelf work tries to make shelf tags authoritative, that would
  conflict with this plan

Resolution:

- only bridge via explicit user actions

### C. `PAPER_GUIDE_P2_DEEP_READING_PLAN.md`

Status:

- no direct conflict

Reason:

- paper-guide P2 is about reading flow inside the reader
- this taxonomy plan is about library organization and metadata editing

Shared surface:

- reader may later expose light `Save to library` actions

Constraint:

- reader must not become the main taxonomy editor

### D. `CONVERSATION_OPEN_PERFORMANCE_PLAN.md`

Status:

- no architectural conflict, but there is sequencing risk

Risk:

- adding heavy metadata UI directly into hot chat surfaces too early would
  work against the performance plan

Resolution:

- keep the primary taxonomy UI centered in `LibraryPage`
- only add lightweight hooks in chat, CiteShelf, and reader

### E. Current `LibraryPage`

Status:

- this plan changes its role significantly

Risk:

- if all UX pieces land at once, the current page may become too crowded

Resolution:

- deliver in the phased order above
- do not mix all taxonomy controls into the first release

## 19. Recommended Execution Order

If this work starts soon, the best order is:

1. backend metadata model from `LITERATURE_CLASSIFICATION_TAGGING_PLAN.md`
2. `UX-A` list signals
3. `UX-B` details panel
4. `UX-C` shared filters
5. `UX-D` categories view
6. `UX-E` tags view
7. `UX-F` batch actions
8. `UX-G` CiteShelf bridge

This keeps the first release useful without overloading the current library
page too early.

## 20. Final Recommendation

The feature should be designed as:

- a persistent library taxonomy layer
- a compact multi-view browsing model
- a dedicated details editing surface
- a clear bridge from temporary conversation annotations into persistent paper
  organization

The most important UX rule is:

`classification should help the user browse and decide faster, not add one more dense admin panel`
