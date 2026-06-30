# Future MCP Server Design

This document sketches a future MCP-compatible tool server for Pi_zaya. It is a
design note only; the current repository does not ship a full MCP server.

## Goals

- Expose local paper-library operations as safe, citation-grounded tools.
- Keep data local by default and avoid uploading PDFs or indexes to third-party
  services without explicit user action.
- Preserve evidence traceability in every tool response.
- Reuse the existing retrieval, reference, reader, and literature-basket modules
  instead of building a parallel stack.

## Proposed Tools

| Tool | Purpose | Output expectations |
|---|---|---|
| `search_papers(query, scope)` | Search current paper, basket, or full library. | Ranked papers/chunks with source IDs, headings, scores, and evidence previews. |
| `read_paper_chunk(paper_id, chunk_id)` | Read a specific indexed chunk. | Chunk text plus source path, heading, page/anchor metadata when available. |
| `resolve_reference(paper_id, ref_id)` | Resolve an in-paper reference. | Bibliographic metadata, DOI if available, citing context, and source paper. |
| `compare_papers(paper_ids, question)` | Compare selected papers for a user question. | Source-specific evidence groups and compact comparison notes. |
| `build_reading_guide(paper_id)` | Suggest a reading path through one paper. | Section waypoints with why each section matters. |
| `export_bibtex(paper_id)` | Export reference metadata. | BibTeX or clear explanation when metadata is incomplete. |
| `add_to_literature_basket(paper_id)` | Add a paper or reference to the local basket. | Basket item ID and normalized citation metadata. |

## Safety And Consent

- Tools should be read-only by default. Mutating tools such as
  `add_to_literature_basket` should require an explicit user action.
- File paths should be sanitized in responses intended for external clients.
- Large source texts should be paginated or chunked to avoid accidentally
  exposing more local data than requested.
- Remote enrichment, if enabled, should clearly disclose which metadata is sent
  outside the local machine.

## Local-First Data Handling

The MCP server should run against the same local stores used by the FastAPI app:

- `db/` for indexed documents and chunks
- `library.sqlite3` for library metadata
- `chat.sqlite3` for conversation history when explicitly requested
- `references_index.json` and Crossref cache for references

The server should not require a hosted vector database or external document
service.

## Citation-Grounded Outputs

Every answer-like tool should return:

- `source_name`
- `source_path` or a stable local source ID
- `heading_path`
- evidence preview or chunk ID
- citation/reference ID when available
- limitations when evidence is missing or low confidence

This keeps downstream agents from presenting unsupported claims as grounded
paper conclusions.

## Implementation Notes

The lowest-risk implementation path would be:

1. Add an MCP adapter layer that wraps existing functions in `kb.retrieval_engine`,
   `kb.reference_index`, `kb.agent.tools`, and library/basket stores.
2. Keep schemas small and JSON-serializable.
3. Add unit tests with synthetic indexes rather than requiring a real PDF corpus.
4. Add a local-only example configuration and document how to disable remote
   enrichment.

Large changes, such as a new storage backend or remote tool execution service,
should stay out of the first MCP version.
