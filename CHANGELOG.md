# Changelog

All notable user-facing changes are recorded here. Pi_zaya follows Semantic Versioning for release identifiers.

## [0.1.0-beta.8] - 2026-08-18

### Added

- Add a small native `Pi_zaya.exe` Windows launcher with the product icon, single-instance behavior, a system-tray menu, direct access to logs and user data, and bounded Chinese startup errors.
- Let the portable launcher automatically select another loopback port when the preferred port is occupied, with bounded startup and confirmed safe-stop behavior.

### Changed

- Make `Pi_zaya.exe` the primary portable-package entrypoint while retaining the start and stop command scripts as diagnostic fallbacks.
- Exercise the native launcher and an intentionally occupied preferred port in the packaged clean-profile smoke gate before release publication.

## [0.1.0-beta.7] - 2026-08-18

### Added

- Ship a dedicated `README-中文.md` in the Windows portable package with startup, API/model setup, data-location, update, and checksum instructions.
- Add safe provider detection and live model discovery for Qwen, DeepSeek, OpenAI, and custom OpenAI-compatible endpoints, with text/vision model-category filtering.
- Provide provider and model dropdowns while preserving free-form model IDs for private, newly released, or otherwise unlisted models.

### Changed

- Bound interactive model discovery to five seconds and connection tests to twelve seconds, disable automatic retries, and add client-side cancellation so unavailable or mismatched providers cannot leave Settings stuck loading.
- Treat generic `sk-` credentials as ambiguous instead of probing multiple providers; the user selects the intended provider before the key is sent anywhere.

## [0.1.0-beta.6] - 2026-08-18

### Added

- Show a dismissible first-run model setup guide when no text-model API key is configured, with a direct, focused path into text credentials and clear guidance that the vision connection is optional but recommended.

### Changed

- Run the formal Windows package acceptance from the checksum-verified ZIP after fresh extraction, with an isolated Windows profile, an empty credential environment, no Python or Node.js on `PATH`, and the default `%LOCALAPPDATA%\Pi_zaya` data path.
- Verify that a clean packaged launch does not inherit a text-model API key before publishing the release assets.

## [0.1.0-beta.5] - 2026-08-18

### Fixed

- Show project-only literature-basket actions only after the active project context is ready, preventing an early evidence-matrix or research-brief click from being discarded during conversation restoration.
- Keep research-gap, evidence-matrix, and research-brief actions disabled while basket metadata is still updating, so a background refresh cannot consume the user's first click.
- Pre-transform large lazy routes and run the Windows core browser suite serially, separating dev-server cold readiness from unchanged functional assertions and interaction budgets.
- Preserve Playwright traces and screenshots on the first failed attempt and upload them from CI and Windows release runs for actionable browser-gate diagnostics.
- Require an untagged Windows release workflow preflight before creating the next immutable beta tag.

## [0.1.0-beta.4] - 2026-08-18

### Fixed

- Scope the generated-artifact ignore rule to the repository root so the Windows portable build and packaged-runtime smoke scripts are tracked and available on clean release runners.
- Make the packaged-runtime smoke gate require the bundled MIT license and reject artifacts whose manifest reports a dirty source tree.
- Wait for restored conversation and project-shelf state before evidence-matrix browser tests open the workspace, preserving the full comparison-review assertions under slow parallel runners.

## [0.1.0-beta.3] - 2026-08-18

### Fixed

- Make the research-gap browser acceptance test wait for restored conversation and project-shelf state before exercising the workspace, eliminating a slow-runner hydration race without relaxing its assertions.

## [0.1.0-beta.2] - 2026-08-18

### Fixed

- Force UTF-8 for Python processes in the Windows release workflow so Unicode research fixtures and conversion paths cannot fail under the runner's legacy console encoding.

## [0.1.0-beta.1] - 2026-08-17

First downloadable beta candidate:

- FastAPI + React desktop-local product path with evidence-grounded PDF Q&A, traceable citations, literature baskets, evidence matrices, research briefs, and research gap/status workflows.
- Conversion quality center with anchored Markdown, safe repair, structured indexing, per-document concurrent task control, cancellation, terminal outcomes, and precise conversion/index retry actions.
- Canonical application version exposed by the API and update panel, including correct prerelease ordering.
- Windows x64 portable ZIP build with a bundled Python runtime and prebuilt frontend; Node.js and system Python are not required for the formal artifact.
- Desktop launcher and safe stop command, release manifest, SHA-256 checksum, and `%LOCALAPPDATA%\Pi_zaya` user-data isolation.
- Tag-driven Windows build that reruns backend, frontend, research-quality, conversion-quality, and packaged-runtime smoke gates before creating a GitHub prerelease.
- Project source and release artifacts licensed under the MIT License.

Known beta boundaries:

- Background conversion state and the recent-result journal are process-local. A restart does not resume queued or active work.
- The portable package is unsigned and has no installer or automatic updater. Updates replace the application folder while preserving the separate user-data directory.
- Model access still requires user-supplied provider credentials.
