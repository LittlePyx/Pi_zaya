# Changelog

All notable user-facing changes are recorded here. Pi_zaya follows Semantic Versioning for release identifiers.

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
