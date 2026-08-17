# Pi_zaya Release Runbook

This runbook governs downloadable Pi_zaya releases. The first target is `v0.1.0-beta.1`, delivered as a self-contained Windows x64 portable ZIP.

## Current release decision

The project is suitable for an explicitly labeled beta release. The application can be downloaded, extracted, and started without installing Node.js or Python. It is not yet a polished generally available desktop release because jobs are not durable across restarts, the artifact is unsigned, and updates are manual.

The owner selected the MIT License on 2026-08-18. Root `LICENSE` carries the standard MIT terms with `Copyright (c) 2026 LittlePyx`. Formal builds and the tag workflow still require that file to be present.

## Release contract

- Canonical version: root `VERSION`, valid Semantic Versioning without a leading `v`.
- Git tag: exactly `v` plus `VERSION`.
- Frontend version: `web/package.json` and the lockfile must match `VERSION`.
- Platform: Windows x64 portable ZIP.
- Runtime: official CPython embeddable distribution at the version in `.python-version`, with backend dependencies installed into the package.
- Dependencies: exact Windows runtime versions in `requirements-release.txt`; the built package records the resolved set in `THIRD_PARTY_PACKAGES.txt` and must pass `pip check`.
- Frontend: production `web/dist`; Node.js is not included or needed at runtime.
- User data: `%LOCALAPPDATA%\Pi_zaya` by default. An explicit `KB_APP_DATA_DIR` still wins.
- Network binding: desktop launcher uses `127.0.0.1` only.
- Integrity: every ZIP has an adjacent SHA-256 file and JSON artifact manifest.
- License: MIT; `LICENSE` must be present both in the repository and inside the ZIP.
- Entry/exit: `Start-Pi-zaya.cmd` and `Stop-Pi-zaya.cmd`.

Development mode is unchanged: repository-local databases and preferences remain the default unless `KB_RELEASE_MODE=1` or `KB_APP_DATA_DIR` is explicitly set. There is no implicit migration of existing development data.

## Local validation build

The following fast build uses the current system Python only to validate staging and startup. It is not an end-user artifact:

```powershell
.\tools\release\build_windows_portable.ps1 `
  -OutputDir .\.runtime\release-smoke `
  -PythonRuntime System `
  -SkipFrontendBuild `
  -KeepStage `
  -AllowDirty

.\tools\release\smoke_windows_portable.ps1 `
  -BundleRoot .\.runtime\release-smoke\Pi_zaya-v0.1.0-beta.1-windows-x64
```

`-AllowDirty` marks the manifest as a non-release build. `-AllowMissingLicense` only exists for mechanical tests on historical branches; it creates an internal artifact whose manifest says `missing-release-blocked`. Never distribute either kind of artifact.

## Formal local build

After the license is present:

```powershell
cd web
npm ci
npm run lint
npm run build
cd ..

.\tools\release\build_windows_portable.ps1 `
  -PythonRuntime Embedded `
  -SkipFrontendBuild `
  -KeepStage

.\tools\release\smoke_windows_portable.ps1 `
  -BundleRoot .\release\Pi_zaya-v0.1.0-beta.1-windows-x64
```

Verify the final checksum independently:

```powershell
Get-FileHash .\release\Pi_zaya-v0.1.0-beta.1-windows-x64.zip -Algorithm SHA256
Get-Content .\release\Pi_zaya-v0.1.0-beta.1-windows-x64.zip.sha256
```

## Tag release

1. Confirm the MIT `LICENSE` and copyright line are still correct.
2. Update `VERSION`, both frontend version fields, and `CHANGELOG.md`.
3. Confirm the normal CI workflow passes on the exact commit.
4. Create and push the exact tag, such as `v0.1.0-beta.1`.
5. `.github/workflows/release-windows.yml` reruns the complete backend, frontend, research, conversion, and browser gates on Windows.
6. The workflow builds the embedded-runtime ZIP, starts the packaged application from its staged folder, checks `/api/health`, `/api/app/version`, and the React root, then stops it through the packaged stop command.
7. Only after those checks pass does the workflow create a GitHub prerelease and attach the ZIP, checksum, and artifact manifest.

## Clean-machine acceptance

Before promoting beyond beta, also test the downloaded ZIP manually on a Windows account that has neither Python nor Node.js on `PATH`:

1. Verify SHA-256, then extract the whole ZIP.
2. Double-click `Start-Pi-zaya.cmd` and confirm the browser opens the library page.
3. Configure provider credentials in Settings and restart once to prove preferences survive.
4. Upload two representative PDFs, run concurrent conversion, cancel one document, and retry it.
5. Confirm each document reports its own terminal outcome; induce or use a fixture for index retry without reconversion.
6. Ask a grounded question, open a citation in the reader, build and export a small evidence matrix and research brief.
7. Run `Stop-Pi-zaya.cmd`, replace the app folder with the same-version test build, restart, and confirm the library persists under `%LOCALAPPDATA%\Pi_zaya`.
8. Inspect `%LOCALAPPDATA%\Pi_zaya\logs` and confirm no API keys are printed.

Record the Windows version, package SHA-256, result, and known exceptions in the release notes.

## 2026-08-17 implementation acceptance

The release foundation passed on the working tree based on `bfe0a825`:

- canonical version and release-path tests: pass;
- Ruff and `git diff --check`: pass;
- backend unit: 4,437 passed, 41 skipped;
- backend sanity: 266 passed, 2 skipped;
- visible Research Agent answer contract: 5 passed;
- frontend lint and production build: pass;
- browser smoke: 128 passed, 2 skipped;
- core citation/library browser regressions: 111 passed;
- ordinary-user surface isolation: 4 passed;
- Research QA fixture/full-library: 56 and 29 cases;
- source grounding: 41/41;
- grounded replay: 6/6;
- Agent golden contract: 10 cases, zero errors;
- reviewed Agent replay: 5/5;
- paired comparison: 5/5 with zero false comparisons;
- comparison candidates: 5/5 with 18 discoveries and zero contract/evidence/prefill failures at `test_results/evidence_comparison_candidates/20260817_205325`;
- project status: 5/5 at `test_results/project_research_status/20260817_205327`;
- project journey: 20/20 at `test_results/project_research_journey/20260817_205351`;
- converter quality: 13/13 at `test_results/converter_quality_eval/20260817_205354`;
- system-Python staging smoke: pass;
- final embedded CPython 3.10.11 package: `pip check`, version API, health API, React root, process record, safe stop, and independent ZIP checksum all pass; observed ZIP size is 90.4 MB.

The local acceptance package was intentionally built with `-AllowMissingLicense`; its manifest records `missing-release-blocked`. It proves package mechanics only and is not an authorized distribution artifact.

### 2026-08-18 MIT acceptance

After the owner selected MIT, the standard license was added with `Copyright (c) 2026 LittlePyx`. The release builder now rejects a dirty Git working tree unless `-AllowDirty` is explicit, validates the MIT contract, copies `LICENSE` into the application root, and records license and source-cleanliness fields in both manifests.

A new embedded-runtime acceptance build used `-AllowDirty` but did not use `-AllowMissingLicense`. It passed `pip check`, packaged-runtime startup, health/version APIs, React root, safe stop, root-versus-package license hash equality, ZIP license presence, and independent SHA-256 verification. The observed ZIP size remained 90.4 MB. Its manifests correctly report `license=MIT`, `license_status=included`, and `source_dirty=true`; it remains a non-distributable mechanics proof until these sources are committed. A clean tagged checkout will report `source_dirty=false` and is the only form the release workflow may publish.

## Promotion gates after beta

- Persist queued/active conversion jobs and recover or honestly fail them after restart.
- Add code signing and decide whether to ship an installer in addition to the portable ZIP.
- Define an automatic or guided updater with rollback and data-schema compatibility checks.
- Exercise backup/restore and data migration across consecutive real release versions.
- Expand clean-machine coverage beyond the GitHub-hosted Windows image.
