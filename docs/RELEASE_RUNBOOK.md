# Pi_zaya Release Runbook

This runbook governs downloadable Pi_zaya releases. `v0.1.0-beta.12` is the current published downloadable beta, and `v0.1.0-beta.13` is the active untagged candidate. The release uses a self-contained Windows x64 current-user installer and a portable ZIP. `v0.1.0-beta.5` was the first published downloadable beta.

The current beta temporarily withholds the project evidence-matrix
workspace and its matrix-dependent brief, gap, and project-status entrances
from ordinary builds while their synthesis quality contract is revised. The
implementation, saved records, APIs, exports, and full internal regression
suite remain present. Internal browser gates explicitly use
`VITE_ENABLE_EVIDENCE_MATRIX_WORKSPACE=1`; downloadable builds must leave it
unset. The immutable `v0.1.0-beta.12` tag advances `v0.1.0-beta.11` and must
never be moved or overwritten. The `v0.1.0-beta.13` candidate must pass its own
untagged Windows preflight before tagging; every later candidate must use a new
version and repeat that sequence.

## Current release decision

The project is suitable for an explicitly labeled beta release. The application can be installed or extracted and started without installing Node.js or Python. Conversion jobs are durable across restarts: queued, running, and cancelling work from an older backend session becomes an explicit recoverable task, never a permanently spinning task. Recovery remains user-triggered so restarting the app cannot silently spend model credits. The project is not yet a polished generally available desktop release because updates remain manual. The Authenticode path is ready, but an artifact is considered signed only when its manifest says `signed: true` and Windows validates its trusted publisher; without configured certificate secrets, beta artifacts remain explicitly unsigned.

The owner selected the MIT License on 2026-08-18. Root `LICENSE` carries the standard MIT terms with `Copyright (c) 2026 LittlePyx`. Formal builds and the tag workflow still require that file to be present.

## Release contract

- Canonical version: root `VERSION`, valid Semantic Versioning without a leading `v`.
- Git tag: exactly `v` plus `VERSION`.
- Frontend version: `web/package.json` and the lockfile must match `VERSION`.
- Release notes: `docs/releases/v<VERSION>.md`, reviewed bilingual Markdown that explains every attached asset before a tag can publish.
- Platform: Windows x64 current-user installer plus portable ZIP.
- Runtime: official CPython embeddable distribution at the version in `.python-version`, with backend dependencies installed into the package.
- Dependencies: exact Windows runtime versions in `requirements-release.txt`; the built package records the resolved set in `THIRD_PARTY_PACKAGES.txt` and must pass `pip check`.
- Frontend: production `web/dist`; Node.js is not included or needed at runtime.
- User data: `%LOCALAPPDATA%\Pi_zaya` by default. An explicit `KB_APP_DATA_DIR` still wins.
- Network binding: desktop launcher uses `127.0.0.1` only.
- Installer: per-user `%LOCALAPPDATA%\Programs\Pi_zaya`, no elevation, stable AppId, Start menu entry, optional desktop shortcut, in-place upgrade, and normal uninstall.
- Data preservation: installer upgrade and uninstall never remove `%LOCALAPPDATA%\Pi_zaya`; complete personal-data deletion remains a separate, explicit user action.
- Integrity: the installer and ZIP each have an adjacent SHA-256 file and JSON artifact manifest.
- Signing: optional trusted Authenticode covers `Pi_zaya.exe`, Setup, and Uninstaller with SHA-256 plus an RFC 3161 timestamp. The release manifest records the observed status, publisher, thumbprint, and timestamp. Self-signed certificates are not release signatures.
- License: MIT; `LICENSE` must be present both in the repository and inside the ZIP.
- Entry/exit: native `Pi_zaya.exe` with a system-tray safe-exit action; `Start-Pi-zaya.cmd` and `Stop-Pi-zaya.cmd` remain diagnostic fallbacks.
- Conversion durability: the conversion ledger lives in `library.sqlite3`, never stores API keys, preserves validated page-cache artifacts, and requires an explicit Continue action after a restart. A missing source PDF or vision credential stays in an actionable blocked recovery state rather than retrying in a loop.
- Conversion quality confirmation: strict blocking remains the default. A user who has inspected the current Markdown may explicitly request a fresh scan and warning-preserving index operation; the blocking codes and confirmation audit remain recorded, and detected unreliable pages remain excluded from answer evidence.
- Targeted conversion repair: a fresh page-scoped repair plan carries its exact page numbers into the durable task ledger. The worker retains validated healthy-page cache entries, snapshots the current Markdown/assets/cache, and accepts the rewrite only when the quality gate passes, every requested page is reliable, and no new blocker appears. Any failed acceptance restores the snapshot.
- Conversion concurrency: the downloadable product always submits pages in source order and uses a shared automatic provider-inflight ceiling of eight. No alternate page scheduler, adaptive page-budget branch, or text-local vision bypass is included. Higher ceilings remain explicit operator experiments and must not be enabled in a formal artifact without repeating the converter speed and structural-quality gates.
- Evidence workspace surface: ordinary downloadable builds leave `VITE_ENABLE_EVIDENCE_MATRIX_WORKSPACE` unset, so the matrix workspace and its dependent brief/gap/status entrances are absent. Internal tests set it to `1` and must keep the existing evidence workflow assertions passing while the redesign proceeds.

Development mode is unchanged: repository-local databases and preferences remain the default unless `KB_RELEASE_MODE=1` or `KB_APP_DATA_DIR` is explicitly set. There is no implicit migration of existing development data.

## Local validation build

The following fast build uses the current system Python only to validate portable staging and startup. It is not an end-user artifact and cannot be used to build the installer, which requires the embedded runtime:

```powershell
.\tools\release\build_windows_portable.ps1 `
  -OutputDir .\.runtime\release-smoke `
  -PythonRuntime System `
  -SkipFrontendBuild `
  -KeepStage `
  -AllowDirty

.\tools\release\smoke_windows_portable.ps1 `
  -BundleRoot .\.runtime\release-smoke\Pi_zaya-v0.1.0-beta.13-windows-x64 `
  -AllowDirty
```

`-AllowDirty` marks the manifest as a non-release build and must be explicit for both the builder and its local smoke. The formal workflow never enables it. `-AllowMissingLicense` only exists for mechanical tests on historical branches; it creates an internal artifact whose manifest says `missing-release-blocked`. Never distribute either kind of artifact.

## Formal local build

After the license and Inno Setup 7 are present:

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

.\tools\release\build_windows_installer.ps1 `
  -StageRoot .\release\Pi_zaya-v0.1.0-beta.13-windows-x64 `
  -InnoSetupCompiler "C:\Program Files\Inno Setup 7\ISCC.exe"

.\tools\release\smoke_windows_portable.ps1 `
  -ArchivePath .\release\Pi_zaya-v0.1.0-beta.13-windows-x64.zip `
  -CleanProfile

.\tools\release\smoke_windows_installer.ps1 `
  -InstallerPath .\release\Pi_zaya-v0.1.0-beta.13-windows-x64-setup.exe
```

Verify the final checksum independently:

```powershell
Get-FileHash .\release\Pi_zaya-v0.1.0-beta.13-windows-x64.zip -Algorithm SHA256
Get-Content .\release\Pi_zaya-v0.1.0-beta.13-windows-x64.zip.sha256
Get-FileHash .\release\Pi_zaya-v0.1.0-beta.13-windows-x64-setup.exe -Algorithm SHA256
Get-Content .\release\Pi_zaya-v0.1.0-beta.13-windows-x64-setup.exe.sha256
```

## Tag release

1. Confirm the MIT `LICENSE` and copyright line are still correct.
2. Update `VERSION`, both frontend version fields, `CHANGELOG.md`, and the reviewed bilingual `docs/releases/v<VERSION>.md` download guide.
3. Confirm the normal CI workflow passes on the exact commit.
4. Manually dispatch `.github/workflows/release-windows.yml` from that untagged commit and require the complete Windows gates, package build, and packaged-runtime smoke to pass. A manual dispatch retains verified artifacts but does not create a GitHub release.
5. Only after the untagged Windows preflight succeeds, create and push the exact tag, such as `v0.1.0-beta.13`.
6. The tag run repeats the complete backend, frontend, research, conversion, browser, portable-package, installer, and packaged-runtime gates on Windows.
7. The workflow verifies and extracts the final ZIP under an isolated Windows profile, checks `/api/health`, `/api/app/version`, `/api/settings`, and the React root, then stops it through the packaged stop command.
8. The workflow also silently installs into an isolated directory, repeats the runtime checks without system Python or Node.js, performs an in-place reinstall, uninstalls, and proves that the separate user-data sentinel remains.
9. Only after those checks pass does the workflow create a GitHub prerelease and attach both artifacts, both checksums, and both artifact manifests.

### Trusted signing configuration

The optional CI path reads `WINDOWS_SIGNING_CERT_BASE64` and `WINDOWS_SIGNING_CERT_PASSWORD` only from protected GitHub Actions secrets. The PFX is imported into the ephemeral runner's current-user certificate store immediately before packaging and removed in an `always()` cleanup step. When neither secret exists, the build remains allowed for beta but both manifests must report the real unsigned state. If only one secret exists, or signing/timestamp verification fails anywhere, packaging fails.

For a signed local build, import the trusted code-signing certificate into the current user's Personal store and pass the same `-SigningThumbprint ... -RequireSignature` options to both builders. Never put a PFX, password, or private key in the repository, build directory, ZIP, logs, or release assets.

### Failed tag recovery

Do not move or overwrite a tag after it has been pushed. If a tagged workflow fails before publishing a release, fix the cause, advance the prerelease version, and create a new tag. The `v0.1.0-beta.1` workflow attempt failed before artifact creation because Python inherited the Windows runner's legacy `cp1252` console encoding; `v0.1.0-beta.2` fixed that issue and passed the research/conversion gates, but a slow-runner hydration race then caused one research-gap browser test to click before restored project state was stable. `v0.1.0-beta.3` preserved all assertions while waiting for the intended conversation and shelf preconditions, then passed every source and browser gate; packaging exposed that the unanchored `release/` ignore rule had also excluded `tools/release/` from clean checkouts. `v0.1.0-beta.4` tracked the release tooling and passed normal CI on both its commit and tag, but its Windows tag run exposed a narrower project-context entrance race before packaging and therefore published no release. `v0.1.0-beta.5` keeps project-only basket actions unavailable until both the project context and background basket metadata updates are stable, retains first-failure browser diagnostics, pre-transforms the large browser-test routes and serializes the Windows core suite without changing its 113 tests or assertions, adds the untagged Windows preflight, and became the first successfully published downloadable beta. `v0.1.0-beta.6` adds first-run API guidance and upgrades the packaged-runtime gate to exercise the verified ZIP under an isolated, runtime-free Windows profile.

`v0.1.0-beta.7` adds an explicit Chinese portable-package guide plus safe provider/model discovery. Ambiguous credentials are never sprayed across provider endpoints; live catalog calls have a short timeout and zero retries, and both built-in model choices and free-form model IDs remain available after failure.

`v0.1.0-beta.8` makes a small native `Pi_zaya.exe` the primary Windows entrypoint. It keeps a single tray instance, opens the local browser surface, exposes log/data/open/exit actions, delegates safe backend lifecycle to the bounded PowerShell launchers, and retains the command files as diagnostic fallbacks. The launch path now detects an occupied preferred loopback port and selects another local port. The packaged-runtime smoke must invoke the EXE itself with no tray/browser, hold the preferred port open, require a different recorded port, and preserve the existing clean-profile health/version/settings/React/safe-stop checks.

`v0.1.0-beta.9` adds a standard per-user installer while retaining the portable ZIP. It uses a stable AppId and active-launcher mutex for safe in-place upgrades, creates normal Windows shortcuts, and keeps program files separate from `%LOCALAPPDATA%\Pi_zaya`. The installer gate must cover clean-profile install, embedded-runtime launch, same-version reinstall, uninstall, registration cleanup, and preserved user data. The release workflow also supports trusted Authenticode signing for the launcher, Setup, and Uninstaller; unsigned beta output remains allowed only when its manifests say so explicitly.

`v0.1.0-beta.10` adds a guided first-success path and durable, user-triggered conversion recovery. Interrupted work is reconciled from the library database after restart, validated page caches are reused, and missing sources or credentials remain actionable without retry loops. Concurrent conversions share one work-conserving provider-inflight ceiling while preserving source-order submission. The release excludes rejected scheduler, adaptive-budget, and text-local bypass experiments and requires paired structural-quality comparison for future speed candidates.

## Clean-machine acceptance

The formal Windows workflow automates both clean-start paths before any asset can be published. It verifies the ZIP and installer checksums; exercises the extracted ZIP and the installed application from fresh temporary locations; replaces the process profile and `%LOCALAPPDATA%`; removes Python and Node.js from `PATH`; clears inherited provider credentials; starts only the bundled runtime; checks the health/version/settings/React surfaces; confirms the expected isolated data path and missing-key state; and stops safely. The installer path additionally reinstalls in place, uninstalls, removes its HKCU registration, and proves that user data was not deleted.

The following hands-on workflow remains required before promotion beyond beta because it exercises real provider credentials, representative papers, user interaction, and a second physical or virtual Windows account that has neither Python nor Node.js on `PATH`:

1. Verify SHA-256 for both artifacts. Install with Setup on one account and extract the whole ZIP on the other.
2. Start from the Start menu or double-click `Pi_zaya.exe`, confirm the browser opens the library page, and confirm the Pi_zaya system-tray menu can reopen it.
3. Configure provider credentials in Settings and restart once to prove preferences survive.
4. Upload two representative PDFs and run concurrent conversion. Exit Pi_zaya while one is active, restart, confirm it is shown as interrupted rather than converting, then explicitly continue it and verify the completed-page reuse count. Cancel and retry the other document.
5. Confirm each document reports its own terminal outcome; induce or use a fixture for index retry without reconversion. Reopen the library and confirm the resumed document appears only once in the document, reference, and structured indexes.
6. Ask a grounded question, open a citation in the reader, and confirm the ordinary downloadable build exposes no evidence-matrix, research-brief, research-gap, or project-research-status entry. Separately run the unchanged internal evidence workflow browser gate with `VITE_ENABLE_EVIDENCE_MATRIX_WORKSPACE=1`.
7. Exit through the Pi_zaya system-tray menu, install the next test build over the installed version, restart, and confirm the library persists under `%LOCALAPPDATA%\Pi_zaya`. Then uninstall and confirm the same data remains. For the portable path, replace the app folder and separately retain `Stop-Pi-zaya.cmd` as a diagnostic fallback.
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

### 2026-08-18 beta.6 clean-profile acceptance

- first-run API guidance regressions: 23 passed, 1 build-mode-specific case skipped;
- complete browser smoke: 132 passed, 2 build-mode-specific cases skipped;
- core citation/library browser regressions: 113 passed;
- ordinary-user surface isolation: 4 passed;
- backend unit: 4,444 passed, 41 skipped;
- backend sanity: 266 passed, 2 skipped;
- Ruff, frontend lint/build, research QA, reviewed replay, version A/B, comparison, comparison candidates, project status/journey, and converter quality gates: pass;
- embedded ZIP clean-profile mechanics: checksum verification, fresh extraction, runtime-free `PATH`, isolated profile, cleared credential environment, default `%LOCALAPPDATA%\Pi_zaya`, health/version/settings/React checks, and packaged safe stop all pass.

The local mechanics artifact was built with `-AllowDirty` and has SHA-256 `540aac9e125dba3e0029ef1952bd27ac4918db3051362f7bdc065a6de9301ddb`; it is not distributable. The subsequent clean untagged preflight and immutable tag workflow both passed, and the published beta.6 ZIP has SHA-256 `9ad3865f4f17404ba9c743b728426087154fb9025c6819b8653c1025b5d6c3fb`.

### 2026-08-18 beta.7 provider-discovery acceptance

- dedicated settings/model-discovery regressions: 25 passed, 1 build-mode-specific case skipped;
- complete browser smoke: 134 passed, 2 build-mode-specific cases skipped;
- core citation/library browser regressions: 113 passed;
- ordinary-user surface isolation: 4 passed;
- backend unit: 4,444 passed, 41 skipped;
- backend sanity: 272 passed, 2 skipped;
- Ruff, frontend lint/build, visible Agent contract, research QA, reviewed replay, version A/B, comparison, comparison candidates, project status/journey, and converter quality gates: pass;
- ambiguous generic API keys remain local until the user selects a provider, including when Save is clicked immediately after paste; selected-provider discovery filters text/vision model categories, and discovery failures retain both built-in model choices and free-form model input;
- backend discovery uses a five-second timeout and zero retries; interactive connection tests use a twelve-second timeout and zero retries; the browser cancels either request on its own bounded deadline;
- the Windows ZIP contains the explicit root-level `README-中文.md`, and the packaged smoke gate now requires it;
- embedded ZIP clean-profile mechanics: checksum verification, fresh extraction, runtime-free `PATH`, isolated profile, cleared credential environment, default `%LOCALAPPDATA%\Pi_zaya`, health/version/settings/React checks, and packaged safe stop all pass.

The final local beta.7 mechanics artifact was rebuilt after the save-race fix with `-AllowDirty` and has SHA-256 `97eb1163f530e4295bd8d83295358b05a0715e759ff38a4e92f846dea122e01b`; it is not distributable. The subsequent normal CI, clean untagged Windows preflight, immutable tag CI, formal Windows tag workflow, and published-asset checksum/manifest/Chinese-README verification all passed. The published beta.7 ZIP has SHA-256 `e637c80ac7669fba4729113efc739800bbca5215642c00a3a67f5092b2b211ce`.

### 2026-08-18 beta.8 native-launcher acceptance

- `Pi_zaya.exe` compiles as a 36 KB Windows GUI launcher with the Pi_zaya icon, product version `0.1.0-beta.8`, and file version `0.1.0.8`;
- the launcher keeps one tray instance, reopens the existing app on a second launch, exposes open/log/data/exit actions, caps startup at 60 seconds, and waits for an in-progress startup before honoring safe exit;
- the PowerShell fallback caps readiness at 45 seconds, selects another loopback port when the preferred port is occupied, and preserves the process record if safe stop cannot confirm process exit within 10 seconds;
- the system-runtime package smoke invokes the EXE with a deliberately occupied preferred port, verifies the different recorded port plus health/version/settings/React surfaces, and safely stops the backend;
- backend unit: 4,444 passed, 41 skipped; backend sanity: 272 passed, 2 skipped; Ruff and release-foundation contract checks: pass;
- frontend lint/build: pass; complete browser smoke: 134 passed, 2 skipped; core citation/library regressions: 113 passed; ordinary-user surface isolation: 4 passed;
- research QA 56-case fixture, version A/B, comparison 5, comparison candidates 5, project status 5, project journey, grounded replay 6, reviewed replay 5, and converter quality 13: pass;
- the embedded CPython 3.10.11 ZIP passes `pip check` and the same EXE-driven clean-profile archive smoke with Python/Node removed from `PATH`, inherited provider keys cleared, isolated `%LOCALAPPDATA%`, an occupied preferred port, and packaged safe stop.

The local beta.8 mechanics artifact has SHA-256 `56512c889ddda7fccaa04f3a5b3c23ff83daa0e90937137bdb187129c9714942`. Its manifest correctly records `source_dirty=true`, so it is not distributable. Normal CI and the untagged Windows preflight subsequently passed on commit `605fb2d283e6`; the immutable tag workflow passed after rerunning the same commit when one four-worker Linux browser attempt timed out, without changing tests, workers, or timeouts. The published beta.8 ZIP has SHA-256 `cf013f161154b1c7e90a89529ce1fe91e47a7b1b44ed004728e038fcb205f5e7`.

### 2026-08-18 beta.9 installer mechanics acceptance

- Inno Setup 7.1.0 was downloaded from the official release, and its installer validated with a trusted `Pyrsys B.V.` Authenticode signature before use;
- the generated current-user Setup compiles as one x64-compatible EXE, requires no elevation, uses a stable AppId, creates Start menu/optional desktop shortcuts, and guards active launcher replacement with the same named mutex;
- an embedded beta.8 stage was used only for the first installer compile/mechanics proof; the complete beta.9 embedded stage then repeated checksum/manifest validation, silent clean install, runtime-free profile launch, occupied-port fallback, health/version/settings/React checks, same-version in-place reinstall, silent uninstall, HKCU registration cleanup, installed-launcher/uninstaller signature-state validation, and user-data preservation;
- backend unit: 4,444 passed, 41 skipped; backend sanity: 272 passed, 2 skipped; Ruff and the release-foundation contract: pass;
- frontend lint/build: pass; complete browser smoke: 134 passed, 2 skipped; core citation/library regressions: 113 passed; ordinary-user surface isolation: 4 passed;
- research QA 56-case fixture, version A/B, comparison 5, comparison candidates 5, project status 5, project journey, grounded replay 6, reviewed replay 5, and converter quality 13: pass;
- the embedded CPython 3.10.11 ZIP passed `pip check` and EXE-driven clean-profile archive smoke. Its local SHA-256 is `edc2104eb6ed87f7b02741b25b1b7802478eaaf68919270fbe8282345e5df8c5`;
- the local installer SHA-256 is `bc3ed076e9c793461b26cd0be5d2e351658230571419671439c90ac66665b67e`. It is intentionally unsigned, and its manifest reports `signed=false` and `signature_status=NotSigned`;
- the release path can optionally require one trusted certificate for the timestamped native launcher, Setup, and signed Uninstaller. The unconfigured path cannot claim or simulate a release signature.

Both local beta.9 artifacts record `source_dirty=true`, so they are mechanics proofs rather than distributable assets. A clean commit, untagged Windows preflight, and exact-tag run are still required before publication.

### 2026-08-18 durable conversion recovery acceptance

- `library.sqlite3` now owns a migration-safe conversion ledger with queued, running, cancelling, interrupted, blocked-recovery metadata, and terminal semantics; task payloads are allowlisted and never contain provider credentials;
- backend startup atomically classifies work owned by older process sessions as interrupted, while repeated startup reconciliation is idempotent and cannot interrupt current-session work;
- recovery is explicit: Continue and Continue all validate the source PDF and vision credential before requeueing the same task id. Missing prerequisites remain visible and recoverable without polling or automatic retries;
- resumed conversions keep `.conversion_cache`, including interrupted quality-repair runs, and expose the number of completed pages available for reuse;
- the durable terminal transition is compare-and-set and the in-memory queue rejects a second resume, so one recovery action cannot start duplicate conversion or indexing work;
- deleting a source dismisses its interrupted recovery, while renaming a source moves the recoverable ledger path with it;
- a spawned-process crash regression writes a running job, terminates without cleanup, restarts with a new session, and verifies recovery of the same task id and persisted page progress;
- exact-tree backend unit: 4,468 passed, 41 skipped; backend sanity: 274 passed, 2 skipped; Ruff and `git diff --check`: pass;
- frontend lint/build: pass; browser smoke: 136 passed, 2 skipped; core citation/library regressions: 113 passed; ordinary-user surface isolation: 4 passed;
- Research QA fixture/full-library dry runs: 56 and 29 cases; source grounding: 41/41; grounded replay: 6/6; Agent golden contract: 10 cases, zero errors; reviewed Agent replay: 5/5;
- paired comparison: 5/5 with zero false comparisons; comparison candidates: 5/5 with 18 discoveries and zero contract/evidence/prefill failures at `test_results/evidence_comparison_candidates/20260818_201707`;
- project status: 5/5 at `test_results/project_research_status/20260818_201707`; project journey: 20/20 at `test_results/project_research_journey/20260818_201727`; converter quality: 13/13 at `test_results/converter_quality_eval/20260818_201705`.

The complete automated release-quality gate passed on the exact working tree. A distributable build still requires a clean committed tree, clean-machine restart/resume acceptance, and the normal untagged/tagged release path.

### 2026-08-19 conversion-speed boundary

- the accepted two-document dynamic path remains global inflight eight, four page workers, three repair workers, and source-order page scheduling;
- exploratory inflight 12 failed the fixed-paper quality gate, while 16 supplied no material advantage over the accepted three-repeat inflight-eight evidence, so high-core hosts no longer raise the automatic ceiling;
- full text-density scheduling was about 24% faster across the two-paper sample but lost images/display math on all three Nature pairs;
- a one-page, locally screened tail promotion still made Nature 6.4% slower and dropped four image links, so neither scheduler advanced to the ten-paper rollout or product default;
- both failed scheduler implementations and every runtime/configuration entry point were removed; the older no-go adaptive page-budget and text-local bypass implementations were removed by the same quality-first audit;
- only the rejected experiments' benchmark reports and decision records remain as historical evidence; none can be enabled through runtime, environment, benchmark, or throughput configuration;
- paired converter benchmarks now record structural regression flags and can fail after artifact capture with `--fail-on-paired-quality-regression`.

The detailed commands, raw-artifact paths, and timing tables are in
`docs/CONVERTER_LLM_SPEED_PLAN.md`, section 27. This decision preserves the
downloadable beta's existing conversion-quality contract; it does not claim an
additional default speedup beyond the already accepted two-document dynamic
coordinator.

The exact-tree release gates then passed after all rejected speed branches were
removed: backend unit 4,452 passed with 41
skipped, sanity 274 passed with two skipped, Agent runtime 5/5, converter quality
13/13, Ruff, frontend lint/build, browser smoke 136 with two build-mode skips,
core browser 113/113, public-surface 4/4, and every existing research fixture
and replay gate. The complete browser sequence passed in one fail-fast run after
the scheduler code was removed.

### 2026-08-19 beta.10 publication acceptance

- the immutable annotated tag `v0.1.0-beta.10` points to commit `fd4519ec9ace7987819422d1c13bf8157e6b234e`, whose source tree is clean and whose canonical/backend/frontend version contract is `0.1.0-beta.10`;
- normal main-branch CI run `32234230467` and the complete untagged Windows preflight run `32235026219` passed on that exact commit before the tag was created;
- formal Windows tag run `32236592990` repeated every backend, research, conversion, frontend, clean-profile ZIP, installer, in-place upgrade, uninstall, data-preservation, checksum, and signature-declaration gate before publishing the prerelease;
- the independent tag CI run `32236592953` initially reached its 20-minute Ubuntu job limit while installing Playwright, before any browser test ran. Its failed job was rerun on the same immutable tag without changing code, workers, tests, assertions, or timeouts; the second attempt passed the complete CI summary;
- the published ZIP is 94,813,511 bytes with SHA-256 `95d95589d3412644cf30805d1f0488c2b30971295354f400ad314b8158051712`;
- the published Setup is 60,875,545 bytes with SHA-256 `dcd2c717e2624e7d2a47a4c9bb4a9fc6f2fcb991e0e5aca636d5651549c41f15`;
- both published manifests record commit `fd4519ec9ace`, `source_dirty=false`, and `license=MIT`. No trusted signing certificate was configured, so Setup, launcher, and Uninstaller are explicitly recorded as `NotSigned` rather than implying publisher trust;
- the GitHub prerelease contains exactly the ZIP, Setup, their two adjacent SHA-256 files, and their two machine-readable manifests.

### 2026-08-20 beta.11 publication acceptance

- the immutable annotated tag `v0.1.0-beta.11` points to commit `a80c4c37144f52ff6e4f6524b840d462388f1ca3`, whose source tree is clean and whose canonical/backend/frontend version contract is `0.1.0-beta.11`;
- normal main-branch CI run `32269444566` initially reached its 20-minute Ubuntu job limit while the Playwright dependency installer waited on the Ubuntu package mirror, before any browser test ran. Only the failed frontend job was rerun on the same commit; its complete frontend gates and CI summary passed. The complete untagged Windows preflight run `32272653921` then passed on that exact commit before the tag was created;
- formal Windows tag run `32274511583` repeated every backend, research, conversion, frontend, clean-profile ZIP, installer, in-place upgrade, uninstall, data-preservation, checksum, and signature-declaration gate before publishing the prerelease at `2026-08-19T16:30:56Z`;
- the independent tag CI run `32274511524` encountered the same 20-minute Ubuntu package-mirror limit while installing Playwright, before any browser test ran. Only its failed frontend job was rerun on the same immutable tag without changing code, workers, tests, assertions, or timeouts; the second attempt passed the complete CI summary;
- the published ZIP is 94,823,315 bytes with SHA-256 `7aa849277c693efa7e30916a12ddb5359e9efc9c32ee96b92713810bb8018801`;
- the published Setup is 60,874,584 bytes with SHA-256 `46e42c87ee2e5b9b17540614a63468f3b0edc5701d09837a1dbc63e9657dfb10`;
- both published manifests record commit `a80c4c37144f`, `source_dirty=false`, `license=MIT`, and the embedded Python runtime. No trusted signing certificate was configured, so Setup, launcher, and Uninstaller are explicitly recorded as `NotSigned` rather than implying publisher trust;
- the GitHub prerelease contains exactly the ZIP, Setup, their two adjacent SHA-256 files, and their two machine-readable manifests. All six assets were independently downloaded after publication, and both artifact hashes matched their adjacent checksum files.

### 2026-08-23 beta.12 candidate preparation

- the strict conversion-quality gate remains the default, while a user who has inspected the current Markdown now has an explicit fresh-scan and warning-preserving indexing path after repeated conservative blocking;
- confirmed documents retain their blocking issue codes and audit marker, and detected unreliable pages remain excluded from answer evidence;
- the implementation commit passed the backend unit and sanity suites, converter quality 13/13, Ruff, frontend lint/build, browser smoke/core/public-surface gates, and the dedicated 20-case library-quality browser suite;
- the beta.12 version contract, bilingual release notes, Ruff, frontend lint/build, and 11 release-foundation tests pass locally;
- at this preparation checkpoint, beta.12 remained an untagged release candidate pending normal CI and the complete untagged Windows preflight on the exact clean release commit. No beta.12 download or checksum was represented as published before that point.

### 2026-08-24 beta.12 targeted-repair acceptance

- fresh page-scoped quality plans now preserve their diagnosed page numbers through the API queue, in-memory state, and durable conversion ledger;
- targeted reconversion reuses validated healthy-page cache entries and snapshots the complete previous conversion output before replacing generated files;
- a repaired result is accepted only after a fresh strict quality scan confirms every requested page is reliable and no new blocking code appears; otherwise the previous Markdown, assets, and page cache are restored and the rollback is recorded;
- the library quality row exposes the affected pages and basic per-page diagnostic evidence without changing document-scoped repair behavior;
- backend unit passed 4,458 with 41 skips, backend sanity passed 275 with 2 skips, and the focused conversion-task suite passed 296 with 2 skips;
- Ruff, the visible Research Agent contract, all CI research/evidence fixture gates, reviewed replay, frontend lint/build, browser smoke (136 passed, 3 build-mode skips), core citation/library browser regressions (113/113), ordinary-user isolation (4/4), evidence-workspace release isolation (1/1), and the dedicated library-quality suite (21/21) pass locally;
- full local converter quality passed 13/13 at `test_results/converter_quality_eval/20260824_131423`;
- at this targeted-repair checkpoint, beta.12 remained untagged pending normal CI and the complete untagged Windows preflight on the exact clean release commit.

### 2026-08-24 beta.12 publication acceptance

- the immutable annotated tag `v0.1.0-beta.12` points to commit `a6ecda9a300dbd966614839b1a1e22a2480ad4e4`, whose source tree is clean and whose canonical/backend/frontend version contract is `0.1.0-beta.12`;
- normal main-branch CI run `32702792498` and the complete untagged Windows preflight run `32703524820` passed on that exact commit before the tag was created;
- independent tag CI run `32708247157` and formal Windows tag run `32708247168` passed every backend, research, conversion, frontend, clean-profile ZIP, installer, in-place upgrade, uninstall, data-preservation, checksum, and signature-declaration gate;
- the GitHub prerelease was published at `2026-08-24T09:08:49Z` with exactly six attached release assets;
- the published ZIP is 94,836,135 bytes with SHA-256 `f45a097008a5b91d8bfa4ceef2cf4b35427bdeb3d294279253fb48cfb4a880d2`;
- the published Setup is 60,891,858 bytes with SHA-256 `fb134b8a28229f9df470c9edd532a0daaefe490a9738b404e49f347fce843926`;
- both published manifests record commit `a6ecda9a300d`, `source_dirty=false`, `license=MIT`, and the embedded Python runtime. No trusted certificate was configured, so Setup, launcher, and Uninstaller are explicitly recorded as `NotSigned`;
- all six assets were downloaded again into a fresh verification directory. Both binaries matched their adjacent checksum files, the downloaded installer reported `NotSigned`, and the ZIP contained `VERSION=0.1.0-beta.12`, `LICENSE`, `Pi_zaya.exe`, and the 3,537-character Chinese guide covering Setup, portable use, API Key configuration, SHA-256 verification, and startup.

### 2026-08-25 beta.13 candidate acceptance

- the candidate adds persistent research notes, direct reader capture for selected text, Markdown tables, equations, and figures, exact source return, duplicate-range prevention, outline composition, and Markdown/Word export with Unicode-safe title filenames;
- a real NatCommun single-photon-imaging paper was used to add a figure, table, and equation to one note, reopen the exact paper source, reject a repeated source range, and verify that the note and its three sources survived an application restart;
- the first-use path now offers a self-authored sample PDF and distinguishes a document that is still converting from a profile that has not imported one;
- `git diff --check`, Ruff, frontend lint/build, and the 11 release-foundation tests pass on the candidate tree;
- backend unit passed 4,487 with 41 skips, and backend sanity passed 280 with 2 skips;
- all research/evidence fixture dry runs, 6/6 grounded replay, 5/5 reviewed replay, and converter quality 13/13 pass;
- browser smoke passed 144 with 3 build-mode skips, core citation/library regressions passed 113/113, ordinary-user isolation passed 4/4, evidence-workspace release isolation passed 1/1, and the dedicated library-quality suite passed 21/21;
- at this checkpoint, beta.13 remains untagged pending normal CI and the complete untagged Windows preflight on the exact clean release commit. No beta.13 download, checksum, or GitHub Release is represented as published before those gates pass.

## Promotion gates after beta

- Acquire and protect a trusted public code-signing certificate before promotion beyond unsigned beta; exercise the required-signature CI branch with the real publisher identity.
- Define an automatic or guided updater with rollback and data-schema compatibility checks.
- Exercise backup/restore and data migration across consecutive real release versions.
- Expand clean-machine coverage beyond the GitHub-hosted Windows image.
