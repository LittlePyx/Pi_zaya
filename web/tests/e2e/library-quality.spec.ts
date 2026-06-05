import { expect, test, type Page } from '@playwright/test'

async function expectNoHorizontalOverflow(page: Page) {
  const metrics = await page.evaluate(() => ({
    bodyScrollWidth: document.body.scrollWidth,
    bodyClientWidth: document.body.clientWidth,
    docScrollWidth: document.documentElement.scrollWidth,
    docClientWidth: document.documentElement.clientWidth,
  }))
  expect(metrics.bodyScrollWidth, 'body should not create horizontal scroll').toBeLessThanOrEqual(metrics.bodyClientWidth + 2)
  expect(metrics.docScrollWidth, 'document should not create horizontal scroll').toBeLessThanOrEqual(metrics.docClientWidth + 2)
}

const baseItem = {
  sha1: '',
  task_state: 'idle',
  status: 'converted',
  replace_task: false,
  queue_pos: 0,
  cur_page_done: 0,
  cur_page_total: 0,
  cur_page_msg: '',
  paper_category: 'Single-Photon Imaging',
  reading_status: 'reading',
  note: '',
  user_tags: ['converter'],
  has_suggestions: false,
  suggested_category: '',
  suggested_tags: [],
}

test.beforeEach(async ({ page }) => {
  const brokenName = 'Optica-2024-Broken conversion.pdf'
  const weakName = 'Applied Optics-2023-Weak anchors.pdf'
  const readerLocateSourcePath = 'F:\\kb\\md\\weak\\weak.en.md'
  const repairRequestedNames = new Set<string>()
  const repairCompletedNames = new Set<string>()
  const isRepairing = (name: string) => repairRequestedNames.has(name) && !repairCompletedNames.has(name)
  let latestRepairRun: Record<string, unknown> | null = null
  let shelfMetadataBackfillState: Record<string, unknown> = {
    ok: true,
    job_id: 'shelf-meta-fixture',
    status: 'completed',
    phase: 'completed',
    running: false,
    progress: { percent: 100, processed: 2, total: 2 },
    scan: {
      ok: true,
      docs: 3,
      scanned: 42,
      ready: 28,
      export_ready: 24,
      needs_repair: 18,
      repairable: 12,
      retryable: 12,
      target_count: 12,
      returned_count: 12,
      target_limit: 120,
      truncated: false,
      missing_fields: [
        { name: 'authors', count: 8 },
        { name: 'venue', count: 6 },
        { name: 'year', count: 3 },
      ],
      issue_codes: [],
      sources: [],
      targets: [],
    },
    result: {
      ok: true,
      requested: 2,
      ready: 2,
      export_ready: 2,
      partial: 0,
      retryable: 0,
      failed: 0,
      unresolved: 0,
      changed: 1,
      persisted: 2,
      preheated: 2,
      remaining_targets: 16,
      items: [],
      verification: {
        type: 'shelf_metadata_repair',
        status: 'passed',
        quality_ok: true,
        target_count: 2,
        metadata_ready_after: 2,
        export_ready_after: 2,
        changed: 1,
        retryable: 0,
        failed: 0,
        unresolved_after: 0,
      },
    },
  }
  await page.addInitScript(() => {
    window.localStorage.removeItem('kb.library.qualityRepairHistory.v1')
  })
  await page.route('**/api/settings', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model: 'test-model',
        base_url: '',
        has_api_key: true,
        db_dir: 'F:\\kb\\db',
        prefs: {
          pdf_dir: 'F:\\kb\\pdfs',
          md_dir: 'F:\\kb\\md',
          theme: 'dark',
          ui_locale: 'zh',
        },
      }),
    })
  })
  await page.route('**/api/projects', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) })
  })
  await page.route('**/api/conversations**', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) })
  })
  await page.route('**/api/references/sync/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'data: {"running":false,"done":true,"status":"idle","stage":"","message":"","current":"","docs_done":0,"docs_total":0}\n\n',
    })
  })
  await page.route('**/api/references/shelf/metadata/backfill/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(shelfMetadataBackfillState),
    })
  })
  await page.route('**/api/references/shelf/metadata/backfill/start', async (route) => {
    shelfMetadataBackfillState = {
      ...shelfMetadataBackfillState,
      status: 'running',
      phase: 'repairing',
      running: true,
      progress: { percent: 34, processed: 0, total: 12 },
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ started: true, job_id: 'shelf-meta-fixture-2', state: shelfMetadataBackfillState }),
    })
  })
  await page.route('**/api/library/quality/overview**', async (route) => {
    const brokenDone = repairCompletedNames.has(brokenName)
    const weakDone = repairCompletedNames.has(weakName)
    const review = (brokenDone ? 0 : 1) + (weakDone ? 0 : 1)
    const good = 1 + (brokenDone ? 1 : 0) + (weakDone ? 1 : 0)
    const recommended = [
      brokenDone ? null : {
        name: brokenName,
        path: `F:\\kb\\pdfs\\${brokenName}`,
        md_path: 'F:\\kb\\md\\broken\\broken.en.md',
        status: 'error',
        score: 38,
        summary: 'Needs repair | Q38 | 0 pages | 0 refs | 1 figures | 1 math',
        task_state: 'idle',
        issues: [
          { code: 'missing_images', label: 'Missing image assets', severity: 'error', count: 1 },
          { code: 'unclosed_display_math', label: 'Unclosed display math', severity: 'error', count: 1 },
        ],
      },
      weakDone ? null : {
        name: weakName,
        path: `F:\\kb\\pdfs\\${weakName}`,
        md_path: 'F:\\kb\\md\\weak\\weak.en.md',
        status: 'warning',
        score: 55,
        summary: 'Needs review | Q55 | 4 pages | 8 refs | 2 figures | 3 math',
        task_state: 'idle',
        issues: [
          { code: 'missing_page_markers', label: 'Missing page markers', severity: 'warning', count: 7 },
        ],
      },
    ].filter(Boolean)
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        status: review > 0 ? 'error' : 'good',
        summary: {
          total_view: 4,
          total_all: 4,
          converted: 3,
          pending: 1,
          queued: 0,
          running: 0,
          assessed: 3,
          good,
          review,
          unknown: 0,
          avg_score: review > 1 ? 70 : (review > 0 ? 82 : 95),
        },
        top_issues: [
          brokenDone ? null : { code: 'missing_images', label: 'Missing image assets', severity: 'error', papers: 1, count: 1 },
          weakDone ? null : { code: 'missing_page_markers', label: 'Missing page markers', severity: 'warning', papers: 1, count: 7 },
        ].filter(Boolean),
        recommended,
        domains: {
          conversion: {
            available: true,
            status: review > 0 ? 'error' : 'good',
            summary: {
              converted: 3,
              assessed: 3,
              good,
              review,
              unknown: 0,
              avg_score: review > 1 ? 70 : (review > 0 ? 82 : 95),
            },
            top_failures: [
              brokenDone ? null : { name: 'Missing image assets', count: 1 },
              weakDone ? null : { name: 'Missing page markers', count: 1 },
            ].filter(Boolean),
          },
          research_qa: {
            available: true,
            status: 'error',
            summary: { total: 4, passed: 3, failed: 1 },
            top_failures: [{ name: 'refs_include_required_docs', count: 1 }],
            latest_path: 'test_results/research_qa_eval/latest',
            report_path: 'test_results/research_qa_eval/latest/report.md',
            updated_at: 1,
          },
          citation_cards: {
            available: true,
            status: 'error',
            summary: {
              tracked_checks: 8,
              failed_checks: 2,
              citation_card_failed: 1,
              shelf_failed: 1,
              ref_card_failed: 0,
              system_b_failed: 0,
              shelf_item_count: 2,
              shelf_metadata_ready_count: 1,
              shelf_export_ready_count: 1,
              shelf_summary_export_ready_count: 1,
              shelf_doi_count: 1,
              shelf_source_clickable_count: 2,
              shelf_review_count: 1,
            },
            top_failures: [{ name: 'citation_card_quality', count: 1 }],
            latest_path: 'test_results/research_qa_eval/latest',
            updated_at: 1,
          },
          reader_locate: {
            available: true,
            status: 'error',
            summary: { total: 2, exact: 1, block: 0, degraded: 0, failed: 1, repairable: 1, strict_miss: 1, affected_sources: 1 },
            top_failures: [{ name: 'target anchor drift', count: 1 }],
          },
        },
        reader_locate: {
          available: true,
          status: 'error',
          summary: { total: 2, exact: 1, block: 0, degraded: 0, failed: 1, repairable: 1, strict_miss: 1, affected_sources: 1 },
          top_failures: [{ name: 'target anchor drift', count: 1 }],
          recommended_sources: [
            {
              source_path: readerLocateSourcePath,
              source_name: 'weak.en.md',
              pdf_path: `F:\\kb\\pdfs\\${weakName}`,
              md_path: readerLocateSourcePath,
              md_exists: true,
              total: 2,
              failed: 1,
              degraded: 0,
              repairable: 1,
              strict_miss: 1,
              latest_status: 'failed',
              latest_precision: 'failed',
              latest_reason: 'target anchor drift',
              latest_at: 1790000000,
              recommended_action: 'repair_conversion_and_reindex',
            },
          ],
          latest: [],
        },
        full_chain: {
          available: true,
          status: 'error',
          score: review > 0 ? 34 : 56,
          summary: review > 0
            ? '4 blocking stages need source-level repair before the app is release-ready.'
            : '3 blocking stages need source-level repair before the app is release-ready.',
          stages: [
            {
              key: 'conversion',
              label: 'PDF conversion',
              status: review > 0 ? 'error' : 'good',
              detail: review > 0 ? `${recommended.length} sources need conversion repair` : 'Q95 average conversion score',
              action: review > 0 ? 'repair_conversion' : 'monitor_conversion',
              count: recommended.length,
              blocking: review > 0,
              metrics: { review, unknown: 0, avg_score: review > 1 ? 70 : (review > 0 ? 82 : 95) },
            },
            {
              key: 'research_qa',
              label: 'Research QA',
              status: 'error',
              detail: '1 failed / 4 QA cases',
              action: 'fix_failed_qa_cases',
              count: 1,
              blocking: true,
            },
            {
              key: 'retrieval',
              label: 'Retrieval coverage',
              status: 'error',
              detail: '1 QA cases missed required retrieval docs',
              action: 'rebuild_index',
              count: 1,
              blocking: true,
            },
            {
              key: 'citations',
              label: 'Citation cards',
              status: 'error',
              detail: '1 citation/card checks failed',
              action: 'repair_citation_cards',
              count: 1,
              blocking: true,
            },
            {
              key: 'shelf',
              label: 'Literature basket',
              status: 'error',
              detail: '1 literature basket checks failed; export-ready 1/2, summaries 1/2',
              action: 'repair_shelf_metadata',
              count: 1,
              blocking: true,
              metrics: { items: 2, export_ready: 1, summary_export_ready: 1 },
            },
            {
              key: 'repair_loop',
              label: 'Repair verification',
              status: 'warning',
              detail: '2 failed/error reruns; latest failed',
              action: 'rerun_failed_cases',
              count: 2,
              blocking: false,
            },
          ],
          root_causes: [
            brokenDone ? null : { code: 'missing_images', label: 'Missing image assets', domain: 'conversion', count: 1, severity: 'error' },
            { code: 'retrieval_missing_expected_docs', label: 'Retrieval missed required documents', domain: 'research_qa', count: 1, severity: 'error' },
            { code: 'citation_card_quality', label: 'Citation card or basket copy is weak', domain: 'citation_cards', count: 1, severity: 'error' },
          ].filter(Boolean),
          action_history: [
            {
              id: 'hist-reindex',
              stage_key: 'retrieval',
              stage_label: 'Retrieval coverage',
              action: 'rebuild_index',
              status: 'success',
              summary: 'Rebuilt retrieval index',
              detail: 'Next QA check: scinerf-admm-origin',
              target_ids: ['scinerf-admm-origin'],
              metrics: { target_count: 1 },
              before: { status: 'error', score: 42, count: 1, detail: '1 missed doc' },
              after: { status: 'good', score: 96, count: 0, detail: 'coverage passed' },
              delta: { improved: true, score_delta: 54, count_delta: 1, summary: 'Improved: error -> good' },
              improved: true,
              verification: { type: 'research_qa_rerun', case_id: 'scinerf-admm-origin', quality_ok: true },
              created_at: 1790000200,
            },
          ],
          next_actions: [
            ...(review > 0 ? [{ domain: 'conversion', severity: 'error', label: 'Repair conversion quality', count: recommended.length }] : []),
            { domain: 'research_qa', severity: 'error', label: 'Fix failed research QA cases', count: 1 },
            { domain: 'citation_cards', severity: 'error', label: 'Fix citation and card quality', count: 2 },
          ],
        },
        feature_health: {
          available: true,
          status: 'error',
          score: review > 0 ? 56 : 64,
          summary: '6 user-facing workflows need attention before the product feels reliable.',
          items: [
            {
              key: 'pdf_conversion',
              label: 'PDF conversion',
              status: review > 0 ? 'error' : 'good',
              score: review > 1 ? 70 : (review > 0 ? 82 : 95),
              summary: review > 0 ? `${review} sources need conversion review` : 'Markdown is ready for retrieval',
              detail: 'Readable Markdown, page markers, figures, formulas, and references.',
              action: review > 0 ? 'repair_conversion' : 'review_conversion',
              target_stage: 'conversion',
              count: review,
              blocking: review > 0,
              metrics: { review, unknown: 0 },
            },
            {
              key: 'general_qa',
              label: 'General QA',
              status: 'error',
              score: 75,
              summary: '1 failed / 4 research QA cases',
              detail: 'Checks whether user questions retrieve the right papers and cite usable evidence.',
              action: 'fix_failed_qa_cases',
              target_stage: 'research_qa',
              count: 1,
              blocking: true,
            },
            {
              key: 'paper_guide',
              label: 'Paper Guide',
              status: 'error',
              score: 38,
              summary: 'Deep-reading quality is limited by source, retrieval, or citation failures',
              detail: 'Depends on conversion quality, focused retrieval, figure/equation/source grounding, and citation surfacing.',
              action: 'inspect_paper_guide',
              target_stage: 'research_qa',
              count: 4,
              blocking: true,
            },
            {
              key: 'citation_cards',
              label: 'Citation cards',
              status: 'error',
              score: 45,
              summary: '1 citation/card checks failed',
              detail: 'Tracks title, source, evidence quote, claim support, System B mapping, and card copy quality.',
              action: 'repair_citation_cards',
              target_stage: 'citations',
              count: 1,
              blocking: true,
            },
            {
              key: 'literature_basket',
              label: 'Literature basket',
              status: 'error',
              score: 44,
              summary: '1 basket checks failed; export-ready 1/2',
              detail: 'Checks structured DOI, authors, venue, grounded summary, source-open, and export readiness.',
              action: 'repair_shelf_metadata',
              target_stage: 'shelf',
              count: 1,
              blocking: true,
              metrics: { shelf_failed: 1, items: 2, metadata_ready: 1, export_ready: 1, summary_export_ready: 1, doi: 1, source_clickable: 2, review: 1 },
            },
            {
              key: 'reader_locate',
              label: 'Reader locate',
              status: 'error',
              score: 43,
              summary: 'Reader locate may be affected by weak citations or source conversion',
              detail: 'Covers citation click-through, source opening, anchors, page markers, and evidence snippets.',
              action: 'inspect_reader_locate',
              target_stage: 'citations',
              count: 2,
              blocking: true,
            },
            {
              key: 'repair_loop',
              label: 'Repair loop',
              status: 'warning',
              score: 76,
              summary: '2 failed/error reruns need follow-up',
              detail: 'Confirms that source repair, metadata repair, reindex, and QA rerun actually improved results.',
              action: 'rerun_failed_cases',
              target_stage: 'repair_loop',
              count: 2,
              blocking: false,
            },
          ],
        },
        priority_actions: [
          ...(review > 0 ? [{ domain: 'conversion', severity: 'error', label: 'Repair conversion quality', count: recommended.length }] : []),
          { domain: 'reader_locate', severity: 'error', label: 'Repair reader locate sources', count: 1 },
          { domain: 'research_qa', severity: 'error', label: 'Fix failed research QA cases', count: 1 },
          { domain: 'citation_cards', severity: 'error', label: 'Fix citation and card quality', count: 2 },
        ],
        rerun_summary: {
          available: true,
          total: 3,
          passed: 1,
          failed: 2,
          error: 0,
          case_count: 2,
          latest_finished_at: 1790000100,
          latest_status: 'failed',
          top_failures: [{ name: 'refs_include_required_docs', count: 2 }],
        },
        repair_runs: latestRepairRun ? [latestRepairRun] : [],
        failure_cases: [
          {
            id: 'scinerf-admm-origin',
            question: 'ADMM 是作者自己发明的吗？',
            status: 'done',
            conv_id: 'qa-conv-1',
            latency_ms: 1234,
            failures: [
              { name: 'citation_card_quality', domain: 'citation_cards', detail: 'missing title' },
              { name: 'refs_include_required_docs', domain: 'research_qa', detail: 'scinerf' },
            ],
            failure_names: ['citation_card_quality', 'refs_include_required_docs'],
            expected_doc_ids: ['scinerf', 'scigs'],
            ref_doc_ids: ['scinerf'],
            citation_doc_ids: [],
            missing_expected_doc_ids: ['scigs'],
            doc_ids: ['scinerf', 'scigs'],
            citation_count: 1,
            system_b_count: 0,
            ref_hit_count: 2,
            diagnostic_summary: {
              citation_routes: { system_a: 1, system_b: 0 },
              missing_expected_doc_count: 1,
              citation_diagnostic_count: 1,
              ref_diagnostic_count: 1,
              shelf_metadata_repair_target_count: 1,
              shelf_missing_export_fields: [{ name: 'doi', count: 1 }],
            },
            citation_diagnostics: [
              {
                route: 'system_a',
                num: 1,
                anchor: 'scinerf-a1',
                title: 'SCINeRF citation',
                source_name: 'SCINeRF',
                source_path: 'db/scinerf/scinerf.en.md',
                heading_path: 'Method / ADMM',
                evidence_quote: 'The reconstruction uses ADMM as an optimization component.',
                raw: 'Author A. SCINeRF upstream optimization reference. Journal A, 2024. doi:10.1234/scinerf-ref',
                metadata_missing_fields: ['doi'],
                metadata_repairable: true,
              },
            ],
            ref_diagnostics: [
              {
                title: 'SCINeRF ref card',
                source_name: 'SCINeRF',
                source_path: 'db/scinerf/scinerf.en.md',
                heading_path: 'Method / ADMM',
                score: 8.8,
                summary_line: 'SCINeRF explains how ADMM is used in the reconstruction pipeline.',
                why_line: 'This card should be visible for the ADMM origin question.',
                polish_status: 'full',
                ref_pack_state: 'ready',
                evidence_quote: 'ADMM appears as an optimization solver.',
              },
            ],
            source_diagnostics: [
              {
                source_path: 'db/scinerf/scinerf.en.md',
                source_name: 'SCINeRF',
                title: 'SCINeRF citation',
                roles: ['citation:system_a', 'reference_basket'],
                pdf_path: 'F:\\kb\\pdfs\\SCINeRF.pdf',
                md_path: 'F:\\kb\\md\\scinerf\\scinerf.en.md',
                md_exists: true,
                repairable: true,
                needs_repair: true,
                quality_status: 'warning',
                quality_score: 74,
                quality_summary: 'Needs review | Q74 | 4 pages | 12 refs | 2 figures | 3 math',
                quality_issues: [{ code: 'missing_page_markers', label: 'Missing page anchors', severity: 'warning', count: 1 }],
              },
            ],
            root_causes: [
              { code: 'source_conversion_quality', label: 'Source conversion needs repair', severity: 'warning', detail: '1 related source needs repair.', action: 'repair_sources' },
              { code: 'retrieval_missing_expected_docs', label: 'Retrieval missed required documents', severity: 'error', detail: 'Missing expected docs: scigs', action: 'rebuild_index' },
              { code: 'citation_card_quality', label: 'Citation card or basket copy is weak', severity: 'error', detail: 'Card copy failed.', action: 'inspect_cards' },
              { code: 'shelf_metadata_export_fields', label: 'Literature basket metadata is not export-ready', severity: 'error', detail: 'Missing structured fields: doi x1.', action: 'repair_shelf_metadata' },
            ],
            shelf_metadata_missing_fields: [{ name: 'doi', count: 1 }],
            shelf_metadata_repair_targets: [
              {
                key: 'scinerf-admm-origin:shelf-meta:0',
                source_path: 'db/scinerf/scinerf.en.md',
                source_name: 'SCINeRF',
                title: 'SCINeRF citation',
                raw: 'Author A. SCINeRF upstream optimization reference. Journal A, 2024. doi:10.1234/scinerf-ref',
                metadata_missing_fields: ['doi'],
                repair_target_kind: 'system_b_citation',
              },
            ],
            repair_actions: [
              {
                id: 'apply_repair_plan',
                kind: 'apply_repair_plan',
                label: 'Fix from source',
                severity: 'error',
                enabled: true,
                source_count: 1,
                detail: 'Repair source conversions -> Repair citation and shelf metadata -> Rebuild retrieval index -> Rerun QA acceptance',
                steps: [
                  { kind: 'repair_sources', label: 'Repair source conversions', source_count: 1 },
                  { kind: 'repair_shelf_metadata', label: 'Repair citation and shelf metadata', target_count: 1, missing_fields: [{ name: 'doi', count: 1 }] },
                  { kind: 'rebuild_index', label: 'Rebuild retrieval index' },
                  { kind: 'rerun_case', label: 'Rerun QA acceptance' },
                ],
                acceptance: 'The case is rerun after repairs.',
              },
              { id: 'open_replay', kind: 'open_replay', label: 'Open replay', severity: 'warning', enabled: true, detail: 'Inspect replay.' },
              { id: 'rerun_case', kind: 'rerun_case', label: 'Rerun case', severity: 'error', enabled: true, detail: 'Run this case again.' },
              { id: 'repair_sources', kind: 'repair_sources', label: 'Repair sources', severity: 'error', enabled: true, source_count: 1, detail: 'Reconvert related sources.' },
              { id: 'rebuild_index', kind: 'rebuild_index', label: 'Rebuild index', severity: 'error', enabled: true, detail: 'Refresh index.' },
              { id: 'open_raw', kind: 'open_artifact', target: 'raw', label: 'Open raw QA', severity: 'warning', enabled: true, detail: 'Inspect raw.' },
            ],
            rerun_status: {
              available: true,
              run_count: 2,
              last_status: 'failed',
              last_quality_ok: false,
              last_finished_at: 1790000100,
              last_latency_ms: 2100,
              last_passed_at: 1790000000,
              consecutive_failed: 1,
              failure_names: ['refs_include_required_docs'],
              report_path: 'test_results/research_qa_eval/rerun/report.md',
              raw_path: 'test_results/research_qa_eval/rerun/raw_results.jsonl',
            },
            answer_preview: 'preview',
          },
          {
            id: 'hadamard-fourier-choice',
            question: 'Hadamard 和 Fourier 到底该怎么选？',
            status: 'done',
            conv_id: 'qa-conv-2',
            latency_ms: 2345,
            failures: [
              { name: 'citation_card_quality', domain: 'citation_cards', detail: 'weak summary' },
            ],
            failure_names: ['citation_card_quality'],
            expected_doc_ids: ['hsi-fsi'],
            ref_doc_ids: ['hsi-fsi'],
            citation_doc_ids: ['hsi-fsi'],
            missing_expected_doc_ids: [],
            doc_ids: ['hsi-fsi'],
            citation_count: 2,
            system_b_count: 1,
            ref_hit_count: 3,
            diagnostic_summary: {
              citation_routes: { system_a: 1, system_b: 1 },
              missing_expected_doc_count: 0,
              citation_diagnostic_count: 2,
              ref_diagnostic_count: 1,
            },
            citation_diagnostics: [
              {
                route: 'system_b',
                num: 2,
                anchor: 'hsi-b1',
                title: 'Hadamard versus Fourier SPI',
                source_name: 'HSI vs FSI',
                source_path: 'db/hsi-fsi/hsi-fsi.en.md',
                heading_path: 'Reference context',
                evidence_quote: 'The paper compares sampling bases for SPI.',
              },
            ],
            ref_diagnostics: [
              {
                title: 'HSI vs FSI ref card',
                source_name: 'HSI vs FSI',
                source_path: 'db/hsi-fsi/hsi-fsi.en.md',
                heading_path: 'Comparison',
                score: 9.2,
                summary_line: 'Hadamard and Fourier basis choices are compared.',
                why_line: 'Relevant for the basis selection question.',
                polish_status: 'full',
                ref_pack_state: 'ready',
                evidence_quote: 'Hadamard and Fourier sampling are evaluated.',
              },
            ],
            source_diagnostics: [
              {
                source_path: 'db/hsi-fsi/hsi-fsi.en.md',
                source_name: 'HSI vs FSI',
                title: 'HSI vs FSI ref card',
                roles: ['citation:system_b', 'reference_basket'],
                pdf_path: 'F:\\kb\\pdfs\\HSI vs FSI.pdf',
                md_path: 'F:\\kb\\md\\hsi-fsi\\hsi-fsi.en.md',
                md_exists: true,
                repairable: true,
                needs_repair: false,
                quality_status: 'good',
                quality_score: 96,
                quality_summary: 'Ready | Q96 | 9 pages | 35 refs | 4 figures | 8 math',
                quality_issues: [],
              },
            ],
            root_causes: [
              { code: 'citation_card_quality', label: 'Citation card or basket copy is weak', severity: 'error', detail: 'Weak summary.', action: 'inspect_cards' },
            ],
            repair_actions: [
              { id: 'open_replay', kind: 'open_replay', label: 'Open replay', severity: 'warning', enabled: true, detail: 'Inspect replay.' },
              { id: 'rerun_case', kind: 'rerun_case', label: 'Rerun case', severity: 'warning', enabled: true, detail: 'Run this case again.' },
              { id: 'open_report', kind: 'open_artifact', target: 'report', label: 'Open report', severity: 'warning', enabled: true, detail: 'Open report.' },
            ],
            rerun_status: {
              available: true,
              run_count: 1,
              last_status: 'passed',
              last_quality_ok: true,
              last_finished_at: 1790000050,
              last_latency_ms: 1400,
              last_passed_at: 1790000050,
              consecutive_failed: 0,
              failure_names: [],
              report_path: 'test_results/research_qa_eval/rerun2/report.md',
              raw_path: 'test_results/research_qa_eval/rerun2/raw_results.jsonl',
            },
            answer_preview: 'preview',
          },
        ],
        queue: { running: false, active_count: 0, active_tasks: [], current: '', done: 0, total: 0 },
        scope: 'all',
        truncated: false,
      }),
    })
  })
  await page.route('**/api/library/quality/artifact/open', async (route) => {
    const payload = route.request().postDataJSON() as { domain?: string, target?: string }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        domain: payload.domain || '',
        target: payload.target || '',
        path: `test_results/research_qa_eval/latest/${payload.target || 'report'}`,
      }),
    })
  })
  await page.route('**/api/library/quality/action-history**', async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true, items: [] }),
      })
      return
    }
    const payload = route.request().postDataJSON() as Record<string, unknown>
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        item: {
          id: `hist-${String(payload.stage_key || 'stage')}`,
          created_at: payload.created_at || 1790000300,
          ...payload,
        },
      }),
    })
  })
  await page.route('**/api/library/quality/figure-assets/scan', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        target_count: 3,
        limit: 1000,
        status: 'error',
        scanned: 3,
        figures: 7,
        docs_with_issues: 2,
        refresh_recommended: 2,
        issue_counts: {
          missing_asset: 1,
          low_resolution: 1,
          duplicate_asset: 2,
          suspicious_crop: 1,
        },
        severity_counts: { error: 1, warning: 4 },
        target_dpi: 320,
        failed: 0,
        errors: [],
        items: [
          {
            ok: true,
            status: 'error',
            source_name: 'Broken conversion',
            pdf_name: brokenName,
            pdf_path: `F:\\kb\\pdfs\\${brokenName}`,
            md_path: 'F:\\kb\\md\\broken\\broken.en.md',
            assets_dir: 'F:\\kb\\md\\broken\\assets',
            source_pdf_path: `F:\\kb\\pdfs\\${brokenName}`,
            source_pdf_available: true,
            target_dpi: 320,
            figures: 2,
            issue_count: 2,
            issue_counts: { missing_asset: 1, low_resolution: 1 },
            severity_counts: { error: 1, warning: 1 },
            refresh_recommended: true,
            issues: [
              {
                code: 'missing_asset',
                severity: 'error',
                asset_name: 'page_1_fig_2.png',
                page: 1,
                figure_number: 2,
                message: 'figure asset is missing or too small to be a valid image',
              },
              {
                code: 'low_resolution',
                severity: 'warning',
                asset_name: 'page_1_fig_1.png',
                page: 1,
                figure_number: 1,
                message: 'figure asset was rendered below the configured figure DPI',
                actual_width: 160,
                actual_height: 160,
                expected_width: 320,
                expected_height: 320,
                estimated_dpi: 160,
              },
            ],
            assets: [],
          },
          {
            ok: true,
            status: 'warning',
            source_name: 'Weak anchors',
            pdf_name: weakName,
            pdf_path: `F:\\kb\\pdfs\\${weakName}`,
            md_path: 'F:\\kb\\md\\weak\\weak.en.md',
            assets_dir: 'F:\\kb\\md\\weak\\assets',
            source_pdf_path: `F:\\kb\\pdfs\\${weakName}`,
            source_pdf_available: true,
            target_dpi: 320,
            figures: 3,
            issue_count: 3,
            issue_counts: { duplicate_asset: 2, suspicious_crop: 1 },
            severity_counts: { warning: 3 },
            refresh_recommended: true,
            issues: [
              {
                code: 'duplicate_asset',
                severity: 'warning',
                asset_name: 'page_2_fig_1.png',
                page: 2,
                figure_number: 3,
                message: 'multiple figure assets have identical image bytes',
                duplicates: ['page_2_fig_1.png', 'page_2_fig_2.png'],
              },
              {
                code: 'suspicious_crop',
                severity: 'warning',
                asset_name: 'page_2_fig_3.png',
                page: 2,
                figure_number: 4,
                message: 'saved crop is much smaller than the detected visual figure box',
              },
            ],
            assets: [],
          },
        ],
      }),
    })
  })
  await page.route('**/api/library/quality/figure-assets/refresh', async (route) => {
    const payload = route.request().postDataJSON() as { sources?: Array<{ source_path?: string, source_name?: string }> }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        requested: payload.sources?.length || 2,
        scanned: 2,
        figures: 5,
        docs_with_issues: 2,
        refresh_recommended: 2,
        issue_counts: { missing_asset: 1, low_resolution: 1, duplicate_asset: 2, suspicious_crop: 1 },
        severity_counts: { error: 1, warning: 4 },
        enqueued: 2,
        skipped_busy: 1,
        failed: 0,
        errors: [],
        items: (payload.sources || []).map((source, idx) => ({
          source_name: source.source_name || '',
          pdf_name: idx === 0 ? brokenName : weakName,
          pdf_path: `F:\\kb\\pdfs\\${idx === 0 ? brokenName : weakName}`,
          md_path: source.source_path || '',
          issue_count: idx === 0 ? 2 : 3,
          issue_codes: idx === 0 ? ['missing_asset', 'low_resolution'] : ['duplicate_asset', 'suspicious_crop'],
          enqueued: true,
          skipped_busy: idx === 1,
          task_id: `fig-refresh-${idx + 1}`,
          error: idx === 1 ? 'already queued or running' : '',
        })),
      }),
    })
  })
  await page.route('**/api/library/quality/repair-runs**', async (route) => {
    const url = route.request().url()
    const runId = (url.split('/quality/repair-runs/')[1]?.split('/advance')[0]?.split(/[?#]/)[0] || 'run-repair-1')
    const method = route.request().method()
    if (method === 'POST' && url.includes('/advance')) {
      const item = {
        run_id: runId,
        status: 'completed',
        phase: 'verification_passed',
        created_at: 1790000400,
        updated_at: 1790000600,
        requested: 1,
        enqueued: 1,
        repaired: 1,
        failed: 0,
        skipped_busy: 0,
        needs_reindex: true,
        reindexed: true,
        target_names: [brokenName],
        target_sources: ['F:\\kb\\md\\broken\\broken.en.md'],
        impact: {
          requested: 1,
          repaired: 1,
          improved: 1,
          enqueued: 1,
          skipped_busy: 0,
          failed: 0,
          needs_reindex: true,
          reindexed: true,
          before_avg_score: 38,
          after_avg_score: 94,
          score_delta: 56,
          fixed_issue_codes: [{ name: 'missing_images', count: 1 }],
          remaining_issue_codes: [],
        },
        verification: {
          type: 'combined_repair_verification',
          status: 'passed',
          quality_ok: true,
          research_qa: {
            type: 'research_qa_rerun',
            case_id: 'scinerf-admm-origin',
            status: 'passed',
            quality_ok: true,
            failure_count: 0,
          },
          reader_locate: {
            type: 'reader_locate_repair',
            status: 'passed',
            quality_ok: true,
            target_count: 1,
            passed: 1,
            failed: 0,
            needs_reader_reopen: 0,
          },
        },
        detail: 'Research QA and Reader locate verification passed for scinerf-admm-origin.',
      }
      latestRepairRun = item
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ok: true,
          advanced: true,
          waiting: false,
          item,
          reindex: {
            ok: true,
            stdout: 'ingest ok',
            stderr: '',
            structured_indices: { version: 2, scanned: 1, rebuilt: 1, skipped: 0, failed: 0, citation_mention_count: 2, errors: [] },
            structured_indices_error: '',
            refsync: { started: true, run_id: 9 },
            refsync_error: '',
          },
          detail: 'index refresh completed',
        }),
      })
      return
    }
    if (method === 'POST') {
      const payload = route.request().postDataJSON() as { status?: string, phase?: string, reindexed?: boolean }
      const item = {
        run_id: runId,
        status: payload.status || 'completed',
        phase: payload.phase || 'reindex_complete',
        created_at: 1790000400,
        updated_at: 1790000500,
        requested: 1,
        enqueued: 0,
        repaired: 1,
        failed: 0,
        skipped_busy: 0,
        needs_reindex: true,
        reindexed: payload.reindexed ?? true,
        target_names: [brokenName],
        target_sources: ['F:\\kb\\md\\broken\\broken.en.md'],
        detail: 'Markdown source repair completed; index refresh is pending.',
      }
      latestRepairRun = item
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ok: true,
          item,
        }),
      })
      return
    }
    const defaultItem = {
      run_id: 'run-repair-1',
      status: 'reindex_pending',
      phase: 'reindex_pending',
      created_at: 1790000400,
      updated_at: 1790000400,
      requested: 1,
      enqueued: 0,
      repaired: 1,
      failed: 0,
      skipped_busy: 0,
      needs_reindex: true,
      reindexed: null,
      target_names: [brokenName],
      target_sources: ['F:\\kb\\md\\broken\\broken.en.md'],
      detail: 'Markdown source repair completed; index refresh is pending.',
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        items: [latestRepairRun || defaultItem],
      }),
    })
  })
  await page.route('**/api/library/convert/status', async (route) => {
    for (const name of repairRequestedNames) repairCompletedNames.add(name)
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'data: {"running":false,"done":true,"total":1,"completed":1,"current":"","active_count":0,"active_tasks":[],"cur_page_done":0,"cur_page_total":0,"cur_page_msg":"","last":""}\n\n',
    })
  })
  await page.route('**/api/library/files**', async (route) => {
    const repairedQuality = {
      status: 'good',
      label: 'Ready',
      score: 94,
      summary: 'Ready | Q94 | 11 pages | 39 refs | 7 figures | 12 math',
      has_review_issue: false,
      issues: [],
      metrics: {
        page_markers: 11,
        references: 39,
        reference_lines: 39,
        figures: 7,
        display_math: 9,
        inline_math: 3,
        missing_images: 0,
        unclosed_display_math: 0,
      },
      conversion_report: {
        available: true,
        stale: false,
        path: 'F:\\kb\\md\\broken\\conversion_quality.json',
        generated_at: '2026-05-29T10:00:00Z',
        auto_repair_enabled: true,
        auto_repair_changed: false,
        auto_repair_unsafe: false,
        auto_repair_applied: [],
        issue_codes_before: ['missing_images', 'missing_references'],
        remaining_issue_codes: [],
        regression_reasons: [],
        repair_plan: { action: 'none', scope: '', speed_mode: '', no_llm: false, replace: true, md_autofix_first: true, reason: '', issue_codes: [], reconvert_issue_codes: [], autofix_issue_codes: [], review_issue_codes: [], issue_actions: [] },
        repair_attempt_count: 1,
        latest_repair_attempt: { event: 'post_convert_quality_gate', status: 'ready', detail: 'Quality gate passed after repair' },
        recommended_action: 'none',
        needs_reconvert: false,
      },
    }
    const brokenQuality = {
      status: 'error',
      label: 'Needs repair',
      score: 38,
      summary: 'Needs repair | Q38 | 0 pages | 0 refs | 1 figures | 1 math',
      has_review_issue: true,
      issues: [
        { code: 'missing_images', label: 'Missing image assets', severity: 'error', count: 1 },
        { code: 'unclosed_display_math', label: 'Unclosed display math', severity: 'error', count: 1 },
        { code: 'missing_references', label: 'Missing reference list', severity: 'warning', count: 0 },
      ],
      metrics: {
        page_markers: 0,
        references: 0,
        reference_lines: 0,
        figures: 1,
        display_math: 1,
        inline_math: 0,
        missing_images: 1,
        unclosed_display_math: 1,
      },
      conversion_report: {
        available: true,
        stale: false,
        path: 'F:\\kb\\md\\broken\\conversion_quality.json',
        generated_at: '2026-05-29T09:00:00Z',
        auto_repair_enabled: true,
        auto_repair_changed: false,
        auto_repair_unsafe: false,
        auto_repair_applied: [],
        issue_codes_before: ['missing_images', 'unclosed_display_math', 'missing_references'],
        remaining_issue_codes: ['missing_images', 'missing_references'],
        regression_reasons: ['missing_references'],
        repair_plan: {
          action: 'reconvert',
          scope: 'full',
          speed_mode: 'ultra_fast',
          no_llm: false,
          replace: true,
          md_autofix_first: true,
          reason: 'Missing images and references block indexing',
          issue_codes: ['missing_images', 'missing_references'],
          reconvert_issue_codes: ['missing_images', 'missing_references'],
          autofix_issue_codes: [],
          review_issue_codes: [],
          issue_actions: [],
        },
        repair_attempt_count: 1,
        latest_repair_attempt: {
          event: 'post_convert_quality_gate',
          status: 'blocked',
          action: 'reconvert',
          reason: 'Missing references block indexing',
          detail: 'Quality gate blocked ingest until reconversion',
        },
        recommended_action: 'reconvert',
        needs_reconvert: true,
      },
    }
    const weakQuality = {
      status: 'warning',
      label: 'Needs review',
      score: 55,
      summary: 'Needs review | Q55 | 4 pages | 8 refs | 2 figures | 3 math',
      has_review_issue: true,
      issues: [
        { code: 'missing_page_markers', label: 'Missing page markers', severity: 'warning', count: 7 },
      ],
      metrics: {
        page_markers: 4,
        references: 8,
        reference_lines: 8,
        figures: 2,
        display_math: 2,
        inline_math: 1,
        missing_images: 0,
      },
      conversion_report: {
        available: true,
        stale: false,
        path: 'F:\\kb\\md\\weak\\conversion_quality.json',
        generated_at: '2026-05-29T09:10:00Z',
        auto_repair_enabled: true,
        auto_repair_changed: false,
        auto_repair_unsafe: false,
        auto_repair_applied: [],
        issue_codes_before: ['missing_page_markers'],
        remaining_issue_codes: ['missing_page_markers'],
        regression_reasons: [],
        repair_plan: {
          action: 'autofix',
          scope: 'markdown',
          speed_mode: '',
          no_llm: true,
          replace: true,
          md_autofix_first: true,
          reason: 'Page anchors need normalization',
          issue_codes: ['missing_page_markers'],
          reconvert_issue_codes: [],
          autofix_issue_codes: ['missing_page_markers'],
          review_issue_codes: [],
          issue_actions: [],
        },
        repair_attempt_count: 1,
        latest_repair_attempt: { event: 'post_convert_quality_gate', status: 'partial', detail: 'Anchors still need repair' },
        recommended_action: 'autofix',
        needs_reconvert: false,
      },
    }
    const items = [
      {
        ...baseItem,
        name: 'NatPhoton-2024-Healthy conversion.pdf',
        path: 'F:\\kb\\pdfs\\NatPhoton-2024-Healthy conversion.pdf',
        md_exists: true,
        md_path: 'F:\\kb\\md\\healthy\\healthy.en.md',
        md_folder: 'F:\\kb\\md\\healthy',
        category: 'converted',
        index_state: 'ready',
        index_status: 'ready',
        index_ready: true,
        index_doc_id: 'healthy-doc',
        index_num_chunks: 8,
        index_chunk_exists: true,
        conversion_quality: {
          status: 'good',
          label: 'Ready',
          score: 96,
          summary: 'Ready | Q96 | 12 pages | 42 refs | 8 figures | 14 math',
          has_review_issue: false,
          issues: [],
          metrics: {
            page_markers: 12,
            references: 42,
            reference_lines: 42,
            figures: 8,
            display_math: 10,
            inline_math: 4,
            missing_images: 0,
          },
          conversion_report: {
            available: true,
            stale: false,
            path: 'F:\\kb\\md\\healthy\\conversion_quality.json',
            generated_at: '2026-05-29T09:20:00Z',
            auto_repair_enabled: true,
            auto_repair_changed: true,
            auto_repair_unsafe: false,
            auto_repair_applied: ['normalize_page_anchors'],
            issue_codes_before: ['weak_page_anchors'],
            remaining_issue_codes: [],
            regression_reasons: [],
            repair_plan: { action: 'none', scope: '', speed_mode: '', no_llm: false, replace: true, md_autofix_first: true, reason: '', issue_codes: [], reconvert_issue_codes: [], autofix_issue_codes: [], review_issue_codes: [], issue_actions: [] },
            repair_attempt_count: 1,
            latest_repair_attempt: { event: 'post_convert_quality_gate', status: 'autofixed', detail: 'Quality gate auto-fixed page anchors' },
            recommended_action: 'none',
            needs_reconvert: false,
          },
        },
      },
      {
        ...baseItem,
        name: brokenName,
        path: `F:\\kb\\pdfs\\${brokenName}`,
        md_exists: true,
        md_path: 'F:\\kb\\md\\broken\\broken.en.md',
        md_folder: 'F:\\kb\\md\\broken',
        category: 'converted',
        paper_category: '3DGS',
        reading_status: 'unread',
        task_state: isRepairing(brokenName) ? 'queued' : 'idle',
        status: isRepairing(brokenName) ? 'queued_reconvert' : 'converted',
        replace_task: isRepairing(brokenName),
        queue_pos: isRepairing(brokenName) ? 1 : 0,
        index_state: repairCompletedNames.has(brokenName) ? 'ready' : 'quality_blocked',
        index_status: repairCompletedNames.has(brokenName) ? 'ready' : 'quality_blocked',
        index_ready: repairCompletedNames.has(brokenName),
        index_doc_id: 'broken-doc',
        index_num_chunks: repairCompletedNames.has(brokenName) ? 7 : 0,
        index_chunk_exists: repairCompletedNames.has(brokenName),
        conversion_quality: repairCompletedNames.has(brokenName) ? repairedQuality : brokenQuality,
      },
      {
        ...baseItem,
        name: weakName,
        path: `F:\\kb\\pdfs\\${weakName}`,
        md_exists: true,
        md_path: 'F:\\kb\\md\\weak\\weak.en.md',
        md_folder: 'F:\\kb\\md\\weak',
        category: 'converted',
        paper_category: 'Adaptive Optics',
        reading_status: 'reading',
        user_tags: ['converter', 'review'],
        task_state: isRepairing(weakName) ? 'queued' : 'idle',
        status: isRepairing(weakName) ? 'queued_reconvert' : 'converted',
        replace_task: isRepairing(weakName),
        queue_pos: isRepairing(weakName) ? 1 : 0,
        index_state: 'ready',
        index_status: 'ready',
        index_ready: true,
        index_doc_id: 'weak-doc',
        index_num_chunks: 5,
        index_chunk_exists: true,
        conversion_quality: repairCompletedNames.has(weakName) ? repairedQuality : weakQuality,
      },
      {
        ...baseItem,
        name: 'Pending paper.pdf',
        path: 'F:\\kb\\pdfs\\Pending paper.pdf',
        md_exists: false,
        md_path: '',
        md_folder: 'F:\\kb\\md\\pending',
        category: 'pending',
        status: 'pending',
        paper_category: '',
        reading_status: '',
        user_tags: [],
        index_state: 'not_converted',
        index_status: '',
        index_ready: false,
        index_doc_id: '',
        index_num_chunks: 0,
        index_chunk_exists: false,
        conversion_quality: null,
      },
    ]
    const qualityReview = items.filter((item) => item.conversion_quality?.has_review_issue).length
    const qualityReady = items.filter((item) => item.conversion_quality?.status === 'good').length
    const reconverting = items.filter((item) => item.task_state !== 'idle').length
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items,
        counts: {
          total_view: items.length,
          total_all: items.length,
          pending: 1,
          converted: 3,
          queued: 0,
          running: 0,
          reconverting,
          quality_review: qualityReview,
          quality_ready: qualityReady,
        },
        truncated: false,
        scope: '200',
        queue: {
          running: false,
          active_count: 0,
          active_tasks: [],
          current: '',
          done: 0,
          total: 0,
        },
      }),
    })
  })
  await page.route('**/api/library/quality/repair', async (route) => {
    const payload = route.request().postDataJSON() as { pdf_names?: string[], sources?: Array<{ source_path?: string, source_name?: string }> }
    const names = Array.isArray(payload.pdf_names) ? payload.pdf_names : []
    const sources = Array.isArray(payload.sources) ? payload.sources : []
    const requested = names.length + sources.length
    const repaired = requested > 0 ? requested : 0
    for (const name of names) repairRequestedNames.add(name)
    const repairRun = {
      run_id: 'run-repair-1',
      status: requested > 0 ? 'queued' : 'completed',
      phase: requested > 0 ? 'source_reconversion_queued' : 'repair_complete',
      created_at: 1790000400,
      updated_at: 1790000400,
      requested,
      enqueued: requested,
      repaired,
      failed: 0,
      skipped_busy: 0,
      needs_reindex: requested > 0,
      reindexed: null,
      target_names: names,
      target_sources: sources.map((source) => String(source.source_path || source.source_name || '')),
      detail: requested > 0 ? 'Source reconversion queued; index refresh should run after conversion.' : 'No source repair required.',
    }
    latestRepairRun = repairRun
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        repair_run_id: 'run-repair-1',
        repair_run: repairRun,
        requested,
        enqueued: requested,
        repaired,
        needs_reindex: requested > 0,
        impact: {
          requested,
          repaired,
          improved: repaired,
          enqueued: requested,
          skipped_busy: 0,
          failed: 0,
          needs_reindex: requested > 0,
          before_avg_score: 38,
          after_avg_score: 94,
          score_delta: 56,
          fixed_issue_codes: [{ name: 'missing_images', count: repaired || 1 }],
          remaining_issue_codes: [],
        },
        skipped_busy: 0,
        failed: 0,
        items: [
          ...names.map((name, idx) => ({
            source_path: '',
            source_name: name,
            pdf_name: name,
            pdf_path: `F:\\kb\\pdfs\\${name}`,
            ok: true,
            enqueued: true,
            repaired: true,
            repair_changed: true,
            repair_applied: ['ensure_page_anchor'],
            repair_before_score: 38,
            repair_after_score: 94,
            quality_before: { status: 'error', score: 38, has_review_issue: true, issues: [{ code: 'missing_images' }] },
            quality_after: { status: 'good', score: 94, has_review_issue: false, issues: [] },
            before_issue_codes: ['missing_images'],
            fixed_issue_codes: ['missing_images'],
            remaining_issue_codes: [],
            skipped_busy: false,
            error: '',
            task_id: `repair-${idx}`,
          })),
          ...sources.map((source, idx) => ({
            source_path: String(source.source_path || ''),
            source_name: String(source.source_name || ''),
            pdf_name: `${source.source_name || `source-${idx}`}.pdf`,
            pdf_path: `F:\\kb\\pdfs\\${source.source_name || `source-${idx}`}.pdf`,
            ok: true,
            enqueued: true,
            repaired: true,
            repair_changed: true,
            repair_applied: ['ensure_page_anchor'],
            repair_before_score: 38,
            repair_after_score: 94,
            quality_before: { status: 'error', score: 38, has_review_issue: true, issues: [{ code: 'missing_images' }] },
            quality_after: { status: 'good', score: 94, has_review_issue: false, issues: [] },
            before_issue_codes: ['missing_images'],
            fixed_issue_codes: ['missing_images'],
            remaining_issue_codes: [],
            skipped_busy: false,
            error: '',
            task_id: `source-repair-${idx}`,
          })),
        ],
      }),
    })
  })
  await page.route('**/api/library/reindex', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        stdout: '',
        stderr: '',
        structured_indices: null,
        structured_indices_error: '',
        refsync: { started: false, reason: 'test' },
        refsync_error: '',
      }),
    })
  })
  await page.route('**/api/library/quality/research-qa/rerun', async (route) => {
    const payload = route.request().postDataJSON() as { case_id?: string }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        case_id: payload.case_id || '',
        status: 'passed',
        quality_ok: true,
        returncode: 0,
        summary: { total: 1, passed: 1, failed: 0 },
        failures: [],
        output_dir: 'test_results/research_qa_eval/rerun',
        report_path: 'test_results/research_qa_eval/rerun/report.md',
        raw_path: 'test_results/research_qa_eval/rerun/raw_results.jsonl',
        stdout_tail: '[OK] research QA eval finished',
        stderr_tail: '',
        started_at: 1,
        finished_at: 2,
        latency_ms: 1200,
      }),
    })
  })
  await page.route('**/api/references/shelf/metadata/repair', async (route) => {
    const payload = route.request().postDataJSON() as { items?: Array<Record<string, unknown>> }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        requested: payload.items?.length || 0,
        ready: payload.items?.length || 0,
        export_ready: payload.items?.length || 0,
        partial: 0,
        retryable: 0,
        failed: 0,
        unresolved: 0,
        changed: 1,
        acceptance: {
          contract_version: 1,
          requested: payload.items?.length || 0,
          quality_ok: true,
          metadata_ready_before: 0,
          metadata_ready_after: payload.items?.length || 0,
          metadata_ready_delta: payload.items?.length || 0,
          export_ready_before: 0,
          export_ready_after: payload.items?.length || 0,
          export_ready_delta: payload.items?.length || 0,
          summary_export_ready_after: 0,
          retryable: 0,
          failed: 0,
          unresolved_after: 0,
          remaining_fields: [],
          remaining_issue_codes: [],
        },
        verification: {
          type: 'shelf_metadata_repair',
          status: 'passed',
          quality_ok: true,
          target_count: payload.items?.length || 0,
          metadata_ready_after: payload.items?.length || 0,
          export_ready_after: payload.items?.length || 0,
          changed: 1,
          retryable: 0,
          failed: 0,
          unresolved_after: 0,
        },
        items: (payload.items || []).map((item, idx) => ({
          key: item.key || `meta-${idx}`,
          ok: true,
          changed: idx === 0,
          changed_fields: idx === 0 ? ['doi', 'authors', 'venue'] : [],
          repair_status: idx === 0 ? 'repaired' : 'ready',
          retryable: false,
          before: { contract_version: 1, ok: false, status: 'warning', score: 76, missing_fields: ['doi'], issues: [], repairable: true, retryable: true },
          after: { contract_version: 1, ok: true, status: 'ready', score: 100, missing_fields: [], issues: [], repairable: true, retryable: false },
          meta: { ...item, doi: '10.1561/2200000016', authors: 'Boyd S', venue: 'Foundations and Trends in Machine Learning' },
        })),
      }),
    })
  })
})

test('library page surfaces conversion quality and filters review items', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/library')

  await expect(page.getByTestId('library-file-row')).toHaveCount(4)
  await expect(page.getByTestId('library-quality-report')).toBeVisible()
  await expect(page.getByTestId('library-quality-center-summary')).toContainText('Needs repair')
  await expect(page.getByTestId('library-quality-center-summary')).toContainText('User-facing risks')
  await expect(page.getByTestId('library-quality-center-details')).toHaveCount(0)
  await expect(page.getByTestId('library-quality-domains')).toHaveCount(0)
  await page.getByTestId('library-quality-center-toggle').click()
  await expect(page.getByTestId('library-quality-center-details')).toBeVisible()
  await expect(page.getByTestId('library-quality-report-review')).toContainText('2')
  await expect(page.getByTestId('library-quality-report-good')).toContainText('1')
  await expect(page.getByTestId('library-quality-report-avg')).toContainText('Q70')
  await expect(page.getByTestId('library-quality-report-source-ready')).toContainText('1')
  await expect(page.getByTestId('library-quality-report-autofixed')).toContainText('1')
  await expect(page.getByTestId('library-quality-report-blocked')).toContainText('1')
  await expect(page.getByTestId('library-quality-domains').locator('[data-quality-domain="research_qa"]')).toContainText('refs_include_required_docs x1')
  await expect(page.getByTestId('library-quality-domains').locator('[data-quality-domain="citation_cards"]')).toContainText('citation_card_quality x1')
  await expect(page.getByTestId('library-quality-domains').locator('[data-quality-domain="citation_cards"]')).toContainText('shelf export 1/2')
  await expect(page.getByTestId('library-quality-domains').locator('[data-quality-domain="reader_locate"]')).toContainText('target anchor drift x1')
  await expect(page.getByTestId('library-metadata-backfill-health')).toContainText('Literature metadata')
  await expect(page.getByTestId('library-metadata-backfill-health')).toContainText('24/42')
  await expect(page.getByTestId('library-metadata-backfill-health')).toContainText('authors x8')
  await expect(page.getByTestId('library-metadata-backfill-health')).toContainText('preheated')
  await expect(page.getByTestId('library-figure-assets-health')).toContainText('Figure assets')
  await expect(page.getByTestId('library-figure-assets-health')).toContainText('Not scanned')
  const figureRefreshButton = page.getByTestId('library-figure-assets-health').getByRole('button', { name: 'Refresh flagged' })
  await expect(figureRefreshButton).toBeDisabled()
  const figureScanRequest = page.waitForRequest('**/api/library/quality/figure-assets/scan')
  await page.getByTestId('library-figure-assets-health').getByRole('button', { name: 'Scan' }).click()
  await expect.poll(async () => {
    const payload = (await figureScanRequest).postDataJSON() as { include_all?: boolean }
    return payload.include_all
  }).toBe(false)
  await expect(page.getByTestId('library-figure-assets-health')).toContainText('2/3 refresh')
  await expect(page.getByTestId('library-figure-assets-issues')).toContainText('missing_asset x1')
  await expect(page.getByTestId('library-figure-assets-issues')).toContainText('duplicate_asset x2')
  await expect(page.getByTestId('library-figure-assets-list')).toContainText('Broken conversion')
  await expect(page.getByTestId('library-figure-assets-list')).toContainText('figure asset is missing')
  await expect(figureRefreshButton).toBeEnabled()
  const figureRefreshRequest = page.waitForRequest('**/api/library/quality/figure-assets/refresh')
  await figureRefreshButton.click()
  await expect.poll(async () => {
    const payload = (await figureRefreshRequest).postDataJSON() as { limit?: number; sources?: Array<{ source_path?: string }> }
    return `${payload.sources?.length || 0}:${payload.limit || 0}`
  }).toBe('2:2')
  await expect(page.getByTestId('library-figure-assets-refresh-result')).toContainText('queued')
  await expect(page.getByTestId('library-figure-assets-refresh-result')).toContainText('2')
  await expect(page.getByTestId('library-quality-feature-health')).toContainText('Feature health')
  await expect(page.getByTestId('library-quality-feature-health')).toContainText('Paper Guide')
  await expect(page.getByTestId('library-quality-feature-health')).toContainText('Literature basket')
  await expect(page.getByTestId('library-quality-feature-health')).toContainText('Reader locate')
  await expect(page.getByTestId('library-quality-full-chain')).toContainText('Full-chain health')
  await expect(page.getByTestId('library-quality-full-chain')).toContainText('Retrieval coverage')
  await expect(page.getByTestId('library-quality-full-chain')).toContainText('Literature basket')
  await expect(page.getByTestId('library-quality-full-chain')).toContainText('citation_card_quality')
  await expect(page.getByTestId('library-quality-full-chain-history')).toContainText('Recent actions')
  await expect(page.getByTestId('library-quality-full-chain-history')).toContainText('Rebuilt retrieval index')
  await expect(page.getByTestId('library-quality-full-chain-history')).toContainText('Improved: error -> good')
  const persistedHistoryRow = page.getByTestId('library-quality-full-chain-history-row').filter({ hasText: 'Rebuilt retrieval index' })
  await expect(persistedHistoryRow.getByTestId('library-quality-full-chain-history-open')).toContainText('Open replay')
  const retrievalStage = page.getByTestId('library-quality-full-chain-stage').filter({ hasText: 'Retrieval coverage' })
  const stageReindexRequest = page.waitForRequest('**/api/library/reindex')
  await retrievalStage.getByTestId('library-quality-full-chain-stage-action').click()
  await stageReindexRequest
  await expect(retrievalStage.getByTestId('library-quality-full-chain-stage-result')).toContainText('Reindex verified')
  await expect(retrievalStage.getByTestId('library-quality-full-chain-stage-result')).toContainText('QA rerun passed')
  const shelfStage = page.getByTestId('library-quality-full-chain-stage').filter({ hasText: 'Literature basket' })
  const stageMetadataRequest = page.waitForRequest('**/api/references/shelf/metadata/repair')
  await shelfStage.getByTestId('library-quality-full-chain-stage-action').click()
  await expect.poll(async () => {
    const payload = (await stageMetadataRequest).postDataJSON() as { items?: Array<Record<string, unknown>> }
    return payload.items?.length || 0
  }).toBeGreaterThan(0)
  await expect(shelfStage.getByTestId('library-quality-full-chain-stage-result')).toContainText('Metadata repair verified')
  await expect(shelfStage.getByTestId('library-quality-full-chain-stage-result')).toContainText('QA rerun passed')
  await expect(page.getByTestId('library-quality-priority-actions').locator('[data-quality-action-domain="research_qa"]')).toContainText('1')
  await expect(page.getByTestId('library-quality-priority-actions').locator('[data-quality-action-domain="citation_cards"]')).toContainText('2')
  await expect(page.getByTestId('library-quality-priority-actions').locator('[data-quality-action-domain="reader_locate"]')).toContainText('1')
  await expect(page.getByTestId('library-quality-rerun-summary')).toContainText('Runs')
  await expect(page.getByTestId('library-quality-rerun-summary')).toContainText('refs_include_required_docs x2')
  await expect(page.getByTestId('library-quality-failure-cases')).toContainText('scinerf-admm-origin')
  await expect(page.getByTestId('library-quality-failure-cases')).toContainText('scinerf')
  await page.getByTestId('library-quality-failure-filter').filter({ hasText: 'refs_include_required_docs' }).click()
  await expect(page.getByTestId('library-quality-failure-case')).toHaveCount(1)
  await expect(page.getByTestId('library-quality-failure-case')).toContainText('scinerf-admm-origin')
  await page.getByTestId('library-quality-failure-filter-all').click()
  await expect(page.getByTestId('library-quality-failure-case')).toHaveCount(2)
  const failedQaCase = page.getByTestId('library-quality-failure-case').filter({ hasText: 'scinerf-admm-origin' })
  await expect(failedQaCase.getByTestId('library-quality-root-causes')).toContainText('Retrieval missed required documents')
  await expect(failedQaCase.getByTestId('library-quality-root-causes')).toContainText('Citation card or basket copy is weak')
  await expect(failedQaCase.getByTestId('library-quality-shelf-metadata-fields')).toContainText('doi x1')
  await expect(failedQaCase.getByTestId('library-quality-source-diagnostics')).toContainText('Q74')
  await expect(failedQaCase.getByTestId('library-quality-rerun-result')).toContainText('Rerun passed')
  await expect(failedQaCase.getByRole('button', { name: 'Fix from source' })).toBeVisible()
  await expect(failedQaCase.getByRole('button', { name: 'Rerun case' })).toBeVisible()
  await expect(failedQaCase.getByRole('button', { name: 'Rebuild index' })).toBeVisible()
  const planRepairRequest = page.waitForRequest('**/api/library/quality/repair')
  const planMetadataRequest = page.waitForRequest('**/api/references/shelf/metadata/repair')
  const planReindexRequest = page.waitForRequest('**/api/library/reindex')
  const planRerunRequest = page.waitForRequest('**/api/library/quality/research-qa/rerun')
  await failedQaCase.getByRole('button', { name: 'Fix from source' }).click()
  const planRepairPayload = planRepairRequest.then((request) => request.postDataJSON() as { sources?: Array<{ source_path?: string }> })
  await expect.poll(async () => (await planRepairPayload).sources?.[0]?.source_path || '').toContain('scinerf')
  await expect.poll(async () => {
    const payload = (await planMetadataRequest).postDataJSON() as { items?: Array<Record<string, unknown>> }
    return payload.items?.length || 0
  }).toBeGreaterThan(0)
  await expect.poll(async () => {
    const payload = (await planMetadataRequest).postDataJSON() as { items?: Array<Record<string, unknown>> }
    const first = payload.items?.[0] || {}
    return Array.isArray(first.metadata_missing_fields) ? first.metadata_missing_fields.join(',') : ''
  }).toContain('doi')
  await planReindexRequest
  await expect.poll(async () => {
    const payload = (await planRerunRequest).postDataJSON() as { case_id?: string }
    return payload.case_id || ''
  }).toBe('scinerf-admm-origin')
  await expect(failedQaCase.getByTestId('library-quality-rerun-result')).toContainText('Rerun passed')
  const rerunRequest = page.waitForRequest('**/api/library/quality/research-qa/rerun')
  await failedQaCase.getByRole('button', { name: 'Rerun case' }).click()
  const rerunPayload = rerunRequest.then((request) => request.postDataJSON() as { case_id?: string })
  await expect.poll(async () => (await rerunPayload).case_id || '').toBe('scinerf-admm-origin')
  await expect(failedQaCase.getByTestId('library-quality-rerun-result')).toContainText('Rerun passed')
  const sourceRepairRequest = page.waitForRequest('**/api/library/quality/repair')
  await failedQaCase.getByRole('button', { name: 'Repair sources' }).click()
  const sourceRepairPayload = sourceRepairRequest.then((request) => request.postDataJSON() as { sources?: Array<{ source_path?: string }> })
  await expect.poll(async () => (await sourceRepairPayload).sources?.[0]?.source_path || '').toContain('scinerf')
  await expect(page).toHaveURL(/\/library/)
  const rawOpenRequest = page.waitForRequest('**/api/library/quality/artifact/open')
  await failedQaCase.getByRole('button', { name: 'Open raw QA' }).click()
  await expect.poll(async () => {
    const payload = (await rawOpenRequest).postDataJSON() as { domain?: string, target?: string }
    return `${payload.domain}:${payload.target}`
  }).toBe('research_qa:raw')
  const reportOpenRequest = page.waitForRequest('**/api/library/quality/artifact/open')
  await page.getByTestId('library-quality-domains').locator('[data-quality-domain="research_qa"] button').first().click()
  await expect.poll(async () => {
    const payload = (await reportOpenRequest).postDataJSON() as { domain?: string, target?: string }
    return `${payload.domain}:${payload.target}`
  }).toBe('research_qa:report')
  const priorityOpenRequest = page.waitForRequest('**/api/library/quality/artifact/open')
  await page.getByTestId('library-quality-priority-actions').locator('[data-quality-action-domain="citation_cards"]').click()
  await expect.poll(async () => {
    const payload = (await priorityOpenRequest).postDataJSON() as { domain?: string, target?: string }
    return `${payload.domain}:${payload.target}`
  }).toBe('citation_cards:report')
  await expect(page.getByTestId('library-quality-report-recommended')).toContainText('Broken conversion')
  await expect(page.getByTestId('library-quality-report-repair-recommended')).toContainText('2')
  await page.getByTestId('library-quality-report-focus-review').click()
  await expect(page.getByTestId('library-file-row')).toHaveCount(2)
  await page.getByTestId('library-quality-issues-filter').click()
  await expect(page.getByTestId('library-file-row')).toHaveCount(4)

  const healthy = page.getByTestId('library-file-row').filter({ hasText: 'Healthy conversion' })
  await expect(healthy.getByTestId('library-file-quality-chip')).toHaveAttribute('data-quality-status', 'good')
  await expect(healthy.getByTestId('library-file-source-readiness')).toHaveAttribute('data-source-readiness', 'autofixed')
  await expect(healthy.getByTestId('library-file-source-readiness')).toContainText('已自动修复')
  await expect(healthy.getByTestId('library-file-quality-line')).toContainText('refs 42')

  const broken = page.getByTestId('library-file-row').filter({ hasText: 'Broken conversion' })
  await expect(broken.getByTestId('library-file-quality-chip')).toHaveAttribute('data-quality-status', 'error')
  await expect(broken.getByTestId('library-file-quality-chip')).toContainText('Repair Q38')
  await expect(broken.getByTestId('library-file-source-readiness')).toHaveAttribute('data-source-readiness', 'blocked')
  await expect(broken.getByTestId('library-file-source-readiness')).toContainText('未入库')
  await expect(broken.getByTestId('library-file-quality-line')).toContainText('Missing image assets')
  await expect(broken.getByTestId('library-file-quality-line')).toContainText('refs 0')
  await expect(broken.getByTestId('library-quality-repair')).toContainText('重转入库')

  await page.getByTestId('library-quality-issues-filter').click()
  await expect(page.getByTestId('library-file-row')).toHaveCount(2)
  await expect(page.getByTestId('library-file-row').filter({ hasText: 'Broken conversion' })).toBeVisible()
  await expect(page.getByTestId('library-file-row').filter({ hasText: 'Weak anchors' })).toBeVisible()
  await page.getByTestId('library-quality-issues-filter').click()
  await expect(page.getByTestId('library-file-row')).toHaveCount(4)

  await expect(broken.getByTestId('library-quality-repair')).toBeVisible()
  const repairRunAutoAdvanceRequest = page.waitForRequest('**/api/library/quality/repair-runs/**/advance')
  const repairRequest = page.waitForRequest('**/api/library/quality/repair')
  await broken.getByTestId('library-quality-repair').click()
  const repairPayload = repairRequest.then((request) => request.postDataJSON() as { pdf_names?: string[] })
  await expect.poll(async () => (await repairPayload).pdf_names?.join('\n') || '').toContain('Optica-2024-Broken conversion.pdf')
  await repairRunAutoAdvanceRequest
  await expect(broken.getByTestId('library-file-quality-chip')).toHaveAttribute('data-quality-status', 'good')
  await expect(broken.getByTestId('library-file-quality-chip')).toContainText('Q94')
  await expect(broken.getByTestId('library-file-source-readiness')).toHaveAttribute('data-source-readiness', 'ready')
  await expect(broken.getByTestId('library-file-quality-line')).toContainText('refs 39')
  await expect(broken.getByTestId('library-quality-repair-result')).toContainText('Q38')
  await expect(broken.getByTestId('library-quality-repair-result')).toContainText('Q94')
  await expect(broken.getByTestId('library-quality-repair-result')).toContainText('Missing image assets')
  await expect(page.getByTestId('library-quality-repair-impact')).toContainText('Repair impact')
  await expect(page.getByTestId('library-quality-repair-run')).toContainText('Run tracked')
  await expect(page.getByTestId('library-quality-repair-run')).toContainText('run-repa')
  await expect(page.getByTestId('library-quality-repair-run')).toContainText('verified')
  await expect(page.getByTestId('library-quality-repair-run')).toContainText('QA rerun passed')
  await expect(page.getByTestId('library-quality-repair-run')).toContainText('Reader locate verified')
  await expect(page.getByTestId('library-quality-repair-impact')).toContainText('Q38')
  await expect(page.getByTestId('library-quality-repair-impact')).toContainText('missing_images')
  await expect(page.getByTestId('library-quality-report-review')).toContainText('1')
  await expect(page.getByTestId('library-quality-report-good')).toContainText('2')
  await expect(page.getByTestId('library-quality-report-repair-recommended')).toContainText('1')
  await expect(page.getByTestId('library-quality-history')).toBeVisible()
  await expect(page.getByTestId('library-quality-history-count')).toContainText('1')
  await expect(page.getByTestId('library-quality-history-row')).toContainText('Q38')
  await expect(page.getByTestId('library-quality-history-row')).toContainText('Q94')
  await expect(page.getByTestId('library-quality-history-row')).toContainText('Missing image assets')
  await expect(page.getByTestId('library-quality-history-focus-remaining')).toBeDisabled()
  await expect(page.getByTestId('library-quality-history-repair-recommended')).toContainText('1')
  await expect(page.getByTestId('library-quality-history-repair-recommended')).toBeEnabled()

  await page.getByTestId('library-quality-history-paper').click()
  await expect(page.getByTestId('library-file-row')).toHaveCount(1)
  await expect(page.getByTestId('library-file-row')).toContainText('Broken conversion')
  await expect(page.getByTestId('library-quality-history-active-filter')).toContainText('历史聚焦 1 篇')
  await page.getByTestId('library-quality-history-active-filter').click()
  await expect(page.getByTestId('library-file-row')).toHaveCount(4)

  await page.getByTestId('library-quality-issues-filter').click()
  await expect(page.getByTestId('library-file-row')).toHaveCount(1)
  await expect(page.getByTestId('library-file-row')).toContainText('Weak anchors')
  const recommendedRepairRequest = page.waitForRequest('**/api/library/quality/repair')
  await page.getByTestId('library-quality-history-repair-recommended').click()
  const recommendedRepairPayload = recommendedRepairRequest.then((request) => request.postDataJSON() as { pdf_names?: string[] })
  await expect.poll(async () => (await recommendedRepairPayload).pdf_names?.join('\n') || '').toContain('Applied Optics-2023-Weak anchors.pdf')
  await expect(page.getByTestId('library-file-row')).toHaveCount(0)
  await expect(page.getByTestId('library-quality-history-count')).toContainText('2')
  await expect(page.getByTestId('library-quality-report-review')).toContainText('0')
  await expect(page.getByTestId('library-quality-report-good')).toContainText('3')
  await page.getByTestId('library-quality-issues-filter').click()
  await expect(page.getByTestId('library-file-row')).toHaveCount(4)
  const weak = page.getByTestId('library-file-row').filter({ hasText: 'Weak anchors' })
  await expect(weak.getByTestId('library-file-quality-chip')).toHaveAttribute('data-quality-status', 'good')
  await expectNoHorizontalOverflow(page)

  await page.getByTestId('library-quality-full-chain-history-row')
    .filter({ hasText: 'Rebuilt retrieval index' })
    .getByTestId('library-quality-full-chain-history-open')
    .click()
  await expect(page).toHaveURL(/__research_qa_replay__\?case=scinerf-admm-origin&source=quality-history/)
  await expect(page.getByTestId('research-qa-diagnostic')).toContainText('citation_card_quality')
  await expect(page.getByTestId('research-qa-diagnostic-docs')).toContainText('scinerf')
  await expect(page.getByTestId('research-qa-diagnostic-missing-docs')).toContainText('scigs')
  await expect(page.getByTestId('research-qa-diagnostic-citations')).toContainText('SCINeRF citation')
  await expect(page.getByTestId('research-qa-diagnostic-refs')).toContainText('SCINeRF ref card')
})
