# Phase 0 (unblock) — status

Tracks the PIPELINE_REWORK.md "Phase 0" correctness fixes for the repos we own
(workflow-provider + memory-unit).

## Done

- **memory-unit `/hydrate` + `/refresh` crash** (in the memory-unit repo) — signature
  mismatch fixed; hydrate works without an OpenAI key; `/refresh` reuses the stored root
  folder and reports `refreshed`. Tests: `memory-unit/tests/integration/test_api_hydrate.py`.
- **Parameters (slots) are no longer dropped** — `ContextItem` moved to `utils/task.py` and
  added as `Task.context_items`. `identify_task` now stores them on the Task, and `/edit_task`
  returns the real items instead of always `[]` (`app.py`). Tests in `tests/task_unit_test.py`
  (`test_edit_task_endpoint_preserves_context_items`).
- **Generated workflows are persisted** — `BuilderAgent` takes the vector store and
  `create_workflow_initial` writes the generated workflow to the `generated_workflows`
  collection (best-effort; a storage failure won't break creation). Tests:
  `tests/test_builder_agent.py`.

## Deliberately deferred (not a cosmetic fix)

- **Chroma durability.** `utils/config.py` defaults to `./chroma_db` (persists in local dev),
  but the **Dockerfile forces `CHROMA_PERSIST_DIR=/tmp/chroma_db`**. Changing that path is
  cosmetic: on Cloud Run the container filesystem is ephemeral regardless of path, so a real
  fix requires a **mounted volume (e.g. GCS FUSE) or a managed/durable vector DB** — that's the
  Phase 1 "shared durable store" work, not a Phase 0 one-liner. Left the Dockerfile as-is to
  avoid a false sense of durability; address it when standing up the shared store.

## Out of scope here (later phases)

- Search-before-create is still only in `/enrich` and not enforced (`/create_workflow` is
  directly callable); generated workflows now persist but **dedup / a promotion gate** are
  Phase 3 concerns. Today's persistence can store near-duplicate or rejected variants.
- Typed slot model, template/instance hierarchy, memory-unit `resolve()`, unified trace loop —
  all later phases.
