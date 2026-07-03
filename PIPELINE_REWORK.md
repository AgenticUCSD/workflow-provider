# Context-Aware Pipeline — Rework Plan (workflow-provider + memory-unit)

**Audience:** the engineers on `workflow-provider` and `memory-unit`.
**Goal:** turn four stateless services that pass JSON into **one context-aware pipeline backed by shared, durable state** — so everything from task identification through workflow execution reads and writes the same tasks, workflows (templates + enriched instances), skills, and context, and every run is one trace that refines the system.

**Scope:** `workflow-provider` and `memory-unit`, plus the **shared store** they shoulder with the executor. This is the provider/memory counterpart to the executor's [`workflow_executor/REWORK.md`](../workflow_executor/REWORK.md) and reuses its core ideas — the **Artifact** envelope, the **promotion gate**, versioning/provenance, and the **two loops** (execution vs. improvement). Where this doc says "Artifact," it means the same thing the executor doc does. The win is that all three services share one model.

---

## Status & corrections (revised 2026-07-01)

This is a **revision** of the original plan, re-grounded against the code as it actually stands
today. The **design is unchanged** — same six goals, same phases, same guarantees, same target
architecture. What changed is accuracy: several bugs the original plan set out to fix are **already
fixed and merged**, and a few forward assumptions **collide with reality**. The corrections:

- **Phase 0 is effectively DONE** (merged this session — provider PR #2 `ed0a3cc`, memory-unit
  PR #1 `4ca5610`). Params are no longer dropped, `/hydrate` no longer 500s, generated workflows
  persist, search-before-create is enforced, dedup exists, memory-unit is tenant-guarded and durable.
  The only Phase-0 leftover is the provider's Chroma persistence path, which folds into Phase 1.
- **The typed `Step` model cannot be sent to the executor as-is** (bug). The executor's
  `/workflow/execute` accepts `workflow_steps: List[str]` **and now rejects unknown fields**
  (`extra="forbid"`). Resolution: typed Steps stay **internal to the provider** and are **serialized
  back to `List[str]` at the executor boundary** — an adapter, no executor change, live contract safe.
- **The Artifact envelope does not exist in any repo** (it is designed-only in both rework docs).
  It is net-new work with **unresolved ownership**, not a foundation to build on. Flagged in Phase 1.
- **Phase 5 (and the promotion gate) is BLOCKED** on the executor's eval harness, which **does not
  exist** (the executor has no test suite / scorers / golden set). Marked cross-team-blocked.
- **`thread_id` is overloaded** (session key vs. trace key vs. the executor's `task_id`). We define a
  mapping rather than assume one shared primary key.
- **Recommended near-term path:** because Phase 0 is done and Phase 1 is cross-team-blocked, do the
  **in-our-repos, unblocked slice first — Phase 2 (typed slots) + Phase 4 (memory-unit `resolve()`,
  wired into the provider).** That is the plan's headline context-aware win and needs neither the
  shared store nor the eval harness. Phases 1/3/5 stay in the plan, tagged with their blockers.

Status legend below: ✅ done · ⚠️ partially done / open · 🚫 cross-team-blocked.

---

## TL;DR

Today the pipeline still can't be fully context-aware because **the shared context it should be aware of does not exist yet.** Each service re-derives most state from its request body; the only cross-run persistence is per-service Chroma collections; the one service meant to supply context (`memory-unit`) is durable and tenant-guarded now but **still unwired** and returns prose, not parameters. The plan, in order:

0. ✅ **Unblock** — fix the correctness bugs that made the goals impossible (param fields that didn't exist, a memory-unit that 500'd, generated workflows that never persisted). **Done.**
1. ⚠️ **One shared datastore** — Postgres + a vector index holding Tasks, Workflow Templates, Enriched Instances, Skills/Capabilities, Context, and per-session state. Replaces the ephemeral per-service Chromas.
2. ⚠️ **A typed slot model** — make task **parameters** first-class (we already carry `context_items`; this types them into a signature with source + confidence). Powers population, search, enrichment, conversation-editing, and eval.
3. ⚠️ **Template → enriched-instance hierarchy** — versioned, parameterized templates; search-before-create (✅ already enforced); generated workflows persisted (✅ already) so the corpus learns.
4. ⚠️ **memory-unit as the context/parameter resolver** — structured slot resolution + a write-back loop, wired in (durable + isolated ✅ already).
5. 🚫 **One trace, one refinement loop** — a single `thread_id` correlates the whole pipeline in DeepEval; traces refine templates and context through the executor's eval gate (**blocked: that gate doesn't exist yet**).

---

## Where we are now (grounded)

**workflow-provider** (multi-agent FastAPI + ChromaDB):
- ✅ **Parameters are carried, not dropped.** `Task.context_items: List[ContextItem]` exists
  (`utils/task.py:42-46,59`); `identify_task` stores them; `/edit_task` round-trips
  `edited_task.context_items` (`app.py:135-146`). *(Original claim "parameters are dropped" is fixed.)*
- ✅ **Search-before-create is enforced on `/create_workflow`** (not just `/enrich`); a fresh create
  reuses a strict match, regeneration still generates (`app.py:98-119`). ✅ **Generated workflows
  persist** to the `generated_workflows` collection (`agents/builder_agent.py:47,50-58`). ✅
  **Content-hash dedup** skips exact-duplicate inserts (`utils/chroma.py:31-51`).
- ⚠️ **Persistence is per-service Chroma**, two collections `manual_workflows` / `generated_workflows`
  on `CHROMA_PERSIST_DIR` (default `./chroma_db`, Docker `/tmp/chroma_db`) — durable to disk but
  **ephemeral on Cloud Run and per-instance**. No Task store, no skill store, no shared state.
- ⚠️ **Flat workflows.** `Workflow = {workflow_id, name, description, steps: List[str]}`
  (`utils/task.py:31-40`) — string steps, no template/instance, no version/lineage. `manual` vs
  `generated` is provenance only. `Objective.inputs` is still a free `Dict[str, Any]`.
- ⚠️ DeepEval traces every LLM agent call (via `thread_id`), but **not retrieval**; the
  `analyzer_agent` writes `knowledge/*.txt` that **nothing reads** — a dead-end, not a loop.

**memory-unit** (agentic-RAG over Drive):
- ✅ **`/hydrate` and `/refresh` work.** The `api.py`↔`core.py` signature mismatch is fixed
  (`MemoryUnit(persist_dir, model_name)` / `hydrate_from_drive(root_folder_id, auth_token)`).
- ✅ **Tenant-guarded + durable.** `require_owner` requires `X-User-Id` and binds the hydrating user
  (`api.py:144-155`); CORS is an `ALLOWED_ORIGINS` allow-list, not `["*"]` (`api.py:41-56`); Chroma is
  a `PersistentClient` on disk (`storage/vector_store.py:28`). BM25 + preferences are still rebuilt
  in-RAM per hydrate (acceptable for a search index).
- ⚠️ **Still an island** — durable and safe, but **no other service calls it** in the live pipeline.
- ⚠️ **Returns prose, not parameters.** `query()` returns a structured `ContextQueryResult`
  (answer + `context_for_*` + preference/pattern lists), but there is **no `resolve()`** producing
  typed `field → value` slots. It cannot yet populate a parameter.
- ⚠️ **No write-back** (machine context is read-only; `drive/client.py` has read methods only —
  `upload_url` is defined but unused), **no tracing** (`X-Thread-Id` is accepted but dropped), **no
  hierarchical scope** (single-tenant, one hydrated user at a time; no org/role/thread partitioning).

**workflow_executor** (owned by another engineer — contract only, we do not modify it):
- Accepts `WorkflowExecuteRequest` with `workflow_steps: **List[str]**` and now
  `model_config = ConfigDict(extra="forbid")` — it **422s on unknown top-level fields**
  (`server.py:455-476`). It already accepts `context_items` and `metadata` (sender/recipient/cc).
- **No Artifact envelope, no typed Step, no template/instance hierarchy, no eval harness** in code —
  those live only in `workflow_executor/REWORK.md`. Its Postgres is **LangGraph checkpointing +
  status/interrupt persistence**, not a shared artifact store. Auth is Bearer + `X-User-Id`.

---

## Target architecture

```
   ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐
   │ Task         │   │ Workflow         │   │ Workflow          │
   │ Identifier   │   │ Provider         │   │ Executor          │
   │ (id + slots) │   │ (search/build)   │   │ (run + HITL)      │
   └──────┬───────┘   └────────┬─────────┘   └─────────┬─────────┘
          │  read/write         │  read/write           │  read/write
          ▼                     ▼                       ▼
   ┌────────────────────────────────────────────────────────────────┐
   │              SHARED STORE  (Postgres + vector index)           │
   │  Sessions · Tasks(+slots) · Templates · Instances · Skills/    │
   │  Capabilities · Context blocks         — all are Artifacts     │
   └──────────────────────────┬─────────────────────────────────────┘
                              │ resolve(slots,user) / write-back
                     ┌────────┴─────────┐
                     │   memory-unit    │  context + parameter resolver
                     └──────────────────┘
   one thread_id  ─────────────────────────────────►  one DeepEval trace
```

The store is the center of gravity. `thread_id` is a **real session key**, not a trace tag — every
stage reads the same task/slots/template/conversation and appends to it, so nothing is re-derived
from request bodies. **Identifier note:** the executor keys its runs on its own `task_id`
(autogenerated when omitted). We do **not** assume `thread_id` and `task_id` are the same PK — the
provider records a `thread_id → executor task_id` mapping on the session when it calls
`/workflow/execute`, and correlates traces through that mapping.

---

## Guarantees we want (promises)

- **P-STORE1 — Shared, durable state.** Tasks, templates, instances, skills, context, and session state live in one store that survives restarts and is shared across services and instances. No pipeline-critical state lives in per-instance memory or `/tmp`.
- **P-SLOT1 — Parameters are first-class.** Every task carries a typed **signature** (slots); slot values carry `source` + `confidence`; the conversation can edit task name, description, **and** slots. *(Partly seeded: `context_items` already carries `field/status/value`.)*
- **P-CTX-POP1 — Context-driven population.** Slots are populated from user context via `memory-unit` before falling to HITL; only genuinely missing/low-confidence slots are asked.
- **P-SBC1 — Search before create (enforced).** A new template is only created after the repository is searched and no match clears the threshold; every generated template is persisted so search improves. *(✅ enforced today for flat workflows; extends to templates in Phase 3.)*
- **P-TMPL1 — Template lineage.** Workflows are versioned **templates** (generic, parameterized) vs **enriched instances** (template + bound slots + specialization); instances record `template_id@version` and are attributable.
- **P-TRACE1 — One pipeline, one trace.** Identify → populate → search → enrich → execute → resume is a single DeepEval trace correlated by `thread_id` (and the `task_id` mapping), recording the artifact versions used (P-EVO1 from the executor doc).
- **P-LEARN1 — Closed refinement loop.** Traces feed evaluation; templates and context blocks are refined and promoted through the same gate as skills/capabilities — no dead-end sinks.

These compose with the executor's `P-SEC*`, `P-REL*`, `P-OBS1`, `P-MCP1`, `P-CTX1`, `P-EVO1`. Error shape stays RFC 9457 `application/problem+json`.

---

## Cross-cutting model (shared with the executor)

Everything that evolves is an **Artifact** with one envelope — defined in the executor REWORK, reused here so provider/memory/executor don't fork the model:

```
Artifact = { id, kind: task|template|instance|skill|capability|context,
             version, content_hash, source: human|generated|ingested,
             provenance, trust_tier, eval_score,
             status: draft|candidate|trusted|deprecated, created_at }
```

Generated templates and distilled context are **untrusted until promoted** through the one **promotion gate** (safety scan + eval win vs incumbent + HITL/canary → `trusted`, version-pinned). The provider's workflow/skill store **is** the executor's skill/capability store — one store, not two. (An MCP server is just a `capability` Artifact; it does not need a separate silo.)

> ⚠️ **Reality check (net-new, unowned).** The Artifact envelope is **designed-only** — it exists in
> neither the executor nor the provider code today. It is a shared contract that must be **built and
> owned before Phase 1 can stand on it** (see Open decisions). Until then, provider content-hash
> dedup (`_content_hash`, already shipped) is the interim provenance/identity mechanism.

---

## Phase 0 — Unblock ✅ DONE

Fixed the bugs that made the goals impossible; no redesign. **Merged** (provider `ed0a3cc`,
memory-unit `4ca5610`):
- ✅ **memory-unit:** reconciled `api.py`↔`core.py` signatures so `/hydrate` and `/refresh` stop
  throwing; added an `X-User-Id` owner guard + `ALLOWED_ORIGINS` CORS. *(Still open — token
  **validation** server-side, and per-user data isolation; folded into Phase 4.)*
- ✅ **provider:** `context_items` is on the `Task` schema so `edit_task` stops dropping it;
  workflows produced by `create_workflow_initial` **persist** into `generated_workflows`; added
  content-hash dedup; enforced search-before-create on `/create_workflow`.
- ⚠️ **both / carry-over:** move Chroma off ephemeral storage. memory-unit uses a disk
  `PersistentClient`; the provider still writes to an in-image path. The real fix (a mounted volume
  or a managed DB) **is Phase 1** — left as-is deliberately (a path change alone is cosmetic on
  Cloud Run).

**Acceptance (met):** `/hydrate` returns 200; an `edit_task` that changes a parameter round-trips it;
a generated workflow is retrievable by a later search; offline suites green (provider 33,
memory-unit 51).

## Phase 1 — The shared datastore + data model  ⭐ foundation ⚠️

Stand up **Postgres + a vector index** (pgvector, or one managed vector DB) as the single store.
Define the schemas once; all services use them.

- **Session** (`thread_id` PK): `user_id, task_id (executor mapping), conversation[], state, chosen_template_id?, chosen_instance_id?, hitl_answers[]`.
- **Task**: `task_id, thread_id, user_id, type, name, description, signature: Slot[], status, chosen_instance_id?`.
- **Slot** (the parameter): `name, type, required, value?, status: present|missing|guessed, source: email|context|user|guessed|tool, confidence`.
- **WorkflowTemplate** (Artifact `kind=template`): `+ required_slots: SlotSpec[], steps: Step[], tags, embedding, scope: global|org|role|user`.
- **Step** (typed, provider-internal): `kind: tool|skill|llm|hitl|subtemplate, ref, input_bindings, output_name`. **Serialized to `List[str]` at the executor boundary** (see the adapter note in Phase 3).
- **EnrichedInstance** (Artifact `kind=instance`): `template_id@version, bound_slots, specialization_scope, task_id, outcome?, trace_id`.
- **Skill / Capability / ContextBlock**: the executor's Artifacts, in the same store.

> ⚠️ **Blockers before starting:** (a) **ownership** of the store is an open decision, and (b) it must
> **reconcile with the executor's existing Postgres** (checkpointing + status). Do not stand up a
> second, competing Postgres. This phase also assumes the Artifact envelope exists (it doesn't yet).

**Acceptance:** every service reads/writes tasks, templates, instances, and session by `thread_id`
from one store; nothing pipeline-critical is in `/tmp` or process memory (P-STORE1).

## Phase 2 — The typed slot model (the spine) ⚠️ — recommended next, unblocked

Make parameters first-class end to end. This is what unlocks five of the six goals, and it is
**doable entirely within the provider** on top of the `context_items` we already carry.

1. **Signature at identification.** `identify_task` emits a `Task` whose `signature` is typed slots
   (evolve the existing `ContextItem{field,status,value}` into `Slot{name,type,required,value,source,confidence}`).
   Persist them on the Task (P-SLOT1). *(Foundation already present — `Task.context_items`.)*
2. **Population step.** Before HITL, resolve slots from user context: call
   `memory-unit.resolve(signature, user_id, thread_id)` (Phase 4) → fill `value/source/confidence`.
   Remaining `missing`/low-confidence slots become HITL questions — and **only** those (P-CTX-POP1).
3. **Conversation edits all three.** `edit_task` operates on a `Task` that includes `signature`, so
   the flow can change **name, description, and slots** with field-level intent. *(edit_task already
   round-trips `context_items`; extend it to typed slot edits.)*
4. **Provenance + confidence drive HITL.** The extension already models `source`; carry it through so
   the UI shows where each value came from and asks only for gaps.

**Acceptance:** a task's slots are populated from context where available, with source+confidence;
the conversation can edit a slot's value and have it persist; HITL only asks for genuinely
missing/uncertain slots.

## Phase 3 — Template → enriched-instance hierarchy + search-before-create ⚠️

1. **Two levels.** Promote `Workflow` to a versioned **Template** (generic, parameterized by
   `required_slots`, typed `Step`s) and an **EnrichedInstance** (template + bound slots +
   org/role/user specialization). Search matches **templates**; enrichment **binds slots** to produce
   an instance.
2. **Search-before-create, enforced.** ✅ already enforced on `/create_workflow` and `/enrich`;
   extend it to templates with a real similarity **threshold/score** (not just an LLM yes/no) and keep
   persisting every created template as a `candidate` Artifact (P-SBC1). Dedup today is exact
   content-hash only; semantic near-dup is the follow-up.
3. **Lineage.** Instances record `template_id@version`; specialized variants are child templates with
   `parent_id` — the structure the refinement loop (Phase 5) needs to attribute outcomes.

> ⚠️ **Executor-boundary adapter (contract-safe, the chosen approach).** Typed `Step`s are
> **provider-internal only.** When the provider calls the executor's `/workflow/execute`, it
> **serializes `Step[]` back down to `workflow_steps: List[str]`** and sends only the fields the
> executor accepts (`task_*`, `workflow_*`, `context_items`, `metadata`). The executor now sets
> `extra="forbid"` and **422s on unknown fields**, so this serialization is mandatory — never send
> the typed Step objects. No executor change is required and the live contract is untouched. A future
> typed-Step contract on the executor is a separate cross-team decision, not a dependency here.

**Acceptance:** a task that matches an existing template never creates a duplicate; a created
template is searchable next time; an executed instance records exactly which template version + bound
slots it ran; the executor still receives `List[str]` steps and returns 2xx.

## Phase 4 — memory-unit as context + parameter resolver (and write-back) ⚠️ — recommended next, unblocked

Turn the durable-but-orphaned retriever into the pipeline's context brain. **Within our two repos.**

1. **Structured resolution.** Add `resolve(task_signature, user_id, scope) → [{field, value, source,
   confidence}]` alongside the existing prose `query()` — typed slot values, not "documents that
   mention the field." Back it with an entity/value store (defaults, preferences) plus the existing
   RAG for evidence.
2. **Hierarchical scope.** Resolve across `global → org → role → user → thread`, specific wins — the
   same hierarchy as the executor's ContextProvider.
3. **Write-back loop.** Completed tasks/traces distill into durable context (default recipient,
   preferred meeting length, project numbers) written back into the store — so the memory
   self-learns (P-LEARN1). *(Needs a Drive upload method — `drive/client.py` is read-only today;
   `upload_url` is stubbed. Write-back may instead live in the extension, which already holds the
   Drive write scope — see Open decisions.)* Distilled context blocks are Artifacts and pass the
   promotion gate.
4. **Wire it in + harden.** Have the task-identifier (Phase 2) and workflow-builder call `resolve()`;
   add server-side token **validation** + per-user data isolation (Phase 0 added the owner guard but
   only *extracts* the bearer); treat retrieved context as untrusted data (P-SEC3 fencing).

**Acceptance:** the task-identifier fills slots from `resolve()`; a value confirmed via HITL on one
task is available (as context) on the next; memory-unit is on the critical path.

## Phase 5 — One trace, one refinement loop 🚫 BLOCKED (needs the executor's eval harness)

1. **One pipeline = one trace.** Thread a single `thread_id` through identify → populate → search →
   enrich → execute → resume; add **retrieval** tracing (provider traces LLM calls only) and add
   tracing to memory-unit (`X-Thread-Id` is dropped today). Record artifact versions (P-TRACE1).
2. **Trace retrieval quality** — search recall/precision, slot-fill accuracy, template-match rate.
3. **Close the analyzer loop.** Replace the dead-end `knowledge/*.txt` writer with the executor's
   improvement loop: traces → evaluation → propose **template** edits and **context** refinements →
   promotion gate → store (P-LEARN1).

> 🚫 **Hard dependency:** steps 3 (and the promotion gate referenced in Phases 3–4) require the
> **executor's evaluation harness — golden set + replay + scorers — which does not exist yet**
> (executor Phase 7, not started; no test suite in that repo). Steps 1–2 (tracing plumbing) can
> proceed independently; the refinement loop cannot close until the gate exists. Track as a
> cross-team dependency, not a "later" item.

**Acceptance:** a whole pipeline run is one queryable DeepEval trace; a recurring successful pattern
yields a `candidate` template/context refinement that is only promoted after beating the incumbent on
eval; refinements are attributable to the traces they came from.

---

## Sequencing

Original linear sequence, **re-annotated for what's done and what's blocked**:

1. ✅ **Phase 0** — unblock (done, merged).
2. ⚠️ **Phase 1** — shared store + schemas. *Foundation, but gated on ownership + reconciling with the
   executor's Postgres + the Artifact envelope existing.*
3. ⚠️ **Phase 2** — typed slots (unlocks population + conversation-edit). **Unblocked, in-repo.**
4. ⚠️ **Phase 3** — template/instance + threshold search-before-create, with the executor-boundary
   adapter.
5. ⚠️ **Phase 4** — memory-unit resolver + write-back, wired in. **Mostly in-repo** (write-back may be
   an extension task).
6. 🚫 **Phase 5** — unified trace + refinement loop (depends on 1–4 **and** the executor's eval
   harness, which is not started).

**Recommended near-term path (unblocked, in our two repos, delivers the context-aware win):**
do **Phase 2 + Phase 4 together** — type the slots on the provider, add `memory-unit.resolve()`, and
wire the provider's population step to call it. This makes memory-unit part of the live pipeline
without waiting on the shared store or the eval harness. Ship behind flags; treat Phase 1 as a
migration that these can later move onto.

> **Implemented this session (flag-gated MVP).** A working slice of this path is in code:
> - memory-unit — `MemoryUnit.resolve(fields, ...)` returns structured `{field, value, evidence,
>   source, confidence, status}` slots (deterministic BM25 over the unified index, vector fallback),
>   exposed as `POST /resolve` behind the owner guard. Values are **extracted** to a concise form
>   (email/number/clause), keeping the snippet as `evidence`. **Server-side Google token validation**
>   now runs at `/hydrate` + `/refresh` (`memory_unit/auth.py`; on by default, `MEMORY_VALIDATE_TOKEN=false`
>   to disable). **Write-back** (`learn()` + `POST /learn`) ingests distilled context into the index
>   and a durable JSONL re-applied on hydrate.
> - provider — `ContextItem` gained optional `source`/`confidence` (additive; the executor ignores the
>   unknown nested fields); `utils/memory_client.py` (stdlib-only, flag-gated on `MEMORY_URL`, never
>   raises) + `utils/population.py` fill *missing* slots and mark them `guessed` with provenance;
>   `POST /populate_task_context` and, behind `MEMORY_AUTO_POPULATE` (default off), the live
>   `/identify_task` flow both wire it in. Behavior unchanged when the flags are unset.
> - Tests: memory-unit 75, provider 41 (offline). Fixed a real bug: snake_case slot names matched
>   nothing (BM25 tokenizer drops `_`-joined words). **Still open in this phase:** hierarchical scope,
>   and *durable* Drive write-back (today's write-back persists to `persist_dir`, ephemeral on Cloud
>   Run until the Phase-1 shared store / extension-owned Drive write lands).

## Risks / gotchas
- **Executor boundary is `List[str]` + `extra="forbid"`.** Any typed-Step or extra field sent to
  `/workflow/execute` is a 422. The adapter that serializes `Step[] → List[str]` (Phase 3) is
  mandatory, not optional.
- **Migration:** existing Chroma corpora (provider) must be backfilled into the shared store with
  stable IDs; the `to_string()`/`_workflow_from_document` round-trip loses fidelity on the string path
  — migrate from **metadata/source objects** (workflows are already recovered metadata-first), not
  parsed docs.
- **Slot typing creep:** keep the type system small (string/date/email/enum/number/ref) — don't build
  a type theory.
- **Resolver confidence:** a wrong high-confidence slot value is worse than a missing one — calibrate,
  and prefer HITL when uncertain.
- **Two teams, one store:** the shared schema is a contract — version it; additive-only within a major
  version. Reconcile with the executor's existing Postgres rather than standing up a second one.
- **Write-back poisoning:** distilling context from a hostile email can persist an injection — pass
  write-back context through the promotion gate's safety scan (same as generated skills).
- **`thread_id` overload:** it is a session PK here but a trace tag elsewhere and *not* the executor's
  `task_id` — keep the `thread_id → task_id` mapping explicit.

## Open decisions (settle before Phase 1)
- **Who builds/owns the Artifact envelope** (net-new; neither repo has it). Blocks Phase 1's "all are Artifacts."
- **Where the shared store lives** and who owns it, **given the executor already runs Postgres.** Recommendation: one Postgres+pgvector, thin shared data-access lib; extend the executor's instance rather than add a second.
- **Vector DB:** pgvector (transactional with the relational data) vs. a managed vector store. Recommendation: pgvector unless scale forces otherwise.
- **Param resolution ownership:** memory-unit (recommended — it already holds context) vs. the task-identifier calling raw retrieval.
- **Write-back ownership:** memory-unit (needs a new Drive upload method) vs. the extension (already holds the Drive write scope + OAuth token). Recommendation: extension, gated by a safety scan.
- **Unify the provider's template/skill store with the executor's capability store** (recommended) vs. keep separate and sync.

## Out of scope (deliberately)
- Rewriting the agents' prompting strategy — this is about state, schema, and the loop, not prompt tuning.
- Replacing ChromaDB everywhere on day one — migrate behind a flag.
- The extension/UI — it already models slot `source`; it consumes these APIs, it doesn't drive the redesign.
- Modifying `workflow_executor` — we consume its contract (`List[str]` steps) and do not change it here.
