"""Phase 1 — Postgres backend for the flat workflow store.

A drop-in replacement for :class:`utils.chroma.ChromaVectorStore` that reads and writes
``planner.workflows`` in the shared Cloud SQL store instead of the per-service Chroma
``manual_workflows`` / ``generated_workflows`` collections. Selected by ``STORE_BACKEND=pg``
(see :func:`utils.config.make_workflow_store`); the Chroma store stays the default and this
module is imported only when the flag picks it.

Implements the four methods any caller actually uses — ``add_workflow``,
``add_single_workflow``, ``query_from_all_workflows_as_objects``, ``get_all_workflows`` —
with the same content-hash dedup (per ``is_generated`` "collection") and the same
``ada-002`` embeddings. Retrieval feeds the SearchAgent's strict 95% LLM re-rank, so there
is no distance threshold to preserve; cosine (matching the HNSW index) is used.

Connects as the least-privilege ``planner_app`` role via ``PLANNER_DATABASE_URL``.
"""

import uuid
import logging
from typing import List, Optional

import chromadb.utils.embedding_functions as embedding_functions

from utils.chroma import ChromaVectorStore
from utils.config import PLANNER_DATABASE_URL, openai_api_key_or_placeholder
from utils.pg_template_store import _vector_literal
from utils.task import Task, Workflow
from utils.tracing import traced

_TABLE = "planner.workflows"


class PGWorkflowStore:
    """Postgres/pgvector-backed flat workflow store (drop-in for ``ChromaVectorStore``)."""

    def __init__(self, database_url: Optional[str] = None):
        url = database_url if database_url is not None else PLANNER_DATABASE_URL
        if not url:
            raise RuntimeError(
                "STORE_BACKEND=pg but PLANNER_DATABASE_URL is not set. Provide the "
                "planner_app connection string, or unset STORE_BACKEND to use Chroma."
            )
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Json

        self._psycopg = psycopg
        self._dict_row = dict_row
        self._Json = Json
        self._url = url
        self._embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key_or_placeholder(),
            model_name="text-embedding-ada-002",
        )

    # ---- helpers ----------------------------------------------------------
    def _connect(self):
        return self._psycopg.connect(
            self._url, autocommit=True, row_factory=self._dict_row
        )

    def _embed(self, text: str) -> Optional[List[float]]:
        """Best-effort embedding (same model as Chroma). On failure returns None so the
        workflow is still stored with a NULL embedding (id-retrievable, not searchable)."""
        try:
            return list(self._embedding_fn([text])[0])
        except Exception:
            return None

    @staticmethod
    def _content_hash(workflow: Workflow) -> str:
        return ChromaVectorStore._content_hash(workflow)

    @staticmethod
    def _workflow_from_row(row) -> Workflow:
        steps = row.get("steps") or []
        if not isinstance(steps, list):
            steps = []
        return Workflow(
            workflow_id=row.get("workflow_id") or str(uuid.uuid4()),
            name=row.get("name") or "",
            description=row.get("description") or "",
            steps=steps,
        )

    def _existing_doc_id(self, is_generated: bool, content_hash: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT doc_id FROM {_TABLE} "
                "WHERE is_generated = %s AND content_hash = %s LIMIT 1",
                (is_generated, content_hash),
            ).fetchone()
        return row["doc_id"] if row else None

    # ---- public interface (mirrors ChromaVectorStore) ---------------------
    def add_workflow(self, workflow: Workflow, is_generated: bool = True) -> str:
        """Persist a workflow into the given "collection". On an exact content-hash match
        within that collection, skip the insert and return the existing ``doc_id``."""
        content_hash = self._content_hash(workflow)

        existing = self._existing_doc_id(is_generated, content_hash)
        if existing:
            return existing

        doc_id = str(uuid.uuid4())
        embedding = _vector_literal(self._embed(workflow.to_string()))
        with self._connect() as conn:
            res = conn.execute(
                f"""
                INSERT INTO {_TABLE}
                    (doc_id, workflow_id, is_generated, name, description,
                     steps, content_hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (is_generated, content_hash) DO NOTHING
                RETURNING doc_id
                """,
                (
                    doc_id,
                    workflow.workflow_id,
                    is_generated,
                    workflow.name,
                    workflow.description,
                    self._Json(list(workflow.steps)),
                    content_hash,
                    embedding,
                ),
            ).fetchone()
        if res:
            return res["doc_id"]
        # Lost a race with a concurrent identical insert — return the winner's id.
        return self._existing_doc_id(is_generated, content_hash) or doc_id

    def add_single_workflow(self, workflow: Workflow, is_generated: bool = False) -> str:
        return self.add_workflow(workflow, is_generated=is_generated)

    @traced(name="retrieval.pg.query_all")
    def query_from_all_workflows_as_objects(
        self, task: Task, top_k: int = 5
    ) -> List[Workflow]:
        """Top-k nearest workflows from each "collection", merged (manual first) and
        deduped by ``workflow_id`` — the same shape ChromaVectorStore returns."""
        qvec = _vector_literal(self._embed(task.to_string()))
        if qvec is None:
            return []

        def _nearest(is_generated: bool):
            with self._connect() as conn:
                return conn.execute(
                    f"""
                    SELECT workflow_id, name, description, steps
                    FROM {_TABLE}
                    WHERE is_generated = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (is_generated, qvec, top_k),
                ).fetchall()

        # Fail open: a DB error (unreachable/slow/permission) degrades to "no
        # candidates" → the caller generates a fresh workflow, exactly like a
        # cold-start Chroma — rather than 500-ing the live request. Logged so a
        # persistent outage is still visible.
        try:
            rows = list(_nearest(False)) + list(_nearest(True))  # manual, then generated
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "pg workflow search failed; degrading to no-candidates: %s", exc
            )
            return []
        return self._dedup_by_workflow_id(self._workflow_from_row(r) for r in rows)

    def get_all_workflows(self) -> List[Workflow]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT workflow_id, name, description, steps FROM {_TABLE}"
            ).fetchall()
        return self._dedup_by_workflow_id(self._workflow_from_row(r) for r in rows)

    @staticmethod
    def _dedup_by_workflow_id(workflows) -> List[Workflow]:
        out: List[Workflow] = []
        seen = set()
        for wf in workflows:
            if wf.workflow_id not in seen:
                seen.add(wf.workflow_id)
                out.append(wf)
        return out
