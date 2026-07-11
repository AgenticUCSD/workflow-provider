"""Phase 1 — Postgres backend for the workflow-template store.

A drop-in replacement for :class:`utils.template_store.TemplateStore` that reads and
writes ``planner.workflow_templates`` in the shared Cloud SQL store instead of Chroma.
Selected by ``STORE_BACKEND=pg`` (see :func:`utils.config.make_template_store`); the
Chroma store stays the default and this module is imported only when the flag picks it.

Same public interface, same content-hash dedup, and the same monotonic
``score = 1/(1+distance)`` search ranking as the Chroma store — only the backend
differs. Embeddings use the same ``text-embedding-ada-002`` model, so vectors are
comparable across backends.

Connects as the least-privilege ``planner_app`` role via ``PLANNER_DATABASE_URL``
(DML-only on ``planner.*``). Vectors are sent as pgvector literals with an explicit
``::vector`` cast, so no extra driver-side adapter is needed (only ``psycopg``); the
pgvector *extension* lives in the database.
"""

from typing import Any, Dict, List, Optional

import chromadb.utils.embedding_functions as embedding_functions

from utils.config import (
    PLANNER_DATABASE_URL,
    openai_api_key_or_placeholder,
    template_near_dup_distance,
)
from utils.template import SlotSpec, Step, WorkflowTemplate, scope_rank
from utils.template_store import TemplateStore
from utils.tracing import traced

_TABLE = "planner.workflow_templates"
# Columns needed to reconstruct a WorkflowTemplate (excludes the heavy embedding).
_COLS = (
    "template_id, version, name, description, required_slots, steps, "
    "tags, parent_id, status, source, scope"
)


def _vector_literal(vec: Optional[List[float]]) -> Optional[str]:
    """pgvector text literal (``[1,2,3]``) for a param cast with ``::vector``; None → NULL."""
    if vec is None:
        return None
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class PGTemplateStore:
    """Postgres/pgvector-backed template store (drop-in for ``TemplateStore``)."""

    def __init__(self, database_url: Optional[str] = None):
        url = database_url if database_url is not None else PLANNER_DATABASE_URL
        if not url:
            raise RuntimeError(
                "STORE_BACKEND=pg but PLANNER_DATABASE_URL is not set. Provide the "
                "planner_app connection string, or unset STORE_BACKEND to use Chroma."
            )
        # Lazy imports so importing this module (e.g. under a skipped test) never
        # hard-requires the pg driver; only constructing the store does.
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

    # ---- connection -------------------------------------------------------
    def _connect(self):
        return self._psycopg.connect(
            self._url, autocommit=True, row_factory=self._dict_row
        )

    # ---- embeddings -------------------------------------------------------
    def _embed(self, text: str) -> Optional[List[float]]:
        """Embed ``text`` with the same model as Chroma. Best-effort: on failure
        returns None so the template is still stored (with a NULL embedding — it is
        id-retrievable, just not returned by semantic search)."""
        try:
            return list(self._embedding_fn([text])[0])
        except Exception:
            return None

    # ---- content hash (reuse the Chroma store's rule verbatim) ------------
    @staticmethod
    def _content_hash(template: WorkflowTemplate) -> str:
        return TemplateStore._content_hash(template)

    # ---- (de)serialization ------------------------------------------------
    @staticmethod
    def _template_from_row(row: Dict[str, Any]) -> WorkflowTemplate:
        return WorkflowTemplate(
            template_id=row["template_id"],
            version=int(row["version"]),
            name=row["name"],
            description=row.get("description") or "",
            required_slots=[SlotSpec(**s) for s in (row.get("required_slots") or [])],
            steps=[Step(**s) for s in (row.get("steps") or [])],
            tags=list(row.get("tags") or []),
            parent_id=(row.get("parent_id") or None),
            source=row.get("source") or "generated",
            status=row.get("status") or "draft",  # envelope default (conforms to executor artifacts)
            scope=row.get("scope") or "global",
        )

    # ---- public interface (mirrors TemplateStore) -------------------------
    def next_version(self, template_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COALESCE(MAX(version), 0) + 1 AS v FROM {_TABLE} "
                "WHERE template_id = %s",
                (template_id,),
            ).fetchone()
        return int(row["v"]) if row and row["v"] is not None else 1

    def _near_duplicate_id(self, embedding: str) -> Optional[str]:
        """``"template_id:version"`` of an existing near-identical template, or None.

        Off unless ``TEMPLATE_NEAR_DUP_DISTANCE`` is set: compares the candidate's
        ``embedding`` (a ``::vector`` literal) to the single nearest stored template;
        a cosine distance ≤ threshold counts as a near-dup. Best-effort — any error
        yields None so the insert proceeds."""
        threshold = template_near_dup_distance()
        if threshold is None:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    f"""
                    SELECT template_id, version, (embedding <=> %s::vector) AS distance
                    FROM {_TABLE}
                    WHERE embedding IS NOT NULL
                    ORDER BY distance
                    LIMIT 1
                    """,
                    (embedding,),
                ).fetchone()
            if row and row["distance"] is not None and float(row["distance"]) <= threshold:
                return f"{row['template_id']}:{row['version']}"
        except Exception:
            return None
        return None

    def add_template(self, template: WorkflowTemplate, dedup: bool = True) -> str:
        """Persist a template. When ``dedup``: an exact content-hash match is skipped
        (returns the existing ``template_id:version``); and if
        ``TEMPLATE_NEAR_DUP_DISTANCE`` is set, a semantically near-identical template
        (cosine distance ≤ threshold) is also skipped."""
        content_hash = self._content_hash(template)

        if dedup:
            with self._connect() as conn:
                row = conn.execute(
                    f"SELECT template_id, version FROM {_TABLE} "
                    "WHERE content_hash = %s LIMIT 1",
                    (content_hash,),
                ).fetchone()
            if row:
                return f"{row['template_id']}:{row['version']}"

        embedding = _vector_literal(self._embed(template.to_string()))

        if dedup and embedding is not None:
            near = self._near_duplicate_id(embedding)
            if near is not None:
                return near

        required_slots = [s.model_dump() for s in template.required_slots]
        steps = [s.model_dump() for s in template.steps]
        version = int(template.version)

        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {_TABLE}
                    (template_id, version, name, description, required_slots, steps,
                     tags, parent_id, content_hash, embedding, status, source, scope)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s)
                ON CONFLICT (template_id, version) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    required_slots = EXCLUDED.required_slots,
                    steps = EXCLUDED.steps,
                    tags = EXCLUDED.tags,
                    parent_id = EXCLUDED.parent_id,
                    content_hash = EXCLUDED.content_hash,
                    embedding = EXCLUDED.embedding,
                    status = EXCLUDED.status,
                    source = EXCLUDED.source,
                    scope = EXCLUDED.scope
                """,
                (
                    template.template_id,
                    version,
                    template.name,
                    template.description,
                    self._Json(required_slots),
                    self._Json(steps),
                    list(template.tags),
                    template.parent_id,
                    content_hash,
                    embedding,
                    template.status,
                    template.source,
                    template.scope,
                ),
            )
        return f"{template.template_id}:{version}"

    def add_new_version(self, template: WorkflowTemplate) -> str:
        """Persist ``template`` as the next version of its lineage (bumps version)."""
        template.version = self.next_version(template.template_id)
        return self.add_template(template, dedup=False)

    def get_template(
        self, template_id: str, version: Optional[int] = None
    ) -> Optional[WorkflowTemplate]:
        """Fetch a template — the latest version by default, or a specific one."""
        with self._connect() as conn:
            if version is not None:
                row = conn.execute(
                    f"SELECT {_COLS} FROM {_TABLE} "
                    "WHERE template_id = %s AND version = %s",
                    (template_id, version),
                ).fetchone()
            else:
                row = conn.execute(
                    f"SELECT {_COLS} FROM {_TABLE} WHERE template_id = %s "
                    "ORDER BY version DESC LIMIT 1",
                    (template_id,),
                ).fetchone()
        return self._template_from_row(row) if row else None

    def list_versions(self, template_id: str) -> List[int]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT version FROM {_TABLE} WHERE template_id = %s ORDER BY version",
                (template_id,),
            ).fetchall()
        return [int(r["version"]) for r in rows]

    def children_of(self, template_id: str) -> List[WorkflowTemplate]:
        """Templates whose ``parent_id`` is ``template_id`` (specializations)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLS} FROM {_TABLE} WHERE parent_id = %s",
                (template_id,),
            ).fetchall()
        return [self._template_from_row(r) for r in rows]

    @traced(name="retrieval.template.search")
    def search_templates(
        self,
        query_text: str,
        top_k: int = 5,
        max_distance: Optional[float] = None,
        scope: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search returning ``[{template, distance, score}]``. ``score =
        1/(1+distance)`` is monotonic; when ``max_distance`` is given, farther
        matches are dropped.

        ``scope`` is an ordered preference list (most-specific first). When given,
        results are ranked by scope specificity **first**, then proximity — a
        more-specific-scoped template can win over a closer-but-less-specific one
        (mirrors memory-unit's ``resolve``); unscoped/unlisted templates rank last
        but are never dropped. When None, ranking is purely proximity (unchanged)."""
        qvec = _vector_literal(self._embed(query_text))
        if qvec is None:
            return []
        # Widen the SQL candidate set when scoping so a farther but more-specific
        # template can surface; scope re-ranking happens Python-side after fetch.
        limit = top_k * 3 if scope else top_k
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT {_COLS}, (embedding <=> %s::vector) AS distance
                FROM {_TABLE}
                WHERE embedding IS NOT NULL
                ORDER BY distance
                LIMIT %s
                """,
                (qvec, limit),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for row in rows:
            distance = float(row["distance"])
            if max_distance is not None and distance > max_distance:
                continue
            out.append(
                {
                    "template": self._template_from_row(row),
                    "distance": distance,
                    "score": round(1.0 / (1.0 + distance), 3),
                }
            )
        if scope:
            out.sort(key=lambda r: (scope_rank(r["template"].scope, scope), r["distance"]))
            out = out[:top_k]
        return out
