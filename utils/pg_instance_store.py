"""Phase 1 — persistence for enriched instances (state layer, slice 4).

`/enrich_template` binds a template's slots into an ``EnrichedInstance`` — the record of
exactly which ``template_id@version`` + bound slots produced a runnable workflow. Today
that instance is returned and discarded. This store persists it to
``planner.enriched_instances`` when ``STORE_BACKEND=pg`` (Phase 3 lineage/attribution).

There is no pre-existing instance store, so the default is **no persistence**
(``NullInstanceStore``) — exactly today's behavior — and the pg store is opt-in via the
flag. Persistence is best-effort at the call site; a storage failure must never fail
enrichment.

The table holds **lineage + bindings only** (template ref, bound_slots, task_id, trace_id,
status/outcome); name/description/steps are derivable from the template and are not stored.
"""

from typing import Any, Dict, Optional

from utils.config import PLANNER_DATABASE_URL
from utils.template import EnrichedInstance

_TABLE = "planner.enriched_instances"


class NullInstanceStore:
    """No-op store — the default when persistence is off (STORE_BACKEND != pg)."""

    def add_instance(self, instance: EnrichedInstance, trace_id: Optional[str] = None) -> None:
        return None

    def get_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        return None


class PGInstanceStore:
    """Postgres-backed enriched-instance store (planner.enriched_instances)."""

    def __init__(self, database_url: Optional[str] = None):
        url = database_url if database_url is not None else PLANNER_DATABASE_URL
        if not url:
            raise RuntimeError(
                "STORE_BACKEND=pg but PLANNER_DATABASE_URL is not set. Provide the "
                "planner_app connection string, or unset STORE_BACKEND to disable persistence."
            )
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Json

        self._psycopg = psycopg
        self._dict_row = dict_row
        self._Json = Json
        self._url = url

    def _connect(self):
        return self._psycopg.connect(
            self._url, autocommit=True, row_factory=self._dict_row
        )

    def add_instance(
        self, instance: EnrichedInstance, trace_id: Optional[str] = None
    ) -> None:
        """Persist an instance's lineage. Idempotent on instance_id."""
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {_TABLE}
                    (instance_id, template_id, template_version, bound_slots,
                     specialization_scope, task_id, trace_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (instance_id) DO NOTHING
                """,
                (
                    instance.instance_id,
                    instance.template_id,
                    int(instance.template_version),
                    self._Json(dict(instance.bound_slots)),
                    instance.specialization_scope,
                    instance.task_id,
                    trace_id,
                ),
            )

    def get_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored lineage row (dict) for an instance, or None."""
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT instance_id, template_id, template_version, bound_slots,
                       specialization_scope, task_id, trace_id, outcome, status
                FROM {_TABLE} WHERE instance_id = %s
                """,
                (instance_id,),
            ).fetchone()
        return dict(row) if row else None
