"""Phase 3 — persistence for workflow templates.

A Chroma-backed store for ``WorkflowTemplate``s, separate from the flat-workflow
collections so it is fully additive. Provides:
- content-hash dedup (exact-content, like the workflow store)
- versioning (``next_version``) and lineage (``parent_id``)
- **score-based threshold search** — returns a distance + monotonic score per
  match so callers can threshold, instead of relying on an LLM yes/no.
"""

import hashlib
import uuid
from typing import Any, Dict, List, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from utils.config import CHROMA_PERSIST_DIR, OPENAI_API_KEY
from utils.template import WorkflowTemplate


class TemplateStore:
    def __init__(self):
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002",
        )
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.templates = self.client.get_or_create_collection(
            name="workflow_templates",
            embedding_function=openai_ef,
        )

    @staticmethod
    def _content_hash(template: WorkflowTemplate) -> str:
        return hashlib.sha256(template.to_string().encode("utf-8")).hexdigest()

    def next_version(self, template_id: str) -> int:
        """Next version number for a template lineage (max existing + 1, else 1)."""
        existing = self.templates.get(where={"template_id": template_id})
        versions = [
            int(m.get("version", 0)) for m in (existing.get("metadatas") or []) if m
        ]
        return (max(versions) + 1) if versions else 1

    def add_template(self, template: WorkflowTemplate, dedup: bool = True) -> str:
        """Persist a template. On an exact content-hash match (when ``dedup``) skip
        the insert and return the existing document id."""
        content_hash = self._content_hash(template)

        if dedup:
            existing = self.templates.get(where={"content_hash": content_hash})
            existing_ids = existing.get("ids") if isinstance(existing, dict) else None
            if existing_ids:
                return existing_ids[0]

        document_id = str(uuid.uuid4())
        self.templates.add(
            documents=[template.to_string()],
            ids=[document_id],
            metadatas=[
                {
                    "template_id": template.template_id,
                    "version": int(template.version),
                    "parent_id": template.parent_id or "",
                    "status": template.status,
                    "source": template.source,
                    "name": template.name,
                    "description": template.description,
                    "content_hash": content_hash,
                    "template_json": template.model_dump_json(),
                }
            ],
        )
        return document_id

    def add_new_version(self, template: WorkflowTemplate) -> str:
        """Persist ``template`` as the next version of its lineage (bumps version)."""
        template.version = self.next_version(template.template_id)
        return self.add_template(template, dedup=False)

    def _template_from_metadata(self, metadata: Optional[dict]) -> Optional[WorkflowTemplate]:
        if not isinstance(metadata, dict):
            return None
        raw = metadata.get("template_json")
        if not raw:
            return None
        try:
            return WorkflowTemplate.model_validate_json(raw)
        except Exception:
            return None

    def _all_for_lineage(self, template_id: str) -> List[WorkflowTemplate]:
        res = self.templates.get(where={"template_id": template_id})
        metas = res.get("metadatas") or []
        return [t for t in (self._template_from_metadata(m) for m in metas) if t]

    def get_template(
        self, template_id: str, version: Optional[int] = None
    ) -> Optional[WorkflowTemplate]:
        """Fetch a template by id — the latest version by default, or a specific one."""
        templates = self._all_for_lineage(template_id)
        if not templates:
            return None
        if version is not None:
            for t in templates:
                if t.version == version:
                    return t
            return None
        return max(templates, key=lambda t: t.version)

    def list_versions(self, template_id: str) -> List[int]:
        return sorted(t.version for t in self._all_for_lineage(template_id))

    def children_of(self, template_id: str) -> List[WorkflowTemplate]:
        """Templates whose ``parent_id`` is ``template_id`` (specializations)."""
        res = self.templates.get(where={"parent_id": template_id})
        metas = res.get("metadatas") or []
        return [t for t in (self._template_from_metadata(m) for m in metas) if t]

    def search_templates(
        self, query_text: str, top_k: int = 5, max_distance: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search returning ``[{template, distance, score}]`` sorted by
        proximity. ``score = 1/(1+distance)`` is monotonic (higher = closer); when
        ``max_distance`` is given, farther matches are dropped."""
        results = self.templates.query(query_texts=[query_text], n_results=top_k)
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        out: List[Dict[str, Any]] = []
        for idx, meta in enumerate(metas):
            template = self._template_from_metadata(meta)
            if template is None:
                continue
            distance = float(dists[idx]) if idx < len(dists) and dists[idx] is not None else 0.0
            if max_distance is not None and distance > max_distance:
                continue
            out.append(
                {
                    "template": template,
                    "distance": distance,
                    "score": round(1.0 / (1.0 + distance), 3),
                }
            )
        out.sort(key=lambda r: r["distance"])
        return out
