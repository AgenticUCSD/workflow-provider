import uuid
import json
import hashlib
from typing import List, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from utils.task import Task, Workflow
from utils.config import CHROMA_PERSIST_DIR, OPENAI_API_KEY


class ChromaVectorStore:
    def __init__(self):
        # There are options for models
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002",
        )

        # Keep a handle so collections re-created in clear_collection() use the
        # same embedding function. Recreating without it persists a "default" EF,
        # which then conflicts with this openai EF on the next startup.
        self._embedding_fn = openai_ef

        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.manual_workflows = self.client.get_or_create_collection(
            name="manual_workflows",
            embedding_function=openai_ef,
        )
        self.generated_workflows = self.client.get_or_create_collection(
            name="generated_workflows",
            embedding_function=openai_ef,
        )

    @staticmethod
    def _content_hash(workflow: Workflow) -> str:
        """Stable content hash of a workflow, used to skip exact-duplicate inserts.

        Based on the same string we embed (`to_string()`), so two workflows that
        would embed identically hash identically.
        """
        return hashlib.sha256(workflow.to_string().encode("utf-8")).hexdigest()

    def add_workflow(self, workflow: Workflow, is_generated=True):
        collection = self.generated_workflows if is_generated else self.manual_workflows
        content_hash = self._content_hash(workflow)

        # Dedup gate: if an identical workflow already lives in this collection, skip
        # the insert and return the existing id. The lookup is a metadata `get` — no
        # embedding call — so it stays cheap and offline-testable. (Exact-content
        # dedup only; semantic near-dup matching is a deliberate follow-up.)
        existing = collection.get(where={"content_hash": content_hash})
        existing_ids = existing.get("ids") if isinstance(existing, dict) else None
        if existing_ids:
            return existing_ids[0]

        document_id = str(uuid.uuid4())
        collection.add(
            documents=[workflow.to_string()],
            ids=[document_id],
            metadatas=[
                {
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "steps": json.dumps(workflow.steps),
                    "content_hash": content_hash,
                }
            ],
        )
        return document_id

    def query_workflows(self, task: Task, top_k=5, is_generated=True):
        collection = self.generated_workflows if is_generated else self.manual_workflows
        results = collection.query(
            query_texts=[task.to_string()],
            n_results=top_k,
        )
        return results

    def query_from_all_workflows(self, task: Task, top_k=5):
        results = {}
        manual_results = self.query_workflows(task, top_k=top_k, is_generated=False)
        generated_results = self.query_workflows(task, top_k=top_k, is_generated=True)
        results["manual_workflows"] = manual_results
        results["generated_workflows"] = generated_results
        return results

    def _workflow_from_metadata(self, metadata: dict | None) -> Optional[Workflow]:
        if not isinstance(metadata, dict):
            return None

        steps_raw = metadata.get("steps", "[]")
        try:
            steps = json.loads(steps_raw) if isinstance(steps_raw, str) else steps_raw
        except Exception:
            steps = []

        if not isinstance(steps, list):
            steps = []

        try:
            return Workflow.model_validate(
                {
                    "workflow_id": metadata.get("workflow_id") or str(uuid.uuid4()),
                    "name": metadata.get("name", ""),
                    "description": metadata.get("description", ""),
                    "steps": steps,
                }
            )
        except Exception:
            return None

    def _workflow_from_document(self, document: str | None) -> Optional[Workflow]:
        if not isinstance(document, str) or not document.strip():
            return None

        lines = [line.strip() for line in document.splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        name_prefix = "Workflow: "
        description_prefix = "Description: "
        steps_prefix = "- "

        name = ""
        description = ""
        steps: List[str] = []

        for line in lines:
            if line.startswith(name_prefix):
                name = line[len(name_prefix):].strip()
            elif line.startswith(description_prefix):
                description = line[len(description_prefix):].strip()
            elif line.startswith(steps_prefix):
                steps.append(line[len(steps_prefix):].strip())

        if not name:
            return None

        synthetic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}|{description}|{'|'.join(steps)}"))
        try:
            return Workflow.model_validate(
                {
                    "workflow_id": synthetic_id,
                    "name": name,
                    "description": description,
                    "steps": steps,
                }
            )
        except Exception:
            return None

    def _extract_workflows_from_query(self, query_results: dict) -> List[Workflow]:
        if not isinstance(query_results, dict):
            return []

        metadatas = query_results.get("metadatas") or [[]]
        documents = query_results.get("documents") or [[]]

        metadata_list = metadatas[0] if metadatas and isinstance(metadatas[0], list) else []
        document_list = documents[0] if documents and isinstance(documents[0], list) else []

        workflows: List[Workflow] = []
        seen_ids = set()

        max_len = max(len(metadata_list), len(document_list))
        for idx in range(max_len):
            metadata = metadata_list[idx] if idx < len(metadata_list) else None
            document = document_list[idx] if idx < len(document_list) else None

            workflow = self._workflow_from_metadata(metadata)
            if workflow is None:
                workflow = self._workflow_from_document(document)

            if workflow and workflow.workflow_id not in seen_ids:
                seen_ids.add(workflow.workflow_id)
                workflows.append(workflow)

        return workflows

    def query_from_all_workflows_as_objects(self, task: Task, top_k=5) -> List[Workflow]:
        manual_results = self.query_workflows(task, top_k=top_k, is_generated=False)
        generated_results = self.query_workflows(task, top_k=top_k, is_generated=True)

        manual_workflows = self._extract_workflows_from_query(manual_results)
        generated_workflows = self._extract_workflows_from_query(generated_results)

        merged = manual_workflows + generated_workflows
        deduped: List[Workflow] = []
        seen_ids = set()
        for workflow in merged:
            if workflow.workflow_id not in seen_ids:
                seen_ids.add(workflow.workflow_id)
                deduped.append(workflow)

        return deduped

    def clear_collection(self, is_generated=True):
        collection_name = (
        "generated_workflows" if is_generated else "manual_workflows"
        )
        self.client.delete_collection(name=collection_name)
        if is_generated:
            self.generated_workflows = self.client.get_or_create_collection(
                name="generated_workflows",
                embedding_function=self._embedding_fn,
            )
        else:
            self.manual_workflows = self.client.get_or_create_collection(
                name="manual_workflows",
                embedding_function=self._embedding_fn,
            )

    def clear_collections(self):
        self.clear_collection(is_generated=False)
        self.clear_collection(is_generated=True)

    def add_single_workflow(self, workflow: Workflow, is_generated: bool = False) -> str:
        """Add a single workflow to the vector store."""
        return self.add_workflow(workflow, is_generated=is_generated)

    def get_all_workflows(self) -> List[Workflow]:
        """Retrieve all workflows from both collections."""
        workflows: List[Workflow] = []

        manual_count = self.manual_workflows.count()
        generated_count = self.generated_workflows.count()

        if manual_count > 0:
            manual_results = self.manual_workflows.get(
                ids=None,
                include=["metadatas", "documents"]
            )
            workflows.extend(self._extract_workflows_from_query({
                "metadatas": [manual_results.get("metadatas", [])],
                "documents": [manual_results.get("documents", [])]
            }))

        if generated_count > 0:
            generated_results = self.generated_workflows.get(
                ids=None,
                include=["metadatas", "documents"]
            )
            workflows.extend(self._extract_workflows_from_query({
                "metadatas": [generated_results.get("metadatas", [])],
                "documents": [generated_results.get("documents", [])]
            }))

        return workflows
