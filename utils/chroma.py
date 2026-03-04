import uuid
import json
from typing import List, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from task_identification.task import Task, Workflow
from utils.config import CHROMA_PERSIST_DIR, OPENAI_API_KEY


class ChromaVectorStore:
    def __init__(self):
        # There are options for models
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002",
        )

        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.manual_workflows = self.client.get_or_create_collection(
            name="manual_workflows",
            embedding_function=openai_ef,
        )
        self.generated_workflows = self.client.get_or_create_collection(
            name="generated_workflows",
            embedding_function=openai_ef,
        )

    def add_workflow(self, workflow: Workflow, is_generated=True):

        document_id = str(uuid.uuid4())
        collection = self.generated_workflows if is_generated else self.manual_workflows
        collection.add(
            documents=[workflow.to_string()],
            ids=[document_id],
            metadatas=[
                {
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "steps": json.dumps(workflow.steps),
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
            )
        else:
            self.manual_workflows = self.client.get_or_create_collection(
                name="manual_workflows",
            )

    def clear_collections(self):
        self.clear_collection(is_generated=False)
        self.clear_collection(is_generated=True)
