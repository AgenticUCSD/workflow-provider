import uuid

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from utils.task import Task, Workflow
from utils.config import OPENAI_API_KEY


class ChromaVectorStore:
    def __init__(self):
        # There are options for models
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002",
        )

        self.client = chromadb.PersistentClient(path="./chroma_db")
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
        collection.add(documents=[workflow.to_string()], ids=[document_id])
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
