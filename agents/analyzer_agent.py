import os
import uuid
from datetime import datetime
from typing import Literal, Optional

import requests
from deepeval.integrations.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from pydantic import BaseModel, Field

from utils.model import extract_structured_output, model

# Confident AI API
CONFIDENT_API_BASE = "https://api.confident-ai.com/v1"

# Knowledge file paths
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")
USER_PREFERENCES_FILE = os.path.join(KNOWLEDGE_DIR, "user_preferences.txt")
TASK_PATTERNS_FILE = os.path.join(KNOWLEDGE_DIR, "task_patterns.txt")
WORKFLOW_TRENDS_FILE = os.path.join(KNOWLEDGE_DIR, "workflow_trends.txt")

# Ensure knowledge directory exists
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# API endpoint path
ANALYZER_PATH = "/analyze_traces"


class TraceData(BaseModel):
    """Represents a single trace from the system."""
    trace_id: str
    input: str
    output: str
    metadata: dict = Field(default_factory=dict)
    spans: list[dict] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Result of analyzing traces."""
    status: Literal["success", "no_insights", "error"]
    summary: str
    user_preferences_added: list[str] = Field(default_factory=list)
    task_patterns_added: list[str] = Field(default_factory=list)
    workflow_trends_added: list[str] = Field(default_factory=list)


class KnowledgeUpdate(BaseModel):
    """What to write to knowledge files."""
    user_preferences: str = ""
    task_patterns: str = ""
    workflow_trends: str = ""


SYSTEM_PROMPT = """You are an analyzer agent that extracts patterns from workflow system traces.

Your job is to analyze trace data and update knowledge files with meaningful insights.

For each trace array you receive:
1. Read the current content of all three knowledge files
2. Analyze the traces to identify patterns in:
   - User preferences (communication style, meeting preferences, etc.)
   - Task patterns (common types, automation candidates)
   - Workflow trends (successful patterns, optimization opportunities)

3. Compare new insights against existing content:
   - If a similar trend exists: fold the new evidence into it (update timestamp/count)
   - If genuinely new: append as new entry with [YYYY-MM-DD HH:MM] timestamp
   - Be conservative - only record patterns with clear evidence

4. Write the updated content back to each file

File Format:
Each entry starts with [timestamp]. Multiple insights separated by newlines.
Example: "[2024-01-15 14:32] Prefers concise emails (seen in 3 traces)"

When folding into existing trends:
- Update the timestamp
- Increment evidence count if tracking it
- Keep the core insight intact
""".strip()


class AnalyzerAgent:
    def __init__(self) -> None:
        self.agent = create_agent(
            model=model,
            response_format=ToolStrategy(KnowledgeUpdate),
            system_prompt=SYSTEM_PROMPT,
            tools=[self.read_knowledge_file, self.write_knowledge_file],
        )

    def _agent_config(self, thread_id: str | None = None) -> dict:
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": [CallbackHandler()],
        }

    @staticmethod
    @tool
    def read_knowledge_file(filename: str) -> str:
        """Read the current content of a knowledge file.

        Args:
            filename: One of 'user_preferences.txt', 'task_patterns.txt', or 'workflow_trends.txt'

        Returns:
            Current file content, or empty string if file doesn't exist
        """
        file_map = {
            "user_preferences.txt": USER_PREFERENCES_FILE,
            "task_patterns.txt": TASK_PATTERNS_FILE,
            "workflow_trends.txt": WORKFLOW_TRENDS_FILE,
        }
        filepath = file_map.get(filename)
        if not filepath:
            return f"Error: Unknown file {filename}"

        if not os.path.exists(filepath):
            return ""

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    def fetch_traces_by_thread(self, thread_id: str) -> list["TraceData"]:
        """Fetch all traces for a given thread ID from Confident AI API.

        Args:
            thread_id: The thread ID to fetch traces for.

        Returns:
            List of TraceData objects from the thread.
        """
        api_key = os.environ.get("CONFIDENT_API_KEY")
        if not api_key:
            raise ValueError("CONFIDENT_API_KEY environment variable is required to fetch thread traces")

        url = f"{CONFIDENT_API_BASE}/threads/{thread_id}"
        headers = {"CONFIDENT_API_KEY": api_key}

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        if not data.get("success"):
            raise ValueError(f"Failed to fetch thread: {data.get('message', 'Unknown error')}")

        thread_data = data.get("data", {})
        traces = thread_data.get("traces", [])

        # Convert API traces to TraceData format
        trace_data_list = []
        for trace in traces:
            trace_data_list.append(
                TraceData(
                    trace_id=trace.get("uuid", str(uuid.uuid4())),
                    input=str(trace.get("input", "")),
                    output=str(trace.get("output", "")),
                    metadata={
                        "name": trace.get("name"),
                        "environment": trace.get("environment"),
                        "threadId": trace.get("threadId"),
                        "userId": trace.get("userId"),
                        "tags": trace.get("tags", []),
                    },
                    spans=[
                        {
                            "name": span.get("name"),
                            "type": span.get("type"),
                            "input": str(span.get("input", "")),
                            "output": str(span.get("output", "")),
                            "status": span.get("status"),
                        }
                        for span in trace.get("spans", [])
                    ],
                )
            )

        return trace_data_list

    @staticmethod
    @tool
    def write_knowledge_file(filename: str, content: str) -> str:
        """Write content to a knowledge file (full overwrite).

        Args:
            filename: One of 'user_preferences.txt', 'task_patterns.txt', or 'workflow_trends.txt'
            content: Complete new content for the file

        Returns:
            Success message or error
        """
        file_map = {
            "user_preferences.txt": USER_PREFERENCES_FILE,
            "task_patterns.txt": TASK_PATTERNS_FILE,
            "workflow_trends.txt": WORKFLOW_TRENDS_FILE,
        }
        filepath = file_map.get(filename)
        if not filepath:
            return f"Error: Unknown file {filename}"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {filename}"
        except Exception as e:
            return f"Error writing file: {e}"

    def analyze_traces(
        self,
        thread_id: str
    ) -> AnalysisResult:
        """Analyze all traces for a given thread ID and update knowledge files.

        Fetches traces from the Confident AI API for the given thread,
        then analyzes them for patterns and updates knowledge files.

        Args:
            thread_id: The thread ID to fetch traces for from Confident AI.

        Returns:
            AnalysisResult with status, summary, and any insights added.
        """
        # Fetch traces from thread
        try:
            traces = self.fetch_traces_by_thread(thread_id)
        except Exception as e:
            return AnalysisResult(
                status="error",
                summary=f"Failed to fetch traces for thread {thread_id}: {str(e)}"
            )

        if not traces:
            return AnalysisResult(
                status="no_insights",
                summary=f"No traces found in thread {thread_id}"
            )

        # Build the analysis prompt
        traces_text = "\n\n".join([
            f"Trace {i+1} (ID: {t.trace_id}):\nInput: {t.input[:500]}...\nOutput: {t.output[:500]}...\nMetadata: {t.metadata}"
            for i, t in enumerate(traces)
        ])

        content = f"""Analyze the following {len(traces)} traces from a workflow system runthrough.

TRACES:
{traces_text}

FIRST: Use read_knowledge_file to read all three knowledge files.

THEN: Analyze the traces and determine what new insights to add, considering:
- User preferences: communication style, format preferences, timing, etc.
- Task patterns: recurring task types, common sequences, automation opportunities
- Workflow trends: successful patterns, common step counts, optimization ideas

When comparing to existing content:
- If a similar trend exists with [timestamp], fold new evidence into it
- Only append genuinely new insights with current timestamp [""" + datetime.now().strftime("%Y-%m-%d %H:%M") + """]
- Be specific: include evidence like "seen in X of Y traces"

FINALLY: Use write_knowledge_file to write the updated content for each file that needs changes.

Return a KnowledgeUpdate with the complete content for each file (read → merge → write).
"""

        chat = [{"role": "user", "content": content}]

        try:
            result = self.agent.invoke({"messages": chat}, config=self._agent_config(thread_id))
            parsed = extract_structured_output(result, KnowledgeUpdate)

            if parsed is None:
                return AnalysisResult(
                    status="error",
                    summary="Could not parse analysis result"
                )

            # Track what was added
            files_updated = []
            user_prefs = []
            task_pats = []
            wf_trends = []

            if parsed.user_preferences:
                files_updated.append("user_preferences.txt")
                # Extract added lines (simplistic - lines not in original)
                user_prefs = [line for line in parsed.user_preferences.split("\n") if line.strip()]

            if parsed.task_patterns:
                files_updated.append("task_patterns.txt")
                task_pats = [line for line in parsed.task_patterns.split("\n") if line.strip()]

            if parsed.workflow_trends:
                files_updated.append("workflow_trends.txt")
                wf_trends = [line for line in parsed.workflow_trends.split("\n") if line.strip()]

            # Build summary
            total_insights = len(user_prefs) + len(task_pats) + len(wf_trends)
            summary = f"Analyzed {len(traces)} traces. Updated {len(files_updated)} files with {total_insights} insights."

            return AnalysisResult(
                status="success",
                summary=summary,
                user_preferences_added=user_prefs,
                task_patterns_added=task_pats,
                workflow_trends_added=wf_trends,
            )

        except Exception as e:
            return AnalysisResult(
                status="error",
                summary=f"Error during analysis: {str(e)}"
            )
