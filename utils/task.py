from enum import Enum
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class TaskTypes(str, Enum):
    NO_TASK = "notask"
    SCHEDULE = "schedule"
    ACTION_REQUIRED = "action_required"
    REPLY_NEEDED = "reply_needed"
    REVIEW_FEEDBACK = "review_feedback"
    FORWARD_DELEGATE = "forward_delegate"

class Status(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Objective(BaseModel):
    objective_id: str
    name: str
    description: str
    inputs: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    success_criteria: str
    expected_output: Dict[str, Any]
    deadline: Optional[str] = None  # ISO 8601 format

class Workflow(BaseModel):
    workflow_id: str
    name: str
    description: str
    steps: List[str]
    
class Task(BaseModel):
    task_id: str
    task_type: TaskTypes
    objective: Objective
    workflow: Optional[Workflow] = None
    status: Status
    metadata: Optional[Dict[str, Any]] = None