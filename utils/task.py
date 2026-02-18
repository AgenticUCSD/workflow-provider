from enum import Enum
from pydantic import BaseModel
from typing import Any, Dict, Optional

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
    constrainsts: Optional[Dict[str, Any]] = None
    success_criteria: str
    expected_output: Dict[str, Any]
    deadline: Optional[str] = None  # ISO 8601 format

#TODO: The definition of step and workflow needs refinement. Currently it's a sample structure
class Step(BaseModel):
    claude_skill_name: str
    api: str
    access: str #can be changed to str | AccessLevel

class Workflow(BaseModel):
    workflow_id: str
    name: str
    description: str
    steps: list[Step]  # Define the structure of steps as needed

    
class Task(BaseModel):
    task_id: str
    task_type: TaskTypes
    objective: Objective 
    candidate_workflow: list[Workflow] #possible workflows
    workflow: Workflow
    status: Status
    metadata: Optional[Dict[str, Any]] = None