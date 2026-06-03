from enum import Enum
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class TaskTypes(str, Enum):
    NO_TASK = "no_task"
    DRAFT = "draft"
    REVIEW = "review"
    SCHEDULE = "schedule"
    RESPOND = "respond"
    EXECUTE = "execute"
    DECISION = "decision"
    DELEGATE = "delegate"

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
    steps: List[str] # Define the structure of steps as needed; this can be a WORKFLOW as well
    
    def to_string(self) -> str:
        """Convert workflow to a string representation for embedding."""
        steps_str = "\n".join(f"- {step}" for step in self.steps)
        return f"Workflow: {self.name}\nDescription: {self.description}\nSteps:\n{steps_str}"
    
class Task(BaseModel):
    task_id: str
    task_type: TaskTypes
    priority: Optional[str] = None  # "low", "normal", "high", "urgent"
    objective: Objective
    candidate_workflows: Optional[List[Workflow]] = None
    workflow: Optional[Workflow] = None
    status: Status
    metadata: Optional[Dict[str, Any]] = None
    
    def to_string(self) -> str:
        """Convert task to a string representation for embedding."""
        obj = self.objective
        result = f"Task Type: {self.task_type.value}\n"
        result += f"Objective: {obj.name}\n"
        result += f"Description: {obj.description}\n"
        result += f"Success Criteria: {obj.success_criteria}\n"
        if obj.deadline:
            result += f"Deadline: {obj.deadline}\n"
        return result
