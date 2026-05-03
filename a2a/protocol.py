"""
Self-Healing AI Pipeline - A2A Protocol Implementation
Agent-to-Agent communication using JSON-over-HTTP
Following the A2A standard: discovery via agent cards, task lifecycle management
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SENT_BACK = "sent_back"  # Agent rejected, sending back to previous


class DriftType(Enum):
    CONFIDENCE_DROP = "confidence_drop"
    ACCURACY_DROP = "accuracy_drop"
    DISTRIBUTION_SHIFT = "distribution_shift"
    OUT_OF_DOMAIN = "out_of_domain"
    UNKNOWN = "unknown"


@dataclass
class AgentCard:
    """A2A Agent Card - describes what an agent can do"""

    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    accepts_from: List[str]  # which agents can send tasks to this one
    sends_to: List[str]  # which agents this one can delegate to
    endpoint: str
    version: str = "1.0"


@dataclass
class A2AMessage:
    """Standard A2A message format for agent communication"""

    task_id: str
    from_agent: str
    to_agent: str
    action: str
    status: TaskStatus
    payload: Dict[str, Any]
    timestamp: str = ""
    parent_task_id: Optional[str] = None
    feedback: Optional[str] = None  # Used when sending back with corrections

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d):
        d["status"] = TaskStatus(d["status"])
        return cls(**d)


# ---- Agent Card Registry ----
AGENT_CARDS = {
    "monitor": AgentCard(
        agent_id="monitor",
        name="Monitor Agent",
        description="Watches real-time prediction metrics and detects data drift by comparing against baseline performance thresholds",
        capabilities=["drift_detection", "metric_comparison", "threshold_alerting"],
        accepts_from=["orchestrator", "verification"],
        sends_to=["diagnostics"],
        endpoint="http://localhost:8001",
    ),
    "diagnostics": AgentCard(
        agent_id="diagnostics",
        name="Diagnostics Agent",
        description="Investigates root cause of detected drift by analyzing prediction logs, confidence distributions, and class patterns via MCP tools",
        capabilities=[
            "root_cause_analysis",
            "log_analysis",
            "pattern_detection",
            "distribution_comparison",
        ],
        accepts_from=["monitor"],
        sends_to=["repair"],
        endpoint="http://localhost:8002",
    ),
    "repair": AgentCard(
        agent_id="repair",
        name="Repair Agent",
        description="Takes corrective action based on diagnosis: retrains model on clean data, adjusts thresholds, or rolls back to previous model version",
        capabilities=[
            "model_retraining",
            "model_swap",
            "threshold_adjustment",
            "rollback",
        ],
        accepts_from=["diagnostics"],
        sends_to=["verification"],
        endpoint="http://localhost:8003",
    ),
    "verification": AgentCard(
        agent_id="verification",
        name="Verification Agent",
        description="Validates that repairs actually fixed the problem by running test data through the new model and comparing before/after metrics",
        capabilities=["model_validation", "metric_comparison", "approval", "rejection"],
        accepts_from=["repair"],
        sends_to=["monitor"],  # Can send back to monitor to restart cycle if fix failed
        endpoint="http://localhost:8004",
    ),
}


def create_task_id():
    return f"task-{uuid.uuid4().hex[:8]}"


def create_message(
    from_agent: str,
    to_agent: str,
    action: str,
    payload: dict,
    status: TaskStatus = TaskStatus.PENDING,
    task_id: str = None,
    parent_task_id: str = None,
    feedback: str = None,
) -> A2AMessage:
    return A2AMessage(
        task_id=task_id or create_task_id(),
        from_agent=from_agent,
        to_agent=to_agent,
        action=action,
        status=status,
        payload=payload,
        parent_task_id=parent_task_id,
        feedback=feedback,
    )


def print_message(msg: A2AMessage, prefix=""):
    """Pretty print an A2A message"""
    status_icons = {
        TaskStatus.PENDING: "⏳",
        TaskStatus.IN_PROGRESS: "🔄",
        TaskStatus.COMPLETED: "✅",
        TaskStatus.FAILED: "❌",
        TaskStatus.SENT_BACK: "↩️",
    }
    icon = status_icons.get(msg.status, "❓")
    print(
        f"{prefix}{icon} [{msg.task_id}] {msg.from_agent} → {msg.to_agent} | {msg.action} | {msg.status.value}"
    )
    if msg.feedback:
        print(f"{prefix}   💬 Feedback: {msg.feedback}")
