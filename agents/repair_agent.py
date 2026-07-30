"""
Self-Healing RL Recommendation Agent — Repair Agent
Retrains the RL policy when performance degrades
"""

import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2a.protocol import *
from mcp.tools import MCPTools
from observability import traceable


class RepairAgent:
    def __init__(self):
        self.card = AGENT_CARDS["repair"]
        self.tools = MCPTools()
        self.repair_count = 0

    @traceable(run_type="chain", name="repair.execute")
    def execute_repair(self, message: A2AMessage) -> A2AMessage:
        print(f"\n  🔧 Repair Agent: Executing...")
        print_message(message, prefix="  📥 ")

        diagnosis = message.payload.get("diagnosis", {})
        action = diagnosis.get("recommended_action", "monitor")
        severity = diagnosis.get("severity", "low")
        root_cause = diagnosis.get("root_cause", "unknown")

        self.tools.log_agent_action(
            "repair",
            "repair_started",
            f"Executing {action} for {root_cause} (severity: {severity})",
            diagnosis,
        )

        if action == "retrain":
            result = self._retrain(diagnosis)
        elif action == "continue_monitoring":
            result = {
                "action": "continue_monitoring",
                "status": "completed",
                "reason": "Severity too low",
            }
        else:
            result = {"action": "no_action", "status": "completed"}

        self.repair_count += 1
        self.tools.log_agent_action(
            "repair",
            "repair_completed",
            f"Repair #{self.repair_count}: {result.get('action')} — {result.get('status')}",
            result,
        )

        verify_msg = create_message(
            from_agent="repair",
            to_agent="verification",
            action="verify_repair",
            payload={
                "repair_result": result,
                "diagnosis": diagnosis,
                "repair_count": self.repair_count,
            },
            parent_task_id=message.task_id,
        )

        print(f"\n  🔧 REPAIR RESULT:")
        print(f"     Action: {result.get('action')}")
        print(f"     Status: {result.get('status')}")
        if "new_model" in result:
            print(f"     New Version: {result['new_model'].get('version')}")
            print(f"     Mean Reward: {result['new_model'].get('mean_reward')}")
        print_message(verify_msg, prefix="  📤 ")
        return verify_msg

    def _retrain(self, diagnosis) -> dict:
        print(f"  🔧 Retraining RL agent on training subreddits...")
        retrain_result = self.tools.retrain_rl_agent(episodes=200)
        print(
            f"  🔧 New model deployed — version {retrain_result['version']}, reward {retrain_result['mean_reward']:.4f}"
        )
        return {
            "action": "retrain",
            "status": "completed",
            "new_model": retrain_result,
            "root_cause_addressed": diagnosis.get("root_cause"),
        }
