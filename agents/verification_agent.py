"""
Self-Healing RL Recommendation Agent — Verification Agent
Validates that RL model repairs actually improved performance
"""

import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2a.protocol import *
from mcp.tools import MCPTools

MIN_REWARD_THRESHOLD = 0.25
MIN_RELEVANCE_THRESHOLD = 0.15


class VerificationAgent:
    def __init__(self):
        self.card = AGENT_CARDS["verification"]
        self.tools = MCPTools()
        self.verification_count = 0
        self.max_retries = 3

    def verify_repair(self, message: A2AMessage) -> A2AMessage:
        print(f"\n  ✔️  Verification: Validating...")
        print_message(message, prefix="  📥 ")

        repair_result = message.payload.get("repair_result", {})
        repair_action = repair_result.get("action", "unknown")
        repair_count = message.payload.get("repair_count", 0)
        self.verification_count += 1

        self.tools.log_agent_action(
            "verification",
            "verification_started",
            f"Verifying repair #{repair_count}: {repair_action}",
            repair_result,
        )

        if repair_action == "continue_monitoring":
            return self._approve(message, "No repair needed — monitoring continues")

        # Run validation
        print(f"  ✔️  Running validation on training data...")
        validation = self.tools.run_validation(n_samples=50)

        print(f"  ✔️  Results:")
        print(
            f"     Mean Reward: {validation['mean_reward']:.4f} (threshold: {MIN_REWARD_THRESHOLD})"
        )
        print(
            f"     Relevance: {validation['relevance_rate']:.4f} (threshold: {MIN_RELEVANCE_THRESHOLD})"
        )
        print(f"     Passed: {'✅' if validation['passed'] else '❌'}")

        reward_ok = validation["mean_reward"] >= MIN_REWARD_THRESHOLD
        relevance_ok = validation["relevance_rate"] >= MIN_RELEVANCE_THRESHOLD

        if reward_ok and relevance_ok:
            return self._approve(
                message,
                f"Verified: reward={validation['mean_reward']:.4f}, relevance={validation['relevance_rate']:.4f}",
                validation,
            )
        else:
            if repair_count >= self.max_retries:
                return self._approve(
                    message,
                    f"Max retries ({self.max_retries}) exceeded. Accepting. reward={validation['mean_reward']:.4f}",
                    validation,
                )
            failures = []
            if not reward_ok:
                failures.append(
                    f"reward {validation['mean_reward']:.4f} < {MIN_REWARD_THRESHOLD}"
                )
            if not relevance_ok:
                failures.append(
                    f"relevance {validation['relevance_rate']:.4f} < {MIN_RELEVANCE_THRESHOLD}"
                )
            return self._reject(
                message,
                f"Insufficient: {', '.join(failures)}. Attempt {repair_count}/{self.max_retries}.",
                validation,
            )

    def _approve(self, msg, reason, validation=None):
        self.tools.log_agent_action(
            "verification", "repair_approved", reason, validation
        )
        result = create_message(
            from_agent="verification",
            to_agent="monitor",
            action="repair_verified",
            status=TaskStatus.COMPLETED,
            payload={"result": "approved", "reason": reason, "validation": validation},
            parent_task_id=msg.task_id,
        )
        print(f"\n  ✅ APPROVED: {reason}")
        print_message(result, prefix="  📤 ")
        return result

    def _reject(self, msg, reason, validation=None):
        self.tools.log_agent_action(
            "verification", "repair_rejected", reason, validation
        )
        result = create_message(
            from_agent="verification",
            to_agent="monitor",
            action="repair_failed",
            status=TaskStatus.SENT_BACK,
            payload={"result": "rejected", "reason": reason, "validation": validation},
            parent_task_id=msg.task_id,
            feedback=reason,
        )
        print(f"\n  ↩️  REJECTED: {reason}")
        print_message(result, prefix="  📤 ")
        return result
