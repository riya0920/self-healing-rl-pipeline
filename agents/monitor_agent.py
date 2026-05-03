"""
Self-Healing RL Recommendation Agent — Monitor Agent
Watches RL reward curves and detects performance degradation
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2a.protocol import *
from mcp.tools import MCPTools
from config import *


class MonitorAgent:
    def __init__(self):
        self.card = AGENT_CARDS["monitor"]
        self.tools = MCPTools()
        self.drift_detected = False

    def check_for_drift(self, window: int = 50) -> Optional[A2AMessage]:
        metrics = self.tools.get_recent_metrics(window=window)

        if metrics.get("error") or metrics.get("count", 0) < 10:
            print(
                f"  📊 Monitor: Not enough data yet ({metrics.get('count', 0)} recommendations)"
            )
            return None

        baseline = self.tools.get_baseline_stats()
        reward_dist = self.tools.get_reward_distribution(window=window)

        drift_signals = []
        drift_type = None

        # Check 1: Reward drop
        reward_drift = metrics["reward_drift"]
        if reward_drift < -REWARD_DROP_THRESHOLD:
            drift_signals.append(
                f"Reward dropped by {abs(reward_drift):.4f} (threshold: {REWARD_DROP_THRESHOLD})"
            )
            drift_type = DriftType.CONFIDENCE_DROP

        # Check 2: Relevance rate drop
        if (
            metrics["relevance_rate"] is not None
            and metrics["relevance_rate"] < ENGAGEMENT_THRESHOLD
        ):
            drift_signals.append(
                f"Relevance rate: {metrics['relevance_rate']:.1%} (threshold: {ENGAGEMENT_THRESHOLD:.1%})"
            )
            drift_type = DriftType.ACCURACY_DROP

        # Check 3: High percentage of low-reward recommendations
        low_reward_pct = reward_dist.get("low_reward_pct", 0)
        if low_reward_pct > LOW_REWARD_PCT_THRESHOLD:
            drift_signals.append(
                f"Low reward recommendations: {low_reward_pct:.1%} (threshold: {LOW_REWARD_PCT_THRESHOLD:.1%})"
            )
            if drift_type is None:
                drift_type = DriftType.OUT_OF_DOMAIN

        # Check 4: OOD detection
        if reward_dist.get("ood_detected", False):
            drift_signals.append(
                f"Out-of-domain posts: {reward_dist['ood_percentage']:.1%} from non-training subreddits"
            )
            drift_type = DriftType.OUT_OF_DOMAIN

        # Check 5: Category collapse
        if reward_dist.get("category_collapse_detected", False):
            drift_signals.append(
                f"Category collapse: {reward_dist['dominant_category']} has {reward_dist['dominant_category_pct']:.1%}"
            )
            if drift_type is None:
                drift_type = DriftType.DISTRIBUTION_SHIFT

        if drift_signals:
            self.drift_detected = True
            self.tools.log_agent_action(
                "monitor",
                "drift_detected",
                f"Detected {len(drift_signals)} drift signal(s): {'; '.join(drift_signals)}",
                {"metrics": metrics, "distribution": reward_dist},
            )

            message = create_message(
                from_agent="monitor",
                to_agent="diagnostics",
                action="investigate_drift",
                payload={
                    "drift_signals": drift_signals,
                    "drift_type": drift_type.value if drift_type else "unknown",
                    "current_metrics": metrics,
                    "baseline": {"mean_reward": baseline["mean_reward"]},
                    "window_size": window,
                },
            )

            print(f"\n  🚨 DRIFT DETECTED!")
            for s in drift_signals:
                print(f"     ⚠️  {s}")
            print_message(message, prefix="  ")
            return message
        else:
            print(
                f"  ✅ Monitor: All metrics normal (reward: {metrics['mean_reward']:.4f})"
            )
            return None

    def handle_verification_result(self, message: A2AMessage):
        if message.status == TaskStatus.COMPLETED:
            print(f"  ✅ Monitor: Fix verified. Drift resolved.")
            self.drift_detected = False
            self.tools.log_agent_action(
                "monitor",
                "drift_resolved",
                "Drift resolved after repair",
                message.payload,
            )
            return None
        elif message.status == TaskStatus.SENT_BACK:
            print(f"  ↩️  Monitor: Fix failed. Re-triggering diagnostics.")
            self.tools.log_agent_action(
                "monitor",
                "fix_failed",
                f"Re-investigating: {message.feedback}",
                message.payload,
            )
            return self.check_for_drift()
        return None
