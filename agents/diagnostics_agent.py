"""
Self-Healing RL Recommendation Agent — Diagnostics Agent
Investigates root cause of RL performance degradation
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2a.protocol import *
from mcp.tools import MCPTools
from config import *


class DiagnosticsAgent:
    def __init__(self):
        self.card = AGENT_CARDS["diagnostics"]
        self.tools = MCPTools()

    def investigate(self, message: A2AMessage) -> A2AMessage:
        print(f"\n  🔬 Diagnostics: Investigating...")
        print_message(message, prefix="  📥 ")

        self.tools.log_agent_action(
            "diagnostics",
            "investigation_started",
            f"Investigating: {message.payload.get('drift_type', 'unknown')}",
            message.payload,
        )

        # Gather evidence
        print(f"  🔬 Step 1: Reward distribution...")
        reward_dist = self.tools.get_reward_distribution(window=100)

        print(f"  🔬 Step 2: Temporal patterns...")
        temporal = self.tools.get_temporal_metrics(bucket_size=10)

        print(f"  🔬 Step 3: Low-reward recommendations...")
        low_reward = self.tools.get_recommendation_logs(limit=30, max_reward=0.15)

        print(f"  🔬 Step 4: High-reward comparison...")
        high_reward = self.tools.get_recommendation_logs(limit=20, min_reward=0.3)

        diagnosis = self._analyze(
            message.payload, reward_dist, temporal, low_reward, high_reward
        )

        self.tools.log_agent_action(
            "diagnostics",
            "diagnosis_complete",
            f"Root cause: {diagnosis['root_cause']}. Action: {diagnosis['recommended_action']}",
            diagnosis,
        )

        repair_msg = create_message(
            from_agent="diagnostics",
            to_agent="repair",
            action="execute_repair",
            payload={"diagnosis": diagnosis, "original_alert": message.payload},
            parent_task_id=message.task_id,
        )

        print(f"\n  📋 DIAGNOSIS:")
        print(f"     Root Cause: {diagnosis['root_cause']}")
        print(f"     Severity: {diagnosis['severity']}")
        for e in diagnosis["evidence"]:
            print(f"     📌 {e}")
        print(f"     Action: {diagnosis['recommended_action']}")
        print_message(repair_msg, prefix="  📤 ")
        return repair_msg

    def _analyze(self, drift_payload, reward_dist, temporal, low_reward, high_reward):
        root_cause = "unknown"
        severity = "low"
        evidence = []
        action = "monitor"

        # Evidence 1: OOD detection
        if reward_dist.get("ood_detected", False):
            ood_pct = reward_dist["ood_percentage"]
            ood_subs = [
                s
                for s in reward_dist.get("unique_post_subreddits", [])
                if s not in TRAINING_SUBREDDITS
            ]
            evidence.append(
                f"OOD detected: {ood_pct:.1%} from non-training subreddits: {ood_subs[:5]}"
            )
            root_cause = "out_of_domain_content"
            severity = "critical"
            action = "retrain"

        # Evidence 2: Category collapse
        if reward_dist.get("category_collapse_detected", False):
            evidence.append(
                f"Category collapse: {reward_dist['dominant_category']} dominates at {reward_dist['dominant_category_pct']:.1%}"
            )
            if root_cause == "unknown":
                root_cause = "category_collapse"
                severity = "high"
                action = "retrain"

        # Evidence 3: Low reward percentage
        low_pct = reward_dist.get("low_reward_pct", 0)
        if low_pct > 0.5:
            evidence.append(
                f"Critical: {low_pct:.1%} of recommendations have reward < 0.2"
            )
            severity = "critical"
            action = "retrain"
        elif low_pct > 0.3:
            evidence.append(f"Elevated low-reward rate: {low_pct:.1%}")
            if root_cause == "unknown":
                root_cause = "reward_degradation"
                severity = "medium"
                action = "retrain"

        # Evidence 4: Temporal analysis
        if temporal and len(temporal) >= 3:
            recent = temporal[-3:]
            early = temporal[:3]
            recent_reward = np.mean([b["mean_reward"] for b in recent])
            early_reward = np.mean([b["mean_reward"] for b in early])
            if early_reward - recent_reward > 0.15:
                evidence.append(
                    f"Temporal decline: reward dropped from {early_reward:.3f} to {recent_reward:.3f}"
                )

        # Evidence 5: Low-reward post analysis
        if low_reward:
            non_training = sum(
                1 for p in low_reward if p["post_subreddit"] not in TRAINING_SUBREDDITS
            )
            if non_training > len(low_reward) * 0.5:
                evidence.append(
                    f"{non_training}/{len(low_reward)} low-reward posts from unknown subreddits"
                )
                root_cause = "out_of_domain_content"
                severity = "critical"
                action = "retrain"

        if not evidence:
            evidence.append("Insufficient evidence")
            action = "continue_monitoring"

        return {
            "root_cause": root_cause,
            "severity": severity,
            "evidence": evidence,
            "recommended_action": action,
            "reward_distribution": reward_dist,
        }
