"""
Self-Healing RL Recommendation Agent — MCP Tools
Tools agents use to access RL metrics, recommendation logs, and retrain the model
"""

import os
import sys
import json
import sqlite3
import pickle
import time
import random
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class MCPTools:
    """MCP Server exposing tools for agent access to RL pipeline data"""

    # ---- MONITORING TOOLS ----

    @staticmethod
    def get_recent_metrics(window: int = 50) -> Dict[str, Any]:
        """Get aggregated metrics for the last N recommendations"""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT reward, engagement_score, recommended_category, is_relevant, timestamp FROM recommendations ORDER BY id DESC LIMIT ?",
            (window,),
        ).fetchall()
        conn.close()

        if not rows:
            return {"error": "No recommendations available", "count": 0}

        rewards = [r[0] for r in rows]
        engagements = [r[1] for r in rows]
        categories = [r[2] for r in rows]
        relevant = [r[3] for r in rows if r[3] is not None]
        timestamps = [r[4] for r in rows]

        mean_reward = float(np.mean(rewards))
        baseline_reward = 0.5  # Expected reward when agent is performing well

        return {
            "window_size": len(rows),
            "mean_reward": round(mean_reward, 4),
            "std_reward": round(float(np.std(rewards)), 4),
            "min_reward": round(float(np.min(rewards)), 4),
            "max_reward": round(float(np.max(rewards)), 4),
            "mean_engagement": round(float(np.mean(engagements)), 4),
            "relevance_rate": round(float(np.mean(relevant)), 4) if relevant else None,
            "count": len(rows),
            "category_distribution": {
                cat: categories.count(cat) for cat in set(categories)
            },
            "time_range": {"oldest": timestamps[-1], "newest": timestamps[0]},
            "baseline_mean_reward": baseline_reward,
            "reward_drift": round(mean_reward - baseline_reward, 4),
        }

    @staticmethod
    def get_baseline_stats() -> Dict[str, Any]:
        """Get baseline performance statistics"""
        return {
            "mean_reward": 0.5,
            "mean_engagement": 0.5,
            "categories": TRAINING_SUBREDDITS,
            "training_subreddits": TRAINING_SUBREDDITS,
            "drift_subreddits": DRIFT_SUBREDDITS,
            "heavy_drift_subreddits": HEAVY_DRIFT_SUBREDDITS,
        }

    # ---- DIAGNOSTICS TOOLS ----

    @staticmethod
    def get_recommendation_logs(
        limit: int = 100,
        min_reward: float = None,
        max_reward: float = None,
        category: str = None,
    ) -> List[Dict]:
        """Query recommendation logs with optional filters"""
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT id, timestamp, post_title, post_subreddit, recommended_category, reward, engagement_score, is_relevant, q_values FROM recommendations WHERE 1=1"
        params = []

        if min_reward is not None:
            query += " AND reward >= ?"
            params.append(min_reward)
        if max_reward is not None:
            query += " AND reward <= ?"
            params.append(max_reward)
        if category is not None:
            query += " AND recommended_category = ?"
            params.append(category)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "post_title": r[2][:100],
                "post_subreddit": r[3],
                "recommended": r[4],
                "reward": r[5],
                "engagement": r[6],
                "relevant": r[7],
                "q_values": json.loads(r[8]) if r[8] else {},
            }
            for r in rows
        ]

    @staticmethod
    def get_reward_distribution(window: int = 100) -> Dict[str, Any]:
        """Analyze reward distribution to detect drift patterns"""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT reward, recommended_category, is_relevant, post_subreddit FROM recommendations ORDER BY id DESC LIMIT ?",
            (window,),
        ).fetchall()
        conn.close()

        if not rows:
            return {"error": "No data"}

        rewards = [r[0] for r in rows]
        rec_cats = [r[1] for r in rows]
        post_subs = [r[3] for r in rows]

        # Reward buckets
        buckets = {"0.0-0.1": 0, "0.1-0.2": 0, "0.2-0.3": 0, "0.3-0.5": 0, "0.5-1.0": 0}
        for r in rewards:
            if r < 0.1:
                buckets["0.0-0.1"] += 1
            elif r < 0.2:
                buckets["0.1-0.2"] += 1
            elif r < 0.3:
                buckets["0.2-0.3"] += 1
            elif r < 0.5:
                buckets["0.3-0.5"] += 1
            else:
                buckets["0.5-1.0"] += 1

        # Check for category collapse
        rec_counts = {cat: rec_cats.count(cat) for cat in set(rec_cats)}
        max_cat_pct = max(rec_counts.values()) / len(rec_cats) if rec_cats else 0

        # Check for OOD: are post subreddits outside training set?
        ood_count = sum(1 for s in post_subs if s not in TRAINING_SUBREDDITS)
        ood_pct = ood_count / len(post_subs) if post_subs else 0

        low_reward_pct = (buckets["0.0-0.1"] + buckets["0.1-0.2"]) / len(rows)

        return {
            "reward_buckets": buckets,
            "recommendation_distribution": rec_counts,
            "category_collapse_detected": max_cat_pct > 0.6,
            "dominant_category": (
                max(rec_counts, key=rec_counts.get) if rec_counts else None
            ),
            "dominant_category_pct": round(max_cat_pct, 4),
            "low_reward_pct": round(low_reward_pct, 4),
            "ood_percentage": round(ood_pct, 4),
            "ood_detected": ood_pct > 0.3,
            "unique_post_subreddits": list(set(post_subs)),
            "training_subreddits": TRAINING_SUBREDDITS,
        }

    @staticmethod
    def get_temporal_metrics(bucket_size: int = 10) -> List[Dict]:
        """Get metrics over time in buckets"""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT id, reward, recommended_category, is_relevant FROM recommendations ORDER BY id ASC"
        ).fetchall()
        conn.close()

        if not rows:
            return []

        buckets = []
        for i in range(0, len(rows), bucket_size):
            chunk = rows[i : i + bucket_size]
            rewards = [r[1] for r in chunk]
            relevant = [r[3] for r in chunk if r[3] is not None]
            buckets.append(
                {
                    "bucket_start_id": chunk[0][0],
                    "bucket_end_id": chunk[-1][0],
                    "count": len(chunk),
                    "mean_reward": round(float(np.mean(rewards)), 4),
                    "relevance_rate": (
                        round(float(np.mean(relevant)), 4) if relevant else None
                    ),
                }
            )

        return buckets

    # ---- REPAIR TOOLS ----

    @staticmethod
    def retrain_rl_agent(episodes: int = 200) -> Dict[str, Any]:
        """Retrain the RL agent on training subreddit data"""
        from rl_agent import DQNAgent, PostFeatureEncoder
        from scraper import generate_synthetic_reddit_data

        agent = DQNAgent()
        encoder = PostFeatureEncoder()

        # Load existing model if available
        model_path = os.path.join(PIPELINE_DIR, "rl_model.pt")
        if os.path.exists(model_path):
            agent.load(model_path)

        # Generate fresh training data from training subreddits
        posts = generate_synthetic_reddit_data(TRAINING_SUBREDDITS, n_per_sub=100)

        total_reward = 0
        for ep in range(episodes):
            post = random.choice(posts)
            state = encoder.encode_post(post)
            action = agent.select_action(state, training=True)

            recommended_cat = TRAINING_SUBREDDITS[action]
            engagement = post["engagement_score"]
            reward = (
                engagement if post["subreddit"] == recommended_cat else engagement * 0.1
            )

            next_post = random.choice(posts)
            next_state = encoder.encode_post(next_post)

            agent.store_experience(state, action, reward, next_state, False)
            agent.episode_rewards.append(reward)
            total_reward += reward

            agent.train_step()

        # Save with new version
        version = f"v{int(time.time())}"
        agent.version = version
        agent.save(model_path)

        mean_reward = total_reward / episodes

        # Register in model registry
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE model_registry SET status = 'retired' WHERE status = 'active'"
        )
        conn.execute(
            "INSERT INTO model_registry (timestamp, version, mean_reward, training_steps, episodes, status, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                version,
                mean_reward,
                agent.training_steps,
                episodes,
                "active",
                "Retrained by Repair Agent",
            ),
        )
        conn.commit()
        conn.close()

        return {
            "version": version,
            "mean_reward": round(mean_reward, 4),
            "training_steps": agent.training_steps,
            "episodes": episodes,
            "status": "deployed",
        }

    @staticmethod
    def run_validation(n_samples: int = 50) -> Dict[str, Any]:
        """Validate current model on training data"""
        from rl_agent import DQNAgent, PostFeatureEncoder
        from scraper import generate_synthetic_reddit_data

        agent = DQNAgent()
        encoder = PostFeatureEncoder()

        model_path = os.path.join(PIPELINE_DIR, "rl_model.pt")
        if os.path.exists(model_path):
            agent.load(model_path)

        posts = generate_synthetic_reddit_data(
            TRAINING_SUBREDDITS, n_per_sub=n_samples // len(TRAINING_SUBREDDITS) + 1
        )
        posts = posts[:n_samples]

        rewards = []
        relevant = 0
        for post in posts:
            state = encoder.encode_post(post)
            action = agent.select_action(state, training=False)
            recommended = TRAINING_SUBREDDITS[action]
            engagement = post["engagement_score"]
            reward = (
                engagement if post["subreddit"] == recommended else engagement * 0.1
            )
            rewards.append(reward)
            if post["subreddit"] == recommended:
                relevant += 1

        mean_reward = float(np.mean(rewards))
        relevance_rate = relevant / len(posts)

        return {
            "mean_reward": round(mean_reward, 4),
            "relevance_rate": round(relevance_rate, 4),
            "samples_tested": len(posts),
            "baseline_reward": 0.5,
            "passed": mean_reward >= 0.3 and relevance_rate >= 0.2,
        }

    # ---- LOGGING TOOLS ----

    @staticmethod
    def log_agent_action(
        agent_name: str,
        action_type: str,
        description: str,
        details: dict = None,
        status: str = "completed",
    ):
        """Log an agent action to the database"""
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO agent_actions (timestamp, agent_name, action_type, description, details, status) VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                agent_name,
                action_type,
                description,
                json.dumps(details) if details else None,
                status,
            ),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get_agent_actions_summary(limit: int = 50) -> List[str]:
        """Get formatted summary of all agent actions"""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT timestamp, agent_name, action_type, description FROM agent_actions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [f"[{r[0][:19]}] {r[1]:15s} | {r[2]:20s} | {r[3]}" for r in rows]
