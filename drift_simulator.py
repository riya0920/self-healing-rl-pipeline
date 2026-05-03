"""
Self-Healing RL Recommendation Agent — Drift Simulator
Feeds the RL recommendation agent live/synthetic Reddit posts in phases:
  Phase 1: Training subreddits (agent performs well)
  Phase 2: Drift subreddits (agent has never seen these topics)
  Phase 3: Heavy drift (completely different content domains)
  Phase 4: Mixed chaos (random mix of everything)
"""

import os
import sys
import json
import time
import random
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from scraper import RedditScraper, generate_synthetic_reddit_data

API_URL = f"http://localhost:{SERVER_PORT}"

PHASES = [
    {
        "name": "Phase 1: In-Distribution (Training Subreddits)",
        "subreddits": TRAINING_SUBREDDITS,
        "n_per_sub": 15,
        "repeats": 2,
        "description": "Agent sees data it was trained on. Should perform well.",
    },
    {
        "name": "Phase 2: Mild Drift (New Topics)",
        "subreddits": DRIFT_SUBREDDITS,
        "n_per_sub": 15,
        "repeats": 2,
        "description": "Agent sees cooking, fitness, legal, medical posts. Has never seen these.",
    },
    {
        "name": "Phase 3: Heavy Drift (Completely Different)",
        "subreddits": HEAVY_DRIFT_SUBREDDITS,
        "n_per_sub": 15,
        "repeats": 2,
        "description": "Agent sees philosophy, art, gardening, astronomy. Total domain mismatch.",
    },
    {
        "name": "Phase 4: Mixed Chaos",
        "subreddits": TRAINING_SUBREDDITS + DRIFT_SUBREDDITS + HEAVY_DRIFT_SUBREDDITS,
        "n_per_sub": 5,
        "repeats": 1,
        "description": "Random mix of everything. Agent is completely confused.",
    },
]


def send_recommendation(post: dict) -> dict:
    """Send a post to the server and get a recommendation"""
    payload = {
        "post_title": post["title"],
        "post_subreddit": post["subreddit"],
        "post_score": post.get("score", 0),
        "post_comments": post.get("num_comments", 0),
        "post_upvote_ratio": post.get("upvote_ratio", 0.5),
        "post_id": post.get("id", ""),
    }
    try:
        resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=5)
        return resp.json()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def run_drift_simulation(delay: float = 0.3, use_live: bool = False):
    """Run the full drift simulation"""

    print("=" * 70)
    print("🌊 RL RECOMMENDATION DRIFT SIMULATOR")
    print("=" * 70)
    print(f"Target: {API_URL}")
    print(f"Phases: {len(PHASES)}")
    print(f"Data source: {'Live Reddit API' if use_live else 'Synthetic Reddit data'}")
    print()

    # Initialize scraper if using live data
    scraper = RedditScraper() if use_live else None

    total_sent = 0
    phase_stats = []

    for phase_idx, phase in enumerate(PHASES):
        print(f"\n{'='*70}")
        print(f"🔄 {phase['name']}")
        print(f"   {phase['description']}")
        print(f"{'='*70}")

        phase_rewards = []
        phase_relevant = []

        for repeat in range(phase["repeats"]):
            # Get posts
            if use_live and scraper:
                posts = scraper.fetch_multi_subreddit(
                    phase["subreddits"], limit_per_sub=phase["n_per_sub"]
                )
            else:
                posts = generate_synthetic_reddit_data(
                    phase["subreddits"], n_per_sub=phase["n_per_sub"]
                )

            random.shuffle(posts)

            for post in posts:
                result = send_recommendation(post)
                if result:
                    reward = result["reward"]
                    recommended = result["recommended_category"]
                    actual = post["subreddit"]
                    relevant = "✅" if actual == recommended else "❌"
                    is_rel = 1 if actual == recommended else 0

                    phase_rewards.append(reward)
                    phase_relevant.append(is_rel)
                    total_sent += 1

                    print(
                        f"  {relevant} [{reward:.3f}] r/{actual:15s} → recommended: {recommended:12s} | {post['title'][:50]}..."
                    )

                time.sleep(delay)

        # Phase summary
        if phase_rewards:
            mean_reward = sum(phase_rewards) / len(phase_rewards)
            relevance = sum(phase_relevant) / len(phase_relevant)
            phase_stats.append(
                {
                    "phase": phase["name"],
                    "mean_reward": mean_reward,
                    "relevance_rate": relevance,
                    "count": len(phase_rewards),
                }
            )
            print(f"\n  📊 Phase Summary:")
            print(f"     Mean Reward: {mean_reward:.4f}")
            print(f"     Relevance Rate: {relevance:.1%}")
            print(f"     Recommendations: {len(phase_rewards)}")

        if phase_idx < len(PHASES) - 1:
            print(f"\n  ⏸️  Waiting 3 seconds before next phase...")
            time.sleep(3)

    # Final summary
    print(f"\n{'='*70}")
    print(f"🏁 DRIFT SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"   Total recommendations: {total_sent}")
    print(f"\n   Phase-by-phase performance:")
    for stat in phase_stats:
        emoji = (
            "✅"
            if stat["mean_reward"] > 0.3
            else "⚠️" if stat["mean_reward"] > 0.15 else "❌"
        )
        print(
            f"   {emoji} {stat['phase'][:40]:40s} | reward: {stat['mean_reward']:.4f} | relevance: {stat['relevance_rate']:.1%}"
        )

    # Fetch final metrics from server
    try:
        metrics = requests.get(f"{API_URL}/metrics/recent?window=100").json()
        print(f"\n   📊 Server Metrics (last 100):")
        print(f"      Mean Reward: {metrics.get('mean_reward', 'N/A')}")
        print(f"      Mean Engagement: {metrics.get('mean_engagement', 'N/A')}")
        print(f"      Reward Drift: {metrics.get('reward_drift', 'N/A')}")
    except:
        pass


if __name__ == "__main__":
    import sys as _sys

    delay = float(_sys.argv[1]) if len(_sys.argv) > 1 else 0.3
    use_live = "--live" in _sys.argv
    run_drift_simulation(delay=delay, use_live=use_live)
