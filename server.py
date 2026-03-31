"""
Self-Healing RL Recommendation Agent — FastAPI Server
Serves RL-powered content recommendations and logs all interactions to SQLite
"""
import os
import sys
import json
import time
import sqlite3
import numpy as np
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from rl_agent import DQNAgent, PostFeatureEncoder
from scraper import RedditScraper, generate_synthetic_reddit_data

# ---- Initialize Components ----
agent = DQNAgent()
encoder = PostFeatureEncoder()
MODEL_VERSION = agent.version

# Try to load existing model
MODEL_PATH = os.path.join(PIPELINE_DIR, "rl_model.pt")
if os.path.exists(MODEL_PATH):
    agent.load(MODEL_PATH)

# ---- Database Setup ----
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        post_id TEXT,
        post_title TEXT NOT NULL,
        post_subreddit TEXT NOT NULL,
        recommended_category TEXT NOT NULL,
        recommended_idx INTEGER NOT NULL,
        engagement_score REAL NOT NULL,
        reward REAL NOT NULL,
        q_values TEXT NOT NULL,
        epsilon REAL NOT NULL,
        model_version TEXT NOT NULL,
        is_relevant INTEGER DEFAULT NULL
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        window_size INTEGER NOT NULL,
        mean_reward REAL NOT NULL,
        std_reward REAL NOT NULL,
        mean_engagement REAL NOT NULL,
        recommendation_count INTEGER NOT NULL,
        category_distribution TEXT NOT NULL,
        model_version TEXT NOT NULL,
        drift_detected INTEGER DEFAULT 0
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS agent_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        agent_name TEXT NOT NULL,
        action_type TEXT NOT NULL,
        description TEXT NOT NULL,
        details TEXT DEFAULT NULL,
        status TEXT NOT NULL
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS model_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        version TEXT NOT NULL,
        mean_reward REAL NOT NULL,
        training_steps INTEGER NOT NULL,
        episodes INTEGER NOT NULL,
        status TEXT NOT NULL,
        notes TEXT DEFAULT NULL
    )''')
    
    # Register initial model
    existing = c.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0]
    if existing == 0:
        c.execute('''INSERT INTO model_registry (timestamp, version, mean_reward, training_steps, episodes, status, notes)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), MODEL_VERSION, 0.0, 0, 0, 'active', 'Initial RL model'))
    
    conn.commit()
    conn.close()

init_db()

# ---- FastAPI App ----
app = FastAPI(
    title="Self-Healing RL Recommendation Agent",
    description="RL-powered content recommendation engine with autonomous drift detection and self-healing",
    version="1.0"
)

class RecommendRequest(BaseModel):
    post_title: str
    post_subreddit: str
    post_score: int = 0
    post_comments: int = 0
    post_upvote_ratio: float = 0.5
    post_id: str = ""

class RecommendResponse(BaseModel):
    recommended_category: str
    recommended_idx: int
    confidence: float
    q_values: Dict[str, float]
    reward: float
    engagement_score: float
    model_version: str
    timestamp: str

class TrainRequest(BaseModel):
    episodes: int = 100
    subreddits: List[str] = TRAINING_SUBREDDITS

START_TIME = time.time()

@app.get("/health")
def health():
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
    conn.close()
    stats = agent.get_stats()
    return {
        "status": "healthy",
        "model_version": agent.version,
        "total_recommendations": count,
        "epsilon": stats["epsilon"],
        "training_steps": stats["training_steps"],
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """Get RL-powered recommendation for a post"""
    post = {
        "id": request.post_id,
        "title": request.post_title,
        "subreddit": request.post_subreddit,
        "score": request.post_score,
        "num_comments": request.post_comments,
        "upvote_ratio": request.post_upvote_ratio,
        "engagement_score": encoder._calc_engagement(request.post_score, request.post_comments, request.post_upvote_ratio)
            if hasattr(encoder, '_calc_engagement') else 0.5
    }
    
    # Encode state and get action
    state = encoder.encode_post(post)
    action = agent.select_action(state, training=False)
    q_values = agent.get_q_values(state)
    
    recommended_cat = TRAINING_SUBREDDITS[action]
    engagement = post.get("engagement_score", 0.0)
    
    # Calculate reward: high if recommendation matches actual subreddit category, engagement-weighted
    is_match = (request.post_subreddit == recommended_cat or 
                request.post_subreddit in TRAINING_SUBREDDITS[:action+1])
    reward = engagement if request.post_subreddit == recommended_cat else engagement * 0.2
    
    # Store experience for training
    next_state = state  # Simplified: next state is current state
    agent.store_experience(state, action, reward, next_state, False)
    agent.episode_rewards.append(reward)
    
    timestamp = datetime.now().isoformat()
    
    # Log to SQLite
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        '''INSERT INTO recommendations (timestamp, post_id, post_title, post_subreddit, 
           recommended_category, recommended_idx, engagement_score, reward, q_values, 
           epsilon, model_version, is_relevant)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (timestamp, request.post_id, request.post_title[:200], request.post_subreddit,
         recommended_cat, action, engagement, reward,
         json.dumps({TRAINING_SUBREDDITS[i]: round(float(q_values[i]), 4) for i in range(len(TRAINING_SUBREDDITS))}),
         agent.epsilon, agent.version, 1 if is_match else 0)
    )
    conn.commit()
    conn.close()
    
    return RecommendResponse(
        recommended_category=recommended_cat,
        recommended_idx=action,
        confidence=round(float(np.max(q_values)) / max(float(np.sum(np.abs(q_values))), 1e-6), 4),
        q_values={TRAINING_SUBREDDITS[i]: round(float(q_values[i]), 4) for i in range(len(TRAINING_SUBREDDITS))},
        reward=round(reward, 4),
        engagement_score=round(engagement, 4),
        model_version=agent.version,
        timestamp=timestamp
    )

@app.post("/train")
def train(request: TrainRequest):
    """Train the RL agent on Reddit data"""
    print(f"\n🔧 Training RL agent for {request.episodes} episodes on {request.subreddits}...")
    
    # Generate training data
    posts = generate_synthetic_reddit_data(request.subreddits, n_per_sub=50)
    
    total_reward = 0
    losses = []
    
    for episode in range(request.episodes):
        # Sample a random post
        post = random.choice(posts)
        state = encoder.encode_post(post)
        action = agent.select_action(state, training=True)
        
        # Reward: based on whether recommendation matches post category
        recommended_cat = TRAINING_SUBREDDITS[action]
        engagement = post["engagement_score"]
        reward = engagement if post["subreddit"] == recommended_cat else engagement * 0.1
        
        # Next state (sample another post)
        next_post = random.choice(posts)
        next_state = encoder.encode_post(next_post)
        
        agent.store_experience(state, action, reward, next_state, False)
        agent.episode_rewards.append(reward)
        total_reward += reward
        
        # Train
        loss = agent.train_step()
        if loss > 0:
            losses.append(loss)
    
    # Save model
    agent.save(MODEL_PATH)
    
    mean_reward = total_reward / request.episodes
    mean_loss = np.mean(losses) if losses else 0.0
    
    return {
        "episodes": request.episodes,
        "mean_reward": round(mean_reward, 4),
        "mean_loss": round(mean_loss, 4),
        "epsilon": round(agent.epsilon, 4),
        "training_steps": agent.training_steps,
        "model_version": agent.version
    }

@app.get("/metrics/recent")
def recent_metrics(window: int = 50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT reward, engagement_score, recommended_category, is_relevant, recommended_idx FROM recommendations ORDER BY id DESC LIMIT ?",
        (window,)
    ).fetchall()
    conn.close()
    
    if not rows:
        return {"error": "No recommendations yet", "count": 0}
    
    rewards = [r[0] for r in rows]
    engagements = [r[1] for r in rows]
    categories = [r[2] for r in rows]
    relevant = [r[3] for r in rows if r[3] is not None]
    
    mean_reward = float(np.mean(rewards))
    
    return {
        "window_size": len(rows),
        "mean_reward": round(mean_reward, 4),
        "std_reward": round(float(np.std(rewards)), 4),
        "mean_engagement": round(float(np.mean(engagements)), 4),
        "relevance_rate": round(float(np.mean(relevant)), 4) if relevant else None,
        "category_distribution": {cat: categories.count(cat) for cat in set(categories)},
        "baseline_mean_reward": 0.5,
        "reward_drift": round(mean_reward - 0.5, 4)
    }

@app.get("/metrics/all_recommendations")
def all_recommendations(limit: int = 100):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, timestamp, post_title, post_subreddit, recommended_category, reward, engagement_score, is_relevant, model_version FROM recommendations ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    
    return [
        {
            "id": r[0], "timestamp": r[1], "post_title": r[2][:80],
            "post_subreddit": r[3], "recommended": r[4],
            "reward": r[5], "engagement": r[6], "relevant": r[7], "version": r[8]
        }
        for r in rows
    ]

@app.get("/metrics/agent_actions")
def get_agent_actions(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, agent_name, action_type, description, status FROM agent_actions ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    
    return [
        {"timestamp": r[0], "agent": r[1], "action": r[2], "description": r[3], "status": r[4]}
        for r in rows
    ]

@app.get("/agent/stats")
def agent_stats():
    return agent.get_stats()

@app.get("/baseline")
def get_baseline():
    return {
        "mean_reward": 0.5,
        "mean_engagement": 0.5,
        "categories": TRAINING_SUBREDDITS,
        "model_version": agent.version
    }

if __name__ == "__main__":
    print(f"🚀 Starting RL Recommendation Server v{agent.version}")
    print(f"📊 Categories: {TRAINING_SUBREDDITS}")
    print(f"💾 Database: {DB_PATH}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
