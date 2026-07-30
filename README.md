# 🤖 Self-Healing RL Recommendation Agent

**A reinforcement learning content recommendation system that autonomously detects when its recommendations start failing, diagnoses the root cause, retrains itself, and verifies the fix. Uses MCP for tool access and A2A (Agent-to-Agent) protocol for multi-agent coordination.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/riya0920/self-healing-rl-pipeline/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## The Problem

RL recommendation agents are trained on historical user behavior. When user preferences shift (new topics trend, seasonal changes occur, or the content distribution changes), the agent's learned policy becomes stale. In production, this means **bad recommendations, dropping engagement, and lost revenue.**

Most systems rely on humans to notice the degradation, diagnose the issue, and manually retrain. **This system does it autonomously.**

## How It Works

### The RL Agent

A **Deep Q-Network (DQN)** learns which content categories to recommend to maximize user engagement. Trained on Reddit post data from subreddits like r/technology, r/sports, r/politics, r/science.

### The Drift

When the content stream shifts to domains the agent has never seen (r/cooking, r/fitness, r/legaladvice), the agent's recommendations become irrelevant. Reward drops. Relevance tanks.

### The Self-Healing Loop

```
Live Reddit Posts → RL Agent serves recommendations
                         ↓ all interactions logged to SQLite
Monitor Agent    → watches reward curves, detects engagement drops
                         ↓ (drift detected)
Diagnostics Agent → analyzes logs via MCP: "75% OOD posts detected"
                         ↓ (root cause identified)
Repair Agent     → retrains DQN on clean training data, deploys new version
                         ↓ (fix applied)
Verification Agent → validates new model, approves or sends back
```

### Demo Output

```
🚨 Monitor detected 5 drift signals:
   ⚠️ Reward dropped by 0.3600 (threshold: 0.25)
   ⚠️ Relevance rate: 12.5% (threshold: 30.0%)
   ⚠️ 90% low reward recommendations
   ⚠️ 75% out-of-domain posts from non-training subreddits
   ⚠️ Category collapse: technology at 65%

🔬 Diagnostics: Root cause: out_of_domain_content (critical)
   30/30 low-reward posts from unknown subreddits

🔧 Repair: Retrained RL policy, new version deployed

✔️ Verification: reward=0.4224, relevance=42%, status=APPROVED

✅ System self-healed autonomously
```

## Architecture

```
self-healing-rl-pipeline/
├── config.py                # Reddit API credentials + all settings
├── scraper.py               # Reddit data scraper (PRAW + JSON fallback)
├── rl_agent.py              # DQN agent (PyTorch) with replay buffer
├── server.py                # FastAPI server serving recommendations
├── train_initial.py         # Initial RL agent training script
├── drift_simulator.py       # 4-phase drift simulation
├── dashboard.py             # Streamlit real-time monitoring
├── observability.py         # LangSmith tracing wrapper (no-op when disabled)
├── .env.example             # Reddit + LangSmith environment variables
├── requirements.txt
├── agents/
│   ├── monitor_agent.py     # Watches reward curves + engagement
│   ├── diagnostics_agent.py # Root cause analysis via MCP tools
│   ├── repair_agent.py      # Retrains DQN + deploys new version
│   └── verification_agent.py # Validates fix, approves or rejects
├── a2a/
│   ├── protocol.py          # A2A message format, agent cards, task lifecycle
│   └── orchestrator.py      # Coordinates full self-healing loop
└── mcp/
    └── tools.py             # MCP tools for DB access, metrics, retraining
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set Reddit API credentials (optional, works without them using synthetic data)
export REDDIT_CLIENT_ID=your_id
export REDDIT_CLIENT_SECRET=your_secret

# Enable LangSmith tracing (optional — leave unset to run without it)
export LANGSMITH_API_KEY=lsv2_...        # from https://smith.langchain.com/settings
export LANGSMITH_PROJECT=self-healing-rl

# 1. Train the RL agent
python train_initial.py

# 2. Start the server (Terminal 1)
python -m uvicorn server:app --port 8000

# 3. Run drift simulation with intensity 0.3 (Terminal 2)
python drift_simulator.py 0.3

# 4. Run self-healing agents (Terminal 2, after drift finishes)
python a2a/orchestrator.py

# 5. View dashboard (Terminal 3)
python -m streamlit run dashboard.py
```

## Tech Stack

| Component | Technology |
| --- | --- |
| RL Agent | PyTorch DQN with experience replay + target network |
| Data | Reddit API (PRAW) with synthetic fallback |
| Serving | FastAPI + Uvicorn |
| Storage | SQLite (recommendations, metrics, agent actions, model registry) |
| Agent Coordination | A2A Protocol (JSON-over-HTTP, agent cards) |
| Tool Access | MCP (Model Context Protocol) |
| Observability | LangSmith tracing of every healing cycle |
| Dashboard | Streamlit (4 tabs: performance, agents, logs, registry) |
| Drift Detection | Reward monitoring, OOD detection, category collapse |

## Drift Phases

| Phase | Data Source | Expected RL Agent Behavior |
| --- | --- | --- |
| 1: In-Distribution | r/technology, r/sports, r/politics, r/science | High reward, good relevance |
| 2: Mild Drift | r/cooking, r/fitness, r/legaladvice, r/medicine | Reward drops, wrong recommendations |
| 3: Heavy Drift | r/philosophy, r/art, r/gardening, r/astronomy | Very low reward, category collapse |
| 4: Mixed Chaos | All subreddits randomly | Complete confusion |

## Agent Communication (A2A Protocol)

Each agent communicates via JSON messages over HTTP. Example task handoff from Monitor to Diagnostics:

```json
{
  "task_id": "task-bb37ed06",
  "from_agent": "monitor",
  "to_agent": "diagnostics",
  "action": "investigate_drift",
  "status": "pending",
  "payload": {
    "drift_signals": ["Reward dropped by 0.36", "75% OOD posts"],
    "drift_type": "out_of_domain"
  }
}
```

## Observability (LangSmith)

Every self-healing run is traced with [LangSmith](https://smith.langchain.com).
Set `LANGSMITH_API_KEY` and each cycle appears as a single nested trace:

```
self_healing_cycle
├── monitor.detect_drift
│   ├── mcp.get_recent_metrics
│   └── mcp.get_reward_distribution      # OOD + reward-distribution analysis
├── diagnostics.investigate
├── repair.execute
│   └── mcp.retrain_rl_agent             # DQN retrain + versioned deploy
└── verification.verify
    └── mcp.run_validation               # reward/relevance thresholds
```

This gives per-agent latency, inputs/outputs, and root-cause payloads for
every drift → diagnose → repair → verify loop. Tracing is fully optional:
`observability.py` degrades to a **no-op** when `langsmith` is not installed
or `LANGSMITH_API_KEY` is unset, so the pipeline runs unchanged either way.
Set `LANGSMITH_TRACING=false` to force it off even with a key present.

## Reddit API Setup (Optional)

1. Go to <https://www.reddit.com/prefs/apps>
2. Click "create app" then select "script"
3. Copy `client_id` and `client_secret`
4. Set in `config.py` or as environment variables

Without API credentials, the system uses realistic synthetic Reddit data.

## License

MIT

## Author

**Riya Soni** · MS Computer Science, Stevens Institute of Technology  
[GitHub](https://github.com/riya0920) · [LinkedIn](https://linkedin.com/in/riya-soni-ml-engineer)
