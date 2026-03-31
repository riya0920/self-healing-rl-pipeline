"""
Self-Healing RL Recommendation Agent — Streamlit Dashboard
Real-time monitoring of RL agent performance, drift detection, and agent activity
Run: streamlit run dashboard.py
"""
import os
import sys
import streamlit as st
import sqlite3
import json
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

st.set_page_config(page_title="Self-Healing RL Recommendation Agent", page_icon="🤖", layout="wide")

def get_db():
    return sqlite3.connect(DB_PATH)

def load_recommendations(limit=200):
    conn = get_db()
    df = pd.read_sql_query(
        f"SELECT id, timestamp, post_title, post_subreddit, recommended_category, reward, engagement_score, is_relevant, model_version FROM recommendations ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

def load_agent_actions(limit=50):
    conn = get_db()
    df = pd.read_sql_query(
        f"SELECT timestamp, agent_name, action_type, description, status FROM agent_actions ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

def load_model_registry():
    conn = get_db()
    df = pd.read_sql_query("SELECT timestamp, version, mean_reward, training_steps, episodes, status, notes FROM model_registry ORDER BY id DESC", conn)
    conn.close()
    return df

# Header
st.title("🤖 Self-Healing RL Recommendation Agent")
st.markdown("**Real-time monitoring of RL-powered content recommendations with autonomous drift detection and self-healing**")
st.markdown("---")

# Metrics
recs = load_recommendations()
actions = load_agent_actions()
registry = load_model_registry()

col1, col2, col3, col4, col5 = st.columns(5)

if not recs.empty:
    recent = recs.head(50)
    mean_reward = recent['reward'].mean()
    relevance = recent[recent['is_relevant'].notna()]['is_relevant'].mean() if recent['is_relevant'].notna().any() else None
    
    col1.metric("Total Recommendations", len(recs))
    col2.metric("Mean Reward", f"{mean_reward:.3f}", delta=f"{mean_reward - 0.5:.3f}", delta_color="normal")
    col3.metric("Baseline Reward", "0.500")
    col4.metric("Relevance Rate", f"{relevance:.1%}" if relevance is not None else "N/A",
                delta=f"{relevance - 0.5:.1%}" if relevance is not None else None, delta_color="normal")
    col5.metric("Agent Actions", len(actions))
else:
    col1.metric("Total Recommendations", 0)
    col2.metric("Mean Reward", "N/A")
    col3.metric("Baseline Reward", "0.500")
    col4.metric("Relevance Rate", "N/A")
    col5.metric("Agent Actions", 0)

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 RL Performance", "🤖 Agent Activity", "📋 Recommendation Logs", "🗄️ Model Registry"])

with tab1:
    st.subheader("RL Agent Performance Over Time")
    if not recs.empty:
        chart = recs[['id', 'reward']].sort_values('id')
        st.markdown("**Reward Per Recommendation**")
        st.line_chart(chart.set_index('id')['reward'], use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Recommended Category Distribution (last 50)**")
            cat_dist = recent['recommended_category'].value_counts()
            st.bar_chart(cat_dist, use_container_width=True)
        with col_b:
            st.markdown("**Post Subreddit Distribution (last 50)**")
            sub_dist = recent['post_subreddit'].value_counts()
            st.bar_chart(sub_dist, use_container_width=True)
        
        if recs['is_relevant'].notna().any():
            st.markdown("**Rolling Relevance Rate (window=10)**")
            rel_data = recs[recs['is_relevant'].notna()].sort_values('id').copy()
            rel_data['rolling_relevance'] = rel_data['is_relevant'].rolling(window=10, min_periods=1).mean()
            st.line_chart(rel_data.set_index('id')['rolling_relevance'], use_container_width=True)
    else:
        st.info("No recommendations yet. Start the server and drift simulator.")

with tab2:
    st.subheader("Agent Activity Log")
    if not actions.empty:
        icons = {'monitor': '🔍', 'diagnostics': '🔬', 'repair': '🔧', 'verification': '✔️'}
        for _, row in actions.iterrows():
            icon = icons.get(row['agent_name'], '❓')
            if 'drift_detected' in row['action_type']:
                st.error(f"{icon} **{row['agent_name'].upper()}** | {row['action_type']} | {row['description']}")
            elif 'approved' in row['action_type'] or 'resolved' in row['action_type']:
                st.success(f"{icon} **{row['agent_name'].upper()}** | {row['action_type']} | {row['description']}")
            elif 'rejected' in row['action_type'] or 'failed' in row['action_type']:
                st.warning(f"{icon} **{row['agent_name'].upper()}** | {row['action_type']} | {row['description']}")
            else:
                st.info(f"{icon} **{row['agent_name'].upper()}** | {row['action_type']} | {row['description']}")
        
        st.markdown("---")
        st.markdown("**Agent Action Counts**")
        st.bar_chart(actions['agent_name'].value_counts(), use_container_width=True)
    else:
        st.info("No agent actions yet. Run the orchestrator.")

with tab3:
    st.subheader("Recent Recommendations")
    if not recs.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            reward_filter = st.slider("Min Reward", 0.0, 1.0, 0.0, 0.05)
        with col_f2:
            cat_filter = st.multiselect("Filter by Recommended", recs['recommended_category'].unique().tolist(),
                                        default=recs['recommended_category'].unique().tolist())
        
        filtered = recs[(recs['reward'] >= reward_filter) & (recs['recommended_category'].isin(cat_filter))]
        display = filtered[['id', 'timestamp', 'post_title', 'post_subreddit', 'recommended_category', 'reward', 'is_relevant', 'model_version']].copy()
        display['post_title'] = display['post_title'].str[:60] + '...'
        display['reward'] = display['reward'].round(4)
        display['is_relevant'] = display['is_relevant'].map({1.0: '✅', 0.0: '❌', None: '❓'})
        st.dataframe(display, use_container_width=True, height=400)
        st.markdown(f"**Showing {len(filtered)} of {len(recs)} recommendations**")
    else:
        st.info("No recommendations yet.")

with tab4:
    st.subheader("Model Registry")
    if not registry.empty:
        for _, row in registry.iterrows():
            icon = "🟢" if row['status'] == 'active' else "⚪"
            st.markdown(f"{icon} **{row['version']}** | Reward: {row['mean_reward']:.4f} | Steps: {row['training_steps']} | Episodes: {row['episodes']} | {row['status']} | {row['notes'] or ''}")
    else:
        st.info("No models registered.")

# Sidebar
st.sidebar.title("🤖 Pipeline Controls")
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Start:**")
st.sidebar.code("""
# 1. Train initial model
python train_initial.py

# 2. Start server
uvicorn server:app --port 8000

# 3. Run drift simulation
python drift_simulator.py 0.3

# 4. Run agents
python a2a/orchestrator.py

# 5. Dashboard
streamlit run dashboard.py
""")
st.sidebar.markdown("---")
st.sidebar.markdown(f"- Training subs: `{', '.join(TRAINING_SUBREDDITS)}`")
st.sidebar.markdown(f"- Drift subs: `{', '.join(DRIFT_SUBREDDITS)}`")
if st.sidebar.button("🔄 Refresh"):
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Riya Soni**")
st.sidebar.markdown("[GitHub](https://github.com/riya0920) | [LinkedIn](https://linkedin.com/in/riya-soni-ml-engineer)")
