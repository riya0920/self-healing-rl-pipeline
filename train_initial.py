"""
Self-Healing RL Recommendation Agent — Initial Training
Trains the DQN agent on training subreddits before deployment
"""
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from rl_agent import DQNAgent, PostFeatureEncoder
from scraper import generate_synthetic_reddit_data

def train_initial(episodes: int = 500):
    print("=" * 60)
    print("🎓 INITIAL RL AGENT TRAINING")
    print("=" * 60)
    
    agent = DQNAgent()
    encoder = PostFeatureEncoder()
    
    print(f"📊 Training subreddits: {TRAINING_SUBREDDITS}")
    print(f"📊 Episodes: {episodes}")
    print(f"📊 State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    print()
    
    # Generate training data
    print("📥 Generating training data...")
    posts = generate_synthetic_reddit_data(TRAINING_SUBREDDITS, n_per_sub=100)
    print(f"   {len(posts)} training posts generated")
    
    # Training loop
    print("\n🔧 Training...")
    total_reward = 0
    rewards_history = []
    losses = []
    
    for ep in range(episodes):
        post = random.choice(posts)
        state = encoder.encode_post(post)
        action = agent.select_action(state, training=True)
        
        recommended_cat = TRAINING_SUBREDDITS[action]
        engagement = post["engagement_score"]
        
        # Reward: high if recommendation matches, scaled by engagement
        if post["subreddit"] == recommended_cat:
            reward = engagement  # Correct recommendation
        else:
            reward = engagement * 0.1  # Wrong recommendation
        
        next_post = random.choice(posts)
        next_state = encoder.encode_post(next_post)
        
        agent.store_experience(state, action, reward, next_state, False)
        agent.episode_rewards.append(reward)
        total_reward += reward
        rewards_history.append(reward)
        
        loss = agent.train_step()
        if loss > 0:
            losses.append(loss)
        
        if (ep + 1) % 100 == 0:
            recent_reward = np.mean(rewards_history[-100:])
            recent_loss = np.mean(losses[-100:]) if losses else 0
            print(f"   Episode {ep+1}/{episodes} | Reward: {recent_reward:.4f} | Loss: {recent_loss:.4f} | Epsilon: {agent.epsilon:.3f}")
    
    # Save model
    model_path = os.path.join(PIPELINE_DIR, "rl_model.pt")
    agent.save(model_path)
    
    # Final stats
    mean_reward = total_reward / episodes
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"   Mean Reward: {mean_reward:.4f}")
    print(f"   Final Epsilon: {agent.epsilon:.4f}")
    print(f"   Training Steps: {agent.training_steps}")
    print(f"   Memory Size: {len(agent.memory)}")
    print(f"   Model saved: {model_path}")
    
    # Quick validation
    print(f"\n🧪 Quick Validation...")
    correct = 0
    total = 50
    test_posts = generate_synthetic_reddit_data(TRAINING_SUBREDDITS, n_per_sub=15)[:total]
    
    for post in test_posts:
        state = encoder.encode_post(post)
        action = agent.select_action(state, training=False)
        recommended = TRAINING_SUBREDDITS[action]
        if post["subreddit"] == recommended:
            correct += 1
    
    accuracy = correct / total
    print(f"   Recommendation Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"   {'✅ Good enough to deploy!' if accuracy > 0.2 else '⚠️  Low accuracy, but drift will be detectable'}")

if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    train_initial(episodes)
