"""
Self-Healing RL Recommendation Agent — DQN Agent
Deep Q-Network that learns which subreddit categories to recommend
to maximize user engagement (upvotes, comments, engagement score)
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class QNetwork(nn.Module):
    """Deep Q-Network: maps state → Q-values for each action (category)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for stable DQN training"""
    
    def __init__(self, capacity: int = MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int = BATCH_SIZE) -> Tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class PostFeatureEncoder:
    """Encodes Reddit post features into a fixed-size state vector"""
    
    def __init__(self, state_dim: int = STATE_DIM):
        self.state_dim = state_dim
        # Simple vocabulary for title encoding
        self.vocab = {}
        self.vocab_size = 0
    
    def encode_post(self, post: Dict) -> np.ndarray:
        """Convert a Reddit post into a state vector"""
        features = np.zeros(self.state_dim, dtype=np.float32)
        
        # Feature 1-4: Engagement metrics (normalized)
        features[0] = min(post.get("score", 0) / 5000, 1.0)
        features[1] = min(post.get("num_comments", 0) / 500, 1.0)
        features[2] = post.get("upvote_ratio", 0.5)
        features[3] = post.get("engagement_score", 0.0)
        
        # Feature 4-8: Time features
        created = post.get("created_utc", 0)
        if created > 0:
            import time
            age_hours = (time.time() - created) / 3600
            features[4] = min(age_hours / 24, 1.0)  # Normalized age
        
        # Feature 8-40: Bag of words from title (simple hash-based encoding)
        title = post.get("title", "").lower()
        words = title.split()
        for i, word in enumerate(words[:16]):
            idx = 8 + (hash(word) % 32)
            features[idx] += 1.0
        
        # Feature 40-52: Subreddit one-hot encoding
        all_subs = TRAINING_SUBREDDITS + DRIFT_SUBREDDITS + HEAVY_DRIFT_SUBREDDITS
        sub = post.get("subreddit", "")
        for i, s in enumerate(all_subs[:12]):
            if sub == s:
                features[40 + i] = 1.0
        
        # Feature 52-64: Title statistics
        features[52] = min(len(words) / 20, 1.0)  # Title length
        features[53] = min(len(title) / 200, 1.0)  # Character count
        features[54] = sum(1 for c in title if c.isupper()) / max(len(title), 1)  # Caps ratio
        features[55] = 1.0 if "?" in title else 0.0  # Is question
        features[56] = 1.0 if "!" in title else 0.0  # Has exclamation
        features[57] = 1.0 if any(w in title for w in ["breaking", "new", "just"]) else 0.0
        
        return features
    
    def encode_batch(self, posts: List[Dict]) -> np.ndarray:
        """Encode a batch of posts"""
        return np.array([self.encode_post(p) for p in posts])
    
    def encode_context(self, recent_posts: List[Dict], window: int = 5) -> np.ndarray:
        """Encode recent interaction context as state"""
        if not recent_posts:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Average the features of recent posts as context
        recent = recent_posts[-window:]
        encoded = self.encode_batch(recent)
        return np.mean(encoded, axis=0)


class DQNAgent:
    """Deep Q-Network agent for content recommendation"""
    
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks (main + target for stable training)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer()
        self.encoder = PostFeatureEncoder(state_dim)
        
        self.epsilon = EPSILON_START
        self.training_steps = 0
        self.episode_rewards = []
        self.version = "v1.0"
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions (for monitoring/debugging)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """Perform one training step using experience replay"""
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values (using target network for stability)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)
        
        # Loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        
        self.training_steps += 1
        
        # Update target network periodically
        if self.training_steps % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "episode_rewards": self.episode_rewards,
            "version": self.version
        }, path)
        print(f"  💾 Model saved: {path} (version {self.version}, {self.training_steps} steps)")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.version = checkpoint.get("version", "v1.0")
        print(f"  📂 Model loaded: {path} (version {self.version}, {self.training_steps} steps)")
    
    def get_stats(self) -> Dict:
        """Get agent statistics for monitoring"""
        recent_rewards = self.episode_rewards[-50:] if self.episode_rewards else []
        return {
            "version": self.version,
            "epsilon": round(self.epsilon, 4),
            "training_steps": self.training_steps,
            "memory_size": len(self.memory),
            "mean_reward_last_50": round(np.mean(recent_rewards), 4) if recent_rewards else 0.0,
            "std_reward_last_50": round(np.std(recent_rewards), 4) if recent_rewards else 0.0,
            "total_episodes": len(self.episode_rewards)
        }


if __name__ == "__main__":
    print("🤖 Testing DQN Agent...")
    agent = DQNAgent()
    encoder = PostFeatureEncoder()
    
    # Simulate a few steps
    for i in range(5):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = agent.select_action(state)
        reward = random.random()
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        agent.store_experience(state, action, reward, next_state, False)
        print(f"  Step {i}: action={TRAINING_SUBREDDITS[action]}, reward={reward:.3f}, epsilon={agent.epsilon:.3f}")
    
    print(f"\n📊 Agent stats: {agent.get_stats()}")
    print("✅ DQN Agent working!")
