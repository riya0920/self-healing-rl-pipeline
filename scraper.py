"""
Self-Healing RL Recommendation Agent — Reddit Data Scraper
Scrapes live posts from Reddit using PRAW (with API keys) or public JSON (fallback)
"""
import os
import sys
import json
import time
import random
import requests
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class RedditScraper:
    """Scrapes Reddit posts using PRAW or public JSON API fallback"""
    
    def __init__(self):
        self.use_praw = False
        self.praw_reddit = None
        self._init_praw()
    
    def _init_praw(self):
        """Try to initialize PRAW, fallback to public JSON if no credentials"""
        if REDDIT_CLIENT_ID != "YOUR_CLIENT_ID_HERE":
            try:
                import praw
                self.praw_reddit = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT
                )
                self.use_praw = True
                print("✅ Reddit API: Using PRAW (authenticated)")
            except Exception as e:
                print(f"⚠️  PRAW failed: {e}. Falling back to public JSON API.")
        else:
            print("⚠️  Reddit API: No credentials found. Using public JSON API (rate-limited).")
            print("   Set credentials in config.py or environment variables for better performance.")
    
    def fetch_posts(self, subreddit: str, limit: int = 25, sort: str = "hot") -> List[Dict]:
        """Fetch posts from a subreddit"""
        if self.use_praw:
            return self._fetch_praw(subreddit, limit, sort)
        else:
            return self._fetch_json(subreddit, limit, sort)
    
    def _fetch_praw(self, subreddit: str, limit: int, sort: str) -> List[Dict]:
        """Fetch using PRAW (authenticated, higher rate limits)"""
        try:
            sub = self.praw_reddit.subreddit(subreddit)
            if sort == "hot":
                posts = sub.hot(limit=limit)
            elif sort == "new":
                posts = sub.new(limit=limit)
            elif sort == "top":
                posts = sub.top(limit=limit, time_filter="day")
            else:
                posts = sub.hot(limit=limit)
            
            results = []
            for post in posts:
                if post.stickied:
                    continue
                results.append({
                    "id": post.id,
                    "title": post.title,
                    "subreddit": subreddit,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "created_utc": post.created_utc,
                    "url": post.url,
                    "selftext": post.selftext[:200] if post.selftext else "",
                    "scraped_at": datetime.now().isoformat(),
                    "engagement_score": self._calc_engagement(post.score, post.num_comments, post.upvote_ratio)
                })
            return results
        except Exception as e:
            print(f"  ❌ PRAW error on r/{subreddit}: {e}")
            return self._fetch_json(subreddit, limit, sort)
    
    def _fetch_json(self, subreddit: str, limit: int, sort: str) -> List[Dict]:
        """Fetch using public JSON API (no auth needed, rate-limited)"""
        try:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
            headers = {"User-Agent": REDDIT_USER_AGENT}
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code == 429:
                print(f"  ⚠️  Rate limited on r/{subreddit}. Waiting 2s...")
                time.sleep(2)
                resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code != 200:
                print(f"  ❌ HTTP {resp.status_code} on r/{subreddit}")
                return []
            
            data = resp.json()
            results = []
            for child in data.get("data", {}).get("children", []):
                post = child["data"]
                if post.get("stickied", False):
                    continue
                results.append({
                    "id": post["id"],
                    "title": post["title"],
                    "subreddit": subreddit,
                    "score": post.get("score", 0),
                    "upvote_ratio": post.get("upvote_ratio", 0.5),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "url": post.get("url", ""),
                    "selftext": post.get("selftext", "")[:200],
                    "scraped_at": datetime.now().isoformat(),
                    "engagement_score": self._calc_engagement(
                        post.get("score", 0), 
                        post.get("num_comments", 0), 
                        post.get("upvote_ratio", 0.5)
                    )
                })
            
            time.sleep(1)  # Be respectful of rate limits
            return results
        except Exception as e:
            print(f"  ❌ JSON API error on r/{subreddit}: {e}")
            return []
    
    def _calc_engagement(self, score: int, num_comments: int, upvote_ratio: float) -> float:
        """Calculate normalized engagement score (0-1)"""
        # Combine upvotes, comments, and ratio into a single engagement metric
        score_norm = min(score / 1000, 1.0)  # Cap at 1000 upvotes
        comments_norm = min(num_comments / 200, 1.0)  # Cap at 200 comments
        ratio_norm = upvote_ratio  # Already 0-1
        
        engagement = 0.4 * score_norm + 0.3 * comments_norm + 0.3 * ratio_norm
        return round(engagement, 4)
    
    def fetch_multi_subreddit(self, subreddits: List[str], limit_per_sub: int = 20) -> List[Dict]:
        """Fetch from multiple subreddits"""
        all_posts = []
        for sub in subreddits:
            posts = self.fetch_posts(sub, limit=limit_per_sub)
            all_posts.extend(posts)
            print(f"  📥 r/{sub}: {len(posts)} posts (avg engagement: {sum(p['engagement_score'] for p in posts)/max(len(posts),1):.3f})")
        
        random.shuffle(all_posts)
        return all_posts


def generate_synthetic_reddit_data(subreddits: List[str], n_per_sub: int = 50) -> List[Dict]:
    """Generate synthetic Reddit-like data for offline testing (no API needed)"""
    
    TEMPLATES = {
        "technology": [
            "New AI model outperforms GPT-4 on benchmarks",
            "Apple announces revolutionary chip architecture",
            "Open source framework for distributed computing released",
            "Cybersecurity breach affects millions of users",
            "Quantum computing breakthrough at MIT",
            "Tesla releases full self-driving update",
            "Microsoft integrates AI into Office suite",
            "Linux kernel gets major performance improvements",
            "New programming language gains popularity among developers",
            "Cloud computing costs drop significantly in 2026",
        ],
        "sports": [
            "Lakers defeat Celtics in overtime thriller",
            "World Cup qualifying match ends in upset",
            "NFL draft picks surprise analysts",
            "Olympic athlete breaks world record",
            "Champions League final draws record viewers",
            "Baseball trade deadline sees major moves",
            "Tennis star announces retirement",
            "MMA fighter wins by knockout in first round",
            "Soccer transfer window opens with big signings",
            "Basketball MVP race heats up in final weeks",
        ],
        "politics": [
            "Senate passes bipartisan infrastructure bill",
            "New immigration policy sparks debate",
            "Election results challenge polling predictions",
            "Supreme Court takes up landmark case",
            "Foreign policy shift announced by administration",
            "Tax reform proposal divides Congress",
            "Governor signs controversial education bill",
            "Climate policy agreement reached at summit",
            "Healthcare reform debate continues in Senate",
            "Defense spending bill faces opposition",
        ],
        "science": [
            "Researchers discover high-temperature superconductor",
            "Mars rover finds evidence of ancient water",
            "New study reveals effects of microplastics on health",
            "CRISPR gene editing shows promise for rare diseases",
            "Astronomers detect signals from distant galaxy",
            "Climate model predicts faster ice sheet melting",
            "Neuroscience breakthrough in memory formation",
            "New species discovered in deep ocean exploration",
            "Physics experiment confirms quantum entanglement theory",
            "Vaccine development enters promising new phase",
        ],
        "cooking": [
            "Best sourdough bread recipe for beginners",
            "How to make authentic Italian carbonara",
            "Air fryer recipes that changed my life",
            "Meal prep ideas for busy professionals",
            "Traditional Japanese ramen from scratch",
            "Vegan desserts that taste amazing",
            "Grilling techniques for perfect steak",
            "Homemade pasta dough masterclass",
            "Quick weeknight dinners under 30 minutes",
            "Fermentation basics for homemade kimchi",
        ],
        "fitness": [
            "5x5 strength program results after 12 weeks",
            "Running form tips that prevent injuries",
            "Best home workout equipment on a budget",
            "How I lost 50 pounds with intermittent fasting",
            "Yoga poses for lower back pain relief",
            "Marathon training plan for beginners",
            "HIIT vs steady state cardio for fat loss",
            "Protein intake recommendations for muscle growth",
            "Swimming workout routine for total body fitness",
            "Stretching routine for desk workers",
        ],
        "legaladvice": [
            "Landlord refuses to return security deposit",
            "Got a speeding ticket in another state",
            "Employer withholding final paycheck",
            "Neighbor's tree fell on my property",
            "HOA imposing unreasonable fines",
            "Car accident with uninsured driver",
            "Wrongful termination question",
            "Small claims court process explained",
            "Copyright infringement on my creative work",
            "Custody agreement modification request",
        ],
        "medicine": [
            "New treatment protocol for autoimmune diseases",
            "Understanding blood test results explained",
            "Breakthrough in Alzheimer's drug development",
            "Mental health resources for healthcare workers",
            "Telemedicine effectiveness study results",
            "Antibiotic resistance growing concern in hospitals",
            "Sleep apnea treatment options compared",
            "Pediatric vaccination schedule updates",
            "Physical therapy techniques for knee rehabilitation",
            "Diabetes management with continuous glucose monitoring",
        ],
        "philosophy": [
            "Is consciousness an illusion or fundamental reality",
            "Existentialism in modern digital age",
            "Ethics of artificial intelligence decision making",
            "Stoicism practical applications for daily life",
            "Free will debate continues in philosophy",
        ],
        "art": [
            "Digital art tools for beginners guide",
            "Contemporary sculpture exhibition opens downtown",
            "Oil painting techniques from the masters",
            "Street art transforms abandoned warehouse district",
            "Photography composition rules you should break",
        ],
        "gardening": [
            "Raised bed garden setup for spring planting",
            "Composting basics for apartment dwellers",
            "Best vegetables to grow in shade",
            "Indoor herb garden maintenance tips",
            "Native plants for pollinator gardens",
        ],
        "astronomy": [
            "Jupiter's moons visible with basic telescope tonight",
            "Comet approaching visible to naked eye this week",
            "Astrophotography guide for beginners",
            "Dark matter theory challenged by new observations",
            "Space station visible passes this month",
        ],
    }
    
    all_posts = []
    for sub in subreddits:
        templates = TEMPLATES.get(sub, [f"Post about {sub} topic {i}" for i in range(10)])
        for i in range(n_per_sub):
            template = random.choice(templates)
            # Add variation
            prefix = random.choice(["", "[Discussion] ", "[News] ", "[OC] ", ""])
            suffix = random.choice(["", " - thoughts?", " - what do you think?", "", ""])
            title = prefix + template + suffix
            
            # Generate realistic engagement scores
            if sub in TRAINING_SUBREDDITS:
                score = random.randint(50, 5000)
                comments = random.randint(10, 500)
                ratio = random.uniform(0.7, 0.98)
            else:
                score = random.randint(10, 2000)
                comments = random.randint(5, 200)
                ratio = random.uniform(0.5, 0.95)
            
            engagement = min(score/1000, 1.0) * 0.4 + min(comments/200, 1.0) * 0.3 + ratio * 0.3
            
            all_posts.append({
                "id": f"synth_{sub}_{i}",
                "title": title,
                "subreddit": sub,
                "score": score,
                "upvote_ratio": round(ratio, 2),
                "num_comments": comments,
                "created_utc": time.time() - random.randint(0, 86400),
                "url": f"https://reddit.com/r/{sub}/synth_{i}",
                "selftext": "",
                "scraped_at": datetime.now().isoformat(),
                "engagement_score": round(engagement, 4)
            })
    
    random.shuffle(all_posts)
    return all_posts


if __name__ == "__main__":
    scraper = RedditScraper()
    print("\n🔍 Testing Reddit Scraper...")
    
    if scraper.use_praw:
        posts = scraper.fetch_multi_subreddit(["technology", "science"], limit_per_sub=5)
    else:
        print("   Using synthetic data for testing (no API credentials)")
        posts = generate_synthetic_reddit_data(["technology", "science"], n_per_sub=5)
    
    print(f"\n📊 Fetched {len(posts)} posts")
    for p in posts[:5]:
        print(f"   [{p['engagement_score']:.3f}] r/{p['subreddit']}: {p['title'][:60]}...")
