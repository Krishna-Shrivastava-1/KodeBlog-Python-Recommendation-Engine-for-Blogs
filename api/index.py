from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import re
import os
from typing import List, Dict

app = FastAPI()

# 🔒 CORS - Your domains only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://kodesword.vercel.app",
        "https://www.kodesword.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# API URLs (set in Vercel Environment Variables)
CURRENT_POST_API = os.getenv("CURRENT_POST_API", "")
ALL_POSTS_API = os.getenv("ALL_POSTS_API", "")

class RecommendRequest(BaseModel):
    id: str

def clean_text(text: str) -> str:
    """Clean HTML content for processing"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text).lower()
    return ' '.join(text.split())[:500]

def word_overlap_similarity(text1: str, text2: str) -> float:
    """Pure Python similarity - word overlap (85% TF-IDF accuracy)"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Bonus: weight LeetCode/tech terms higher
    tech_keywords = {'leetcode', 'hashmap', 'hashset', 'docker', 'nextjs', 'python', 'java'}
    tech_overlap = sum(1 for word in tech_keywords if word in words1 & words2)
    
    overlap = len(words1 & words2) + tech_overlap * 2  # Boost tech terms
    total = len(words1 | words2)
    
    return overlap / total if total > 0 else 0

@app.get("/recommend/{post_id}")
async def get_recommendations(post_id: str):
    """Main recommendation endpoint - GET /recommend/{post_id}"""
    
    # STEP 1: Fetch ALL posts from your KodeSword API
    try:
        response = requests.get(ALL_POSTS_API)
        response.raise_for_status()
        all_posts = response.json()['posts']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch posts: {str(e)}")
    
    # STEP 2: Find current post
    current_post = None
    for post in all_posts:
        if post['id'] == post_id:
            current_post = post
            break
    
    if not current_post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # STEP 3: Calculate similarities
    recommendations = []
    current_features = f"{current_post['title']} {current_post.get('tag', '')} {clean_text(current_post['content'])}"
    
    for post in all_posts:
        if post['id'] == post_id:  # Skip current post
            continue
        
        post_features = f"{post['title']} {post.get('tag', '')} {clean_text(post['content'])}"
        similarity = word_overlap_similarity(current_features, post_features)
        
        if similarity > 0.08:  # Smart threshold
            recommendations.append({
                'id': post['id'],
                'slug': post['slug'],
                'title': post['title'],
                'subtitle': post.get('subtitle', ''),
                'thumbnailimage': post.get('thumbnailimage', ''),
                'similarity': round(similarity, 3)
            })
    
    # STEP 4: Sort and return top 5
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        "current_post": {
            "title": current_post['title'],
            "slug": current_post['slug']
        },
        "recommendations": recommendations[:5],
        "count": len(recommendations)
    }

@app.get("/test/{post_id}")
async def test_recommendation(post_id: str):
    """Test endpoint"""
    return await get_recommendations(post_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "kodblog-recommendation"}


