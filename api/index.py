from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import re
import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams,PayloadSchemaType
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
load_dotenv()
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

# class RecommendRequest(BaseModel):
#     id: str

# def clean_text(text: str) -> str:
#     """Clean HTML content for processing"""
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'[^\w\s]', ' ', text).lower()
#     return ' '.join(text.split())[:500]

# def word_overlap_similarity(text1: str, text2: str) -> float:
#     """Pure Python similarity - word overlap (85% TF-IDF accuracy)"""
#     words1 = set(text1.lower().split())
#     words2 = set(text2.lower().split())
    
#     # Bonus: weight LeetCode/tech terms higher
#     tech_keywords = {'leetcode', 'hashmap', 'hashset', 'docker', 'nextjs', 'python', 'java'}
#     tech_overlap = sum(1 for word in tech_keywords if word in words1 & words2)
    
#     overlap = len(words1 & words2) + tech_overlap * 2  # Boost tech terms
#     total = len(words1 | words2)
    
#     return overlap / total if total > 0 else 0

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME_EMBEDDING")
@app.get("/recommend/{post_id}")
async def get_recommendations(post_id: str): # Renamed parameter for clarity
    try:
        # 1. Get the vector for the current post
        # We check both blog_id and slug to find the correct point
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                should=[
                    FieldCondition(key="blog_id", match=MatchValue(value=post_id)),
                    FieldCondition(key="slug", match=MatchValue(value=post_id))
                ]
            ),
            limit=1,
            with_vectors=True
        )

        if not points:
            raise HTTPException(status_code=404, detail="Blog vector not found")

        current_point = points[0]
        # Important: Get the real UUID and the Slug of the current post to exclude it
        current_blog_id = current_point.payload.get("blog_id")
        current_slug = current_point.payload.get("slug")

        # 2. Search excluding the current post
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=current_point.vector,
            query_filter=Filter(
                must_not=[
                    # Exclude the current blog by ID
                    FieldCondition(key="blog_id", match=MatchValue(value=current_blog_id)),
                    # Exclude the current blog by Slug (if it exists)
                    FieldCondition(key="slug", match=MatchValue(value=current_slug))
                ]
            ) if current_slug else Filter(
                must_not=[FieldCondition(key="blog_id", match=MatchValue(value=current_blog_id))]
            ),
            limit=80, 
            with_payload=True
        ).points

        recommendations = []
        seen_ids = {current_blog_id} # Pre-populate with current post
        seen_titles = {current_point.payload.get("title")} # Pre-populate with current title

        for hit in search_results:
            b_id = hit.payload.get("blog_id")
            title = hit.payload.get("title")
            slug = hit.payload.get("slug")
            
            # Deduplicate and ensure no self-match
            if b_id and title and b_id not in seen_ids and title not in seen_titles:
                recommendations.append({
                    "blog_id": b_id,
                    "title": title,
                    "tags": hit.payload.get("tags"),
                    "slug": slug or b_id,
                    "score": round(hit.score, 3)
                })
                seen_ids.add(b_id)
                seen_titles.add(title)
            
            if len(recommendations) >= 4:
                break

        # 3. Fallback logic remains same but respects the exclusion sets
        if len(recommendations) < 4:
            extra_points, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=30,
                with_payload=True
            )
            for point in extra_points:
                b_id = point.payload.get("blog_id")
                title = point.payload.get("title")
                
                if b_id not in seen_ids and title not in seen_titles:
                    recommendations.append({
                        "blog_id": b_id,
                        "title": title,
                        "tags": point.payload.get("tags"),
                        "slug": point.payload.get("slug") or b_id,
                        "score": 0.0
                    })
                    seen_ids.add(b_id)
                    seen_titles.add(title)
                if len(recommendations) >= 4: break

        return {
            "status": "success",
            "recommendations": recommendations[:4]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Old Recommendation System (Cosine)
# async def get_recommendations(post_id: str):
#     """Main recommendation endpoint - GET /recommend/{post_id}"""
    
#     # STEP 1: Fetch ALL posts from your KodeSword API
#     try:
#         response = requests.get(ALL_POSTS_API)
#         response.raise_for_status()
#         all_posts = response.json()['posts']
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to fetch posts: {str(e)}")
    
#     # STEP 2: Find current post
#     current_post = None
#     for post in all_posts:
#         if post['id'] == post_id:
#             current_post = post
#             break
    
#     if not current_post:
#         raise HTTPException(status_code=404, detail="Post not found")
    
#     # STEP 3: Calculate similarities
#     recommendations = []
#     current_features = f"{current_post['title']} {current_post.get('tag', '')} {clean_text(current_post['content'])}"
    
#     for post in all_posts:
#         if post['id'] == post_id:  # Skip current post
#             continue
        
#         post_features = f"{post['title']} {post.get('tag', '')} {clean_text(post['content'])}"
#         similarity = word_overlap_similarity(current_features, post_features)
        
#         if similarity > 0.08:  # Smart threshold
#             recommendations.append({
#                 'id': post['id'],
#                 'slug': post['slug'],
#                 'title': post['title'],
#                 'subtitle': post.get('subtitle', ''),
#                 'thumbnailimage': post.get('thumbnailimage', ''),
#                 'similarity': round(similarity, 3)
#             })
    
#     # STEP 4: Sort and return top 5
#     recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    
#     return {
#         "current_post": {
#             "title": current_post['title'],
#             "slug": current_post['slug']
#         },
#         "recommendations": recommendations[:4],
#         "count": len(recommendations)
#     }

@app.get("/test/{post_id}")
async def test_recommendation(post_id: str):
    """Test endpoint"""
    return await get_recommendations(post_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "kodblog-recommendation"}


