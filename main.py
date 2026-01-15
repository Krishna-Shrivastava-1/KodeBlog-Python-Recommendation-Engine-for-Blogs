# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import requests
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from typing import List
# import re
# import os
# from fastapi.middleware.cors import CORSMiddleware 
# app = FastAPI()
# from dotenv import load_dotenv

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         os.getenv("BASE_URL"),                    # Your local dev
#         os.getenv("PROD_BASE_URL"),          # Your production                     
#     ],
#     allow_credentials=True,
#     allow_methods=["GET"],                              # POST, GET for /recommend
#     allow_headers=["*"],
# )


# load_dotenv(verbose=True)
# # YOUR API ENDPOINTS (replace with your real URLs)
# CURRENT_POST_API =os.getenv("CURRENT_POST_API")  # Your getPostBySlug
# ALL_POSTS_API = os.getenv("ALL_POSTS_API")          # Your getAllPosts

# class RecommendRequest(BaseModel):
#     id: str

# def clean_text(text: str) -> str:
#     """Extract words from HTML content + tags"""
#     # Remove HTML tags, keep LeetCode/HashMap etc
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'[^\w\s]', ' ', text).lower()
#     return text[:500]  # First 500 chars


# async def recommend_posts(request: RecommendRequest):
#     id = request.id  # "unique-number-of-occurrences"
#     # STEP 1: Get CURRENT post user is reading
#     try:
#         current_post = requests.get(f"{CURRENT_POST_API}{id}").json()
#     except:
#         raise HTTPException(status_code=404, detail="Current post not found")
#     # STEP 2: Get ALL your blog posts
#     all_posts = requests.get(ALL_POSTS_API).json()['posts']
    
#     # STEP 3: Create features (title + tag + content preview)
#     posts_data = []
#     current_features = ""
    
#     for post in all_posts:
#         if post['id'] == id:
#             # Current post (index 0)
#             features = f"{post['title']} {post['tag']} {clean_text(post['content'])}"
#             current_features = features
#             posts_data.insert(0, {
#                 'id': post['id'],
#                 'slug': post['slug'],
#                 'title': post['title'],
#                 'features': features
#             })
#         else:
#             # Other posts
#             features = f"{post['title']} {post['tag']} {clean_text(post['content'])}"
#             posts_data.append({
#                 'id': post['id'],
#                 'slug': post['slug'], 
#                 'title': post['title'],
#                 'features': features
#             })
    
#     # STEP 4: TF-IDF Magic (compares current post vs all others)
#     df = pd.DataFrame(posts_data)
#     tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
#     tfidf_matrix = tfidf.fit_transform(df['features'])
    
#     # STEP 5: Cosine similarity scores
#     cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)  # Current vs ALL
#     sim_scores = list(enumerate(cosine_sim[0]))
    
#     # STEP 6: Top 5 most similar (exclude current post itself)
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:5]
    
#     # STEP 7: Return recommended post IDs/slugs
#     recommendations = []
#     for idx, score in sim_scores:
#         if score > 0.1:  # Minimum similarity threshold
#             rec_post = df.iloc[idx]
#             recommendations.append({
#                 'id': rec_post['id'],
#                 'slug': rec_post['slug'],
#                 'title': rec_post['title'],
#                 'similarity': float(score)
#             })
    
#     return {
#         "current_post": current_post["postbyid"]['title'],
#         "recommendations": recommendations,
#         "count": len(recommendations)
#     }

# # Test endpoint
# @app.get("/recommend/{id}")
# async def test_recommend(id: str):
#     return await recommend_posts(RecommendRequest(id=id))

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("REC_API_PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port, workers=1)




from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from typing import List, Dict

app = FastAPI()

# 🔒 CORS - Only your domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("BASE_URL"),                    # Your local dev
        os.getenv("PROD_BASE_URL"),          # Your production                     
    ],
    allow_credentials=True,
    allow_methods=["GET"],                              # POST, GET for /recommend
    allow_headers=["*"],
)

from dotenv import load_dotenv
load_dotenv()

CURRENT_POST_API = os.getenv("CURRENT_POST_API")
ALL_POSTS_API = os.getenv("ALL_POSTS_API")

class RecommendRequest(BaseModel):
    id: str

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text).lower()
    return ' '.join(text.split())[:500]

def simple_tfidf_vectorizer(posts: List[str], max_features: int = 1000):
    """Pure NumPy TF-IDF - NO scikit-learn needed"""
    # Get all unique words
    all_words = set()
    for post in posts:
        words = post.split()
        all_words.update(words)
    
    words = list(all_words)[:max_features]
    word_to_idx = {word: i for i, word in enumerate(words)}
    
    # TF: Term Frequency
    tf_matrix = np.zeros((len(posts), len(words)))
    doc_freq = np.zeros(len(words))
    
    for i, post in enumerate(posts):
        word_counts = Counter(post.split())
        total_words = sum(word_counts.values())
        for word, count in word_counts.items():
            if word in word_to_idx:
                idx = word_to_idx[word]
                tf_matrix[i, idx] = count / total_words
                doc_freq[idx] += 1
    
    # IDF: Inverse Document Frequency
    n_docs = len(posts)
    idf = np.log(n_docs / (doc_freq + 1)) + 1
    
    # TF-IDF
    tfidf_matrix = tf_matrix * idf
    
    return tfidf_matrix, words

def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pure NumPy cosine similarity"""
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))


async def recommend_posts(request: RecommendRequest):
    post_id = request.id
    
    # STEP 1: Get ALL posts
    try:
        all_posts_resp = requests.get(ALL_POSTS_API)
        all_posts = all_posts_resp.json()['posts']
    except:
        raise HTTPException(status_code=500, detail="Failed to fetch posts")
    
    # STEP 2: Find current post
    current_post = None
    for post in all_posts:
        if post['id'] == post_id:
            current_post = post
            break
    
    if not current_post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # STEP 3: Create features
    posts_data = []
    features_list = []
    
    for post in all_posts:
        features = f"{post['title']} {post.get('tag', '')} {clean_text(post['content'])}"
        features_list.append(features)
        posts_data.append({
                'id': post['id'],
                'slug': post['slug'],
                'title': post['title'],
                'subtitle': post.get('subtitle', ''),
                'thumbnailimage': post.get('thumbnailimage', ''),
                'similarity': round(similarity, 3)
            })
    
    # STEP 4: Pure NumPy TF-IDF
    tfidf_matrix, _ = simple_tfidf_vectorizer(features_list)
    
    # STEP 5: Find current post index
    current_idx = next(i for i, post in enumerate(posts_data) if post['id'] == post_id)
    
    # STEP 6: Cosine similarity
    cosine_scores = cosine_similarity_numpy(tfidf_matrix[[current_idx]], tfidf_matrix)[0]
    
    # STEP 7: Top 5 recommendations (exclude self)
    sim_scores = list(enumerate(cosine_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = []
    for idx, score in sim_scores:
        if score > 0.05:  # Lower threshold for NumPy version
            rec_post = posts_data[idx]
            recommendations.append({
                'id': rec_post['id'],
                'slug': rec_post['slug'],
                'title': rec_post['title'],
                'similarity': float(score)
            })
    
    return {
        "current_post": current_post['title'],
        "recommendations": recommendations,
        "count": len(recommendations)
    }

@app.get("/recommend/{post_id}")
async def test_recommend(post_id: str):
    return await recommend_posts(RecommendRequest(id=post_id))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("REC_API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
