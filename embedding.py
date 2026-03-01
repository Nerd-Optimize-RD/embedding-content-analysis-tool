import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
import base64
from io import BytesIO
import requests
from flask import Flask, request, render_template_string, jsonify


class PrefixMiddleware:
    """WSGI middleware to serve the app under APPLICATION_ROOT (e.g. /embedding-content-analysis-tool)."""
    def __init__(self, app, prefix):
        self.app = app
        self.prefix = prefix.rstrip('/')

    def __call__(self, environ, start_response):
        if self.prefix:
            path = environ.get('PATH_INFO', '') or '/'
            if path == self.prefix or path.startswith(self.prefix + '/'):
                environ['SCRIPT_NAME'] = self.prefix
                environ['PATH_INFO'] = path[len(self.prefix):] or '/'
        return self.app(environ, start_response)
from google import genai
from dotenv import load_dotenv
import trafilatura

# Load environment variables
load_dotenv()

# Configure APIs
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "xxx")  # Get from environment variable
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "xxx")  # Get from environment variable
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")  # SerpAPI key for competitor search

# Initialize Gemini client (google.genai SDK)
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

app = Flask(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Configure Flask to use the custom encoder
app.json_encoder = NumpyEncoder

def get_embedding(text):
    """Get embedding from Google Gemini API (google.genai SDK)"""
    try:
        result = genai_client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text,
        )
        # Response has .embeddings (list); each item has .values
        emb = result.embeddings[0]
        embedding = getattr(emb, "values", None) or emb
        if hasattr(embedding, "__iter__") and not isinstance(embedding, (str, bytes)):
            return list(embedding)
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a random embedding for testing if API fails
        print("Using random embedding instead")
        return np.random.normal(0, 0.1, 3072).tolist()

def analyze_with_deepseek(embedding_data, content_snippet):
    """Get analysis from Deepseek API"""
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์การฝังตัวสำหรับ SEO และ NLP วิเคราะห์ข้อมูลการฝังตัวที่ให้มาเพื่อดึงข้อมูลเชิงลึกเกี่ยวกับคุณภาพเนื้อหา โครงสร้างความหมาย และโอกาสในการปรับปรุง SEO เน้นที่รูปแบบการกระตุ้น กลุ่มมิติ และตัวชี้วัดคุณภาพ ให้คำแนะนำที่ปฏิบัติได้จริง โปรดตอบเป็นภาษาไทย"
                },
                {
                    "role": "user",
                    "content": f"""วิเคราะห์ข้อมูลการฝังตัว 3,072 มิติจากเนื้อหาชิ้นนี้ เน้นที่ตัวชี้วัดคุณภาพ โครงสร้างความหมาย และผลกระทบต่อ SEO

ตัวอย่างเนื้อหา (18,500 ตัวอักษรแรก): 
{content_snippet[:18500]}...

สถิติข้อมูลการฝังตัว:
- จำนวนมิติ: {len(embedding_data)}
- ค่าเฉลี่ย: {np.mean(embedding_data):.6f}
- ส่วนเบี่ยงเบนมาตรฐาน: {np.std(embedding_data):.6f}
- ค่าต่ำสุด: {np.min(embedding_data):.6f} ที่มิติ {np.argmin(embedding_data)}
- ค่าสูงสุด: {np.max(embedding_data):.6f} ที่มิติ {np.argmax(embedding_data)}
- 5 มิติที่มีค่าสูงสุด: {sorted(range(len(embedding_data)), key=lambda i: abs(embedding_data[i]), reverse=True)[:5]}

โปรดให้การวิเคราะห์ที่กระชับโดยเน้นที่:
1. การประเมินคุณภาพเนื้อหาตามรูปแบบการฝังตัว
2. กลุ่มมิติหลักและหน้าที่ทางความหมายที่น่าจะเป็น
3. คำแนะนำการปรับปรุง SEO ตามโครงสร้างการฝังตัว
4. จุดแข็งและจุดอ่อนของหัวข้อ

โปรดตอบเป็นภาษาไทยทั้งหมด"""
                }
            ],
            "temperature": 1,
            "max_tokens": 8000
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text content from Deepseek's response
        if "choices" in result and len(result["choices"]) > 0:
            message_content = result["choices"][0].get("message", {}).get("content", "")
            return message_content
        else:
            return "Error: Unexpected response format from Deepseek API."
            
    except Exception as e:
        print(f"Error getting Content Analysis: {e}")
        return "Error getting analysis from Deepseek. Please check your API key and try again."

def plot_embedding_overview(embedding):
    """Create overview plot of embedding values"""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(embedding)), embedding)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Embedding Values Across All 3k Dimensions')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_top_dimensions(embedding):
    """Plot top dimensions by magnitude"""
    # Get indices of top 20 dimensions by magnitude
    top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:20]
    top_values = [embedding[i] for i in top_indices]
    
    plt.figure(figsize=(12, 6))
    colors = ['blue' if v >= 0 else 'red' for v in top_values]
    plt.bar(range(len(top_indices)), top_values, color=colors)
    plt.xticks(range(len(top_indices)), top_indices, rotation=45)
    plt.title('Top 20 Dimensions by Magnitude')
    plt.xlabel('Dimension Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_dimension_clusters(embedding):
    """Plot dimension clusters heatmap"""
    # Reshape embedding to highlight patterns
    embedding_reshaped = np.array(embedding).reshape(64, 48)
    
    plt.figure(figsize=(12, 8))
    # Create a custom colormap from blue to white to red
    cmap = LinearSegmentedColormap.from_list('BrBG', ['blue', 'white', 'red'], N=256)
    plt.imshow(embedding_reshaped, cmap=cmap, aspect='auto')
    plt.colorbar(label='Activation Value')
    plt.title('Embedding Clusters Heatmap (Reshaped to 64x48)')
    plt.xlabel('Dimension Group')
    plt.ylabel('Dimension Group')
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_pca(embedding):
    """Plot PCA visualization of embedding dimensions"""
    # Create a 2D array where each row is a segment of the original embedding
    segment_size = 256
    num_segments = len(embedding) // segment_size
    data_matrix = np.zeros((num_segments, segment_size))
    
    # Fill the matrix with segments
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size
        data_matrix[i] = embedding[start:end]
    
    # Apply PCA
    if num_segments > 1:
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(data_matrix)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_results[:, 0], pca_results[:, 1])
        
        # Label each point with its segment range
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size - 1
            plt.annotate(f"{start}-{end}", 
                         (pca_results[i, 0], pca_results[i, 1]),
                         fontsize=8)
        
        plt.title('PCA of Embedding Segments')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, alpha=0.3)
    else:
        # If we don't have enough segments, create a simpler visualization
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "Not enough segments for PCA visualization", 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_activation_histogram(embedding):
    """Plot histogram of embedding activation values"""
    plt.figure(figsize=(10, 6))
    plt.hist(embedding, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.title('Distribution of Embedding Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def analyze_embedding(embedding):
    """Analyze embedding for key metrics"""
    embedding = np.array(embedding)  # Convert to numpy array for easier processing
    abs_embedding = np.abs(embedding)
    
    # Calculate key metrics - CONVERT NUMPY TYPES TO PYTHON NATIVE TYPES
    metrics = {
        "dimension_count": int(len(embedding)),
        "mean_value": float(np.mean(embedding)),
        "std_dev": float(np.std(embedding)),
        "min_value": float(np.min(embedding)),
        "min_dimension": int(np.argmin(embedding)),
        "max_value": float(np.max(embedding)),
        "max_dimension": int(np.argmax(embedding)),
        "median_value": float(np.median(embedding)),
        "positive_count": int(np.sum(embedding > 0)),
        "negative_count": int(np.sum(embedding < 0)),
        "zero_count": int(np.sum(embedding == 0)),
        "abs_mean": float(np.mean(abs_embedding)),
        "significant_dims": int(np.sum(abs_embedding > 0.1))
    }
    
    # Find activation clusters
    significant_threshold = 0.1
    significant_dims = np.where(abs_embedding > significant_threshold)[0]
    
    # Find clusters (dimensions that are close to each other)
    clusters = []
    if len(significant_dims) > 0:
        current_cluster = [int(significant_dims[0])]  # Convert to int
        
        for i in range(1, len(significant_dims)):
            if significant_dims[i] - significant_dims[i-1] <= 5:  # If dimensions are close
                current_cluster.append(int(significant_dims[i]))  # Convert to int
            else:
                if len(current_cluster) > 0:
                    clusters.append(current_cluster)
                current_cluster = [int(significant_dims[i])]  # Convert to int
        
        if len(current_cluster) > 0:
            clusters.append(current_cluster)
    
    # Filter to meaningful clusters (more than 1 dimension)
    clusters = [c for c in clusters if len(c) > 1]
    
    # Format clusters for display
    cluster_info = []
    for i, cluster in enumerate(clusters):
        values = [float(embedding[dim]) for dim in cluster]  # Convert to float
        cluster_info.append({
            "id": i+1,
            "dimensions": [int(d) for d in cluster],  # Convert to int
            "start_dim": int(min(cluster)),
            "end_dim": int(max(cluster)),
            "size": int(len(cluster)),
            "avg_value": float(np.mean(values)),
            "max_value": float(np.max(values)),
            "max_dim": int(cluster[np.argmax(values)])
        })
    
    # Top dimensions by magnitude
    top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:10]
    top_dimensions = [{"dimension": int(idx), "value": float(embedding[idx])} for idx in top_indices]
    
    return {
        "metrics": metrics,
        "clusters": cluster_info,
        "top_dimensions": top_dimensions
    }

# ============================================================
# NEW FUNCTIONS FOR COMPETITOR ANALYSIS
# ============================================================

def parse_markdown_to_chunks(text):
    """
    แบ่ง content ตามหัวข้อ Markdown (# H1, ## H2, ### H3)
    Returns: list of {"level": 1-3, "title": str, "content": str}
    """
    chunks = []
    lines = text.split('\n')
    current_chunk = {"level": 0, "title": "Introduction", "content": ""}

    for line in lines:
        # Check for headers
        h3_match = re.match(r'^###\s+(.+)$', line)
        h2_match = re.match(r'^##\s+(.+)$', line)
        h1_match = re.match(r'^#\s+(.+)$', line)

        if h1_match:
            if current_chunk["content"].strip():
                chunks.append(current_chunk)
            current_chunk = {"level": 1, "title": h1_match.group(1).strip(), "content": ""}
        elif h2_match:
            if current_chunk["content"].strip():
                chunks.append(current_chunk)
            current_chunk = {"level": 2, "title": h2_match.group(1).strip(), "content": ""}
        elif h3_match:
            if current_chunk["content"].strip():
                chunks.append(current_chunk)
            current_chunk = {"level": 3, "title": h3_match.group(1).strip(), "content": ""}
        else:
            current_chunk["content"] += line + "\n"

    # Add last chunk
    if current_chunk["content"].strip():
        chunks.append(current_chunk)

    return chunks

def extract_plain_text(chunks):
    """รวม chunks เป็น plain text สำหรับ embedding"""
    text_parts = []
    for chunk in chunks:
        if chunk["title"]:
            text_parts.append(chunk["title"])
        if chunk["content"]:
            text_parts.append(chunk["content"].strip())
    return "\n\n".join(text_parts)

def search_google_top3(keyword):
    """
    ค้นหา Top 3 URLs จาก SerpAPI
    Returns: {"results": [{"url", "title", "snippet"}], "success": bool, "error"?: str}
    """
    if not SERPAPI_API_KEY or not SERPAPI_API_KEY.strip() or SERPAPI_API_KEY in ("xxx", "your_serpapi_key"):
        return {"results": [], "success": False, "error": "SERPAPI_API_KEY is not configured"}

    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": keyword,
            "api_key": SERPAPI_API_KEY,
            "num": 10,
            "gl": "th",
            "hl": "th",
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("organic_results", [])[:3]:
            url_val = item.get("link") or item.get("url") or ""
            if not url_val:
                continue
            results.append({
                "url": url_val,
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
            })
        results = results[:3]
        return {"results": results, "success": True}
    except Exception as e:
        print(f"Error searching SerpAPI: {e}")
        return {"results": [], "success": False, "error": str(e)}

def extract_web_content(url):
    """
    ใช้ Trafilatura ดึง main body content จาก URL
    Returns: {"content": str, "title": str, "success": bool}
    """
    try:
        # Download the page
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return {"content": "", "title": "", "success": False, "error": "Failed to download page"}

        # Extract main content
        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False
        )

        # Try to get title
        title = ""
        try:
            metadata = trafilatura.extract_metadata(downloaded)
            if metadata:
                title = metadata.title or ""
        except:
            pass

        if content:
            return {"content": content, "title": title, "success": True}
        else:
            return {"content": "", "title": title, "success": False, "error": "No content extracted"}

    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return {"content": "", "title": "", "success": False, "error": str(e)}

def calculate_cosine_sim(emb1, emb2):
    """คำนวณ cosine similarity ระหว่าง 2 embeddings"""
    emb1 = np.array(emb1).reshape(1, -1)
    emb2 = np.array(emb2).reshape(1, -1)
    return float(cosine_similarity(emb1, emb2)[0][0])

def analyze_embedding_differences(user_emb, competitor_embs, competitor_labels):
    """
    วิเคราะห์ความแตกต่างของ embeddings
    Returns dict with common dimensions, user gaps, user strengths, similarity scores
    """
    user_emb = np.array(user_emb)
    comp_embs = [np.array(e) for e in competitor_embs]

    # Calculate similarity scores
    similarity_scores = []
    for i, comp_emb in enumerate(comp_embs):
        sim = calculate_cosine_sim(user_emb, comp_emb)
        similarity_scores.append({
            "label": competitor_labels[i],
            "similarity": round(sim * 100, 2)
        })

    # Calculate average competitor embedding
    avg_comp_emb = np.mean(comp_embs, axis=0)

    # Find dimensions where user is significantly different
    # Use adaptive threshold based on data variance
    diff = user_emb - avg_comp_emb
    abs_diff = np.abs(diff)

    # Calculate adaptive threshold (use percentile-based approach)
    threshold = max(0.01, np.percentile(abs_diff, 75) * 0.5)  # More sensitive threshold

    # User gaps: dimensions where competitors have significantly higher values
    # Lower the secondary threshold for more results
    gap_indices = np.where((diff < -threshold) & (np.abs(avg_comp_emb) > 0.01))[0]
    user_gaps = []
    for idx in sorted(gap_indices, key=lambda i: diff[i])[:30]:  # Increased to 30
        user_gaps.append({
            "dimension": int(idx),
            "user_value": float(user_emb[idx]),
            "competitor_avg": float(avg_comp_emb[idx]),
            "difference": float(diff[idx])
        })

    # User strengths: dimensions where user has higher values
    strength_indices = np.where((diff > threshold) & (np.abs(user_emb) > 0.01))[0]
    user_strengths = []
    for idx in sorted(strength_indices, key=lambda i: diff[i], reverse=True)[:30]:  # Increased to 30
        user_strengths.append({
            "dimension": int(idx),
            "user_value": float(user_emb[idx]),
            "competitor_avg": float(avg_comp_emb[idx]),
            "difference": float(diff[idx])
        })

    # Common high activation dimensions (where all have high values)
    common_threshold = 0.08
    user_high = set(np.where(np.abs(user_emb) > common_threshold)[0])
    comp_high_sets = [set(np.where(np.abs(e) > common_threshold)[0]) for e in comp_embs]
    common_dims = user_high.intersection(*comp_high_sets)

    common_dimensions = []
    for idx in sorted(common_dims, key=lambda i: np.abs(user_emb[i]), reverse=True)[:20]:
        common_dimensions.append({
            "dimension": int(idx),
            "user_value": float(user_emb[idx]),
            "competitor_avg": float(avg_comp_emb[idx])
        })

    # Calculate overall coverage score
    all_comp_significant = set()
    for e in comp_embs:
        all_comp_significant.update(np.where(np.abs(e) > 0.05)[0])

    user_significant = set(np.where(np.abs(user_emb) > 0.05)[0])
    coverage = len(user_significant.intersection(all_comp_significant)) / max(len(all_comp_significant), 1)

    return {
        "similarity_scores": similarity_scores,
        "average_similarity": round(np.mean([s["similarity"] for s in similarity_scores]), 2),
        "user_gaps": user_gaps,
        "user_strengths": user_strengths,
        "common_dimensions": common_dimensions,
        "coverage_score": round(coverage * 100, 2),
        "gap_count": len(user_gaps),
        "strength_count": len(user_strengths)
    }

def analyze_competitors_with_deepseek(user_content, competitor_contents, analysis_data, keyword):
    """
    ใช้ DeepSeek วิเคราะห์เปรียบเทียบ:
    - Content Gap
    - จุดแข็ง-จุดอ่อน
    - คำแนะนำในการปรับปรุง
    """
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        # Prepare competitor snippets (first 3000 chars each)
        comp_snippets = []
        for i, content in enumerate(competitor_contents):
            comp_snippets.append(f"คู่แข่ง {i+1} (ตัวอย่างเนื้อหา):\n{content[:3000]}...")

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """คุณเป็นผู้เชี่ยวชาญด้าน SEO Content Analysis โปรดวิเคราะห์เปรียบเทียบบทความของผู้ใช้กับคู่แข่งที่ติดอันดับ Top 3 บน Google

ให้การวิเคราะห์ที่เป็นประโยชน์และปฏิบัติได้จริง เน้นที่:
1. Content Gap - เนื้อหาที่คู่แข่งมีแต่ผู้ใช้ยังไม่มี
2. จุดแข็งของบทความผู้ใช้
3. จุดที่ควรปรับปรุง
4. คำแนะนำเฉพาะเจาะจงในการปรับปรุงบทความ

โปรดตอบเป็นภาษาไทยทั้งหมด และจัดรูปแบบให้อ่านง่าย"""
                },
                {
                    "role": "user",
                    "content": f"""วิเคราะห์เปรียบเทียบบทความสำหรับ Keyword: "{keyword}"

📝 บทความของผู้ใช้ (ตัวอย่างเนื้อหา):
{user_content[:5000]}...

📊 ข้อมูลการวิเคราะห์ Embedding:
- ความคล้ายคลึงเฉลี่ยกับคู่แข่ง: {analysis_data['average_similarity']}%
- Coverage Score: {analysis_data['coverage_score']}%
- จำนวน Content Gap ที่พบ: {analysis_data['gap_count']} มิติ
- จำนวนจุดแข็ง: {analysis_data['strength_count']} มิติ

🏆 เนื้อหาคู่แข่ง Top 3:
{chr(10).join(comp_snippets)}

โปรดให้การวิเคราะห์ที่ครอบคลุม:
1. 📌 Content Gap Analysis - เนื้อหาสำคัญที่ขาดหายไป
2. 💪 จุดแข็งของบทความ - สิ่งที่ทำได้ดี
3. ⚠️ จุดที่ควรปรับปรุง - สิ่งที่ต้องแก้ไข
4. 📝 คำแนะนำเฉพาะ - ขั้นตอนที่ควรทำเพื่อปรับปรุงบทความ
5. 🎯 สรุป - ภาพรวมและลำดับความสำคัญ"""
                }
            ],
            "temperature": 0.7,
            "max_tokens": 8000
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("message", {}).get("content", "")
        else:
            return "Error: Unexpected response format from DeepSeek API."

    except Exception as e:
        print(f"Error getting DeepSeek competitor analysis: {e}")
        return f"Error getting analysis from DeepSeek: {str(e)}"

# ============================================================
# NEW VISUALIZATION FUNCTIONS FOR COMPETITOR COMPARISON
# ============================================================

def plot_comparison_scatter(user_embedding, competitor_embeddings, competitor_labels):
    """
    สร้าง Cluster Graph แยกสีตามเว็บไซต์
    แบ่ง embedding แต่ละบทความเป็น segments แล้ว plot เหมือน PCA Visualization
    ทำให้เห็นความคล้าย/แตกต่างของแต่ละส่วนชัดเจน
    """
    from scipy.spatial import ConvexHull

    # Define segment size (split 3072 dimensions into segments)
    segment_size = 256
    num_segments = len(user_embedding) // segment_size

    # Colors: Brand palette (Calming blue, Navy black, secondary teal, secondary red)
    colors = ['#4d62a7', '#44a2a5', '#f7991a', '#bf415c']
    all_labels = ["Your Article"] + competitor_labels
    all_embeddings = [user_embedding] + competitor_embeddings

    # Collect all segments for PCA
    all_segments = []
    segment_sources = []  # Which source each segment belongs to
    segment_labels = []   # Label for each segment (e.g., "0-255")

    for source_idx, embedding in enumerate(all_embeddings):
        for seg_idx in range(num_segments):
            start = seg_idx * segment_size
            end = start + segment_size
            segment = embedding[start:end]
            all_segments.append(segment)
            segment_sources.append(source_idx)
            segment_labels.append(f"{start}-{end-1}")

    # Apply PCA to all segments
    segments_array = np.array(all_segments)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(segments_array)

    plt.figure(figsize=(14, 10))

    # Plot each source with different color
    for source_idx, source_label in enumerate(all_labels):
        # Get indices for this source
        indices = [i for i, s in enumerate(segment_sources) if s == source_idx]
        x = pca_results[indices, 0]
        y = pca_results[indices, 1]

        plt.scatter(x, y, c=colors[source_idx], s=80, label=source_label,
                   alpha=0.7, edgecolors='white', linewidths=1)

        # Add annotations for segments
        for i, idx in enumerate(indices):
            plt.annotate(segment_labels[idx], (pca_results[idx, 0], pca_results[idx, 1]),
                        fontsize=6, ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        alpha=0.7, color=colors[source_idx])

    # Draw convex hull around each cluster (visual aid to see overlap)
    for source_idx, source_label in enumerate(all_labels):
        indices = [i for i, s in enumerate(segment_sources) if s == source_idx]
        if len(indices) >= 3:
            points = pca_results[indices]
            try:
                hull = ConvexHull(points)
                hull_points = np.append(hull.vertices, hull.vertices[0])
                plt.plot(points[hull_points, 0], points[hull_points, 1],
                        color=colors[source_idx], alpha=0.3, linestyle='--', linewidth=1)
                plt.fill(points[hull_points, 0], points[hull_points, 1],
                        color=colors[source_idx], alpha=0.1)
            except:
                pass  # Skip if convex hull fails

    plt.title('Content Similarity Map (Embedding Segments)', fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_chunk_scatter(all_chunk_data):
    """
    สร้าง Cluster Graph แสดง embedding ของแต่ละ chunk/section
    แยกสีตามแหล่งที่มา (Your Article, Competitor 1, 2, 3)

    Parameters:
    all_chunk_data: list of dict with keys:
        - source: str (source label)
        - chunks: list of dict with 'title', 'embedding'
        - color: str (hex color)
    """
    # Collect all embeddings and metadata
    all_embeddings = []
    all_labels = []
    all_colors = []
    all_sources = []

    source_colors = ['#4d62a7', '#44a2a5', '#f7991a', '#bf415c']  # Calming blue, Teal, Orange, Red

    for i, source_data in enumerate(all_chunk_data):
        source = source_data['source']
        chunks = source_data['chunks']
        color = source_colors[i % len(source_colors)]

        for chunk in chunks:
            all_embeddings.append(chunk['embedding'])
            # Short label: source + chunk title (truncated)
            chunk_title = chunk.get('title', 'Section')[:20]
            all_labels.append(f"{chunk_title}")
            all_colors.append(color)
            all_sources.append(source)

    if len(all_embeddings) < 2:
        # Not enough data points for PCA
        plt.figure(figsize=(12, 10))
        plt.text(0.5, 0.5, 'Not enough chunks to create visualization',
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    # Apply PCA
    embeddings_array = np.array(all_embeddings)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings_array)

    plt.figure(figsize=(14, 10))

    # Plot points grouped by source for legend
    unique_sources = []
    for i, source_data in enumerate(all_chunk_data):
        unique_sources.append(source_data['source'])

    # Create scatter for each source
    for source_idx, source in enumerate(unique_sources):
        color = source_colors[source_idx % len(source_colors)]
        indices = [i for i, s in enumerate(all_sources) if s == source]

        if indices:
            x = pca_results[indices, 0]
            y = pca_results[indices, 1]
            plt.scatter(x, y, c=color, s=150, label=source, alpha=0.7,
                       edgecolors='white', linewidths=1.5)

            # Add annotations for each point
            for idx in indices:
                plt.annotate(all_labels[idx],
                           (pca_results[idx, 0], pca_results[idx, 1]),
                           fontsize=7, ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points',
                           alpha=0.8)

    plt.title('Content Chunk Similarity Map (PCA)', fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_similarity_scorecard(analysis_data):
    """
    สร้าง Scorecard แสดง similarity scores และ metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Average Similarity Gauge
    ax1 = axes[0]
    similarity = max(0, min(100, analysis_data['average_similarity']))  # Clamp 0-100
    colors_gauge = ['#bf415c', '#f7991a', '#e8cb5a', '#44a2a5']
    color_idx = min(int(similarity / 25), 3)

    pie_val = max(0.01, similarity)  # Ensure non-zero for pie
    pie_remainder = max(0.01, 100 - similarity)
    wedges, _ = ax1.pie([pie_val, pie_remainder], colors=[colors_gauge[color_idx], '#e5e7eb'],
                        startangle=90, counterclock=False)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax1.add_patch(centre_circle)
    ax1.text(0, 0, f'{similarity}%', ha='center', va='center', fontsize=24, fontweight='bold')
    ax1.text(0, -0.3, 'Similarity', ha='center', va='center', fontsize=10, color='gray')
    ax1.set_title('Average Similarity', fontsize=12, fontweight='bold')

    # Coverage Score Gauge
    ax2 = axes[1]
    coverage = max(0, min(100, analysis_data['coverage_score']))  # Clamp 0-100
    color_idx = min(int(coverage / 25), 3)

    pie_val = max(0.01, coverage)  # Ensure non-zero for pie
    pie_remainder = max(0.01, 100 - coverage)
    wedges, _ = ax2.pie([pie_val, pie_remainder], colors=[colors_gauge[color_idx], '#e5e7eb'],
                        startangle=90, counterclock=False)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax2.add_patch(centre_circle)
    ax2.text(0, 0, f'{coverage}%', ha='center', va='center', fontsize=24, fontweight='bold')
    ax2.text(0, -0.3, 'Coverage', ha='center', va='center', fontsize=10, color='gray')
    ax2.set_title('Topic Coverage', fontsize=12, fontweight='bold')

    # Gap/Strength Bar
    ax3 = axes[2]
    gaps = analysis_data['gap_count']
    strengths = analysis_data['strength_count']

    bars = ax3.barh(['Gaps', 'Strengths'], [gaps, strengths],
                    color=['#bf415c', '#44a2a5'], height=0.5)
    ax3.set_xlim(0, max(gaps, strengths) * 1.3)
    ax3.set_title('Gap vs Strength', fontsize=12, fontweight='bold')

    for bar, val in zip(bars, [gaps, strengths]):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=12, fontweight='bold')

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_radar_chart(user_embedding, competitor_embeddings, competitor_labels):
    """
    Radar chart เปรียบเทียบ key metrics
    """
    # Calculate metrics for each article
    def calc_metrics(emb):
        emb = np.array(emb)
        return {
            'Content Depth': min(np.std(emb) * 100, 100),
            'Topic Coverage': min(np.sum(np.abs(emb) > 0.05) / 30.72, 100),  # % of 3072
            'Semantic Richness': min(np.mean(np.abs(emb)) * 500, 100),
            'Activation Strength': min(np.max(np.abs(emb)) * 100, 100),
            'Balance': 100 - min(np.abs(np.sum(emb > 0) - np.sum(emb < 0)) / 30.72, 100)
        }

    user_metrics = calc_metrics(user_embedding)
    comp_metrics = [calc_metrics(e) for e in competitor_embeddings]
    avg_comp_metrics = {k: np.mean([m[k] for m in comp_metrics]) for k in user_metrics.keys()}

    # Radar chart
    categories = list(user_metrics.keys())
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # User values
    user_values = list(user_metrics.values())
    user_values += user_values[:1]
    ax.plot(angles, user_values, 'o-', linewidth=2, label='Your Article', color='#4d62a7')
    ax.fill(angles, user_values, alpha=0.25, color='#4d62a7')

    # Average competitor values
    comp_values = list(avg_comp_metrics.values())
    comp_values += comp_values[:1]
    ax.plot(angles, comp_values, 'o-', linewidth=2, label='Competitors Avg', color='#44a2a5')
    ax.fill(angles, comp_values, alpha=0.25, color='#44a2a5')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Content Quality Comparison', fontsize=14, fontweight='bold', y=1.08)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_comparison_heatmap(user_embedding, competitor_embeddings, competitor_labels):
    """
    Heatmap แสดงค่า embedding ของทุกบทความเทียบกัน
    """
    all_embeddings = [user_embedding] + competitor_embeddings
    all_labels = ["Your Article"] + competitor_labels

    # Get top 50 most variable dimensions across all embeddings
    embeddings_array = np.array(all_embeddings)
    variance = np.var(embeddings_array, axis=0)
    top_dims = np.argsort(variance)[-50:][::-1]

    # Extract values for top dimensions
    heatmap_data = embeddings_array[:, top_dims]

    plt.figure(figsize=(14, 6))

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('custom', ['#4d62a7', 'white', '#bf415c'])

    sns.heatmap(heatmap_data, cmap=cmap, center=0,
                yticklabels=all_labels,
                xticklabels=[f'D{d}' for d in top_dims],
                cbar_kws={'label': 'Activation Value'})

    plt.title('Top 50 Variable Dimensions Across Articles', fontsize=14, fontweight='bold')
    plt.xlabel('Dimension Index')
    plt.ylabel('Article')
    plt.xticks(rotation=45, ha='right', fontsize=7)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_dimension_comparison(user_embedding, competitor_embeddings, top_n=20):
    """
    Bar chart เปรียบเทียบ Top N dimensions
    แสดงค่าของ user vs average competitor
    """
    user_emb = np.array(user_embedding)
    avg_comp = np.mean(competitor_embeddings, axis=0)

    # Find dimensions with biggest differences
    diff = user_emb - avg_comp
    top_diff_indices = np.argsort(np.abs(diff))[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(top_diff_indices))
    width = 0.35

    user_vals = [user_emb[i] for i in top_diff_indices]
    comp_vals = [avg_comp[i] for i in top_diff_indices]

    bars1 = ax.bar(x - width/2, user_vals, width, label='Your Article', color='#4d62a7', alpha=0.8)
    bars2 = ax.bar(x + width/2, comp_vals, width, label='Competitors Avg', color='#44a2a5', alpha=0.8)

    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Activation Value')
    ax.set_title(f'Top {top_n} Dimensions with Biggest Differences', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{i}' for i in top_diff_indices], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_similarity_matrix(user_embedding, competitor_embeddings, competitor_labels):
    """
    แสดง Similarity Matrix ระหว่างทุกบทความ
    """
    all_embeddings = [user_embedding] + competitor_embeddings
    all_labels = ["Your Article"] + competitor_labels

    # Calculate pairwise similarities
    n = len(all_embeddings)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sim_matrix[i][j] = calculate_cosine_sim(all_embeddings[i], all_embeddings[j]) * 100

    plt.figure(figsize=(8, 6))

    sns.heatmap(sim_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=all_labels, yticklabels=all_labels,
                vmin=0, vmax=100, cbar_kws={'label': 'Similarity %'})

    plt.title('Content Similarity Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# HTML template (single page application)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Content Analysis Tool</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Mitr:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'navy': '#1d252d',
                        'calming': '#4d62a7',
                        'smoky': '#dfe6ef',
                        'eggshell': '#d9d4c8',
                    },
                    fontFamily: {
                        'poppins': ['Poppins', 'sans-serif'],
                        'mitr': ['Mitr', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    <style>
        * { font-family: 'Poppins', 'Mitr', sans-serif; }
        :lang(th), [lang="th"] { font-family: 'Mitr', 'Poppins', sans-serif; }
        body { background: #1d252d; min-height: 100vh; }
        .loading { display: inline-block; width: 50px; height: 50px; border: 3px solid rgba(223,230,239,.3); border-radius: 50%; border-top-color: #4d62a7; animation: spin 1s ease-in-out infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .card { background: rgba(255, 255, 255, 0.97); backdrop-filter: blur(10px); border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15); }
        .btn-primary { background: #4d62a7; color: #ffffff; font-weight: 600; padding: 12px 32px; border-radius: 8px; border: none; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(77, 98, 167, 0.4); }
        .btn-primary:hover { background: #3d5097; transform: translateY(-2px); box-shadow: 0 6px 20px rgba(77, 98, 167, 0.6); }
        .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .btn-secondary { background: #dfe6ef; color: #1d252d; font-weight: 600; padding: 10px 24px; border-radius: 8px; border: 2px solid #dfe6ef; cursor: pointer; transition: all 0.3s ease; }
        .btn-secondary:hover { border-color: #4d62a7; background: #eef2f7; }
        .input-field { border: 2px solid #dfe6ef; border-radius: 8px; transition: all 0.3s ease; width: 100%; padding: 12px 16px; color: #1d252d; }
        .input-field:focus { border-color: #4d62a7; outline: none; box-shadow: 0 0 0 3px rgba(77, 98, 167, 0.15); }
        .metric-card { background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%); border: 1px solid #dfe6ef; border-radius: 12px; padding: 20px; }
        .section-title { color: #1d252d; font-weight: 700; font-size: 1.5rem; margin-bottom: 1.5rem; }
        .tab-btn { padding: 12px 24px; font-weight: 600; border-radius: 8px 8px 0 0; border: none; cursor: pointer; transition: all 0.3s ease; }
        .tab-btn.active { background: rgba(255,255,255,0.97); color: #4d62a7; }
        .tab-btn:not(.active) { background: rgba(77,98,167,0.5); color: #dfe6ef; }
        .tab-btn:not(.active):hover { background: rgba(77,98,167,0.7); color: #ffffff; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .competitor-card { background: #f9fafb; border: 2px solid #dfe6ef; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
        .competitor-card.success { border-color: #44a2a5; background: #f0faf9; }
        .competitor-card.error { border-color: #bf415c; background: #fef2f4; }
        .status-badge { display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
        .status-badge.success { background: #e0f0ef; color: #2a7a7d; }
        .status-badge.error { background: #fde8ec; color: #991b3b; }
        .status-badge.pending { background: #fdf3e0; color: #92400e; }
        .progress-bar { height: 4px; background: #dfe6ef; border-radius: 2px; overflow: hidden; }
        .progress-bar-fill { height: 100%; background: linear-gradient(135deg, #4d62a7 0%, #1d252d 100%); transition: width 0.5s ease; }
        /* HTML Analysis Styles - Enhanced for better readability */
        .markdown-content { 
            line-height: 1.9; 
            color: #374151; 
            font-size: 1rem;
            background: #ffffff;
            padding: 2rem;
            border-radius: 12px;
        }
        .markdown-content h1 { 
            font-size: 2rem; 
            font-weight: 800; 
            color: #1f2937; 
            margin: 2rem 0 1.5rem 0; 
            padding-bottom: 0.75rem; 
            border-bottom: 3px solid #4d62a7;
            letter-spacing: -0.02em;
        }
        .markdown-content h2 { 
            font-size: 1.5rem; 
            font-weight: 700; 
            color: #374151; 
            margin: 1.75rem 0 1rem 0;
            padding: 0.75rem 0;
            border-bottom: 2px solid #e5e7eb;
        }
        .markdown-content h3 { 
            font-size: 1.25rem; 
            font-weight: 600; 
            color: #4b5563; 
            margin: 1.5rem 0 0.75rem 0;
            padding-left: 0.5rem;
            border-left: 4px solid #4d62a7;
        }
        .markdown-content h4 { 
            font-size: 1.1rem; 
            font-weight: 600; 
            color: #6b7280; 
            margin: 1.25rem 0 0.5rem 0;
        }
        .markdown-content p { 
            margin: 1rem 0; 
            text-align: justify;
            word-spacing: 0.05em;
        }
        .markdown-content ul, .markdown-content ol { 
            margin: 1rem 0; 
            padding-left: 2rem; 
        }
        .markdown-content li { 
            margin: 0.6rem 0; 
            line-height: 1.7;
        }
        .markdown-content ul li { 
            list-style-type: disc; 
            list-style-position: outside;
        }
        .markdown-content ol li { 
            list-style-type: decimal; 
            list-style-position: outside;
        }
        .markdown-content ul ul, .markdown-content ol ol {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }
        .markdown-content strong { 
            font-weight: 700; 
            color: #1f2937;
            background: linear-gradient(120deg, rgba(77, 98, 167, 0.1) 0%, rgba(77, 98, 167, 0.1) 100%);
            padding: 0.1rem 0.2rem;
            border-radius: 3px;
        }
        .markdown-content em { 
            font-style: italic; 
            color: #4b5563;
        }
        .markdown-content code { 
            background: #f3f4f6; 
            padding: 0.25rem 0.5rem; 
            border-radius: 5px; 
            font-size: 0.9em; 
            color: #dc2626;
            font-family: 'Courier New', monospace;
            border: 1px solid #e5e7eb;
        }
        .markdown-content pre { 
            background: #1f2937; 
            color: #e5e7eb; 
            padding: 1.25rem; 
            border-radius: 10px; 
            overflow-x: auto; 
            margin: 1.25rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .markdown-content pre code { 
            background: none; 
            color: inherit; 
            padding: 0;
            border: none;
        }
        .markdown-content blockquote { 
            border-left: 5px solid #4d62a7; 
            padding: 1rem 1.5rem; 
            margin: 1.5rem 0; 
            background: #f9fafb;
            color: #4b5563; 
            font-style: italic;
            border-radius: 0 8px 8px 0;
        }
        .markdown-content table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 1.5rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            overflow: hidden;
        }
        .markdown-content th, .markdown-content td { 
            border: 1px solid #e5e7eb; 
            padding: 1rem; 
            text-align: left;
        }
        .markdown-content th { 
            background: linear-gradient(135deg, #4d62a7 0%, #1d252d 100%);
            color: #ffffff;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.875rem;
            letter-spacing: 0.05em;
        }
        .markdown-content td {
            background: #ffffff;
        }
        .markdown-content tr:nth-child(even) td {
            background: #f9fafb;
        }
        .markdown-content tr:hover td {
            background: #f3f4f6;
        }
        .markdown-content hr { 
            border: none; 
            border-top: 3px solid #e5e7eb; 
            margin: 2rem 0;
            border-radius: 2px;
        }
        .markdown-content a { 
            color: #4d62a7; 
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: all 0.2s ease;
        }
        .markdown-content a:hover { 
            color: #1d252d;
            border-bottom-color: #1d252d;
        }
        /* Special styling for analysis sections */
        .markdown-content > *:first-child {
            margin-top: 0;
        }
        .markdown-content > *:last-child {
            margin-bottom: 0;
        }
    </style>
</head>
<body>
    <div class="container mx-auto px-4 py-8 max-w-7xl">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold" style="color: #dfe6ef;">Embedding Content Analysis Tool</h1>
        </div>

        <!-- Tab Navigation -->
        <div class="flex gap-2 mb-0">
            <button class="tab-btn active" onclick="switchTab('single')">Single Analysis</button>
            <button class="tab-btn" onclick="switchTab('compare')">Competitor Comparison</button>
        </div>

        <!-- Single Analysis Tab -->
        <div id="single-tab" class="tab-content active">
            <div class="card p-8 rounded-tl-none">
                <h2 class="section-title">Content Input</h2>
                <form id="single-form" class="space-y-6">
                    <div>
                        <label class="block text-sm font-semibold text-gray-700 mb-2">Paste your content here:</label>
                        <textarea id="single-content" rows="10" class="input-field" placeholder="Enter the content you want to analyze..."></textarea>
                    </div>
                    <div class="flex justify-end">
                        <button type="submit" class="btn-primary">Analyze Content</button>
                    </div>
                </form>
            </div>
        </div>

        <!-- Competitor Comparison Tab -->
        <div id="compare-tab" class="tab-content">
            <div class="card p-8 rounded-tl-none">
                <h2 class="section-title">Competitor Analysis</h2>

                <!-- User Article Input -->
                <div class="mb-8">
                    <label class="block text-sm font-semibold text-gray-700 mb-2">
                        Your Article (ใช้ Markdown headers: # H1, ## H2, ### H3)
                    </label>
                    <textarea id="user-content" rows="8" class="input-field" placeholder="# หัวข้อหลัก (H1)
บทนำของบทความ...

## หัวข้อรอง (H2)
เนื้อหาในส่วนนี้...

### หัวข้อย่อย (H3)
รายละเอียดเพิ่มเติม..."></textarea>
                </div>

                <!-- Target Keyword -->
                <div class="mb-8">
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Target Keyword</label>
                    <input type="text" id="target-keyword" class="input-field" placeholder="เช่น มะเร็งลำไส้">
                </div>

                <!-- Competitor Data Source Toggle -->
                <div class="mb-6">
                    <label class="block text-sm font-semibold text-gray-700 mb-3">Competitor Data Source</label>
                    <div class="flex gap-4">
                        <label class="flex items-center cursor-pointer">
                            <input type="radio" name="comp-source" value="auto" checked class="mr-2" onchange="toggleCompetitorMode()">
                            <span class="font-medium">Auto SERP API</span>
                        </label>
                        <label class="flex items-center cursor-pointer">
                            <input type="radio" name="comp-source" value="manual" class="mr-2" onchange="toggleCompetitorMode()">
                            <span class="font-medium">Manual Input</span>
                        </label>
                    </div>
                </div>

                <!-- Auto Mode -->
                <div id="auto-mode" class="mb-8">
                    <div class="flex items-center gap-4 mb-4">
                        <button type="button" onclick="searchCompetitors()" class="btn-secondary" id="search-btn">
                            Search Top 3 Competitors
                        </button>
                        <span id="search-status" class="text-sm text-gray-500"></span>
                    </div>
                    <div id="search-results" class="space-y-3"></div>
                </div>

                <!-- Manual Mode -->
                <div id="manual-mode" class="hidden mb-8">
                    <div class="space-y-4">
                        <div class="competitor-card">
                            <label class="block text-sm font-semibold text-gray-700 mb-2">Competitor 1</label>
                            <textarea id="manual-comp-1" rows="4" class="input-field" placeholder="Paste competitor 1 content here..."></textarea>
                        </div>
                        <div class="competitor-card">
                            <label class="block text-sm font-semibold text-gray-700 mb-2">Competitor 2</label>
                            <textarea id="manual-comp-2" rows="4" class="input-field" placeholder="Paste competitor 2 content here..."></textarea>
                        </div>
                        <div class="competitor-card">
                            <label class="block text-sm font-semibold text-gray-700 mb-2">Competitor 3</label>
                            <textarea id="manual-comp-3" rows="4" class="input-field" placeholder="Paste competitor 3 content here..."></textarea>
                        </div>
                    </div>
                </div>

                <!-- Submit Button -->
                <div class="flex justify-end">
                    <button type="button" onclick="runComparison()" class="btn-primary" id="compare-btn">
                        Analyze & Compare
                    </button>
                </div>
            </div>
        </div>

        <!-- Loading Indicator -->
        <div id="loading-container" class="hidden flex flex-col items-center justify-center py-16">
            <div class="loading mb-6"></div>
            <p id="loading-text" class="text-white text-lg font-medium">Analyzing content... This may take a minute.</p>
            <div class="w-64 mt-4">
                <div class="progress-bar">
                    <div id="progress-fill" class="progress-bar-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>

        <!-- Single Analysis Results -->
        <div id="single-results" class="hidden space-y-8 mt-8">
            <div class="card p-6">
                <h2 class="section-title">Embedding Overview</h2>
                <img id="s-overview-chart" class="w-full h-auto rounded-lg" />
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="card p-6">
                    <h2 class="section-title">Top Dimensions</h2>
                    <img id="s-top-dimensions-chart" class="w-full h-auto rounded-lg" />
                </div>
                <div class="card p-6">
                    <h2 class="section-title">Activation Distribution</h2>
                    <img id="s-histogram-chart" class="w-full h-auto rounded-lg" />
                </div>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="card p-6">
                    <h2 class="section-title">Dimension Clusters</h2>
                    <img id="s-clusters-chart" class="w-full h-auto rounded-lg" />
                </div>
                <div class="card p-6">
                    <h2 class="section-title">PCA Visualization</h2>
                    <img id="s-pca-chart" class="w-full h-auto rounded-lg" />
                </div>
            </div>
            <div class="card p-6">
                <h2 class="section-title">Key Metrics</h2>
                <div id="s-metrics-container" class="grid grid-cols-1 md:grid-cols-3 gap-4"></div>
            </div>
            <div class="card p-6">
                <h2 class="section-title">Dimension Clusters</h2>
                <div id="s-clusters-container" class="space-y-4"></div>
            </div>
            <div class="card p-6">
                <h2 class="section-title">Content Analysis</h2>
                <div id="s-deepseek-analysis" class="markdown-content"></div>
            </div>
        </div>

        <!-- Comparison Results -->
        <div id="compare-results" class="hidden space-y-8 mt-8">
            <!-- Scorecard -->
            <div class="card p-6">
                <h2 class="section-title">Scorecard Overview</h2>
                <img id="c-scorecard-chart" class="w-full h-auto rounded-lg" />
            </div>

            <!-- Chunk-Level Similarity Map (Main Visualization) -->
            <div class="card p-6">
                <h2 class="section-title">📊 Content Chunk Similarity Map</h2>
                <p class="text-gray-600 text-sm mb-4">แสดง embedding ของแต่ละ section/chunk ในบทความ เพื่อเปรียบเทียบความคล้ายคลึงของเนื้อหาในแต่ละส่วน</p>
                <img id="c-chunk-scatter-chart" class="w-full h-auto rounded-lg" />
            </div>

            <!-- Comparison Charts Row 1 -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="card p-6">
                    <h2 class="section-title">Embedding Segments Map</h2>
                    <p class="text-gray-500 text-xs mb-2">แบ่ง embedding เป็น segments (256 dims) เพื่อดูความ overlap/แตกต่าง • พื้นที่ทับซ้อน = เนื้อหาคล้ายกัน</p>
                    <img id="c-scatter-chart" class="w-full h-auto rounded-lg" />
                </div>
                <div class="card p-6">
                    <h2 class="section-title">Similarity Matrix</h2>
                    <img id="c-similarity-matrix-chart" class="w-full h-auto rounded-lg" />
                </div>
            </div>

            <!-- Comparison Charts Row 2 -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="card p-6">
                    <h2 class="section-title">Quality Comparison (Radar)</h2>
                    <img id="c-radar-chart" class="w-full h-auto rounded-lg" />
                </div>
                <div class="card p-6">
                    <h2 class="section-title">Top Dimension Differences</h2>
                    <img id="c-dimension-chart" class="w-full h-auto rounded-lg" />
                </div>
            </div>

            <!-- Heatmap -->
            <div class="card p-6">
                <h2 class="section-title">Embedding Comparison Heatmap</h2>
                <img id="c-heatmap-chart" class="w-full h-auto rounded-lg" />
            </div>

            <!-- Similarity Scores -->
            <div class="card p-6">
                <h2 class="section-title">Similarity Scores</h2>
                <div id="c-similarity-scores" class="grid grid-cols-1 md:grid-cols-3 gap-4"></div>
            </div>

            <!-- Content Gap Analysis -->
            <div class="card p-6">
                <h2 class="section-title">Content Gap & Strengths</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h3 class="text-lg font-semibold text-red-600 mb-3">Content Gaps (Areas to Improve)</h3>
                        <div id="c-gaps-list" class="space-y-2"></div>
                    </div>
                    <div>
                        <h3 class="text-lg font-semibold text-green-600 mb-3">Your Strengths</h3>
                        <div id="c-strengths-list" class="space-y-2"></div>
                    </div>
                </div>
            </div>

            <!-- Your Article Analysis -->
            <div class="card p-6">
                <h2 class="section-title">Your Article - Detailed Analysis</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div>
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">Embedding Overview</h3>
                        <img id="c-overview-chart" class="w-full h-auto rounded-lg" />
                    </div>
                    <div>
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">Top Dimensions</h3>
                        <img id="c-top-dimensions-chart" class="w-full h-auto rounded-lg" />
                    </div>
                </div>
            </div>

            <!-- Content Analysis -->
            <div class="card p-6">
                <h2 class="section-title">DeepSeek Competitor Analysis</h2>
                <div id="c-deepseek-analysis" class="markdown-content"></div>
            </div>
        </div>
    </div>

    <script>
        const BASE = '{{ request.script_root or "" }}';
        // Store competitor data
        let competitorData = [];

        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            if (tab === 'single') {
                document.querySelector('.tab-btn:first-child').classList.add('active');
                document.getElementById('single-tab').classList.add('active');
            } else {
                document.querySelector('.tab-btn:last-child').classList.add('active');
                document.getElementById('compare-tab').classList.add('active');
            }

            // Hide results when switching tabs
            document.getElementById('single-results').classList.add('hidden');
            document.getElementById('compare-results').classList.add('hidden');
        }

        function toggleCompetitorMode() {
            const isAuto = document.querySelector('input[name="comp-source"]:checked').value === 'auto';
            document.getElementById('auto-mode').classList.toggle('hidden', !isAuto);
            document.getElementById('manual-mode').classList.toggle('hidden', isAuto);
        }

        async function searchCompetitors() {
            const keyword = document.getElementById('target-keyword').value.trim();
            if (!keyword) {
                alert('Please enter a target keyword first.');
                return;
            }

            const btn = document.getElementById('search-btn');
            const status = document.getElementById('search-status');
            btn.disabled = true;
            status.textContent = 'Searching...';

            try {
                const response = await fetch(BASE + '/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ keyword })
                });
                const data = await response.json();

                if (data.success && data.results.length > 0) {
                    status.textContent = `Found ${data.results.length} results`;
                    displaySearchResults(data.results);
                } else {
                    status.textContent = 'No results found. Try manual input.';
                    document.getElementById('search-results').innerHTML = `
                        <div class="text-red-600 text-sm">
                            ${data.error || 'No results found. Please try manual input mode.'}
                        </div>`;
                }
            } catch (error) {
                status.textContent = 'Search failed';
                console.error(error);
            }

            btn.disabled = false;
        }

        function truncateUrl(url, maxLen) {
            if (!url) return '';
            maxLen = maxLen || 55;
            return url.length > maxLen ? url.substring(0, maxLen) + '...' : url;
        }
        function escapeHtml(s) {
            if (!s) return '';
            const div = document.createElement('div');
            div.textContent = s;
            return div.innerHTML;
        }

        function displaySearchResults(results) {
            competitorData = [];
            const container = document.getElementById('search-results');
            container.innerHTML = '';

            results.forEach((result, i) => {
                const displayUrl = truncateUrl(result.url, 55);
                const safeUrl = escapeHtml(result.url);
                const safeDisplayUrl = escapeHtml(displayUrl);
                const card = document.createElement('div');
                card.className = 'competitor-card';
                card.id = `comp-card-${i}`;
                card.innerHTML = `
                    <div class="flex justify-between items-start mb-2">
                        <div class="flex-1 min-w-0">
                            <h4 class="font-semibold text-gray-900">${i + 1}. ${escapeHtml(result.title)}</h4>
                            <a href="${safeUrl}" target="_blank" rel="noopener noreferrer" class="text-sm text-blue-600 hover:underline block truncate max-w-full" title="${safeUrl}">${safeDisplayUrl}</a>
                        </div>
                        <span id="comp-status-${i}" class="status-badge pending">Pending</span>
                    </div>
                    <p class="text-sm text-gray-600 mb-3">${escapeHtml(result.snippet)}</p>
                    <button onclick="extractContent(${i})" class="btn-secondary text-sm" id="extract-btn-${i}">
                        Extract Content
                    </button>
                `;
                container.appendChild(card);

                competitorData.push({
                    url: result.url,
                    title: result.title,
                    content: null,
                    source: 'auto'
                });
            });
        }

        async function extractContent(index) {
            const url = competitorData[index] && competitorData[index].url;
            if (!url) return;
            const btn = document.getElementById(`extract-btn-${index}`);
            const status = document.getElementById(`comp-status-${index}`);
            const card = document.getElementById(`comp-card-${index}`);

            btn.disabled = true;
            btn.textContent = 'Extracting...';
            status.className = 'status-badge pending';
            status.textContent = 'Extracting...';

            try {
                const response = await fetch(BASE + '/extract', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url })
                });
                const data = await response.json();

                if (data.success && data.content) {
                    competitorData[index].content = data.content;
                    competitorData[index].title = data.title || competitorData[index].title;

                    status.className = 'status-badge success';
                    status.textContent = `${data.content.length} chars`;
                    card.classList.add('success');
                    card.classList.remove('error');
                    btn.textContent = 'Done';
                } else {
                    status.className = 'status-badge error';
                    status.textContent = 'Failed';
                    card.classList.add('error');
                    card.classList.remove('success');
                    btn.textContent = 'Retry';
                    btn.disabled = false;
                }
            } catch (error) {
                status.className = 'status-badge error';
                status.textContent = 'Error';
                card.classList.add('error');
                btn.textContent = 'Retry';
                btn.disabled = false;
                console.error(error);
            }
        }

        function showLoading(text) {
            document.getElementById('loading-container').classList.remove('hidden');
            document.getElementById('loading-text').textContent = text;
            document.getElementById('single-results').classList.add('hidden');
            document.getElementById('compare-results').classList.add('hidden');
        }

        function hideLoading() {
            document.getElementById('loading-container').classList.add('hidden');
        }

        function updateProgress(percent) {
            document.getElementById('progress-fill').style.width = percent + '%';
        }

        // Single Analysis Form
        document.getElementById('single-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const content = document.getElementById('single-content').value.trim();
            if (!content) {
                alert('Please enter content to analyze.');
                return;
            }

            showLoading('Analyzing content... This may take a minute.');
            updateProgress(20);

            try {
                const response = await fetch(BASE + '/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content })
                });
                updateProgress(80);

                if (!response.ok) throw new Error('Analysis failed');
                const data = await response.json();
                updateProgress(100);

                // Update charts
                document.getElementById('s-overview-chart').src = 'data:image/png;base64,' + data.overview_chart;
                document.getElementById('s-top-dimensions-chart').src = 'data:image/png;base64,' + data.top_dimensions_chart;
                document.getElementById('s-clusters-chart').src = 'data:image/png;base64,' + data.clusters_chart;
                document.getElementById('s-pca-chart').src = 'data:image/png;base64,' + data.pca_chart;
                document.getElementById('s-histogram-chart').src = 'data:image/png;base64,' + data.histogram_chart;

                // Update metrics
                const metrics = data.analysis.metrics;
                const metricsContainer = document.getElementById('s-metrics-container');
                metricsContainer.innerHTML = '';
                [
                    { label: 'Dimensions', value: metrics.dimension_count },
                    { label: 'Mean Value', value: metrics.mean_value.toFixed(6) },
                    { label: 'Std Deviation', value: metrics.std_dev.toFixed(6) },
                    { label: 'Min Value', value: `${metrics.min_value.toFixed(6)} (dim ${metrics.min_dimension})` },
                    { label: 'Max Value', value: `${metrics.max_value.toFixed(6)} (dim ${metrics.max_dimension})` },
                    { label: 'Significant Dims', value: `${metrics.significant_dims} (>0.1)` }
                ].forEach(m => {
                    metricsContainer.innerHTML += `<div class="metric-card"><h3 class="font-semibold text-gray-600 text-sm mb-2">${m.label}</h3><p class="text-2xl font-bold text-gray-900">${m.value}</p></div>`;
                });

                // Update clusters
                const clustersContainer = document.getElementById('s-clusters-container');
                clustersContainer.innerHTML = data.analysis.clusters.length === 0
                    ? '<p class="text-gray-500 text-center py-4">No significant clusters detected.</p>'
                    : data.analysis.clusters.map(c => `
                        <div class="metric-card">
                            <h3 class="font-semibold text-gray-900 mb-2">Cluster #${c.id}: Dims ${c.start_dim}-${c.end_dim}</h3>
                            <p class="text-sm text-gray-600">Size: ${c.size} | Avg: ${c.avg_value.toFixed(4)} | Max: ${c.max_value.toFixed(4)}</p>
                        </div>
                    `).join('');

                // Update Content Analysis (render Markdown as HTML with enhanced options)
                if (typeof marked !== 'undefined') {
                    marked.setOptions({
                        breaks: true,
                        gfm: true,
                        headerIds: true,
                        mangle: false
                    });
                }
                document.getElementById('s-deepseek-analysis').innerHTML = marked.parse(data.claude_analysis || '');

                hideLoading();
                document.getElementById('single-results').classList.remove('hidden');
                document.getElementById('single-results').scrollIntoView({ behavior: 'smooth' });

            } catch (error) {
                hideLoading();
                alert('Analysis failed. Please try again.');
                console.error(error);
            }
        });

        // Comparison Analysis
        async function runComparison() {
            const userContent = document.getElementById('user-content').value.trim();
            const keyword = document.getElementById('target-keyword').value.trim();
            const isAuto = document.querySelector('input[name="comp-source"]:checked').value === 'auto';

            if (!userContent) {
                alert('Please enter your article content.');
                return;
            }

            let competitors = [];
            if (isAuto) {
                competitors = competitorData.filter(c => c.content);
                if (competitors.length === 0) {
                    alert('Please search and extract competitor content first.');
                    return;
                }
            } else {
                for (let i = 1; i <= 3; i++) {
                    const content = document.getElementById(`manual-comp-${i}`).value.trim();
                    if (content) {
                        competitors.push({ content, source: 'manual', title: `Competitor ${i}` });
                    }
                }
                if (competitors.length === 0) {
                    alert('Please enter at least one competitor content.');
                    return;
                }
            }

            showLoading('Analyzing and comparing content... This may take 2-3 minutes.');
            updateProgress(10);

            try {
                const response = await fetch(BASE + '/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_content: userContent,
                        keyword: keyword,
                        competitors: competitors
                    })
                });
                updateProgress(80);

                const data = await response.json();
                updateProgress(100);

                if (!data.success) {
                    throw new Error(data.error || 'Comparison failed');
                }

                // Update comparison charts
                document.getElementById('c-chunk-scatter-chart').src = 'data:image/png;base64,' + data.chunk_scatter_chart;
                document.getElementById('c-scatter-chart').src = 'data:image/png;base64,' + data.scatter_chart;
                document.getElementById('c-scorecard-chart').src = 'data:image/png;base64,' + data.scorecard_chart;
                document.getElementById('c-radar-chart').src = 'data:image/png;base64,' + data.radar_chart;
                document.getElementById('c-heatmap-chart').src = 'data:image/png;base64,' + data.heatmap_chart;
                document.getElementById('c-dimension-chart').src = 'data:image/png;base64,' + data.dimension_chart;
                document.getElementById('c-similarity-matrix-chart').src = 'data:image/png;base64,' + data.similarity_matrix_chart;
                document.getElementById('c-overview-chart').src = 'data:image/png;base64,' + data.overview_chart;
                document.getElementById('c-top-dimensions-chart').src = 'data:image/png;base64,' + data.top_dimensions_chart;

                // Update similarity scores
                const scoresContainer = document.getElementById('c-similarity-scores');
                scoresContainer.innerHTML = data.comparison_analysis.similarity_scores.map(s => `
                    <div class="metric-card">
                        <h3 class="font-semibold text-gray-600 text-sm mb-2">${s.label}</h3>
                        <p class="text-3xl font-bold ${s.similarity >= 70 ? 'text-green-600' : s.similarity >= 50 ? 'text-yellow-600' : 'text-red-600'}">${s.similarity}%</p>
                    </div>
                `).join('');

                // Update gaps and strengths
                const gapsList = document.getElementById('c-gaps-list');
                const strengthsList = document.getElementById('c-strengths-list');

                const gaps = data.comparison_analysis?.user_gaps || [];
                const strengths = data.comparison_analysis?.user_strengths || [];

                console.log('Gaps found:', gaps.length, 'Strengths found:', strengths.length);

                if (gaps.length > 0) {
                    gapsList.innerHTML = gaps.slice(0, 15).map(g => `
                        <div class="p-3 bg-red-50 rounded-lg border border-red-200">
                            <div class="flex justify-between items-center">
                                <span class="font-semibold text-red-700">Dimension ${g.dimension}</span>
                                <span class="text-xs text-red-500">Diff: ${g.difference.toFixed(4)}</span>
                            </div>
                            <div class="text-sm text-gray-600 mt-1">
                                <span>You: <strong>${g.user_value.toFixed(4)}</strong></span> vs
                                <span>Competitors: <strong>${g.competitor_avg.toFixed(4)}</strong></span>
                            </div>
                        </div>
                    `).join('');
                } else {
                    gapsList.innerHTML = '<p class="text-gray-500 text-center py-4">ไม่พบ Content Gap ที่มีนัยสำคัญ</p>';
                }

                if (strengths.length > 0) {
                    strengthsList.innerHTML = strengths.slice(0, 15).map(s => `
                        <div class="p-3 bg-green-50 rounded-lg border border-green-200">
                            <div class="flex justify-between items-center">
                                <span class="font-semibold text-green-700">Dimension ${s.dimension}</span>
                                <span class="text-xs text-green-500">Diff: +${s.difference.toFixed(4)}</span>
                            </div>
                            <div class="text-sm text-gray-600 mt-1">
                                <span>You: <strong>${s.user_value.toFixed(4)}</strong></span> vs
                                <span>Competitors: <strong>${s.competitor_avg.toFixed(4)}</strong></span>
                            </div>
                        </div>
                    `).join('');
                } else {
                    strengthsList.innerHTML = '<p class="text-gray-500 text-center py-4">ไม่พบจุดแข็งที่มีนัยสำคัญ</p>';
                }

                // Update Content Analysis (render Markdown as HTML with enhanced options)
                if (typeof marked !== 'undefined') {
                    marked.setOptions({
                        breaks: true,
                        gfm: true,
                        headerIds: true,
                        mangle: false
                    });
                }
                document.getElementById('c-deepseek-analysis').innerHTML = marked.parse(data.deepseek_analysis || '');

                hideLoading();
                document.getElementById('compare-results').classList.remove('hidden');
                document.getElementById('compare-results').scrollIntoView({ behavior: 'smooth' });

            } catch (error) {
                hideLoading();
                alert('Comparison failed: ' + error.message);
                console.error(error);
            }
        }
    </script>
    <footer class="text-center py-6 mt-8 text-sm" style="color: #9ca3af;">
        Embedding Content Analysis Tool © <a href="https://nerdoptimize.com" target="_blank" rel="noopener" class="underline hover:text-white">NerdOptimize</a>. Use permitted; commercial resale and removal of this notice prohibited.
    </footer>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    content = data.get('content', '')
    
    # Get embedding from Gemini API
    embedding = get_embedding(content)
    
    # Generate charts
    overview_chart = plot_embedding_overview(embedding)
    top_dimensions_chart = plot_top_dimensions(embedding)
    clusters_chart = plot_dimension_clusters(embedding)
    pca_chart = plot_pca(embedding)
    histogram_chart = plot_activation_histogram(embedding)
    
    # Analyze embedding
    analysis = analyze_embedding(embedding)
    
    # Get Content Analysis
    deepseek_analysis = analyze_with_deepseek(embedding, content)
    
    # Return all data
    return jsonify({
        'overview_chart': overview_chart,
        'top_dimensions_chart': top_dimensions_chart,
        'clusters_chart': clusters_chart,
        'pca_chart': pca_chart,
        'histogram_chart': histogram_chart,
        'analysis': analysis,
        'claude_analysis': deepseek_analysis
    })

# ============================================================
# NEW API ENDPOINTS FOR COMPETITOR ANALYSIS
# ============================================================

@app.route('/search', methods=['POST'])
def search_competitors():
    """Search for top 3 competitors using SerpAPI"""
    data = request.json
    keyword = data.get('keyword', '')

    if not keyword:
        return jsonify({'success': False, 'error': 'Keyword is required'}), 400

    result = search_google_top3(keyword)
    return jsonify(result)

@app.route('/extract', methods=['POST'])
def extract_content_endpoint():
    """Extract content from a URL using Trafilatura"""
    data = request.json
    url = data.get('url', '')

    if not url:
        return jsonify({'success': False, 'error': 'URL is required'}), 400

    result = extract_web_content(url)
    return jsonify(result)

@app.route('/compare', methods=['POST'])
def compare_with_competitors():
    """
    Main comparison endpoint
    Input: {
        "user_content": str,
        "keyword": str,
        "competitors": [{"content": str, "source": "auto"|"manual", "url"?: str, "title"?: str}, ...]
    }
    """
    data = request.json
    user_content = data.get('user_content', '')
    keyword = data.get('keyword', '')
    competitors = data.get('competitors', [])

    if not user_content:
        return jsonify({'success': False, 'error': 'User content is required'}), 400

    if len(competitors) == 0:
        return jsonify({'success': False, 'error': 'At least one competitor is required'}), 400

    try:
        # Parse user content into chunks
        user_chunks = parse_markdown_to_chunks(user_content)
        user_plain_text = extract_plain_text(user_chunks)

        # Get user embedding (full article)
        print("Getting embedding for user content...")
        user_embedding = get_embedding(user_plain_text)

        # Get embeddings for each user chunk
        print("Getting embeddings for user chunks...")
        user_chunk_embeddings = []
        for chunk in user_chunks:
            chunk_text = chunk.get('content', '')
            if chunk_text and len(chunk_text.strip()) > 50:  # Only embed substantial chunks
                chunk_emb = get_embedding(chunk_text)
                user_chunk_embeddings.append({
                    'title': chunk.get('title', 'Section'),
                    'embedding': chunk_emb
                })

        # Process competitors
        competitor_embeddings = []
        competitor_labels = []
        competitor_contents = []
        competitor_chunk_data = []  # For chunk-level visualization

        for i, comp in enumerate(competitors[:3]):  # Max 3 competitors
            comp_content = comp.get('content', '')
            comp_title = comp.get('title', f'Competitor {i+1}')
            comp_url = comp.get('url', '')

            if comp_content:
                print(f"Getting embedding for {comp_title}...")
                comp_embedding = get_embedding(comp_content)
                competitor_embeddings.append(comp_embedding)

                # Create label
                if comp_url:
                    # Extract domain from URL
                    from urllib.parse import urlparse
                    domain = urlparse(comp_url).netloc
                    label = domain if domain else comp_title
                else:
                    label = comp_title

                competitor_labels.append(label[:30])  # Truncate long labels
                competitor_contents.append(comp_content)

                # Parse competitor into chunks and get embeddings
                print(f"Getting chunk embeddings for {label}...")
                comp_chunks = parse_markdown_to_chunks(comp_content)
                comp_chunk_embeddings = []
                for chunk in comp_chunks[:10]:  # Limit to 10 chunks per competitor
                    chunk_text = chunk.get('content', '')
                    if chunk_text and len(chunk_text.strip()) > 50:
                        chunk_emb = get_embedding(chunk_text)
                        comp_chunk_embeddings.append({
                            'title': chunk.get('title', 'Section'),
                            'embedding': chunk_emb
                        })

                competitor_chunk_data.append({
                    'source': label,
                    'chunks': comp_chunk_embeddings
                })

        if len(competitor_embeddings) == 0:
            return jsonify({'success': False, 'error': 'No valid competitor content provided'}), 400

        # Analyze embedding differences
        print("Analyzing embedding differences...")
        analysis_data = analyze_embedding_differences(user_embedding, competitor_embeddings, competitor_labels)

        # Generate comparison charts
        print("Generating comparison charts...")
        scatter_chart = plot_comparison_scatter(user_embedding, competitor_embeddings, competitor_labels)
        scorecard_chart = plot_similarity_scorecard(analysis_data)
        radar_chart = plot_radar_chart(user_embedding, competitor_embeddings, competitor_labels)
        heatmap_chart = plot_comparison_heatmap(user_embedding, competitor_embeddings, competitor_labels)
        dimension_chart = plot_dimension_comparison(user_embedding, competitor_embeddings)
        similarity_matrix_chart = plot_similarity_matrix(user_embedding, competitor_embeddings, competitor_labels)

        # Generate chunk-level scatter chart
        print("Generating chunk-level scatter chart...")
        all_chunk_data = [{'source': 'Your Article', 'chunks': user_chunk_embeddings}] + competitor_chunk_data
        chunk_scatter_chart = plot_chunk_scatter(all_chunk_data)

        # Generate original single-article charts for user's article
        overview_chart = plot_embedding_overview(user_embedding)
        top_dimensions_chart = plot_top_dimensions(user_embedding)
        clusters_chart = plot_dimension_clusters(user_embedding)
        histogram_chart = plot_activation_histogram(user_embedding)

        # Analyze user article metrics
        user_analysis = analyze_embedding(user_embedding)

        # Get DeepSeek competitor analysis
        print("Getting Content Analysis...")
        deepseek_analysis = analyze_competitors_with_deepseek(
            user_content, competitor_contents, analysis_data, keyword
        )

        # Return all data
        return jsonify({
            'success': True,
            # Comparison charts
            'scatter_chart': scatter_chart,
            'chunk_scatter_chart': chunk_scatter_chart,  # NEW: Chunk-level similarity map
            'scorecard_chart': scorecard_chart,
            'radar_chart': radar_chart,
            'heatmap_chart': heatmap_chart,
            'dimension_chart': dimension_chart,
            'similarity_matrix_chart': similarity_matrix_chart,
            # Original single-article charts
            'overview_chart': overview_chart,
            'top_dimensions_chart': top_dimensions_chart,
            'clusters_chart': clusters_chart,
            'histogram_chart': histogram_chart,
            # Analysis data
            'comparison_analysis': analysis_data,
            'user_analysis': user_analysis,
            'deepseek_analysis': deepseek_analysis,
            # Metadata
            'user_chunks_count': len(user_chunks),
            'competitors_count': len(competitor_embeddings),
            'competitor_labels': competitor_labels
        })

    except Exception as e:
        print(f"Error in compare_with_competitors: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Serve under APPLICATION_ROOT when set (e.g. /embedding-content-analysis-tool for nginx deploy)
_application_root = (os.getenv('APPLICATION_ROOT') or '').strip().rstrip('/')
if _application_root:
    app.wsgi_app = PrefixMiddleware(app.wsgi_app, _application_root)

if __name__ == '__main__':
    print("Starting Embedding Content Analysis Tool...")
    print("=" * 60)
    
    # Check if API keys are set
    if GOOGLE_API_KEY == "xxx" or not GOOGLE_API_KEY:
        print("⚠️  WARNING: GOOGLE_API_KEY is not set!")
        print("   Please set it in your .env file or as an environment variable.")
        print("   Get your key from: https://makersuite.google.com/app/apikey")
    else:
        print("✓ GOOGLE_API_KEY is configured")
    
    if DEEPSEEK_API_KEY == "xxx" or not DEEPSEEK_API_KEY:
        print("⚠️  WARNING: DEEPSEEK_API_KEY is not set!")
        print("   Please set it in your .env file or as an environment variable.")
        print("   Get your key from: https://platform.deepseek.com/")
    else:
        print("✓ DEEPSEEK_API_KEY is configured")

    if SERPAPI_API_KEY and SERPAPI_API_KEY.strip() and SERPAPI_API_KEY not in ("xxx", "your_serpapi_key"):
        print("✓ SERPAPI_API_KEY is configured (competitor search)")
    else:
        print("⚠️  WARNING: SERPAPI_API_KEY is not set!")
        print("   Required for competitor search. Get your key from: https://serpapi.com/")

    print("=" * 60)
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5001))
    print(f"Mode: {'Development' if debug else 'Production'}")
    print(f"Visit http://{host}:{port} in your browser to use the tool.")
    print("=" * 60)
    app.run(debug=debug, host=host, port=port)
