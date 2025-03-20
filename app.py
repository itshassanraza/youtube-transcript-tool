# app.py
from streamlit.runtime.scriptrunner import add_script_run_ctx
import streamlit as st
import os
os.environ["STREAMLIT_GLOBAL_DISABLE_SECRETS_WARNING"] = "true"
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from requests.exceptions import Timeout
from threading import Thread
import time
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_RESULTS = 10
APP_VERSION = "2.0"
CREATED_BY = "Your Name"
CREATED_DATE = datetime.now().strftime("%Y-%m-%d")
MAX_RETRIES = 3
RETRY_DELAY = 1
REQUEST_TIMEOUT = 20
MAX_TRANSCRIPT_LENGTH = 3000

# Configure page settings
st.set_page_config(
    page_title="YouTube Content Analyzer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_env_vars():
    """Validate required environment variables"""
    required_vars = {
        'YOUTUBE_API_KEY': os.environ.get("YOUTUBE_API_KEY"),
        'GEMINI_API_KEY': os.environ.get("GEMINI_API_KEY"),
        'OPENROUTER_API_KEY': os.environ.get("OPENROUTER_API_KEY")
    }
    
    missing = [name for name, val in required_vars.items() if not val]
    if missing:
        st.error(f"Missing required environment variables: {', '.join(missing)}")
        st.stop()
    
    return required_vars

class AIService:
    """Handles AI analysis through multiple providers"""
    def __init__(self, gemini_api_key: str, openrouter_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.openrouter_api_key = openrouter_api_key
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

    def _handle_api_error(self, error: Exception, service_name: str):
        """Log and handle API errors"""
        logger.error("%s API error: %s", service_name, str(error))
        return None

    def analyze_with_gemini(self, text: str) -> Optional[str]:
        """Analyze text using Google's Gemini API"""
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.gemini_url,
                    json={
                        "contents": [{
                            "parts": [{"text": text[:MAX_TRANSCRIPT_LENGTH]}]
                        }],
                    },
                    params={"key": self.gemini_api_key},
                    headers={"Content-Type": "application/json"},
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Timeout:
                logger.warning("Gemini API timeout (attempt %d/%d)", attempt+1, MAX_RETRIES)
                if attempt == MAX_RETRIES-1:
                    return None
                time.sleep(RETRY_DELAY)
            except Exception as e:
                return self._handle_api_error(e, "Gemini")
        return None

    def analyze_with_openrouter(self, text: str) -> Optional[str]:
        """Fallback analysis using OpenRouter API"""
        try:
            response = requests.post(
                self.openrouter_url,
                json={
                    "model": "google/palm-2-chat-bison",
                    "messages": [{
                        "role": "user",
                        "content": text[:MAX_TRANSCRIPT_LENGTH]
                    }]
                },
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "HTTP-Referer": os.environ.get("APP_URL", "https://github.com/"),
                    "Content-Type": "application/json"
                },
                timeout=REQUEST_TIMEOUT
            )
            return response.json()['choices'][0]['message']['content'] if response.ok else None
        except Exception as e:
            return self._handle_api_error(e, "OpenRouter")

    def analyze_content(self, text: str) -> str:
        """Get analysis from available AI services"""
        result = self.analyze_with_gemini(text)
        if not result:
            result = self.analyze_with_openrouter(text)
        return result or "Analysis unavailable - please try again later"

class YouTubeClient:
    """Handles YouTube API interactions"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.session = requests.Session()

    def _make_request(self, endpoint: str, params: dict) -> Optional[dict]:
        """Generic request handler for YouTube API"""
        try:
            response = self.session.get(
                f"{self.base_url}/{endpoint}",
                params={**params, "key": self.api_key},
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("YouTube API error: %s", str(e))
            return None

    def search_videos(self, query: str, max_results: int) -> List[Dict]:
        """Search for YouTube videos"""
        result = self._make_request("search", {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results
        })
        return result.get("items", []) if result else []

    def get_video_stats(self, video_id: str) -> Dict:
        """Get video statistics"""
        result = self._make_request("videos", {
            "part": "statistics",
            "id": video_id
        })
        if result and result.get("items"):
            stats = result["items"][0]["statistics"]
            return {
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0))
            }
        return {"views": 0, "likes": 0, "comments": 0}

    def get_video_comments(self, video_id: str, max_comments: int = 5) -> List[Dict]:
        """Get video comments"""
        result = self._make_request("commentThreads", {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": max_comments,
            "order": "relevance"
        })
        return self._parse_comments(result) if result else []

    def _parse_comments(self, result: dict) -> List[Dict]:
        """Parse comments from API response"""
        comments = []
        for item in result.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            text = comment["textDisplay"]
            analysis = TextBlob(text)
            comments.append({
                "text": text,
                "sentiment": analysis.sentiment.polarity,
                "author": comment["authorDisplayName"],
                "likes": comment["likeCount"]
            })
        return comments

class TranscriptService:
    """Handles transcript retrieval and processing"""
    @staticmethod
    def get_transcript(video_id: str) -> str:
        """Get transcript with fallback to auto-translation"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en'],
                timeout=REQUEST_TIMEOUT
            )
            return TranscriptService._clean_transcript(transcript)
        except Exception as e:
            logger.warning("Transcript error: %s", str(e))
            return ""

    @staticmethod
    def _clean_transcript(transcript: list) -> str:
        """Clean and format transcript"""
        return " ".join(entry["text"] for entry in transcript)[:MAX_TRANSCRIPT_LENGTH]

class class AnalysisEngine:
    def __init__(self, youtube_client: YouTubeClient, ai_service: AIService):
        self.youtube = youtube_client
        self.ai = ai_service
        self.analysis_cache = {}

    def full_analysis(self, query: str, max_results: int) -> List[Dict]:
        """Optimized analysis workflow with progress tracking"""
        try:
            videos = self.youtube.search_videos(query, max_results)
            if not videos:
                logger.warning("No videos found for query: %s", query)
                return []

            results = []
            progress = st.progress(0)
            status_text = st.empty()
            
            for idx, video in enumerate(videos):
                try:
                    status_text.markdown(f"🔍 Analyzing video {idx+1}/{len(videos)}...")
                    progress.progress((idx+1)/len(videos))
                    
                    video_id = video["id"]["videoId"]
                    cached = self.analysis_cache.get(video_id)
                    
                    if cached:
                        results.append(cached)
                    else:
                        analysis = self._analyze_video(video, query)
                        self.analysis_cache[video_id] = analysis
                        results.append(analysis)
                        
                    # Yield intermediate results
                    if idx % 2 == 0:
                        yield results[-1]

                except Exception as e:
                    logger.error("Error analyzing video %s: %s", video_id, str(e))
                    continue

            return results
        finally:
            progress.empty()
            status_text.empty()

    def _analyze_video(self, video: dict, query: str) -> Dict:
        """Optimized video analysis with timeout handling"""
        video_id = video["id"]["videoId"]
        title = video["snippet"]["title"]
        
        # Fast relevance check before full analysis
        if not self._quick_relevance_check(query, title):
            return {
                "video_id": video_id,
                "title": title,
                "status": "skipped (low relevance)"
            }

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._full_video_analysis, video_id, title)
                return future.result(timeout=45)  # 45-second timeout
        except TimeoutError:
            logger.warning("Analysis timeout for video: %s", video_id)
            return {
                "video_id": video_id,
                "title": title,
                "status": "timeout"
            }

    def _full_video_analysis(self, video_id: str, title: str) -> Dict:
        """Core analysis with fallback mechanisms"""
        try:
            transcript = TranscriptService.get_transcript(video_id)
            stats = self.youtube.get_video_stats(video_id)
            
            # Parallelize comment fetching and AI analysis
            with ThreadPoolExecutor(max_workers=2) as executor:
                comments_future = executor.submit(self.youtube.get_video_comments, video_id)
                summary_future = executor.submit(self.ai.analyze_content, f"Title: {title}\nTranscript: {transcript[:1500]}")

                comments = comments_future.result(timeout=20)
                summary = summary_future.result(timeout=30)

            return {
                "video_id": video_id,
                "title": title,
                "stats": stats,
                "summary": summary,
                "comments": comments[:3],
                "status": "complete"
            }
        except Exception as e:
            logger.error("Full analysis failed: %s", str(e))
            return {
                "video_id": video_id,
                "title": title,
                "status": f"error: {str(e)}"
            }

    def _get_video_analysis(self, video_id: str, title: str) -> Dict:
        """Get combined video analysis"""
        transcript = TranscriptService.get_transcript(video_id)
        stats = self.youtube.get_video_stats(video_id)
        comments = self.youtube.get_video_comments(video_id)
        
        return {
            "stats": stats,
            "sentiment": self._calculate_sentiment(comments),
            "comments": comments[:3],
            "summary": self.ai.analyze_content(f"Title: {title}\nTranscript: {transcript}"),
            "engagement": self._calculate_engagement(stats),
            "misleading": self._check_content_match(title, transcript)
        }

    @staticmethod
    def _calculate_relevance(query: str, title: str) -> float:
        """Calculate relevance score between query and title"""
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        return len(query_words & title_words) / len(query_words) if query_words else 0

    @staticmethod
    def _calculate_engagement(stats: dict) -> float:
        """Calculate engagement score"""
        return ((stats["likes"] + stats["comments"]) / stats["views"]) * 100 if stats["views"] else 0

    @staticmethod
    def _calculate_sentiment(comments: list) -> str:
        """Calculate average comment sentiment"""
        if not comments:
            return "No comments"
        avg = sum(c["sentiment"] for c in comments) / len(comments)
        return "Positive" if avg > 0 else "Negative" if avg < 0 else "Neutral"

    @staticmethod
    def _check_content_match(title: str, transcript: str) -> bool:
        """Check for content mismatch between title and transcript"""
        if not transcript:
            return False
        title_keywords = set(title.lower().split())
        content_keywords = set(transcript.lower().split())
        return len(title_keywords - content_keywords) / len(title_keywords) > 0.5

# UI Components
def setup_sidebar() -> tuple:
    """Configure sidebar and return settings"""
    st.sidebar.image("https://www.youtube.com/img/desktop/yt_1200.png", width=100)
    st.sidebar.title("Settings")
    st.sidebar.subheader("Analysis Parameters")
    
    max_results = st.sidebar.slider("Max Results", 5, 50, MAX_RESULTS)
    min_relevance = st.sidebar.slider("Min Relevance Score", 0.0, 1.0, 0.5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Version:** {APP_VERSION} | **By:** {CREATED_BY}")
    st.sidebar.markdown(f"**Last Updated:** {CREATED_DATE}")
    
    return max_results, min_relevance

def display_video(video: dict):
    """Display video analysis results"""
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.video(f"https://youtube.com/watch?v={video['video_id']}")
        
        with col2:
            st.subheader(video["title"])
            stats = video["analysis"]["stats"]
            
            st.metric("Engagement Score", 
                     f"{video['analysis']['engagement']:.1f}%",
                     help="(Likes + Comments) / Views * 100")
            
            cols = st.columns(3)
            cols[0].metric("Views", f"{stats['views']:,}")
            cols[1].metric("Likes", f"{stats['likes']:,}")
            cols[2].metric("Comments", f"{stats['comments']:,}")
            
            if video["analysis"]["misleading"]:
                st.warning("⚠️ Potential title/content mismatch")
            
            st.caption(f"**Sentiment:** {video['analysis']['sentiment']}")
            
            with st.expander("💬 Top Comments"):
                for comment in video["analysis"]["comments"]:
                    icon = "👍" if comment["sentiment"] > 0 else "👎" if comment["sentiment"] < 0 else "🤔"
                    st.markdown(f"""
                        **{comment['author']}** {icon}  
                        {comment['text']}  
                        *{comment['likes']} likes*
                    """)
            
            st.markdown("#### AI Analysis")
            st.write(video["analysis"]["summary"])

def main():
    """Main application with real-time updates"""
    st.title("🎥 YouTube Content Analyzer")
    max_results, min_relevance = setup_sidebar()
    
    query = st.text_input("🔍 Enter your search query", key="search_input")
    results_placeholder = st.empty()
    
    if st.button("Analyze Videos", type="primary", use_container_width=True) and query:
        with results_placeholder.container():
            st.write("🚀 Starting analysis...")
            
            try:
                analyzer = get_analyzer()
                results_container = st.container()
                relevant_videos = []
                
                for result in analyzer.full_analysis(query, max_results):
                    if result.get("status") != "complete":
                        continue
                    
                    if result["relevance"] >= min_relevance:
                        relevant_videos.append(result)
                        
                        with results_container:
                            display_video(result)
                            st.divider()
                
                if not relevant_videos:
                    st.warning("No relevant videos found matching your criteria")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.exception("Main analysis error")

# Keep-alive and production setup
def keep_alive():
    """Maintain application wake status"""
    while True:
        time.sleep(300)
        try:
            requests.get(os.environ.get("APP_URL", "http://localhost:8501/"), timeout=10)
        except:
            pass

if __name__ == "__main__":
    # Start keep-alive thread
    t = Thread(target=keep_alive)
    add_script_run_ctx(t)
    t.daemon = True
    t.start()
    
    # Run main application
    try:
        main()
    except Exception as e:
        logger.critical("Application crash: %s", str(e))
        st.error("Critical application error - please refresh the page")
