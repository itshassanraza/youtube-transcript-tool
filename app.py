# app.py
from streamlit.runtime.scriptrunner import add_script_run_ctx
from concurrent.futures import ThreadPoolExecutor
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
import plotly.express as px

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_RESULTS = 10
APP_VERSION = "2.3"
CREATED_BY = "Your Name"
CREATED_DATE = datetime.now().strftime("%Y-%m-%d")
MAX_RETRIES = 3
RETRY_DELAY = 1
REQUEST_TIMEOUT = 30
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

    def analyze_with_gemini(self, text: str) -> Optional[str]:
        """Analyze text using Google's Gemini API"""
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.gemini_url,
                    json={
                        "contents": [{
                            "parts": [{"text": text[:MAX_TRANSCRIPT_LENGTH]}]
                        }]
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
                logger.error("Gemini API error: %s", str(e))
                return None
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
            logger.error("OpenRouter API error: %s", str(e))
            return None

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
            "maxResults": max_results,
            "fields": "items(id(videoId),snippet(title,description))"
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

class AnalysisEngine:
    """Main analysis engine with caching and timeout handling"""
    def __init__(self, youtube_client: YouTubeClient, ai_service: AIService):
        self.youtube = youtube_client
        self.ai = ai_service
        self.analysis_cache = {}

    def _calculate_relevance(self, query: str, title: str, description: str, summary: str) -> float:
        """Improved relevance scoring using multiple factors"""
        try:
            query = query.lower().strip()
            if not query:
                return 0
            
            content = f"{title} {description} {summary}".lower()
            query_words = set(query.split())
            content_words = set(content.split())
            
            exact_matches = len(query_words & content_words)
            partial_matches = sum(
                1 for q_word in query_words 
                if any(q_word in c_word for c_word in content_words)
            )
            
            total_score = (exact_matches * 0.8) + (partial_matches * 0.2)
            max_possible = len(query_words)
            
            return total_score / max_possible if max_possible > 0 else 0
            
        except Exception as e:
            logger.error(f"Relevance calculation error: {str(e)}")
            return 0

    def full_analysis(self, query: str, max_results: int) -> List[Dict]:
        """Optimized analysis workflow with progress tracking"""
        if not query.strip():
            return []

        try:
            videos = self.youtube.search_videos(query, max_results)
            if not videos:
                logger.info("YouTube API returned no videos for query: %s", query)
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
                        if analysis.get('relevance', 0) >= 0.1:
                            results.append(analysis)

                except Exception as e:
                    logger.error("Error analyzing video %s: %s", video_id, str(e))
                    continue

            return sorted(results, key=lambda x: x["relevance"], reverse=True)
        except Exception as e:
            logger.error("Analysis failed: %s", str(e))
            return []
        finally:
            progress.empty()
            status_text.empty()

    def _analyze_video(self, video: dict, query: str) -> Dict:
        """Optimized video analysis with timeout handling"""
        video_id = video["id"]["videoId"]
        title = video["snippet"]["title"]
        description = video["snippet"]["description"]

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._full_video_analysis, video_id, title)
                result = future.result(timeout=45)
            
            result["relevance"] = self._calculate_relevance(
                query, title, description, result["analysis"].get("summary", "")
            )
            return result
        except TimeoutError:
            logger.warning("Analysis timeout for video: %s", video_id)
            return self._create_video_result(video_id, title, "timeout")
        except Exception as e:
            logger.error("Analysis failed: %s", str(e))
            return self._create_video_result(video_id, title, f"error: {str(e)}")

    def _full_video_analysis(self, video_id: str, title: str) -> Dict:
        """Core analysis with fallback mechanisms"""
        try:
            transcript = TranscriptService.get_transcript(video_id)
            stats = self.youtube.get_video_stats(video_id)
            comments = self.youtube.get_video_comments(video_id)
            sentiment = self._calculate_sentiment(comments)
            
            summary_prompt = f"Analyze this YouTube video content:\nTitle: {title}\nTranscript: {transcript[:1500]}\nProvide: 1. Summary 2. Main topics 3. Content quality assessment"
            summary = self.ai.analyze_with_gemini(summary_prompt)
            if not summary:
                summary = self.ai.analyze_with_openrouter(summary_prompt) or "No summary available"

            return self._create_video_result(
                video_id,
                title,
                "complete",
                stats=stats,
                summary=summary,
                comments=comments[:3],
                sentiment=sentiment,
                engagement=self._calculate_engagement(stats),
                misleading=self._check_content_match(title, transcript)
            )
        except Exception as e:
            logger.error("Full analysis failed: %s", str(e))
            return self._create_video_result(video_id, title, f"error: {str(e)}")

    def _create_video_result(self, video_id, title, status, **kwargs):
        """Create standardized video result dictionary"""
        return {
            "video_id": video_id,
            "title": title,
            "status": status,
            "relevance": kwargs.get("relevance", 0),
            "analysis": {
                "stats": kwargs.get("stats", {"views": 0, "likes": 0, "comments": 0}),
                "summary": kwargs.get("summary", "No analysis available"),
                "comments": kwargs.get("comments", []),
                "sentiment": kwargs.get("sentiment", "No comments"),
                "engagement": kwargs.get("engagement", 0),
                "misleading": kwargs.get("misleading", False)
            }
        }

    @staticmethod
    def _calculate_sentiment(comments: list) -> str:
        """Calculate average comment sentiment"""
        if not comments:
            return "No comments"
        try:
            avg = sum(c["sentiment"] for c in comments) / len(comments)
            if avg > 0.1:
                return "Positive"
            elif avg < -0.1:
                return "Negative"
            return "Neutral"
        except Exception as e:
            logger.error("Sentiment calculation error: %s", str(e))
            return "Unknown"

    @staticmethod
    def _calculate_engagement(stats: dict) -> float:
        """Calculate engagement score"""
        try:
            return ((stats["likes"] + stats["comments"]) / stats["views"]) * 100 if stats["views"] else 0
        except KeyError:
            return 0

    @staticmethod
    def _check_content_match(title: str, transcript: str) -> bool:
        """Check for content mismatch between title and transcript"""
        if not transcript:
            return False
        try:
            title_keywords = set(title.lower().split())
            content_keywords = set(transcript.lower().split())
            return len(title_keywords - content_keywords) / len(title_keywords) > 0.5
        except Exception as e:
            logger.error("Content match check error: %s", str(e))
            return False

def setup_sidebar() -> tuple:
    """Configure sidebar and return settings"""
    st.sidebar.image("https://www.youtube.com/img/desktop/yt_1200.png", width=100)
    st.sidebar.title("Settings")
    st.sidebar.subheader("Analysis Parameters")
    
    max_results = st.sidebar.slider("Max Results", 5, 50, MAX_RESULTS)
    min_relevance = st.sidebar.slider("Min Relevance Score", 0.0, 1.0, 0.2)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Version:** {APP_VERSION} | **By:** {CREATED_BY}")
    st.sidebar.markdown(f"**Last Updated:** {CREATED_DATE}")
    
    if st.sidebar.checkbox("Show Diagnostics"):
        st.sidebar.subheader("Diagnostics")
        test_query = st.sidebar.text_input("Test query")
        test_title = st.sidebar.text_input("Test title")
        if test_query and test_title:
            analyzer = get_analyzer()
            score = analyzer._calculate_relevance(test_query, test_title, "", "")
            st.sidebar.metric("Calculated Relevance", f"{score:.2f}")
    
    return max_results, min_relevance

def display_video(video: dict):
    """Display video analysis results with keyword cloud"""
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.video(f"https://youtube.com/watch?v={video['video_id']}")
        
        with col2:
            st.subheader(video["title"])
            st.caption(f"Relevance score: {video.get('relevance', 0):.2f}")
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
                comments = video["analysis"]["comments"]
                if not comments:
                    st.write("No comments available")
                else:
                    for comment in comments:
                        icon = "👍" if comment.get("sentiment", 0) > 0 else "👎" if comment.get("sentiment", 0) < 0 else "🤔"
                        st.markdown(f"""
                            **{comment['author']}** {icon}  
                            {comment['text']}  
                            *{comment['likes']} likes*
                        """)
            
            st.markdown("#### AI Analysis")
            st.write(video["analysis"]["summary"])
            
            transcript = TranscriptService.get_transcript(video['video_id'])
            if transcript:
                keywords = " ".join(set(transcript.lower().split()) - {'the', 'a', 'and', 'to', 'in', 'is', 'of'})
                st.markdown("**Top Keywords:**")
                st.write(keywords[:200] + "..." if len(keywords) > 200 else keywords)

def get_analyzer():
    """Initialize and return analyzer instance"""
    env_vars = check_env_vars()
    youtube_client = YouTubeClient(env_vars["YOUTUBE_API_KEY"])
    ai_service = AIService(env_vars["GEMINI_API_KEY"], env_vars["OPENROUTER_API_KEY"])
    return AnalysisEngine(youtube_client, ai_service)

def main():
    """Main application with tabs and new features"""
    st.title("🎥 YouTube Content Analyzer")
    max_results, min_relevance = setup_sidebar()
    
    query = st.text_input("🔍 Enter your search query", key="search_input")
    results_placeholder = st.empty()
    
    if st.button("Analyze Videos", type="primary", use_container_width=True) and query:
        with results_placeholder.container():
            tab1, tab2 = st.tabs(["📺 Videos", "📋 Summary"])
            analyzer = get_analyzer()
            relevant_videos = []
            
            with st.spinner("Analyzing videos..."):
                results = analyzer.full_analysis(query, max_results)
                for result in results:
                    if result.get("status") == "complete" and result.get("relevance", 0) >= min_relevance:
                        relevant_videos.append(result)
            
            with tab1:
                if relevant_videos:
                    for video in relevant_videos:
                        display_video(video)
                        st.divider()
                else:
                    st.warning("No relevant videos found matching your criteria")
            
            with tab2:
                if relevant_videos:
                    df = pd.DataFrame([{
                        'Title': v['title'],
                        'Relevance': f"{v.get('relevance', 0):.2f}",
                        'Summary': v['analysis'].get('summary', 'No summary available'),
                        'Views': v['analysis']['stats'].get('views', 0),
                        'Engagement': f"{v['analysis'].get('engagement', 0):.1f}%",
                        'Link': f"https://youtube.com/watch?v={v['video_id']}"
                    } for v in relevant_videos])
                    
                    st.dataframe(
                        df,
                        column_config={
                            "Link": st.column_config.LinkColumn(),
                            "Views": st.column_config.NumberColumn(format="%d"),
                            "Relevance": st.column_config.NumberColumn(format="%.2f")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Summary as CSV",
                        data=csv,
                        file_name=f"youtube_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    fig = px.bar(
                        df,
                        x="Title",
                        y="Engagement",
                        title="Engagement by Video",
                        hover_data=["Summary"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No analysis available yet")

def keep_alive():
    """Maintain application wake status"""
    while True:
        time.sleep(300)
        try:
            requests.get(os.environ.get("APP_URL", "http://localhost:8501/"), timeout=10)
        except:
            pass

if __name__ == "__main__":
    t = Thread(target=keep_alive)
    add_script_run_ctx(t)
    t.daemon = True
    t.start()
    
    try:
        main()
    except Exception as e:
        logger.critical("Application crash: %s", str(e))
        st.error("Critical application error - please refresh the page")
