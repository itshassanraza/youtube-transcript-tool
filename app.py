import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
import os
from requests.exceptions import Timeout
from streamlit.report_thread import add_report_ctx
from threading import Thread
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_RESULTS = 10
APP_VERSION = "1.1"
CREATED_BY = "Your Name"
CREATED_DATE = datetime.now().strftime("%Y-%m-%d")
MAX_RETRIES = 3
RETRY_DELAY = 1

# API Keys (Replace with your own)
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Configure page settings
st.set_page_config(
    page_title="YouTube Content Analyzer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_env_vars():
    required_vars = ['YOUTUBE_API_KEY', 'GEMINI_API_KEY', 'OPENROUTER_API_KEY']
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        st.error(f"Missing required environment variables: {', '.join(missing)}")
        st.stop()

class AIService:
    def __init__(self, gemini_api_key: str, openrouter_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.openrouter_api_key = openrouter_api_key
        
    def analyze_with_gemini(self, text: str) -> Optional[str]:
        for attempt in range(MAX_RETRIES):
            try:
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.gemini_api_key}
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": text[:5000]  # Limit input size
                        }]
                    }]
                }
                
                response = requests.post(
                    url, 
                    json=payload, 
                    headers=headers, 
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Timeout:
                logger.warning(f"Gemini API timeout (attempt {attempt+1})")
                if attempt == MAX_RETRIES-1:
                    return None
                time.sleep(RETRY_DELAY)
            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")
                return None
        return None

    def analyze_with_openrouter(self, text: str) -> Optional[str]:
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "google/palm-2-chat-bison",
                "messages": [{
                    "role": "user",
                    "content": text[:5000]  # Limit input size
                }]
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return None
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            return None

    def analyze_content(self, text: str) -> Optional[str]:
        result = self.analyze_with_gemini(text)
        return result if result else self.analyze_with_openrouter(text)

class VideoAnalyzer:
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
        return list(set(w for w in words if w not in stop_words))

    @staticmethod
    def calculate_relevance_score(query: str, content: str) -> float:
        query_keywords = set(VideoAnalyzer.extract_keywords(query))
        content_keywords = set(VideoAnalyzer.extract_keywords(content))
        return len(query_keywords & content_keywords) / len(query_keywords) if query_keywords else 0.0

class YouTubeAnalyzer:
    def __init__(self, ai_service: AIService):
        self.video_analyzer = VideoAnalyzer()
        self.ai_service = ai_service

    @st.cache_data(ttl=3600, show_spinner=False)
    def search_videos(_self, query: str, max_results: int = 10) -> List[Dict]:
        url = 'https://www.googleapis.com/youtube/v3/search'
        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': max_results,
            'key': YOUTUBE_API_KEY
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('items', [])
        except Exception as e:
            st.error(f"Error fetching videos: {str(e)}")
            return []

    def get_video_transcript(self, video_id: str, language_code: str = 'en') -> str:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=[language_code],
                timeout=10
            )
            return ' '.join(entry['text'] for entry in transcript_list)
        except Exception as e:
            logger.error(f"Transcript error: {str(e)}")
            return ""

    def get_video_statistics(self, video_id: str) -> Dict:
        url = 'https://www.googleapis.com/youtube/v3/videos'
        params = {'part': 'statistics', 'id': video_id, 'key': YOUTUBE_API_KEY}
        try:
            response = requests.get(url, params=params, timeout=10)
            stats = response.json()['items'][0]['statistics']
            return {
                'likes': int(stats.get('likeCount', 0)),
                'views': int(stats.get('viewCount', 0)),
                'comments': int(stats.get('commentCount', 0))
            }
        except Exception as e:
            logger.error(f"Stats error: {str(e)}")
            return {'likes': 0, 'views': 0, 'comments': 0}

    def get_video_comments(self, video_id: str, max_comments: int = 5) -> List[Dict]:
        url = 'https://www.googleapis.com/youtube/v3/commentThreads'
        params = {'part': 'snippet', 'videoId': video_id, 'maxResults': max_comments, 'key': YOUTUBE_API_KEY}
        try:
            response = requests.get(url, params=params, timeout=10)
            comments = []
            for item in response.json().get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                text = comment['textDisplay']
                analysis = TextBlob(text)
                comments.append({
                    'text': text,
                    'sentiment': analysis.sentiment.polarity,
                    'author': comment['authorDisplayName'],
                    'likes': comment['likeCount']
                })
            return comments
        except Exception as e:
            logger.error(f"Comments error: {str(e)}")
            return []

    def check_content_mismatch(self, title: str, transcript: str) -> bool:
        title_keywords = set(self.video_analyzer.extract_keywords(title))
        transcript_keywords = set(self.video_analyzer.extract_keywords(transcript))
        return len(title_keywords - transcript_keywords)/len(title_keywords) > 0.5 if title_keywords else False

    def calculate_engagement_score(self, stats: Dict) -> float:
        if stats['views'] == 0:
            return 0.0
        return ((stats['likes'] + stats['comments']) / stats['views']) * 100

    @st.cache_data(ttl=3600, max_entries=20)
    def analyze_video_content(_self, video_id: str, title: str, description: str) -> Dict[str, Any]:
        try:
            analysis = {'basic_stats': _self.get_video_statistics(video_id)}
            transcript = _self.get_video_transcript(video_id)
            
            if transcript:
                analysis.update({
                    'misleading': _self.check_content_mismatch(title, transcript),
                    'engagement_score': _self.calculate_engagement_score(analysis['basic_stats']),
                    'ai_summary': _self.ai_service.analyze_content(f"""
                        Analyze this YouTube video content:
                        Title: {title}
                        Transcript: {transcript[:3000]}
                        Provide: 1. Summary 2. Main topics 3. Content quality assessment""")
                })

            comments = _self.get_video_comments(video_id)
            if comments:
                analysis['top_comments'] = comments
                avg_sentiment = sum(c['sentiment'] for c in comments) / len(comments)
                analysis['sentiment'] = 'Positive' if avg_sentiment > 0 else 'Negative' if avg_sentiment < 0 else 'Neutral'

            return analysis
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {'basic_stats': {'views': 0, 'likes': 0, 'comments': 0}}

def keep_alive():
    while True:
        time.sleep(10)
        st.experimental_rerun()

def main():
    st.title("🎥 YouTube Content Analyzer")
    st.sidebar.image("https://www.youtube.com/img/desktop/yt_1200.png", width=100)
    st.sidebar.title("Settings")

    analyzer = YouTubeAnalyzer(ai_service)
    max_results = st.sidebar.slider("Maximum Results", 5, 50, MAX_RESULTS)
    min_relevance = st.sidebar.slider("Minimum Relevance Score", 0.0, 1.0, 0.5)

    query = st.text_input("🔍 Enter your search query", key="search_input")
    if st.button("Search", type="primary", key="search_button"):
        if query:
            with st.spinner("Searching for relevant videos..."):
                videos = analyzer.search_videos(query, max_results)
                if videos:
                    tab1, tab2 = st.tabs(["📺 Videos", "📋 Summary"])
                    relevant_videos = []

                    with tab1:
                        progress_bar = st.progress(0)
                        total_videos = len(videos)
                        
                        for index, video in enumerate(videos):
                            video_id = video['id']['videoId']
                            title = video['snippet']['title']
                            description = video['snippet']['description']
                            progress_bar.progress((index+1)/total_videos)

                            try:
                                analysis = analyzer.analyze_video_content(video_id, title, description)
                                relevance_score = analyzer.video_analyzer.calculate_relevance_score(
                                    query, f"{title} {description} {analysis.get('ai_summary', '')}"
                                )

                                if relevance_score >= min_relevance:
                                    relevant_videos.append({
                                        'video_id': video_id,
                                        'title': title,
                                        'relevance_score': relevance_score,
                                        'analysis': analysis
                                    })

                                    with st.container():
                                        col1, col2 = st.columns([1, 2])
                                        with col1:
                                            st.video(f"https://youtube.com/watch?v={video_id}")
                                        with col2:
                                            st.subheader(title)
                                            stats = analysis.get('basic_stats', {})
                                            st.markdown(f"""
                                                **Engagement Score:** {analysis.get('engagement_score', 0):.1f}%  
                                                👁️ {stats.get('views', 0)} | 👍 {stats.get('likes', 0)} | 💬 {stats.get('comments', 0)}
                                            """)

                                            if analysis.get('misleading'):
                                                st.warning("⚠️ Potential content mismatch detected!")

                                            if 'sentiment' in analysis:
                                                st.markdown(f"**Comment Sentiment:** {analysis['sentiment']}")

                                            if 'top_comments' in analysis:
                                                with st.expander("💬 Top Comments"):
                                                    for comment in analysis['top_comments'][:3]:
                                                        sentiment_icon = '👍' if comment['sentiment'] > 0 else '👎' if comment['sentiment'] < 0 else '🤔'
                                                        st.markdown(f"""
                                                            **{comment['author']}** ({comment['likes']} likes) {sentiment_icon}  
                                                            {comment['text']}
                                                        """)

                                            st.markdown("**AI Analysis:**")
                                            st.write(analysis.get('ai_summary', 'No analysis available'))
                                        st.divider()
                            except Exception as e:
                                st.error(f"Failed to analyze {title[:50]}: {str(e)}")
                                continue
                        progress_bar.empty()

                    with tab2:
                        if relevant_videos:
                            df = pd.DataFrame([{
                                'Title': v['title'],
                                'Relevance': f"{v['relevance_score']:.2f}",
                                'Summary': v['analysis'].get('ai_summary', ''),
                                'Link': f"https://youtube.com/watch?v={v['video_id']}"
                            } for v in relevant_videos])

                            st.dataframe(
                                df,
                                column_config={"Link": st.column_config.LinkColumn()},
                                hide_index=True,
                                use_container_width=True
                            )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Version:** {APP_VERSION} | **By:** {CREATED_BY} | **Updated:** {CREATED_DATE}")

if __name__ == "__main__":
    check_env_vars()
    ai_service = AIService(GEMINI_API_KEY, OPENROUTER_API_KEY)
    
    # Start keep-alive thread
    t = Thread(target=keep_alive)
    add_report_ctx(t)
    t.daemon = True
    t.start()
    
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.exception("Critical application error:")
