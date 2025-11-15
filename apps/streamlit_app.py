"""
Streamlit WhatsApp Chat Analysis Application

Advanced tool for analyzing WhatsApp chat exports with AI-powered insights.
Features: ChatGPT-like interface, chat history, specialized AI analysis, EDA, and more.
"""

import os
import sys
import tempfile
import traceback
from datetime import datetime
from io import BytesIO, StringIO
from typing import Dict, Any, List, Optional
from collections import Counter
import re
import json
import uuid
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing analysis modules
from src.data_extraction import load_whatsapp_file, extract_chat_data
from src.data_cleaning import preprocess_chat_data
from src.data_wrangling import enrich_dataframe
from src.nlp_analysis import add_sentiment_features
# from src.ai_integration import combined_ai_chat
from src.visualization import (
    plot_message_frequency,
    plot_temporal_distribution,
    plot_sentiment_distribution,
    plot_activity_heatmap,
    plot_engagement_metrics
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analysis - AI Powered",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and better styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #10a37f;
        --bg-dark: #1e1e1e;
    }

    .stat-card {
        background-color: #2b2b2b;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #444;
        text-align: center;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #10a37f;
    }

    .stat-label {
        color: #888;
        font-size: 0.9rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==================== SESSION MANAGEMENT ====================

# Session storage directory
SESSION_DIR = Path(__file__).parent / ".sessions"
SESSION_DIR.mkdir(exist_ok=True)

def get_session_id() -> str:
    """Get or create a unique session ID, persisting it in URL query params"""
    # Try to get session_id from URL query parameters first
    query_params = st.query_params

    # Check if session_id exists in query params
    query_session_id = query_params.get('session_id', None)

    if query_session_id:
        # Load from URL - this is the most important case for page refresh
        st.session_state.session_id = query_session_id
        return query_session_id
    elif 'session_id' in st.session_state:
        # Already have session_id in state, make sure it's in URL too
        session_id = st.session_state.session_id
        st.query_params['session_id'] = session_id
        return session_id
    else:
        # Create new session ID
        session_id = str(uuid.uuid4())
        st.session_state.session_id = session_id
        # Save to URL query params
        st.query_params['session_id'] = session_id
        return session_id

def get_session_file() -> Path:
    """Get the session file path"""
    session_id = get_session_id()
    return SESSION_DIR / f"{session_id}.pkl"

def save_session():
    """Save current session to disk"""
    try:
        session_file = get_session_file()
        session_data = {
            'chat_history': st.session_state.get('chat_history', []),
            'analysis_results': st.session_state.get('analysis_results', {}),
            'uploaded': st.session_state.get('uploaded', False),
            'stats': st.session_state.get('stats', {}),
            'context_messages': st.session_state.get('context_messages', []),
            'df': st.session_state.get('df'),
            'last_updated': datetime.now().isoformat()
        }

        with open(session_file, 'wb') as f:
            pickle.dump(session_data, f)
    except Exception as e:
        st.error(f"Error saving session: {e}")
        traceback.print_exc()

def load_session():
    """Load session from disk if exists"""
    try:
        session_file = get_session_file()
        if session_file.exists():
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)

            # Restore ALL session state from saved file
            st.session_state.chat_history = session_data.get('chat_history', [])
            st.session_state.analysis_results = session_data.get('analysis_results', {})
            st.session_state.uploaded = session_data.get('uploaded', False)
            st.session_state.stats = session_data.get('stats', {})
            st.session_state.context_messages = session_data.get('context_messages', [])
            st.session_state.df = session_data.get('df')

            return True
    except Exception as e:
        st.warning(f"Could not load previous session: {e}")
        traceback.print_exc()
    return False

def clear_session():
    """Clear current session and start fresh"""
    try:
        session_file = get_session_file()
        if session_file.exists():
            session_file.unlink()

        # Reset session state
        st.session_state.df = None
        st.session_state.chat_history = []
        st.session_state.analysis_results = {}
        st.session_state.uploaded = False
        st.session_state.stats = {}
        st.session_state.context_messages = []

        # Generate new session ID and update URL
        new_session_id = str(uuid.uuid4())
        st.session_state.session_id = new_session_id
        st.query_params['session_id'] = new_session_id
    except Exception as e:
        st.error(f"Error clearing session: {e}")

def cleanup_old_sessions(days=7):
    """Clean up session files older than specified days"""
    try:
        current_time = datetime.now()
        for session_file in SESSION_DIR.glob("*.pkl"):
            file_time = datetime.fromtimestamp(session_file.stat().st_mtime)
            if (current_time - file_time).days > days:
                session_file.unlink()
    except Exception as e:
        print(f"Error cleaning up sessions: {e}")


# ==================== INITIALIZATION ====================

def init_session_state():
    """Initialize session state variables and load persistent session"""
    # Initialize session ID first (this must happen before anything else)
    session_id = get_session_id()

    # Try to load existing session first
    session_loaded = load_session()

    # Only initialize default values if session wasn't loaded
    if not session_loaded:
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'uploaded' not in st.session_state:
            st.session_state.uploaded = False
        if 'stats' not in st.session_state:
            st.session_state.stats = {}
        if 'context_messages' not in st.session_state:
            st.session_state.context_messages = []

    # Add debug info (will show at top of page)
    if os.getenv('DEBUG_SESSION'):
        st.info(f"Session ID: {session_id[:8]}... | Loaded: {session_loaded} | Uploaded: {st.session_state.uploaded}")

    # Cleanup old sessions on startup
    cleanup_old_sessions()


# ==================== UTILITY FUNCTIONS ====================

def process_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Process uploaded WhatsApp chat file"""
    try:
        # Save to temporary file
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Extract and process chat data
        df = extract_chat_data(tmp_path)

        # Preprocess
        df = preprocess_chat_data(df, chat_type='auto')

        # Enrich with features
        df = enrich_dataframe(df, chat_type='auto')

        # Add sentiment analysis
        df = add_sentiment_features(df, method='vader')

        # Map sentiment scores to labels
        df['sentiment_label'] = df['sentiment'].apply(
            lambda x: 'Positive' if x >= 0.6 else ('Negative' if x <= 0.4 else 'Neutral')
        )

        # Add message length and word count if not present
        if 'message_length' not in df.columns and 'message' in df.columns:
            df['message_length'] = df['message'].astype(str).str.len()
        if 'word_count' not in df.columns and 'message' in df.columns:
            df['word_count'] = df['message'].astype(str).str.split().str.len()

        # Clean up temporary file
        os.remove(tmp_path)

        return df

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        traceback.print_exc()
        return None


def calculate_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate statistics from DataFrame"""
    stats = {
        "n_messages": int(df.shape[0]),
        "n_senders": int(df['sender'].nunique()) if 'sender' in df.columns else 0,
        "top_senders": list(df['sender'].value_counts().head(5).to_dict().keys()) if 'sender' in df.columns else [],
        "messages_per_sender": df['sender'].value_counts().head(10).to_dict() if 'sender' in df.columns else {},
        "sentiment_counts": df['sentiment_label'].value_counts().to_dict() if 'sentiment_label' in df.columns else {}
    }
    return stats


def setup_context_messages(df: pd.DataFrame) -> List[str]:
    """Setup context messages for AI chat"""
    context_messages = []
    if 'message' in df.columns and 'sender' in df.columns:
        for _, row in df.head(100).iterrows():
            context_messages.append(f"{row['sender']}: {row['message']}")
    return context_messages


# ==================== HELPER FUNCTIONS ====================

def extract_hour(hour_value) -> int:
    """Extract hour integer from various hour formats (int, '11:00', etc.)"""
    if pd.isna(hour_value):
        return 0
    if isinstance(hour_value, int):
        return hour_value
    if isinstance(hour_value, str):
        # Handle formats like '11:00' or '11'
        return int(hour_value.split(':')[0])
    return int(hour_value)


# ==================== ANALYSIS FUNCTIONS ====================

def plot_message_length_distribution(df: pd.DataFrame):
    """Plot message length distribution"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['message_length'],
        nbinsx=50,
        name="Message Length"
    ))
    fig.update_layout(
        title="Message Length Distribution",
        xaxis_title="Length (characters)",
        yaxis_title="Count",
        template="plotly_dark"
    )
    return fig


def plot_sentiment_distributions(df: pd.DataFrame):
    """Plot sentiment distributions"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['sentiment_compound'],
        nbinsx=50,
        name="Sentiment"
    ))
    fig.update_layout(
        title="Sentiment Score Distribution",
        xaxis_title="Sentiment Score",
        yaxis_title="Count",
        template="plotly_dark"
    )
    return fig


def plot_temporal_distributions(df: pd.DataFrame):
    """Plot temporal distributions"""
    if 'hour' in df.columns:
        hourly = df.groupby('hour').size()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hourly.index,
            y=hourly.values,
            name="Messages by Hour"
        ))
        fig.update_layout(
            title="Message Distribution by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Message Count",
            template="plotly_dark"
        )
        return fig
    return None


def plot_member_sentiment_analysis(df: pd.DataFrame):
    """Plot member sentiment analysis"""
    if 'sentiment_compound' in df.columns:
        avg_sentiment = df.groupby('sender')['sentiment_compound'].mean().sort_values(ascending=False).head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=avg_sentiment.values,
            y=avg_sentiment.index,
            orientation='h',
            name="Average Sentiment"
        ))
        fig.update_layout(
            title="Average Sentiment by Sender",
            xaxis_title="Sentiment Score",
            yaxis_title="Sender",
            template="plotly_dark"
        )
        return fig
    return None


def run_analysis(analysis_type: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Run specified analysis type"""
    result_text = ""
    charts = []

    try:
        if analysis_type == "univariate-messages":
            # Message characteristics analysis
            if 'message_length' not in df.columns:
                df['message_length'] = df['message'].astype(str).str.len()
            if 'word_count' not in df.columns:
                df['word_count'] = df['message'].astype(str).str.split().str.len()

            result_text = f"""**Univariate Analysis: Message Characteristics**

📊 **Message Length Distribution**
- Average: {df['message_length'].mean():.1f} characters
- Median: {df['message_length'].median():.1f} characters
- Std Dev: {df['message_length'].std():.1f}

📝 **Word Count Distribution**
- Average: {df['word_count'].mean():.1f} words
- Median: {df['word_count'].median():.1f} words

📅 **Frequency Analysis**
- Total Messages: {len(df):,}
- By Sender: Top is {df['sender'].value_counts().index[0]} with {df['sender'].value_counts().iloc[0]} messages
- Peak Day: {df['day'].mode()[0] if 'day' in df.columns else 'N/A'}
- Peak Month: {df['month'].mode()[0] if 'month' in df.columns else 'N/A'}
- Peak Hour: {extract_hour(df['hour'].mode()[0]) if 'hour' in df.columns else 'N/A'}:00
"""
            try:
                fig = plot_message_length_distribution(df)
                charts.append(fig)
            except:
                pass

        elif analysis_type == "univariate-sentiment":
            # Sentiment distribution analysis
            if 'sentiment_compound' in df.columns:
                result_text = f"""**Univariate Analysis: Sentiment Distribution**

📊 **Overall Sentiment Scores**
- Mean: {df['sentiment_compound'].mean():.3f}
- Median: {df['sentiment_compound'].median():.3f}
- Std Dev: {df['sentiment_compound'].std():.3f}

📈 **Distribution**
- Positive (>0.5): {(df['sentiment_compound'] > 0.5).sum()} ({(df['sentiment_compound'] > 0.5).sum()/len(df)*100:.1f}%)
- Neutral (-0.5 to 0.5): {((df['sentiment_compound'] >= -0.5) & (df['sentiment_compound'] <= 0.5)).sum()} ({((df['sentiment_compound'] >= -0.5) & (df['sentiment_compound'] <= 0.5)).sum()/len(df)*100:.1f}%)
- Negative (<-0.5): {(df['sentiment_compound'] < -0.5).sum()} ({(df['sentiment_compound'] < -0.5).sum()/len(df)*100:.1f}%)

🎭 **Mood Swing Analysis**
"""
                # Mood swing per member
                for sender in df['sender'].value_counts().head(5).index:
                    sender_sentiment_std = df[df['sender'] == sender]['sentiment_compound'].std()
                    result_text += f"- {sender}: {sender_sentiment_std:.3f}\n"

                try:
                    fig = plot_sentiment_distributions(df)
                    charts.append(fig)
                except:
                    pass
            else:
                result_text = "Sentiment data not available"

        elif analysis_type == "univariate-text":
            # Text features analysis
            df['has_url'] = df['message'].astype(str).str.contains('http', case=False, na=False)
            df['has_question'] = df['message'].astype(str).str.contains('\\?', regex=True, na=False)
            df['has_exclamation'] = df['message'].astype(str).str.contains('!', regex=False, na=False)
            df['has_mention'] = df['message'].astype(str).str.contains('@', regex=False, na=False)

            result_text = f"""**Univariate Analysis: Text Features**

🔗 **URL Detection**
- Messages with URLs: {df['has_url'].sum()} ({df['has_url'].sum()/len(df)*100:.1f}%)

❓ **Question Marks**
- Messages with questions: {df['has_question'].sum()} ({df['has_question'].sum()/len(df)*100:.1f}%)

❗ **Exclamation Marks**
- Messages with exclamations: {df['has_exclamation'].sum()} ({df['has_exclamation'].sum()/len(df)*100:.1f}%)

@ **Mentions**
- Messages with mentions: {df['has_mention'].sum()} ({df['has_mention'].sum()/len(df)*100:.1f}%)
"""

        elif analysis_type == "bivariate-temporal":
            result_text = f"""**Bivariate Analysis: Temporal Patterns**

📅 **Message Frequency vs Day of Week**
"""
            if 'day' in df.columns:
                day_counts = df['day'].value_counts()
                for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                    if day in day_counts.index:
                        result_text += f"- {day}: {day_counts[day]} messages\n"

            result_text += f"\n⏰ **Message Frequency vs Hour of Day**\n"
            if 'hour' in df.columns:
                hourly = df.groupby('hour').size()
                peak_hour = extract_hour(hourly.idxmax())
                quiet_hour = extract_hour(hourly.idxmin())
                result_text += f"- Peak Hour: {peak_hour:02d}:00 ({hourly.max()} messages)\n"
                result_text += f"- Quiet Hour: {quiet_hour:02d}:00 ({hourly.min()} messages)\n"

            try:
                fig = plot_temporal_distributions(df)
                if fig:
                    charts.append(fig)
            except:
                pass

        elif analysis_type == "bivariate-sender":
            result_text = f"""**Bivariate Analysis: Sender Relationships**

👥 **Message Count vs Sender**
"""
            top_senders = df['sender'].value_counts().head(10)
            for sender, count in top_senders.items():
                result_text += f"- {sender}: {count} messages ({count/len(df)*100:.1f}%)\n"

            if 'message_length' not in df.columns:
                df['message_length'] = df['message'].astype(str).str.len()

            result_text += f"\n📏 **Average Message Length vs Sender**\n"
            avg_lengths = df.groupby('sender')['message_length'].mean().sort_values(ascending=False).head(10)
            for sender, avg_len in avg_lengths.items():
                result_text += f"- {sender}: {avg_len:.1f} characters\n"

            if 'sentiment_compound' in df.columns:
                result_text += f"\n😊 **Sentiment vs Sender**\n"
                avg_sentiment = df.groupby('sender')['sentiment_compound'].mean().sort_values(ascending=False).head(10)
                for sender, sent in avg_sentiment.items():
                    result_text += f"- {sender}: {sent:.3f}\n"

            try:
                fig = plot_engagement_metrics(df)
                charts.append(fig)
            except:
                pass

        elif analysis_type == "multivariate-heatmaps":
            result_text = f"""**Multivariate Analysis: Activity Heatmaps**

🔥 **Day × Hour Activity**
"""
            if 'day' in df.columns and 'hour' in df.columns:
                heatmap_data = pd.crosstab(df['day'], df['hour'])
                peak_day = heatmap_data.sum(axis=1).idxmax()
                peak_hour = extract_hour(heatmap_data.sum(axis=0).idxmax())
                result_text += f"- Peak Day: {peak_day}\n"
                result_text += f"- Peak Hour: {peak_hour:02d}:00\n"
                result_text += f"- Busiest Combination: {peak_day} at {peak_hour:02d}:00\n"

                try:
                    fig = plot_activity_heatmap(df)
                    charts.append(fig)
                except:
                    pass
            else:
                result_text += "Temporal data not available\n"

        elif analysis_type == "multivariate-engagement":
            result_text = f"""**Multivariate Analysis: Engagement Metrics**

📊 **Comprehensive Member Statistics**
"""
            if 'message_length' not in df.columns:
                df['message_length'] = df['message'].astype(str).str.len()
            if 'word_count' not in df.columns:
                df['word_count'] = df['message'].astype(str).str.split().str.len()

            for sender in df['sender'].value_counts().head(10).index:
                sender_df = df[df['sender'] == sender]
                msg_count = len(sender_df)
                avg_len = sender_df['message_length'].mean()
                avg_words = sender_df['word_count'].mean()
                avg_sent = sender_df['sentiment_compound'].mean() if 'sentiment_compound' in df.columns else 0

                result_text += f"\n**{sender}**\n"
                result_text += f"- Messages: {msg_count}\n"
                result_text += f"- Avg Length: {avg_len:.1f} chars, {avg_words:.1f} words\n"
                result_text += f"- Avg Sentiment: {avg_sent:.3f}\n"

            try:
                fig = plot_engagement_metrics(df)
                charts.append(fig)
            except:
                pass

        elif analysis_type == "temporal-trends":
            result_text = f"""**Temporal Analysis: Time-based Patterns**

📈 **Message Frequency Over Time**
"""
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                daily = df.groupby(df['date'].dt.date).size()
                result_text += f"- Daily Average: {daily.mean():.1f} messages\n"
                result_text += f"- Busiest Day: {daily.idxmax()} ({daily.max()} messages)\n"
                result_text += f"- Quietest Day: {daily.idxmin()} ({daily.min()} messages)\n"

            if 'sentiment_compound' in df.columns and 'date' in df.columns:
                result_text += f"\n😊 **Sentiment Trends Over Time**\n"
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_compound'].mean()
                result_text += f"- Most Positive Day: {daily_sentiment.idxmax()} ({daily_sentiment.max():.3f})\n"
                result_text += f"- Most Negative Day: {daily_sentiment.idxmin()} ({daily_sentiment.min():.3f})\n"

            result_text += f"\n⏰ **Activity Patterns by Time of Day**\n"
            if 'part_of_day' in df.columns:
                pod_counts = df['part_of_day'].value_counts()
                for pod in ['Morning', 'Afternoon', 'Evening', 'Night', 'Midnight']:
                    if pod in pod_counts.index:
                        result_text += f"- {pod}: {pod_counts[pod]} messages\n"

            try:
                fig = plot_temporal_distribution(df, frequency='day')
                charts.append(fig)
            except:
                pass

        elif analysis_type == "temporal-series":
            result_text = f"""**Temporal Analysis: Time Series Features**

📅 **Date Range Analysis**
"""
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                date_range = (df['date'].max() - df['date'].min()).days
                result_text += f"- Total Duration: {date_range} days\n"
                result_text += f"- Start Date: {df['date'].min().strftime('%Y-%m-%d')}\n"
                result_text += f"- End Date: {df['date'].max().strftime('%Y-%m-%d')}\n"
                result_text += f"- Messages per Day: {len(df)/max(date_range, 1):.1f}\n"

            result_text += f"\n🔥 **Peak Activity Periods**\n"
            if 'month' in df.columns:
                monthly = df.groupby('month').size()
                result_text += f"- Peak Month: {monthly.idxmax()} ({monthly.max()} messages)\n"

            if 'day' in df.columns:
                weekly = df.groupby('day').size()
                result_text += f"- Peak Day: {weekly.idxmax()} ({weekly.max()} messages)\n"

            if 'hour' in df.columns:
                hourly = df.groupby('hour').size()
                peak_hour_raw = hourly.idxmax()
                peak_hour = extract_hour(peak_hour_raw)
                result_text += f"- Peak Hour: {peak_hour:02d}:00 ({hourly.max()} messages)\n"

        elif analysis_type == "nlp-words":
            result_text = f"""**Text/NLP Analysis: Word Analysis**

💬 **Most Frequent Words**
"""
            all_text = ' '.join(df['message'].astype(str)).lower()
            words = re.findall(r'\b[a-z]{4,}\b', all_text)
            word_counts = Counter(words)

            for word, count in word_counts.most_common(20):
                result_text += f"- {word}: {count} times\n"

            result_text += f"\n🔤 **Keyword Extraction (TF-IDF)**\n"
            try:
                vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(df['message'].astype(str))
                feature_names = vectorizer.get_feature_names_out()
                for term in feature_names:
                    result_text += f"- {term}\n"
            except:
                pass

        elif analysis_type == "nlp-sentiment":
            if 'sentiment_compound' in df.columns:
                result_text = f"""**Text/NLP Analysis: Advanced Sentiment**

📊 **VADER Sentiment Scores**
- Compound: {df['sentiment_compound'].mean():.3f} (±{df['sentiment_compound'].std():.3f})
"""
                if 'sentiment_pos' in df.columns:
                    result_text += f"- Positive: {df['sentiment_pos'].mean():.3f}\n"
                    result_text += f"- Neutral: {df['sentiment_neu'].mean():.3f}\n"
                    result_text += f"- Negative: {df['sentiment_neg'].mean():.3f}\n"

                result_text += f"\n😊 **Sentiment Categorization**\n"
                positive = (df['sentiment_compound'] > 0.5).sum()
                neutral = ((df['sentiment_compound'] >= -0.5) & (df['sentiment_compound'] <= 0.5)).sum()
                negative = (df['sentiment_compound'] < -0.5).sum()
                result_text += f"- Positive: {positive} ({positive/len(df)*100:.1f}%)\n"
                result_text += f"- Neutral: {neutral} ({neutral/len(df)*100:.1f}%)\n"
                result_text += f"- Negative: {negative} ({negative/len(df)*100:.1f}%)\n"

                try:
                    fig = plot_member_sentiment_analysis(df)
                    if fig:
                        charts.append(fig)
                except:
                    pass
            else:
                result_text = "Sentiment data not available"

        elif analysis_type == "group-members":
            result_text = f"""**Group Analysis: Member Statistics**

👥 **Messages per Sender**
"""
            sender_counts = df['sender'].value_counts()
            for sender, count in sender_counts.head(15).items():
                percentage = count / len(df) * 100
                result_text += f"- {sender}: {count} messages ({percentage:.1f}%)\n"

            if 'message_length' not in df.columns:
                df['message_length'] = df['message'].astype(str).str.len()

            result_text += f"\n📏 **Average Message Length per Sender**\n"
            avg_lengths = df.groupby('sender')['message_length'].mean().sort_values(ascending=False)
            for sender, avg_len in avg_lengths.head(10).items():
                result_text += f"- {sender}: {avg_len:.1f} characters\n"

            if 'sentiment_compound' in df.columns:
                result_text += f"\n😊 **Sentiment per Member**\n"
                avg_sentiment = df.groupby('sender')['sentiment_compound'].mean().sort_values(ascending=False)
                for sender, sent in avg_sentiment.head(10).items():
                    result_text += f"- {sender}: {sent:.3f}\n"

            try:
                fig = plot_message_frequency(df, by='sender', orientation='h')
                charts.append(fig)
            except:
                pass

        elif analysis_type == "group-dynamics":
            result_text = f"""**Group Analysis: Group Dynamics**

🔄 **Member Activity Patterns**
"""
            if 'day' in df.columns:
                for sender in df['sender'].value_counts().head(5).index:
                    sender_peak_day = df[df['sender'] == sender]['day'].mode()[0] if not df[df['sender'] == sender].empty else 'N/A'
                    result_text += f"- {sender}: Most active on {sender_peak_day}\n"

            if '@' in ' '.join(df['message'].astype(str)):
                result_text += f"\n@ **Mention Patterns**\n"
                mentions = df[df['message'].astype(str).str.contains('@', na=False)]
                result_text += f"- Total Mentions: {len(mentions)}\n"
                result_text += f"- Most Mentions by: {mentions['sender'].value_counts().index[0] if not mentions.empty else 'N/A'}\n"

            result_text += f"\n📊 **Conversation Participation**\n"
            sender_counts = df['sender'].value_counts()
            total_msgs = len(df)
            for sender, count in sender_counts.head(10).items():
                participation = count / total_msgs * 100
                result_text += f"- {sender}: {participation:.1f}% participation\n"

        elif analysis_type == "stats-descriptive":
            if 'message_length' not in df.columns:
                df['message_length'] = df['message'].astype(str).str.len()
            if 'word_count' not in df.columns:
                df['word_count'] = df['message'].astype(str).str.split().str.len()

            result_text = f"""**Statistical Summary: Descriptive Statistics**

📊 **Overall Statistics**
- Total Messages: {len(df):,}
- Unique Senders: {df['sender'].nunique()}
- Date Range: {(pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()).days if 'date' in df.columns else 'N/A'} days

📝 **Message Length Statistics**
- Mean: {df['message_length'].mean():.1f} characters
- Median: {df['message_length'].median():.1f} characters
- Std Dev: {df['message_length'].std():.1f}
- Min: {df['message_length'].min()}
- Max: {df['message_length'].max()}
- 25th Percentile: {df['message_length'].quantile(0.25):.1f}
- 75th Percentile: {df['message_length'].quantile(0.75):.1f}

💬 **Word Count Statistics**
- Mean: {df['word_count'].mean():.1f} words
- Median: {df['word_count'].median():.1f} words
- Std Dev: {df['word_count'].std():.1f}
"""

            if 'sentiment_compound' in df.columns:
                result_text += f"\n😊 **Sentiment Statistics**\n"
                result_text += f"- Mean: {df['sentiment_compound'].mean():.3f}\n"
                result_text += f"- Median: {df['sentiment_compound'].median():.3f}\n"
                result_text += f"- Std Dev: {df['sentiment_compound'].std():.3f}\n"
                result_text += f"- 25th Percentile: {df['sentiment_compound'].quantile(0.25):.3f}\n"
                result_text += f"- 75th Percentile: {df['sentiment_compound'].quantile(0.75):.3f}\n"

            result_text += f"\n📈 **Distribution Statistics**\n"
            result_text += f"- Messages per Sender (Mean): {df.groupby('sender').size().mean():.1f}\n"
            result_text += f"- Messages per Sender (Median): {df.groupby('sender').size().median():.1f}\n"

        else:
            result_text = f"Unknown analysis type: {analysis_type}"

    except Exception as e:
        result_text = f"Analysis failed: {str(e)}"
        traceback.print_exc()

    return {"result": result_text, "charts": charts}


# ==================== MAIN APPLICATION ====================

def main():
    """Main application function"""

    # Initialize session state
    init_session_state()

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.title("💬 WhatsApp Analysis")

        # Session indicator (small, bottom of sidebar)
        session_id = st.session_state.get('session_id', 'unknown')
        st.caption(f"🔗 Session: `{session_id[:8]}`")

        # File upload
        st.subheader("📤 Upload Chat")
        uploaded_file = st.file_uploader(
            "Choose a WhatsApp chat file",
            type=['txt', 'zip'],
            help="Export your WhatsApp chat and upload the .txt or .zip file"
        )

        if uploaded_file is not None and not st.session_state.uploaded:
            with st.spinner("Analyzing chat..."):
                df = process_uploaded_file(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.stats = calculate_stats(df)
                    st.session_state.context_messages = setup_context_messages(df)
                    st.session_state.uploaded = True
                    st.session_state.chat_history = []
                    save_session()  # Save session after upload
                    st.success("✅ Chat analyzed successfully!")
                    st.rerun()

        # New analysis button
        if st.session_state.uploaded:
            if st.button("➕ New Analysis", width='stretch', help="Clear all data and start fresh"):
                clear_session()  # Clear session and generate new ID
                save_session()   # Save the cleared state
                st.rerun()

        st.divider()

        # Analysis tools (only show if data is loaded)
        if st.session_state.uploaded:
            st.subheader("📊 Analysis Tools")

            # Univariate Analysis
            with st.expander("📊 Univariate Analysis"):
                if st.button("Message Characteristics", key="uni-msg", width='stretch'):
                    result = run_analysis("univariate-messages", st.session_state.df)
                    st.session_state.analysis_results["univariate-messages"] = result
                    save_session()
                if st.button("Sentiment Distribution", key="uni-sent", width='stretch'):
                    result = run_analysis("univariate-sentiment", st.session_state.df)
                    st.session_state.analysis_results["univariate-sentiment"] = result
                    save_session()
                if st.button("Text Features", key="uni-text", width='stretch'):
                    result = run_analysis("univariate-text", st.session_state.df)
                    st.session_state.analysis_results["univariate-text"] = result
                    save_session()

            # Bivariate Analysis
            with st.expander("🔗 Bivariate Analysis"):
                if st.button("Temporal Patterns", key="bi-temp", width='stretch'):
                    result = run_analysis("bivariate-temporal", st.session_state.df)
                    st.session_state.analysis_results["bivariate-temporal"] = result
                    save_session()
                if st.button("Sender Relationships", key="bi-send", width='stretch'):
                    result = run_analysis("bivariate-sender", st.session_state.df)
                    st.session_state.analysis_results["bivariate-sender"] = result
                    save_session()

            # Multivariate Analysis
            with st.expander("📈 Multivariate Analysis"):
                if st.button("Activity Heatmaps", key="multi-heat", width='stretch'):
                    result = run_analysis("multivariate-heatmaps", st.session_state.df)
                    st.session_state.analysis_results["multivariate-heatmaps"] = result
                    save_session()
                if st.button("Engagement Analysis", key="multi-eng", width='stretch'):
                    result = run_analysis("multivariate-engagement", st.session_state.df)
                    st.session_state.analysis_results["multivariate-engagement"] = result
                    save_session()

            # Temporal Analysis
            with st.expander("⏰ Temporal Analysis"):
                if st.button("Time-based Patterns", key="temp-pat", width='stretch'):
                    result = run_analysis("temporal-trends", st.session_state.df)
                    st.session_state.analysis_results["temporal-trends"] = result
                    save_session()
                if st.button("Time Series Features", key="temp-ser", width='stretch'):
                    result = run_analysis("temporal-series", st.session_state.df)
                    st.session_state.analysis_results["temporal-series"] = result
                    save_session()

            # Text/NLP Analysis
            with st.expander("💬 Text/NLP Analysis"):
                if st.button("Word Analysis", key="nlp-word", width='stretch'):
                    result = run_analysis("nlp-words", st.session_state.df)
                    st.session_state.analysis_results["nlp-words"] = result
                    save_session()
                if st.button("Advanced Sentiment", key="nlp-sent", width='stretch'):
                    result = run_analysis("nlp-sentiment", st.session_state.df)
                    st.session_state.analysis_results["nlp-sentiment"] = result
                    save_session()

            # Group Analysis
            with st.expander("👥 Group Analysis"):
                if st.button("Member Statistics", key="grp-mem", width='stretch'):
                    result = run_analysis("group-members", st.session_state.df)
                    st.session_state.analysis_results["group-members"] = result
                    save_session()
                if st.button("Group Dynamics", key="grp-dyn", width='stretch'):
                    result = run_analysis("group-dynamics", st.session_state.df)
                    st.session_state.analysis_results["group-dynamics"] = result
                    save_session()

            # Statistical Summary
            with st.expander("📋 Statistical Summary"):
                if st.button("Descriptive Statistics", key="stat-desc", width='stretch'):
                    result = run_analysis("stats-descriptive", st.session_state.df)
                    st.session_state.analysis_results["stats-descriptive"] = result
                    save_session()

            st.divider()

            # Data Export
            st.subheader("💾 Data Export")
            if st.button("📥 Download CSV", width='stretch'):
                csv = st.session_state.df.to_csv(index=False)
                st.download_button(
                    label="Download CSV File",
                    data=csv,
                    file_name="chat_analysis.csv",
                    mime="text/csv",
                    width='stretch'
                )

    # ==================== MAIN CONTENT ====================

    if not st.session_state.uploaded:
        # Welcome screen
        st.markdown("<h1 style='text-align: center; background: linear-gradient(135deg, #10a37f 0%, #00d4ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>WhatsApp Chat Analysis</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888; font-size: 1.2rem;'>Upload a chat export to start analyzing with AI-powered insights</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👈 Use the sidebar to upload your WhatsApp chat export file (.txt or .zip)")

    else:
        # Header with stats
        st.header("📊 Chat Analysis Dashboard")

        # Display stats cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{st.session_state.stats['n_messages']:,}</div>
                <div class="stat-label">Total Messages</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{st.session_state.stats['n_senders']}</div>
                <div class="stat-label">Participants</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            positive_count = st.session_state.stats['sentiment_counts'].get('Positive', 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{positive_count}</div>
                <div class="stat-label">Positive</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            negative_count = st.session_state.stats['sentiment_counts'].get('Negative', 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{negative_count}</div>
                <div class="stat-label">Negative</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Create tabs for Dashboard and Chat
        tab1, tab2 = st.tabs(["📊 Analysis Dashboard", "💬 AI Chat Assistant"])

        # ==================== DASHBOARD TAB ====================
        with tab1:
            # Display analysis results if any
            if st.session_state.analysis_results:
                st.subheader("📈 Analysis Results")
                for analysis_key, analysis_data in st.session_state.analysis_results.items():
                    with st.expander(f"View: {analysis_key.replace('-', ' ').title()}", expanded=True):
                        st.markdown(analysis_data['result'])
                        for idx, chart in enumerate(analysis_data['charts']):
                            st.plotly_chart(chart, width='stretch', key=f"{analysis_key}_chart_{idx}")
            else:
                st.info("👈 Select an analysis from the sidebar to view results here")

        # ==================== CHAT TAB ====================
        with tab2:
            st.markdown("### 🤖 AI-Powered Chat Analysis")
            st.markdown("Ask anything about your WhatsApp chat and get detailed insights")

            # Add clear history button at the top
            col1, col2, col3 = st.columns([5, 1, 1])
            with col3:
                if st.button("🗑️ Clear Chat", width='stretch', help="Clear chat history only"):
                    st.session_state.chat_history = []
                    save_session()  # Save after clearing chat
                    st.rerun()

            st.divider()

            # Display chat history using st.chat_message
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(msg['content'])
                elif msg['role'] == 'assistant':
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg['content'])

            # Chat input using st.chat_input
            user_input = st.chat_input(
                "Ask anything about your chat... (e.g., 'Analyze the communication patterns' or 'What are the main topics?')",
                key="chat_input"
            )

            if user_input:
                # Display user message immediately
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_input)

                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                # Prepare comprehensive stats for AI context
                df_stats = {
                    'total_messages': len(st.session_state.df),
                    'n_senders': st.session_state.df['sender'].nunique(),
                    'date_range': f"{st.session_state.df['date'].min()} to {st.session_state.df['date'].max()}" if 'date' in st.session_state.df.columns else 'N/A',
                    'avg_message_length': st.session_state.df['message_length'].mean() if 'message_length' in st.session_state.df.columns else 0,
                    'avg_word_count': st.session_state.df['word_count'].mean() if 'word_count' in st.session_state.df.columns else 0,
                    'sentiment_summary': st.session_state.df['sentiment_label'].value_counts().to_dict() if 'sentiment_label' in st.session_state.df.columns else {},
                    'top_senders': st.session_state.df['sender'].value_counts().head(10).to_dict() if 'sender' in st.session_state.df.columns else {}
                }

                # Get AI response from Groq only (single model to save tokens)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Analyzing..."):
                        try:
                            from src.ai_integration import groq_chat

                            response_data = groq_chat(
                                query=user_input,
                                context_lines=st.session_state.context_messages,
                                n_context=150,  # Increased context
                                stats=df_stats  # Pass statistics for better context
                            )

                            if response_data.get("response"):
                                ai_response = response_data["response"]
                                st.markdown(ai_response)

                                # Add AI response to history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": ai_response
                                })
                                save_session()  # Save after AI response
                            else:
                                error_msg = response_data.get("error", "Unknown error occurred")
                                st.error(f"Error: {error_msg}")

                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                st.rerun()


if __name__ == "__main__":
    main()
