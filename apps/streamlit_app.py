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
            name="Average Sentiment",
            marker=dict(color=avg_sentiment.values, colorscale='RdYlGn', showscale=True)
        ))
        fig.update_layout(
            title="Average Sentiment by Member",
            xaxis_title="Sentiment Score",
            yaxis_title="Member",
            template="plotly_dark",
            height=400
        )
        return fig
    return None


def plot_word_frequency(df: pd.DataFrame, top_n=20):
    """Plot word frequency bar chart"""
    all_text = ' '.join(df['message'].astype(str)).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    word_counts = Counter(words).most_common(top_n)

    words_list = [w[0] for w in word_counts]
    counts_list = [w[1] for w in word_counts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts_list,
        y=words_list,
        orientation='h',
        marker=dict(color=counts_list, colorscale='Viridis')
    ))
    fig.update_layout(
        title=f"Top {top_n} Most Frequent Words",
        xaxis_title="Frequency",
        yaxis_title="Word",
        template="plotly_dark",
        height=500
    )
    return fig


def plot_text_features(df: pd.DataFrame):
    """Plot text features pie chart"""
    df_temp = df.copy()
    df_temp['has_url'] = df_temp['message'].astype(str).str.contains('http', case=False, na=False)
    df_temp['has_question'] = df_temp['message'].astype(str).str.contains('\\?', regex=True, na=False)
    df_temp['has_exclamation'] = df_temp['message'].astype(str).str.contains('!', regex=False, na=False)
    df_temp['has_mention'] = df_temp['message'].astype(str).str.contains('@', regex=False, na=False)

    features = {
        'URLs': df_temp['has_url'].sum(),
        'Questions': df_temp['has_question'].sum(),
        'Exclamations': df_temp['has_exclamation'].sum(),
        'Mentions': df_temp['has_mention'].sum()
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(features.keys()),
        y=list(features.values()),
        marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ))
    fig.update_layout(
        title="Message Features",
        xaxis_title="Feature Type",
        yaxis_title="Count",
        template="plotly_dark"
    )
    return fig


def plot_day_of_week(df: pd.DataFrame):
    """Plot messages by day of week"""
    if 'day' in df.columns:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df['day'].value_counts()
        ordered_counts = [day_counts.get(day, 0) for day in day_order]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=day_order,
            y=ordered_counts,
            marker=dict(color=ordered_counts, colorscale='Blues')
        ))
        fig.update_layout(
            title="Messages by Day of Week",
            xaxis_title="Day",
            yaxis_title="Message Count",
            template="plotly_dark"
        )
        return fig
    return None


def plot_sender_comparison(df: pd.DataFrame, top_n=10):
    """Plot multi-metric sender comparison"""
    from plotly.subplots import make_subplots

    if 'message_length' not in df.columns:
        df['message_length'] = df['message'].astype(str).str.len()

    top_senders = df['sender'].value_counts().head(top_n).index

    # Prepare data
    msg_counts = []
    avg_lengths = []
    avg_sentiments = []

    for sender in top_senders:
        sender_df = df[df['sender'] == sender]
        msg_counts.append(len(sender_df))
        avg_lengths.append(sender_df['message_length'].mean())
        if 'sentiment_compound' in df.columns:
            avg_sentiments.append(sender_df['sentiment_compound'].mean())

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Message Count', 'Avg Message Length', 'Avg Sentiment')
    )

    fig.add_trace(
        go.Bar(x=msg_counts, y=list(top_senders), orientation='h', name='Messages'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=avg_lengths, y=list(top_senders), orientation='h', name='Length', marker_color='orange'),
        row=1, col=2
    )

    if avg_sentiments:
        fig.add_trace(
            go.Bar(x=avg_sentiments, y=list(top_senders), orientation='h', name='Sentiment', marker_color='green'),
            row=1, col=3
        )

    fig.update_layout(
        title_text=f"Top {top_n} Members - Comparison",
        template="plotly_dark",
        height=400,
        showlegend=False
    )

    return fig


def run_analysis(analysis_type: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Run specified analysis type"""
    result_text = ""
    charts = []

    try:
        if analysis_type == "univariate-messages":
            if 'message_length' not in df.columns:
                df['message_length'] = df['message'].astype(str).str.len()
            if 'word_count' not in df.columns:
                df['word_count'] = df['message'].astype(str).str.split().str.len()

            result_text = f"""**Message Characteristics**

Average Length: {df['message_length'].mean():.0f} characters | Average Words: {df['word_count'].mean():.1f}
"""

            # Message length distribution
            try:
                fig = plot_message_length_distribution(df)
                charts.append(fig)
            except:
                pass

            # Word count distribution
            try:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df['word_count'], nbinsx=30, marker_color='lightblue'))
                fig.update_layout(title="Word Count Distribution", xaxis_title="Words", yaxis_title="Count", template="plotly_dark")
                charts.append(fig)
            except:
                pass

            # Messages by sender
            try:
                top_senders = df['sender'].value_counts().head(10)
                fig = go.Figure()
                fig.add_trace(go.Bar(y=top_senders.index, x=top_senders.values, orientation='h', marker_color='teal'))
                fig.update_layout(title="Top 10 Most Active Members", xaxis_title="Messages", yaxis_title="Member", template="plotly_dark")
                charts.append(fig)
            except:
                pass

        elif analysis_type == "univariate-sentiment":
            if 'sentiment_compound' in df.columns:
                positive_pct = (df['sentiment_compound'] > 0.5).sum()/len(df)*100
                negative_pct = (df['sentiment_compound'] < -0.5).sum()/len(df)*100
                neutral_pct = 100 - positive_pct - negative_pct

                result_text = f"""**Sentiment Distribution**

Positive: {positive_pct:.1f}% | Neutral: {neutral_pct:.1f}% | Negative: {negative_pct:.1f}%
"""

                # Sentiment histogram
                try:
                    fig = plot_sentiment_distributions(df)
                    charts.append(fig)
                except:
                    pass

                # Sentiment pie chart
                try:
                    fig = go.Figure()
                    fig.add_trace(go.Pie(
                        labels=['Positive', 'Neutral', 'Negative'],
                        values=[positive_pct, neutral_pct, negative_pct],
                        marker_colors=['#2ecc71', '#95a5a6', '#e74c3c']
                    ))
                    fig.update_layout(title="Sentiment Breakdown", template="plotly_dark")
                    charts.append(fig)
                except:
                    pass

                # Sentiment by member
                try:
                    fig = plot_member_sentiment_analysis(df)
                    charts.append(fig)
                except:
                    pass
            else:
                result_text = "Sentiment data not available"

        elif analysis_type == "univariate-text":
            result_text = "**Text Features**"

            try:
                fig = plot_text_features(df)
                charts.append(fig)
            except:
                pass

        elif analysis_type == "bivariate-temporal":
            result_text = "**Temporal Patterns**"

            # Hour distribution
            try:
                fig = plot_temporal_distributions(df)
                if fig:
                    charts.append(fig)
            except:
                pass

            # Day of week
            try:
                fig = plot_day_of_week(df)
                if fig:
                    charts.append(fig)
            except:
                pass

        elif analysis_type == "bivariate-sender":
            result_text = "**Member Activity Analysis**"

            try:
                fig = plot_sender_comparison(df, top_n=10)
                charts.append(fig)
            except:
                pass

            try:
                fig = plot_engagement_metrics(df)
                charts.append(fig)
            except:
                pass

        elif analysis_type == "multivariate-heatmaps":
            if 'day' in df.columns and 'hour' in df.columns:
                heatmap_data = pd.crosstab(df['day'], df['hour'])
                peak_day = heatmap_data.sum(axis=1).idxmax()
                peak_hour = extract_hour(heatmap_data.sum(axis=0).idxmax())

                result_text = f"""**Activity Heatmap**

Peak Activity: {peak_day} at {peak_hour:02d}:00
"""

                try:
                    fig = plot_activity_heatmap(df)
                    charts.append(fig)
                except:
                    pass
            else:
                result_text = "Temporal data not available"

        elif analysis_type == "multivariate-engagement":
            result_text = "**Member Engagement Metrics**"

            try:
                fig = plot_sender_comparison(df, top_n=10)
                charts.append(fig)
            except:
                pass

            try:
                fig = plot_engagement_metrics(df)
                charts.append(fig)
            except:
                pass

        elif analysis_type == "temporal-trends":
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                daily = df.groupby(df['date'].dt.date).size()
                result_text = f"""**Time-based Patterns**

Daily Average: {daily.mean():.0f} messages
"""
            else:
                result_text = "**Time-based Patterns**"

            try:
                fig = plot_temporal_distribution(df, frequency='day')
                charts.append(fig)
            except:
                pass

            # Add time of day breakdown
            if 'part_of_day' in df.columns:
                try:
                    pod_counts = df['part_of_day'].value_counts()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=pod_counts.index, y=pod_counts.values, marker_color='coral'))
                    fig.update_layout(title="Messages by Time of Day", xaxis_title="Time Period", yaxis_title="Messages", template="plotly_dark")
                    charts.append(fig)
                except:
                    pass

        elif analysis_type == "temporal-series":
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                date_range = (df['date'].max() - df['date'].min()).days
                result_text = f"""**Time Series Analysis**

Duration: {date_range} days | Messages/Day: {len(df)/max(date_range, 1):.1f}
"""
            else:
                result_text = "**Time Series Analysis**"

            # Monthly trend
            if 'month' in df.columns:
                try:
                    monthly = df.groupby('month').size()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=monthly.index, y=monthly.values, marker_color='skyblue'))
                    fig.update_layout(title="Messages by Month", xaxis_title="Month", yaxis_title="Messages", template="plotly_dark")
                    charts.append(fig)
                except:
                    pass

        elif analysis_type == "nlp-words":
            result_text = "**Word Analysis**"

            try:
                fig = plot_word_frequency(df, top_n=20)
                charts.append(fig)
            except:
                pass

        elif analysis_type == "nlp-sentiment":
            if 'sentiment_compound' in df.columns:
                avg_sentiment = df['sentiment_compound'].mean()
                result_text = f"""**Advanced Sentiment Analysis**

Average Sentiment: {avg_sentiment:.3f}
"""

                try:
                    fig = plot_sentiment_distributions(df)
                    charts.append(fig)
                except:
                    pass

                try:
                    fig = plot_member_sentiment_analysis(df)
                    if fig:
                        charts.append(fig)
                except:
                    pass
            else:
                result_text = "Sentiment data not available"

        elif analysis_type == "group-members":
            result_text = "**Member Statistics**"

            try:
                fig = plot_message_frequency(df, by='sender', orientation='h')
                charts.append(fig)
            except:
                pass

            try:
                fig = plot_sender_comparison(df, top_n=10)
                charts.append(fig)
            except:
                pass

        elif analysis_type == "group-dynamics":
            sender_counts = df['sender'].value_counts()
            top_member = sender_counts.index[0]
            participation = sender_counts.iloc[0] / len(df) * 100

            result_text = f"""**Group Dynamics**

Most Active: {top_member} ({participation:.1f}%)
"""

            # Participation pie chart
            try:
                top_10 = sender_counts.head(10)
                others = sender_counts[10:].sum()
                labels = list(top_10.index) + ['Others']
                values = list(top_10.values) + [others]

                fig = go.Figure()
                fig.add_trace(go.Pie(labels=labels, values=values))
                fig.update_layout(title="Member Participation Distribution", template="plotly_dark")
                charts.append(fig)
            except:
                pass

        elif analysis_type == "stats-descriptive":
            if 'message_length' not in df.columns:
                df['message_length'] = df['message'].astype(str).str.len()
            if 'word_count' not in df.columns:
                df['word_count'] = df['message'].astype(str).str.split().str.len()

            result_text = f"""**Statistical Summary**

Total Messages: {len(df):,} | Members: {df['sender'].nunique()} | Avg Length: {df['message_length'].mean():.0f} chars
"""

            # Box plots for message length and word count
            try:
                from plotly.subplots import make_subplots
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Message Length', 'Word Count'))

                fig.add_trace(go.Box(y=df['message_length'], name='Length', marker_color='lightblue'), row=1, col=1)
                fig.add_trace(go.Box(y=df['word_count'], name='Words', marker_color='lightgreen'), row=1, col=2)

                fig.update_layout(title="Distribution Statistics", template="plotly_dark", showlegend=False, height=400)
                charts.append(fig)
            except:
                pass

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
                for analysis_key, analysis_data in st.session_state.analysis_results.items():
                    with st.expander(f"{analysis_key.replace('-', ' ').title()}", expanded=True):
                        # Show brief text summary
                        if analysis_data['result']:
                            st.markdown(analysis_data['result'])
                            if analysis_data['charts']:
                                st.divider()

                        # Display charts prominently
                        if analysis_data['charts']:
                            for idx, chart in enumerate(analysis_data['charts']):
                                st.plotly_chart(chart, use_container_width=True, key=f"{analysis_key}_chart_{idx}")
                        else:
                            st.info("No charts available for this analysis")
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
