"""
WhatsApp Chat Analysis Package

A modular package for analyzing WhatsApp chat exports (both group and individual chats).
Supports zip and txt file formats, includes NLP analysis, visualizations, and AI-powered insights.
"""

__version__ = "1.0.0"

from .chat_analyzer import WhatsAppChatAnalyzer
from .data_extraction import load_whatsapp_file, extract_chat_data, detect_chat_type
from .data_cleaning import clean_messages, preprocess_chat_data
from .data_wrangling import enrich_dataframe
from .nlp_analysis import analyze_sentiment_vader, add_sentiment_features
from .visualization import (
    plot_message_frequency,
    plot_temporal_distribution,
    plot_activity_heatmap,
    plot_sentiment_distribution,
)
from .ai_insights import generate_chat_summary, identify_topics, generate_insights

__all__ = [
    "WhatsAppChatAnalyzer",
    "load_whatsapp_file",
    "extract_chat_data",
    "detect_chat_type",
    "clean_messages",
    "preprocess_chat_data",
    "enrich_dataframe",
    "analyze_sentiment_vader",
    "add_sentiment_features",
    "plot_message_frequency",
    "plot_temporal_distribution",
    "plot_activity_heatmap",
    "plot_sentiment_distribution",
    "generate_chat_summary",
    "identify_topics",
    "generate_insights",
]

