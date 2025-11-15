"""
NLP Analysis Module

Sentiment analysis, text processing, and keyword extraction for WhatsApp chats.
"""

from typing import Dict, List, Optional
import pandas as pd
from wordcloud import WordCloud
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Initialize VADER analyzer (thread-safe, can be reused)
_vader_analyzer = None


def _get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Get or create VADER sentiment analyzer instance."""
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def analyze_sentiment_vader(message: str) -> Dict[str, float]:
    """
    Analyze sentiment using VADER sentiment analyzer.

    Args:
        message: Text message to analyze

    Returns:
        Dictionary with sentiment scores: compound, pos, neu, neg
    """
    analyzer = _get_vader_analyzer()
    scores = analyzer.polarity_scores(str(message))
    return scores


def analyze_sentiment_textblob(message: str) -> Dict[str, float]:
    """
    Analyze sentiment using TextBlob.

    Args:
        message: Text message to analyze

    Returns:
        Dictionary with polarity and subjectivity scores
    """
    blob = TextBlob(str(message))
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }


def add_sentiment_features(
    df: pd.DataFrame,
    method: str = 'vader'
) -> pd.DataFrame:
    """
    Add sentiment analysis columns to DataFrame.

    Args:
        df: DataFrame with 'message' column
        method: 'vader' or 'textblob'

    Returns:
        DataFrame with sentiment columns added
    """
    if df.empty or 'message' not in df.columns:
        return df

    df = df.copy()

    if method == 'vader':
        # Add VADER sentiment scores
        sentiment_scores = df['message'].apply(analyze_sentiment_vader)
        df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        df['sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
        df['sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
        df['sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])

        # Categorize sentiment
        df['sentiment'] = df['sentiment_compound'].apply(
            lambda x: 1 if x >= 0.5 else (0.5 if x > -0.5 else 0)
        )

    elif method == 'textblob':
        # Add TextBlob sentiment scores
        sentiment_scores = df['message'].apply(analyze_sentiment_textblob)
        df['sentiment_polarity'] = sentiment_scores.apply(lambda x: x['polarity'])
        df['sentiment_subjectivity'] = sentiment_scores.apply(lambda x: x['subjectivity'])

        # Categorize sentiment
        df['sentiment'] = df['sentiment_polarity'].apply(
            lambda x: 1 if x >= 0.1 else (0.5 if x > -0.1 else 0)
        )

    else:
        raise ValueError(f"Unknown sentiment method: {method}. Use 'vader' or 'textblob'")

    return df


def tokenize_and_clean(
    text: str,
    remove_stopwords: bool = True,
    custom_stopwords: Optional[List[str]] = None
) -> List[str]:
    """
    Tokenize and clean text for analysis.

    Args:
        text: Text to tokenize
        remove_stopwords: Whether to remove stopwords
        custom_stopwords: Additional stopwords to remove

    Returns:
        List of cleaned tokens
    """
    tokens = word_tokenize(str(text).lower())

    # Filter alphabetic tokens
    tokens = [t for t in tokens if t.isalpha()]

    # Remove stopwords
    if remove_stopwords:
        stopwords_set = set(ENGLISH_STOP_WORDS)
        if custom_stopwords:
            stopwords_set.update(custom_stopwords)
        tokens = [t for t in tokens if t not in stopwords_set]

    return tokens


def generate_wordcloud(
    text: str,
    width: int = 800,
    height: int = 400,
    background_color: str = 'white',
    colormap: str = 'viridis',
    **kwargs
) -> WordCloud:
    """
    Generate word cloud from text.

    Args:
        text: Text to generate word cloud from
        width: Width of word cloud
        height: Height of word cloud
        background_color: Background color
        colormap: Colormap for words
        **kwargs: Additional arguments for WordCloud

    Returns:
        WordCloud object
    """
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        **kwargs
    ).generate(text)

    return wordcloud


def extract_keywords(
    text: str,
    n: int = 10,
    remove_stopwords: bool = True,
    custom_stopwords: Optional[List[str]] = None
) -> List[str]:
    """
    Extract top keywords from text.

    Args:
        text: Text to extract keywords from
        n: Number of keywords to return
        remove_stopwords: Whether to remove stopwords
        custom_stopwords: Additional stopwords to remove

    Returns:
        List of top keywords
    """
    from collections import Counter

    tokens = tokenize_and_clean(text, remove_stopwords, custom_stopwords)
    word_freq = Counter(tokens)
    top_keywords = [word for word, _ in word_freq.most_common(n)]

    return top_keywords


def prepare_text_for_wordcloud(
    df: pd.DataFrame,
    sender: Optional[str] = None,
    custom_stopwords: Optional[List[str]] = None
) -> str:
    """
    Prepare text from DataFrame for word cloud generation.

    Args:
        df: DataFrame with 'message' column
        sender: Optional sender name to filter by
        custom_stopwords: Additional stopwords to remove

    Returns:
        Cleaned text string ready for word cloud
    """
    if df.empty or 'message' not in df.columns:
        return ""

    # Filter by sender if provided
    if sender and 'sender' in df.columns:
        messages = df[df['sender'] == sender]['message']
    else:
        messages = df['message']

    # Combine all messages
    all_text = ' '.join(messages.astype(str))

    # Tokenize and clean
    tokens = tokenize_and_clean(all_text, remove_stopwords=True, custom_stopwords=custom_stopwords)

    # Join back into text
    cleaned_text = " ".join(tokens)

    return cleaned_text


