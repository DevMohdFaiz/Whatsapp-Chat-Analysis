"""
Data Wrangling Module

Feature engineering and data transformations for WhatsApp chat analysis.
Handles both group and individual chat specific features.
"""

import re
from typing import Optional
import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features: month, day, hour, part_of_day.

    Args:
        df: DataFrame with 'date_and_time' column

    Returns:
        DataFrame with temporal features added
    """
    if df.empty or 'date_and_time' not in df.columns:
        return df

    df = df.copy()

    # Extract temporal components
    df['month'] = df['date_and_time'].dt.month_name()
    df['day'] = df['date_and_time'].dt.day_name()
    df['hour'] = df['date_and_time'].dt.hour
    df['date'] = df['date_and_time'].dt.date

    # Categorize part of day
    time_bins = [-1, 6, 12, 16, 19, 24]
    time_labels = ['Midnight', 'Morning', 'Afternoon', 'Evening', 'Night']
    df['part_of_day'] = pd.cut(
        df['hour'],
        bins=time_bins,
        labels=time_labels
    )

    # Format hour as HH:00
    df['hour'] = df['hour'].apply(
        lambda x: f"0{x}:00" if x < 10 else f"{x}:00"
    )

    return df


def add_message_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add message-level features: character_length, word_length, is_url.

    Args:
        df: DataFrame with 'message' column

    Returns:
        DataFrame with message features added
    """
    if df.empty or 'message' not in df.columns:
        return df

    df = df.copy()

    # Character length
    df['character_length'] = df['message'].str.len()

    # Word length
    df['word_length'] = df['message'].apply(
        lambda x: len(str(x).split(" "))
    )

    # URL detection
    df['is_url'] = df['message'].apply(
        lambda x: bool(re.search(r"(http://|https://)", str(x)))
    )

    return df


def add_response_time_features(
    df: pd.DataFrame,
    chat_type: str = 'group'
) -> pd.DataFrame:
    """
    Calculate response times with logic specific to chat type.

    Args:
        df: DataFrame with 'date_and_time' and 'sender' columns
        chat_type: 'group' or 'individual'

    Returns:
        DataFrame with response_time features added
    """
    if df.empty or 'date_and_time' not in df.columns:
        return df

    df = df.copy()

    if chat_type == 'individual':
        # For individual chats, calculate time between "You" and contact messages
        if 'sender' in df.columns:
            df['response_time'] = df['date_and_time'].diff()
            # Mark response times that are between different senders
            df['is_response'] = df['sender'] != df['sender'].shift(1)
        else:
            df['response_time'] = df['date_and_time'].diff()
            df['is_response'] = True
    else:
        # For group chats, response time is time between any two consecutive messages
        df['response_time'] = df['date_and_time'].diff()
        df['is_response'] = True

    return df


def add_engagement_features(
    df: pd.DataFrame,
    chat_type: str = 'group'
) -> pd.DataFrame:
    """
    Add engagement metrics specific to chat type.

    Args:
        df: DataFrame with temporal and message features
        chat_type: 'group' or 'individual'

    Returns:
        DataFrame with engagement features added
    """
    if df.empty:
        return df

    df = df.copy()

    if chat_type == 'group':
        # Group chat specific features
        if 'sender' in df.columns:
            # Messages per sender
            sender_counts = df['sender'].value_counts().to_dict()
            df['sender_message_count'] = df['sender'].map(sender_counts)

            # Average message length per sender
            sender_avg_length = df.groupby('sender')['word_length'].mean().to_dict()
            df['sender_avg_length'] = df['sender'].map(sender_avg_length)

    else:
        # Individual chat specific features
        if 'sender' in df.columns:
            # Identify conversation turns
            df['conversation_turn'] = (df['sender'] != df['sender'].shift(1)).cumsum()

            # Calculate conversation gaps (long pauses)
            df['conversation_gap'] = df['date_and_time'].diff()
            df['is_long_gap'] = df['conversation_gap'] > pd.Timedelta(hours=24)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction-based features like mentions, replies, etc.

    Args:
        df: DataFrame with message and sender columns

    Returns:
        DataFrame with interaction features added
    """
    if df.empty or 'message' not in df.columns:
        return df

    df = df.copy()

    # Detect mentions (common in group chats)
    if 'sender' in df.columns:
        # Check if message contains @mentions or sender names
        unique_senders = df['sender'].unique()
        df['has_mention'] = df['message'].apply(
            lambda x: any(f"@{sender.lower()}" in str(x).lower() or
                         sender.lower() in str(x).lower()
                         for sender in unique_senders
                         if sender and len(sender) > 2)
        )

    # Detect questions
    df['is_question'] = df['message'].str.contains(r'\?', regex=True)

    # Detect exclamations
    df['is_exclamation'] = df['message'].str.contains(r'!', regex=True)

    return df


def enrich_dataframe(
    df: pd.DataFrame,
    chat_type: str = 'group'
) -> pd.DataFrame:
    """
    Main feature engineering pipeline.

    Args:
        df: Preprocessed DataFrame from preprocess_chat_data
        chat_type: 'group' or 'individual'

    Returns:
        Enriched DataFrame with all features
    """
    if df.empty:
        return df

    df = df.copy()

    # Ensure chat_type is stored
    if 'chat_type' not in df.attrs:
        df.attrs['chat_type'] = chat_type

    # Add temporal features
    df = add_temporal_features(df)

    # Add message features
    df = add_message_features(df)

    # Add response time features
    df = add_response_time_features(df, chat_type=chat_type)

    # Add engagement features
    df = add_engagement_features(df, chat_type=chat_type)

    # Add interaction features
    df = add_interaction_features(df)

    # Reset index for clean output
    df = df.reset_index(drop=True)

    return df


