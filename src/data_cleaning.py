"""
Data Cleaning Module

Handles cleaning and preprocessing of WhatsApp chat data.
Supports configurable cleaning rules and handles both group and individual chats.
"""

import re
from typing import Dict, Optional, List
import pandas as pd


def clean_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove media placeholders, system messages, and other non-text content.

    Args:
        df: DataFrame with at least a 'message' column

    Returns:
        Cleaned DataFrame
    """
    if df.empty or 'message' not in df.columns:
        return df

    df = df.copy()

    # Common media and system message patterns
    media_patterns = [
        '<Media omitted>',
        'Waiting for this message',
        'This message was deleted',
        'image omitted',
        'video omitted',
        'audio omitted',
        'document omitted',
        'sticker omitted',
        'GIF omitted',
    ]

    # Remove rows with media placeholders (case-insensitive)
    mask = ~df['message'].str.lower().isin([p.lower() for p in media_patterns])
    df = df[mask].reset_index(drop=True)

    # Remove empty messages
    df = df[df['message'].str.strip() != ''].reset_index(drop=True)

    return df


def normalize_sender_names(
    df: pd.DataFrame,
    name_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Clean and normalize sender names with configurable name mapping.

    Args:
        df: DataFrame with at least a 'sender' column
        name_mapping: Optional dictionary mapping old names to new names
                     Example: {'tmak 3': 'Tmak', 'John Doe': 'John'}

    Returns:
        DataFrame with normalized sender names
    """
    if df.empty or 'sender' not in df.columns:
        return df

    df = df.copy()

    # Apply custom name mapping if provided
    if name_mapping:
        df['sender'] = df['sender'].replace(name_mapping)

    # Remove leading/trailing whitespace
    df['sender'] = df['sender'].str.strip()

    # Remove empty sender names (shouldn't happen in valid exports, but handle it)
    df = df[df['sender'] != ''].reset_index(drop=True)

    return df


def validate_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure valid datetime parsing and remove rows with invalid dates.

    Args:
        df: DataFrame with 'date' and 'time' columns

    Returns:
        DataFrame with valid datetime entries only
    """
    if df.empty or 'date' not in df.columns or 'time' not in df.columns:
        return df

    df = df.copy()

    # Combine date and time
    df['date_and_time'] = df['date'] + " " + df['time']

    # Parse datetime with dayfirst=True (common for DD/MM/YYYY format)
    df['date_and_time'] = pd.to_datetime(
        df['date_and_time'],
        dayfirst=True,
        errors='coerce'
    )

    # Remove rows where datetime parsing failed
    df = df.dropna(subset=['date_and_time']).reset_index(drop=True)

    return df


def handle_multiline_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle multi-line messages that may have been split incorrectly.

    WhatsApp exports sometimes split long messages or messages with line breaks.
    This function attempts to merge continuation lines with their parent messages.

    Args:
        df: DataFrame with parsed messages

    Returns:
        DataFrame with merged multi-line messages
    """
    if df.empty:
        return df

    df = df.copy()
    # For now, return as-is. Advanced multi-line handling can be added later
    # This is a placeholder for future enhancement
    return df


def preprocess_chat_data(
    df: pd.DataFrame,
    chat_type: str = 'group',
    name_mapping: Optional[Dict[str, str]] = None,
    remove_first_row: bool = True
) -> pd.DataFrame:
    """
    Main preprocessing pipeline for WhatsApp chat data.

    Args:
        df: Raw DataFrame from extract_chat_data
        chat_type: 'group' or 'individual'
        name_mapping: Optional dictionary for sender name normalization
        remove_first_row: Whether to remove the first row (often contains metadata)

    Returns:
        Preprocessed DataFrame with date_and_time column added
    """
    if df.empty:
        return df

    df = df.copy()

    # Remove first row if it contains metadata (common in WhatsApp exports)
    if remove_first_row and len(df) > 0:
        df = df.iloc[1:].reset_index(drop=True)

    # Clean messages (remove media placeholders, etc.)
    df = clean_messages(df)

    # Normalize sender names
    if chat_type == 'group':
        df = normalize_sender_names(df, name_mapping)

    # Validate and parse datetime
    df = validate_datetime(df)

    # Handle multi-line messages
    df = handle_multiline_messages(df)

    # Store chat type as metadata
    df.attrs['chat_type'] = chat_type

    return df

