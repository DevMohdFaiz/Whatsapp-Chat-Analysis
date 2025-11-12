"""
Data Extraction Module

Handles loading and parsing WhatsApp chat exports from various formats.
Supports both zip and txt files, group and individual chats, iOS and Android formats.
"""

import re
import os
from typing import List, Dict, Optional, Union
from zipfile import ZipFile
import pandas as pd


def load_whatsapp_file(file_path: str, return_dataframe: bool = False, chat_type: str = 'auto') -> Union[List[str], pd.DataFrame]:
    """
    Unified loader supporting both .zip and .txt files.

    Args:
        file_path: Path to the WhatsApp export file (with or without extension)
        return_dataframe: If True, returns a parsed DataFrame instead of raw lines
        chat_type: 'auto', 'group', or 'individual' (only used if return_dataframe=True)

    Returns:
        List of chat message lines if return_dataframe=False,
        DataFrame with columns: date, time, sender, message if return_dataframe=True

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    # Handle file path - remove extension if provided, but keep directory structure
    if file_path.endswith('.zip') or file_path.endswith('.txt'):
        base_path = file_path.rsplit('.', 1)[0]
    else:
        base_path = file_path

    # Try zip file first
    zip_path = f"{base_path}.zip"
    txt_path = f"{base_path}.txt"

    if os.path.exists(zip_path):
        # Extract to a temporary directory to avoid conflicts
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            with ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(temp_dir)

            # Look for the extracted .txt file
            # WhatsApp exports typically have names like "WhatsApp Chat with [Name].txt"
            txt_files = [f for f in os.listdir(temp_dir) if f.endswith('.txt')]

            if not txt_files:
                raise FileNotFoundError(
                    f"No .txt file found in {zip_path}. "
                    f"Extracted files: {os.listdir(temp_dir)}"
                )

            # Use the first .txt file found (usually there's only one)
            extracted_txt_path = os.path.join(temp_dir, txt_files[0])
            with open(extracted_txt_path, "r", encoding='utf-8') as file:
                lines = file.readlines()

            # Return DataFrame if requested
            if return_dataframe:
                return _lines_to_dataframe(lines, chat_type)
            return lines

    elif os.path.exists(txt_path):
        with open(txt_path, "r", encoding='utf-8') as file:
            lines = file.readlines()

        # Return DataFrame if requested
        if return_dataframe:
            return _lines_to_dataframe(lines, chat_type)
        return lines

    else:
        raise FileNotFoundError(
            f"Neither {zip_path} nor {txt_path} found. "
            f"Please provide a valid WhatsApp export file."
        )


def _lines_to_dataframe(lines: List[str], chat_type: str = 'auto') -> pd.DataFrame:
    """
    Helper function to convert list of lines to DataFrame.
    Handles multi-line messages by merging continuation lines with the previous message.

    Args:
        lines: List of chat message lines
        chat_type: 'auto', 'group', or 'individual'

    Returns:
        DataFrame with columns: date, time, sender, message
    """
    extracted_data_dict = {
        'date': [],
        'time': [],
        'sender': [],
        'message': []
    }

    current_message = None

    for chat_line in lines:
        parsed = parse_whatsapp_message(chat_line)

        # Check if this is a new message (has date/time/sender) or a continuation
        if parsed['date'] and parsed['time'] and parsed['sender']:
            # This is a new message - save the previous one if it exists
            if current_message is not None:
                extracted_data_dict['date'].append(current_message['date'])
                extracted_data_dict['time'].append(current_message['time'])
                extracted_data_dict['sender'].append(current_message['sender'])
                extracted_data_dict['message'].append(current_message['message'])

            # Start tracking this new message
            current_message = {
                'date': parsed['date'],
                'time': parsed['time'],
                'sender': parsed['sender'],
                'message': parsed['message']
            }
        elif current_message is not None and chat_line.strip():
            # This is a continuation line - append to current message
            current_message['message'] += '\n' + chat_line.strip()

    # Don't forget to add the last message
    if current_message is not None:
        extracted_data_dict['date'].append(current_message['date'])
        extracted_data_dict['time'].append(current_message['time'])
        extracted_data_dict['sender'].append(current_message['sender'])
        extracted_data_dict['message'].append(current_message['message'])

    # Validate all lists have same length
    lengths = [
        len(extracted_data_dict['date']),
        len(extracted_data_dict['time']),
        len(extracted_data_dict['sender']),
        len(extracted_data_dict['message'])
    ]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("Data extraction resulted in mismatched column lengths")

    df = pd.DataFrame({
        'date': extracted_data_dict['date'],
        'time': extracted_data_dict['time'],
        'sender': extracted_data_dict['sender'],
        'message': extracted_data_dict['message']
    })

    # Auto-detect chat type if requested
    if chat_type == 'auto':
        chat_type = detect_chat_type(df)

    # Store chat type as metadata
    df.attrs['chat_type'] = chat_type

    return df


def parse_whatsapp_message(line: str) -> Dict[str, Optional[str]]:
    """
    Parse individual message lines handling iOS/Android formats.

    Supports multiple WhatsApp export formats:
    - Android: DD/MM/YYYY, HH:MM - Sender: Message
    - iOS: [DD/MM/YYYY, HH:MM:SS] Sender: Message
    - Individual chat: DD/MM/YYYY, HH:MM - You: Message or Contact: Message

    Args:
        line: Single line from WhatsApp export

    Returns:
        Dictionary with date, time, sender, and message fields
    """
    result = {
        'date': None,
        'time': None,
        'sender': None,
        'message': None
    }

    try:
        # Pattern 1: iOS format - [DD/MM/YYYY, HH:MM:SS] Sender: Message
        # Handle Unicode marks (left-to-right mark \u200e, right-to-left mark \u200f) that may appear
        # Match with or without Unicode marks before bracket, after bracket, and around sender/message
        ios_pattern = r".*?\[(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2}:\d{2})\]\s*(.+?):\s*(.+)"
        match = re.match(ios_pattern, line)
        if match:
            result['date'] = match.group(1)
            result['time'] = match.group(2)[:5]  # Take only HH:MM
            # Clean sender and message from Unicode marks
            result['sender'] = re.sub(r'[\u200e\u200f]', '', match.group(3)).strip()
            result['message'] = re.sub(r'[\u200e\u200f]', '', match.group(4)).strip()
            return result

        # Pattern 2: Android format - DD/MM/YYYY, HH:MM - Sender: Message
        android_pattern = r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2})\s*-\s*([^:]+):\s*(.+)"
        match = re.match(android_pattern, line)
        if match:
            result['date'] = match.group(1)
            result['time'] = match.group(2)
            result['sender'] = match.group(3).strip()
            result['message'] = match.group(4).strip()
            return result

        # Pattern 3: Android format without seconds - DD/MM/YYYY, HH:MM - Sender: Message
        android_pattern2 = r"(\d{2}/\d{2}/\d{4}),\s*(\d{2}:\d{2})\s*-\s*([^:]+):\s*(.+)"
        match = re.match(android_pattern2, line)
        if match:
            result['date'] = match.group(1)
            result['time'] = match.group(2)
            result['sender'] = match.group(3).strip()
            result['message'] = match.group(4).strip()
            return result

    except Exception:
        # Return empty result if parsing fails
        pass

    return result


def detect_chat_type(df: pd.DataFrame) -> str:
    """
    Auto-detect if chat is group or individual based on sender patterns.

    Args:
        df: DataFrame with at least a 'sender' column

    Returns:
        'group' or 'individual'
    """
    if df.empty or 'sender' not in df.columns:
        return 'group'  # Default to group

    unique_senders = df['sender'].nunique()
    sender_values = df['sender'].unique()

    # Check for individual chat indicators
    # Individual chats typically have "You" or only 1-2 unique senders
    if unique_senders <= 2:
        # Check if one of the senders is "You" or similar
        you_indicators = ['You', 'you', 'Me', 'me']
        if any(indicator in sender_values for indicator in you_indicators):
            return 'individual'

    # If more than 2 senders, it's likely a group chat
    if unique_senders > 2:
        return 'group'

    # Default to group if uncertain
    return 'group'


def extract_chat_data(
    file_path: str,
    chat_type: str = 'auto'
) -> pd.DataFrame:
    """
    Main extraction function with auto-detection for group vs individual chats.
    Properly handles multi-line messages.

    Args:
        file_path: Path to the WhatsApp export file
        chat_type: 'auto', 'group', or 'individual'. If 'auto', will detect automatically

    Returns:
        DataFrame with columns: date, time, sender, message

    Raises:
        ValueError: If chat_type is invalid
    """
    if chat_type not in ['auto', 'group', 'individual']:
        raise ValueError("chat_type must be 'auto', 'group', or 'individual'")

    whatsapp_chat = load_whatsapp_file(file_path)

    # Use the helper function that handles multi-line messages
    return _lines_to_dataframe(whatsapp_chat, chat_type)

