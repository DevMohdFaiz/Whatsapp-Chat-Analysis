"""
Legacy helpers module for backward compatibility.

DEPRECATED: This module is maintained for backward compatibility with existing notebooks.
New code should import from the src package modules directly.

Migration guide:
- extract_chat_data() -> src.data_extraction.extract_chat_data()
- preprocess_df() -> Use src.chat_analyzer.WhatsAppChatAnalyzer class
- unzip_chat() -> src.data_extraction.load_whatsapp_file()
"""

import warnings
import pandas as pd

# Import from new modules
try:
    from src.data_extraction import extract_chat_data as _new_extract_chat_data, load_whatsapp_file
    from src.data_cleaning import preprocess_chat_data
    from src.data_wrangling import enrich_dataframe
    _NEW_MODULES_AVAILABLE = True
except ImportError:
    # Fallback to old implementation if src package not available
    import re
    import pandas as pd
    import numpy as np
    from zipfile import ZipFile
    _NEW_MODULES_AVAILABLE = False


def _deprecation_warning(function_name: str, new_location: str):
    """Issue deprecation warning."""
    warnings.warn(
        f"{function_name}() is deprecated. "
        f"Please use {new_location} instead. "
        f"This function will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )


def unzip_chat(whatsapp_file_path):
    """
    DEPRECATED: Use src.data_extraction.load_whatsapp_file() instead.

    Unzips the whatsapp.zip file and returns the loaded .txt file as a list
    Params: whatsapp_file_path= Path of the whatsapp zip file
    """
    _deprecation_warning("unzip_chat", "src.data_extraction.load_whatsapp_file")

    if _NEW_MODULES_AVAILABLE:
        # Try new implementation first
        return load_whatsapp_file(whatsapp_file_path)
    else:
        # Fallback to old implementation
        with ZipFile(f"{whatsapp_file_path}.zip", "r") as f:
            f.extractall()
        with open(f"{whatsapp_file_path}.txt", "r", encoding='utf-8') as file:
            whatsapp_chat = file.readlines()
        return whatsapp_chat


def extract_chat_data(whatsapp_file_path):
    """
    DEPRECATED: Use src.data_extraction.extract_chat_data() instead.

    Extracts important information like date, time, sender and message content
    from whatsapp.zip file and returns a pandas dataframe with the columns:
    date, time, sender and message

    Params: whatsapp_file_path= Path of the whatsapp zip file
    """
    _deprecation_warning("extract_chat_data", "src.data_extraction.extract_chat_data")

    if _NEW_MODULES_AVAILABLE:
        # Use new implementation
        return _new_extract_chat_data(whatsapp_file_path)
    else:
        # Fallback to old implementation
        whatsapp_chat = unzip_chat(whatsapp_file_path=whatsapp_file_path)
        extracted_data_dict = {
            'date': [], 'time': [], 'sender': [], 'message': []
        }

        for chat in whatsapp_chat:
            try:
                date, time = re.search(r"\d{2}/\d{2}/\d{4}", chat), re.search(r"\d{2}:\d{2}", chat)
                sender, message = re.search(r"-\s*([^:]+):", chat), re.search(r": (.+)\n$", chat)

                extracted_data_dict['date'].append(date.group(0) if date else "")
                extracted_data_dict['time'].append(time.group(0) if time else "")
                extracted_data_dict['sender'].append(sender.group(1) if sender else "")
                extracted_data_dict['message'].append(message.group(1) if message else "")
            except:
                extracted_data_dict['date'].append("")
                extracted_data_dict['time'].append("")
                extracted_data_dict['sender'].append("")
                extracted_data_dict['message'].append("")

        assert len(extracted_data_dict['date'])== len(extracted_data_dict['time'])==\
              len(extracted_data_dict['sender'])== len(extracted_data_dict['message'])

        return pd.DataFrame({
            'date': extracted_data_dict['date'],
            'time': extracted_data_dict['time'],
            'sender': extracted_data_dict['sender'],
            'message': extracted_data_dict['message']
        })


def preprocess_df(df: pd.DataFrame):
    """
    DEPRECATED: Use src.chat_analyzer.WhatsAppChatAnalyzer class instead.

    Preprocesses the chat dataframe with cleaning and feature engineering.

    Args:
        df: Raw DataFrame from extract_chat_data

    Returns:
        Preprocessed DataFrame
    """
    _deprecation_warning("preprocess_df", "src.chat_analyzer.WhatsAppChatAnalyzer")

    if _NEW_MODULES_AVAILABLE:
        # Use new implementation
        # First preprocess, then enrich
        df_processed = preprocess_chat_data(df, chat_type='group')
        df_enriched = enrich_dataframe(df_processed, chat_type='group')
        return df_enriched
    else:
        # Fallback to old implementation
        df = df.iloc[1:, :]
        df['sender'].replace('tmak 3', 'Tmak', inplace=True)
        df = df[~df['message'].isin(['<Media omitted>', 'Waiting for this message'])].reset_index()
        df['date_and_time']=df['date']+ " " + df['time']
        df['date_and_time']= pd.to_datetime(df['date_and_time'], dayfirst=True, errors='coerce')
        df = df.dropna()
        df=df.drop(['date', 'time', 'index'], axis=1)
        df['month']= df['date_and_time'].dt.month_name()
        df['day'] = df['date_and_time'].dt.day_name()
        df['hour'] = df['date_and_time'].dt.hour
        df['date'] = df['date_and_time'].dt.date
        df['character_length']= df['message'].str.len()
        df['word_length'] = df['message'].apply((lambda x: len(x.split(" "))))
        time_bins= [-1, 6, 12, 16, 19, 24]
        time_labels= ['Midnight', 'Morning', 'Afternoon', 'Evening', 'Night']
        df['part_of_day']= pd.cut(df['hour'], bins=time_bins, labels=time_labels)
        df['hour']=df['hour'].apply(lambda x : f"0{x}:00" if x <10 else f"{x}:00")
        df['is_url']= df['message'].apply((lambda x: bool(re.search(r"(http:/|https:/)", str(x)))))
        df['response_time'] = df['date_and_time'].diff()
        df = df.reset_index().drop('index', axis=1)
        return df
