import re
import io
import nltk
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from zipfile import ZipFile
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path
from wordcloud import WordCloud
from typing import List
# from nltk.corpus import stopwords

nltk.download('stopwords')
english_stopwords = set(nltk.corpus.stopwords.words('english'))

vader_analyzer = SentimentIntensityAnalyzer()


def unzip_chat(whatsapp_file_path: str):
    """
    Unzips the whatsapp.zip file and returns the loaded .txt file as a list
    Params: whatsapp_file_path= Path of the whatsapp zip file
    """
    with ZipFile(f"{whatsapp_file_path}.zip", "r") as f:
        f.extractall()
    with open(f"{whatsapp_file_path}.txt", "r", encoding='utf-8') as file:
        whatsapp_chat = file.readlines()
    return whatsapp_chat

def unzip_chat_for_st(uploaded_file: ZipFile):
    if uploaded_file is not None:
        zip_bytes = io.BytesIO(uploaded_file.read())

        with ZipFile(zip_bytes, "r") as zip_f:
            unzipped_file = [f for f in zip_f.namelist() if f.endswith('.txt')]
            if not unzipped_file:
                st.warning("No .txt file(s) found!")
            else:
                txt_file = unzipped_file[0]
                with zip_f.open(txt_file, "r") as file:
                    file_content = file.read().decode("utf-8")
                    file_content = file_content.splitlines(keepends=True)
        return file_content
    

def extract_chat_data(whatsapp_chats: List) -> pd.DataFrame:
    """
    Extracts important information like date, time, sender and message content from whatsapp.zip file 
    and returns a pandas dataframe with the columns: date, time, sender and message
    Args
        whatsapp_file_path: Path of the whatsapp zip file
    """

    extracted_data_dict = {
        'date': [], 'time': [], 'sender': [], 'message': []
    }
    
    for chat in whatsapp_chats:
        try:
            date, time = re.search(r"\d+/\d+/\d+", chat), re.search(r"\d+:\d+(?:\s+\w+)?", chat)
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
    
    return pd.DataFrame({'date': extracted_data_dict['date'], 'time': extracted_data_dict['time'],\
               'sender': extracted_data_dict['sender'], 'message': extracted_data_dict['message']})


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess and clean the dataframe

    Args:
        df: pandas dataframe
    """
    df = df.iloc[1:, :]
    df['sender'].replace('tmak 3', 'Tmak', inplace=True)
    df = df[~df['message'].isin(['<Media omitted>', 'Waiting for this message'])].reset_index()
    if df[df['time'].str.contains('AM|PM')].shape[0]>df.shape[0] * .5:
        df['time'] = pd.to_datetime(df['time'], format="%I:%M %p", errors='coerce').dt.strftime("%H:%M")
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



def simple_tokenize(text: str) -> List:
    return [word for word in re.findall(r"\b[a-zA-Z]+\b", text.lower()) if len(word)>1]

def get_member_texts(member_texts):
    all_member_words = []
    for msg in member_texts:
        words = [w for w in simple_tokenize(msg) if w not in english_stopwords]
        all_member_words.extend(words)
        # assert len(all_member_words), len([w for w in all_member_words if w.isalpha()])
    return " ".join(all_member_words)

@st.cache_data(ttl=3600)
def generate_word_cloud(words):
    if len(words)>0:
        wc = WordCloud(height=600, width=600, colormap='viridis', background_color='white').generate(words)
        wc = wc.to_image()
    else:
        wc = None
    return wc


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    messages = df['message']
    sentiment_results_dict = {'sentiment': [], 'sentiment_score': []}
    for msg in messages:
        sentiment_score= vader_analyzer.polarity_scores(msg)['compound']
        if (sentiment_score>=0.5):
            sentiment = 1
        elif (sentiment_score> -0.5 and sentiment_score<0.5):
            sentiment = 0.5
        else:
            sentiment = 0
        sentiment_results_dict['sentiment'].append(sentiment)
        sentiment_results_dict['sentiment_score'].append(sentiment_score)
    # sentiment_analysis_results= vader_sent_analyzer(messages=messages)
    df['sentiment'] = sentiment_results_dict['sentiment']
    df['sentiment_score'] = sentiment_results_dict['sentiment_score']
    return df