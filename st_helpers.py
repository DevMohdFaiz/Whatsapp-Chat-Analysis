import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from helpers import extract_chat_data, preprocess_df
from pathlib import Path
from zipfile import ZipFile
from wordcloud import WordCloud

def unzip_chat_for_st(uploaded_file):
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


@st.cache_data(show_spinner=True)
def generate_word_cloud(texts):
    word_cloud = WordCloud(width=3000, height=2000, colormap='viridis', background_color='white').generate(texts)
    return word_cloud.to_array()