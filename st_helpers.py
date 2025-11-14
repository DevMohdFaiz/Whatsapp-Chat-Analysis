import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from helpers import extract_chat_data, preprocess_df
from pathlib import Path
from zipfile import ZipFile

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
                    file_content = file.read().decode("utf")
    return file_content