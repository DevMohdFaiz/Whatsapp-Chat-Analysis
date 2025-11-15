import io
import helpers, st_helpers
import importlib
import math
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path
from zipfile import ZipFile
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from nltk import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

importlib.reload(helpers)
importlib.reload(st_helpers)
from helpers import extract_chat_data, preprocess_df, vader_sent_analyzer
from st_helpers import generate_word_cloud

# Page config
st.set_page_config(
    page_title="WhatsApp Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        color: black;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
        height: 43vh;
    }.metric-card {
    background-color: #1e1e1e;
    color: white;
    padding: 16px 20px;
    border-radius: 12px;
    border-left: 6px solid #1f77b4;
    margin: 8px 0;
    font-size: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }.metric-card h3 {
        margin: 0;
        padding: 0;
        font-size: 22px;
        font-weight: 600;
        color: #4dabff;
    }.metric-card p {
        margin: 6px 0 0 0;
        font-size: 26px;
        font-weight: 500;
    }.metric-card small {
        font-size: 14px;
        opacity: 0.8;
    }

    </style>
""", unsafe_allow_html=True)

# Title
st.title("💬 WhatsApp Chat Sentiment Analyzer")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Dashboard Controls")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type=['txt'])
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This dashboard analyzes sentiment in your WhatsApp chats using VADER sentiment analysis.
    
    **Features:**
    - Overall sentiment distribution
    - Sentiment trends over time
    - Individual member analysis
    - Time-based patterns
    """)
uploaded_file = st.file_uploader("Upload your file")


def unzip_chat_for_st(uploaded_file):
    # st.header(bool(uploaded_file is not None))
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


st.set_page_config(page_title="Performance Dashboard", layout="centered")


if uploaded_file is not None:
    unzipped_chat = unzip_chat_for_st(uploaded_file=uploaded_file)
    chat_data = extract_chat_data(unzipped_chat)
    df = preprocess_df(chat_data)
    df = vader_sent_analyzer(df)


    st.header("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:    
        st.markdown(f"""
        <div class="metric-card"><h3>Total Messages</h3>
            <p><strong>{len(df)}</strong><strong></strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card"><h3>No. of members</h3>
            <p><strong>{df['sender'].nunique()}</strong><strong></strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card"><h3>Avg. word length</h3>
            <p><strong>{int(df['word_length'].mean())}</strong><strong></strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Insights Section
    st.header("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_positive = df.groupby('sender')['sentiment_score'].mean().idxmax()
        most_positive_score = df.groupby('sender')['sentiment_score'].mean().max()
        
        st.markdown(f"""
        <div class="insight-box">
            <h3>😊 Most Positive Member</h3>
            <p><strong>{most_positive}</strong> has the highest average sentiment score of <strong>{most_positive_score:.3f}</strong></p>
            <p>They tend to bring positive vibes to the conversation!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        happiest_day = df.groupby('date')['sentiment_score'].mean().idxmax()
        happiest_score = df.groupby('date')['sentiment_score'].mean().max()
        
        st.markdown(f"""
        <div class="insight-box">
            <h3>🌟 Happiest Day</h3>
            <p><strong>{happiest_day}</strong> was the most positive day with a sentiment score of <strong>{happiest_score:.3f}</strong></p>
            <p>Something great must have happened!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        most_active_member = df['sender'].value_counts()        
        st.markdown(f"""
        <div class="insight-box">
            <h3>🌟 Most active member</h3>
            <p><strong>{most_active_member.index[0]}</strong> is the most active group member <strong></strong></p>
            <p>They have sent a total of <strong>{most_active_member.values[0]}</strong> messages!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualization Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "👥 Members", "⏰ Time Patterns", "💬 Messages", "Others"])
    
    with tab1:
        col1, col2 = st.columns(2)        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=['Neutral', 'Positive', 'Negative'], color=['red', 'green', 'yellow'],
                hole=0.5, 
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout({'height':400})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Over Time")
            daily_sentiment = df.groupby(df['date'])['sentiment_score'].mean().reset_index()
            fig = px.line(daily_sentiment, x='date', y='sentiment_score', markers=True)
            fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Neutral")
            fig.update_layout(xaxis_title="Date",yaxis_title="Average Sentiment Score",height=400)
            st.plotly_chart(fig, use_container_width=True)

    
    with tab2:
        st.subheader("Sentiment by Member")
        fig = px.bar(df['sender'].value_counts().sort_values(ascending=False), orientation='v', title='Group members message count')
        fig.update_layout(xaxis_title='Members', yaxis_title='No. of messages',)
        st.plotly_chart(fig, use_container_width=False)
        col1, col2= st.columns(2)
        
        with col1:
            member_sentiment = df.groupby('sender')['sentiment_score'].mean().sort_values(ascending=True)
            fig = px.bar(x=member_sentiment.values, y=member_sentiment.index, orientation='h', color=member_sentiment.values, color_continuous_scale=["#1b84ee", "#151bb8", "#0058f0"])
            fig.update_layout(xaxis_title="Average Sentiment Score",yaxis_title="Member",height=600,showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            member_dist = df.groupby(['sender', 'sentiment']).size().unstack(fill_value=0)
            fig = px.bar(member_dist,  color_discrete_map={'positive': "#1b84ee",'neutral': "#151bb8",'negative': "#0058f0"})
            fig.update_layout(xaxis_title="Member",yaxis_title="Message Count",height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:        
        st.subheader("Time Patterns")
        time_order_dict = {
        'day_order':['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
        'month_order': ['June', 'July', 'August', 'September', 'October'],
        'hour_order': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00',
            '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
        }
        freq_cols = ['date', 'month', 'day', 'hour']
        fig = make_subplots(rows=len(freq_cols), cols=1, subplot_titles=[f"Message frequency by {col.capitalize()}" for col in freq_cols])

        for idx, col_name in enumerate(freq_cols):
            df_plot = df.groupby(col_name).size().reindex(time_order_dict[f"{col_name}_order"]).reset_index(name='count') if col_name != 'date' else df.groupby(col_name).size().reset_index(name='count')
            # if col_name != 'date': 
            #     df_plot = df.groupby(col_name).size().reindex(time_order_dict[f"{col_name}_order"]).reset_index(name='count')
            # else:
            #     df_plot = df.groupby(col_name).size().reset_index(name='count')
            trace = px.line(df_plot, x=col_name, y='count').data[0]
            fig.add_trace(trace, row=idx + 1,  col=1)
        fig.update_layout(height=1000, title_text="Message Frequency Distributions")
        st.plotly_chart(fig, use_container_width=True)


        fig = make_subplots(rows=len(freq_cols), cols=1, subplot_titles=[f"Sentiment Distribution by {col.capitalize()}" for col in freq_cols])
        for idx, col_name in enumerate(freq_cols):
            sentiment_df = df.groupby(col_name)['sentiment'].mean().reindex(time_order_dict[f"{col_name}_order"]).reset_index(name='count') if col_name != 'date' else df.groupby(col_name)['sentiment'].mean().reset_index(name='count') 
            trace = px.line(sentiment_df, x=col_name, y='count').data[0]
            fig.add_trace(trace, row=idx + 1,  col=1)
        fig.update_layout(height=1000, title_text="Sentiment Score Distributions")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(df.groupby('date')['message'].count(), title='Message Count Over time')
        fig.update_layout({'xaxis_title': 'Date', 'yaxis_title': 'No. of messages', 'showlegend':False})
        st.plotly_chart(fig, use_container_width=True)
        # st.subheader("Time-Based Patterns")
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     hourly_sentiment = df.groupby('hour')['sentiment_score'].mean()
        #     fig = px.line(
        #         x=hourly_sentiment.index,
        #         y=hourly_sentiment.values,
        #         markers=True
        #     )
        #     fig.add_hline(y=0, line_dash="dash", line_color="red")
        #     fig.update_layout(
        #         xaxis_title="Hour of Day",
        #         yaxis_title="Average Sentiment Score",
        #         title="Mood Throughout the Day",
        #         height=400
        #     )
        #     st.plotly_chart(fig, use_container_width=True)
        
    #     with col2:
    #         df['day_name'] = df['datetime'].dt.day_name()
    #         day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    #         daily_sentiment = df.groupby('day_name')['sentiment_score'].mean().reindex(day_order)
            
    #         fig = go.Figure(data=[
    #             go.Bar(
    #                 x=daily_sentiment.index,
    #                 y=daily_sentiment.values,
    #                 marker_color=['green' if x > 0 else 'red' for x in daily_sentiment.values]
    #             )
    #         ])
    #         fig.add_hline(y=0, line_dash="dash", line_color="black")
    #         fig.update_layout(
    #             xaxis_title="Day of Week",
    #             yaxis_title="Average Sentiment Score",
    #             title="Which Days Are Happiest?",
    #             height=400
    #         )
    #         st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Messages")
        group_members= list(df['sender'].unique())
        chosen_member = st.selectbox("Select a group member to generate a word cloud", options=group_members, index=None)
        member_messsages = df[df['sender']==chosen_member]['message']
        member_tokens = word_tokenize(" ".join(t.lower() for t in member_messsages))
        member_texts = " ".join([t for t in member_tokens if t.isalpha() and t not in ENGLISH_STOP_WORDS])
        if len(member_texts) >0:
            word_cloud = generate_word_cloud(member_texts)
            fig = go.Figure()
            fig.update_layout(title=f"{chosen_member}'s Frequent Words",margin=dict(l=0, r=0, t=50, b=0),xaxis_visible=False,yaxis_visible=False)
            fig.add_trace(go.Image(z=word_cloud))
            fig.update_traces(zsmooth=False)
            st.plotly_chart(fig, use_container_width=True)
        

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌟 Most Positive Messages")
            top_positive = df.nlargest(5, 'sentiment_score')[['sender', 'message', 'sentiment_score']]
            for idx, row in top_positive.iterrows():
                st.markdown(f"""
                **{row['sender']}** (Score: {row['sentiment_score']:.3f})  
                _{row['message'][:100]}..._
                """)
                st.markdown("---")
        
        with col2:
            st.markdown("### 💔 Most Negative Messages")
            top_negative = df.nsmallest(5, 'sentiment_score')[['sender', 'message', 'sentiment_score']]
            for idx, row in top_negative.iterrows():
                st.markdown(f"""
                **{row['sender']}** (Score: {row['sentiment_score']:.3f})  
                _{row['message'][:100]}..._
                """)
                st.markdown("---")
    
    # # Download section
    # st.markdown("---")
    # st.header("📥 Export Results")
    
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     csv = df.to_csv(index=False).encode('utf-8')
    #     st.download_button(
    #         label="Download CSV",
    #         data=csv,
    #         file_name="sentiment_analysis.csv",
    #         mime="text/csv"
    #     )
    col1, col2= st.columns(2)
    # with st.container():
        # Create summary report
    st.markdown(
         f"""
        . * 5
        WhatsApp Sentiment Analysis Report
        
        
        Total Messages: {len(df)}
        Date Range: {df['date'].min()} to {df['date'].max()}
        
        Overall Sentiment:
        - Average Score: {df['sentiment_score'].mean():.3f}
        # - Positive: {(df['sentiment']=='positive').sum()} (df['sentiment']==1).sum() / len(df)*100
        - Neutral: {(df['sentiment']=='neutral').sum()} ({(df['sentiment']==0.5).sum()/len(df)*100:.1f}%)
        - Negative: {(df['sentiment']=='negative').sum()} ({(df['sentiment']==0).sum()/len(df)*100:.1f}%)
        
    #     Most Positive Member: {most_positive}
    #     Happiest Day: {happiest_day}
    #     """
    ) 
    #     st.download_button(
    #         label="Download Report",
    #         data=summary,
    #         file_name="sentiment_report.txt",
    #         mime="text/plain"
    #     )
    
    # with col3:
    #     st.info("More export options coming soon!")


    with tab5:
        st.header("Others")
        st.subheader("Choose to see most/lest active members")
        label = st.selectbox("Choose to see top/least active members", options=['Top', 'Least'], index=None)
        if label== 'Top':
            chosen_members = df['sender'].value_counts().reset_index().rename({'sender': 'member', 'count':'message_count'}, axis=1).nlargest(10, 'message_count')
        else:
            chosen_members = df['sender'].value_counts().reset_index().rename({'sender': 'member', 'count':'message_count'}, axis=1).nsmallest(10, 'message_count')
        if label is not None:
            chart = alt.Chart(chosen_members).mark_bar().encode(
                x=alt.X('message_count:Q', title='Number of Messages'),
                y=alt.Y('member:N', title='Participant', sort='-x'),
                color=alt.Color('member:N', legend=None),
                tooltip=['member:N', 'message_count:Q']
            ).properties(width=500, height=550, title=f'{label} active group members').configure_axis(grid=False)
            st.altair_chart(chart, use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to WhatsApp Sentiment Analyzer!
    
    ### How to get started:
    
    1. **Export your WhatsApp chat**
       - Open WhatsApp chat
       - Click ⋮ (menu) → More → Export chat
       - Choose "Without Media"
    
    2. **Upload the .txt file** using the sidebar
    
    3. **Explore the insights!**
       - See overall sentiment trends
       - Analyze individual members
       - Discover time-based patterns
       - Find most positive/negative messages
    
    ### What you'll discover:
    
    - 📊 Overall mood of the conversation
    - 👥 Who's the most positive/negative person
    - ⏰ Best and worst times of day
    - 📅 Happiest and saddest days
    - 💬 Most emotional messages
    
    ### Privacy Note:
    
    🔒 Your data is processed locally and **not stored** on any server. 
    Everything happens in your browser session.
    """)
    
    # Sample visualization
    st.markdown("---")
    st.subheader("📸 Preview")
    st.image("https://via.placeholder.com/800x400.png?text=Sample+Dashboard+Preview", 
             caption="Sample dashboard view")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Made with ❤️ using Streamlit | Powered by VADER Sentiment Analysis</p>
</div>
""", unsafe_allow_html=True)