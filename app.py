import io
import helpers
import importlib
import math
import nltk
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
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('punkt_tab')
importlib.reload(helpers)
from helpers import extract_chat_data, preprocess_df, vader_sent_analyzer, unzip_chat_for_st, get_member_texts, generate_word_cloud


st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="💬", layout="centered", initial_sidebar_state="expanded")
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">  
    <style>
    .main {
        padding: 0rem 1rem;
    }.stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }.h1 {
        color: #1f77b4;
    }.insight-box {
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
    }.member-profile-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        text-align: center;
        height: 30vh;
    }.member-profile-label {
        color: #666;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 10px;
    }.member-profile-value {
        color: #1f77b4;
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }.member-profile-delta {
        color: #333;
        font-size: 14px;
        background-color: #f0f0f0;
        padding: 4px 8px;
        border-radius: 4px;
    }.summary-card{
        background-color: #f8f9fa;
        padding: 30px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 30px;
    }.summary-card-inner{
        font-size: 16px;
        color: #333; 
        line-height: 1.8;       
    }
    </style>
""", unsafe_allow_html=True)

st.title("WhatsApp Chat Analyzer") 
st.markdown("---")

with st.sidebar:
    st.header("📊 Dashboard Controls")    
    uploaded_file = st.file_uploader("Upload WhatsApp Chat (.zip)", type=['zip'])    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This dashboard analyzes you whatsapp chats and provides interesting insights.
    
    **Features:**
    - Overall chat analysis
    - Chat trends over time
    - Group Member profile
    - Time-based patterns
    """)
# st.set_page_config(page_title="Performance Dashboard", layout="wide")


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
            <p><strong>{len(df):,}</strong><strong></strong></p>
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
        <div class="metric-card"><h3>Avg. msg. length</h3>
            <p><strong>{int(df['character_length'].mean())}</strong><strong></strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card"><h3>Avg. word length</h3>
            <p><strong>{int(df['word_length'].mean())}</strong><strong></strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.header("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_positive = df.groupby('sender')['sentiment_score'].mean().idxmax()
        most_positive_score = df.groupby('sender')['sentiment_score'].mean().max()
        
        st.markdown(f"""
        <div class="insight-box">
            <h3>Most Positive Member</h3>
            <p><strong>{most_positive}</strong> has the highest average sentiment score of <strong>{most_positive_score:.3f}</strong></p>
            <p>They tend to bring positive vibes to the conversation!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        happiest_day = df.groupby('date')['sentiment_score'].mean().idxmax()
        happiest_score = df.groupby('date')['sentiment_score'].mean().max()
        
        st.markdown(f"""
        <div class="insight-box">
            <h3>Happiest Day</h3>
            <p><strong>{happiest_day}</strong> was the most positive day with a sentiment score of <strong>{happiest_score:.3f}</strong></p>
            <p>Something great must have happened!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        most_active_member = df['sender'].value_counts()        
        st.markdown(f"""
        <div class="insight-box">
            <h3>Most active member</h3>
            <p><strong>{most_active_member.index[0]}</strong> is the most active group member <strong></strong></p>
            <p>They have sent a total of <strong>{most_active_member.values[0]:,}</strong> messages!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "👥 Members Profile", "⏰ Time Patterns", "💬 Messages", "Others"])
    with tab1:      
        st.subheader("Group members message Count")
        members_df = df.groupby('sender').size().sort_values(ascending=False)
        # st.write(members_df)
        fig = px.bar(x=members_df.index, y=members_df.values, orientation='v',)
        fig.update_layout(xaxis_title='Members', yaxis_title='No. of messages', showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=False)
        col1, col2 = st.columns(2)        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=['Neutral', 'Positive', 'Negative'], color=['red', 'green', 'yellow'], hole=0.5)
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
        st.subheader("Members Profile")
        selected_member = st.selectbox("Select a member to analyze:",options=sorted(df['sender'].unique()),index=None)        
        if selected_member is not None:
            member_df = df[df['sender'] == selected_member].copy()   
            avg_msg_length = member_df['character_length'].mean()         
            st.markdown("---")
            st.markdown(f"## 📊 {selected_member}'s Profile")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div class="member-profile-card">
                        <div class="member-profile-label">Total Messages</div>
                        <div class="member-profile-value">{len(member_df):,}</div>
                        <div class="member-profile-delta">{len(member_df)/len(df)*100:.1f}% of total</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                avg_sentiment = member_df['sentiment_score'].mean()
                sentiment = "Positive" if avg_sentiment > 0 else "Negative"
                st.markdown(f"""
                    <div class="member-profile-card">
                        <div class="member-profile-label">Average Sentiment</div>
                        <div class="member-profile-value">{round(avg_sentiment, 2)}</div>
                        <div class="member-profile-delta">{sentiment}</div>                          
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                volatility = member_df['sentiment_score'].std()
                stability = "Stable" if volatility < 0.3 else "Unstable"
                st.markdown(f"""
                    <div class="member-profile-card">
                        <div class="member-profile-label">Mood Volatility</div>
                        <div class="member-profile-value">{volatility:.3f}</div> 
                        <div class="member-profile-delta">{stability}</div>                                                 
                    </div>
                """, unsafe_allow_html=True)

            with col4:                
                avg_msg_len_comment = "Verbose" if avg_msg_length > 50 else "Concise"
                st.markdown(f"""
                    <div class="member-profile-card">
                        <div class="member-profile-label">Avg msg. length</div>
                        <div class="member-profile-value">{avg_msg_length:.0f} chars</div>
                        <div class="member-profile-delta">{avg_msg_len_comment}</div>                          
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")          
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎭 Personality Traits")      
                
                traits = []                            
                if volatility > 0.4:
                    traits.append("🎢 **Emotional Rollercoaster** - Mood swings often")
                elif volatility < 0.2:
                    traits.append("😌 **Steady Eddie** - Very consistent mood")
                
                if avg_msg_length > 60:
                    traits.append("📝 **Storyteller** - Loves long messages")
                elif avg_msg_length < 30:
                    traits.append("⚡ **Straight to the point**")
                
                member_msg_pct = len(member_df) / len(df) * 100
                if member_msg_pct > 10:
                    traits.append("💬 **Active member!**")
                elif member_msg_pct < 10:
                    traits.append("🤫 **Quiet member**")
                                
                for trait in traits:
                    st.markdown(trait)
                    st.markdown("")
            
            with col2:
                st.markdown("### 📈 Activity Stats")
                
                # Most active time
                most_active_hour = member_df['hour'].mode()[0] if len(member_df) > 0 else 0
                st.markdown(f"⏰ **Most Active Hour:** {most_active_hour}:00")
                
                most_active_day = member_df['day'].mode()[0] if len(member_df) > 0 else "N/A"
                st.markdown(f"📅 **Most Active Day:** {most_active_day}")
                
                
                daily_activity = member_df['date'].value_counts().sort_index()
                longest_streak = 0
                for i in range(len(daily_activity) - 1):
                    if (daily_activity.index[i+1] - daily_activity.index[i]).days == 1:
                        longest_streak += 1
                st.markdown(f"🔥 **Longest Streak:** {longest_streak} days")
                
                st.markdown(f"📊 **Total Contribution:** {member_msg_pct:.1f}% of all messages")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Sentiment Distribution")
                sentiment_counts = member_df['sentiment'].value_counts()
                fig = px.pie(values=sentiment_counts.values,names=sentiment_counts.index,color=sentiment_counts.index,
                    color_discrete_map={'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}, hole=0.5)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=300, showlegend=False, annotations=[dict(text=selected_member, x=0.5, y=0.5, font_size=16, showarrow=False)])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Sentiment Over Time")
                member_daily = member_df.groupby(member_df['date'])['sentiment_score'].mean().reset_index()
                fig = px.line(
                    member_daily,
                    x='date',
                    y='sentiment_score',
                    markers=True
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Neutral")
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Sentiment Score",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⏰ Activity by Hour")
                hourly_activity = member_df['hour'].value_counts().sort_index()
                fig = px.bar(
                    x=hourly_activity.index,
                    y=hourly_activity.values,
                    color=hourly_activity.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    xaxis_title="Hour of Day",
                    yaxis_title="Message Count",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📅 Activity by Day")
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_activity = member_df['day'].value_counts().reindex(day_order, fill_value=0)
                fig = px.bar(
                    x=day_activity.index,
                    y=day_activity.values,
                    color=day_activity.values,
                    color_continuous_scale='Greens'
                )
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Message Count",
                    height=300,
                    showlegend=False
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Member's Most Emotional Messages
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Most Positive Messages")
                top_positive = member_df.nlargest(3, 'sentiment_score')[['message', 'sentiment_score', 'date']]
                for idx, row in top_positive.iterrows():
                    with st.expander(f"Score: {row['sentiment_score']:.3f} - {row['date']}"):
                        st.write(row['message'])
            
            with col2:
                st.markdown("### Most Negative Messages")
                top_negative = member_df.nsmallest(3, 'sentiment_score')[['message', 'sentiment_score', 'date']]
                for idx, row in top_negative.iterrows():
                    with st.expander(f"Score: {row['sentiment_score']:.3f} - {row['date']}"):
                        st.write(row['message'])

        # member_sentiment = df.groupby('sender')['sentiment_score'].mean().sort_values(ascending=True)
        # fig = px.bar(x=member_sentiment.values, y=member_sentiment.index, orientation='v', color=member_sentiment.values, color_continuous_scale=["#1b84ee", "#151bb8", "#0058f0"])
        # fig.update_layout(xaxis_title="Average Sentiment Score",yaxis_title="Member",height=600,showlegend=False)
        # st.plotly_chart(fig, use_container_width=True)
        
        # with col2:
        #     member_dist = df.groupby(['sender', 'sentiment']).size().unstack(fill_value=0)
        #     fig = px.bar(member_dist,  color_discrete_map={'positive': "#1b84ee",'neutral': "#151bb8",'negative': "#0058f0"})
        #     fig.update_layout(xaxis_title="Member",yaxis_title="Message Count",height=400)
        #     st.plotly_chart(fig, use_container_width=True)
    
    with tab3:        
        st.subheader("Time Patterns")
        time_order_dict = {
        'day_order':['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'month_order': ['June', 'July', 'August', 'September', 'October'],
        'hour_order': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
        }
        freq_cols = ['date', 'month', 'day', 'hour']
        fig = make_subplots(rows=len(freq_cols), cols=1, subplot_titles=[f"Message frequency by {col.capitalize()}" for col in freq_cols])

        for idx, col_name in enumerate(freq_cols):
            df_plot = df.groupby(col_name).size().reindex(time_order_dict[f"{col_name}_order"]).reset_index(name='count') if col_name != 'date' else df.groupby(col_name).size().reset_index(name='count')
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

        st.subheader("Sentiment Patterns Over Time")        
        col1, col2 = st.columns(2)        
        with col1:
            hourly_sentiment = df.groupby('hour')['sentiment_score'].mean()
            fig = px.line(x=hourly_sentiment.index, y=hourly_sentiment.values, markers=True)
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(xaxis_title="Hour of Day",yaxis_title="Average Sentiment Score",title="Mood Throughout the Day",height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            daily_sentiment = df.groupby('day')['sentiment_score'].mean().reindex(time_order_dict['day_order'])            
            fig = go.Figure([go.Bar(x=daily_sentiment.index,y=daily_sentiment.values,marker_color=['green' if x > 0 else 'red' for x in daily_sentiment.values])])
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.update_layout(xaxis_title="Day of Week",yaxis_title="Average Sentiment Score",title="Which Days Are Happiest?",height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Messages")
            group_members= list(df['sender'].unique())
            chosen_member = st.selectbox("Select a group member to see their most used words", options=group_members, index=None)
            member_messages = df[df['sender']==chosen_member]['message']
            # member_tokens = word_tokenize(" ".join(t.lower() for t in member_messsages))
            # member_texts = " ".join([t for t in member_tokens if t.isalpha() and t not in ENGLISH_STOP_WORDS])
            member_texts = get_member_texts(member_messages)
            if len(member_texts)>0:
                fig, ax = plt.subplots(figsize=(5,5))
                word_cloud = generate_word_cloud(member_texts)
                ax.imshow(word_cloud)
                ax.axis('off')
                ax.set_title(f"{chosen_member}'s most frequent words", fontsize=15)   
                st.pyplot(fig, use_container_width=False)                 


    with tab5:
        st.header("Others")
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
    st.markdown("""
    ## 👋 Welcome to WhatsApp Chat Analyzer! 
    
    ### How to get started:
    
    1. **Export your WhatsApp chat**
       - Open WhatsApp chat
       - Click ⋮ (menu) → More → Export chat
       - Choose "Without Media"
    
    2. **Upload the .zip file** via the sidebar
    
    3. **Explore the insights!**
       - See overall chat analysis
       - Analyze individual members profile
       - Discover time-based patterns
       - Find most positive/negative messages
    
    ### What you'll discover:
    
    - 📊 Overall mood of the conversation
    - 👥 The most active and least active group members
    - ⏰ Chat patterns over time
    - 💬 Most emotional messages
    
    ### Privacy Note:
    
    🔒 Your data is processed locally and **not stored** on any server. 
    Everything happens in your browser session.
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by Streamlit</p>
    <p>Copyright 2025</p>
</div>
""", unsafe_allow_html=True)