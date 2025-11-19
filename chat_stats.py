import pandas as pd
from typing import Optional, Dict
from helpers import unzip_chat, preprocess_df, extract_chat_data
from groq import Groq
from dotenv import get_key
import streamlit as st
from ai_integration import _get_groq_client, _call_groq_api



def generate_basic_stats(df: pd.DataFrame)->Dict[str, str]:
    if df.empty:
        print(f"Empty Dataframe")
    sample_msg = df['message'].sample(300).tolist()
    basic_stats = {
        'total_messages': len(df),
        'date_range_start': df['date'].min(),
        'date_range_end': df['date'].max(),
        'num_members': df['sender'].nunique(),
        'members': df['sender'].unique().tolist(),
        'avg_sentiment': df['sentiment_score'].mean(),
        'avg_volatility': df['sentiment_score'].mean(),
        'positive_count': (df['sentiment'] == 1).sum(),
        'neutral_count': (df['sentiment'] == 0.5).sum(),
        'negative_count': (df['sentiment'] == 0).sum(),
        'positive_pct': (df['sentiment'] == 1).sum() / len(df),
        'neutral_pct':  (df['sentiment'] == 0.5).sum() / len(df),
        'negative_pct': (df['sentiment'] == 0).sum() / len(df),
        'most_active_date': df['date'].mode()[0],
        'most_active_day': df['day'].mode()[0],
        'most_active_month':df['month'].mode()[0],
        'most_active_part_of_day': df['part_of_day'].mode()[0],
        'sample_messages': ", ".join([m for m in sample_msg])
    }
    return basic_stats


def generate_member_stats(df: pd.DataFrame, member: str, model="llama-3.3-70b-versatile")->str:
    if df.empty or 'sender' not in df.columns:
        return {'error': "Insufficeint data"}

    if member not in df['sender'].unique():
        return {'error': "{member} not found"}

    member_df= df[df['sender']==member].copy()

    msg_count= member_df.shape[0]
    msg_percent = (msg_count/df.shape[0])*100
    avg_word_length = member_df['word_length'].mean()
    avg_char_count =member_df['character_length'].mean()
    avg_volatility = member_df['sentiment_score'].std()
    avg_sentiment = member_df['sentiment'].mode()[0]   
    positive_count= (member_df['sentiment'] == 1).sum()
    neutral_count= (member_df['sentiment'] == 0.5).sum()
    negative_count= (member_df['sentiment'] == 0).sum()
    positive_pct= (member_df['sentiment'] == 1).sum() / len(df)
    neutral_pct = (member_df['sentiment'] == 0.5).sum() / len(df)
    negative_pct = (member_df['sentiment'] == 0).sum() / len(df)

    if avg_sentiment== 1:
        avg_sentiment='positive'
    elif avg_sentiment ==0.5:
        avg_sentiment='neutral'
    elif avg_sentiment==0:
        avg_sentiment= 'negative'
    most_active_date = member_df['date'].mode()[0]
    most_active_day = member_df['day'].mode()[0]
    most_active_month =member_df['month'].mode()[0]
    most_active_part_of_day = member_df['part_of_day'].mode()[0]

    # try:
    #     sample_messages= member_df['message'].sample(10).tolist()
    # except:
    #     sample_messages= member_df['message'].head(10).tolist()
    # sample_texts = ", ".join(m for m in sample_messages)

    prompt = f"""
        Analyze the messaging behaviour of {member} in this chat

        Statistics:
        - Total messages: {msg_count}
        - Average message length: {avg_word_length:.1f} words
        - Average sentiment score: {avg_volatility}
        - Average sentiment: {avg_sentiment} 
        - Average word_length: {avg_word_length}
        - Average character count: {avg_char_count}
        - Most active day: {most_active_date}
        - Most active day: {most_active_day} 
        - Most active month: {most_active_month}
        - Most active part of day: {most_active_part_of_day}
        - Positive message count: {positive_count}
        - Neutral message count {neutral_count}
        - Negative message count {negative_count}
        - Positive percent {positive_pct}
        - Neutral percent {neutral_pct}
        - Negative percent {negative_pct}

       
        Please provide insights on:
        1. Communication style
        2. Topics of interest
        3. Engagement patterns
        4. Sentiment trends
        5. Notable characteristics

        Provide your analysis:
        """
    

    # analysis= _call_groq_api(prompt, model=model, max_tokens=1024)

    return {
        'member': member,
        'statistics': {
            'msg_count': msg_count,
            'msg_percent': msg_percent,
            'avg_word_length': avg_word_length,
            'avg_char_count': avg_char_count,
            'avg_sentiment': avg_sentiment,
            'avg_volatility': avg_volatility,
            'most_active_date': most_active_date,
            'most_active_day': most_active_day,
            'most_active_month': most_active_month,
            'most_active_part_of_day': most_active_part_of_day
            # 'sample_messages': {sample_texts}
        },
        # 'insights': analysis
    }



def generate_all_members_stats(df: pd.DataFrame)-> Dict[str, Dict]:
    """Generate summary statistics for all group members"""
    if df.empty:
        return {'error': 'The provided dataframe seems to be empty'}
    
    all_members = df['sender'].unique()
    all_members_stats = []
    for member in all_members:
        all_members_stats.append(generate_member_stats(df=df, member=member))
        all_members_stats = sorted(all_members_stats, key=lambda x: x['statistics']['msg_count'], reverse=True)
    return all_members_stats
    

def generate_chat_summary(df: pd.DataFrame, period, model: str="llama-3.3-70b-versatile", max_msg=1000)->str:
    """
    Generate chat summary via AI.

    Uses llama-3.3-70b-versatile by default for high-quality comprehensive summaries.
    For faster summaries, use 'llama-3.1-8b-instant'.
    For highest quality, use 'openai/gpt-oss-120b'.

    Args:
        df: DataFrame with chat messages
        period: Time period ('all', 'daily', 'weekly', 'monthly')
        model: Groq model to use (default: llama-3.3-70b-versatile)
        max_messages: Maximum number of messages to include in summary

    Returns:
        Generated summary text
    """
    if df.empty or 'message' not in df.columns:
        return "No messages found."

    
    df_filtered = df.copy()
    if period != 'all' and 'date_and_time' in df.columns:
        if period == 'daily':
            df_filtered = df_filtered[df_filtered['date_and_time'] == df_filtered['date_and_time'].dt.date.max()]
        elif period == 'weekly':
            week_ago = df_filtered['date_and_time'].max() - pd.Timedelta(days=7)
            df_filtered = df_filtered[df_filtered['date_and_time'] >= week_ago]
        elif period == 'monthly':
            month_ago = df_filtered['date_and_time'].max() - pd.Timedelta(days=30)
            df_filtered = df_filtered[df_filtered['date_and_time'] >= month_ago]

    # Limit messages for context
    messages = df_filtered['message'].sample(max_msg).tolist()
    messages_text = "\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(messages)])

    # Get chat type
    chat_type = df.attrs.get('chat_type', 'group')
    chat_type_str = "group chat" if chat_type == 'group' else "individual chat"



def create_text_summary(df: pd.DataFrame, basic_stats: Dict, member_stats: Dict)-> str:
    """Convert statistics into a formatted text summary for the LLM"""
    
    summary = f"""
        WHATSAPP CHAT ANALYSIS SUMMARY
        ================================

        OVERVIEW:
        - Total Messages: {basic_stats['total_messages']:,}
        - Date Range: {basic_stats['date_range_start']} to {basic_stats['date_range_end']}
        - Number of Members: {basic_stats['num_members']}
        - Members: {', '.join(basic_stats['members'])}
        - Overall Sentiment Score: {basic_stats['avg_sentiment']:.3f}
        - Most Active Date: {basic_stats['most_active_date']}
        - Most Active Day of Week: {basic_stats['most_active_day']}
        - Most Active Day month: {basic_stats['most_active_month']}
        - Most Active Part of day: {basic_stats['most_active_part_of_day']}


        OVERALL SENTIMENT BREAKDOWN:
        - Positive: {basic_stats['positive_count']:,} messages)
        - Neutral: {basic_stats['neutral_count']:,} messages)
        - Negative: {basic_stats['negative_count']:,} messages)

        MOOD INTERPRETATION:
    """
    
    # Add mood interpretation
    avg_sent = basic_stats['avg_sentiment']
    if avg_sent > 0.3:
        summary += "- The overall mood is VERY POSITIVE. This chat has great vibes!\n"
    elif avg_sent > 0.1:
        summary += "- The overall mood is POSITIVE. Generally good energy.\n"
    elif avg_sent > -0.1:
        summary += "- The overall mood is NEUTRAL. Balanced conversation.\n"
    elif avg_sent > -0.3:
        summary += "- The overall mood is SLIGHTLY NEGATIVE. Some concerns or complaints.\n"
    else:
        summary += "- The overall mood is NEGATIVE. Many complaints or frustrations.\n"
    
    summary += f"""\nSample messages {basic_stats['sample_messages']}\n"""
    summary += "\nMEMBER ANALYSIS:\n"
    summary += "=" * 50 + "\n\n"
    
    # Add each member's stats
    for idx, member in enumerate(member_stats, 1):
        summary += f"{idx}. {member['member']}"
        
        # Mark most active
        if idx == 1:
            summary += " (MOST ACTIVE)"
        
        summary += f"\n"
        summary += f"   - Total Messages: {member['statistics']['msg_count']:,} ({member['statistics']['msg_percent']:.2f}% of all messages)\n"
        summary += f"   - Average Sentiment: {member['statistics']['avg_sentiment']}\n"
        summary += f"   - Average word length: {member['statistics']['avg_word_length']:.0f} words\n"
        summary += f" - Average character count: {member['statistics']['avg_char_count']:.0f} characters\n"
        summary += f" - Most active date: {member['statistics']['most_active_date']} \n"
        summary += f" - Most active day: {member['statistics']['most_active_day']} \n"
        summary += f" - Most active month: {member['statistics']['most_active_month']} \n"
        summary += f" - Most active part of day: {member['statistics']['most_active_part_of_day']} \n"
        # summary += f" - Sample messages: {member['statistics']['sample_messages']}\n"
        summary += "\n"
    
    # Add key insights
    summary += "\nKEY INSIGHTS:\n"
    summary += "=" * 50 + "\n"
    
    # Most positive member
    most_positive = max(member_stats, key=lambda x: x['statistics']['avg_sentiment'])
    summary += f"🌟 Most Positive Member: {most_positive['member']} (sentiment: {most_positive['statistics']['avg_sentiment']})\n"
    
    # Most negative member
    most_negative = min(member_stats, key=lambda x: x['statistics']['avg_sentiment'])
    summary += f"😔 Most Negative Member: {most_negative['member']} (sentiment: {most_negative['statistics']['avg_sentiment']})\n"
    
    # Most active member
    most_active = member_stats[0]  # Already sorted by message count
    summary += f"💬 Most Active Member: {most_active['member']} ({most_active['statistics']['msg_count']:,} messages)\n"
    
    # Most volatile member
    most_volatile = max(member_stats, key=lambda x: x['statistics']['avg_volatility'])
    summary += f"🎭 Most Emotionally Volatile: {most_volatile['member']} (volatility: {most_volatile['statistics']['avg_volatility']:.3f})\n"
    
    # Most stable member
    most_stable = min(member_stats, key=lambda x: x['statistics']['avg_volatility'])
    summary += f"😌 Most Emotionally Stable: {most_stable['member']} (volatility: {most_stable['statistics']['avg_volatility']:.3f})\n"
    
    
    
    summary += "\n" + "=" * 50 + "\n"
    summary += "END OF SUMMARY\n"
    
    return summary

@st.cache_data
def generate_chat_summary_context(df: pd.DataFrame)-> Dict[str, str]:
    basic_stats = generate_basic_stats(df)
    all_members_stats = generate_all_members_stats(df)
    chat_summary = create_text_summary(df, basic_stats, member_stats=all_members_stats)
    return {
        'summary': chat_summary,
        'basic_stats': basic_stats,
        'all_members_stats': all_members_stats,
        'df': df
    }