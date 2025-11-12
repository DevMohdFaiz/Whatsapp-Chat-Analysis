"""
AI Insights Module

AI-powered analysis using Groq LLMs for chat summarization, topic modeling, and insights.
"""

import os
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from groq import Groq
except ImportError:
    Groq = None
    print("Warning: groq package not installed. AI features will not work.")


def _get_groq_client() -> Optional['Groq']:
    """Get Groq client instance."""
    if Groq is None:
        return None

    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )

    return Groq(api_key=api_key)


def _call_groq_api(
    prompt: str,
    model: str = 'llama-3.1-8b-instant',
    max_tokens: int = 1024,
    temperature: float = 0.7,
    retries: int = 3
) -> str:
    """
    Internal Groq API wrapper with error handling and retries.

    Args:
        prompt: Prompt to send to the model
        model: Model name (default: llama-3.1-8b-instant)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        retries: Number of retry attempts

    Returns:
        Model response text

    Raises:
        ValueError: If Groq client cannot be initialized
        RuntimeError: If API call fails after retries
    """
    client = _get_groq_client()
    if client is None:
        raise ValueError("Groq client not available. Please install groq package and set GROQ_API_KEY.")

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Groq API call failed after {retries} attempts: {str(e)}")
            continue

    raise RuntimeError("Unexpected error in Groq API call")


def generate_chat_summary(
    df: pd.DataFrame,
    period: str = 'all',
    model: str = 'llama-3.1-8b-instant',
    max_messages: int = 1000
) -> str:
    """
    Generate chat summary using AI.

    Args:
        df: DataFrame with chat messages
        period: Time period ('all', 'daily', 'weekly', 'monthly')
        model: Groq model to use
        max_messages: Maximum number of messages to include in summary

    Returns:
        Generated summary text
    """
    if df.empty or 'message' not in df.columns:
        return "No messages to summarize."

    # Filter by period if needed
    df_filtered = df.copy()
    if period != 'all' and 'date_and_time' in df.columns:
        if period == 'daily':
            df_filtered = df_filtered[df_filtered['date_and_time'].dt.date == df_filtered['date_and_time'].dt.date.max()]
        elif period == 'weekly':
            week_ago = df_filtered['date_and_time'].max() - pd.Timedelta(days=7)
            df_filtered = df_filtered[df_filtered['date_and_time'] >= week_ago]
        elif period == 'monthly':
            month_ago = df_filtered['date_and_time'].max() - pd.Timedelta(days=30)
            df_filtered = df_filtered[df_filtered['date_and_time'] >= month_ago]

    # Limit messages for context
    messages = df_filtered['message'].head(max_messages).tolist()
    messages_text = "\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(messages)])

    # Get chat type
    chat_type = df.attrs.get('chat_type', 'group')
    chat_type_str = "group chat" if chat_type == 'group' else "individual chat"

    prompt = f"""Analyze the following {chat_type_str} messages and provide a comprehensive summary.

{messages_text}

Please provide:
1. Main topics discussed
2. Overall tone and sentiment
3. Key highlights or important moments
4. Any notable patterns or trends

Summary:"""

    return _call_groq_api(prompt, model=model, max_tokens=1024)


def identify_topics(
    df: pd.DataFrame,
    n_topics: int = 5,
    model: str = 'llama-3.1-8b-instant',
    max_messages: int = 500
) -> List[Dict[str, str]]:
    """
    Identify main topics in the chat using AI.

    Args:
        df: DataFrame with chat messages
        n_topics: Number of topics to identify
        model: Groq model to use
        max_messages: Maximum number of messages to analyze

    Returns:
        List of dictionaries with topic information
    """
    if df.empty or 'message' not in df.columns:
        return []

    # Sample messages for analysis
    messages = df['message'].head(max_messages).tolist()
    messages_text = "\n".join([f"- {msg}" for msg in messages])

    prompt = f"""Analyze the following chat messages and identify the {n_topics} main topics discussed.

Messages:
{messages_text}

For each topic, provide:
- Topic name
- Brief description
- Key points or themes

Format your response as a numbered list with clear topic names and descriptions."""

    response = _call_groq_api(prompt, model=model, max_tokens=1024)

    # Parse response into structured format
    topics = []
    lines = response.split('\n')
    current_topic = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to extract topic information
        if line[0].isdigit() or line.startswith('-') or line.startswith('*'):
            if current_topic:
                topics.append(current_topic)
            current_topic = {'name': line, 'description': ''}
        elif current_topic:
            current_topic['description'] += ' ' + line

    if current_topic:
        topics.append(current_topic)

    # Limit to n_topics
    return topics[:n_topics]


def generate_insights(
    df: pd.DataFrame,
    chat_type: str = 'group',
    model: str = 'llama-3.1-8b-instant'
) -> Dict[str, str]:
    """
    Generate smart insights about the chat.

    Args:
        df: DataFrame with analyzed chat data
        chat_type: 'group' or 'individual'
        model: Groq model to use

    Returns:
        Dictionary with various insights
    """
    if df.empty:
        return {"error": "No data to analyze"}

    insights = {}

    # Prepare data summary
    total_messages = len(df)
    date_range = ""
    if 'date_and_time' in df.columns:
        start_date = df['date_and_time'].min()
        end_date = df['date_and_time'].max()
        date_range = f"from {start_date.date()} to {end_date.date()}"

    if chat_type == 'group' and 'sender' in df.columns:
        unique_senders = df['sender'].nunique()
        top_sender = df['sender'].value_counts().index[0] if len(df) > 0 else "N/A"
        avg_sentiment = df['sentiment'].mean() if 'sentiment' in df.columns else None

        summary_text = f"""
Chat Statistics:
- Total messages: {total_messages}
- Date range: {date_range}
- Number of participants: {unique_senders}
- Most active member: {top_sender}
- Average sentiment: {avg_sentiment:.2f} (if available)
"""
    else:
        summary_text = f"""
Chat Statistics:
- Total messages: {total_messages}
- Date range: {date_range}
"""

    # Get sample messages for context
    sample_messages = df['message'].head(100).tolist()
    messages_text = "\n".join([f"- {msg}" for msg in sample_messages])

    prompt = f"""Based on the following chat statistics and sample messages, provide intelligent insights:

{summary_text}

Sample Messages:
{messages_text}

Please provide insights on:
1. Communication patterns
2. Engagement levels
3. Conversation dynamics
4. Notable trends or behaviors
5. Recommendations or observations

Format your response clearly with numbered insights."""

    insights_text = _call_groq_api(prompt, model=model, max_tokens=1024)
    insights['general'] = insights_text

    return insights


def analyze_conversation_flow(
    df: pd.DataFrame,
    model: str = 'llama-3.1-8b-instant'
) -> Dict[str, str]:
    """
    Analyze conversation flow for individual chats.

    Args:
        df: DataFrame with individual chat data
        model: Groq model to use

    Returns:
        Dictionary with conversation flow analysis
    """
    if df.empty or 'sender' not in df.columns:
        return {"error": "Insufficient data for conversation flow analysis"}

    # Calculate response times
    if 'response_time' in df.columns:
        avg_response_time = df['response_time'].mean()
        response_times_text = f"Average response time: {avg_response_time}"
    else:
        response_times_text = "Response time data not available"

    # Get conversation sample
    sample_messages = df[['sender', 'message']].head(50)
    messages_text = "\n".join([
        f"{row['sender']}: {row['message']}"
        for _, row in sample_messages.iterrows()
    ])

    prompt = f"""Analyze the conversation flow of this individual chat:

{response_times_text}

Sample Conversation:
{messages_text}

Please analyze:
1. Conversation rhythm and pacing
2. Response patterns
3. Engagement levels
4. Communication style
5. Any notable conversation dynamics

Provide your analysis:"""

    analysis = _call_groq_api(prompt, model=model, max_tokens=1024)

    return {
        'flow_analysis': analysis,
        'avg_response_time': str(avg_response_time) if 'response_time' in df.columns else "N/A"
    }


def get_member_insights(
    df: pd.DataFrame,
    member: str,
    model: str = 'llama-3.1-8b-instant'
) -> Dict[str, str]:
    """
    Get AI-powered insights for a specific group member.

    Args:
        df: DataFrame with group chat data
        member: Name of the member to analyze
        model: Groq model to use

    Returns:
        Dictionary with member-specific insights
    """
    if df.empty or 'sender' not in df.columns:
        return {"error": "Insufficient data for member analysis"}

    if member not in df['sender'].values:
        return {"error": f"Member '{member}' not found in chat"}

    member_df = df[df['sender'] == member].copy()

    # Calculate statistics
    message_count = len(member_df)
    avg_length = member_df['word_length'].mean() if 'word_length' in member_df.columns else 0
    avg_sentiment = member_df['sentiment'].mean() if 'sentiment' in member_df.columns else None

    # Get sample messages
    sample_messages = member_df['message'].head(50).tolist()
    messages_text = "\n".join([f"- {msg}" for msg in sample_messages])

    prompt = f"""Analyze the messaging behavior of '{member}' in this group chat:

Statistics:
- Total messages: {message_count}
- Average message length: {avg_length:.1f} words
- Average sentiment: {avg_sentiment:.2f} (if available)

Sample Messages:
{messages_text}

Please provide insights on:
1. Communication style
2. Topics of interest
3. Engagement patterns
4. Sentiment trends
5. Notable characteristics

Provide your analysis:"""

    analysis = _call_groq_api(prompt, model=model, max_tokens=1024)

    return {
        'member': member,
        'statistics': {
            'message_count': message_count,
            'avg_length': avg_length,
            'avg_sentiment': avg_sentiment
        },
        'insights': analysis
    }

