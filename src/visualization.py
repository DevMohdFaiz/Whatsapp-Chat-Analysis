"""
Visualization Module

All plotting and visualization functions for WhatsApp chat analysis.
Returns plotly/matplotlib figure objects for integration into notebooks or apps.
"""

import math
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

from .nlp_analysis import prepare_text_for_wordcloud


# Time ordering dictionaries for consistent plotting
TIME_ORDER_DICT = {
    'day_order': ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
    'month_order': ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                    'August', 'September', 'October', 'November', 'December'],
    'hour_order': [f"{i:02d}:00" for i in range(24)],
    'part_of_day_order': ['Midnight', 'Morning', 'Afternoon', 'Evening', 'Night']
}


def plot_message_frequency(
    df: pd.DataFrame,
    by: str = 'sender',
    orientation: str = 'v',
    title: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Plot message frequency by sender, day, month, or hour.

    Args:
        df: DataFrame with analysis data
        by: Column to group by ('sender', 'day', 'month', 'hour')
        orientation: 'v' for vertical, 'h' for horizontal
        title: Plot title (auto-generated if None)
        **kwargs: Additional arguments for plotly

    Returns:
        Plotly figure object
    """
    if df.empty or by not in df.columns:
        raise ValueError(f"DataFrame must contain '{by}' column")

    counts = df[by].value_counts().sort_values(ascending=False)

    # Reorder if time-based column
    if by in ['day', 'month', 'hour', 'part_of_day']:
        order_key = f"{by}_order"
        if order_key in TIME_ORDER_DICT:
            available_values = [v for v in TIME_ORDER_DICT[order_key] if v in counts.index]
            counts = counts.reindex(available_values)

    if title is None:
        title = f'Message frequency by {by}'

    fig = px.bar(
        x=counts.index if orientation == 'v' else counts.values,
        y=counts.values if orientation == 'v' else counts.index,
        orientation=orientation,
        title=title,
        **kwargs
    )

    x_label = by.capitalize() if orientation == 'v' else 'No. of messages'
    y_label = 'No. of messages' if orientation == 'v' else by.capitalize()

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False
    )

    return fig


def plot_temporal_distribution(
    df: pd.DataFrame,
    frequency: str = 'day',
    title: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Plot temporal distribution of messages (day, month, hour).

    Args:
        df: DataFrame with temporal columns
        frequency: 'day', 'month', or 'hour'
        title: Plot title
        **kwargs: Additional arguments for plotly

    Returns:
        Plotly figure object
    """
    if df.empty or frequency not in df.columns:
        raise ValueError(f"DataFrame must contain '{frequency}' column")

    order_key = f"{frequency}_order"
    if order_key not in TIME_ORDER_DICT:
        raise ValueError(f"Unknown frequency: {frequency}")

    df_plot = df.groupby(frequency).size().reset_index(name='count')
    df_plot = df_plot.reindex(
        [i for i in TIME_ORDER_DICT[order_key] if i in df_plot[frequency].values]
    )

    if title is None:
        title = f"{frequency.capitalize()}ly Message Frequency"

    fig = px.line(
        df_plot,
        x=frequency,
        y='count',
        title=title,
        markers=True,
        **kwargs
    )

    fig.update_layout(
        xaxis_title=frequency.capitalize(),
        yaxis_title='No. of messages'
    )

    return fig


def plot_temporal_distributions(
    df: pd.DataFrame,
    frequencies: List[str] = ['month', 'day', 'hour'],
    title: Optional[str] = None,
    height: int = 1000,
    **kwargs
) -> go.Figure:
    """
    Plot multiple temporal distributions in subplots.

    Args:
        df: DataFrame with temporal columns
        frequencies: List of frequency types to plot
        title: Overall plot title
        height: Total height of figure
        **kwargs: Additional arguments for plotly

    Returns:
        Plotly figure with subplots
    """
    n_plots = len(frequencies)
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        subplot_titles=[f"{freq.capitalize()}ly Message Frequency" for freq in frequencies],
        vertical_spacing=0.1
    )

    for idx, freq in enumerate(frequencies):
        if freq not in df.columns:
            continue

        order_key = f"{freq}_order"
        if order_key not in TIME_ORDER_DICT:
            continue

        df_plot = df.groupby(freq).size().reset_index(name='count')
        available_values = [v for v in TIME_ORDER_DICT[order_key] if v in df_plot[freq].values]
        df_plot = df_plot.set_index(freq).reindex(available_values).reset_index()

        trace = px.line(df_plot, x=freq, y='count', markers=True).data[0]
        fig.add_trace(trace, row=idx + 1, col=1)

    if title is None:
        title = "Message Frequency Distributions"

    fig.update_layout(height=height, title_text=title)
    fig.update_xaxes(title_text="", row=n_plots, col=1)
    fig.update_yaxes(title_text="No. of messages", row=n_plots, col=1)

    return fig


def plot_message_length_distribution(
    df: pd.DataFrame,
    length_type: str = 'word_length',
    title: Optional[str] = None,
    nbins: int = 30,
    **kwargs
) -> go.Figure:
    """
    Plot distribution of message lengths.

    Args:
        df: DataFrame with length columns
        length_type: 'word_length' or 'character_length'
        title: Plot title
        nbins: Number of bins for histogram
        **kwargs: Additional arguments for plotly

    Returns:
        Plotly figure object
    """
    if df.empty or length_type not in df.columns:
        raise ValueError(f"DataFrame must contain '{length_type}' column")

    if title is None:
        title = 'Message Length Distribution'

    fig = px.histogram(
        df,
        x=length_type,
        title=title,
        nbins=nbins,
        **kwargs
    )

    fig.update_layout(
        xaxis_title='Message Length',
        yaxis_title='Frequency',
        showlegend=False
    )

    return fig


def plot_activity_heatmap(
    df: pd.DataFrame,
    x: str = 'day',
    y: str = 'hour',
    title: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Plot activity heatmap showing message frequency across time dimensions.

    Args:
        df: DataFrame with temporal columns
        x: X-axis dimension (e.g., 'day')
        y: Y-axis dimension (e.g., 'hour')
        title: Plot title
        **kwargs: Additional arguments for plotly

    Returns:
        Plotly figure object
    """
    if df.empty or x not in df.columns or y not in df.columns:
        raise ValueError(f"DataFrame must contain '{x}' and '{y}' columns")

    heatmap_df = df.groupby([x, y]).size().unstack(fill_value=0)

    # Reorder rows and columns if time-based
    if x in TIME_ORDER_DICT:
        order_key = f"{x}_order"
        available_x = [v for v in TIME_ORDER_DICT[order_key] if v in heatmap_df.index]
        heatmap_df = heatmap_df.reindex(available_x)

    if y in TIME_ORDER_DICT:
        order_key = f"{y}_order"
        available_y = [v for v in TIME_ORDER_DICT[order_key] if v in heatmap_df.columns]
        heatmap_df = heatmap_df[available_y]

    if title is None:
        title = f"Activity Heatmap: {x.capitalize()} vs {y.capitalize()}"

    fig = px.imshow(
        heatmap_df,
        text_auto=True,
        aspect='auto',
        title=title,
        **kwargs
    )

    fig.update_layout(
        xaxis_title=y.capitalize(),
        yaxis_title=x.capitalize()
    )

    return fig


def plot_sentiment_distribution(
    df: pd.DataFrame,
    by: str = 'day',
    title: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Plot sentiment distribution over time or by sender.

    Args:
        df: DataFrame with sentiment column
        by: Column to group by ('day', 'month', 'hour', 'sender')
        title: Plot title
        **kwargs: Additional arguments for plotly

    Returns:
        Plotly figure object
    """
    if df.empty or 'sentiment' not in df.columns or by not in df.columns:
        raise ValueError(f"DataFrame must contain 'sentiment' and '{by}' columns")

    sentiment_df = df.groupby(by)['sentiment'].mean().reset_index(name='avg_sentiment')

    # Reorder if time-based
    if by in TIME_ORDER_DICT:
        order_key = f"{by}_order"
        available_values = [v for v in TIME_ORDER_DICT[order_key] if v in sentiment_df[by].values]
        sentiment_df = sentiment_df.set_index(by).reindex(available_values).reset_index()

    if title is None:
        title = f"Sentiment Distribution by {by.capitalize()}"

    fig = px.line(
        sentiment_df,
        x=by,
        y='avg_sentiment',
        title=title,
        markers=True,
        **kwargs
    )

    fig.update_layout(
        xaxis_title=by.capitalize(),
        yaxis_title='Average Sentiment Score'
    )

    return fig


def plot_sentiment_distributions(
    df: pd.DataFrame,
    frequencies: List[str] = ['month', 'day', 'hour'],
    title: Optional[str] = None,
    height: int = 1000,
    **kwargs
) -> go.Figure:
    """
    Plot multiple sentiment distributions in subplots.

    Args:
        df: DataFrame with sentiment and temporal columns
        frequencies: List of frequency types to plot
        title: Overall plot title
        height: Total height of figure
        **kwargs: Additional arguments for plotly

    Returns:
        Plotly figure with subplots
    """
    if df.empty or 'sentiment' not in df.columns:
        raise ValueError("DataFrame must contain 'sentiment' column")

    n_plots = len(frequencies)
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        subplot_titles=[f"{freq.capitalize()}ly Sentiment Distribution" for freq in frequencies],
        vertical_spacing=0.1
    )

    for idx, freq in enumerate(frequencies):
        if freq not in df.columns:
            continue

        order_key = f"{freq}_order"
        if order_key not in TIME_ORDER_DICT:
            continue

        sentiment_df = df.groupby(freq)['sentiment'].mean().reset_index(name='avg_sentiment')
        available_values = [v for v in TIME_ORDER_DICT[order_key] if v in sentiment_df[freq].values]
        sentiment_df = sentiment_df.set_index(freq).reindex(available_values).reset_index()

        trace = px.line(sentiment_df, x=freq, y='avg_sentiment', markers=True).data[0]
        fig.add_trace(trace, row=idx + 1, col=1)

    if title is None:
        title = "Sentiment Score Distributions"

    fig.update_layout(height=height, title_text=title)
    fig.update_xaxes(title_text="", row=n_plots, col=1)
    fig.update_yaxes(title_text="Average Sentiment Score", row=n_plots, col=1)

    return fig


def plot_member_wordclouds(
    df: pd.DataFrame,
    custom_stopwords: Optional[List[str]] = None,
    figsize: tuple = (15, 5),
    **kwargs
) -> plt.Figure:
    """
    Plot word clouds for each member in group chat.

    Args:
        df: DataFrame with 'sender' and 'message' columns
        custom_stopwords: Additional stopwords to remove
        figsize: Figure size (width, height per subplot)
        **kwargs: Additional arguments for WordCloud

    Returns:
        Matplotlib figure object
    """
    if df.empty or 'sender' not in df.columns or 'message' not in df.columns:
        raise ValueError("DataFrame must contain 'sender' and 'message' columns")

    unique_senders = df['sender'].unique()
    n_members = len(unique_senders)
    n_cols = 2
    n_rows = math.ceil(n_members / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, member in enumerate(unique_senders):
        member_text = prepare_text_for_wordcloud(df, sender=member, custom_stopwords=custom_stopwords)

        if len(member_text) > 0:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                **kwargs
            ).generate(member_text)

            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].set_title(f"{member}'s Frequent Words", fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, 'No messages', ha='center', va='center',
                          transform=axes[idx].transAxes)
            axes[idx].set_title(f"{member}", fontsize=12)
            axes[idx].axis('off')

    # Remove unused subplots
    for j in range(len(unique_senders), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Group Members Word Clouds', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_engagement_metrics(
    df: pd.DataFrame,
    chat_type: str = 'group',
    title: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Plot engagement-specific metrics.

    Args:
        df: DataFrame with engagement features
        chat_type: 'group' or 'individual'
        title: Plot title
        **kwargs: Additional arguments for plotly

    Returns:
        Plotly figure object
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if chat_type == 'group' and 'sender' in df.columns:
        # Plot average message length per sender
        message_length_df = df.groupby('sender')[['word_length', 'character_length']].mean()
        message_length_df = message_length_df.sort_values('word_length', ascending=False)

        if title is None:
            title = 'Average Message Length per Group Member'

        fig = px.bar(
            message_length_df,
            orientation='v',
            title=title,
            **kwargs
        )

        fig.update_layout(
            xaxis_title="Group Member",
            yaxis_title="Average Message Length"
        )

    else:
        # Individual chat engagement metrics
        if 'conversation_gap' in df.columns:
            # Plot conversation gaps over time
            df['gap_hours'] = df['conversation_gap'].dt.total_seconds() / 3600
            fig = px.line(
                df,
                x='date_and_time',
                y='gap_hours',
                title=title or 'Conversation Gaps Over Time',
                **kwargs
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Gap (hours)"
            )
        else:
            raise ValueError("No engagement metrics available for individual chat")

    return fig


def plot_member_sentiment_analysis(
    df: pd.DataFrame,
    title: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Plot sentiment analysis per member (group chats only).

    Args:
        df: DataFrame with 'sender' and 'sentiment' columns
        title: Plot title
        **kwargs: Additional arguments for plotly

    Returns:
        Plotly figure object
    """
    if df.empty or 'sender' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("DataFrame must contain 'sender' and 'sentiment' columns")

    member_sentiment_df = df.groupby('sender')['sentiment'].agg(['mean', 'std', 'median'])
    member_sentiment_df = member_sentiment_df.sort_values('mean', ascending=False)

    if title is None:
        title = 'Sentiment Analysis by Group Member'

    fig = px.bar(
        member_sentiment_df[['mean', 'median']],
        orientation='v',
        title=title,
        barmode='group',
        **kwargs
    )

    fig.update_layout(
        xaxis_title="Group Member",
        yaxis_title="Sentiment Score"
    )

    return fig


def plot_mood_swings(
    df: pd.DataFrame,
    title: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Plot mood swing analysis (sentiment standard deviation) per member.

    Args:
        df: DataFrame with 'sender' and 'sentiment' columns
        title: Plot title
        **kwargs: Additional arguments for plotly

    Returns:
        Plotly figure object
    """
    if df.empty or 'sender' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("DataFrame must contain 'sender' and 'sentiment' columns")

    member_sentiment_df = df.groupby('sender')['sentiment'].agg(['mean', 'std', 'median'])
    mood_swings = member_sentiment_df['std'].sort_values(ascending=False)

    if title is None:
        title = 'Group Members with Most Mood Swings'

    fig = px.bar(
        x=mood_swings.index,
        y=mood_swings.values,
        orientation='v',
        title=title,
        **kwargs
    )

    fig.update_layout(
        xaxis_title='Group Member',
        yaxis_title='Mood Swing (Std Dev)',
        showlegend=False
    )

    return fig

