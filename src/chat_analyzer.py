"""
Main Chat Analyzer Class

Orchestrator class that combines all analysis modules for easy use.
"""

import json
from typing import Dict, Optional, List
import pandas as pd

# Handle both relative and absolute imports
try:
    # Try relative imports first (when used as a package)
    from .data_extraction import extract_chat_data, detect_chat_type
    from .data_cleaning import preprocess_chat_data
    from .data_wrangling import enrich_dataframe
    from .nlp_analysis import add_sentiment_features
    from .visualization import (
        plot_message_frequency,
        plot_temporal_distribution,
        plot_temporal_distributions,
        plot_message_length_distribution,
        plot_activity_heatmap,
        plot_sentiment_distribution,
        plot_sentiment_distributions,
        plot_member_wordclouds,
        plot_engagement_metrics,
        plot_member_sentiment_analysis,
        plot_mood_swings
    )
    from .ai_insights import (
        generate_chat_summary,
        identify_topics,
        generate_insights,
        analyze_conversation_flow,
        get_member_insights
    )
except ImportError:
    # Fall back to absolute imports (when imported directly)
    from data_extraction import extract_chat_data, detect_chat_type
    from data_cleaning import preprocess_chat_data
    from data_wrangling import enrich_dataframe
    from nlp_analysis import add_sentiment_features
    from visualization import (
        plot_message_frequency,
        plot_temporal_distribution,
        plot_temporal_distributions,
        plot_message_length_distribution,
        plot_activity_heatmap,
        plot_sentiment_distribution,
        plot_sentiment_distributions,
        plot_member_wordclouds,
        plot_engagement_metrics,
        plot_member_sentiment_analysis,
        plot_mood_swings
    )
    from ai_insights import (
        generate_chat_summary,
        identify_topics,
        generate_insights,
        analyze_conversation_flow,
        get_member_insights
    )


class WhatsAppChatAnalyzer:
    """
    Main analyzer class for WhatsApp chat analysis.

    Provides a unified interface for loading, processing, analyzing, and visualizing
    WhatsApp chat exports (both group and individual chats).
    """

    def __init__(
        self,
        file_path: str,
        chat_type: str = 'auto',
        name_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the analyzer.

        Args:
            file_path: Path to WhatsApp export file (zip or txt)
            chat_type: 'auto', 'group', or 'individual'
            name_mapping: Optional dictionary for sender name normalization
        """
        self.file_path = file_path
        self.chat_type = chat_type
        self.name_mapping = name_mapping or {}
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.analysis_results: Dict = {}

    def load_and_parse(self) -> pd.DataFrame:
        """
        Load and parse WhatsApp chat file.

        Returns:
            Raw DataFrame with extracted chat data
        """
        self.raw_df = extract_chat_data(self.file_path, chat_type=self.chat_type)

        # Update chat_type if auto-detected
        if self.chat_type == 'auto':
            self.chat_type = self.raw_df.attrs.get('chat_type', 'group')

        return self.raw_df

    def preprocess(self) -> pd.DataFrame:
        """
        Clean and preprocess the chat data.

        Returns:
            Preprocessed DataFrame
        """
        if self.raw_df is None:
            raise ValueError("Must call load_and_parse() first")

        self.processed_df = preprocess_chat_data(
            self.raw_df,
            chat_type=self.chat_type,
            name_mapping=self.name_mapping
        )

        return self.processed_df

    def enrich(self) -> pd.DataFrame:
        """
        Add features and enrich the dataframe.

        Returns:
            Enriched DataFrame with all features
        """
        if self.processed_df is None:
            raise ValueError("Must call preprocess() first")

        self.processed_df = enrich_dataframe(
            self.processed_df,
            chat_type=self.chat_type
        )

        return self.processed_df

    def analyze(self, include_sentiment: bool = True) -> Dict:
        """
        Run full analysis pipeline.

        Args:
            include_sentiment: Whether to include sentiment analysis

        Returns:
            Dictionary with analysis results
        """
        # Ensure data is loaded and processed
        if self.processed_df is None:
            if self.raw_df is None:
                self.load_and_parse()
            self.preprocess()
            self.enrich()

        # Add sentiment analysis if requested
        if include_sentiment and 'sentiment' not in self.processed_df.columns:
            self.processed_df = add_sentiment_features(self.processed_df, method='vader')

        # Calculate statistics
        results = {
            'chat_type': self.chat_type,
            'total_messages': len(self.processed_df),
            'date_range': {},
            'statistics': {}
        }

        # Date range
        if 'date_and_time' in self.processed_df.columns:
            results['date_range'] = {
                'start': str(self.processed_df['date_and_time'].min()),
                'end': str(self.processed_df['date_and_time'].max())
            }

        # Basic statistics
        if 'word_length' in self.processed_df.columns:
            results['statistics']['avg_word_length'] = float(self.processed_df['word_length'].mean())
            results['statistics']['avg_char_length'] = float(self.processed_df['character_length'].mean())

        # Group-specific statistics
        if self.chat_type == 'group' and 'sender' in self.processed_df.columns:
            results['statistics']['unique_senders'] = int(self.processed_df['sender'].nunique())
            results['statistics']['messages_per_sender'] = self.processed_df['sender'].value_counts().to_dict()

        # Sentiment statistics
        if 'sentiment' in self.processed_df.columns:
            results['statistics']['avg_sentiment'] = float(self.processed_df['sentiment'].mean())
            results['statistics']['sentiment_distribution'] = self.processed_df['sentiment'].value_counts().to_dict()

        self.analysis_results = results
        return results

    def visualize(
        self,
        analysis_type: str,
        **kwargs
    ):
        """
        Generate visualizations.

        Args:
            analysis_type: Type of visualization to generate:
                - 'message_frequency': Message count by sender/day/month/hour
                - 'temporal_distribution': Temporal distribution plots
                - 'temporal_distributions': Multiple temporal distributions
                - 'message_length': Message length distribution
                - 'activity_heatmap': Activity heatmap
                - 'sentiment_distribution': Sentiment over time
                - 'sentiment_distributions': Multiple sentiment distributions
                - 'member_wordclouds': Word clouds per member
                - 'engagement': Engagement metrics
                - 'member_sentiment': Sentiment analysis per member
                - 'mood_swings': Mood swing analysis
            **kwargs: Additional arguments for the visualization function

        Returns:
            Plotly or matplotlib figure object
        """
        if self.processed_df is None:
            raise ValueError("Must call analyze() first")

        df = self.processed_df

        visualization_map = {
            'message_frequency': lambda: plot_message_frequency(df, **kwargs),
            'temporal_distribution': lambda: plot_temporal_distribution(df, **kwargs),
            'temporal_distributions': lambda: plot_temporal_distributions(df, **kwargs),
            'message_length': lambda: plot_message_length_distribution(df, **kwargs),
            'activity_heatmap': lambda: plot_activity_heatmap(df, **kwargs),
            'sentiment_distribution': lambda: plot_sentiment_distribution(df, **kwargs),
            'sentiment_distributions': lambda: plot_sentiment_distributions(df, **kwargs),
            'member_wordclouds': lambda: plot_member_wordclouds(df, **kwargs),
            'engagement': lambda: plot_engagement_metrics(df, self.chat_type, **kwargs),
            'member_sentiment': lambda: plot_member_sentiment_analysis(df, **kwargs),
            'mood_swings': lambda: plot_mood_swings(df, **kwargs),
        }

        if analysis_type not in visualization_map:
            raise ValueError(
                f"Unknown analysis_type: {analysis_type}. "
                f"Available types: {list(visualization_map.keys())}"
            )

        return visualization_map[analysis_type]()

    def get_ai_insights(
        self,
        insight_type: str,
        **kwargs
    ) -> Dict:
        """
        Get AI-powered insights.

        Args:
            insight_type: Type of insight to generate:
                - 'summary': Chat summary
                - 'topics': Topic identification
                - 'insights': General insights
                - 'conversation_flow': Conversation flow analysis (individual chats)
                - 'member_insights': Per-member insights (group chats)
            **kwargs: Additional arguments for the insight function

        Returns:
            Dictionary with insights
        """
        if self.processed_df is None:
            raise ValueError("Must call analyze() first")

        df = self.processed_df

        insight_map = {
            'summary': lambda: {'summary': generate_chat_summary(df, **kwargs)},
            'topics': lambda: {'topics': identify_topics(df, **kwargs)},
            'insights': lambda: generate_insights(df, self.chat_type, **kwargs),
            'conversation_flow': lambda: analyze_conversation_flow(df, **kwargs),
            'member_insights': lambda: get_member_insights(df, **kwargs),
        }

        if insight_type not in insight_map:
            raise ValueError(
                f"Unknown insight_type: {insight_type}. "
                f"Available types: {list(insight_map.keys())}"
            )

        return insight_map[insight_type]()

    def export_results(
        self,
        output_path: str,
        format: str = 'json'
    ) -> None:
        """
        Export analysis results to file.

        Args:
            output_path: Path to output file
            format: Export format ('json' or 'csv')
        """
        if self.analysis_results == {}:
            raise ValueError("Must call analyze() first")

        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
        elif format == 'csv':
            if self.processed_df is not None:
                self.processed_df.to_csv(output_path, index=False)
            else:
                raise ValueError("No processed data to export")
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'")

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the processed DataFrame.

        Returns:
            Processed DataFrame
        """
        if self.processed_df is None:
            raise ValueError("Must call analyze() first")
        return self.processed_df.copy()


