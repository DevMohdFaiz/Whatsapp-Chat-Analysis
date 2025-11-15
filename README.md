# WhatsApp Chat Analysis

A comprehensive data analysis tool for WhatsApp chat exports featuring statistical analysis, natural language processing, sentiment analysis, and AI-powered insights.

## Overview

This application provides deep insights into WhatsApp conversations through multiple analytical approaches including temporal patterns, sentiment analysis, communication dynamics, and AI-generated insights. The tool supports both individual and group chat analysis with an interactive web interface.

## Features

### Core Analysis
- **Statistical Analysis**: Descriptive statistics, message frequency, engagement metrics
- **Temporal Analysis**: Activity patterns by hour, day, week, and month
- **Sentiment Analysis**: VADER-based sentiment scoring with positive/neutral/negative classification
- **NLP Analysis**: Word frequency, TF-IDF keyword extraction, text pattern recognition
- **Group Dynamics**: Member participation rates, interaction patterns, communication styles

### Visualization
- Interactive charts using Plotly
- Activity heatmaps (day x hour)
- Temporal distribution plots
- Sentiment distribution graphs
- Member engagement metrics
- Message frequency analysis

### AI Integration
- Groq API integration for conversational analysis
- Context-aware chat interface
- Natural language query support
- Automated insight generation

### Web Applications
- **Streamlit App**: Full-featured interactive dashboard with session persistence
- **Flask App**: RESTful API for programmatic access

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DevMohdFaiz/Whatsapp-Chat-Analysis.git
cd Whatsapp-Chat-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp env.example .env
```

Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

Get your API key from: https://console.groq.com/

## Usage

### Streamlit Application

Launch the interactive web interface:
```bash
streamlit run apps/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

**Features:**
- Upload WhatsApp chat exports (.txt or .zip)
- Session persistence across page refreshes
- Multiple analysis categories with one-click execution
- AI-powered chat assistant
- Data export to CSV
- Interactive visualizations

### Flask Application

Start the Flask API server:
```bash
python apps/flask_app.py
```

API will be available at `http://localhost:5000`

### Jupyter Notebook

For custom analysis workflows:
```bash
jupyter notebook whatsapp_chat_analysis.ipynb
```

## Exporting WhatsApp Chats

### Android
1. Open WhatsApp chat
2. Tap the three dots (menu)
3. Select "More" > "Export chat"
4. Choose "Without media"
5. Save the .txt file

### iOS
1. Open WhatsApp chat
2. Tap contact/group name at top
3. Scroll down and tap "Export Chat"
4. Choose "Without Media"
5. Save the .txt file

## Project Structure

```
Whatsapp-Chat-Analysis/
├── apps/
│   ├── streamlit_app.py    # Interactive web dashboard
│   ├── flask_app.py         # RESTful API server
│   └── .sessions/           # Session storage
├── src/
│   ├── data_extraction.py   # Chat file parsing
│   ├── data_cleaning.py     # Data preprocessing
│   ├── data_wrangling.py    # Feature engineering
│   ├── nlp_analysis.py      # NLP and sentiment analysis
│   ├── visualization.py     # Chart generation
│   ├── ai_integration.py    # AI model integration
│   └── ai_insights.py       # Insight generation
├── chats/                   # Sample chat files
├── requirements.txt         # Python dependencies
├── env.example             # Environment template
└── README.md
```

## Analysis Modules

### Data Extraction
Supports multiple chat formats:
- iOS and Android exports
- Group and individual chats
- Compressed (.zip) and text (.txt) files
- Automatic format detection

### Data Cleaning
- Message parsing and validation
- Timestamp normalization
- Special character handling
- Duplicate removal
- Missing data imputation

### Data Wrangling
Feature engineering including:
- Temporal features (hour, day, month, part of day)
- Message length and word count
- URL and mention detection
- Message type classification

### NLP Analysis
- VADER sentiment analysis
- Word frequency analysis
- TF-IDF keyword extraction
- Text pattern recognition
- Emoji analysis

### Visualization
Interactive visualizations using Plotly:
- Time series plots
- Distribution histograms
- Heatmaps
- Bar charts
- Line graphs

## Analysis Types

### Univariate Analysis
- Message characteristics (length, word count)
- Sentiment distribution
- Text features (URLs, questions, mentions)

### Bivariate Analysis
- Temporal patterns vs message count
- Sender relationships and activity
- Sentiment vs sender correlation

### Multivariate Analysis
- Activity heatmaps (day x hour)
- Comprehensive engagement metrics
- Multi-dimensional member statistics

### Temporal Analysis
- Time-based message patterns
- Peak activity periods
- Trend analysis over time

### Group Analysis
- Member participation statistics
- Group dynamics and interaction patterns
- Communication style analysis

### Statistical Summary
- Descriptive statistics
- Distribution analysis
- Correlation metrics

## API Reference (Flask)

### Endpoints

**POST /analyze**
```json
{
  "file_path": "path/to/chat.txt",
  "analysis_types": ["sentiment", "temporal", "nlp"]
}
```

Returns comprehensive analysis results in JSON format.

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Required for AI features
- `GROQ_MODEL`: Optional, defaults to llama-3.1-8b-instant
- `DEBUG_SESSION`: Optional, enables session debugging in Streamlit

### Session Management

The Streamlit app implements persistent sessions:
- Session data stored in `.sessions/` directory
- Automatic session recovery on page refresh
- Session cleanup after 7 days
- Unique session IDs in URL parameters

## Dependencies

Core libraries:
- pandas, numpy: Data manipulation
- plotly, matplotlib, seaborn: Visualization
- nltk, textblob, vaderSentiment: NLP
- scikit-learn: Machine learning utilities
- groq: AI integration
- streamlit, flask: Web frameworks

See `requirements.txt` for complete list.

## Performance Considerations

- Large chat files (>10,000 messages) may take longer to process
- AI features require active internet connection
- Session files are automatically cleaned up after 7 days
- Recommended to process chats in smaller date ranges for better performance

## Limitations

- Maximum file size dependent on system memory
- AI responses limited by API rate limits
- Sentiment analysis optimized for English language
- Requires proper date format in chat exports

## Troubleshooting

### Common Issues

**Session not persisting:**
- Ensure browser allows cookies and local storage
- Check that `.sessions/` directory is writable
- Verify session ID appears in URL parameters

**Import errors:**
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (requires 3.8+)

**API errors:**
- Verify GROQ_API_KEY in `.env` file
- Check internet connection
- Confirm API key is valid at https://console.groq.com/

**Chat parsing errors:**
- Ensure chat export format is correct
- Try re-exporting the chat
- Verify file encoding is UTF-8

## Contributing

Contributions are welcome. Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request with detailed description

## License

This project is provided as-is for educational and analytical purposes.

## Contact

For questions, issues, or feature requests, please open an issue on GitHub.

## Acknowledgments

Built using open-source libraries and tools from the Python data science ecosystem.
