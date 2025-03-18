# AI Financial Advisor

A machine learning-powered financial advisor application that analyzes stock data, performs sentiment analysis on financial news, and provides investment recommendations.

## Features

- **Stock Data Analysis**: View historical stock data and real-time technical indicators
- **Sentiment Analysis**: Analyze financial news sentiment to gauge market perception
- **Price Prediction**: ML-based price predictions for future stock performance
- **Technical Analysis**: Key technical indicators including RSI, MACD, Bollinger Bands, etc.
- **Investment Recommendations**: Personalized investment advice based on risk profile

## No API Keys Required!

This application retrieves all data directly from the web without requiring any API keys:

- Stock data is fetched using yfinance with built-in web scraping fallback
- Financial news is scraped directly from Yahoo Finance
- All processing is done locally using lightweight ML models

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/ai-financial-advisor.git
cd ai-financial-advisor
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python run.py
```

Or directly with Streamlit:
```
streamlit run app/main.py
```

## How It Works

1. **Data Collection**: The app fetches stock data and news using web scraping techniques
2. **Technical Analysis**: Calculates various technical indicators to identify patterns and signals
3. **Sentiment Analysis**: Analyzes financial news sentiment using NLTK VADER with domain-specific enhancements
4. **Price Prediction**: Uses machine learning models (Random Forest or Linear Regression) to predict future prices
5. **Recommendation Engine**: Combines technical indicators, sentiment, and predictions to generate investment advice

## Project Structure

```
ai-financial-advisor/
├── app/
│   ├── api/              # Data retrieval modules
│   ├── components/       # UI components
│   ├── data/             # Data processing
│   ├── models/           # ML models
│   ├── utils/            # Utility functions
│   └── main.py           # Main Streamlit app
├── requirements.txt      # Project dependencies
├── run.py                # Application runner
└── README.md             # This file
```

## Technologies Used

- **Streamlit**: For the interactive web interface
- **Pandas & NumPy**: For data manipulation and analysis
- **Scikit-learn**: For machine learning models
- **NLTK**: For natural language processing and sentiment analysis
- **Plotly**: For interactive charts and visualizations
- **BeautifulSoup**: For web scraping data

## Disclaimer

This application is for educational and informational purposes only. The investment recommendations provided should not be considered financial advice. Always consult with a professional financial advisor before making investment decisions. 