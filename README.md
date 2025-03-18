# AI Financial Advisor

A self-learning AI application for real-time stock analysis and financial advice.

## Features

- Real-time stock data collection and analysis
- Machine learning models for price prediction
- Sentiment analysis of financial news
- Technical indicator analysis
- Personalized investment recommendations
- Interactive dashboard for data visualization

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up API keys in `.env` file
4. Run the application:
   ```
   streamlit run app/main.py
   ```

## Project Structure

- `app/api/`: API connections to financial data sources
- `app/models/`: Machine learning models for prediction and analysis
- `app/data/`: Data processing and storage
- `app/utils/`: Utility functions
- `app/components/`: UI components for the dashboard

## Usage

After starting the application, navigate to the provided URL in your browser. The dashboard provides:

- Stock price predictions
- Market sentiment analysis
- Technical indicator signals
- Investment recommendations based on your risk profile
- Real-time market news analysis 