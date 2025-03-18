"""
AI Financial Advisor

A machine learning-powered financial advisor application that analyzes stock data,
provides price predictions, sentiment analysis, and investment recommendations.

This application uses data directly from the web without requiring API keys.
"""

# Import main modules for easier access
from app.api.stock_data import get_stock_data, get_stock_info, get_related_stocks
from app.api.news_data import get_financial_news
from app.models.price_prediction import PricePredictionModel
from app.models.sentiment_analysis import SentimentAnalyzer
from app.models.recommendation_engine import RecommendationEngine
from app.utils.technical_indicators import calculate_indicators, generate_signals, identify_patterns

__version__ = "1.0.0"
__author__ = "AI Financial Advisor Team" 