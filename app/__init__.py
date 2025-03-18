# AI Financial Advisor

# Import main modules for easier access
from app.api.stock_data import get_stock_data, get_stock_info, get_related_stocks
from app.api.news_data import get_financial_news
from app.models.price_prediction import PricePredictionModel
from app.models.sentiment_analysis import SentimentAnalyzer
from app.models.recommendation_engine import RecommendationEngine
from app.utils.technical_indicators import calculate_indicators, generate_signals, identify_patterns

__version__ = '0.1.0' 