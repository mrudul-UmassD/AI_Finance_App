import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# Import our utils first to handle numpy compatibility
try:
    from app.utils.numpy_compat import get_nan, is_nan, safe_divide
except ImportError:
    try:
        from utils.numpy_compat import get_nan, is_nan, safe_divide
    except ImportError:
        # Fallback implementations
        def get_nan(): return float('nan')
        def is_nan(x): return x != x
        def safe_divide(a, b): return a / b if b != 0 else get_nan()

# Now import the rest of our modules
try:
    from app.api.stock_data import get_stock_data, get_stock_info
    from app.api.news_data import get_financial_news
    from app.utils.technical_indicators import calculate_indicators, generate_signals
    from app.models.sentiment_analysis import SentimentAnalyzer
    from app.models.price_prediction import PricePredictionModel
    from app.models.recommendation_engine import RecommendationEngine
except ImportError as e:
    st.error(f"Error importing modules from app package: {str(e)}")
    # Try alternative import paths
    try:
        from api.stock_data import get_stock_data, get_stock_info
        from api.news_data import get_financial_news
        from utils.technical_indicators import calculate_indicators, generate_signals
        from models.sentiment_analysis import SentimentAnalyzer
        from models.price_prediction import PricePredictionModel
        from models.recommendation_engine import RecommendationEngine
    except ImportError as e2:
        st.error(f"Alternative import also failed: {str(e2)}")
        st.stop()

# Set page config
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define sidebar
st.sidebar.title("AI Financial Advisor")
st.sidebar.markdown("---")

# Stock selection
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", "AAPL")
days = st.sidebar.slider("Historical Data (Days)", 30, 365, 180)

# User risk profile
st.sidebar.markdown("---")
st.sidebar.subheader("Risk Profile")
risk_tolerance = st.sidebar.select_slider(
    "Risk Tolerance",
    options=["Very Low", "Low", "Medium", "High", "Very High"],
    value="Medium"
)
investment_horizon = st.sidebar.select_slider(
    "Investment Horizon",
    options=["Short-term", "Medium-term", "Long-term"],
    value="Medium-term"
)

# Main app
st.title("AI Financial Advisor 📈")
st.markdown("*AI-powered stock analysis and investment recommendations*")

# Function to load data with caching
@st.cache_data(ttl=3600)
def load_data(ticker, days):
    try:
        data = get_stock_data(ticker, days)
        info = get_stock_info(ticker)
        news = get_financial_news(ticker)
        return data, info, news
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Show loading message while data is being fetched
with st.spinner("Loading data... (This might take a moment)"):
    data, info, news = load_data(ticker, days)

# Check if data was successfully loaded
if data is None or info is None:
    st.error(f"Failed to load data for {ticker}. Please try another ticker.")
    st.stop()

# Display basic stock info
st.subheader(f"{info['longName']} ({ticker})")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"${data['Close'].iloc[0]:.2f}")
with col2:
    price_change = data['Close'].iloc[0] - data['Close'].iloc[1]
    percent_change = (price_change / data['Close'].iloc[1]) * 100
    st.metric("Daily Change", f"${price_change:.2f}", f"{percent_change:.2f}%")
with col3:
    st.metric("Sector", info['sector'])
with col4:
    st.metric("Industry", info['industry'])

# Display stock chart
st.subheader("Stock Price History")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Candlestick"
))
fig.update_layout(
    title=f"{ticker} Stock Price",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# Simple Technical Analysis
st.subheader("Technical Analysis")

# Calculate technical indicators
with st.spinner("Calculating technical indicators..."):
    try:
        indicators = calculate_indicators(data)
        signals = generate_signals(indicators)
        
        # Display buy/sell signals
        signal_df = pd.DataFrame({
            'Signal': signals,
            'Close': data['Close']
        })
        
        buy_signals = signal_df[signal_df['Signal'] == 'buy']
        sell_signals = signal_df[signal_df['Signal'] == 'sell']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=signal_df.index,
            y=signal_df['Close'],
            mode='lines',
            name='Close Price'
        ))
        
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            marker=dict(
                color='green',
                size=10,
                symbol='triangle-up',
            ),
            name='Buy Signal'
        ))
        
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down',
            ),
            name='Sell Signal'
        ))
        
        fig.update_layout(
            title="Buy/Sell Signals based on Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display indicators
        st.subheader("Key Technical Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rsi_value = indicators['RSI_14'].iloc[0] if 'RSI_14' in indicators else 0
            rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
            st.metric("RSI (14)", f"{rsi_value:.2f}", rsi_status)
            
        with col2:
            macd_value = indicators['MACD'].iloc[0] if 'MACD' in indicators else 0
            macd_signal = indicators['MACD_Signal'].iloc[0] if 'MACD_Signal' in indicators else 0
            macd_diff = macd_value - macd_signal
            st.metric("MACD", f"{macd_value:.2f}", f"{macd_diff:.2f}")
            
        with col3:
            bb_upper = indicators['BB_Upper'].iloc[0] if 'BB_Upper' in indicators else 0
            bb_lower = indicators['BB_Lower'].iloc[0] if 'BB_Lower' in indicators else 0
            bb_middle = indicators['BB_Middle'].iloc[0] if 'BB_Middle' in indicators else 1
            
            # Use safe division to avoid divide by zero
            bb_width = ((bb_upper - bb_lower) / bb_middle * 100) if bb_middle != 0 else 0
            st.metric("Bollinger Band Width", f"{bb_width:.2f}%")
            
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")

# News Analysis
st.subheader("Latest News & Sentiment Analysis")

if news:
    with st.spinner("Analyzing sentiment..."):
        try:
            sentiment_analyzer = SentimentAnalyzer()
            
            # Create columns for news display
            for i, article in enumerate(news[:5]):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**{article['title']}**")
                    st.caption(f"Source: {article['source']} | {article['publishedAt'][:10]}")
                    st.markdown(f"{article['description']}")
                    st.markdown(f"[Read more]({article['url']})")
                
                with col2:
                    # Calculate sentiment for this article
                    sentiment = sentiment_analyzer.analyze(article['title'] + " " + article['description'])
                    sentiment_label = sentiment_analyzer.get_sentiment_label(sentiment)
                    
                    # Display sentiment score with appropriate color
                    sentiment_color = "green" if sentiment > 0.05 else "red" if sentiment < -0.05 else "gray"
                    st.markdown(f"<h3 style='text-align: center; color: {sentiment_color};'>{sentiment_label}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='text-align: center; color: {sentiment_color};'>{sentiment:.2f}</h4>", unsafe_allow_html=True)
                
                st.markdown("---")
            
            # Calculate overall sentiment
            all_text = " ".join([a['title'] + " " + a['description'] for a in news])
            overall_sentiment = sentiment_analyzer.analyze(all_text)
            overall_label = sentiment_analyzer.get_sentiment_label(overall_sentiment)
            
            st.subheader("Overall News Sentiment")
            sentiment_color = "green" if overall_sentiment > 0.05 else "red" if overall_sentiment < -0.05 else "gray"
            st.markdown(f"<h2 style='text-align: center; color: {sentiment_color};'>{overall_label}: {overall_sentiment:.2f}</h2>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
else:
    st.warning(f"No news articles found for {ticker}")

# Price Prediction
st.subheader("Price Prediction (30 Days)")

with st.spinner("Training prediction model..."):
    try:
        model = PricePredictionModel()
        model.train(data)
        
        # Make predictions
        predictions = model.predict(data, future_days=30)
        
        # Create dates for prediction
        last_date = data.index[0]
        future_dates = [last_date + timedelta(days=i+1) for i in range(30)]
        
        # Display prediction chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Historical Data'
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            line=dict(dash='dash'),
            name='Predictions'
        ))
        
        fig.update_layout(
            title=f"{ticker} 30-Day Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction metrics
        st.metric("Prediction Accuracy", f"{model.accuracy:.2f}%")
        st.caption("Note: Predictions are based on historical patterns and technical indicators. Always conduct your own research before making investment decisions.")
        
    except Exception as e:
        st.error(f"Error in price prediction: {str(e)}")

# Investment Recommendation
st.subheader("Investment Recommendation")

try:
    # Generate a simple recommendation based on technical signals, sentiment and risk profile
    recent_signals = signals[:7]  # Last week of signals
    buy_count = sum(1 for s in recent_signals if s == 'buy')
    sell_count = sum(1 for s in recent_signals if s == 'sell')
    hold_count = sum(1 for s in recent_signals if s == 'hold')
    
    # Convert risk tolerance to numerical value
    risk_scores = {
        "Very Low": 1,
        "Low": 2,
        "Medium": 3,
        "High": 4,
        "Very High": 5
    }
    
    horizon_scores = {
        "Short-term": 1,
        "Medium-term": 2,
        "Long-term": 3
    }
    
    risk_score = risk_scores[risk_tolerance]
    horizon_score = horizon_scores[investment_horizon]
    
    # Calculate technical score (-1 to 1)
    technical_score = (buy_count - sell_count) / max(1, buy_count + sell_count)
    
    # Calculate overall score weighted by risk profile
    # Higher risk tolerance gives more weight to sentiment
    # Longer horizon gives more weight to prediction trend
    
    # Get prediction trend (positive or negative)
    pred_start = predictions[0]
    pred_end = predictions[-1]
    prediction_trend = (pred_end - pred_start) / pred_start if pred_start != 0 else 0
    
    # Weight components based on risk profile
    if risk_score <= 2:  # Conservative
        technical_weight = 0.6
        sentiment_weight = 0.1
        prediction_weight = 0.3
    elif risk_score >= 4:  # Aggressive
        technical_weight = 0.3
        sentiment_weight = 0.3
        prediction_weight = 0.4
    else:  # Moderate
        technical_weight = 0.4
        sentiment_weight = 0.2
        prediction_weight = 0.4
    
    # Adjust for investment horizon
    if horizon_score == 1:  # Short-term
        technical_weight += 0.1
        sentiment_weight += 0.1
        prediction_weight -= 0.2
    elif horizon_score == 3:  # Long-term
        technical_weight -= 0.1
        sentiment_weight -= 0.1
        prediction_weight += 0.2
    
    # Calculate final score (-1 to 1)
    final_score = (
        technical_weight * technical_score + 
        sentiment_weight * overall_sentiment + 
        prediction_weight * prediction_trend
    )
    
    # Determine recommendation
    if final_score > 0.3:
        recommendation = "Strong Buy"
        explanation = "Technical signals, sentiment analysis, and price predictions all indicate positive momentum."
        color = "green"
    elif final_score > 0.1:
        recommendation = "Buy"
        explanation = "Most indicators suggest this stock is likely to perform well in the near future."
        color = "lightgreen"
    elif final_score > -0.1:
        recommendation = "Hold"
        explanation = "Mixed signals suggest maintaining current positions without significant changes."
        color = "gray"
    elif final_score > -0.3:
        recommendation = "Sell"
        explanation = "Several indicators suggest this stock may underperform in the near future."
        color = "lightcoral"
    else:
        recommendation = "Strong Sell"
        explanation = "Technical signals, sentiment analysis, and price predictions all indicate negative momentum."
        color = "red"
    
    # Display recommendation
    st.markdown(f"<h1 style='text-align: center; color: {color};'>{recommendation}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{explanation}</p>", unsafe_allow_html=True)
    
    # Display the factors considered
    st.subheader("Factors Considered")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Technical Analysis", f"{technical_score:.2f}", 
                 "Bullish" if technical_score > 0 else "Bearish")
    
    with col2:
        st.metric("News Sentiment", f"{overall_sentiment:.2f}", 
                 "Positive" if overall_sentiment > 0 else "Negative")
    
    with col3:
        st.metric("Price Trend (30d)", f"{prediction_trend*100:.2f}%", 
                 "Upward" if prediction_trend > 0 else "Downward")
    
    # Get more detailed recommendations from the engine
    try:
        engine = RecommendationEngine()
        recommendations = engine.get_recommendations(ticker, risk_tolerance)
        
        # Display detailed recommendations
        st.subheader("Detailed Recommendations")
        for rec in recommendations[:3]:  # Show top 3 recommendations
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: rgba(0,0,0,0.05);">
                <h4>{rec['title']}</h4>
                <p>{rec['description']}</p>
                <p><small>Confidence: {rec['confidence']*100:.0f}%</small></p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load detailed recommendations: {str(e)}")
    
    # Add disclaimer
    st.caption("**Disclaimer:** This recommendation is generated by an AI model based on historical data analysis and should be used for informational purposes only. Always consult with a professional financial advisor before making investment decisions.")
    
except Exception as e:
    st.error(f"Error generating recommendation: {str(e)}")

# Footer
st.markdown("---")
st.markdown("AI Financial Advisor | v1.0.0")
st.markdown("Powered by machine learning and web scraping")
st.caption("Data obtained from Yahoo Finance and other public sources") 