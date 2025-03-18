import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import custom modules
from api.stock_data import get_stock_data
from api.news_data import get_financial_news
from models.price_prediction import PricePredictionModel
from models.sentiment_analysis import SentimentAnalyzer
from models.recommendation_engine import RecommendationEngine
from utils.technical_indicators import calculate_indicators

# Page configuration
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💹",
    layout="wide"
)

# Sidebar
st.sidebar.title("AI Financial Advisor")
st.sidebar.image("https://img.icons8.com/color/96/000000/financial-growth.png")

# User inputs
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
time_period = st.sidebar.selectbox(
    "Select Time Period",
    ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"]
)
risk_profile = st.sidebar.select_slider(
    "Risk Tolerance",
    options=["Very Low", "Low", "Medium", "High", "Very High"],
    value="Medium"
)

# Convert time period to days
period_map = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "5 Years": 1825
}
days = period_map[time_period]

# Main function
def main():
    # Header
    st.title(f"Financial Analysis for {ticker}")
    
    try:
        # Get stock data
        with st.spinner('Fetching stock data...'):
            stock_data = get_stock_data(ticker, days)
            
        # Display stock data
        display_stock_data(stock_data, ticker)
        
        # Financial news and sentiment
        display_news_sentiment(ticker)
        
        # Technical analysis
        display_technical_analysis(stock_data)
        
        # Price prediction
        display_price_prediction(stock_data, ticker)
        
        # Investment recommendations
        display_recommendations(ticker, risk_profile)
        
    except Exception as e:
        st.error(f"Error processing data: {e}")

def display_stock_data(data, ticker):
    st.subheader("Stock Price History")
    
    # Create price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    col1.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
    col2.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
    col3.metric("52w High", f"${data['High'].max():.2f}")
    col4.metric("52w Low", f"${data['Low'].min():.2f}")

def display_news_sentiment(ticker):
    st.subheader("News Sentiment Analysis")
    
    # Get news data
    with st.spinner('Analyzing financial news...'):
        news_data = get_financial_news(ticker, limit=5)
        
        # Initialize sentiment analyzer
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_scores = []
        
        # Create columns for news display
        cols = st.columns(len(news_data))
        
        for i, news in enumerate(news_data):
            # Analyze sentiment
            sentiment = sentiment_analyzer.analyze(news['title'] + " " + news['description'])
            sentiment_scores.append(sentiment)
            
            # Display news with sentiment color
            with cols[i]:
                if sentiment > 0.2:
                    st.markdown(f"<div style='background-color:#d4f1d4;padding:10px;border-radius:5px'>"
                                f"<h5>{news['title']}</h5>"
                                f"<p>{news['description'][:100]}...</p>"
                                f"<p>Sentiment: Positive ({sentiment:.2f})</p></div>", 
                                unsafe_allow_html=True)
                elif sentiment < -0.2:
                    st.markdown(f"<div style='background-color:#ffcccb;padding:10px;border-radius:5px'>"
                                f"<h5>{news['title']}</h5>"
                                f"<p>{news['description'][:100]}...</p>"
                                f"<p>Sentiment: Negative ({sentiment:.2f})</p></div>", 
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color:#f0f0f0;padding:10px;border-radius:5px'>"
                                f"<h5>{news['title']}</h5>"
                                f"<p>{news['description'][:100]}...</p>"
                                f"<p>Sentiment: Neutral ({sentiment:.2f})</p></div>", 
                                unsafe_allow_html=True)
        
        # Overall sentiment
        avg_sentiment = np.mean(sentiment_scores)
        st.subheader("Overall Market Sentiment")
        sentiment_gauge = {
            "data": [go.Indicator(
                mode="gauge+number",
                value=avg_sentiment,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [-1, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [-1, -0.5], "color": "red"},
                        {"range": [-0.5, -0.2], "color": "orange"},
                        {"range": [-0.2, 0.2], "color": "yellow"},
                        {"range": [0.2, 0.5], "color": "lightgreen"},
                        {"range": [0.5, 1], "color": "green"},
                    ],
                }
            )],
            "layout": {"height": 300, "width": 500}
        }
        st.plotly_chart(sentiment_gauge)

def display_technical_analysis(data):
    st.subheader("Technical Analysis")
    
    # Calculate technical indicators
    indicators = calculate_indicators(data)
    
    col1, col2 = st.columns(2)
    
    # Moving averages
    with col1:
        st.write("Moving Averages")
        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price'))
        ma_fig.add_trace(go.Scatter(x=indicators.index, y=indicators['SMA_20'], name='SMA 20'))
        ma_fig.add_trace(go.Scatter(x=indicators.index, y=indicators['SMA_50'], name='SMA 50'))
        ma_fig.add_trace(go.Scatter(x=indicators.index, y=indicators['EMA_20'], name='EMA 20'))
        
        ma_fig.update_layout(height=400)
        st.plotly_chart(ma_fig, use_container_width=True)
    
    # Oscillators
    with col2:
        st.write("Oscillators")
        osc_fig = go.Figure()
        osc_fig.add_trace(go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI'))
        
        # Add RSI overbought/oversold lines
        osc_fig.add_shape(type="line", x0=indicators.index[0], y0=70, x1=indicators.index[-1], y1=70,
                         line=dict(color="red", width=2, dash="dash"))
        osc_fig.add_shape(type="line", x0=indicators.index[0], y0=30, x1=indicators.index[-1], y1=30,
                         line=dict(color="green", width=2, dash="dash"))
        
        osc_fig.update_layout(height=400)
        st.plotly_chart(osc_fig, use_container_width=True)
    
    # Technical signals
    signals = []
    if indicators['RSI'].iloc[-1] > 70:
        signals.append("RSI indicates overbought conditions (Bearish)")
    elif indicators['RSI'].iloc[-1] < 30:
        signals.append("RSI indicates oversold conditions (Bullish)")
    
    if indicators['Close'].iloc[-1] > indicators['SMA_20'].iloc[-1]:
        signals.append("Price above 20-day SMA (Bullish)")
    else:
        signals.append("Price below 20-day SMA (Bearish)")
    
    if indicators['SMA_20'].iloc[-1] > indicators['SMA_50'].iloc[-1]:
        signals.append("20-day SMA above 50-day SMA - Golden Cross (Bullish)")
    else:
        signals.append("20-day SMA below 50-day SMA - Death Cross (Bearish)")
    
    if indicators['MACD'].iloc[-1] > indicators['MACD_Signal'].iloc[-1]:
        signals.append("MACD above Signal line (Bullish)")
    else:
        signals.append("MACD below Signal line (Bearish)")
    
    # Display signals
    st.subheader("Technical Signals")
    for signal in signals:
        if "Bullish" in signal:
            st.markdown(f"<div style='color:green'>✅ {signal}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:red'>❌ {signal}</div>", unsafe_allow_html=True)

def display_price_prediction(data, ticker):
    st.subheader("Price Prediction")
    
    # Initialize model
    model = PricePredictionModel()
    
    # Train model
    with st.spinner('Training prediction model...'):
        model.train(data)
    
    # Make predictions for next 30 days
    future_days = 30
    predictions = model.predict(data, future_days)
    
    # Create prediction chart
    pred_fig = go.Figure()
    
    # Historical data
    pred_fig.add_trace(go.Scatter(
        x=data.index[-60:],
        y=data['Close'][-60:],
        name='Historical Price',
        line=dict(color='blue')
    ))
    
    # Prediction data
    pred_dates = pd.date_range(start=data.index[-1], periods=future_days+1)[1:]
    pred_fig.add_trace(go.Scatter(
        x=pred_dates,
        y=predictions,
        name='Price Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    # Prediction confidence interval
    upper_bound = predictions * 1.05
    lower_bound = predictions * 0.95
    
    pred_fig.add_trace(go.Scatter(
        x=pred_dates, 
        y=upper_bound,
        fill=None,
        mode='lines',
        line_color='rgba(255,0,0,0.1)',
        name='Upper Bound'
    ))
    
    pred_fig.add_trace(go.Scatter(
        x=pred_dates, 
        y=lower_bound,
        fill='tonexty',
        mode='lines',
        line_color='rgba(255,0,0,0.1)',
        name='Lower Bound'
    ))
    
    pred_fig.update_layout(
        title=f"30-Day Price Prediction for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500
    )
    
    st.plotly_chart(pred_fig, use_container_width=True)
    
    # Prediction metrics
    last_price = data['Close'].iloc[-1]
    pred_last_day = predictions[-1]
    pred_change = ((pred_last_day - last_price) / last_price) * 100
    
    st.metric(
        "Predicted Price (30 days)", 
        f"${pred_last_day:.2f}", 
        f"{pred_change:.2f}%"
    )
    
    # Model accuracy
    st.write(f"Model Accuracy: {model.accuracy:.2f}%")
    st.caption("Note: Predictions are estimates based on historical data and market patterns. Actual results may vary.")

def display_recommendations(ticker, risk_profile):
    st.subheader("Investment Recommendations")
    
    # Initialize recommendation engine
    engine = RecommendationEngine()
    
    # Get recommendations
    recommendations = engine.get_recommendations(ticker, risk_profile)
    
    # Display recommendations
    st.write(f"Based on your {risk_profile} risk profile and our analysis of {ticker}, we recommend:")
    
    for rec in recommendations:
        st.markdown(f"**{rec['title']}**")
        st.markdown(f"_{rec['description']}_")
        
        # Display recommendation confidence
        conf = rec['confidence']
        if conf >= 0.7:
            conf_color = "green"
        elif conf >= 0.4:
            conf_color = "orange"
        else:
            conf_color = "red"
            
        st.markdown(f"<div style='color:{conf_color}'>Confidence: {conf*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("---")

if __name__ == "__main__":
    main() 