"""
AI Financial Advisor - Main runner script

This script launches the AI Financial Advisor application.
"""

import os
import sys
import subprocess

def main():
    """Main function to run the application"""
    try:
        # Check if required packages are installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        
        # Check if technical_indicators.py exists, create it if not
        utils_dir = os.path.join("app", "utils")
        tech_indicators_file = os.path.join(utils_dir, "technical_indicators.py")
        
        if not os.path.exists(tech_indicators_file):
            print("⚠️ Creating missing technical_indicators.py file...")
            with open(tech_indicators_file, "w") as f:
                f.write("""import pandas as pd
import numpy as np

def calculate_indicators(data):
    \"\"\"
    Calculate technical indicators for stock data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock price data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with calculated indicators
    \"\"\"
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * 2)
    
    # Percentage Price Oscillator
    df['PPO'] = ((df['EMA_12'] - df['EMA_26']) / df['EMA_26']) * 100
    
    # Average Directional Index
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df['ADX'] = dx.rolling(window=14).mean()
    
    # Return indicators DataFrame with most recent values first
    return df.iloc[::-1]

def generate_signals(indicators):
    \"\"\"
    Generate buy/sell signals based on technical indicators
    
    Parameters:
    -----------
    indicators : pandas.DataFrame
        DataFrame with calculated technical indicators
        
    Returns:
    --------
    list
        List of 'buy', 'sell', or 'hold' signals for each row
    \"\"\"
    signals = []
    
    for i in range(len(indicators)):
        # Extract indicator values
        macd = indicators['MACD'].iloc[i]
        macd_signal = indicators['MACD_Signal'].iloc[i]
        rsi = indicators['RSI_14'].iloc[i]
        
        # Previous values (if available)
        prev_macd = indicators['MACD'].iloc[i+1] if i+1 < len(indicators) else macd
        prev_macd_signal = indicators['MACD_Signal'].iloc[i+1] if i+1 < len(indicators) else macd_signal
        
        # Check for MACD crossover (positive)
        macd_bullish = macd > macd_signal and prev_macd <= prev_macd_signal
        
        # Check for MACD crossover (negative)
        macd_bearish = macd < macd_signal and prev_macd >= prev_macd_signal
        
        # Check if RSI is overbought or oversold
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        
        # Moving averages
        price = indicators['Close'].iloc[i]
        sma_short = indicators['SMA_20'].iloc[i]
        sma_long = indicators['SMA_50'].iloc[i]
        
        # Check for moving average crossover
        ma_bullish = sma_short > sma_long and price > sma_short
        ma_bearish = sma_short < sma_long and price < sma_short
        
        # Generate signal based on combined indicators
        if (macd_bullish or rsi_oversold) and ma_bullish:
            signals.append('buy')
        elif (macd_bearish or rsi_overbought) and ma_bearish:
            signals.append('sell')
        else:
            signals.append('hold')
    
    return signals

def identify_patterns(data):
    \"\"\"
    Identify chart patterns in stock price data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock price data
        
    Returns:
    --------
    dict
        Dictionary of identified patterns
    \"\"\"
    patterns = {}
    
    # Very basic pattern identification
    # In a real application, this would be much more sophisticated
    
    # Check for double bottom
    min_idx = data['Low'].rolling(window=10).apply(lambda x: x.argmin(), raw=True)
    if len(min_idx.unique()) > 1 and min_idx.nunique() < len(min_idx) / 2:
        patterns['double_bottom'] = True
    
    # Check for double top
    max_idx = data['High'].rolling(window=10).apply(lambda x: x.argmax(), raw=True)
    if len(max_idx.unique()) > 1 and max_idx.nunique() < len(max_idx) / 2:
        patterns['double_top'] = True
    
    return patterns
""")
            print("✅ Created technical_indicators.py")
        
        # Run the Streamlit app
        print("🚀 Launching AI Financial Advisor...")
        os.system(f"{sys.executable} -m streamlit run app/main.py")
        
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 