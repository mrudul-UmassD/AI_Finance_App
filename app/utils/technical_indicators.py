import pandas as pd
# Replace numpy import with our numpy_compat module
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
import pandas_ta as ta

def calculate_indicators(data):
    """
    Calculate technical indicators for stock data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing stock price data with columns 'Open', 'High', 'Low', 'Close', 'Volume'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original data and technical indicators
    """
    # Create a copy of the data
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    
    # Exponential Moving Averages
    df['EMA_12'] = ta.ema(df['Close'], length=12)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_26'] = ta.ema(df['Close'], length=26)
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Histogram'] = macd['MACDh_12_26_9']
    
    # Relative Strength Index
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Bollinger Bands
    bollinger = ta.bbands(df['Close'], length=20)
    df['BB_Upper'] = bollinger['BBU_20_2.0']
    df['BB_Middle'] = bollinger['BBM_20_2.0']
    df['BB_Lower'] = bollinger['BBL_20_2.0']
    
    # Average True Range
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # On-Balance Volume
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    
    # Stochastic Oscillator
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    df['Stoch_K'] = stoch['STOCHk_14_3_3']
    df['Stoch_D'] = stoch['STOCHd_14_3_3']
    
    # Average Directional Index
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['DI+'] = adx['DMP_14']
    df['DI-'] = adx['DMN_14']
    
    # Fibonacci Retracement Levels (not a traditional indicator, but useful)
    max_price = df['Close'].max()
    min_price = df['Close'].min()
    diff = max_price - min_price
    
    df['Fib_0.0'] = min_price
    df['Fib_0.236'] = min_price + 0.236 * diff
    df['Fib_0.382'] = min_price + 0.382 * diff
    df['Fib_0.5'] = min_price + 0.5 * diff
    df['Fib_0.618'] = min_price + 0.618 * diff
    df['Fib_0.786'] = min_price + 0.786 * diff
    df['Fib_1.0'] = max_price
    
    # Return the DataFrame with all indicators
    return df

def generate_signals(data):
    """
    Generate trading signals based on technical indicators
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing technical indicators
    
    Returns:
    --------
    dict
        Dictionary with signal type and strength
    """
    signals = {}
    
    # MACD Signal
    if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] and data['MACD'].iloc[-2] <= data['MACD_Signal'].iloc[-2]:
        signals['MACD'] = {'signal': 'BUY', 'strength': 0.7}
    elif data['MACD'].iloc[-1] < data['MACD_Signal'].iloc[-1] and data['MACD'].iloc[-2] >= data['MACD_Signal'].iloc[-2]:
        signals['MACD'] = {'signal': 'SELL', 'strength': 0.7}
    else:
        signals['MACD'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    # RSI Signal
    if data['RSI'].iloc[-1] < 30:
        signals['RSI'] = {'signal': 'BUY', 'strength': 0.8}
    elif data['RSI'].iloc[-1] > 70:
        signals['RSI'] = {'signal': 'SELL', 'strength': 0.8}
    else:
        signals['RSI'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    # Moving Average Signal
    if data['Close'].iloc[-1] > data['SMA_20'].iloc[-1] and data['Close'].iloc[-1] > data['SMA_50'].iloc[-1]:
        signals['MA'] = {'signal': 'BUY', 'strength': 0.6}
    elif data['Close'].iloc[-1] < data['SMA_20'].iloc[-1] and data['Close'].iloc[-1] < data['SMA_50'].iloc[-1]:
        signals['MA'] = {'signal': 'SELL', 'strength': 0.6}
    else:
        signals['MA'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    # Bollinger Bands Signal
    if data['Close'].iloc[-1] < data['BB_Lower'].iloc[-1]:
        signals['BB'] = {'signal': 'BUY', 'strength': 0.7}
    elif data['Close'].iloc[-1] > data['BB_Upper'].iloc[-1]:
        signals['BB'] = {'signal': 'SELL', 'strength': 0.7}
    else:
        signals['BB'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    # Stochastic Oscillator Signal
    if data['Stoch_K'].iloc[-1] < 20 and data['Stoch_D'].iloc[-1] < 20 and data['Stoch_K'].iloc[-1] > data['Stoch_D'].iloc[-1]:
        signals['Stoch'] = {'signal': 'BUY', 'strength': 0.6}
    elif data['Stoch_K'].iloc[-1] > 80 and data['Stoch_D'].iloc[-1] > 80 and data['Stoch_K'].iloc[-1] < data['Stoch_D'].iloc[-1]:
        signals['Stoch'] = {'signal': 'SELL', 'strength': 0.6}
    else:
        signals['Stoch'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    # ADX Signal (Trend Strength)
    if data['ADX'].iloc[-1] > 25:
        # Strong trend exists, check DI lines for direction
        if data['DI+'].iloc[-1] > data['DI-'].iloc[-1]:
            signals['ADX'] = {'signal': 'BUY', 'strength': 0.7}
        else:
            signals['ADX'] = {'signal': 'SELL', 'strength': 0.7}
    else:
        signals['ADX'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    # Overall Signal
    buy_count = sum(1 for s in signals.values() if s['signal'] == 'BUY')
    sell_count = sum(1 for s in signals.values() if s['signal'] == 'SELL')
    
    buy_strength = sum(s['strength'] for s in signals.values() if s['signal'] == 'BUY')
    sell_strength = sum(s['strength'] for s in signals.values() if s['signal'] == 'SELL')
    
    if buy_count > sell_count and buy_strength > sell_strength:
        signals['OVERALL'] = {'signal': 'BUY', 'strength': buy_strength / (buy_count if buy_count > 0 else 1)}
    elif sell_count > buy_count and sell_strength > buy_strength:
        signals['OVERALL'] = {'signal': 'SELL', 'strength': sell_strength / (sell_count if sell_count > 0 else 1)}
    else:
        signals['OVERALL'] = {'signal': 'NEUTRAL', 'strength': 0.5}
    
    return signals

def identify_patterns(data, window=5):
    """
    Identify common chart patterns in the data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing price data
    window : int
        Window size for pattern detection
        
    Returns:
    --------
    dict
        Dictionary with identified patterns and their confidence levels
    """
    patterns = {}
    
    # Head and Shoulders Pattern (a reversal pattern)
    # This is a simplified implementation - real pattern detection would be more complex
    if len(data) >= 3 * window:
        # Divide into three windows
        left = data['Close'].iloc[-3 * window:-2 * window]
        head = data['Close'].iloc[-2 * window:-window]
        right = data['Close'].iloc[-window:]
        
        left_max = left.max()
        left_max_idx = left.idxmax()
        head_max = head.max()
        head_max_idx = head.idxmax()
        right_max = right.max()
        right_max_idx = right.idxmax()
        
        # Check if the pattern matches head and shoulders
        if (head_max > left_max and head_max > right_max and 
            abs(left_max - right_max) / left_max < 0.1):  # Left and right shoulders roughly equal height
            patterns['Head and Shoulders'] = {
                'type': 'Bearish Reversal', 
                'confidence': 0.6
            }
    
    # Double Bottom Pattern (a bullish reversal pattern)
    if len(data) >= 2 * window:
        recent = data['Close'].iloc[-2 * window:]
        
        # Find the two lowest points in the window
        lowest = recent.nsmallest(2)
        low1_idx = lowest.index[0]
        low2_idx = lowest.index[1]
        
        # Check if they are roughly the same level
        if (abs(data.loc[low1_idx, 'Close'] - data.loc[low2_idx, 'Close']) / data.loc[low1_idx, 'Close'] < 0.05 and
            abs(low1_idx - low2_idx) > window / 2):  # The bottoms should be separated
            patterns['Double Bottom'] = {
                'type': 'Bullish Reversal', 
                'confidence': 0.7
            }
    
    # Ascending Triangle (usually a continuation pattern)
    if len(data) >= 2 * window:
        recent = data['Close'].iloc[-2 * window:]
        
        # Identify resistance level
        upper_bound = recent.quantile(0.8)
        
        # Count how many times price approaches but doesn't break resistance
        touches = sum(1 for x in recent if abs(x - upper_bound) / upper_bound < 0.02)
        
        # Check for higher lows
        lows = []
        for i in range(-2 * window, -1, window // 2):
            lows.append(data['Low'].iloc[i-window//2:i].min())
        
        if len(lows) >= 3 and all(lows[i] < lows[i+1] for i in range(len(lows)-1)) and touches >= 2:
            patterns['Ascending Triangle'] = {
                'type': 'Bullish Continuation', 
                'confidence': 0.65
            }
    
    # Descending Triangle (usually a continuation pattern)
    if len(data) >= 2 * window:
        recent = data['Close'].iloc[-2 * window:]
        
        # Identify support level
        lower_bound = recent.quantile(0.2)
        
        # Count how many times price approaches but doesn't break support
        touches = sum(1 for x in recent if abs(x - lower_bound) / lower_bound < 0.02)
        
        # Check for lower highs
        highs = []
        for i in range(-2 * window, -1, window // 2):
            highs.append(data['High'].iloc[i-window//2:i].max())
        
        if len(highs) >= 3 and all(highs[i] > highs[i+1] for i in range(len(highs)-1)) and touches >= 2:
            patterns['Descending Triangle'] = {
                'type': 'Bearish Continuation', 
                'confidence': 0.65
            }
    
    return patterns 