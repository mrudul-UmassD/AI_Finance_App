"""
AI Financial Advisor - Main runner script

This script launches the AI Financial Advisor application.
"""

import os
import sys
import subprocess
import time

def main():
    """Main function to run the application"""
    try:
        print("🚀 Setting up AI Financial Advisor...")
        
        # Check if required packages are installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        
        # Need to make sure numpy is properly installed before trying to use it
        print("🔍 Verifying numpy installation...")
        try:
            import numpy as np
            print(f"✅ NumPy version {np.__version__} loaded successfully")
        except ImportError:
            print("⚠️ NumPy not found, reinstalling...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.26.2"])
            print("✅ NumPy reinstalled")
            time.sleep(1)  # Give a moment for the installation to settle
        
        # Make sure NLTK resources are installed
        try:
            print("🔍 Setting up NLTK resources...")
            import nltk
            nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
            os.makedirs(nltk_data_path, exist_ok=True)
            
            # Download essential resources
            for resource in ['vader_lexicon', 'punkt', 'stopwords']:
                try:
                    nltk.data.find(f"{resource}")
                    print(f"✅ NLTK resource '{resource}' already available")
                except LookupError:
                    print(f"⚠️ Downloading NLTK resource '{resource}'...")
                    nltk.download(resource, quiet=True, download_dir=nltk_data_path)
                    print(f"✅ Downloaded NLTK resource '{resource}'")
        except Exception as e:
            print(f"⚠️ Warning: Error setting up NLTK: {str(e)}")
            print("⚠️ The application may have limited functionality without NLTK resources.")
        
        # Make sure utils directory exists
        utils_dir = os.path.join("app", "utils")
        os.makedirs(utils_dir, exist_ok=True)
        
        # Check if numpy_compat.py exists, create it if not
        numpy_compat_file = os.path.join(utils_dir, "numpy_compat.py")
        if not os.path.exists(numpy_compat_file):
            print("⚠️ Creating numpy_compat.py for better compatibility...")
            with open(numpy_compat_file, "w") as f:
                f.write('''"""
Compatibility module for numpy functions to ensure compatibility with different Python versions
"""

def is_nan(value):
    """
    Check if a value is NaN in a way that's compatible with all numpy versions
    """
    try:
        # Try to compare with itself - NaN is the only value that doesn't equal itself
        return value != value
    except:
        # If comparison fails, it's not a NaN
        return False

def get_nan():
    """
    Get a NaN value in a way that's compatible with all numpy versions
    """
    try:
        import numpy as np
        return np.nan
    except (ImportError, AttributeError):
        # Fallback to create NaN without numpy
        return float('nan')

def safe_divide(a, b):
    """
    Safe division that returns NaN on divide by zero
    """
    try:
        if b == 0:
            return get_nan()
        return a / b
    except:
        return get_nan()
''')
            print("✅ Created numpy_compat.py")
        
        # Check if technical_indicators.py exists, create it if not
        tech_indicators_file = os.path.join(utils_dir, "technical_indicators.py")
        
        if not os.path.exists(tech_indicators_file):
            print("⚠️ Creating missing technical_indicators.py file...")
            with open(tech_indicators_file, "w") as f:
                f.write("""import pandas as pd
# Import our utils for numpy compatibility
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
    
    rs = safe_divide(avg_gain, avg_loss)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * 2)
    
    # Percentage Price Oscillator
    df['PPO'] = safe_divide((df['EMA_12'] - df['EMA_26']), df['EMA_26']) * 100
    
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
    
    plus_di = 100 * safe_divide(plus_dm.rolling(window=14).mean(), atr)
    minus_di = 100 * safe_divide(minus_dm.rolling(window=14).mean(), atr)
    
    dx = 100 * safe_divide(abs(plus_di - minus_di), (plus_di + minus_di))
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