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

# For numpy functions we need to use, import them carefully
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class PricePredictionModel:
    """
    Machine learning-based price prediction model for stocks
    """
    
    def __init__(self):
        """Initialize the model with default parameters"""
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.accuracy = 0
        self.window_size = 20  # Number of days to look back
        self.trained = False
        self.model_type = 'random_forest'  # 'linear' or 'random_forest'
    
    def _create_features(self, data):
        """
        Create features for the model
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stock price data including 'Close' column
            
        Returns:
        --------
        tuple
            X (features) and y (target)
        """
        df = data.copy()
        
        # Create lag features
        for i in range(1, self.window_size + 1):
            df[f'lag_{i}'] = df['Close'].shift(i)
        
        # Create moving average features
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_10'] = df['Close'].rolling(window=10).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()
        
        # Volatility features
        df['volatility_5'] = df['Close'].rolling(window=5).std()
        df['volatility_10'] = df['Close'].rolling(window=10).std()
        
        # Price momentum
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Create target
        df['target'] = df['Close']
        
        # Drop NaN rows
        df = df.dropna()
        
        # Define features and target
        features = [f'lag_{i}' for i in range(1, self.window_size + 1)] + \
                  ['ma_5', 'ma_10', 'ma_20', 'ma_50', 'volatility_5', 'volatility_10', 'momentum_5', 'momentum_10']
        
        X = df[features]
        y = df['target']
        
        return X, y
    
    def train(self, stock_data):
        """
        Train the model on historical stock data
        
        Parameters:
        -----------
        stock_data : pandas.DataFrame
            DataFrame with stock price data including 'Close' column
        """
        # Ensure data is in the right format
        data = stock_data.copy()
        data = data[['Close']]
        
        # Adjust window size based on available data
        min_data_points = 30  # absolute minimum required
        if len(data) < min_data_points:
            raise ValueError(f"Not enough data to train model. Need at least {min_data_points} data points, but got {len(data)}.")
            
        # Adjust window size if needed
        if len(data) < self.window_size * 2:
            self.window_size = max(5, len(data) // 4)  # Use at most 1/4 of data for window
        
        # Create features
        X, y = self._create_features(data)
        
        if len(X) < 10:  # Need at least 10 rows after feature creation
            raise ValueError(f"Not enough usable data points after feature creation. Got {len(X)} rows.")
        
        # Split data into train and test sets (80/20)
        # For very small datasets, use more data for training
        train_percent = 0.9 if len(X) < 30 else 0.8
        split_idx = int(len(X) * train_percent)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Handle cases where test set might be empty
        if len(X_test) == 0:
            split_idx = max(1, int(len(X) * 0.7))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train the model
        if self.model_type == 'linear':
            self.model = LinearRegression()
        else:  # random_forest
            # Use fewer estimators for smaller datasets
            n_estimators = min(100, max(10, len(X_train) // 2))
            self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate accuracy metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate a simplified accuracy percentage
        # Using Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs(safe_divide(y_test - y_pred, y_test))) * 100
        # Clip accuracy to reasonable range (0-100%)
        self.accuracy = max(0, min(100, 100 - mape))
        
        # Set trained flag
        self.trained = True
        
        # Save the last row of data for prediction
        self.last_data = data.tail(self.window_size)
    
    def predict(self, stock_data, future_days=30):
        """
        Generate price predictions for future days
        
        Parameters:
        -----------
        stock_data : pandas.DataFrame
            DataFrame with stock price data including 'Close' column
        future_days : int
            Number of days to predict into the future
            
        Returns:
        --------
        numpy.ndarray
            Array of predicted prices
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get the latest data for prediction
        data = stock_data.copy()
        data = data[['Close']]
        
        # Prepare for predictions
        predictions = []
        
        # Create a new DataFrame with continuous integer index to avoid index-related errors
        pred_data = data.copy().tail(self.window_size + future_days).reset_index(drop=True)
        
        # Iteratively predict future days
        for i in range(future_days):
            try:
                # Create features for the current step
                X_pred, _ = self._create_features(pred_data)
                if len(X_pred) == 0:
                    # Not enough data to make prediction (e.g., after dropping NaNs)
                    # Use a simple extrapolation based on recent trend
                    last_prices = pred_data['Close'].iloc[-5:].values
                    if len(last_prices) >= 2:
                        # Simple trend extrapolation
                        avg_change = np.mean(np.diff(last_prices))
                        pred = last_prices[-1] + avg_change
                    else:
                        # Use last price if not enough data for trend
                        pred = pred_data['Close'].iloc[-1]
                else:
                    X_pred = X_pred.iloc[-1:].reset_index(drop=True)  # Take only the last row
                    
                    # Scale features
                    X_pred_scaled = self.scaler.transform(X_pred)
                    
                    # Make prediction
                    pred = self.model.predict(X_pred_scaled)[0]
                
                predictions.append(pred)
                
                # Add the new prediction to the data, using next integer index
                next_idx = len(pred_data)
                new_row = pd.DataFrame({'Close': [pred]}, index=[next_idx])
                pred_data = pd.concat([pred_data, new_row])
                
                # Calculate new features directly (safer than trying to assign to specific indices)
                # This creates a fresh set of features for all data including the new prediction
                # These will be used in the next iteration
                if i < future_days - 1:  # No need to calculate features on the last iteration
                    # Create lag features (we do this explicitly to avoid indexing errors)
                    for j in range(1, self.window_size + 1):
                        pred_data[f'lag_{j}'] = pred_data['Close'].shift(j)
                    
                    # Create moving average features
                    window_sizes = [5, 10, 20, 50]
                    for size in window_sizes:
                        if len(pred_data) >= size:
                            pred_data[f'ma_{size}'] = pred_data['Close'].rolling(window=size).mean()
                    
                    # Volatility features
                    if len(pred_data) >= 5:
                        pred_data['volatility_5'] = pred_data['Close'].rolling(window=5).std()
                    if len(pred_data) >= 10:
                        pred_data['volatility_10'] = pred_data['Close'].rolling(window=10).std()
                    
                    # Price momentum
                    if len(pred_data) >= 6:  # Need at least 6 rows for 5-period momentum
                        pred_data['momentum_5'] = pred_data['Close'] / pred_data['Close'].shift(5) - 1
                    if len(pred_data) >= 11:  # Need at least 11 rows for 10-period momentum
                        pred_data['momentum_10'] = pred_data['Close'] / pred_data['Close'].shift(10) - 1
            
            except Exception as e:
                # If prediction fails for this step, use a simple forecasting method
                if len(predictions) > 0:
                    # Use the last prediction
                    pred = predictions[-1]
                else:
                    # Use the last actual price
                    pred = data['Close'].iloc[0]
                
                predictions.append(pred)
                # Print the error but continue with predictions
                print(f"Warning: Error during prediction step {i}: {str(e)}")
        
        return np.array(predictions)
    
    def evaluate(self, stock_data, days_to_evaluate=30):
        """
        Evaluate model performance on historical data
        
        Parameters:
        -----------
        stock_data : pandas.DataFrame
            DataFrame with stock price data including 'Close' column
        days_to_evaluate : int
            Number of days to use for backtesting
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Extract validation data
        validation_data = stock_data.iloc[-days_to_evaluate:]['Close'].values
        
        # Get training data excluding validation period
        training_data = stock_data.iloc[:-days_to_evaluate]
        
        # Make predictions for validation period
        predictions = self.predict(training_data, future_days=days_to_evaluate)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(validation_data, predictions)
        rmse = np.sqrt(mean_squared_error(validation_data, predictions))
        mape = np.mean(np.abs(safe_divide(validation_data - predictions, validation_data))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'accuracy': 100 - mape
        } 