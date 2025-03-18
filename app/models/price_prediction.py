import pandas as pd
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
        
        # Create features
        X, y = self._create_features(data)
        
        if len(X) < self.window_size + 10:
            raise ValueError(f"Not enough data to train model. Need at least {self.window_size + 10} data points.")
        
        # Split data into train and test sets (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train the model
        if self.model_type == 'linear':
            self.model = LinearRegression()
        else:  # random_forest
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate accuracy metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate a simplified accuracy percentage
        # Using Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        self.accuracy = 100 - mape  # Convert error to accuracy
        
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
        pred_data = data.copy().tail(self.window_size + future_days)
        
        # Iteratively predict future days
        for i in range(future_days):
            # Create features for the current step
            X_pred, _ = self._create_features(pred_data)
            X_pred = X_pred.tail(1)  # Take only the last row
            
            # Scale features
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Make prediction
            pred = self.model.predict(X_pred_scaled)[0]
            predictions.append(pred)
            
            # Update data for next prediction
            next_idx = len(pred_data)
            pred_data.loc[next_idx] = [pred]  # Add new prediction to Close
            
            # Calculate new features for the next step
            for j in range(1, self.window_size + 1):
                if next_idx - j >= 0:
                    pred_data.loc[next_idx, f'lag_{j}'] = pred_data.loc[next_idx - j, 'Close']
            
            # Calculate MAs
            pred_data.loc[next_idx, 'ma_5'] = pred_data['Close'].tail(5).mean()
            pred_data.loc[next_idx, 'ma_10'] = pred_data['Close'].tail(10).mean()
            pred_data.loc[next_idx, 'ma_20'] = pred_data['Close'].tail(20).mean()
            pred_data.loc[next_idx, 'ma_50'] = pred_data['Close'].tail(50).mean()
            
            # Calculate volatility
            pred_data.loc[next_idx, 'volatility_5'] = pred_data['Close'].tail(5).std()
            pred_data.loc[next_idx, 'volatility_10'] = pred_data['Close'].tail(10).std()
            
            # Calculate momentum
            if next_idx - 5 >= 0:
                pred_data.loc[next_idx, 'momentum_5'] = pred_data.loc[next_idx, 'Close'] / pred_data.loc[next_idx - 5, 'Close'] - 1
            if next_idx - 10 >= 0:
                pred_data.loc[next_idx, 'momentum_10'] = pred_data.loc[next_idx, 'Close'] / pred_data.loc[next_idx - 10, 'Close'] - 1
        
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
        mape = np.mean(np.abs((validation_data - predictions) / validation_data)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'accuracy': 100 - mape
        } 