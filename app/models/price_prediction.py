import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

class PricePredictionModel:
    """
    LSTM-based price prediction model for stocks
    """
    
    def __init__(self):
        """Initialize the model with default parameters"""
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.accuracy = 0
        self.sequence_length = 60  # Number of time steps to look back
        self.trained = False
    
    def _create_sequences(self, data):
        """
        Create sequences for LSTM model
        
        Parameters:
        -----------
        data : numpy.ndarray
            Scaled price data
            
        Returns:
        --------
        tuple
            X (sequences) and y (labels)
        """
        X = []
        y = []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])
            
        return np.array(X), np.array(y)
    
    def train(self, stock_data):
        """
        Train the LSTM model on historical stock data
        
        Parameters:
        -----------
        stock_data : pandas.DataFrame
            DataFrame with stock price data including 'Close' column
        """
        # Extract and reshape close prices
        data = stock_data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Build LSTM model
        self.model = Sequential()
        
        # First LSTM layer with dropout
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        
        # Second LSTM layer with dropout
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        # In a real application, you might want to use callbacks for early stopping
        # and learning rate scheduling
        self.model.fit(
            X_train, y_train,
            epochs=25,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Inverse transform the predictions
        y_pred_actual = self.scaler.inverse_transform(y_pred)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate accuracy metrics
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        r2 = r2_score(y_test_actual, y_pred_actual)
        
        # Calculate a simplified accuracy percentage
        # Using Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
        self.accuracy = 100 - mape  # Convert error to accuracy
        
        # Set trained flag
        self.trained = True
    
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
        
        # Extract and reshape close prices
        data = stock_data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.transform(data)
        
        # Create input sequence for prediction
        # Use the last `sequence_length` days for the initial prediction
        input_data = scaled_data[-self.sequence_length:]
        
        # Initialize array for predictions
        predictions = []
        
        # Generate predictions for each future day
        for _ in range(future_days):
            # Reshape for LSTM [samples, time steps, features]
            x_input = np.reshape(input_data, (1, self.sequence_length, 1))
            
            # Predict next day
            pred = self.model.predict(x_input, verbose=0)
            
            # Append prediction to the list
            predictions.append(pred[0][0])
            
            # Update input sequence (remove oldest day, add prediction)
            input_data = np.append(input_data[1:], pred, axis=0)
        
        # Convert predictions to actual prices
        predictions = np.array(predictions).reshape(-1, 1)
        predicted_prices = self.scaler.inverse_transform(predictions)
        
        return predicted_prices.flatten()
    
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