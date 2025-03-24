#!/usr/bin/env python3
"""
Tesla Stock Prediction Model

This script builds and evaluates a machine learning model for Tesla stock price prediction.
It downloads recent stock data, processes it with technical indicators, trains a neural network,
and provides trading signals based on the predictions.

Dependencies: pandas, numpy, matplotlib, scikit-learn, joblib, yfinance
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("tesla_model.log"), logging.StreamHandler()]
)
logger = logging.getLogger("tesla_stock_model")


class TeslaStockPredictor:
    """Model to predict Tesla stock price movements using technical indicators."""
    
    def __init__(
        self,
        ticker: str = "TSLA",
        data_period: int = 730,  # 2 years in days
        features: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        model_filename: str = "tesla_stock_model.pkl",
        scaler_filename: str = "tesla_scaler.pkl",
        csv_filename: str = "tesla_stock_data.csv"
    ):
        """
        Initialize the predictor with configuration parameters.
        
        Args:
            ticker: Stock ticker symbol
            data_period: Number of days of historical data to use
            features: List of features to use for prediction
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            model_filename: Filename to save the trained model
            scaler_filename: Filename to save the feature scaler
            csv_filename: Filename to save the downloaded stock data
        """
        self.ticker = ticker
        self.data_period = data_period
        self.features = features or ['Close', 'Volume', 'SMA_10', 'SMA_20', 'RSI_14', 'Price_Change']
        self.test_size = test_size
        self.random_state = random_state
        self.model_filename = model_filename
        self.scaler_filename = scaler_filename
        self.csv_filename = csv_filename
        
        # Will be initialized during processing
        self.df = None
        self.model = None
        self.scaler = None
        self.accuracy = None
    
    def download_data(self) -> pd.DataFrame:
        """
        Download the latest stock data for the configured ticker.
        
        Returns:
            DataFrame with the downloaded stock data
        
        Raises:
            ValueError: If download fails or returns empty data
        """
        logger.info(f"Downloading {self.ticker} stock data for the past {self.data_period} days...")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.data_period)
            
            stock_df = yf.download(
                self.ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if stock_df.empty:
                raise ValueError(f"No data retrieved for {self.ticker}")
                
            stock_df.reset_index(inplace=True)
            stock_df.to_csv(self.csv_filename, index=False)
            logger.info(f"Downloaded {len(stock_df)} rows of {self.ticker} data")
            
            return stock_df
            
        except Exception as e:
            logger.error(f"Error downloading stock data: {str(e)}")
            raise
    
    def preprocess_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Process raw stock data into a format suitable for modeling.
        
        Args:
            df: Raw stock DataFrame, or None to load from CSV
            
        Returns:
            Processed DataFrame with technical indicators
            
        Raises:
            ValueError: If required columns are missing or data processing fails
        """
        logger.info("Preprocessing data and engineering features...")
        
        try:
            # Load from CSV if no DataFrame provided
            if df is None:
                if not os.path.exists(self.csv_filename):
                    logger.warning(f"CSV file {self.csv_filename} not found, downloading data...")
                    self.download_data()
                
                raw_df = pd.read_csv(self.csv_filename)
            else:
                raw_df = df.copy()
            
            # Standardize column names and datatypes
            raw_df.columns = raw_df.columns.str.strip().str.lower()
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'date']
            
            for col in required_cols:
                if col not in raw_df.columns:
                    raise ValueError(f"Missing required column: '{col}'")
            
            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
            
            # Process date column
            raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')
            raw_df.dropna(subset=['date'], inplace=True)
            raw_df.set_index('date', inplace=True)
            
            # Standardize column names for features
            processed_df = raw_df.rename(columns={
                'close': 'Close',
                'volume': 'Volume',
                'open': 'Open',
                'high': 'High',
                'low': 'Low'
            })
            
            # Create technical indicators
            processed_df = self._add_technical_indicators(processed_df)
            
            # Add target variable (binary classification: price up or down next day)
            processed_df['Target'] = (processed_df['Close'].shift(-1) > processed_df['Close']).astype(int)
            
            # Drop rows with missing values
            processed_df.dropna(inplace=True)
            
            if processed_df.empty:
                raise ValueError("No data available after preprocessing")
                
            self.df = processed_df
            return processed_df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Simple Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Price Changes
        df['Price_Change'] = df['Close'].diff()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Handle division by zero
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        return df
    
    def train_model(self) -> Tuple[MLPClassifier, float]:
        """
        Train the neural network model on the processed data.
        
        Returns:
            Tuple of (trained model, accuracy score)
            
        Raises:
            ValueError: If training fails or data is not properly prepared
        """
        logger.info("Training neural network model...")
        
        try:
            if self.df is None or self.df.empty:
                logger.warning("No preprocessed data available, running preprocessing...")
                self.preprocess_data()
            
            # Prepare features and target
            X = self.df[self.features]
            y = self.df['Target']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=False
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            self.accuracy = accuracy_score(y_test, y_pred)
            
            # Save model and scaler
            joblib.dump(self.model, self.model_filename)
            joblib.dump(self.scaler, self.scaler_filename)
            
            logger.info(f"Model trained with accuracy: {self.accuracy:.4f}")
            
            # Detailed evaluation
            report = classification_report(y_test, y_pred)
            logger.info(f"Classification Report:\n{report}")
            
            return self.model, self.accuracy
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def visualize_feature_importance(self, save_path: Optional[str] = None) -> None:
        """
        Create and optionally save a visualization of feature importance.
        
        Args:
            save_path: Path to save the visualization (None for display only)
        """
        logger.info("Visualizing feature importance...")
        
        try:
            if self.model is None:
                logger.warning("No trained model available, loading or training model...")
                self.load_or_train_model()
            
            plt.figure(figsize=(12, 6))
            
            # Approximate feature importance using first layer weights
            feature_importance = np.mean(np.abs(self.model.coefs_[0]), axis=1)
            
            if len(feature_importance) == len(self.features):
                sorted_idx = np.argsort(feature_importance)
                plt.barh(np.array(self.features)[sorted_idx], feature_importance[sorted_idx])
                plt.title("Feature Importance (MLP First Layer Weights)")
                plt.xlabel("Importance Score")
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path)
                    logger.info(f"Feature importance visualization saved to {save_path}")
                else:
                    plt.show()
            else:
                logger.error(f"Feature importance shape mismatch: expected {len(self.features)} but got {len(feature_importance)}")
                
        except Exception as e:
            logger.error(f"Error visualizing feature importance: {str(e)}")
            raise
    
    def load_or_train_model(self) -> None:
        """Load an existing model or train a new one if it doesn't exist."""
        try:
            # Try to load existing model
            if os.path.exists(self.model_filename) and os.path.exists(self.scaler_filename):
                logger.info(f"Loading existing model from {self.model_filename}")
                self.model = joblib.load(self.model_filename)
                self.scaler = joblib.load(self.scaler_filename)
            else:
                logger.info("No existing model found, training new model...")
                self.train_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}. Training new model.")
            self.train_model()
    
    def get_trading_decision(self, data_row: pd.Series) -> Dict[str, Any]:
        """
        Get a trading decision based on the latest data.
        
        Args:
            data_row: Data row with features needed for prediction
            
        Returns:
            Dictionary with decision and supporting information
        """
        try:
            if self.model is None or self.scaler is None:
                self.load_or_train_model()
            
            # Extract and scale features
            features_values = data_row[self.features].values.reshape(1, -1)
            features_scaled = self.scaler.transform(features_values)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Determine confidence and decision
            confidence = probabilities[prediction]
            decision = "Hold"  # Default decision
            
            # Apply decision logic
            if confidence >= 0.6:
                decision = "Buy" if prediction == 1 else "Sell"
            
            return {
                "decision": decision,
                "prediction": "Up" if prediction == 1 else "Down",
                "confidence": confidence,
                "price": data_row['Close'],
                "date": data_row.name if hasattr(data_row, 'name') else datetime.now().date()
            }
            
        except Exception as e:
            logger.error(f"Error getting trading decision: {str(e)}")
            raise
    
    def get_latest_decision(self) -> Dict[str, Any]:
        """
        Get trading decision based on the most recent available data.
        
        Returns:
            Dictionary with decision and supporting information
        """
        logger.info("Getting latest trading decision...")
        
        if self.df is None or self.df.empty:
            self.preprocess_data()
        
        latest_row = self.df.iloc[-1]
        decision_data = self.get_trading_decision(latest_row)
        
        # Format the output
        now = datetime.now()
        market_status = "before market open" if now.hour < 9 and now.weekday() < 5 else "after market close" if now.hour >= 16 and now.weekday() < 5 else "during market hours" if now.weekday() < 5 else "weekend"
        
        logger.info(f"Latest decision: {decision_data['decision']} with {decision_data['confidence']:.2f} confidence")
        
        # Add market context
        decision_data["market_status"] = market_status
        return decision_data


def main():
    """Entry point for the Tesla stock prediction script."""
    try:
        # Create and initialize predictor
        predictor = TeslaStockPredictor()
        
        # Run the complete workflow
        predictor.download_data()
        predictor.preprocess_data()
        predictor.train_model()
        
        # Get latest trading decision
        decision = predictor.get_latest_decision()
        
        # Display results
        print("\n=== Tesla Stock Trading Decision ===")
        print(f"Date: {decision['date']}")
        print(f"Current Price: ${decision['price']:.2f}")
        print(f"Prediction: {decision['prediction']} (Confidence: {decision['confidence']:.2f})")
        print(f"Recommended Action: {decision['decision']}")
        print(f"Market Status: {decision['market_status']}")
        
        # Generate visualization
        predictor.visualize_feature_importance("feature_importance.png")
        print("\nFeature importance visualization saved to 'feature_importance.png'")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
