# This script is compatible with Anaconda Python environment
# To run: Ensure dependencies are installed via conda or pip

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import yfinance as yf

# Download latest Tesla stock data
print("Downloading latest TSLA stock data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 2)  # last 2 years of data
tsla_df = yf.download('TSLA', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
tsla_df.reset_index(inplace=True)
tsla_df.to_csv("tesla_stock_data.csv", index=False)  # save to CSV

# Load Tesla stock data from CSV
raw_df = pd.read_csv("tesla_stock_data.csv")
raw_df.columns = raw_df.columns.str.strip().str.lower()
for col in ['open', 'high', 'low', 'close', 'volume']:
    if col in raw_df.columns:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
    else:
        raise KeyError(f"Missing expected column: '{col}'")

raw_df.columns = raw_df.columns.str.strip().str.lower()
raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')
raw_df.dropna(subset=['date'], inplace=True)
raw_df.set_index('date', inplace=True)

# Rename columns to match feature expectations
df = raw_df.rename(columns={
    'close': 'Close',
    'volume': 'Volume',
    'open': 'Open',
    'high': 'High',
    'low': 'Low'
})

# Feature Engineering: Create technical indicators
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['Price_Change'] = df['Close'].diff()

# Robust RSI calculation
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
df['RSI_14'] = 100 - (100 / (1 + rs))

# Target Variable: 1 if next day's closing price is higher, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop missing values
df.dropna(inplace=True)

# Feature Selection
features = ['Close', 'Volume', 'SMA_10', 'SMA_20', 'RSI_14', 'Price_Change']
X = df[features]
y = df['Target']

# Validate dataset size before splitting
if X.shape[0] == 0:
    raise ValueError("No data available after preprocessing. Check your CSV or feature engineering steps.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training (Multi-Layer Perceptron Classifier)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                   max_iter=1000, random_state=42, verbose=True)
mlp.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(mlp, 'tsla_stock_mlp_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Predictions
y_pred = mlp.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"MLP Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Feature Importance (Approximated using first layer weights)
plt.figure(figsize=(12, 6))
feature_importance = np.mean(np.abs(mlp.coefs_[0]), axis=1)
if len(feature_importance) == len(features):
    sorted_idx = np.argsort(feature_importance)
    plt.barh(np.array(features)[sorted_idx], feature_importance[sorted_idx])
else:
    print(f"Feature importance mismatch: expected {len(features)} but got {len(feature_importance)}")
plt.title("Feature Importance (MLP First Layer Weights)")
plt.xlabel("Importance Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Trading Decision Function ---
def get_trading_decision(latest_row):
    latest_features = latest_row[features].values.reshape(1, -1)
    latest_features_scaled = scaler.transform(latest_features)
    prediction = mlp.predict(latest_features_scaled)[0]
    proba = mlp.predict_proba(latest_features_scaled)[0][prediction]

    if proba >= 0.6 and prediction == 1:
        return "Buy"
    elif proba >= 0.6 and prediction == 0:
        return "Sell"
    else:
        return "Hold"

# --- Simulate Future Predictions Based on Latest Available Data ---
print("\n=== Agent Simulated Predictions for March 24–28, 2025 ===")
latest_known = df.iloc[-1]
initial_cash = 10000.0
cash = initial_cash
shares = 0

future_dates = pd.date_range(start="2025-03-24", end="2025-03-28", freq='B')
for date in future_dates:
    simulated_row = latest_known.copy()
    simulated_row.name = date  # assign the new date as index
    decision = get_trading_decision(simulated_row)

    price = simulated_row['Close']
    if decision == "Buy" and cash > 0:
        shares = cash / price * 0.99  # account for 1% fee
        cash = 0
    elif decision == "Sell" and shares > 0:
        cash = shares * price * 0.99  # account for 1% fee
        shares = 0
    portfolio_value = cash + shares * simulated_row['Close']
    print(f"{date.date()} → {decision} (simulated), Portfolio Value: ${portfolio_value:.2f}")
