# Tesla-Stock-Prediction-Model


**Agents Performance Summary**

Final Account Value: 
Initial Capital: $10,000
Transaction Fee: 1% on every Buy/Sell
Period Simulated: Based on data from tesla_stock_data.csv

**Trading Strategy Logic**

**Buy Conditions:**

Model predicts the price will go up (next day)
Current price is above the 5-day Moving Average (indicates upward trend)
Invest 50% of current capital into Tesla shares

**Sell Conditions:**

Model predicts the price will fall, OR
Current price drops >3% below average buy price
Sell all shares to protect against loss

**Hold:**

No trade action if neither Buy nor Sell conditions are met

**ML Model Summary**

Model Used: Multi-Layer Perceptron (MLP)
Inputs: Last 10 days of Close and Volume (windowed input)
Target: Binary prediction – whether price will rise the next day
Test Accuracy: ~ 51%

**Analysis of Results**

**Strengths:**

MLP captured some patterns in Tesla’s historical movements
Strategy limited risk by avoiding impulsive selling
Prevented major losses through a 3% stop-loss rule

**Limitations:**

Accuracy just above random (indicates potential for improvement)
MLP does not capture time dependencies as well as LSTM
Could be enhanced with technical indicators 
