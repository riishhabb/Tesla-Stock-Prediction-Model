<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Agent Performance Summary</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 40px;
            background-color: #f9f9f9;
        }
        h1, h2 {
            color: #333366;
        }
        ul {
            margin-left: 20px;
        }
        .note {
            background-color: #fff3cd;
            padding: 10px;
            border-left: 5px solid #ffc107;
            margin-top: 20px;
        }
    </style>
</head>
<body>

    <h1>Agent Performance Summary</h1>

    <h2>Final Account Value</h2>
    <ul>
        <li>Initial Capital: $10,000</li>
        <li>Transaction Fee: 1% on every Buy/Sell</li>
        <li>Period Simulated: Based on data from <code>tesla_stock_data.csv</code></li>
    </ul>

    <h2>Trading Strategy Logic</h2>

    <h3>Buy Conditions:</h3>
    <ul>
        <li>Model predicts the price will go up (next day)</li>
        <li>Current price is above the 5-day Moving Average (indicates upward trend)</li>
        <li>Invest 50% of current capital into Tesla shares</li>
    </ul>

    <h3>Sell Conditions:</h3>
    <ul>
        <li>Model predicts the price will fall, OR</li>
        <li>Current price drops &gt;3% below average buy price</li>
        <li>Sell all shares to protect against loss</li>
    </ul>

    <h3>Hold:</h3>
    <ul>
        <li>No trade action if neither Buy nor Sell conditions are met</li>
    </ul>

    <h2>ML Model Summary</h2>
    <ul>
        <li>Model Used: Multi-Layer Perceptron (MLP)</li>
        <li>Inputs: Last 10 days of Close and Volume (windowed input)</li>
        <li>Target: Binary prediction – whether price will rise the next day</li>
        <li>Test Accuracy: ~51%</li>
    </ul>

    <h2>Analysis of Results</h2>

    <h3>Strengths:</h3>
    <ul>
        <li>MLP captured some patterns in Tesla’s historical movements</li>
        <li>Strategy limited risk by avoiding impulsive selling</li>
        <li>Prevented major losses through a 3% stop-loss rule</li>
    </ul>

    <h3>Limitations:</h3>
    <ul>
        <li>Accuracy just above random (indicates potential for improvement)</li>
        <li>MLP does not capture time dependencies as well as LSTM</li>
        <li>Could be enhanced with technical indicators</li>
    </ul>

    <div class="note">
        <strong>Note:</strong> This agent code must be run using a Python environment (e.g., Anaconda, Jupyter Notebook, or standard Python interpreter).
    </div>

</body>
</html>
