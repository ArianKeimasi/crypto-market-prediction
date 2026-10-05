# Cryptocurrency Market Analysis & Forecasting Application

**Python desktop application for analysing Bitcoin and Ethereum market data using technical indicators, linear regression and rule-based trading signals.**

This project was originally developed as an A-Level Computer Science project and has been refined for my portfolio. It combines live and historical cryptocurrency data with a simple machine-learning forecasting model and an interactive desktop interface.

> **Purpose:** Educational market analysis and exploration. This application is not financial advice and is not intended to be used as an automated trading system.

## Features

- Retrieves live **Bitcoin (BTC)** and **Ethereum (ETH)** prices using the Binance public API
- Downloads historical daily candlestick data
- Processes market data using **pandas**
- Calculates **7-day and 30-day moving averages**
- Calculates the **Relative Strength Index (RSI)**
- Uses **scikit-learn LinearRegression** to explore next-period price forecasting
- Generates rule-based signals: **Strong Buy, Buy, Hold, Sell and Strong Sell**
- Provides a heuristic **signal-strength score**
- Displays market trends and indicators using embedded **Matplotlib** charts
- Provides separate Bitcoin and Ethereum analysis windows through a **Tkinter GUI**
- Includes fallback/error handling when market or prediction data is unavailable

## How It Works

The application follows a simple analysis pipeline:

1. Retrieve historical cryptocurrency data from Binance.
2. Convert the data into a pandas DataFrame.
3. Calculate moving-average and RSI indicators.
4. Use closing price, MA7, MA30 and volume as inputs to a linear-regression model.
5. Produce an exploratory next-price forecast.
6. Combine the forecast direction with RSI information to generate a rule-based market signal.
7. Display the results through the graphical interface.

## Forecasting Model

The project uses `LinearRegression` from scikit-learn as an interpretable baseline model.

The model uses historical:

- Closing price
- 7-day moving average
- 30-day moving average
- Trading volume

to explore short-term price forecasting.

The forecasting component should not be interpreted as evidence that cryptocurrency prices can be reliably predicted. It was designed to demonstrate the application of regression, time-series feature engineering and market-data processing within a complete software project.

## Trading Signal Logic

Forecast information is combined with RSI to produce one of five rule-based signals:

- Strong Buy
- Buy
- Hold
- Sell
- Strong Sell

The displayed **signal strength** is a heuristic score associated with the signal category. It is not a statistical confidence interval or probability of the forecast being correct.

## Technologies

- Python
- pandas
- scikit-learn
- Matplotlib
- Tkinter
- Requests
- Binance public API

## Screenshots

Screenshots of the application interface will be added here.

## Running the Project

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python crypto_predictor.py
```

Tkinter is included with many standard Python installations but may need to be installed separately depending on the operating system and Python distribution.

## Limitations

This project is an educational prototype rather than a production forecasting or trading system.

Important limitations include:

- The forecasting model is a simple linear-regression baseline and cannot represent many nonlinear market relationships.
- The application uses a relatively short historical window.
- The original model does not implement a formal train/test split or time-series cross-validation.
- The trading signals are rule-based and rely on technical indicators that are derived from historical prices.
- The signal-strength values are heuristic rather than statistically estimated probabilities.
- Analysis is limited to Bitcoin and Ethereum.
- Results can depend on the availability and quality of external Binance API data.
- The application does not execute trades.

These limitations mean that apparent agreement between a prediction and subsequent market behaviour should not be interpreted as demonstrated predictive accuracy.

## Possible Future Improvements

Future development could include:

- Time-series train/test evaluation and walk-forward validation
- MAE/RMSE and benchmark-model comparisons
- Additional assets and market features
- More sophisticated feature engineering
- Alternative forecasting models
- Longer historical datasets
- Saving/exporting analysis results
- Improved interface customisation

## Background

The original project involved problem analysis, stakeholder research, requirements gathering, iterative development, testing and evaluation. During development, alternative forecasting approaches were explored before the final application used linear regression as a relatively simple and interpretable baseline.

The public repository focuses on the finished software and its methodology rather than reproducing the full assessed coursework documentation.

---

**Disclaimer:** For educational purposes only. Nothing in this repository constitutes financial or investment advice.
