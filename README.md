# Advanced Trading Analysis System

## 1. Overview

This project is a sophisticated, command-line-based trading analysis system designed to generate daily trading plans for financial instruments. It fuses traditional technical analysis with market profile techniques and uses a machine learning model to generate predictive insights.

The system is built on a modular architecture, allowing for easy extension and maintenance. Its core purpose is to provide traders with a data-driven, systematic approach to analyzing market conditions and formulating a daily trading strategy.

### Key Concepts
- **Market Profile:** A charting technique that displays price, volume, and time information in a statistical distribution. It helps traders identify key levels of support and resistance, such as the **Point of Control (POC)** and the **Value Area (VA)**.
- **Fusion Analysis:** The system's signals are not based on a single methodology. It combines market profile structure, classical technical indicators (RSI, MACD, etc.), support/resistance levels, and ML predictions into a single, weighted score.
- **Machine Learning Predictions:** The system uses an **XGBoost** model to predict the probability of key market events, such as the likelihood of the price breaking above the day's initial balance high (IBH) or below the initial balance low (IBL).

---

## 2. Setup and Installation

Follow these steps to set up and run the application.

### Prerequisites
- Python 3.10 or higher
- `pip` for package management

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install Dependencies:**
    All required Python packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    Launch the main interactive interface by running the `main.py` script:
    ```bash
    python3 main.py
    ```
    This will start the system and display the main menu.

---

## 3. How to Use the System

The application is operated through a simple command-line menu. The menu options are context-aware and will differ depending on whether the market is currently open or closed.

### Main Menu Options

#### 1. Generate Next Day Trading Plan
This is the primary function of the system. It performs a comprehensive analysis to prepare a trading plan for the upcoming session. The process includes:
-   Fetching the latest market data (daily, 4h, 30m, 5m).
-   Calculating a full suite of technical indicators.
-   Generating market profiles for the last 100 days.
-   Calculating historical statistics based on market profile opening types.
-   Identifying key support and resistance zones.
-   Running XGBoost models to predict the session's likely behavior.
-   Generating a final trading signal (LONG, SHORT, or NEUTRAL) with a confidence score.
-   Saving a detailed **HTML report** in the `/reports` directory.

#### 2. View Historical Profile Statistics
This option provides a statistical breakdown of how the market has behaved following different types of openings (e.g., "Open Drive Up," "Open Test Drive"). The output shows the historical probability of the price breaking the initial balance high/low and the average risk/reward ratio for each scenario.

#### 3. Train/Update ML Models
This allows you to manually trigger the training process for the XGBoost models. The system will:
-   Download 200 days of historical data.
-   Generate market profiles and technical features.
-   Train the models for three targets: `target_broke_ibh`, `target_broke_ibl`, and `target_next_day_direction`.
-   Save the trained models to the `/models` directory.

**Note:** The system will automatically run this training process if it cannot find the required model files when generating a trading plan.

#### 4. Configure Notifications
You can configure the system to send high-confidence trading signals directly to a Discord channel.
-   Follow the in-app instructions to create a **Discord Webhook URL**.
-   Enter the URL when prompted. The system will test it and, if successful, will send alerts for signals with "HIGH" confidence.

#### 5. View Cached Data Summary
The system uses a `sqlite3` database (`data_cache/market_data.db`) to cache market data, which speeds up subsequent runs. This option displays a summary of the cached data for the current ticker, including the timeframes, date ranges, and total number of rows. It also provides an option to download more historical data.

#### 6. Set New Ticker
Allows you to change the financial instrument being analyzed.
-   A list of pre-configured tickers from `src/config.py` is displayed.
-   You can enter any valid ticker symbol recognized by Yahoo Finance (e.g., `AAPL`, `GC=F`, `BTC-USD`).
-   The system will reinitialize all its analytical components for the new ticker.

**Note:** As of now, the system analyzes one ticker at a time. Multi-ticker analysis is a potential future enhancement.

### Live Market Menu (Only when market is open)

#### Start Live Scanner
This launches a real-time scanner for the current ticker. It fetches data at a user-defined interval (e.g., every 60 seconds) and provides a live-updating view of key technical indicators and a simple bullish/bearish signal score. This is useful for monitoring intraday price action.

---

## 4. Understanding the Output

### The Trading Signal
The final output of a trading plan is a signal summary, which looks like this:

```
============================================================
   TRADING SIGNAL SUMMARY
============================================================
🟢 Signal: LONG
📊 Score: 66.2/100
🎯 Confidence: MEDIUM

📋 Key Evidence:
  • Bullish opening type: Open Drive Up
  • Price above Value Area High
  • High probability (100.0%) of IBH break
  • RSI overbought (80.4)
  • Strong uptrend (ADX: 28.0)

🎯 Component Scores:
  • Market Profile: 95.0
  • Technical: 65.0
  • Sr Levels: 20.0
  • Ml Prediction: 70.0
```
- **Signal:** The final trading bias (LONG, SHORT, or NEUTRAL).
- **Score:** A numerical score from 0-100 indicating the strength of the signal.
- **Confidence:** A qualitative assessment (LOW, MEDIUM, HIGH) of the signal's reliability.
- **Key Evidence:** The top five factors that contributed to the signal.
- **Component Scores:** A breakdown of the score from each analytical module.

### The HTML Report
For a full, in-depth analysis, open the generated HTML file. It contains:
-   A summary of the trading plan.
-   An interactive candlestick chart with technical indicators and S/R levels.
-   Detailed Market Profile statistics.
-   The full ML prediction probabilities.
-   A table of historical statistics.
