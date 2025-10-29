# Advanced Trading Analysis System

## 1. Overview

This project is a sophisticated, command-line-based trading analysis system designed to generate daily trading plans for financial instruments. It supports both single-ticker deep analysis and multi-ticker watchlist scanning. The system fuses traditional technical analysis with market profile techniques and uses a machine learning model to generate predictive insights.

### Key Features
- **Multi-Ticker Dashboard:** Analyze an entire watchlist of tickers in parallel and view a summary dashboard of the top trading signals.
- **Continuous Scanner:** Run a background scanner on a watchlist to get real-time alerts for high-confidence setups.
- **In-Depth Single Ticker Analysis:** Generate a detailed HTML report for a single ticker, including interactive charts, market profile analysis, and ML predictions.
- **Machine Learning Integration:** Uses an **XGBoost** model to predict key market events, such as the probability of breaking the initial balance high/low.

---

## 2. Setup and Installation

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
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    python3 main.py
    ```

---

## 3. How to Use the System

The application is operated through a command-line menu.

### Multi-Ticker Analysis

#### 1. Analyze Watchlist
This option allows you to run a one-time analysis on a predefined watchlist from `src/watchlists.py`.
-   The system will analyze all tickers in the list in parallel.
-   Once complete, it will display a summary dashboard showing the signal, score, confidence, and key indicators for each ticker, sorted by signal strength.

#### 2. Start Multi-Ticker Scanner
This launches a continuous, live scanner for a selected watchlist.
-   You will be prompted to choose a watchlist and set a refresh interval (in seconds).
-   The system will repeatedly analyze the watchlist in the background.
-   After each cycle, it will refresh the summary dashboard.
-   If any high-confidence signals are found, it will print a real-time alert to the console and send a notification to Discord (if configured).

### Single-Ticker Analysis
First, set the ticker you want to analyze using the "Set New Ticker" option.

#### 3. Generate Trading Plan for Current Ticker
This performs a deep-dive analysis for the currently selected ticker and generates a detailed HTML report. The process includes:
-   Fetching extensive market data.
-   Generating market profiles and historical statistics.
-   Running ML predictions.
-   Saving a comprehensive **HTML report** in the `/reports` directory with interactive charts.

#### 4. View Historical Profile Statistics
This displays a statistical breakdown of how the market has behaved following different types of openings (e.g., "Open Drive Up") for the current ticker.

#### 5. Train ML Models for Current Ticker
This allows you to manually trigger the training process for the XGBoost models for the current ticker. Note that the system will automatically skip ML predictions for any ticker that does not have a pre-trained model.

### System Options

#### 6. Set New Ticker
Change the instrument for the single-ticker analysis functions.

#### 7. Configure Notifications
Set up Discord webhook integration to receive real-time alerts for high-confidence signals from the multi-ticker scanner.

#### 8. View Cached Data Summary
Display a summary of the locally cached market data for the current ticker.

---

## 4. Understanding the Output

### The Dashboard
The multi-ticker dashboard provides a high-level overview of your watchlist, allowing you to quickly spot the most promising setups.

### The HTML Report
For a deep dive into a single instrument, the HTML report provides a comprehensive view, including:
-   An interactive candlestick chart with indicators and support/resistance levels.
-   Detailed Market Profile statistics and visualizations.
-   Full ML prediction probabilities.
