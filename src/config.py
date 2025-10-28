# Configuration module
# src/config.py
"""Enhanced Configuration Module with Comprehensive Instrument Settings"""

import os
from pathlib import Path
from datetime import time

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_CACHE_DIR = BASE_DIR / "data_cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
for dir_path in [DATA_CACHE_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Database configuration
DB_PATH = DATA_CACHE_DIR / "market_data.db"

# Notification settings
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1422203042270548089/SU8WVNF3XJrdn_uXg9SlXLVD8g0HxFeum0lPyOth93JVmz8f1bUgpk3qSRwMXCr-WoXn"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ==========================================
# COMPREHENSIVE INSTRUMENT SETTINGS
# ==========================================

INSTRUMENT_SETTINGS = {
    # ========== E-MINI FUTURES ==========
    "ES=F": {
        "name": "E-mini S&P 500 Futures",
        "tick_size": 0.25,
        "point_value": 50,  # $50 per point
        "rth_start": "08:30",
        "rth_end": "15:15",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 13200,  # Approximate initial margin
        "description": "E-mini S&P 500 Futures",
        "category": "Index Futures"
    },
    "NQ=F": {
        "name": "E-mini Nasdaq 100 Futures",
        "tick_size": 0.25,
        "point_value": 20,  # $20 per point
        "rth_start": "08:30",
        "rth_end": "15:15",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 16500,
        "description": "E-mini Nasdaq 100 Futures",
        "category": "Index Futures"
    },
    "YM=F": {
        "name": "E-mini Dow Jones Futures",
        "tick_size": 1.0,
        "point_value": 5,  # $5 per point
        "rth_start": "08:30",
        "rth_end": "15:15",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 8800,
        "description": "E-mini Dow Jones Futures",
        "category": "Index Futures"
    },
    "RTY=F": {
        "name": "E-mini Russell 2000 Futures",
        "tick_size": 0.10,
        "point_value": 50,  # $50 per point
        "rth_start": "08:30",
        "rth_end": "15:15",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 7150,
        "description": "E-mini Russell 2000 Futures",
        "category": "Index Futures"
    },
    
    # ========== MICRO FUTURES ==========
    "MES=F": {
        "name": "Micro E-mini S&P 500 Futures",
        "tick_size": 0.25,
        "point_value": 5,  # $5 per point (1/10th of ES)
        "rth_start": "08:30",
        "rth_end": "15:15",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 1320,
        "description": "Micro E-mini S&P 500 Futures",
        "category": "Micro Futures"
    },
    "MNQ=F": {
        "name": "Micro E-mini Nasdaq 100 Futures",
        "tick_size": 0.25,
        "point_value": 2,  # $2 per point (1/10th of NQ)
        "rth_start": "08:30",
        "rth_end": "15:15",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 1650,
        "description": "Micro E-mini Nasdaq 100 Futures",
        "category": "Micro Futures"
    },
    "MYM=F": {
        "name": "Micro E-mini Dow Jones Futures",
        "tick_size": 1.0,
        "point_value": 0.50,  # $0.50 per point (1/10th of YM)
        "rth_start": "08:30",
        "rth_end": "15:15",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 880,
        "description": "Micro E-mini Dow Jones Futures",
        "category": "Micro Futures"
    },
    "M2K=F": {
        "name": "Micro E-mini Russell 2000 Futures",
        "tick_size": 0.10,
        "point_value": 5,  # $5 per point (1/10th of RTY)
        "rth_start": "08:30",
        "rth_end": "15:15",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 715,
        "description": "Micro E-mini Russell 2000 Futures",
        "category": "Micro Futures"
    },
    
    # ========== COMMODITY FUTURES ==========
    "GC=F": {
        "name": "Gold Futures",
        "tick_size": 0.10,
        "point_value": 100,  # $100 per point
        "rth_start": "07:20",
        "rth_end": "12:30",
        "timezone": "US/Central",
        "exchange": "COMEX",
        "margin_required": 10000,
        "description": "Gold Futures",
        "category": "Metals"
    },
    "SI=F": {
        "name": "Silver Futures",
        "tick_size": 0.005,
        "point_value": 5000,  # $5000 per point
        "rth_start": "07:25",
        "rth_end": "12:25",
        "timezone": "US/Central",
        "exchange": "COMEX",
        "margin_required": 9000,
        "description": "Silver Futures",
        "category": "Metals"
    },
    "CL=F": {
        "name": "Crude Oil WTI Futures",
        "tick_size": 0.01,
        "point_value": 1000,  # $1000 per point
        "rth_start": "08:00",
        "rth_end": "13:30",
        "timezone": "US/Central",
        "exchange": "NYMEX",
        "margin_required": 6000,
        "description": "Light Sweet Crude Oil",
        "category": "Energy"
    },
    "NG=F": {
        "name": "Natural Gas Futures",
        "tick_size": 0.001,
        "point_value": 10000,  # $10000 per point
        "rth_start": "08:00",
        "rth_end": "13:30",
        "timezone": "US/Central",
        "exchange": "NYMEX",
        "margin_required": 3500,
        "description": "Natural Gas",
        "category": "Energy"
    },
    
    # ========== CURRENCY FUTURES ==========
    "6E=F": {
        "name": "Euro FX Futures",
        "tick_size": 0.00005,
        "point_value": 125000,  # €125,000 per contract
        "rth_start": "07:20",
        "rth_end": "14:00",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 2500,
        "description": "Euro/USD Futures",
        "category": "Currency"
    },
    "6B=F": {
        "name": "British Pound Futures",
        "tick_size": 0.0001,
        "point_value": 62500,  # £62,500 per contract
        "rth_start": "07:20",
        "rth_end": "14:00",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 3000,
        "description": "GBP/USD Futures",
        "category": "Currency"
    },
    "6J=F": {
        "name": "Japanese Yen Futures",
        "tick_size": 0.0000005,
        "point_value": 12500000,  # ¥12,500,000 per contract
        "rth_start": "07:20",
        "rth_end": "14:00",
        "timezone": "US/Central",
        "exchange": "CME",
        "margin_required": 3500,
        "description": "USD/JPY Futures",
        "category": "Currency"
    },
    
    # ========== AGRICULTURAL FUTURES ==========
    "ZC=F": {
        "name": "Corn Futures",
        "tick_size": 0.25,
        "point_value": 50,  # 5000 bushels, $50 per point
        "rth_start": "08:30",
        "rth_end": "13:20",
        "timezone": "US/Central",
        "exchange": "CBOT",
        "margin_required": 2500,
        "description": "Corn Futures",
        "category": "Agriculture"
    },
    "ZS=F": {
        "name": "Soybean Futures",
        "tick_size": 0.25,
        "point_value": 50,  # 5000 bushels, $50 per point
        "rth_start": "08:30",
        "rth_end": "13:20",
        "timezone": "US/Central",
        "exchange": "CBOT",
        "margin_required": 4000,
        "description": "Soybean Futures",
        "category": "Agriculture"
    },
    "ZW=F": {
        "name": "Wheat Futures",
        "tick_size": 0.25,
        "point_value": 50,  # 5000 bushels, $50 per point
        "rth_start": "08:30",
        "rth_end": "13:20",
        "timezone": "US/Central",
        "exchange": "CBOT",
        "margin_required": 3000,
        "description": "Wheat Futures",
        "category": "Agriculture"
    },
    
    # ========== TREASURY FUTURES ==========
    "ZB=F": {
        "name": "30-Year Treasury Bond Futures",
        "tick_size": 0.03125,  # 1/32
        "point_value": 1000,
        "rth_start": "07:20",
        "rth_end": "14:00",
        "timezone": "US/Central",
        "exchange": "CBOT",
        "margin_required": 3500,
        "description": "30-Year T-Bond Futures",
        "category": "Treasuries"
    },
    "ZN=F": {
        "name": "10-Year Treasury Note Futures",
        "tick_size": 0.015625,  # 1/64
        "point_value": 1000,
        "rth_start": "07:20",
        "rth_end": "14:00",
        "timezone": "US/Central",
        "exchange": "CBOT",
        "margin_required": 2000,
        "description": "10-Year T-Note Futures",
        "category": "Treasuries"
    },
    "ZF=F": {
        "name": "5-Year Treasury Note Futures",
        "tick_size": 0.0078125,  # 1/128
        "point_value": 1000,
        "rth_start": "07:20",
        "rth_end": "14:00",
        "timezone": "US/Central",
        "exchange": "CBOT",
        "margin_required": 1500,
        "description": "5-Year T-Note Futures",
        "category": "Treasuries"
    },
    
    # ========== VIX FUTURES ==========
    "VX=F": {
        "name": "CBOE Volatility Index Futures",
        "tick_size": 0.05,
        "point_value": 1000,
        "rth_start": "08:30",
        "rth_end": "15:15",
        "timezone": "US/Central",
        "exchange": "CFE",
        "margin_required": 8000,
        "description": "VIX Futures",
        "category": "Volatility"
    },
    
    # ========== MAJOR ETFs ==========
    "SPY": {
        "name": "SPDR S&P 500 ETF",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "S&P 500 ETF",
        "category": "ETF"
    },
    "QQQ": {
        "name": "Invesco QQQ Trust",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NASDAQ",
        "margin_required": None,
        "description": "Nasdaq 100 ETF",
        "category": "ETF"
    },
    "IWM": {
        "name": "iShares Russell 2000 ETF",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "Russell 2000 ETF",
        "category": "ETF"
    },
    "DIA": {
        "name": "SPDR Dow Jones Industrial Average ETF",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "Dow Jones ETF",
        "category": "ETF"
    },
    "GLD": {
        "name": "SPDR Gold Shares",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "Gold ETF",
        "category": "ETF"
    },
    "SLV": {
        "name": "iShares Silver Trust",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "Silver ETF",
        "category": "ETF"
    },
    "TLT": {
        "name": "iShares 20+ Year Treasury Bond ETF",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NASDAQ",
        "margin_required": None,
        "description": "Long-Term Treasury ETF",
        "category": "ETF"
    },
    "XLF": {
        "name": "Financial Select Sector SPDR Fund",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "Financial Sector ETF",
        "category": "ETF"
    },
    "XLE": {
        "name": "Energy Select Sector SPDR Fund",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "Energy Sector ETF",
        "category": "ETF"
    },
    "XLK": {
        "name": "Technology Select Sector SPDR Fund",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "Technology Sector ETF",
        "category": "ETF"
    },
    "VXX": {
        "name": "iPath Series B S&P 500 VIX Short-Term Futures ETN",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "VIX Short-Term ETN",
        "category": "ETF"
    },
    "UVXY": {
        "name": "ProShares Ultra VIX Short-Term Futures ETF",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "2x VIX ETF",
        "category": "ETF"
    },
    
    # ========== MAJOR STOCKS ==========
    "AAPL": {
        "name": "Apple Inc.",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NASDAQ",
        "margin_required": None,
        "description": "Apple Inc.",
        "category": "Stock"
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NASDAQ",
        "margin_required": None,
        "description": "Microsoft Corp.",
        "category": "Stock"
    },
    "GOOGL": {
        "name": "Alphabet Inc. Class A",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NASDAQ",
        "margin_required": None,
        "description": "Google Class A",
        "category": "Stock"
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NASDAQ",
        "margin_required": None,
        "description": "Amazon",
        "category": "Stock"
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NASDAQ",
        "margin_required": None,
        "description": "Tesla",
        "category": "Stock"
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NASDAQ",
        "margin_required": None,
        "description": "NVIDIA",
        "category": "Stock"
    },
    "META": {
        "name": "Meta Platforms Inc.",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NASDAQ",
        "margin_required": None,
        "description": "Meta (Facebook)",
        "category": "Stock"
    },
    "JPM": {
        "name": "JPMorgan Chase & Co.",
        "tick_size": 0.01,
        "point_value": 1,
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "US/Central",
        "exchange": "NYSE",
        "margin_required": None,
        "description": "JPMorgan",
        "category": "Stock"
    },
}

# Default ticker
DEFAULT_TICKER = "ES=F"

# ==========================================
# COMPREHENSIVE INDICATOR PARAMETERS
# ==========================================

INDICATOR_PARAMS = {
    # Moving Averages
    "ema_periods": [9, 21, 34, 50, 89, 200],
    "sma_periods": [20, 50, 100, 200],
    "wma_periods": [10, 20],
    "vwma_periods": [20],
    
    # Momentum Indicators
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "stoch_k_period": 14,
    "stoch_d_period": 3,
    "stoch_overbought": 80,
    "stoch_oversold": 20,
    "momentum_period": 10,
    "roc_period": 10,
    "williams_r_period": 14,
    "cci_period": 20,
    "mfi_period": 14,
    "mfi_overbought": 80,
    "mfi_oversold": 20,
    
    # Trend Indicators
    "adx_period": 14,
    "adx_threshold": 25,  # Above this = trending
    "aroon_period": 25,
    "psar_af": 0.02,
    "psar_max_af": 0.2,
    "supertrend_period": 10,
    "supertrend_multiplier": 3,
    
    # Volatility Indicators
    "atr_period": 14,
    "bb_period": 20,
    "bb_std": 2,
    "keltner_period": 20,
    "keltner_multiplier": 2,
    "donchian_period": 20,
    
    # Volume Indicators
    "volume_ma_period": 20,
    "obv_signal_period": 9,
    "cmf_period": 21,
    "vwap_periods": ["Session", "Weekly", "Monthly"],
    
    # MACD Parameters
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    
    # Linear Regression
    "lr_period": 20,
    "lr_deviation": 2,
    
    # Fibonacci Levels
    "fib_levels": [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0, 2.618],
    
    # Market Profile Specific
    "tpo_period": 30,  # minutes per TPO
    "value_area_percent": 0.70,  # 70% of volume
    "ib_periods": 2,  # First 2 periods for IB (first hour)
}

# ==========================================
# MACHINE LEARNING PARAMETERS
# ==========================================

ML_PARAMS = {
    # Data Preparation
    "lookback_days": 100,
    "min_samples": 50,
    "test_size": 0.2,
    "validation_size": 0.1,
    
    # Random Forest Parameters
    "rf_n_estimators": 100,
    "rf_max_depth": 10,
    "rf_min_samples_split": 5,
    "rf_min_samples_leaf": 2,
    "rf_max_features": "sqrt",
    
    # XGBoost Parameters (if using)
    "xgb_n_estimators": 100,
    "xgb_max_depth": 6,
    "xgb_learning_rate": 0.1,
    "xgb_subsample": 0.8,
    
    # Feature Engineering
    "feature_selection_threshold": 0.01,  # Min feature importance
    "correlation_threshold": 0.95,  # Remove highly correlated features
    
    # Model Settings
    "random_state": 42,
    "n_jobs": -1,  # Use all CPU cores
    "verbose": 0,
    
    # Cross Validation
    "cv_folds": 5,
    "cv_scoring": "accuracy",
}

# ==========================================
# SIGNAL GENERATION PARAMETERS
# ==========================================

SIGNAL_PARAMS = {
    # Confluence Weights (must sum to 1.0)
    "weights": {
        "market_profile": 0.30,
        "technical": 0.25,
        "sr_levels": 0.20,
        "ml_prediction": 0.25
    },
    
    # Signal Thresholds
    "thresholds": {
        "strong_long": 70,
        "long": 55,
        "neutral_high": 54,
        "neutral_low": 46,
        "short": 45,
        "strong_short": 30
    },
    
    # Confidence Levels
    "confidence": {
        "high": 0.70,
        "medium": 0.50,
        "low": 0.30
    },
    
    # Risk Management
    "stop_loss_atr_multiplier": 2.0,
    "take_profit_atr_multiplier": 3.0,
    "position_size_risk_percent": 1.0,  # Risk 1% per trade
    
    # Alert Settings
    "alert_on_strong_signals": True,
    "alert_confidence_threshold": 0.70,
    "cooldown_minutes": 15,  # Min time between alerts
}

# ==========================================
# MARKET PROFILE PARAMETERS
# ==========================================

MARKET_PROFILE_PARAMS = {
    # TPO Settings
    "tpo_size_multiplier": 2,  # TPO size = tick_size * multiplier
    "min_tpos_for_poc": 3,  # Minimum TPOs to consider as POC
    
    # Value Area Calculation
    "value_area_volume_percent": 70,  # Standard 70%
    "use_volume_profile": False,  # Use TPO count if False
    
    # Opening Type Classification
    "opening_range_percent": 0.1,  # 10% of prior range for "in range"
    "opening_drive_threshold": 0.5,  # 50% of prior range for "drive"
    
    # Day Type Classification  
    "trend_day_extension": 1.5,  # IB extension for trend day
    "normal_day_range": 1.15,  # Max range for normal day
    
    # Composite Profile
    "composite_days": 20,  # Days for composite profile
    "hvn_threshold": 1.5,  # High Volume Node threshold
    "lvn_threshold": 0.5,  # Low Volume Node threshold
}

# ==========================================
# SUPPORT/RESISTANCE PARAMETERS
# ==========================================

SR_PARAMS = {
    # Swing Detection
    "swing_order": 5,  # Points on each side for swing
    "min_swing_percent": 0.5,  # Min % move for valid swing
    
    # Clustering
    "cluster_threshold_ticks": 10,  # Group levels within N ticks
    "min_touches": 2,  # Min touches to confirm level
    
    # Level Strength
    "recency_weight": 1.5,  # Weight recent levels higher
    "volume_weight": 2.0,  # Weight high volume levels
    
    # Psychological Levels
    "psych_round_numbers": [10, 25, 50, 100],  # Round number intervals
    "psych_range_percent": 0.1,  # 10% range for psych levels
}

# ==========================================
# STATISTICS PARAMETERS
# ==========================================

STATISTICS_PARAMS = {
    # Minimum samples for reliable statistics
    "min_samples_per_type": 10,
    
    # Events to track (can be customized)
    "track_events": [
        "IBH", "IBL", "IBH or IBL", "IBH & IBL (Neutral)",
        "1.5X IBH", "1.5X IBL", "2X IBH", "2X IBL",
        "pVAH", "pVAL", "pPOC", "pCL (Gap)",
        "pHI (Range Gap)", "pLO", "pMID", "1/2 Gap",
        "Inside Day", "Outside Day"
    ],
    
    # Probability thresholds for highlighting
    "high_prob_threshold": 70,  # >= 70% is high probability
    "medium_prob_threshold": 50,  # >= 50% is medium probability
}

# ==========================================
# REPORTING PARAMETERS
# ==========================================

REPORTING_PARAMS = {
    # Chart Settings
    "chart_height": 800,
    "chart_width": 1400,
    "use_dark_theme": False,
    
    # Report Sections
    "include_sections": {
        "signal": True,
        "key_levels": True,
        "market_profile": True,
        "technical_analysis": True,
        "statistics": True,
        "sr_zones": True,
        "ml_predictions": True,
        "risk_management": True
    },
    
    # Data Display
    "max_sr_zones": 10,
    "max_evidence_items": 10,
    "decimal_places": 2,
    
    # File Settings
    "report_format": "html",  # html, pdf (requires additional setup)
    "auto_open": True,  # Auto open report in browser
    "keep_reports": 30,  # Days to keep old reports
}

# ==========================================
# LIVE SCANNER PARAMETERS
# ==========================================

SCANNER_PARAMS = {
    # Refresh Settings
    "default_refresh_seconds": 60,
    "min_refresh_seconds": 10,
    "max_refresh_seconds": 300,
    
    # Alert Conditions
    "scan_conditions": {
        "ib_break": True,
        "new_high_low": True,
        "poc_touch": True,
        "signal_change": True,
        "volume_spike": True
    },
    
    # Volume Spike Detection
    "volume_spike_multiplier": 2.0,  # 2x average volume
    "volume_lookback": 20,  # Periods for average
}

# ==========================================
# SYSTEM PARAMETERS
# ==========================================

SYSTEM_PARAMS = {
    # Logging
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_to_file": True,
    "log_file": "trading_system.log",
    
    # Performance
    "max_workers": 4,  # For parallel processing
    "cache_expire_hours": 24,
    "memory_limit_mb": 2048,
    
    # Data Management
    "max_cache_size_mb": 500,
    "cleanup_old_data_days": 30,
    "backup_database": True,
    
    # Error Handling
    "max_retries": 3,
    "retry_delay_seconds": 5,
    "fallback_to_cache": True,
}

# ==========================================
# API KEYS AND CREDENTIALS
# ==========================================

API_KEYS = {
    # Data Providers (if using premium sources)
    "polygon_api_key": os.getenv("POLYGON_API_KEY", ""),
    "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
    "iex_cloud_api_key": os.getenv("IEX_CLOUD_API_KEY", ""),
    
    # Broker APIs (for live trading - future enhancement)
    "interactive_brokers": {
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 1
    },
    
    # Cloud Storage (for backups)
    "aws_access_key": os.getenv("AWS_ACCESS_KEY", ""),
    "aws_secret_key": os.getenv("AWS_SECRET_KEY", ""),
    "aws_bucket": os.getenv("AWS_BUCKET", ""),
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_instrument_by_category(category: str) -> list:
    """Get all instruments in a specific category"""
    return [
        ticker for ticker, settings in INSTRUMENT_SETTINGS.items()
        if settings.get('category') == category
    ]

def get_futures_instruments() -> list:
    """Get all futures instruments"""
    futures_categories = ['Index Futures', 'Micro Futures', 'Metals', 'Energy', 
                         'Currency', 'Agriculture', 'Treasuries', 'Volatility']
    instruments = []
    for category in futures_categories:
        instruments.extend(get_instrument_by_category(category))
    return instruments

def get_equity_instruments() -> list:
    """Get all equity instruments (stocks and ETFs)"""
    return get_instrument_by_category('Stock') + get_instrument_by_category('ETF')

def validate_config() -> bool:
    """Validate configuration settings"""
    errors = []
    
    # Check that weights sum to 1.0
    weight_sum = sum(SIGNAL_PARAMS['weights'].values())
    if abs(weight_sum - 1.0) > 0.001:
        errors.append(f"Signal weights sum to {weight_sum}, should be 1.0")
    
    # Check that all instruments have required fields
    required_fields = ['tick_size', 'rth_start', 'rth_end', 'timezone']
    for ticker, settings in INSTRUMENT_SETTINGS.items():
        for field in required_fields:
            if field not in settings:
                errors.append(f"{ticker} missing required field: {field}")
    
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

# Validate on import
if not validate_config():
    print("Warning: Configuration validation failed. Please check settings.")