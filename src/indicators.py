# Technical indicators module
# src/indicators.py
"""Enhanced Technical Indicators Module with ML-Focused Features"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from scipy.signal import argrelextrema

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period (default 14)
    
    Returns:
        Series with ATR values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using exponential moving average
    atr = tr.ewm(span=period, adjust=False).mean()
    
    return atr

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        df: DataFrame with OHLC data
        period: RSI period (default 14)
    
    Returns:
        Series with RSI values
    """
    close = df['Close']
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX) with +DI and -DI
    
    Args:
        df: DataFrame with OHLC data
        period: ADX period (default 14)
    
    Returns:
        Tuple of (ADX, +DI, -DI) Series
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate directional movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Where both are positive, keep only the larger
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm[mask & (plus_dm < minus_dm)] = 0
    minus_dm[mask & (minus_dm < plus_dm)] = 0
    
    # Calculate ATR for ADX
    atr = calculate_atr(df, period)
    
    # Smooth the directional movements
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx, plus_di, minus_di

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        df: DataFrame with OHLC data
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
    
    Returns:
        Dictionary with MACD line, signal line, and histogram
    """
    close = df['Close']
    
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        df: DataFrame with OHLC data
        period: Moving average period (default 20)
        std_dev: Number of standard deviations (default 2)
    
    Returns:
        Dictionary with upper band, middle band (SMA), lower band, and %B
    """
    close = df['Close']
    
    middle_band = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    # Calculate %B (position within bands)
    percent_b = (close - lower_band) / (upper_band - lower_band)
    
    # Calculate band width (volatility indicator)
    band_width = (upper_band - lower_band) / middle_band
    
    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band,
        'percent_b': percent_b,
        'band_width': band_width
    }

def calculate_ema(df: pd.DataFrame, periods: list = [9, 21, 50, 200]) -> Dict[str, pd.Series]:
    """
    Calculate Exponential Moving Averages for multiple periods
    
    Args:
        df: DataFrame with OHLC data
        periods: List of periods for EMAs
    
    Returns:
        Dictionary with EMAs for each period
    """
    close = df['Close']
    emas = {}
    
    for period in periods:
        emas[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
    
    return emas

def calculate_linear_regression(df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
    """
    Calculate Linear Regression Channel
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period for regression
    
    Returns:
        Dictionary with regression line, slope, r-squared, and channels
    """
    close = df['Close']
    
    # Initialize arrays
    lr_line = pd.Series(index=close.index, dtype=float)
    lr_slope = pd.Series(index=close.index, dtype=float)
    lr_r2 = pd.Series(index=close.index, dtype=float)
    lr_upper = pd.Series(index=close.index, dtype=float)
    lr_lower = pd.Series(index=close.index, dtype=float)
    
    for i in range(period, len(close)):
        # Get window data
        y = close.iloc[i-period+1:i+1].values
        x = np.arange(period)
        
        # Calculate linear regression
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        
        # Store values
        lr_line.iloc[i] = intercept + slope * (period - 1)
        lr_slope.iloc[i] = slope
        lr_r2.iloc[i] = r_value ** 2
        
        # Calculate standard error for channels
        y_pred = intercept + slope * x
        residuals = y - y_pred
        std_error = np.std(residuals)
        
        lr_upper.iloc[i] = lr_line.iloc[i] + (2 * std_error)
        lr_lower.iloc[i] = lr_line.iloc[i] - (2 * std_error)
    
    return {
        'lr_line': lr_line,
        'lr_slope': lr_slope,
        'lr_r2': lr_r2,
        'lr_upper': lr_upper,
        'lr_lower': lr_lower
    }

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator
    
    Args:
        df: DataFrame with OHLC data
        k_period: Period for %K line
        d_period: Period for %D line (signal)
    
    Returns:
        Dictionary with %K and %D values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %D (3-period SMA of %K)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return {
        'stoch_k': k_percent,
        'stoch_d': d_percent
    }

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Series with VWAP values
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV)
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Series with OBV values
    """
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['Volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Calculate Price Momentum
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period
    
    Returns:
        Series with momentum values
    """
    return df['Close'].diff(period)

def calculate_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Calculate Rate of Change (ROC)
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period
    
    Returns:
        Series with ROC values (percentage)
    """
    return ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100

def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Williams %R
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period
    
    Returns:
        Series with Williams %R values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    return williams_r

def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI)
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period
    
    Returns:
        Series with CCI values
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    
    cci = (typical_price - sma) / (0.015 * mad)
    
    return cci

def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index (MFI)
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
    
    Returns:
        Series with MFI values
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    # Determine positive and negative money flow
    positive_flow = pd.Series(0, index=df.index, dtype=float)
    negative_flow = pd.Series(0, index=df.index, dtype=float)
    
    # Compare typical prices to determine flow direction
    for i in range(1, len(df)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = raw_money_flow.iloc[i]
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            negative_flow.iloc[i] = raw_money_flow.iloc[i]
    
    # Calculate money flow ratio and MFI
    positive_flow_sum = positive_flow.rolling(window=period).sum()
    negative_flow_sum = negative_flow.rolling(window=period).sum()
    
    # Avoid division by zero
    money_flow_ratio = positive_flow_sum / negative_flow_sum.replace(0, 0.0001)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    return mfi

def calculate_volatility_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate various volatility-based features for ML
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Dictionary with volatility features
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Historical volatility (standard deviation of returns)
    returns = close.pct_change()
    hv_10 = returns.rolling(window=10).std() * np.sqrt(252)  # Annualized
    hv_20 = returns.rolling(window=20).std() * np.sqrt(252)
    
    # Parkinson volatility (using high-low range)
    parkinson = np.sqrt(252 / (4 * np.log(2))) * np.sqrt(
        ((np.log(high / low)) ** 2).rolling(window=20).mean()
    )
    
    # Garman-Klass volatility
    gk = np.sqrt(
        252 * (
            0.5 * ((np.log(high / low)) ** 2).rolling(window=20).mean() -
            (2 * np.log(2) - 1) * ((np.log(close / close.shift(1))) ** 2).rolling(window=20).mean()
        )
    )
    
    return {
        'hv_10': hv_10,
        'hv_20': hv_20,
        'parkinson_vol': parkinson,
        'garman_klass_vol': gk
    }

def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate Pivot Points (Traditional)
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Dictionary with pivot levels
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Traditional pivot point
    pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    
    # Support and resistance levels
    r1 = 2 * pivot - low.shift(1)
    s1 = 2 * pivot - high.shift(1)
    r2 = pivot + (high.shift(1) - low.shift(1))
    s2 = pivot - (high.shift(1) - low.shift(1))
    r3 = high.shift(1) + 2 * (pivot - low.shift(1))
    s3 = low.shift(1) - 2 * (high.shift(1) - pivot)
    
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }

def calculate_price_patterns(df: pd.DataFrame, window: int = 20) -> Dict[str, pd.Series]:
    """
    Detect basic price patterns for ML features
    
    Args:
        df: DataFrame with OHLC data
        window: Lookback window for pattern detection
    
    Returns:
        Dictionary with pattern indicators
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Higher highs and higher lows (uptrend)
    hh = (high > high.shift(1)).rolling(window=window).sum() / window
    hl = (low > low.shift(1)).rolling(window=window).sum() / window
    
    # Lower highs and lower lows (downtrend)
    lh = (high < high.shift(1)).rolling(window=window).sum() / window
    ll = (low < low.shift(1)).rolling(window=window).sum() / window
    
    # Inside bars
    inside_bar = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
    
    # Outside bars
    outside_bar = ((high > high.shift(1)) & (low < low.shift(1))).astype(int)
    
    # Doji detection (small body relative to range)
    body = abs(close - df['Open'])
    range_hl = high - low
    doji = (body / range_hl.replace(0, np.nan) < 0.1).astype(int)
    
    return {
        'higher_highs_ratio': hh,
        'higher_lows_ratio': hl,
        'lower_highs_ratio': lh,
        'lower_lows_ratio': ll,
        'inside_bar': inside_bar,
        'outside_bar': outside_bar,
        'doji': doji
    }

def calculate_fibonacci_targets(df: pd.DataFrame, lookback: int = 252) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement and extension levels for price targets.

    Args:
        df: DataFrame with OHLC data.
        lookback: Number of periods to find the swing high/low.

    Returns:
        Dictionary with Fibonacci levels.
    """
    if len(df) < lookback:
        lookback = len(df)

    recent_data = df.tail(lookback)
    high_price = recent_data['High'].max()
    low_price = recent_data['Low'].min()

    high_idx = recent_data['High'].idxmax()
    low_idx = recent_data['Low'].idxmin()

    swing_range = high_price - low_price
    if swing_range == 0: return {}

    # Primary trend direction for extensions
    if high_idx > low_idx: # Uptrend
        fib_levels = {
            'Retracement 0.382': high_price - swing_range * 0.382,
            'Retracement 0.500': high_price - swing_range * 0.500,
            'Retracement 0.618': high_price - swing_range * 0.618,
            'Extension 1.272': high_price + swing_range * 0.272,
            'Extension 1.618': high_price + swing_range * 0.618,
        }
    else: # Downtrend
        fib_levels = {
            'Retracement 0.382': low_price + swing_range * 0.382,
            'Retracement 0.500': low_price + swing_range * 0.500,
            'Retracement 0.618': low_price + swing_range * 0.618,
            'Extension 1.272': low_price - swing_range * 0.272,
            'Extension 1.618': low_price - swing_range * 0.618,
        }

    return fib_levels

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators and add them to the dataframe
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with all indicators added as columns
    """
    result = df.copy()
    
    # Core indicators
    result['ATR'] = calculate_atr(df)
    result['RSI'] = calculate_rsi(df)
    
    # ADX with +DI and -DI
    adx, plus_di, minus_di = calculate_adx(df)
    result['ADX'] = adx
    result['Plus_DI'] = plus_di
    result['Minus_DI'] = minus_di
    
    # MACD
    macd_dict = calculate_macd(df)
    result['MACD'] = macd_dict['macd']
    result['MACD_Signal'] = macd_dict['signal']
    result['MACD_Histogram'] = macd_dict['histogram']
    
    # Bollinger Bands
    bb_dict = calculate_bollinger_bands(df)
    result['BB_Upper'] = bb_dict['upper']
    result['BB_Middle'] = bb_dict['middle']
    result['BB_Lower'] = bb_dict['lower']
    result['BB_PercentB'] = bb_dict['percent_b']
    result['BB_Width'] = bb_dict['band_width']
    
    # EMAs
    ema_dict = calculate_ema(df, [9, 21, 50, 200])
    for key, value in ema_dict.items():
        result[key] = value
    
    # Linear Regression
    lr_dict = calculate_linear_regression(df)
    result['LR_Line'] = lr_dict['lr_line']
    result['LR_Slope'] = lr_dict['lr_slope']
    result['LR_R2'] = lr_dict['lr_r2']
    result['LR_Upper'] = lr_dict['lr_upper']
    result['LR_Lower'] = lr_dict['lr_lower']
    
    # Stochastic
    stoch_dict = calculate_stochastic(df)
    result['Stoch_K'] = stoch_dict['stoch_k']
    result['Stoch_D'] = stoch_dict['stoch_d']
    
    # Volume indicators
    result['VWAP'] = calculate_vwap(df)
    result['OBV'] = calculate_obv(df)
    result['MFI'] = calculate_mfi(df)
    
    # Momentum indicators
    result['Momentum'] = calculate_momentum(df)
    result['ROC'] = calculate_roc(df)
    result['Williams_R'] = calculate_williams_r(df)
    result['CCI'] = calculate_cci(df)
    
    # Volatility features
    vol_dict = calculate_volatility_features(df)
    for key, value in vol_dict.items():
        result[f'Vol_{key}'] = value
    
    # Pivot points
    pivot_dict = calculate_pivot_points(df)
    for key, value in pivot_dict.items():
        result[f'Pivot_{key}'] = value
    
    # Price patterns
    pattern_dict = calculate_price_patterns(df)
    for key, value in pattern_dict.items():
        result[f'Pattern_{key}'] = value
    
    return result

def get_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract key features specifically for machine learning
    
    Args:
        df: DataFrame with all indicators
    
    Returns:
        DataFrame with selected ML features
    """
    ml_features = [
        'RSI', 'ADX', 'MACD_Histogram', 'BB_PercentB', 'BB_Width',
        'LR_Slope', 'LR_R2', 'Stoch_K', 'MFI', 'ROC', 'CCI',
        'Vol_hv_20', 'Pattern_higher_highs_ratio', 'Pattern_higher_lows_ratio'
    ]
    
    # Add price position relative to EMAs
    for ema_period in [9, 21, 50, 200]:
        ema_col = f'EMA_{ema_period}'
        if ema_col in df.columns:
            df[f'Price_vs_{ema_col}'] = (df['Close'] - df[ema_col]) / df[ema_col]
            ml_features.append(f'Price_vs_{ema_col}')
    
    # Add cross indicators
    if 'EMA_9' in df.columns and 'EMA_21' in df.columns:
        df['EMA_9_21_Cross'] = (df['EMA_9'] > df['EMA_21']).astype(int)
        ml_features.append('EMA_9_21_Cross')
    
    if 'EMA_50' in df.columns and 'EMA_200' in df.columns:
        df['EMA_50_200_Cross'] = (df['EMA_50'] > df['EMA_200']).astype(int)
        ml_features.append('EMA_50_200_Cross')
    
    return df[ml_features]