# Support/Resistance levels module
# src/sr_levels.py
"""Support and Resistance Levels Module with Advanced Analysis"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import argrelextrema
from scipy.cluster.hierarchy import fclusterdata
import warnings
warnings.filterwarnings('ignore')

class SRLevelAnalyzer:
    def __init__(self, ticker: str, tick_size: float):
        """
        Initialize S/R Level Analyzer
        
        Args:
            ticker: Symbol being analyzed
            tick_size: Minimum price movement
        """
        self.ticker = ticker
        self.tick_size = tick_size
        self.cluster_threshold = tick_size * 10  # Cluster levels within 10 ticks
    
    def find_swing_points(self, df: pd.DataFrame, order: int = 5) -> Dict[str, List[Tuple[pd.Timestamp, float]]]:
        """
        Find swing highs and lows using local extrema
        
        Args:
            df: DataFrame with OHLC data
            order: Number of points on each side to compare
        
        Returns:
            Dictionary with swing highs and lows
        """
        if len(df) < order * 2 + 1:
            return {'highs': [], 'lows': []}
        
        # Find local maxima and minima
        high_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
        low_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]
        
        swing_highs = [(df.index[i], df['High'].iloc[i]) for i in high_idx]
        swing_lows = [(df.index[i], df['Low'].iloc[i]) for i in low_idx]
        
        return {
            'highs': swing_highs,
            'lows': swing_lows
        }
    
    def calculate_atr_levels(self, df: pd.DataFrame, atr: pd.Series, 
                           multipliers: List[float] = [1.0, 1.5, 2.0, 3.0]) -> Dict[str, List[float]]:
        """
        Calculate support/resistance levels based on ATR extensions
        
        Args:
            df: DataFrame with OHLC data
            atr: ATR series
            multipliers: ATR multipliers for different levels
        
        Returns:
            Dictionary with ATR-based levels
        """
        if df.empty or atr.empty:
            return {'support': [], 'resistance': []}
        
        current_price = df['Close'].iloc[-1]
        current_atr = atr.iloc[-1]
        
        support_levels = []
        resistance_levels = []
        
        for mult in multipliers:
            extension = current_atr * mult
            support_levels.append(current_price - extension)
            resistance_levels.append(current_price + extension)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement and extension levels
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of periods to find high/low
        
        Returns:
            Dictionary with Fibonacci levels
        """
        if len(df) < lookback:
            lookback = len(df)
        
        recent_data = df.tail(lookback)
        
        # Find significant high and low
        high_price = recent_data['High'].max()
        low_price = recent_data['Low'].min()
        
        # Determine trend direction
        high_idx = recent_data['High'].idxmax()
        low_idx = recent_data['Low'].idxmin()
        
        if high_idx > low_idx:  # Uptrend
            swing_range = high_price - low_price
            base = low_price
            
            fib_levels = {
                'fib_0': high_price,  # 0% (high)
                'fib_236': high_price - swing_range * 0.236,
                'fib_382': high_price - swing_range * 0.382,
                'fib_500': high_price - swing_range * 0.500,
                'fib_618': high_price - swing_range * 0.618,
                'fib_786': high_price - swing_range * 0.786,
                'fib_100': low_price,  # 100% (low)
                # Extensions
                'fib_ext_1272': high_price + swing_range * 0.272,
                'fib_ext_1618': high_price + swing_range * 0.618,
                'fib_ext_2000': high_price + swing_range * 1.000,
            }
        else:  # Downtrend
            swing_range = high_price - low_price
            base = high_price
            
            fib_levels = {
                'fib_0': low_price,  # 0% (low)
                'fib_236': low_price + swing_range * 0.236,
                'fib_382': low_price + swing_range * 0.382,
                'fib_500': low_price + swing_range * 0.500,
                'fib_618': low_price + swing_range * 0.618,
                'fib_786': low_price + swing_range * 0.786,
                'fib_100': high_price,  # 100% (high)
                # Extensions
                'fib_ext_1272': low_price - swing_range * 0.272,
                'fib_ext_1618': low_price - swing_range * 0.618,
                'fib_ext_2000': low_price - swing_range * 1.000,
            }
        
        # Add all-time high/low Fibonacci if in price discovery
        all_time_high = df['High'].max()
        all_time_low = df['Low'].min()
        current_price = df['Close'].iloc[-1]
        
        # Check if near all-time highs/lows
        if current_price > all_time_high * 0.95:  # Within 5% of ATH
            ath_range = all_time_high - all_time_low
            fib_levels['ath_ext_1618'] = all_time_high + ath_range * 0.618
            fib_levels['ath_ext_2618'] = all_time_high + ath_range * 1.618
        
        return fib_levels
    
    def find_volume_levels(self, df: pd.DataFrame, volume_threshold: float = 1.5) -> List[float]:
        """
        Find price levels with high volume (potential S/R)
        
        Args:
            df: DataFrame with OHLCV data
            volume_threshold: Multiplier for average volume
        
        Returns:
            List of high-volume price levels
        """
        if df.empty or 'Volume' not in df.columns:
            return []
        
        avg_volume = df['Volume'].mean()
        high_volume_bars = df[df['Volume'] > avg_volume * volume_threshold]
        
        levels = []
        for idx, row in high_volume_bars.iterrows():
            # Use VWAP of high volume bar as level
            vwap = ((row['High'] + row['Low'] + row['Close']) / 3)
            levels.append(vwap)
        
        return levels
    
    def find_psychological_levels(self, current_price: float, range_percent: float = 0.1) -> List[float]:
        """
        Find psychological price levels (round numbers)
        
        Args:
            current_price: Current market price
            range_percent: Percentage range around current price
        
        Returns:
            List of psychological levels
        """
        levels = []
        
        # Determine the magnitude
        if current_price < 10:
            round_to = 0.25
        elif current_price < 100:
            round_to = 1.0
        elif current_price < 1000:
            round_to = 5.0
        else:
            round_to = 10.0
        
        # Calculate range
        price_range = current_price * range_percent
        min_price = current_price - price_range
        max_price = current_price + price_range
        
        # Find round levels in range
        start = int(min_price / round_to) * round_to
        while start <= max_price:
            if min_price <= start <= max_price:
                levels.append(start)
            start += round_to
        
        return levels
    
    def cluster_levels(self, all_levels: List[float]) -> List[Dict]:
        """
        Cluster nearby S/R levels into zones
        
        Args:
            all_levels: List of all S/R levels from various sources
        
        Returns:
            List of clustered S/R zones with details
        """
        if not all_levels:
            return []
        
        # Remove NaN and sort
        valid_levels = [l for l in all_levels if not np.isnan(l)]
        if not valid_levels:
            return []
        
        # Convert to numpy array for clustering
        levels_array = np.array(valid_levels).reshape(-1, 1)
        
        # Perform hierarchical clustering
        try:
            clusters = fclusterdata(levels_array, self.cluster_threshold, 
                                  criterion='distance', metric='euclidean')
        except:
            # Fallback if clustering fails
            return [{'zone_center': l, 'levels': [l], 'strength': 1} for l in valid_levels]
        
        # Group levels by cluster
        cluster_groups = {}
        for level, cluster_id in zip(valid_levels, clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(level)
        
        # Create zone information
        zones = []
        for cluster_id, levels in cluster_groups.items():
            zone = {
                'zone_center': np.mean(levels),
                'zone_high': max(levels),
                'zone_low': min(levels),
                'levels': levels,
                'strength': len(levels),  # More confluent levels = stronger zone
                'zone_width': max(levels) - min(levels)
            }
            zones.append(zone)
        
        # Sort by zone center
        zones.sort(key=lambda x: x['zone_center'])
        
        return zones
    
    def analyze_all_sr_levels(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Comprehensive S/R analysis using multiple timeframes and methods
        
        Args:
            data: Dictionary with dataframes for different timeframes
        
        Returns:
            Complete S/R analysis with clustered zones
        """
        all_levels = []
        level_sources = {}
        
        # 1. Multi-timeframe swing points
        for timeframe, df in data.items():
            if df.empty:
                continue
                
            # Different order for different timeframes
            order = 10 if timeframe == '1d' else 5
            swings = self.find_swing_points(df, order)
            
            for date, price in swings['highs'][-10:]:  # Last 10 highs
                all_levels.append(price)
                if price not in level_sources:
                    level_sources[price] = []
                level_sources[price].append(f"{timeframe} Swing High")
            
            for date, price in swings['lows'][-10:]:  # Last 10 lows
                all_levels.append(price)
                if price not in level_sources:
                    level_sources[price] = []
                level_sources[price].append(f"{timeframe} Swing Low")
        
        # 2. ATR-based levels (using daily)
        if '1d' in data and not data['1d'].empty:
            from src.indicators import calculate_atr
            daily_df = data['1d']
            atr = calculate_atr(daily_df)
            
            atr_levels = self.calculate_atr_levels(daily_df, atr)
            for level in atr_levels['support']:
                all_levels.append(level)
                if level not in level_sources:
                    level_sources[level] = []
                level_sources[level].append("ATR Support")
            
            for level in atr_levels['resistance']:
                all_levels.append(level)
                if level not in level_sources:
                    level_sources[level] = []
                level_sources[level].append("ATR Resistance")
        
        # 3. Fibonacci levels
        if '1d' in data and not data['1d'].empty:
            fib_levels = self.calculate_fibonacci_levels(data['1d'])
            for name, level in fib_levels.items():
                all_levels.append(level)
                if level not in level_sources:
                    level_sources[level] = []
                level_sources[level].append(f"Fibonacci {name}")
        
        # 4. Volume levels
        if '30m' in data and not data['30m'].empty:
            volume_levels = self.find_volume_levels(data['30m'].tail(100))
            for level in volume_levels:
                all_levels.append(level)
                if level not in level_sources:
                    level_sources[level] = []
                level_sources[level].append("High Volume")
        
        # 5. Psychological levels
        if '1d' in data and not data['1d'].empty:
            current_price = data['1d']['Close'].iloc[-1]
            psych_levels = self.find_psychological_levels(current_price)
            for level in psych_levels:
                all_levels.append(level)
                if level not in level_sources:
                    level_sources[level] = []
                level_sources[level].append("Psychological")
        
        # Cluster the levels
        zones = self.cluster_levels(all_levels)
        
        # Add source information to zones
        for zone in zones:
            zone['confluent_reasons'] = []
            for level in zone['levels']:
                if level in level_sources:
                    zone['confluent_reasons'].extend(level_sources[level])
            
            # Remove duplicates and count
            unique_reasons = list(set(zone['confluent_reasons']))
            zone['confluent_reasons'] = unique_reasons
            zone['confluence_score'] = len(unique_reasons)
        
        # Classify zones as support or resistance
        if '1d' in data and not data['1d'].empty:
            current_price = data['1d']['Close'].iloc[-1]
            
            for zone in zones:
                if zone['zone_center'] < current_price:
                    zone['type'] = 'Support'
                else:
                    zone['type'] = 'Resistance'
                
                # Calculate distance from current price
                zone['distance'] = abs(zone['zone_center'] - current_price)
                zone['distance_percent'] = (zone['distance'] / current_price) * 100
        
        # Sort by confluence score
        zones.sort(key=lambda x: x['confluence_score'], reverse=True)
        
        # Identify key levels (top 3 support and resistance)
        support_zones = [z for z in zones if z.get('type') == 'Support'][:3]
        resistance_zones = [z for z in zones if z.get('type') == 'Resistance'][:3]
        
        return {
            'all_zones': zones,
            'key_support': support_zones,
            'key_resistance': resistance_zones,
            'total_levels_analyzed': len(all_levels),
            'total_zones_identified': len(zones)
        }
    
    def get_nearest_levels(self, zones: List[Dict], current_price: float, count: int = 2) -> Dict:
        """
        Get nearest support and resistance levels from current price
        
        Args:
            zones: List of S/R zones
            current_price: Current market price
            count: Number of levels to return
        
        Returns:
            Dictionary with nearest support and resistance
        """
        support_zones = [z for z in zones if z['zone_center'] < current_price]
        resistance_zones = [z for z in zones if z['zone_center'] > current_price]
        
        # Sort by distance
        support_zones.sort(key=lambda x: current_price - x['zone_center'])
        resistance_zones.sort(key=lambda x: x['zone_center'] - current_price)
        
        return {
            'nearest_support': support_zones[:count],
            'nearest_resistance': resistance_zones[:count],
            'current_price': current_price
        }