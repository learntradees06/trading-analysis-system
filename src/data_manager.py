# Data management module
# src/data_manager.py
"""Enhanced Data Management Module with Maximum Data Collection and Intelligent Caching"""

import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import json
import time
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import numpy as np
import logging

from src.config import DB_PATH, INSTRUMENT_SETTINGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Enhanced Data Manager with:
    - Maximum historical data download
    - Intelligent append/update strategy
    - Duplicate removal
    - Gap detection and filling
    - Optimized caching
    """

    # Yahoo Finance maximum data limits (approximate)
    YF_MAX_PERIODS = {
        '1m': 7,      # 7 days max for 1-minute data
        '2m': 60,     # 60 days
        '5m': 60,     # 60 days
        '15m': 60,    # 60 days
        '30m': 60,    # 60 days
        '60m': 730,   # 730 days (2 years)
        '90m': 60,    # 60 days
        '1h': 730,    # 730 days
        '1d': 10000,  # All available history
        '5d': 10000,  # All available
        '1wk': 10000, # All available
        '1mo': 10000  # All available
    }

    # Map our timeframes to yfinance intervals
    TIMEFRAME_MAPPING = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '60m',
        '4h': '60m',  # Will aggregate from 1h
        '1d': '1d',
        '1wk': '1wk'
    }

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_database()

        # Get instrument settings
        if ticker in INSTRUMENT_SETTINGS:
            self.settings = INSTRUMENT_SETTINGS[ticker]
        else:
            # Default settings for unknown tickers
            self.settings = {
                "tick_size": 0.01,
                "rth_start": "08:30",
                "rth_end": "15:00",
                "timezone": "US/Central"
            }

        # Initialize ticker object
        self.yf_ticker = yf.Ticker(ticker)

    def _init_database(self):
        """Initialize SQLite database with optimized schema"""
        cursor = self.conn.cursor()

        # Create cache metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_metadata (
                ticker TEXT,
                timeframe TEXT,
                first_date TIMESTAMP,
                last_date TIMESTAMP,
                last_update TIMESTAMP,
                total_rows INTEGER,
                data_quality REAL,
                PRIMARY KEY (ticker, timeframe)
            )
        ''')

        # Create OHLCV data table with index for faster queries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                ticker TEXT,
                timeframe TEXT,
                timestamp TIMESTAMP,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (ticker, timeframe, timestamp)
            )
        ''')

        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp
            ON ohlcv_data(ticker, timeframe, timestamp)
        ''')

        # Create data quality log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_quality_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                timeframe TEXT,
                check_date TIMESTAMP,
                gaps_found INTEGER,
                duplicates_removed INTEGER,
                rows_added INTEGER,
                rows_updated INTEGER
            )
        ''')

        self.conn.commit()

    def fetch_data(self, timeframe: str, days_back: Optional[int] = None,
                   force_refresh: bool = False, max_data: bool = True) -> pd.DataFrame:
        """
        Enhanced fetch with maximum data collection and intelligent caching

        Args:
            timeframe: One of '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk'
            days_back: Specific days to fetch (None = maximum available)
            force_refresh: Force complete data refresh
            max_data: Download maximum available data

        Returns:
            DataFrame with OHLCV data
        """

        # Determine how much data to fetch
        if max_data and days_back is None:
            days_back = self._get_max_days_for_timeframe(timeframe)
        elif days_back is None:
            days_back = 30  # Default

        # Check if we need to update
        if not force_refresh:
            cached_data, needs_update = self._get_cached_data_smart(timeframe)

            if cached_data is not None and not cached_data.empty and not needs_update:
                logger.info(f"Using cached data for {self.ticker} {timeframe}")
                return cached_data

        # Fetch new data
        logger.info(f"Fetching {timeframe} data for {self.ticker} (max {days_back} days)")

        if force_refresh:
            # Complete refresh - download all available data
            new_data = self._download_all_available_data(timeframe)
        else:
            # Incremental update - download only what's needed
            new_data = self._download_incremental_data(timeframe, days_back)

        if new_data is not None and not new_data.empty:
            # Process and cache the data
            processed_data = self._process_and_cache_data(new_data, timeframe, force_refresh)
            return processed_data
        else:
            logger.warning(f"No new data received for {self.ticker} {timeframe}")
            # Return cached data if available
            cached_data, _ = self._get_cached_data_smart(timeframe)
            return cached_data if cached_data is not None else pd.DataFrame()

    def _get_max_days_for_timeframe(self, timeframe: str) -> int:
        """Get maximum days of data for a timeframe"""
        yf_interval = self.TIMEFRAME_MAPPING.get(timeframe, '1d')
        return self.YF_MAX_PERIODS.get(yf_interval, 30)

    def _download_all_available_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Download all available historical data for a timeframe"""
        yf_interval = self.TIMEFRAME_MAPPING.get(timeframe, '1d')

        try:
            # For daily and higher timeframes, get all available data
            if timeframe in ['1d', '1wk']:
                logger.info(f"Downloading complete history for {self.ticker} {timeframe}")
                df = self.yf_ticker.history(period="max", interval=yf_interval, auto_adjust=False)
            else:
                # For intraday, get maximum allowed
                max_days = self._get_max_days_for_timeframe(timeframe)
                logger.info(f"Downloading {max_days} days of {timeframe} data for {self.ticker}")

                # Try different approaches for better data
                if max_days <= 60:
                    # Use days for short periods
                    df = self.yf_ticker.history(period=f"{max_days}d", interval=yf_interval, auto_adjust=False)
                else:
                    # Use months for longer periods
                    months = max_days // 30
                    df = self.yf_ticker.history(period=f"{months}mo", interval=yf_interval, auto_adjust=False)

            if df.empty:
                # Fallback to download method
                logger.info("Trying alternative download method...")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self._get_max_days_for_timeframe(timeframe))
                df = self.yf_ticker.history(start=start_date, end=end_date, interval=yf_interval, auto_adjust=False)

            # Handle 4h aggregation
            if timeframe == '4h' and not df.empty:
                df = self._aggregate_to_4h(df)

            return df

        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            logger.info("Trying a shorter period...")
            try:
                # Fallback to a shorter period
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self._get_max_days_for_timeframe(timeframe) // 2)
                df = self.yf_ticker.history(start=start_date, end=end_date, interval=yf_interval, auto_adjust=False)
                if timeframe == '4h' and not df.empty:
                    df = self._aggregate_to_4h(df)
                return df
            except Exception as e2:
                logger.error(f"Shorter period also failed: {e2}")
                return None

    def _download_incremental_data(self, timeframe: str, days_back: int) -> Optional[pd.DataFrame]:
        """Download only new/missing data to append to cache"""

        # Get current cache status
        metadata = self._get_cache_metadata(timeframe)

        if metadata is None:
            # No cache, download all
            return self._download_all_available_data(timeframe)

        last_date = pd.to_datetime(metadata['last_date'])
        now = datetime.now(pytz.utc)

        # Determine if we need to download new data
        if timeframe in ['1m', '5m', '15m', '30m', '1h', '4h']:
            # For intraday, update if market is open or if last update was before last close
            if self._is_market_open() or (now - last_date).total_seconds() > 3600:
                # Download recent data
                start_date = last_date - timedelta(days=1)  # Overlap for safety
            else:
                logger.info("No update needed - market closed and data is recent")
                return None
        else:
            # For daily, update if last update was before today's close
            if last_date.date() < now.date():
                start_date = last_date - timedelta(days=5)  # Overlap for weekly bars
            else:
                logger.info("Daily data is up to date")
                return None

        # Download incremental data
        yf_interval = self.TIMEFRAME_MAPPING.get(timeframe, '1d')

        try:
            logger.info(f"Downloading incremental data from {start_date} for {self.ticker} {timeframe}")
            df = self.yf_ticker.history(start=start_date, end=now, interval=yf_interval, auto_adjust=False)

            # Handle 4h aggregation
            if timeframe == '4h' and not df.empty:
                df = self._aggregate_to_4h(df)

            return df

        except Exception as e:
            logger.error(f"Error downloading incremental data: {e}")
            return None

    def _process_and_cache_data(self, new_data: pd.DataFrame, timeframe: str,
                                force_refresh: bool) -> pd.DataFrame:
        """Process new data and update cache intelligently"""

        if new_data.empty:
            return new_data

        # Get existing cached data
        if force_refresh:
            existing_data = pd.DataFrame()
        else:
            existing_data, _ = self._get_cached_data_smart(timeframe)
            if existing_data is None:
                existing_data = pd.DataFrame()

        # Combine data
        if not existing_data.empty:
            # Append new data to existing, remove duplicates
            combined_data = self._merge_dataframes(existing_data, new_data)
        else:
            combined_data = new_data

        # Clean data
        combined_data = self._clean_data(combined_data)

        # Detect and log data quality issues
        quality_report = self._check_data_quality(combined_data, timeframe)

        # Cache the processed data
        self._cache_data(combined_data, timeframe)

        # Log the update
        self._log_data_update(timeframe, quality_report)

        return combined_data

    def _merge_dataframes(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        """Intelligently merge existing and new data, removing duplicates"""

        # Standardize both to UTC
        if not existing.empty:
            if not isinstance(existing.index, pd.DatetimeIndex):
                existing.index = pd.to_datetime(existing.index, utc=True)
            if existing.index.tz is None:
                existing = existing.tz_localize('UTC')
            else:
                existing = existing.tz_convert('UTC')

        if not new.empty:
            if not isinstance(new.index, pd.DatetimeIndex):
                new.index = pd.to_datetime(new.index, utc=True)
            if new.index.tz is None:
                new = new.tz_localize('UTC')
            else:
                new = new.tz_convert('UTC')

        # Combine dataframes
        combined = pd.concat([existing, new])

        # Remove exact duplicates
        combined = combined[~combined.index.duplicated(keep='last')]

        # Sort by index
        combined = combined.sort_index()

        # Handle any remaining duplicates by keeping the most recent
        combined = combined.groupby(combined.index).last()

        return combined

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data: remove NaN, fix splits, handle gaps"""

        if df.empty:
            return df

        # Remove rows with NaN in OHLC
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

        # Fix zero volume (set to NaN then forward fill)
        df.loc[df['Volume'] == 0, 'Volume'] = np.nan
        df['Volume'] = df['Volume'].fillna(method='ffill')

        # Detect and adjust for stock splits (simple detection)
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            # Detect sudden price jumps/drops > 20%
            pct_change = df[col].pct_change()
            split_points = pct_change.abs() > 0.20

            if split_points.any():
                logger.warning(f"Potential split detected in {col} - investigating...")
                # You could implement more sophisticated split detection here

        # Handle gaps in time series
        df = self._fill_gaps(df)

        return df

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in time series data"""

        if df.empty or len(df) < 2:
            return df

        # Determine expected frequency
        freq_counts = df.index.to_series().diff().value_counts()
        if not freq_counts.empty:
            expected_freq = freq_counts.index[0]

            # Create complete date range
            full_range = pd.date_range(start=df.index[0], end=df.index[-1], freq=expected_freq)

            # Reindex to fill gaps
            df = df.reindex(full_range)

            # Forward fill gaps (markets closed)
            df = df.fillna(method='ffill')

        return df

    def _check_data_quality(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Check data quality and return report"""

        report = {
            'total_rows': len(df),
            'gaps_found': 0,
            'duplicates_removed': 0,
            'missing_values': 0,
            'quality_score': 100.0
        }

        if df.empty:
            report['quality_score'] = 0
            return report

        # Check for gaps
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            expected_freq = time_diffs.mode()[0] if not time_diffs.mode().empty else time_diffs.median()
            gaps = time_diffs[time_diffs > expected_freq * 2]
            report['gaps_found'] = len(gaps)

        # Check for missing values
        report['missing_values'] = df[['Open', 'High', 'Low', 'Close']].isnull().sum().sum()

        # Calculate quality score
        penalties = (
            report['gaps_found'] * 0.5 +
            report['missing_values'] * 1.0
        )
        report['quality_score'] = max(0, 100 - penalties)

        return report

    def _get_cached_data_smart(self, timeframe: str) -> Tuple[Optional[pd.DataFrame], bool]:
        """
        Get cached data and determine if update is needed

        Returns:
            Tuple of (DataFrame or None, needs_update boolean)
        """

        # Get metadata
        metadata = self._get_cache_metadata(timeframe)

        if metadata is None:
            return None, True

        # Check if data is stale
        last_update = pd.to_datetime(metadata['last_update'])
        now = datetime.now()

        # Different staleness thresholds for different timeframes
        if timeframe in ['1m', '5m']:
            is_stale = (now - last_update).total_seconds() > 300  # 5 minutes
        elif timeframe in ['15m', '30m', '1h']:
            is_stale = (now - last_update).total_seconds() > 900  # 15 minutes
        elif timeframe == '4h':
            is_stale = (now - last_update).total_seconds() > 3600  # 1 hour
        else:  # Daily or longer
            is_stale = (now - last_update).total_seconds() > 86400  # 24 hours

        # Load data from cache
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE ticker = ? AND timeframe = ?
            ORDER BY timestamp
        '''

        df = pd.read_sql_query(query, self.conn, params=(self.ticker, timeframe))

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Ensure timezone aware
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
                df.index = df.index.tz_localize('UTC')

        return df if not df.empty else None, is_stale

    def _cache_data(self, df: pd.DataFrame, timeframe: str):
        """Enhanced caching with transaction and bulk insert"""

        if df.empty:
            return

        cursor = self.conn.cursor()

        try:
            # Start transaction
            cursor.execute('BEGIN TRANSACTION')

            # Clear existing data for complete refresh
            cursor.execute('''
                DELETE FROM ohlcv_data
                WHERE ticker = ? AND timeframe = ?
            ''', (self.ticker, timeframe))

            # Prepare bulk insert data
            records = []
            for timestamp, row in df.iterrows():
                records.append((
                    self.ticker,
                    timeframe,
                    timestamp.isoformat(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row['Volume'])
                ))

            # Bulk insert
            cursor.executemany('''
                INSERT OR REPLACE INTO ohlcv_data
                (ticker, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)

            # Update metadata
            first_date = df.index[0]
            last_date = df.index[-1]

            # Calculate data quality score
            quality_score = self._calculate_quality_score(df)

            cursor.execute('''
                INSERT OR REPLACE INTO cache_metadata
                (ticker, timeframe, first_date, last_date, last_update, total_rows, data_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (self.ticker, timeframe, first_date.isoformat(), last_date.isoformat(),
                  datetime.now().isoformat(), len(df), quality_score))

            # Commit transaction
            cursor.execute('COMMIT')
            logger.info(f"Cached {len(df)} rows for {self.ticker} {timeframe}")

        except Exception as e:
            cursor.execute('ROLLBACK')
            logger.error(f"Error caching data: {e}")
            raise

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-100)"""

        if df.empty:
            return 0.0

        score = 100.0

        # Check for missing values
        missing_pct = df[['Open', 'High', 'Low', 'Close']].isnull().sum().sum() / (len(df) * 4)
        score -= missing_pct * 50

        # Check for zero volume
        zero_vol_pct = (df['Volume'] == 0).sum() / len(df)
        score -= zero_vol_pct * 20

        # Check for time gaps (for intraday data)
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            if not time_diffs.empty:
                expected_freq = time_diffs.mode()[0] if not time_diffs.mode().empty else time_diffs.median()
                gap_pct = (time_diffs > expected_freq * 2).sum() / len(time_diffs)
                score -= gap_pct * 30

        return max(0.0, min(100.0, score))

    def _get_cache_metadata(self, timeframe: str) -> Optional[Dict]:
        """Get cache metadata for a timeframe"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM cache_metadata
            WHERE ticker = ? AND timeframe = ?
        ''', (self.ticker, timeframe))

        result = cursor.fetchone()
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        return None

    def _log_data_update(self, timeframe: str, quality_report: Dict):
        """Log data update to quality log"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO data_quality_log
            (ticker, timeframe, check_date, gaps_found, duplicates_removed, rows_added, rows_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (self.ticker, timeframe, datetime.now().isoformat(),
              quality_report.get('gaps_found', 0),
              quality_report.get('duplicates_removed', 0),
              quality_report.get('total_rows', 0),
              0))  # rows_updated could be tracked separately
        self.conn.commit()

    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        tz = pytz.timezone(self.settings['timezone'])
        now = datetime.now(tz)

        # Parse RTH times
        start_hour, start_min = map(int, self.settings['rth_start'].split(':'))
        end_hour, end_min = map(int, self.settings['rth_end'].split(':'))

        market_open = now.replace(hour=start_hour, minute=start_min, second=0)
        market_close = now.replace(hour=end_hour, minute=end_min, second=0)

        # Check if weekend
        if now.weekday() >= 5:
            return False

        return market_open <= now <= market_close

    def _aggregate_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 1h data to 4h bars"""
        if df.empty:
            return df

        return df.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    def get_rth_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL FUNCTION: Filter dataframe to Regular Trading Hours only

        Args:
            df: DataFrame with datetime index

        Returns:
            DataFrame filtered to RTH only
        """
        if df.empty:
            return df

        # Get RTH times from settings
        rth_start = self.settings['rth_start']
        rth_end = self.settings['rth_end']
        timezone = self.settings['timezone']

        # Convert index to timezone-aware if not already
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')

        # Convert to market timezone
        df_tz = df.copy()
        df_tz.index = df_tz.index.tz_convert(timezone)

        # Filter to RTH
        start_hour, start_min = map(int, rth_start.split(':'))
        end_hour, end_min = map(int, rth_end.split(':'))

        rth_mask = (
            (df_tz.index.hour > start_hour) |
            ((df_tz.index.hour == start_hour) & (df_tz.index.minute >= start_min))
        ) & (
            (df_tz.index.hour < end_hour) |
            ((df_tz.index.hour == end_hour) & (df_tz.index.minute <= end_min))
        )

        return df_tz[rth_mask]

    def get_data_summary(self) -> pd.DataFrame:
        """Get summary of all cached data for this ticker"""
        query = '''
            SELECT
                timeframe,
                first_date,
                last_date,
                total_rows,
                data_quality,
                last_update
            FROM cache_metadata
            WHERE ticker = ?
            ORDER BY timeframe
        '''

        summary = pd.read_sql_query(query, self.conn, params=(self.ticker,))

        if not summary.empty:
            summary['first_date'] = pd.to_datetime(summary['first_date'])
            summary['last_date'] = pd.to_datetime(summary['last_date'])
            summary['last_update'] = pd.to_datetime(summary['last_update'])
            summary['days_covered'] = (summary['last_date'] - summary['first_date']).dt.days

        return summary

    def optimize_cache(self):
        """Optimize database cache by removing old/redundant data"""
        cursor = self.conn.cursor()

        # Remove data older than 1 year for minute timeframes
        one_year_ago = (datetime.now() - timedelta(days=365)).isoformat()

        cursor.execute('''
            DELETE FROM ohlcv_data
            WHERE ticker = ? AND timeframe IN ('1m', '5m') AND timestamp < ?
        ''', (self.ticker, one_year_ago))

        # Vacuum to reclaim space
        cursor.execute('VACUUM')

        self.conn.commit()
        logger.info(f"Cache optimized for {self.ticker}")

    def export_to_csv(self, timeframe: str, filepath: str):
        """Export cached data to CSV file"""
        df, _ = self._get_cached_data_smart(timeframe)
        if df is not None and not df.empty:
            df.to_csv(filepath)
            logger.info(f"Exported {len(df)} rows to {filepath}")
        else:
            logger.warning(f"No data to export for {timeframe}")

    def __del__(self):
        """Close database connection on cleanup"""
        if hasattr(self, 'conn'):
            self.conn.close()

# Utility functions
def download_multiple_tickers(tickers: List[str], timeframes: List[str], max_data: bool = True):
    """Download data for multiple tickers and timeframes"""
    results = {}

    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        dm = DataManager(ticker)
        results[ticker] = {}

        for timeframe in timeframes:
            try:
                df = dm.fetch_data(timeframe, max_data=max_data)
                results[ticker][timeframe] = len(df)
                logger.info(f"  {timeframe}: {len(df)} rows")
            except Exception as e:
                logger.error(f"  Error with {timeframe}: {e}")
                results[ticker][timeframe] = 0

        # Optimize cache after bulk download
        dm.optimize_cache()

    return results

def get_cache_statistics():
    """Get overall cache statistics"""
    conn = sqlite3.connect(DB_PATH)

    # Get total cache size
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total_rows FROM ohlcv_data")
    total_rows = cursor.fetchone()[0]

    # Get cache by ticker
    cursor.execute('''
        SELECT ticker, COUNT(*) as rows,
               MIN(timestamp) as first_date,
               MAX(timestamp) as last_date
        FROM ohlcv_data
        GROUP BY ticker
    ''')

    stats = pd.DataFrame(cursor.fetchall(), columns=['ticker', 'rows', 'first_date', 'last_date'])

    # Get database file size
    db_size = Path(DB_PATH).stat().st_size / (1024 * 1024)  # Size in MB

    conn.close()

    return {
        'total_rows': total_rows,
        'database_size_mb': round(db_size, 2),
        'ticker_stats': stats
    }
