# src/data_manager.py
"""Data Management Module with Thread-Safe Singleton DB Connection and Incremental Caching"""
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import threading
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging

from src.config import DB_PATH, INSTRUMENT_SETTINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Thread-Safe Database Singleton ---
class Database:
    _instance = None
    _lock = threading.Lock()
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                cls._instance.conn.execute('PRAGMA journal_mode=WAL;')
                cls._instance.lock = threading.Lock()
        return cls._instance
    def get_connection(self): return self.conn

class DataManager:
    YF_INTERVAL_MAP = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '1d': '1d', '1wk': '1wk'}
    YF_MAX_DAYS_INTRADAY = 60 # yfinance limit for intraday < 1h
    YF_MAX_DAYS_HOURLY = 730 # yfinance limit for 1h

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.db = Database()
        self.conn = self.db.get_connection()
        self._init_database()
        self.settings = INSTRUMENT_SETTINGS.get(ticker, {
            "tick_size": 0.01, "rth_start": "08:30", "rth_end": "15:00", "timezone": "US/Central"
        })
        self.yf_ticker = yf.Ticker(ticker)

    def _init_database(self):
        with self.db.lock:
            cursor = self.conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS cache_metadata (
                                ticker TEXT, timeframe TEXT, first_date TIMESTAMP,
                                last_date TIMESTAMP, last_update TIMESTAMP, total_rows INTEGER,
                                PRIMARY KEY (ticker, timeframe))''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS ohlcv_data (
                                ticker TEXT, timeframe TEXT, timestamp TIMESTAMP, open REAL, high REAL,
                                low REAL, close REAL, volume REAL, PRIMARY KEY (ticker, timeframe, timestamp))''')
            self.conn.commit()

    def fetch_data(self, timeframe: str, days_back: int = 365, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches historical data, using and updating a local cache incrementally.
        For ML training, you may want to call this with a very large `days_back` value
        to ensure you get the full cached history.
        """
        # 1. Get cached data and metadata
        cached_df = self._get_cached_data(timeframe) if use_cache else pd.DataFrame()
        metadata = self._get_cache_metadata(timeframe) if use_cache else {}

        last_cached_date = pd.to_datetime(metadata.get('last_date'), utc=True) if metadata.get('last_date') else None

        # 2. Determine if the cache is fresh enough
        is_fresh = last_cached_date and (datetime.now(pytz.UTC) - last_cached_date) < timedelta(minutes=15)

        # 3. Define the slicing logic for the final output
        slicer = self._get_slicer(timeframe, days_back)

        # 4. If cache is fresh, return sliced data from the complete cached set
        if not cached_df.empty and is_fresh:
            logger.info(f"Using fresh cached data for {self.ticker} {timeframe}")
            return slicer(cached_df)

        # 5. Determine the start date for the API call
        start_date = None
        if last_cached_date:
            # Fetch one extra day to handle potential partial data
            start_date = (last_cached_date - timedelta(days=1)).strftime('%Y-%m-%d')

        yf_interval = self.YF_INTERVAL_MAP.get(timeframe)
        if not yf_interval:
            logger.error(f"Invalid timeframe requested: {timeframe}")
            return pd.DataFrame()

        # 6. Fetch new data from yfinance
        logger.info(f"Fetching new data for {self.ticker} {timeframe} since {start_date or 'beginning'}")
        new_data = self._fetch_yf_data(yf_interval, start_date)

        if new_data.empty:
            logger.warning(f"No new data returned from yfinance for {self.ticker} {timeframe}")
            return slicer(cached_df) if not cached_df.empty else pd.DataFrame()

        # 7. Process and combine data
        new_data = self._process_new_data(new_data)
        combined_df = pd.concat([cached_df, new_data])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()

        # 8. Cache the new combined data
        if use_cache:
            self._cache_data(new_data, timeframe) # Only cache the newly downloaded portion

        return slicer(combined_df)

    def _get_slicer(self, timeframe: str, days_back: int):
        """Returns a lambda function to slice a DataFrame correctly based on timeframe."""
        if timeframe in ['1d', '1wk']:
            return lambda d: d.tail(days_back)
        else:
            calendar_days = int(days_back * 365.25 / 252) + 5
            cutoff_date = datetime.now(pytz.UTC) - timedelta(days=calendar_days)
            return lambda d: d[d.index >= cutoff_date]

    def _fetch_yf_data(self, yf_interval: str, start_date: Optional[str]) -> pd.DataFrame:
        """Handles the actual data fetching from yfinance with appropriate params."""
        fetch_params = {"interval": yf_interval, "auto_adjust": False}
        if start_date:
             fetch_params['start'] = start_date
        elif yf_interval in ['1m', '5m', '15m', '30m']:
            fetch_params['period'] = f"{self.YF_MAX_DAYS_INTRADAY}d"
        elif yf_interval == '1h':
             fetch_params['period'] = f"{self.YF_MAX_DAYS_HOURLY}d"
        else:
             fetch_params['period'] = "max"
        return self.yf_ticker.history(**fetch_params)

    def _process_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Standardizes columns and timezone for a new DataFrame from yfinance."""
        new_data.columns = [col.capitalize() for col in new_data.columns]
        if 'Adj close' in new_data.columns:
            new_data.rename(columns={'Adj close': 'Adj_close'}, inplace=True)

        if new_data.index.tz is None:
            return new_data.tz_localize('UTC')
        return new_data.tz_convert('UTC')

    def _get_cached_data(self, timeframe: str) -> pd.DataFrame:
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv_data WHERE ticker = ? AND timeframe = ? ORDER BY timestamp"
        with self.db.lock:
            try:
                df = pd.read_sql(query, self.conn, params=(self.ticker, timeframe), index_col='timestamp', parse_dates=['timestamp'])
                if not df.empty:
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
                return df
            except Exception as e:
                logger.warning(f"Could not read from cache for {self.ticker} {timeframe}: {e}")
                return pd.DataFrame()

    def _get_cache_metadata(self, timeframe: str) -> Dict:
        """Retrieves the metadata for a given ticker and timeframe."""
        query = "SELECT * FROM cache_metadata WHERE ticker = ? AND timeframe = ?"
        with self.db.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(query, (self.ticker, timeframe))
                row = cursor.fetchone()
                if row:
                    cols = [desc[0] for desc in cursor.description]
                    return dict(zip(cols, row))
            except Exception as e:
                logger.warning(f"Could not read metadata for {self.ticker} {timeframe}: {e}")
        return {}

    def _cache_data(self, df: pd.DataFrame, timeframe: str):
        """Caches new data incrementally using an upsert operation and updates metadata."""
        if df.empty: return

        df_to_cache = df.copy()
        df_to_cache.columns = [col.lower() for col in df_to_cache.columns]
        db_columns = ['open', 'high', 'low', 'close', 'volume']
        df_to_cache = df_to_cache[[col for col in db_columns if col in df_to_cache.columns]]
        df_to_cache.reset_index(inplace=True)
        df_to_cache.rename(columns={df_to_cache.columns[0]: 'timestamp'}, inplace=True)
        df_to_cache['ticker'] = self.ticker
        df_to_cache['timeframe'] = timeframe

        with self.db.lock:
            df_to_cache.to_sql('ohlcv_data', self.conn, if_exists='append', index=False, method=self._sqlite_upsert)

            cursor = self.conn.cursor()
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM ohlcv_data WHERE ticker = ? AND timeframe = ?", (self.ticker, timeframe))
            first_date, last_date, total_rows = cursor.fetchone()

            cursor.execute("""
                INSERT OR REPLACE INTO cache_metadata (ticker, timeframe, first_date, last_date, last_update, total_rows)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (self.ticker, timeframe, first_date, last_date, datetime.now().isoformat(), total_rows))
            self.conn.commit()

        logger.info(f"Upserted {len(df_to_cache)} rows for {self.ticker} {timeframe}. Total cached rows: {total_rows}")

    @staticmethod
    def _sqlite_upsert(table, conn, keys, data_iter):
        """Performs an 'upsert' (INSERT OR REPLACE) operation for sqlite."""
        column_names = ", ".join(keys)
        placeholders = ", ".join(["?"] * len(keys))
        sql = f"INSERT OR REPLACE INTO {table.name} ({column_names}) VALUES ({placeholders})"
        conn.executemany(sql, data_iter)

    def get_rth_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df_tz = df.copy()
        if df_tz.index.tz is None: df_tz.index = df_tz.index.tz_localize('UTC')
        df_tz.index = df_tz.index.tz_convert(self.settings['timezone'])
        return df_tz.between_time(self.settings['rth_start'], self.settings['rth_end'])

def get_cache_statistics() -> Dict[str, Dict[str, Any]]:
    db = Database()
    with db.lock:
        cursor = db.get_connection().cursor()
        try:
            cursor.execute("SELECT ticker, timeframe, total_rows, first_date, last_date FROM cache_metadata ORDER BY ticker, timeframe")
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            return {}
    summary = {}
    for ticker, timeframe, total_rows, first_date, last_date in rows:
        if ticker not in summary: summary[ticker] = {}
        summary[ticker][timeframe] = {'rows': total_rows, 'start_date': first_date, 'end_date': last_date}
    return summary
