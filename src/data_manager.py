# src/data_manager.py
"""Data Management Module with Thread-Safe Singleton DB Connection"""
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import threading
from typing import Optional, Dict, Any, List
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
                # Use WAL mode for better concurrency
                cls._instance.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                cls._instance.conn.execute('PRAGMA journal_mode=WAL;')
                cls._instance.lock = threading.Lock()
        return cls._instance
    def get_connection(self): return self.conn

class DataManager:
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
            # Correct schema for cache_metadata (6 columns)
            cursor.execute('''CREATE TABLE IF NOT EXISTS cache_metadata (
                                ticker TEXT, timeframe TEXT, first_date TIMESTAMP,
                                last_date TIMESTAMP, last_update TIMESTAMP, total_rows INTEGER,
                                PRIMARY KEY (ticker, timeframe))''')
            # Schema with lowercase column names for consistency
            cursor.execute('''CREATE TABLE IF NOT EXISTS ohlcv_data (
                                ticker TEXT, timeframe TEXT, timestamp TIMESTAMP, open REAL, high REAL,
                                low REAL, close REAL, volume REAL, PRIMARY KEY (ticker, timeframe, timestamp))''')
            self.conn.commit()

    def fetch_data(self, timeframe: str, days_back: int = 365) -> pd.DataFrame:
        df = self._get_cached_data(timeframe)
        if not df.empty:
            logger.info(f"Loaded {len(df)} rows from cache for {self.ticker} {timeframe}")
            return df.tail(days_back) # Return requested slice

        logger.info(f"Fetching new data for {self.ticker} {timeframe}")
        yf_interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '60m', '4h': '60m', '1d': '1d'}
        yf_interval = yf_interval_map.get(timeframe, '1d')

        # Fetch a bit more data for indicator calculations
        period_to_fetch = days_back + 100

        new_data = self.yf_ticker.history(period=f"{period_to_fetch}d", interval=yf_interval, auto_adjust=False)

        if new_data.empty:
            logger.warning(f"No data returned from yfinance for {self.ticker} {timeframe}")
            return pd.DataFrame()

        # --- Enforce PascalCase Naming Convention for application use ---
        new_data.columns = [col.capitalize() for col in new_data.columns]
        if 'Adj close' in new_data.columns:
            new_data = new_data.rename(columns={'Adj close': 'Adj_close'})

        if new_data.index.tz is None:
            new_data.index = new_data.index.tz_localize('UTC')
        else:
            new_data.index = new_data.index.tz_convert('UTC')

        self._cache_data(new_data, timeframe)
        return new_data.tail(days_back)

    def _get_cached_data(self, timeframe: str) -> pd.DataFrame:
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv_data WHERE ticker = ? AND timeframe = ? ORDER BY timestamp"
        with self.db.lock:
            try:
                df = pd.read_sql(query, self.conn, params=(self.ticker, timeframe), index_col='timestamp', parse_dates=['timestamp'])
                if not df.empty:
                    # --- Enforce PascalCase Naming Convention after loading from DB ---
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
                return df
            except Exception as e:
                logger.warning(f"Could not read from cache for {self.ticker} {timeframe}: {e}")
                return pd.DataFrame()

    def _cache_data(self, df: pd.DataFrame, timeframe: str):
        df_to_cache = df.copy()

        # --- Enforce lowercase Naming Convention for DB storage ---
        df_to_cache.columns = [col.lower() for col in df_to_cache.columns]

        # Select only the columns that exist in our DB schema
        db_columns = ['open', 'high', 'low', 'close', 'volume']
        cols_to_keep = [col for col in db_columns if col in df_to_cache.columns]
        df_to_cache = df_to_cache[cols_to_keep]

        df_to_cache.reset_index(inplace=True)
        df_to_cache = df_to_cache.rename(columns={'index': 'timestamp', 'Date': 'timestamp'})

        df_to_cache['ticker'] = self.ticker
        df_to_cache['timeframe'] = timeframe

        with self.db.lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM ohlcv_data WHERE ticker = ? AND timeframe = ?", (self.ticker, timeframe))
            cursor.execute("DELETE FROM cache_metadata WHERE ticker = ? AND timeframe = ?", (self.ticker, timeframe))

            df_to_cache.to_sql('ohlcv_data', self.conn, if_exists='append', index=False)

            # Correct INSERT statement with 6 values for 6 columns
            cursor.execute("""
                INSERT INTO cache_metadata (ticker, timeframe, first_date, last_date, last_update, total_rows)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (self.ticker, timeframe, df.index.min().isoformat(), df.index.max().isoformat(), datetime.now().isoformat(), len(df)))
            self.conn.commit()
        logger.info(f"Cached {len(df)} rows for {self.ticker} {timeframe}")

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
