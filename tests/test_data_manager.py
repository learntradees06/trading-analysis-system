# tests/test_data_manager.py
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import pytz
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_manager import DataManager, Database, get_cache_statistics

TEST_DB_PATH = ":memory:"

def create_sample_dataframe(days, tz='UTC', end_date=None):
    """Creates a sample DataFrame for testing."""
    if end_date is None:
        end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=days - 1)
    dates = pd.date_range(start=start_date, periods=days, freq='D', tz=tz)
    data = {'Open': range(days), 'High': range(days), 'Low': range(days), 'Close': range(days), 'Volume': range(days)}
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'timestamp'
    return df

class TestDataManager(unittest.TestCase):

    @patch('src.data_manager.Database')
    def setUp(self, MockDatabase):
        """Set up a clean, in-memory database and a DataManager instance for each test."""
        # Mock the Database singleton to use an in-memory SQLite DB
        self.conn = sqlite3.connect(TEST_DB_PATH, check_same_thread=False)
        mock_db_instance = MockDatabase.return_value
        mock_db_instance.get_connection.return_value = self.conn
        mock_db_instance.lock = MagicMock() # Simple lock mock

        # Mock yfinance.Ticker
        self.mock_yf_ticker_patch = patch('src.data_manager.yf.Ticker')
        self.mock_yf_ticker = self.mock_yf_ticker_patch.start()

        self.ticker = "TEST"
        self.dm = DataManager(self.ticker)

    def tearDown(self):
        """Clean up by stopping patches and closing the connection."""
        self.mock_yf_ticker_patch.stop()
        self.conn.close()

    def test_initialization(self):
        """Test that the DataManager initializes correctly."""
        self.assertEqual(self.dm.ticker, self.ticker)
        self.assertIsNotNone(self.dm.conn)

        # Check if tables were created
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_data'")
        self.assertIsNotNone(cursor.fetchone())
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cache_metadata'")
        self.assertIsNotNone(cursor.fetchone())

    def test_fetch_data_empty_cache(self):
        """Test fetching data when the cache is empty."""
        sample_df = create_sample_dataframe(10)
        self.mock_yf_ticker.return_value.history.return_value = sample_df

        df = self.dm.fetch_data('1d', days_back=10)

        self.mock_yf_ticker.return_value.history.assert_called_once()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 10)
        # Compare only the essential columns as yfinance might add more
        pd.testing.assert_frame_equal(df.sort_index()[sample_df.columns], sample_df.sort_index(), check_dtype=False)

    def test_fetch_data_fresh_cache(self):
        """Test that fresh cached data is used and no API call is made."""
        # 1. Pre-populate the cache
        initial_df = create_sample_dataframe(20)
        self.dm._cache_data(initial_df, '1d')

        # 2. Mock yfinance to ensure it's NOT called
        self.mock_yf_ticker.return_value.history.return_value = pd.DataFrame() # Should not be returned

        # 3. Fetch data
        df = self.dm.fetch_data('1d', days_back=20)

        self.mock_yf_ticker.return_value.history.assert_not_called()
        self.assertEqual(len(df), 20)
        self.assertEqual(df.index.min().date(), (datetime.now(pytz.UTC) - timedelta(days=19)).date())

    def test_fetch_data_stale_cache(self):
        """Test fetching when cache is stale, requiring an update."""
        # 1. Create data that is explicitly in the past
        stale_end_date = datetime.now(pytz.UTC) - timedelta(days=5)
        old_df = create_sample_dataframe(10, end_date=stale_end_date)
        self.dm._cache_data(old_df, '1d')

        # 2. New data to be "downloaded"
        # yfinance will return data from the last cached date to now
        new_start_date = old_df.index.max() + timedelta(days=1)
        days_to_fetch = (datetime.now(pytz.UTC) - new_start_date).days + 1
        new_df = create_sample_dataframe(days_to_fetch, end_date=datetime.now(pytz.UTC))
        self.mock_yf_ticker.return_value.history.return_value = new_df

        # 3. Fetch data
        df = self.dm.fetch_data('1d', days_back=10 + days_to_fetch)

        self.mock_yf_ticker.return_value.history.assert_called_once()
        self.assertEqual(len(df), 10 + days_to_fetch) # Should be combined

    def test_cache_and_retrieve(self):
        """Test caching data and then retrieving it."""
        sample_df = create_sample_dataframe(5)
        self.dm._cache_data(sample_df, '1d')

        cached_df = self.dm._get_cached_data('1d')
        self.assertFalse(cached_df.empty)
        self.assertEqual(len(cached_df), 5)
        # Adjust column names for comparison as they are different in DB vs DataFrame
        cached_df.columns = [c.capitalize() for c in cached_df.columns]
        pd.testing.assert_frame_equal(cached_df.sort_index(), sample_df.sort_index(), check_freq=False, check_dtype=False)

        metadata = self.dm._get_cache_metadata('1d')
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['total_rows'], 5)

    def test_get_rth_data(self):
        """Test filtering for regular trading hours."""
        # Define timestamps in UTC that correspond to specific times in US/Central
        # RTH: 08:30 - 15:00 inclusive
        utc_dates = pd.to_datetime([
            '2023-10-26 13:29:00',  # 08:29 CT -> Outside
            '2023-10-26 13:30:00',  # 08:30 CT -> Inside
            '2023-10-26 19:59:00',  # 14:59 CT -> Inside
            '2023-10-26 20:00:00'   # 15:00 CT -> Inside (inclusive)
        ]).tz_localize('UTC')

        df = pd.DataFrame(index=utc_dates, data={'Close': [1, 2, 3, 4]})

        # Use the default settings from the DataManager instance
        self.dm.settings['timezone'] = 'US/Central'
        self.dm.settings['rth_start'] = '08:30'
        self.dm.settings['rth_end'] = '15:00'

        rth_df = self.dm.get_rth_data(df)

        self.assertEqual(len(rth_df), 3)
        # Check that the correct rows were kept
        self.assertEqual(rth_df.iloc[0]['Close'], 2) # 08:30
        self.assertEqual(rth_df.iloc[1]['Close'], 3) # 14:59
        self.assertEqual(rth_df.iloc[2]['Close'], 4) # 15:00

    @patch('src.data_manager.Database')
    def test_get_cache_statistics_empty(self, MockDatabase):
        """Test get_cache_statistics when the database is empty."""
        conn = sqlite3.connect(':memory:')
        mock_db_instance = MockDatabase.return_value
        mock_db_instance.get_connection.return_value = conn
        mock_db_instance.lock = MagicMock()

        # Need to create the table, but it will be empty
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS cache_metadata (
                            ticker TEXT, timeframe TEXT, first_date TIMESTAMP,
                            last_date TIMESTAMP, last_update TIMESTAMP, total_rows INTEGER,
                            PRIMARY KEY (ticker, timeframe))''')
        conn.commit()

        stats = get_cache_statistics()
        self.assertEqual(stats, {})

if __name__ == '__main__':
    unittest.main()
