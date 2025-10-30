# tests/test_indicators.py
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indicators import (
    calculate_atr,
    calculate_rsi,
    calculate_adx,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_all_indicators
)

def create_ohlc_dataframe():
    """Creates a sample OHLCV DataFrame for testing."""
    data = {
        'Open': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110, 112, 111, 113, 115],
        'High': [103, 104, 103, 105, 106, 106, 108, 109, 109, 111, 112, 113, 113, 115, 116],
        'Low': [99, 101, 100, 102, 104, 103, 105, 107, 106, 108, 109, 111, 110, 112, 114],
        'Close': [102, 103, 102, 104, 105, 105, 107, 108, 108, 110, 111, 112, 112, 114, 115],
        'Volume': [1000, 1500, 1200, 1800, 2000, 1700, 2200, 2500, 2300, 2800, 3000, 3200, 3100, 3500, 4000]
    }
    df = pd.DataFrame(data)
    return df

class TestIndicators(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame for all tests."""
        self.df = create_ohlc_dataframe()

    def test_calculate_atr(self):
        """Test the ATR calculation."""
        atr = calculate_atr(self.df, period=14)
        self.assertIsInstance(atr, pd.Series)
        self.assertFalse(atr.isnull().all())
        # A simple check on the last value (not a precise calculation, but confirms it runs)
        self.assertGreater(atr.iloc[-1], 0)
        self.assertEqual(len(atr), len(self.df))

    def test_calculate_rsi(self):
        """Test the RSI calculation."""
        rsi = calculate_rsi(self.df, period=14)
        self.assertIsInstance(rsi, pd.Series)
        self.assertFalse(rsi.isnull().all())
        # RSI should be between 0 and 100
        self.assertTrue((rsi.dropna() >= 0).all() and (rsi.dropna() <= 100).all())
        self.assertEqual(len(rsi), len(self.df))

    def test_calculate_adx(self):
        """Test the ADX calculation."""
        adx, plus_di, minus_di = calculate_adx(self.df, period=14)
        self.assertIsInstance(adx, pd.Series)
        self.assertIsInstance(plus_di, pd.Series)
        self.assertIsInstance(minus_di, pd.Series)
        self.assertFalse(adx.isnull().all())
        self.assertEqual(len(adx), len(self.df))

    def test_calculate_macd(self):
        """Test the MACD calculation."""
        macd_dict = calculate_macd(self.df)
        self.assertIsInstance(macd_dict, dict)
        self.assertIn('macd', macd_dict)
        self.assertIn('signal', macd_dict)
        self.assertIn('histogram', macd_dict)
        self.assertIsInstance(macd_dict['macd'], pd.Series)
        self.assertEqual(len(macd_dict['macd']), len(self.df))

    def test_calculate_bollinger_bands(self):
        """Test the Bollinger Bands calculation."""
        bb_dict = calculate_bollinger_bands(self.df, period=5)
        self.assertIsInstance(bb_dict, dict)
        self.assertIn('upper', bb_dict)
        self.assertIn('middle', bb_dict)
        self.assertIn('lower', bb_dict)
        # The upper band should always be greater than or equal to the lower band
        self.assertTrue((bb_dict['upper'].dropna() >= bb_dict['lower'].dropna()).all())
        self.assertEqual(len(bb_dict['upper']), len(self.df))

    def test_calculate_all_indicators(self):
        """Test the wrapper function that calculates all indicators."""
        df_with_indicators = calculate_all_indicators(self.df)
        self.assertIsInstance(df_with_indicators, pd.DataFrame)
        # Check if a few key indicator columns were added
        self.assertIn('ATR', df_with_indicators.columns)
        self.assertIn('RSI', df_with_indicators.columns)
        self.assertIn('MACD', df_with_indicators.columns)
        self.assertIn('BB_Upper', df_with_indicators.columns)
        # The original columns should still be there
        self.assertIn('Close', df_with_indicators.columns)
        self.assertEqual(len(df_with_indicators), len(self.df))

if __name__ == '__main__':
    unittest.main()
