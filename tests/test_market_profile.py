# tests/test_market_profile.py
import unittest
import pandas as pd
from datetime import datetime
import pytz
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market_profile import MarketProfile

def create_intraday_dataframe(session_date):
    """Creates a sample intraday DataFrame for testing a single session."""
    tz = pytz.UTC
    start_time = datetime.combine(session_date, datetime.min.time(), tzinfo=tz)

    # Create 30-minute intervals for a typical RTH session
    timestamps = pd.to_datetime([
        start_time.replace(hour=13, minute=30),  # 08:30 CT
        start_time.replace(hour=14, minute=0),   # 09:00 CT
        start_time.replace(hour=14, minute=30),  # 09:30 CT
        start_time.replace(hour=15, minute=0),   # 10:00 CT
        start_time.replace(hour=15, minute=30),  # 10:30 CT
    ])

    data = {
        'Open':  [100, 101, 102, 103, 104],
        'High':  [102, 103, 104, 105, 106],
        'Low':   [99,  100, 101, 102, 103],
        'Close': [101, 102, 103, 104, 105],
        'Volume':[1000, 1200, 1100, 1300, 1400]
    }
    df = pd.DataFrame(data, index=timestamps)
    return df

class TestMarketProfile(unittest.TestCase):

    def setUp(self):
        """Set up a MarketProfile instance and sample data."""
        self.ticker = "TEST"
        self.tick_size = 0.25
        self.mp = MarketProfile(self.ticker, self.tick_size)

        self.session_date = datetime(2023, 1, 10, tzinfo=pytz.UTC)
        self.df_intraday = create_intraday_dataframe(self.session_date.date())

    def test_calculate_tpo_profile(self):
        """Test the TPO profile calculation for a single session."""
        profile = self.mp.calculate_tpo_profile(self.df_intraday, self.session_date)

        self.assertIsInstance(profile, dict)
        self.assertTrue(profile)

        # Check key metrics
        self.assertIn('poc', profile)
        self.assertIn('vah', profile)
        self.assertIn('val', profile)
        self.assertIn('ib_high', profile)
        self.assertIn('ib_low', profile)

        # Verify POC is a plausible value
        self.assertGreater(profile['poc'], 0)
        # VAH should be greater than or equal to VAL
        self.assertGreaterEqual(profile['vah'], profile['val'])
        # POC should be within the Value Area
        self.assertTrue(profile['val'] <= profile['poc'] <= profile['vah'])

    def test_classify_opening_type(self):
        """Test the opening type classification logic."""
        prior_profile = {
            'session_high': 100,
            'session_low': 90,
            'session_close': 95
        }

        # Test Higher Outside Range (HOR)
        self.assertEqual(self.mp.classify_opening_type(101, prior_profile), 'HOR')
        # Test Higher Inside Range (HIR)
        self.assertEqual(self.mp.classify_opening_type(98, prior_profile), 'HIR')
        # Test Lower Inside Range (LIR)
        self.assertEqual(self.mp.classify_opening_type(92, prior_profile), 'LIR')
        # Test Lower Outside Range (LOR)
        self.assertEqual(self.mp.classify_opening_type(89, prior_profile), 'LOR')

    def test_empty_data_tpo_profile(self):
        """Test that TPO calculation handles empty DataFrames gracefully."""
        empty_df = pd.DataFrame()
        profile = self.mp.calculate_tpo_profile(empty_df, self.session_date)
        self.assertEqual(profile, {})

    def test_unknown_opening_type(self):
        """Test that opening type is 'Unknown' with incomplete prior day data."""
        incomplete_profile = {'session_high': 100} # Missing low and close
        self.assertEqual(self.mp.classify_opening_type(105, incomplete_profile), 'Unknown')

if __name__ == '__main__':
    unittest.main()
