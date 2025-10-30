# tests/test_statistics.py
import unittest
import pandas as pd
from typing import Dict, List
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.statistics import StatisticalAnalyzer

def create_mock_profiles() -> List[Dict]:
    """Creates a list of mock profile dictionaries for testing."""
    profiles = [
        # Day 0 (Prior Day for Day 1)
        {'session_high': 100, 'session_low': 90, 'vah': 98, 'val': 92, 'poc': 95, 'session_close': 97},

        # Day 1: HOR. Breaks pVAH (98). Closes above pVAH.
        {
            'opening_type': 'HOR', 'session_open': 101,
            'session_high': 105, 'session_low': 97, 'session_close': 102,
            'ib_high': 102, 'ib_low': 100,
            'vah': 104, 'val': 100, 'poc': 102
        },

        # Day 2: HIR. Does NOT break pVAH (104).
        {
            'opening_type': 'HIR', 'session_open': 103,
            'session_high': 103.5, 'session_low': 101, 'session_close': 102,
            'ib_high': 103, 'ib_low': 102,
            'vah': 103, 'val': 101.5, 'poc': 102, 'session_close': 102
        },

        # Day 3: LIR. Breaks pPOC (102).
        {
            'opening_type': 'LIR', 'session_open': 101,
            'session_high': 103, 'session_low': 99, 'session_close': 100,
            'ib_high': 102, 'ib_low': 100,
            'vah': 102, 'val': 100, 'poc': 101, 'session_close': 100
        },

        # Day 4: HOR. Breaks pVAH (102). Closes below pVAH.
         {
            'opening_type': 'HOR', 'session_open': 104,
            'session_high': 106, 'session_low': 101, 'session_close': 101,
            'ib_high': 105, 'ib_low': 103,
            'vah': 105, 'val': 102, 'poc': 104, 'session_close': 101
        },
    ]
    return profiles

class TestStatisticalAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up the analyzer and mock data."""
        self.ticker = "TEST"
        self.analyzer = StatisticalAnalyzer(self.ticker)
        self.mock_profiles = create_mock_profiles()

    def test_calculate_opening_type_statistics(self):
        """Test the main statistics calculation."""
        stats = self.analyzer.calculate_opening_type_statistics(self.mock_profiles)

        self.assertIsInstance(stats, dict)
        self.assertIn('HOR', stats)
        self.assertIn('HIR', stats)
        self.assertIn('LIR', stats)
        self.assertNotIn('Unknown', stats)

        # --- Assert HOR stats (2 instances) ---
        hor_stats = stats['HOR']
        self.assertEqual(hor_stats['count'], 2)
        # 1 of 2 closed above pVAH
        self.assertAlmostEqual(hor_stats['stats']['close_above']['pVAH'], 50.0)
        # 2 of 2 closed above pPOC
        self.assertAlmostEqual(hor_stats['stats']['close_above']['pPOC'], 100.0)
        # 2 of 2 broke pVAH
        self.assertAlmostEqual(hor_stats['stats']['broken_during_rth']['pVAH'], 100.0)

        # --- Assert HIR stats (1 instance) ---
        hir_stats = stats['HIR']
        self.assertEqual(hir_stats['count'], 1)
        # 0 of 1 closed above pVAH (using prior day of HOR)
        self.assertAlmostEqual(hir_stats['stats']['close_above']['pVAH'], 0.0)
        # 1 of 1 broke IBH
        self.assertAlmostEqual(hir_stats['stats']['broken_during_rth']['IBH'], 100.0)

        # --- Assert LIR stats (1 instance) ---
        lir_stats = stats['LIR']
        self.assertEqual(lir_stats['count'], 1)
        # 0 of 1 closed above pCL
        self.assertAlmostEqual(lir_stats['stats']['close_above']['pCL'], 0.0)
        # 1 of 1 broke pPOC
        self.assertAlmostEqual(lir_stats['stats']['broken_during_rth']['pPOC'], 100.0)

    def test_get_probabilities_for_day(self):
        """Test the flattening of statistics for ML consumption."""
        stats = self.analyzer.calculate_opening_type_statistics(self.mock_profiles)

        hor_probs = self.analyzer.get_probabilities_for_day('HOR', stats)

        self.assertIsInstance(hor_probs, dict)
        self.assertIn('stat_close_above_pVAH', hor_probs)
        self.assertIn('stat_broken_during_rth_pPOC', hor_probs)
        self.assertAlmostEqual(hor_probs['stat_close_above_pVAH'], 50.0)

if __name__ == '__main__':
    unittest.main()
