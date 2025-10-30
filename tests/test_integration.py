# tests/test_integration.py
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import pytz

# Add src and root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from main import TradingSystem

def create_mock_ohlcv_data(days, base_price=100):
    """Creates a mock OHLCV DataFrame."""
    dates = pd.to_datetime([datetime(2023, 1, 1) + pd.Timedelta(days=i) for i in range(days)], utc=True)
    data = {
        'Open': [base_price + i for i in range(days)],
        'High': [base_price + i + 2 for i in range(days)],
        'Low': [base_price + i - 1 for i in range(days)],
        'Close': [base_price + i + 1 for i in range(days)],
        'Volume': [1000 * (i + 1) for i in range(days)]
    }
    return pd.DataFrame(data, index=dates)

class TestIntegration(unittest.TestCase):

    @patch('main.DataManager')
    @patch('main.MLPredictor')
    def test_run_single_ticker_analysis_workflow(self, MockMLPredictor, MockDataManager):
        """
        Integration test for the main analysis workflow (_run_single_ticker_analysis).
        Mocks DataManager and MLPredictor to test the data processing pipeline.
        """
        # --- 1. Setup Mocks ---

        # Mock DataManager instance and its fetch_data method
        mock_dm_instance = MockDataManager.return_value
        mock_dm_instance.fetch_data.side_effect = lambda tf, days_back: create_mock_ohlcv_data(days_back)

        # Mock MLPredictor instance and its predict method
        mock_ml_instance = MockMLPredictor.return_value
        mock_ml_instance.load_model.return_value = True
        mock_ml_instance.predict.return_value = {
            'predicted_opening_type': 'HOR',
            'confidence': 0.85,
            'probabilities': {'HOR': 0.85, 'HIR': 0.10, 'LIR': 0.03, 'LOR': 0.02}
        }
        # Mock the feature creation to return a non-empty DataFrame
        mock_ml_instance.create_prediction_features.return_value = pd.DataFrame([{'dummy': 1}])

        # --- 2. Initialize TradingSystem and Run Analysis ---

        system = TradingSystem()
        system.ticker = "MOCKTICKER"

        # This is the core function we are testing
        result = system._run_single_ticker_analysis("MOCKTICKER", generate_report=True)

        # --- 3. Assertions ---

        # Check that the main components were called
        mock_dm_instance.fetch_data.assert_called()
        mock_ml_instance.predict.assert_called_once()

        # Check the structure of the output
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)
        self.assertIn("signal", result)
        self.assertIn("current_profile", result)
        self.assertIn("daily_with_indicators", result)
        self.assertIn("sr_analysis", result)
        self.assertIn("ml_predictions", result)
        self.assertIn("statistics", result)

        # Check the content of some key outputs
        # Verify the ML prediction was passed through correctly
        self.assertEqual(result['ml_predictions']['predicted_opening_type'], 'HOR')

        # Verify that indicators were calculated
        self.assertIn('RSI', result['daily_with_indicators'].columns)
        self.assertIn('ATR', result['daily_with_indicators'].columns)

        # Verify that the signal was generated
        signal = result['signal']
        self.assertIsInstance(signal, dict)
        self.assertIn('signal', signal)
        self.assertIn('confidence', signal)
        self.assertIn('evidence', signal)

if __name__ == '__main__':
    unittest.main()
