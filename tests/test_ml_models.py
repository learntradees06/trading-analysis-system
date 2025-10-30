# tests/test_ml_models.py
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml_models import MLPredictor
from src.statistics import StatisticalAnalyzer

TEST_MODELS_DIR = Path("./test_models")

def create_mock_ml_profiles():
    """Creates a list of more complete mock profile dictionaries for ML feature testing."""
    profiles = [
        # Day 0 (Prior Day for Day 1)
        {'date': pd.Timestamp('2023-01-01'), 'session_high': 100, 'session_low': 90, 'vah': 98, 'val': 92, 'poc': 95, 'session_close': 97, 'ib_high': 96, 'ib_low': 91},
        # Day 1
        {'date': pd.Timestamp('2023-01-02'), 'opening_type': 'HOR', 'session_open': 101, 'session_high': 105, 'session_low': 97, 'session_close': 102, 'ib_high': 102, 'ib_low': 100, 'vah': 104, 'val': 100, 'poc': 102},
        # Day 2
        {'date': pd.Timestamp('2023-01-03'), 'opening_type': 'HIR', 'session_open': 103, 'session_high': 103.5, 'session_low': 101, 'session_close': 102, 'ib_high': 103, 'ib_low': 102, 'vah': 103, 'val': 101.5, 'poc': 102},
        # Day 3
        {'date': pd.Timestamp('2023-01-04'), 'opening_type': 'LIR', 'session_open': 101, 'session_high': 103, 'session_low': 99, 'session_close': 100, 'ib_high': 102, 'ib_low': 100, 'vah': 102, 'val': 100, 'poc': 101},
        # Day 4
        {'date': pd.Timestamp('2023-01-05'), 'opening_type': 'HOR', 'session_open': 104, 'session_high': 106, 'session_low': 101, 'session_close': 101, 'ib_high': 105, 'ib_low': 103, 'vah': 105, 'val': 102, 'poc': 104},
    ]
    # Add calculated fields
    for p in profiles:
        if 'ib_high' in p and 'ib_low' in p:
            p['ib_range'] = p['ib_high'] - p['ib_low']
        if 'vah' in p and 'val' in p:
            p['va_width'] = p['vah'] - p['val']
    return profiles

def create_mock_technicals(dates):
    """Creates mock technical indicator DataFrames."""
    daily_tech = pd.DataFrame({
        'ATR': [2.5] * len(dates),
        'RSI': np.linspace(40, 60, len(dates)),
        'ADX': np.linspace(20, 30, len(dates)),
        'Volume_pct_change': [0.1] * len(dates)
    }, index=pd.to_datetime(dates))

    hourly_tech = pd.DataFrame({
        'RSI': np.linspace(45, 55, len(dates)),
        'ADX': np.linspace(22, 28, len(dates)),
        'EMA_21': np.linspace(100, 110, len(dates))
    }, index=pd.to_datetime(dates))

    return daily_tech, hourly_tech

class TestMLPredictor(unittest.TestCase):

    def setUp(self):
        """Set up the MLPredictor and mock data."""
        self.ticker = "TEST_ML"
        TEST_MODELS_DIR.mkdir(exist_ok=True)
        self.predictor = MLPredictor(self.ticker, TEST_MODELS_DIR)

        self.profiles = create_mock_ml_profiles()
        self.stats_analyzer = StatisticalAnalyzer(self.ticker)
        self.all_stats = self.stats_analyzer.calculate_opening_type_statistics(self.profiles)

    def tearDown(self):
        """Remove the test models directory."""
        if TEST_MODELS_DIR.exists():
            shutil.rmtree(TEST_MODELS_DIR)

    def test_create_features(self):
        """Test the feature creation process."""
        dates = [p['date'] for p in self.profiles]
        daily_tech, hourly_tech = create_mock_technicals(dates)

        features_df = self.predictor.create_features(self.profiles, daily_tech, hourly_tech, self.all_stats)

        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertFalse(features_df.empty)
        self.assertIn('target_opening_type', features_df.columns)
        self.assertIn('target_opening_type_encoded', features_df.columns)
        self.assertIn('poc_migration_norm', features_df.columns)
        self.assertIn('stat_close_above_pVAH', features_df.columns) # Check for stat features

        # We have 5 profiles, so we can create features for profiles 1, 2, and 3 (needs prior and next)
        self.assertEqual(len(features_df), 3)

    def test_train_and_predict_flow(self):
        """Test the model training, saving, loading, and prediction flow."""
        # 1. Create a dataset with all four classes represented
        feature_data = {
            'daily_rsi': [
                20, 22, 24, 26, 28,  # LOR (low)
                80, 82, 84, 86, 88,  # HOR (high)
                40, 42, 44, 46, 48,  # LIR (mid-low)
                60, 62, 64, 66, 68   # HIR (mid-high)
            ],
            'target_opening_type': [
                'LOR', 'LOR', 'LOR', 'LOR', 'LOR',
                'HOR', 'HOR', 'HOR', 'HOR', 'HOR',
                'LIR', 'LIR', 'LIR', 'LIR', 'LIR',
                'HIR', 'HIR', 'HIR', 'HIR', 'HIR'
            ]
        }
        features_df = pd.DataFrame(feature_data)
        features_df['target_opening_type_encoded'] = self.predictor.encoder.transform(features_df['target_opening_type'])
        self.predictor.feature_cols = ['daily_rsi']

        # 2. Train the model
        self.predictor.train_model(features_df)
        self.assertTrue(self.predictor.is_trained)
        self.assertIsNotNone(self.predictor.model)

        # 3. Check if model files were saved
        self.assertTrue((TEST_MODELS_DIR / f"{self.ticker}_opening_type_model.pkl").exists())
        self.assertTrue((TEST_MODELS_DIR / f"{self.ticker}_opening_type_scaler.pkl").exists())

        # 4. Load the model into a new predictor instance
        new_predictor = MLPredictor(self.ticker, TEST_MODELS_DIR)
        loaded = new_predictor.load_model()
        self.assertTrue(loaded)
        self.assertTrue(new_predictor.is_trained)

        # 5. Predict all four classes
        test_cases = {
            'LOR': 15, # low RSI
            'HOR': 85, # high RSI
            'LIR': 45, # mid-low RSI
            'HIR': 65  # mid-high RSI
        }
        for expected_type, rsi_value in test_cases.items():
            features = pd.DataFrame([{'daily_rsi': rsi_value}])
            prediction = new_predictor.predict(features)
            self.assertEqual(prediction['predicted_opening_type'], expected_type)

if __name__ == '__main__':
    unittest.main()
