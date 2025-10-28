# test_modules.py
"""
A simple smoke test to ensure all modules can be imported and their main
classes instantiated without raising an exception. This helps catch basic
syntax errors, circular dependencies, or initialization failures.
"""

import sys
from pathlib import Path
import pandas as pd
import traceback  # Import traceback

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import all the main classes from the src directory
from src.config import DEFAULT_TICKER, MODELS_DIR, REPORTS_DIR
from src.data_manager import DataManager
from src.indicators import calculate_all_indicators
from src.market_profile import MarketProfile
from src.sr_levels import SRLevelAnalyzer
from src.statistics import StatisticalAnalyzer
from src.ml_models import MLPredictor
from src.signals import SignalGenerator
from src.reporting import ReportGenerator
from src.notifications import NotificationManager

def run_smoke_tests():
    """
    Instantiates each class and calls a basic method to ensure they load correctly.
    """
    print("--- Starting Smoke Tests ---")

    ticker = DEFAULT_TICKER
    test_passed = True

    try:
        print("\n[1] Testing DataManager...")
        data_manager = DataManager(ticker)
        # Fetch a small amount of data to test functionality
        df = data_manager.fetch_data('1d', days_back=10)
        assert not df.empty, "DataManager failed to fetch data."
        print("✅ DataManager OK")
    except Exception as e:
        print(f"❌ DataManager FAILED: {e}")
        traceback.print_exc()  # Print the full traceback
        test_passed = False

    try:
        print("\n[2] Testing MarketProfile...")
        # Requires settings, which are typically handled by the TradingSystem class
        from src.config import INSTRUMENT_SETTINGS
        settings = INSTRUMENT_SETTINGS.get(ticker, {})
        market_profile = MarketProfile(ticker, settings.get('tick_size', 0.01))
        # This class needs data to do anything useful, so we just check instantiation
        print("✅ MarketProfile OK")
    except Exception as e:
        print(f"❌ MarketProfile FAILED: {e}")
        traceback.print_exc()
        test_passed = False

    try:
        print("\n[3] Testing SRLevelAnalyzer...")
        sr_analyzer = SRLevelAnalyzer(ticker, 0.01)
        print("✅ SRLevelAnalyzer OK")
    except Exception as e:
        print(f"❌ SRLevelAnalyzer FAILED: {e}")
        traceback.print_exc()
        test_passed = False

    try:
        print("\n[4] Testing StatisticalAnalyzer...")
        stats_analyzer = StatisticalAnalyzer(ticker)
        print("✅ StatisticalAnalyzer OK")
    except Exception as e:
        print(f"❌ StatisticalAnalyzer FAILED: {e}")
        traceback.print_exc()
        test_passed = False

    try:
        print("\n[5] Testing MLPredictor...")
        ml_predictor = MLPredictor(ticker, MODELS_DIR)
        assert ml_predictor.models_dir == MODELS_DIR
        print("✅ MLPredictor OK")
    except Exception as e:
        print(f"❌ MLPredictor FAILED: {e}")
        traceback.print_exc()
        test_passed = False

    try:
        print("\n[6] Testing SignalGenerator...")
        signal_generator = SignalGenerator(ticker)
        print("✅ SignalGenerator OK")
    except Exception as e:
        print(f"❌ SignalGenerator FAILED: {e}")
        traceback.print_exc()
        test_passed = False

    try:
        print("\n[7] Testing ReportGenerator...")
        report_generator = ReportGenerator(ticker, REPORTS_DIR)
        assert report_generator.reports_dir == REPORTS_DIR
        print("✅ ReportGenerator OK")
    except Exception as e:
        print(f"❌ ReportGenerator FAILED: {e}")
        traceback.print_exc()
        test_passed = False

    try:
        print("\n[8] Testing NotificationManager...")
        notification_manager = NotificationManager(webhook_url=None)
        # Test sending a message (it should gracefully fail without a URL)
        result = notification_manager.send_to_discord("Test message")
        assert not result, "NotificationManager should fail without a URL."
        print("✅ NotificationManager OK")
    except Exception as e:
        print(f"❌ NotificationManager FAILED: {e}")
        traceback.print_exc()
        test_passed = False

    # Final Summary
    print("\n--- Smoke Tests Complete ---")
    if test_passed:
        print("🎉 All modules loaded and instantiated successfully!")
    else:
        print("🔥 Some modules failed to load. Please check the errors above.")
        # Exit with a non-zero code to indicate failure in automated environments
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_tests()
