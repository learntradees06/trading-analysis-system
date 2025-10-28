# main.py
"""Main Application - Interactive Command-Line Interface"""

import os
import sys
from pathlib import Path
import pandas as pd
import pytz
from datetime import datetime
import time
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import INSTRUMENT_SETTINGS, DEFAULT_TICKER, DISCORD_WEBHOOK_URL, MODELS_DIR, REPORTS_DIR
from src.data_manager import DataManager, get_cache_statistics
from src.indicators import calculate_all_indicators
from src.market_profile import MarketProfile
from src.sr_levels import SRLevelAnalyzer
from src.statistics import StatisticalAnalyzer
from src.ml_models import MLPredictor
from src.signals import SignalGenerator
from src.reporting import ReportGenerator
from src.notifications import NotificationManager
from src.watchlists import WATCHLISTS
from src.portfolio_manager import PortfolioManager
from src.dashboard import Dashboard

class TradingSystem:
    def __init__(self):
        """Initialize the Trading System"""
        self.ticker = DEFAULT_TICKER
        self.notifier = NotificationManager(DISCORD_WEBHOOK_URL)
        self.dashboard = Dashboard()
        self.portfolio_manager = PortfolioManager(self._run_single_ticker_analysis)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize components that depend on the current ticker."""
        print(f"\n🔄 Initializing components for {self.ticker}...")
        if self.ticker in INSTRUMENT_SETTINGS:
            self.settings = INSTRUMENT_SETTINGS[self.ticker]
        else:
            print(f"⚠️  Warning: No specific settings for {self.ticker}, using defaults")
            self.settings = INSTRUMENT_SETTINGS['DEFAULT']
        print(f"✅ Components initialized successfully for {self.ticker}")

    def display_menu(self):
        """Display the main menu."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 80)
        print("   ADVANCED TRADING ANALYSIS SYSTEM")
        print("=" * 80)
        print("\n--- Multi-Ticker Analysis ---")
        print("[1] Analyze Watchlist")
        print("[2] Start Multi-Ticker Scanner")
        print("\n--- Single-Ticker Analysis ---")
        print(f"   (Current Ticker: {self.ticker})")
        print("[3] Generate Trading Plan for Current Ticker")
        print("[4] View Historical Statistics for Current Ticker")
        print("[5] Train ML Models for Current Ticker")
        print("\n--- System ---")
        print("[6] Set New Ticker")
        print("[7] Configure Notifications")
        print("[8] View Cached Data Summary")
        print("[9] Exit")

    def _run_single_ticker_analysis(self, ticker: str, generate_report: bool = False) -> Dict:
        """
        The core analysis logic for a single ticker.
        This function is designed to be called by the PortfolioManager's threads.
        If generate_report is True, it returns all data artifacts needed for the report.
        """
        # (Same component initialization as before)
        data_manager = DataManager(ticker)
        if ticker in INSTRUMENT_SETTINGS:
            settings = INSTRUMENT_SETTINGS[ticker]
        else:
            settings = INSTRUMENT_SETTINGS['DEFAULT']
        market_profile = MarketProfile(ticker, settings['tick_size'])
        sr_analyzer = SRLevelAnalyzer(ticker, settings['tick_size'])
        stats_analyzer = StatisticalAnalyzer(ticker)
        ml_predictor = MLPredictor(ticker, MODELS_DIR)
        signal_generator = SignalGenerator(ticker)

        try:
            # (Same data fetching and processing logic as before)
            daily_data = data_manager.fetch_data('1d', days_back=100)
            hourly_data = data_manager.fetch_data('4h', days_back=100)
            thirty_min_data = data_manager.fetch_data('30m', days_back=100)

            if daily_data.empty or thirty_min_data.empty: return {"error": "Insufficient data."}

            daily_with_indicators = calculate_all_indicators(daily_data)

            profiles = []
            intraday_data_dict = {}
            unique_dates_30m = {ts.date() for ts in thirty_min_data.index}
            unique_dates_daily = {ts.date() for ts in daily_data.index}
            relevant_dates = sorted(list(unique_dates_30m.intersection(unique_dates_daily)))

            for date_obj in relevant_dates:
                day_data = thirty_min_data[thirty_min_data.index.date == date_obj]
                matching_daily_timestamp = daily_data[pd.to_datetime(daily_data.index, utc=True).date == date_obj].index[0]
                if not day_data.empty:
                    rth_data = data_manager.get_rth_data(day_data)
                    if not rth_data.empty:
                        profile = market_profile.calculate_tpo_profile(rth_data, matching_daily_timestamp)
                        if profile:
                            if profiles:
                                profile['opening_type'] = market_profile.classify_opening_type(
                                    rth_data['Open'].iloc[0], profiles[-1])
                            profiles.append(profile)
                            intraday_data_dict[matching_daily_timestamp] = rth_data

            statistics = stats_analyzer.calculate_opening_type_statistics(profiles, intraday_data_dict)
            sr_analysis = sr_analyzer.analyze_all_sr_levels({'1d': daily_data, '4h': hourly_data, '30m': thirty_min_data})
            current_profile = profiles[-1] if profiles else {}
            ml_features = ml_predictor.create_fusion_features(profiles, daily_with_indicators, sr_analysis)

            # (Same ML prediction logic as before)
            models_to_check = ['target_broke_ibh', 'target_broke_ibl', 'target_next_day_direction']
            missing_models = [target for target in models_to_check if not ml_predictor.model_exists(target)]
            if missing_models:
                ml_predictions = {}
            else:
                all(ml_predictor.load_model(target) for target in models_to_check)
                ml_predictions = ml_predictor.predict(ml_features) if not ml_features.empty else {}

            signal = signal_generator.generate_signal(
                current_profile, daily_with_indicators.iloc[-1], sr_analysis, ml_predictions, statistics)

            # Return different payloads based on the context
            if generate_report:
                return {
                    "signal": signal, "current_profile": current_profile,
                    "daily_with_indicators": daily_with_indicators, "sr_analysis": sr_analysis,
                    "ml_predictions": ml_predictions, "statistics": statistics,
                    "daily_data": daily_data
                }
            else:
                return {
                    "signal": signal,
                    "technicals": {
                        "RSI": daily_with_indicators.iloc[-1].get('RSI', 0),
                        "ADX": daily_with_indicators.iloc[-1].get('ADX', 0)
                    }
                }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed: {e}"}

    def analyze_watchlist_and_display(self):
        """Analyzes a full watchlist and displays a summary dashboard."""
        print("\n--- Analyze Watchlist ---")
        for i, (name, data) in enumerate(WATCHLISTS.items(), 1):
            print(f"[{i}] {name} ({data['description']})")

        choice = input("\nSelect a watchlist: ").strip()
        if choice.isdigit():
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(WATCHLISTS):
                watchlist_name = list(WATCHLISTS.keys())[choice_idx]
                tickers = WATCHLISTS[watchlist_name]['tickers']

                print(f"\nAnalyzing '{watchlist_name}' watchlist...")
                results = self.portfolio_manager.analyze_watchlist(tickers)

                print("\n--- Analysis Complete ---")
                self.dashboard.display_summary_table(results)
                return
        print("\n⚠️ Invalid selection.")

    def start_multi_ticker_scanner(self):
        """Starts a continuous scanner for a selected watchlist."""
        print("\n--- Multi-Ticker Live Scanner ---")
        for i, (name, data) in enumerate(WATCHLISTS.items(), 1):
            print(f"[{i}] {name} ({data['description']})")

        choice = input("\nSelect a watchlist to scan: ").strip()
        if not choice.isdigit():
            print("\n⚠️ Invalid selection."); return

        choice_idx = int(choice) - 1
        if not (0 <= choice_idx < len(WATCHLISTS)):
            print("\n⚠️ Invalid selection."); return

        watchlist_name = list(WATCHLISTS.keys())[choice_idx]
        tickers = WATCHLISTS[watchlist_name]['tickers']
        refresh_rate = input("Refresh rate in seconds (default 300): ").strip()
        refresh_rate = int(refresh_rate) if refresh_rate.isdigit() else 300

        print(f"\n🚀 Starting live scanner for '{watchlist_name}' (Refresh: {refresh_rate}s). Press Ctrl+C to stop.")

        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"--- Live Scanner: {watchlist_name} | {datetime.now().strftime('%H:%M:%S')} ---")
                results = self.portfolio_manager.analyze_watchlist(tickers)
                self.dashboard.display_summary_table(results)

                # Check for and display alerts
                print("\n--- High Confidence Alerts ---")
                alerts_found = False
                for ticker, result in results.items():
                    if "signal" in result and result["signal"].get("confidence") == "HIGH":
                        self.dashboard.display_alert(ticker, result["signal"])
                        self.notifier.send_signal_alert(ticker, result["signal"])
                        alerts_found = True

                if not alerts_found:
                    print("None")

                print(f"\nNext scan in {refresh_rate} seconds...")
                time.sleep(refresh_rate)
        except KeyboardInterrupt:
            print("\n\n🛑 Scanner stopped.")

    def generate_single_ticker_plan(self):
        """Generate a trading plan and detailed HTML report for the current ticker."""
        print(f"\n--- Generating Plan for {self.ticker} ---")
        report_generator = ReportGenerator(self.ticker, REPORTS_DIR)

        # Call the analysis function with the flag to get all data artifacts
        analysis_result = self._run_single_ticker_analysis(self.ticker, generate_report=True)

        if "error" in analysis_result:
            print(f"\n❌ Error generating trading plan: {analysis_result['error']}")
            return

        # Display summary in console
        self._display_signal_summary(analysis_result['signal'])

        # Generate and save the detailed HTML report
        print("\n📝 Creating detailed HTML report...")
        try:
            report_path = report_generator.generate_report(
                market_profile=analysis_result['current_profile'],
                technical_data=analysis_result['daily_with_indicators'],
                sr_analysis=analysis_result['sr_analysis'],
                ml_predictions=analysis_result['ml_predictions'],
                signal=analysis_result['signal'],
                statistics=analysis_result['statistics'],
                price_data=analysis_result['daily_data']
            )
            print(f"✅ Report saved to: {report_path}")
        except Exception as e:
            print(f"\n❌ Error generating report: {e}")


    def set_new_ticker(self):
        """Set a new ticker for single-ticker analysis."""
        print("\n--- Set New Ticker ---")
        new_ticker = input(f"Enter new ticker symbol (current: {self.ticker}): ").strip().upper()
        if new_ticker:
            self.ticker = new_ticker
            self._initialize_components()
            print(f"✅ Ticker changed to {self.ticker}")
        else:
            print("⚠️ No ticker entered.")

    def configure_notifications(self):
        """Configure Discord notifications."""
        # This function can remain largely the same.
        pass

    def _display_signal_summary(self, signal: Dict):
        """Display signal summary in console"""
        print("\n" + "=" * 60)
        print("   TRADING SIGNAL SUMMARY")
        print("=" * 60)
        signal_type = signal.get('signal', 'NEUTRAL')
        score = signal.get('score', 0)
        confidence = signal.get('confidence', 'LOW')

        if 'LONG' in signal_type: print(f"🟢 Signal: {signal_type}")
        elif 'SHORT' in signal_type: print(f"🔴 Signal: {signal_type}")
        else: print(f"🟡 Signal: {signal_type}")

        print(f"📊 Score: {score:.1f}/100")
        print(f"🎯 Confidence: {confidence}")
        print("\n📋 Key Evidence:")
        for evidence in signal.get('evidence', [])[:5]:
            print(f"  • {evidence}")
        if 'component_scores' in signal and signal['component_scores']:
            print("\n🎯 Component Scores:")
            for component, score in signal['component_scores'].items():
                print(f"  • {component.replace('_', ' ').title()}: {score:.1f}")

    def view_historical_statistics(self):
        """View historical profile statistics for the current ticker."""
        print(f"\n--- Historical Statistics for {self.ticker} ---")
        try:
            data_manager = DataManager(self.ticker)
            if self.ticker in INSTRUMENT_SETTINGS:
                settings = INSTRUMENT_SETTINGS[self.ticker]
            else:
                settings = INSTRUMENT_SETTINGS['DEFAULT']
            market_profile = MarketProfile(self.ticker, settings['tick_size'])
            stats_analyzer = StatisticalAnalyzer(self.ticker)

            daily_data = data_manager.fetch_data('1d', days_back=100)
            thirty_min_data = data_manager.fetch_data('30m', days_back=100)

            if daily_data.empty or thirty_min_data.empty:
                print("\n❌ Insufficient data for statistics.")
                return

            profiles = []
            intraday_data_dict = {}
            unique_dates_30m = {ts.date() for ts in thirty_min_data.index}
            unique_dates_daily = {ts.date() for ts in daily_data.index}
            relevant_dates = sorted(list(unique_dates_30m.intersection(unique_dates_daily)))

            for date_obj in relevant_dates:
                day_data = thirty_min_data[thirty_min_data.index.date == date_obj]
                matching_daily_timestamp = daily_data[pd.to_datetime(daily_data.index, utc=True).date == date_obj].index[0]
                if not day_data.empty:
                    rth_data = data_manager.get_rth_data(day_data)
                    if not rth_data.empty:
                        profile = market_profile.calculate_tpo_profile(rth_data, matching_daily_timestamp)
                        if profile:
                            if profiles:
                                profile['opening_type'] = market_profile.classify_opening_type(
                                    rth_data['Open'].iloc[0], profiles[-1]
                                )
                            profiles.append(profile)
                            intraday_data_dict[matching_daily_timestamp] = rth_data

            statistics = stats_analyzer.calculate_opening_type_statistics(profiles, intraday_data_dict)
            print(stats_analyzer.format_statistics_table(statistics))

        except Exception as e:
            print(f"\n❌ Error calculating statistics: {e}")

    def train_ml_models_for_ticker(self):
        """Train or update ML models for the current ticker."""
        print(f"\n--- Training ML Models for {self.ticker} ---")
        try:
            data_manager = DataManager(self.ticker)
            if self.ticker in INSTRUMENT_SETTINGS:
                settings = INSTRUMENT_SETTINGS[self.ticker]
            else:
                settings = INSTRUMENT_SETTINGS['DEFAULT']
            market_profile = MarketProfile(self.ticker, settings['tick_size'])
            ml_predictor = MLPredictor(self.ticker, MODELS_DIR)

            print("📥 Fetching historical data (200 days)...")
            daily_data = data_manager.fetch_data('1d', days_back=200)
            thirty_min_data = data_manager.fetch_data('30m', days_back=200)

            if daily_data.empty or thirty_min_data.empty:
                print("\n❌ Insufficient data for training.")
                return

            print("📈 Calculating indicators...")
            daily_with_indicators = calculate_all_indicators(daily_data)

            print("📊 Generating market profiles...")
            profiles = []
            unique_dates_30m = {ts.date() for ts in thirty_min_data.index}
            unique_dates_daily = {ts.date() for ts in daily_data.index}
            relevant_dates = sorted(list(unique_dates_30m.intersection(unique_dates_daily)))

            for date_obj in relevant_dates:
                day_data = thirty_min_data[thirty_min_data.index.date == date_obj]
                matching_daily_timestamp = daily_data[pd.to_datetime(daily_data.index, utc=True).date == date_obj].index[0]
                if not day_data.empty:
                    rth_data = data_manager.get_rth_data(day_data)
                    if not rth_data.empty:
                        profile = market_profile.calculate_tpo_profile(rth_data, matching_daily_timestamp)
                        if profile:
                            if profiles:
                                profile['opening_type'] = market_profile.classify_opening_type(
                                    rth_data['Open'].iloc[0], profiles[-1]
                                )
                            profiles.append(profile)

            print("🔧 Creating fusion features...")
            ml_features = ml_predictor.create_fusion_features(profiles, daily_with_indicators, {})

            if ml_features.empty or len(ml_features) < 20:
                print(f"❌ Insufficient features for training ({len(ml_features)} samples).")
                return

            print(f"🎯 Training models with {len(ml_features)} samples...")
            results = ml_predictor.train_models(ml_features)

            print("\n📊 Training Results:")
            for target, result in results.items():
                if result.get('status') == 'success':
                    print(f"  ✅ {target}: Test Score = {result.get('test_score', 0):.2f}, CV Score = {result.get('cv_score_mean', 0):.2f}")
                else:
                    print(f"  ❌ {target}: {result.get('status', 'Failed')}")

            print("\n✅ Models trained and saved successfully!")

        except Exception as e:
            print(f"\n❌ Error training models: {e}")

    def view_cache_summary(self):
        """View summary of cached data for the current ticker."""
        print(f"\n--- Data Cache Summary for {self.ticker} ---")
        try:
            data_manager = DataManager(self.ticker)
            summary = data_manager.get_data_summary()

            if summary.empty:
                print(f"❌ No cached data found for {self.ticker}")
                return

            print(f"{'Timeframe':<12} {'First Date':<20} {'Last Date':<20} {'Rows':<10}")
            print("-" * 62)
            for _, row in summary.iterrows():
                print(f"{row['timeframe']:<12} "
                      f"{row['first_date'].strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{row['last_date'].strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{row['total_rows']:<10}")

            # Display overall stats
            overall_stats = get_cache_statistics()
            print("\n--- Overall Cache Statistics ---")
            print(f"  Database Size: {overall_stats['database_size_mb']:.2f} MB")
            print(f"  Total Tickers: {len(overall_stats['ticker_stats'])}")

        except Exception as e:
            print(f"\n❌ Error viewing cache summary: {e}")

    def run(self):
        """Main application loop."""
        while True:
            self.display_menu()
            choice = input("\nEnter choice (1-9): ").strip()

            if choice == '1': self.analyze_watchlist_and_display()
            elif choice == '2': self.start_multi_ticker_scanner()
            elif choice == '3': self.generate_single_ticker_plan()
            elif choice == '4': self.view_historical_statistics()
            elif choice == '5': self.train_ml_models_for_ticker()
            elif choice == '6': self.set_new_ticker()
            elif choice == '7': self.configure_notifications()
            elif choice == '8': self.view_cache_summary()
            elif choice == '9': print("\n👋 Goodbye!"); break
            else: print("\n⚠️  Invalid choice. Please try again.")

            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        system = TradingSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n👋 System interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
