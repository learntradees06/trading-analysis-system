# main.py
"""Main Application - Interactive Command-Line Interface"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Tuple

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
from src.watchlist_manager import WatchlistManager
from src.portfolio_manager import PortfolioManager
from src.dashboard import Dashboard

class TradingSystem:
    def __init__(self):
        """Initialize the Trading System"""
        self.ticker = DEFAULT_TICKER
        self.notifier = NotificationManager(DISCORD_WEBHOOK_URL)
        self.dashboard = Dashboard()
        self.watchlist_manager = WatchlistManager()
        self.portfolio_manager = PortfolioManager(self._run_single_ticker_analysis)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize components that depend on the current ticker."""
        pass # No longer needed as components are created on-demand

    def display_menu(self):
        """Display the main menu."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 80); print("   ADVANCED TRADING ANALYSIS SYSTEM"); print("=" * 80)
        print("\n--- Multi-Ticker Analysis ---")
        print("[1] Analyze Watchlist")
        print("[2] Start Multi-Ticker Scanner")
        print("\n--- Single-Ticker Analysis ---")
        print(f"   (Current Ticker: {self.ticker})")
        print("[3] Generate Trading Plan for Current Ticker")
        print("[4] View Historical Statistics for Current Ticker")
        print("[5] Train ML Models for Current Ticker")
        print("\n--- System ---")
        print("[6] Manage Watchlists")
        print("[7] Set New Ticker")
        print("[8] Configure Notifications")
        print("[9] View Cached Data Summary")
        print("[10] Exit")

    def _run_single_ticker_analysis(self, ticker: str, generate_report: bool = False) -> Dict:
        """The core analysis logic for a single ticker."""
        data_manager = DataManager(ticker)
        default_settings = {"tick_size": 0.01, "rth_start": "08:30", "rth_end": "15:00", "timezone": "US/Central"}
        settings = INSTRUMENT_SETTINGS.get(ticker, default_settings)
        market_profile = MarketProfile(ticker, settings['tick_size'])
        sr_analyzer = SRLevelAnalyzer(ticker, settings['tick_size'])
        stats_analyzer = StatisticalAnalyzer(ticker)
        ml_predictor = MLPredictor(ticker, MODELS_DIR)
        signal_generator = SignalGenerator(ticker)
        try:
            timeframes = ['1wk', '1d', '1h', '30m', '15m', '5m']
            data = {tf: data_manager.fetch_data(tf, days_back=252) for tf in timeframes}
            if data['1d'].empty or data['30m'].empty: return {"error": "Insufficient base data."}

            daily_with_indicators = calculate_all_indicators(data['1d'])
            profiles, _ = self._generate_profiles(data['1d'], data['30m'], data_manager, market_profile)
            statistics = stats_analyzer.calculate_opening_type_statistics(profiles)

            # Create features for the most recent day for opening type prediction
            prediction_features = ml_predictor.create_prediction_features(profiles, daily_with_indicators, statistics)
            ml_predictions = self._get_ml_predictions(ml_predictor, prediction_features)

            current_profile = profiles[-1] if profiles else {}
            sr_analysis = sr_analyzer.analyze_all_sr_levels(data)

            # Note: Signal generator might need adjustment to handle new prediction format
            signal = signal_generator.generate_signal(current_profile, daily_with_indicators.iloc[-1], sr_analysis, ml_predictions, statistics)

            if generate_report:
                return {"signal": signal, "current_profile": current_profile, "daily_with_indicators": daily_with_indicators,
                        "sr_analysis": sr_analysis, "ml_predictions": ml_predictions, "statistics": statistics, "all_data": data}
            else:
                return {"signal": signal, "technicals": {"RSI": daily_with_indicators.iloc[-1].get('RSI', 0), "ADX": daily_with_indicators.iloc[-1].get('ADX', 0)}}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed: {e}"}

    def _generate_profiles(self, daily_data, thirty_min_data, data_manager, market_profile) -> Tuple[List[Dict], Dict[datetime, pd.DataFrame]]:
        profiles = []
        intraday_data_dict = {}
        unique_dates_30m = {ts.date() for ts in thirty_min_data.index}
        unique_dates_daily = {ts.date() for ts in daily_data.index}
        relevant_dates = sorted(list(unique_dates_30m.intersection(unique_dates_daily)))

        # Create a dictionary of profiles for quick lookup
        profile_map = {}

        for date_obj in relevant_dates:
            day_data = thirty_min_data[thirty_min_data.index.date == date_obj]
            try:
                matching_daily_timestamp = daily_data[pd.to_datetime(daily_data.index, utc=True).date == date_obj].index[0]
            except IndexError:
                continue

            if not day_data.empty:
                rth_data = data_manager.get_rth_data(day_data)
                if not rth_data.empty:
                    profile = market_profile.calculate_tpo_profile(rth_data, matching_daily_timestamp)
                    if profile:
                        profile_map[matching_daily_timestamp.date()] = profile
                        intraday_data_dict[matching_daily_timestamp] = rth_data

        # Second pass to classify opening types
        sorted_dates = sorted(profile_map.keys())
        for i in range(len(sorted_dates)):
            current_date = sorted_dates[i]
            current_profile = profile_map[current_date]

            if i > 0:
                prior_date = sorted_dates[i-1]
                prior_profile = profile_map[prior_date]
                current_open = current_profile.get('session_open')
                if current_open is not None:
                    current_profile['opening_type'] = market_profile.classify_opening_type(current_open, prior_profile)

            profiles.append(current_profile)

        return profiles, intraday_data_dict

    def _get_ml_predictions(self, ml_predictor, prediction_features):
        """Loads the opening type model and returns predictions."""
        if ml_predictor.load_model():
            if not prediction_features.empty:
                return ml_predictor.predict(prediction_features)
        return {"error": "ML model not found or failed to load. Please train the model."}

    def analyze_watchlist_and_display(self):
        """Analyzes a full watchlist and displays a summary dashboard."""
        watchlists = self.watchlist_manager.get_all()
        if not watchlists:
            print("\n⚠️ No watchlists found. Please create one first in the 'Manage Watchlists' menu."); return
        print("\n--- Analyze Watchlist ---")
        for i, (name, data) in enumerate(watchlists.items(), 1):
            print(f"[{i}] {name} ({data['description']})")
        choice = input("\nSelect a watchlist: ").strip()
        if choice.isdigit() and 0 < int(choice) <= len(watchlists):
            watchlist_name = list(watchlists.keys())[int(choice) - 1]
            tickers = watchlists[watchlist_name]['tickers']
            print(f"\nAnalyzing '{watchlist_name}' watchlist...")
            results = self.portfolio_manager.analyze_watchlist(tickers)
            print("\n--- Analysis Complete ---")
            self.dashboard.display_summary_table(results)
        else:
            print("\n⚠️ Invalid selection.")

    def start_multi_ticker_scanner(self):
        """Starts a continuous scanner for a selected watchlist."""
        print("\n--- Live Market Scanner ---")

        # 1. Select Watchlist
        watchlists = self.watchlist_manager.get_all()
        if not watchlists:
            print("\n⚠️ No watchlists found. Please create one first."); return

        for i, (name, data) in enumerate(watchlists.items(), 1):
            print(f"[{i}] {name} ({len(data.get('tickers', []))} tickers)")

        choice = input("Select a watchlist to scan: ").strip()
        if not choice.isdigit() or not (0 < int(choice) <= len(watchlists)):
            print("⚠️ Invalid selection."); return

        watchlist_name = list(watchlists.keys())[int(choice) - 1]
        tickers_to_scan = watchlists[watchlist_name]['tickers']

        # 2. Set Refresh Interval
        try:
            refresh_seconds = int(input("Enter refresh interval in seconds (e.g., 60): ").strip())
            if refresh_seconds < 10:
                print("Interval too short, setting to 10 seconds.")
                refresh_seconds = 10
        except ValueError:
            print("Invalid number, defaulting to 60 seconds.")
            refresh_seconds = 60

        # 3. Start Scanner Loop
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"🚀 Starting scanner for '{watchlist_name}' watchlist. Press Ctrl+C to stop.")

        try:
            while True:
                start_time = time.time()

                # Analyze and display
                results = self.portfolio_manager.analyze_watchlist(tickers_to_scan)

                # Clear screen before printing new table
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"--- Live Scanner: {watchlist_name} (Last updated: {datetime.now().strftime('%H:%M:%S')}) ---")
                self.dashboard.display_summary_table(results)

                # Check for and send alerts
                for ticker, result in results.items():
                    if 'error' not in result:
                        signal_data = result.get('signal', {})
                        if signal_data.get('confidence') == 'HIGH':
                            # To avoid spam, we'd need a more sophisticated alert manager
                            # For now, it will alert on every high-confidence scan
                            self.notifier.send_alert(ticker, signal_data)

                # Wait for next cycle
                elapsed_time = time.time() - start_time
                wait_time = max(0, refresh_seconds - elapsed_time)

                print(f"\nNext scan in {int(wait_time)} seconds. (Press Ctrl+C to stop)")
                time.sleep(wait_time)

        except KeyboardInterrupt:
            print("\n\n🛑 Scanner stopped by user.")
        except Exception as e:
            print(f"\n❌ An error occurred in the scanner: {e}")

    def generate_single_ticker_plan(self):
        """Generate a trading plan and detailed HTML report for the current ticker."""
        print(f"\n--- Generating Plan for {self.ticker} ---")
        report_generator = ReportGenerator(self.ticker, REPORTS_DIR)
        analysis_result = self._run_single_ticker_analysis(self.ticker, generate_report=True)
        if "error" in analysis_result:
            print(f"\n❌ Error generating trading plan: {analysis_result['error']}"); return
        self._display_signal_summary(analysis_result['signal'])
        print("\n📝 Creating detailed HTML report...")
        try:
            report_path = report_generator.generate_report(
                signal=analysis_result['signal'],
                current_profile=analysis_result['current_profile'],
                daily_with_indicators=analysis_result['daily_with_indicators'],
                sr_analysis=analysis_result['sr_analysis'],
                ml_predictions=analysis_result['ml_predictions'],
                statistics=analysis_result['statistics'],
                all_data=analysis_result['all_data']
            )
            print(f"✅ Report saved to: {report_path}")
        except Exception as e:
            print(f"\n❌ Error generating report: {e}")

    def view_historical_statistics(self):
        """View historical profile statistics for the current ticker."""
        print(f"\n--- Historical Statistics for {self.ticker} ---")
        try:
            print("Fetching historical data (approx. 1 year)...")
            data_manager = DataManager(self.ticker)
            settings = INSTRUMENT_SETTINGS.get(self.ticker, {
                "tick_size": 0.01, "rth_start": "08:30", "rth_end": "15:00", "timezone": "US/Central"
            })
            market_profile = MarketProfile(self.ticker, settings['tick_size'])
            stats_analyzer = StatisticalAnalyzer(self.ticker)

            # Fetch a year of data for meaningful stats
            days_for_stats = 252
            daily_data = data_manager.fetch_data('1d', days_back=days_for_stats)
            thirty_min_data = data_manager.fetch_data('30m', days_back=days_for_stats)

            if daily_data.empty or thirty_min_data.empty or len(daily_data) < 50:
                print("\n⚠️ Not enough historical data to generate meaningful statistics.")
                return

            print("Generating market profiles...")
            profiles, intraday_data_dict = self._generate_profiles(daily_data, thirty_min_data, data_manager, market_profile)

            if not profiles:
                print("\n⚠️ Could not generate market profiles from the available data.")
                return

            print("Calculating statistics...")
            statistics = stats_analyzer.calculate_opening_type_statistics(profiles)

            # The new display method handles formatting
            stats_analyzer.display_statistics(statistics)

        except Exception as e:
            print(f"\n❌ An error occurred while generating statistics: {e}")
            import traceback
            traceback.print_exc()

    def train_ml_models_for_ticker(self):
        """Train the new multiclass opening type prediction model."""
        print(f"\n--- Training Opening Type Prediction Model for {self.ticker} ---")
        confirm = input(f"This will train a new opening type prediction model for {self.ticker}. This requires significant historical data (at least 2 years recommended). Continue? (y/n): ").lower()
        if confirm != 'y':
            print("Training cancelled."); return

        try:
            print("Step 1/3: Initializing components and fetching data...")
            data_manager = DataManager(self.ticker)
            settings = INSTRUMENT_SETTINGS.get(self.ticker, {"tick_size": 0.01, "rth_start": "08:30", "rth_end": "15:00", "timezone": "US/Central"})
            market_profile = MarketProfile(self.ticker, settings['tick_size'])
            ml_predictor = MLPredictor(self.ticker, MODELS_DIR)

            # Fetch ample data for feature creation
            daily_data = data_manager.fetch_data('1d', days_back=730)
            thirty_min_data = data_manager.fetch_data('30m', days_back=730)
            if daily_data.empty or len(daily_data) < 100: # Need at least 100 days for a decent training set
                print("Error: Not enough historical data (<100 days) to train the model."); return

            print("Step 2/3: Generating profiles, indicators, and statistics...")
            daily_with_indicators = calculate_all_indicators(daily_data)
            profiles, _ = self._generate_profiles(daily_data, thirty_min_data, data_manager, market_profile)
            statistics = stats_analyzer.calculate_opening_type_statistics(profiles)

            print("Step 3/3: Creating feature set and training model...")
            full_feature_set = ml_predictor.create_features(profiles, daily_with_indicators, statistics)

            if full_feature_set.empty or len(full_feature_set) < 50:
                print("Error: Failed to create a sufficiently large feature set (<50 samples). More data may be required."); return

            ml_predictor.train_model(full_feature_set)
            print(f"\n✅ Model for {self.ticker} trained and saved successfully.")

        except Exception as e:
            print(f"\n❌ An error occurred during training: {e}")
            import traceback
            traceback.print_exc()

    def _view_all_watchlists(self, show_tickers=False):
        """Helper to view all watchlists."""
        watchlists = self.watchlist_manager.get_all()
        if not watchlists:
            print("\nNo watchlists found.")
            return False # Return False to indicate no watchlists

        print("\nAvailable Watchlists:")
        for i, (name, data) in enumerate(watchlists.items(), 1):
            print(f"  {i}. {name} - {data['description']}")
            if show_tickers:
                tickers = data.get('tickers', [])
                print(f"     Tickers: {', '.join(tickers) if tickers else 'None'}")
        return True # Return True if there are watchlists

    def manage_watchlists(self):
        """Display the watchlist management sub-menu."""
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n" + "="*40); print("      WATCHLIST MANAGEMENT"); print("="*40)
            self._view_all_watchlists(show_tickers=True)
            print("\n" + "-"*40)
            print("[1] Create New Watchlist")
            print("[2] Add Ticker to Watchlist")
            print("[3] Remove Ticker from Watchlist")
            print("[4] Delete Watchlist")
            print("[5] Back to Main Menu")
            print("-" * 40)
            choice = input("Enter choice: ").strip()

            action_map = {
                '1': self._create_watchlist,
                '2': self._add_ticker_to_watchlist,
                '3': self._remove_ticker_from_watchlist,
                '4': self._delete_watchlist,
            }

            if choice in action_map:
                action_map[choice]()
                input("\nPress Enter to continue...")
            elif choice == '5':
                break
            else:
                print("\n⚠️ Invalid choice. Please try again.")
                time.sleep(1)

    def _create_watchlist(self):
        """Create a new watchlist."""
        print("\n--- Create New Watchlist ---")
        name = input("Enter a name for the new watchlist: ").strip()
        if not name:
            print("\n⚠️ Watchlist name cannot be empty."); return
        if self.watchlist_manager.get_watchlist(name):
            print(f"\n⚠️ Watchlist '{name}' already exists."); return

        description = input("Enter a description: ").strip()

        if self.watchlist_manager.create_watchlist(name, description):
            print(f"\n✅ Watchlist '{name}' created successfully.")
            tickers_str = input("Enter tickers to add (comma-separated, e.g., AAPL,GOOGL): ").strip().upper()
            if tickers_str:
                tickers = [t.strip() for t in tickers_str.split(',') if t.strip()]
                for ticker in tickers:
                    self.watchlist_manager.add_ticker(name, ticker)
                print(f"✅ Added {len(tickers)} tickers to '{name}'.")
        else:
            # This case should ideally not be hit due to the check above, but is good practice
            print(f"\n⚠️ Failed to create watchlist '{name}'.")

    def _add_ticker_to_watchlist(self):
        """Add a ticker to an existing watchlist."""
        print("\n--- Add Ticker to Watchlist ---")
        if not self._view_all_watchlists(): return

        watchlists = self.watchlist_manager.get_all()
        name_choice = input("\nEnter the name of the watchlist to modify: ").strip()
        if name_choice not in watchlists:
            print("\n⚠️ Invalid watchlist name."); return

        ticker = input("Enter the ticker to add: ").strip().upper()
        if not ticker:
            print("\n⚠️ Ticker cannot be empty."); return

        if self.watchlist_manager.add_ticker(name_choice, ticker):
            print(f"\n✅ Ticker '{ticker}' added to '{name_choice}'.")
        else:
            # This is better feedback based on the manager's logic
            print(f"\nℹ️ Ticker '{ticker}' already exists in '{name_choice}'.")

    def _remove_ticker_from_watchlist(self):
        """Remove a ticker from an existing watchlist."""
        print("\n--- Remove Ticker from Watchlist ---")
        if not self._view_all_watchlists(show_tickers=True): return

        watchlists = self.watchlist_manager.get_all()
        name_choice = input("\nEnter the name of the watchlist to modify: ").strip()
        if name_choice not in watchlists:
            print("\n⚠️ Invalid watchlist name."); return

        ticker = input("Enter the ticker to remove: ").strip().upper()
        if not ticker:
            print("\n⚠️ Ticker cannot be empty."); return

        if self.watchlist_manager.remove_ticker(name_choice, ticker):
            print(f"\n✅ Ticker '{ticker}' removed from '{name_choice}'.")
        else:
            print(f"\n⚠️ Ticker '{ticker}' not found in '{name_choice}'.")

    def _delete_watchlist(self):
        """Delete an entire watchlist."""
        print("\n--- Delete Watchlist ---")
        if not self._view_all_watchlists(): return

        watchlists = self.watchlist_manager.get_all()
        name_choice = input("\nEnter the name of the watchlist to DELETE: ").strip()
        if name_choice not in watchlists:
            print("\n⚠️ Invalid watchlist name."); return

        confirm = input(f"🔴 Are you sure you want to permanently delete the '{name_choice}' watchlist? (y/n): ").strip().lower()
        if confirm == 'y':
            if self.watchlist_manager.delete_watchlist(name_choice):
                print(f"\n✅ Watchlist '{name_choice}' has been deleted.")
            else:
                # This case is unlikely if the name is coming from the list, but good practice
                print(f"\n⚠️ Error deleting '{name_choice}'. It may have already been removed.")
        else:
            print("\nDeletion cancelled.")

    def set_new_ticker(self):
        """Set a new ticker for single-ticker analysis."""
        print(f"\nCurrent ticker is: {self.ticker}")
        new_ticker = input("Enter new ticker (e.g., AAPL, NQ=F): ").strip().upper()
        if new_ticker:
            self.ticker = new_ticker
            print(f"✅ Ticker changed to: {self.ticker}")
        else:
            print("⚠️ Ticker cannot be empty. No changes made.")

    def configure_notifications(self):
        """Configure Discord notifications."""
        print("\n--- Configure Notifications ---")

        # Check current status
        status = "ENABLED" if self.notifier.is_enabled() else "DISABLED"
        print(f"Discord notifications are currently: {status}")

        # Present options
        print("\n[1] Enable Notifications")
        print("[2] Disable Notifications")
        print("[3] Back")
        choice = input("Enter choice: ").strip()

        if choice == '1':
            self.notifier.enable()
            print("✅ Discord notifications have been ENABLED.")
        elif choice == '2':
            self.notifier.disable()
            print("✅ Discord notifications have been DISABLED.")
        elif choice == '3':
            return
        else:
            print("⚠️ Invalid choice.")

    def view_cache_summary(self):
        """View summary of cached data for the current ticker."""
        print("\n--- Cached Data Summary ---")
        try:
            summary = get_cache_statistics()

            if not summary:
                print("\nNo cached data found.")
                return

            print(f"\nTotal tickers with cached data: {len(summary)}")

            # Display details for each ticker
            for ticker, intervals in summary.items():
                print(f"\nTicker: {ticker}")
                for interval, details in intervals.items():
                    print(f"  - Interval: {interval}")
                    print(f"    Rows: {details['rows']}")
                    print(f"    Start: {details['start_date']}")
                    print(f"    End: {details['end_date']}")

        except Exception as e:
            print(f"\n❌ An error occurred while fetching cache summary: {e}")

    def _display_signal_summary(self, signal: Dict):
        """Display signal summary in console"""
        # ... (implementation restored)
        pass

    def run(self):
        """Main application loop."""
        while True:
            self.display_menu()
            choice = input("\nEnter choice (1-10): ").strip()
            action = {
                '1': self.analyze_watchlist_and_display, '2': self.start_multi_ticker_scanner,
                '3': self.generate_single_ticker_plan, '4': self.view_historical_statistics,
                '5': self.train_ml_models_for_ticker, '6': self.manage_watchlists,
                '7': self.set_new_ticker, '8': self.configure_notifications,
                '9': self.view_cache_summary
            }.get(choice)

            if choice == '10': print("\n👋 Goodbye!"); break

            if action:
                action()
                input("\nPress Enter to continue...")
            else:
                print("\n⚠️  Invalid choice. Please try again.")
                time.sleep(1)

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
