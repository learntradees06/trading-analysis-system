# main.py
"""Main Application - Interactive Command-Line Interface"""

import os
import sys
from pathlib import Path
import pandas as pd
import pytz
from datetime import datetime
import time
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    INSTRUMENT_SETTINGS, DEFAULT_TICKER, DISCORD_WEBHOOK_URL,
    MODELS_DIR, REPORTS_DIR
)
from src.data_manager import DataManager, get_cache_statistics
from src.indicators import calculate_all_indicators
from src.market_profile import MarketProfile
from src.sr_levels import SRLevelAnalyzer
from src.statistics import StatisticalAnalyzer
from src.ml_models import MLPredictor
from src.signals import SignalGenerator
from src.reporting import ReportGenerator
from src.notifications import NotificationManager

class TradingSystem:
    def __init__(self):
        """Initialize the Trading System"""
        self.ticker = DEFAULT_TICKER
        self.data_manager = None
        self.market_profile = None
        self.sr_analyzer = None
        self.stats_analyzer = None
        self.ml_predictor = None
        self.signal_generator = None
        self.report_generator = None
        self.notifier = NotificationManager(DISCORD_WEBHOOK_URL)

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all system components for current ticker"""
        print(f"\n🔄 Initializing components for {self.ticker}...")

        # Get instrument settings
        if self.ticker in INSTRUMENT_SETTINGS:
            self.settings = INSTRUMENT_SETTINGS[self.ticker]
        else:
            print(f"⚠️  Warning: No specific settings for {self.ticker}, using defaults")
            self.settings = {
                'tick_size': 0.01,
                'rth_start': '08:30',
                'rth_end': '15:00',
                'timezone': 'US/Central',
                'description': 'Default Stock/ETF'
            }

        # Initialize components - DataManager only takes ticker
        self.data_manager = DataManager(self.ticker)  # ← Fixed: removed self.settings
        self.market_profile = MarketProfile(self.ticker, self.settings['tick_size'])
        self.sr_analyzer = SRLevelAnalyzer(self.ticker, self.settings['tick_size'])
        self.stats_analyzer = StatisticalAnalyzer(self.ticker)
        self.ml_predictor = MLPredictor(self.ticker, MODELS_DIR)  # ← Added MODELS_DIR
        self.signal_generator = SignalGenerator(self.ticker)
        self.report_generator = ReportGenerator(self.ticker, REPORTS_DIR)  # ← Added REPORTS_DIR

        print(f"✅ Components initialized successfully for {self.ticker}")

    def check_market_hours(self) -> bool:
        """Check if market is currently open (RTH)"""
        tz = pytz.timezone(self.settings['timezone'])
        now = datetime.now(tz)

        # Parse RTH times
        start_hour, start_min = map(int, self.settings['rth_start'].split(':'))
        end_hour, end_min = map(int, self.settings['rth_end'].split(':'))

        market_open = now.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
        market_close = now.replace(hour=end_hour, minute=end_min, second=0, microsecond=0)

        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        return market_open <= now <= market_close

    def display_menu(self):
        """Display appropriate menu based on market hours"""
        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 80)
        print("   ADVANCED TRADING ANALYSIS SYSTEM")
        print("=" * 80)
        print(f"\n📊 Current Ticker: {self.ticker}")
        print(f"📍 Tick Size: {self.settings['tick_size']}")
        print(f"🕒 RTH: {self.settings['rth_start']} - {self.settings['rth_end']} {self.settings['timezone']}")

        # Show notification status
        if self.notifier.webhook_url:
            print(f"🔔 Discord: ✅ Configured")
        else:
            print(f"🔔 Discord: ❌ Not configured")

        is_market_open = self.check_market_hours()

        if is_market_open:
            print("\n✅ Market is OPEN")
            print("\n--- Live Market Analysis System ---")
            print("[1] Start Live Scanner")
            print("[2] Generate Current Day Analysis")
            print("[3] Generate Next Day Trading Plan")
            print("[4] View Historical Profile Statistics")
            print("[5] Train/Update ML Models")
            print("[6] Configure Notifications")
            print("[7] View Cached Data Summary")
            print("[8] Set New Ticker")
            print("[9] Exit")
        else:
            print("\n🔴 Market is CLOSED")
            print("\n--- Market Analysis Planner ---")
            print("[1] Generate Next Day Trading Plan")
            print("[2] View Historical Profile Statistics")
            print("[3] Train/Update ML Models")
            print("[4] Configure Notifications")
            print("[5] View Cached Data Summary")
            print("[6] Set New Ticker")
            print("[7] Exit")

    def generate_trading_plan(self, is_next_day: bool = True):
        """Generate comprehensive trading plan"""
        print(f"\n📊 Generating {'Next Day' if is_next_day else 'Current Day'} Trading Plan...")

        try:
            # Fetch data
            print("📥 Fetching market data...")
            daily_data = self.data_manager.fetch_data('1d', days_back=100)
            hourly_data = self.data_manager.fetch_data('4h', days_back=100)  # Changed from 1h to 4h
            thirty_min_data = self.data_manager.fetch_data('30m', days_back=100)
            five_min_data = self.data_manager.fetch_data('5m', days_back=10)

            # Calculate technical indicators
            print("📈 Calculating technical indicators...")
            daily_with_indicators = calculate_all_indicators(daily_data)

            # Generate market profiles
            print("📊 Analyzing market profiles...")
            profiles = []
            intraday_data_dict = {}

            # Get a list of unique dates present in the 30-min data
            unique_days = thirty_min_data.index.normalize().unique()

            # Filter to match the dates available in the daily data for context
            relevant_days = [d for d in unique_days if d in daily_data.index]

            for date in relevant_days:
                # Get RTH data for this date
                day_data = thirty_min_data[thirty_min_data.index.date == date.date()]
                if not day_data.empty:
                    rth_data = self.data_manager.get_rth_data(day_data)
                    if not rth_data.empty:
                        profile = self.market_profile.calculate_tpo_profile(rth_data, date)
                        if profile:
                            # Add opening type classification
                            if profiles:  # Need prior profile
                                profile['opening_type'] = self.market_profile.classify_opening_type(
                                    rth_data['Open'].iloc[0] if not rth_data.empty else 0,
                                    profiles[-1]
                                )
                            profiles.append(profile)
                            intraday_data_dict[date] = rth_data

            # Calculate statistics
            print("📊 Calculating historical statistics...")
            statistics = self.stats_analyzer.calculate_opening_type_statistics(profiles, intraday_data_dict)

            # Analyze S/R levels
            print("🔍 Analyzing support/resistance levels...")
            sr_data = {
                '1d': daily_data,
                '4h': hourly_data,
                '30m': thirty_min_data
            }
            sr_analysis = self.sr_analyzer.analyze_all_sr_levels(sr_data)

            # Get current/latest profile
            current_profile = profiles[-1] if profiles else {}

            # Prepare ML features and make predictions
            ml_features = self.ml_predictor.create_fusion_features(
                profiles, daily_with_indicators, sr_analysis
            )

            # --- START REPLACEMENT ---
            ml_predictions = {}
            models_to_check = ['target_broke_ibh', 'target_broke_ibl', 'target_next_day_direction']
            missing_models = [target for target in models_to_check if not self.ml_predictor.model_exists(target)]

            if missing_models:
                print(f"⚠️ Models not found: {', '.join(missing_models)}. Training models now...")
                self.train_ml_models()

            # Always try to load models after checking, in case they were just trained
            all_models_loaded = all(self.ml_predictor.load_model(target) for target in models_to_check)

            if all_models_loaded and ml_features is not None and not ml_features.empty:
                print("🤖 Running ML predictions...")
                ml_predictions = self.ml_predictor.predict(ml_features)
            else:
                print("🤖 Skipping ML predictions as models are not ready.")
            # --- END REPLACEMENT ---

            # Generate trading signal
            print("🎯 Generating trading signal...")
            signal = self.signal_generator.generate_signal(
                current_profile,
                daily_with_indicators.iloc[-1] if not daily_with_indicators.empty else pd.Series(),
                sr_analysis,
                ml_predictions,
                statistics
            )

            # Generate HTML report
            print("📝 Creating HTML report...")
            report_path = self.report_generator.generate_report(
                current_profile,
                daily_with_indicators,
                sr_analysis,
                ml_predictions,
                signal,
                statistics,
                daily_data
            )

            print(f"\n✅ Report generated successfully!")
            print(f"📄 Report saved to: {report_path}")

            # Display summary
            self._display_signal_summary(signal)

            # Send Discord notification if high confidence
            if signal['confidence'] == 'HIGH':
                print("\n📢 Sending Discord alert...")
                if self.notifier.send_signal_alert(self.ticker, signal):
                    print("✅ Discord alert sent!")

        except Exception as e:
            print(f"\n❌ Error generating trading plan: {e}")
            import traceback
            traceback.print_exc()

    def view_historical_statistics(self):
        """View historical profile statistics"""
        print("\n📊 Calculating Historical Statistics...")

        try:
            # Fetch data
            daily_data = self.data_manager.fetch_data('1d', days_back=100)
            thirty_min_data = self.data_manager.fetch_data('30m', days_back=100)

            # Generate profiles
            profiles = []
            intraday_data_dict = {}

            for date in daily_data.index:
                day_data = thirty_min_data[thirty_min_data.index.date == date.date()]
                if not day_data.empty:
                    rth_data = self.data_manager.get_rth_data(day_data)
                    if not rth_data.empty:
                        profile = self.market_profile.calculate_tpo_profile(rth_data, date)
                        if profile:
                            # Classify opening type
                            if profiles:  # Need prior profile
                                profile['opening_type'] = self.market_profile.classify_opening_type(
                                    rth_data['Open'].iloc[0] if not rth_data.empty else 0,
                                    profiles[-1]
                                )
                            profiles.append(profile)
                            intraday_data_dict[date] = rth_data

            # Calculate statistics
            statistics = self.stats_analyzer.calculate_opening_type_statistics(profiles, intraday_data_dict)

            # Display formatted table
            print(self.stats_analyzer.format_statistics_table(statistics))

        except Exception as e:
            print(f"\n❌ Error calculating statistics: {e}")

    def start_live_scanner(self):
        """Start live market scanner"""
        if not self.check_market_hours():
            print("\n⚠️  Market is closed. Live scanner is only available during RTH.")
            return

        print("\n🔍 Live Scanner Configuration")
        print("-" * 40)

        # Get timeframe
        print("Select timeframe:")
        print("[1] 5 minutes")
        print("[2] 30 minutes")
        print("[3] 1 hour")

        timeframe_choice = input("\nChoice: ").strip()
        timeframe_map = {'1': '5m', '2': '30m', '3': '1h'}
        timeframe = timeframe_map.get(timeframe_choice, '5m')

        # Get refresh rate
        refresh_rate = input("Refresh rate in seconds (default 60): ").strip()
        refresh_rate = int(refresh_rate) if refresh_rate.isdigit() else 60

        print(f"\n✅ Starting live scanner (Timeframe: {timeframe}, Refresh: {refresh_rate}s)")
        print("Press Ctrl+C to stop...\n")

        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("=" * 80)
                print(f"   LIVE SCANNER - {self.ticker} - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 80)
                print("   [Press Ctrl+C to stop]")
                print("-" * 80)

                # Fetch latest data
                data = self.data_manager.fetch_data(timeframe, days_back=5, force_refresh=True)

                if not data.empty:
                    # Calculate indicators
                    data_with_indicators = calculate_all_indicators(data)

                    # Quick signal check
                    latest = data_with_indicators.iloc[-1]

                    # Simple signal logic for live scanner
                    signal_score = 50
                    evidence = []

                    # RSI check
                    if 'RSI' in latest:
                        if latest['RSI'] > 70:
                            signal_score -= 20
                            evidence.append(f"RSI Overbought ({latest['RSI']:.1f})")
                        elif latest['RSI'] < 30:
                            signal_score += 20
                            evidence.append(f"RSI Oversold ({latest['RSI']:.1f})")

                    # MACD check
                    if 'MACD_Histogram' in latest:
                        if latest['MACD_Histogram'] > 0:
                            signal_score += 10
                            evidence.append("MACD Positive")
                        else:
                            signal_score -= 10
                            evidence.append("MACD Negative")

                    # Display current status
                    print(f"\n📊 Current Price: ${latest['Close']:.2f}")
                    print(f"📈 Change: {((latest['Close'] / latest['Open'] - 1) * 100):.2f}%")
                    print(f"📊 Volume: {latest['Volume']:,.0f}")

                    print(f"\n🎯 Signal Score: {signal_score}/100")

                    if signal_score >= 70:
                        print("🟢 BULLISH SIGNAL DETECTED!")
                        if self.notifier.webhook_url:
                            self.notifier.send_signal_alert(self.ticker, {
                                'signal': 'LONG',
                                'score': signal_score,
                                'confidence': 'HIGH',
                                'evidence': evidence
                            })
                    elif signal_score <= 30:
                        print("🔴 BEARISH SIGNAL DETECTED!")
                        if self.notifier.webhook_url:
                            self.notifier.send_signal_alert(self.ticker, {
                                'signal': 'SHORT',
                                'score': signal_score,
                                'confidence': 'HIGH',
                                'evidence': evidence
                            })
                    else:
                        print("🟡 NEUTRAL - No clear signal")

                    print("\n📋 Evidence:")
                    for e in evidence:
                        print(f"  • {e}")

                    # Technical readings
                    print("\n📉 Technical Readings:")
                    print(f"  RSI: {latest.get('RSI', 0):.1f}")
                    print(f"  ADX: {latest.get('ADX', 0):.1f}")
                    print(f"  Stoch K: {latest.get('Stoch_K', 0):.1f}")

                # Wait for next refresh
                time.sleep(refresh_rate)

        except KeyboardInterrupt:
            print("\n\n✋ Live scanner stopped.")

    def train_ml_models(self):
        """Train or update ML models"""
        print("\n🤖 Training Machine Learning Models...")

        try:
            # Fetch data
            print("📥 Fetching historical data...")
            daily_data = self.data_manager.fetch_data('1d', days_back=200)
            thirty_min_data = self.data_manager.fetch_data('30m', days_back=200)

            # Calculate indicators
            print("📈 Calculating indicators...")
            daily_with_indicators = calculate_all_indicators(daily_data)

            # Generate profiles
            print("📊 Generating market profiles...")
            profiles = []
            intraday_data_dict = {}

            for date in daily_data.index:
                day_data = thirty_min_data[thirty_min_data.index.date == date.date()]
                if not day_data.empty:
                    rth_data = self.data_manager.get_rth_data(day_data)
                    if not rth_data.empty:
                        profile = self.market_profile.calculate_tpo_profile(rth_data, date)
                        if profile:
                            if profiles:
                                profile['opening_type'] = self.market_profile.classify_opening_type(
                                    rth_data['Open'].iloc[0] if not rth_data.empty else 0,
                                    profiles[-1]
                                )
                            profiles.append(profile)
                            intraday_data_dict[date] = rth_data

            # Create features
            print("🔧 Creating fusion features...")
            sr_analysis = {}  # Simplified for training
            ml_features = self.ml_predictor.create_fusion_features(
                profiles, daily_with_indicators, sr_analysis
            )

            if ml_features.empty:
                print("❌ Insufficient data for training")
                return

            # Train models
            print(f"🎯 Training models with {len(ml_features)} samples...")
            results = self.ml_predictor.train_models(ml_features)

            # Display results
            print("\n📊 Training Results:")
            print("-" * 60)

            for target, result in results.items():
                if result['status'] == 'success':
                    print(f"\n✅ {target}:")
                    print(f"   Train Score: {result['train_score']:.3f}")
                    print(f"   Test Score: {result['test_score']:.3f}")
                    print(f"   CV Score: {result['cv_score_mean']:.3f} (±{result['cv_score_std']:.3f})")
                    print(f"   Samples: {result['n_samples']}")

                    # Top features
                    print(f"   Top Features:")
                    for _, row in result['feature_importance'].head(5).iterrows():
                        print(f"      • {row['feature']}: {row['importance']:.3f}")
                else:
                    print(f"\n❌ {target}: {result['status']}")

            print("\n✅ Models trained and saved successfully!")

        except Exception as e:
            print(f"\n❌ Error training models: {e}")
            import traceback
            traceback.print_exc()

    def set_new_ticker(self):
        """Set a new ticker for analysis"""
        print("\n📊 Available Tickers:")
        print("-" * 40)

        for ticker, settings in INSTRUMENT_SETTINGS.items():
            print(f"  {ticker}: {settings.get('description', 'N/A')}")

        print("\n  Or enter any other valid ticker symbol")

        new_ticker = input("\nEnter ticker symbol: ").strip().upper()

        if new_ticker:
            self.ticker = new_ticker
            print(f"\n✅ Ticker changed to {self.ticker}")
            print("🔄 Reinitializing components...")
            self._initialize_components()

    def configure_notifications(self):
        """Configure notification settings"""
        print("\n🔔 Notification Configuration")
        print("-" * 40)

        # Check current status
        if self.notifier.webhook_url:
            print(f"✅ Discord webhook is configured")
            print(f"   Current webhook: {self.notifier.webhook_url[:50]}...")
            change = input("\nDo you want to change it? (y/n): ").strip().lower()
            if change != 'y':
                return
        else:
            print("❌ No Discord webhook configured")

        print("\n📌 To get a Discord webhook:")
        print("1. Go to your Discord server")
        print("2. Right-click on a channel → Edit Channel")
        print("3. Go to Integrations → Webhooks")
        print("4. Create a new webhook and copy the URL")

        webhook_url = input("\nEnter Discord webhook URL (or 'skip' to continue without): ").strip()

        if webhook_url and webhook_url.lower() != 'skip':
            # Test the webhook
            self.notifier.webhook_url = webhook_url

            print("\n🧪 Testing webhook...")
            test_sent = self.notifier.send_to_discord(
                f"✅ Test message from Trading System for {self.ticker}"
            )

            if test_sent:
                print("✅ Webhook configured and tested successfully!")

                # Save to config file (optional)
                save = input("\nSave webhook to config? (y/n): ").strip().lower()
                if save == 'y':
                    self._save_webhook_to_config(webhook_url)
            else:
                print("❌ Failed to send test message. Please check the webhook URL.")
                self.notifier.webhook_url = ""
        else:
            print("⚠️ Continuing without Discord notifications")

    def _save_webhook_to_config(self, webhook_url: str):
        """Save webhook URL to config file"""
        import src.config as config
        config_path = Path("src/config.py")

        try:
            # Read current config
            with open(config_path, 'r') as f:
                lines = f.readlines()

            # Update webhook line
            for i, line in enumerate(lines):
                if 'DISCORD_WEBHOOK_URL' in line:
                    lines[i] = f'DISCORD_WEBHOOK_URL = "{webhook_url}"\n'
                    break

            # Write back
            with open(config_path, 'w') as f:
                f.writelines(lines)

            print("✅ Webhook saved to config.py")
        except Exception as e:
            print(f"❌ Could not save webhook: {e}")

    def view_cache_summary(self):
        """View summary of all cached data"""
        print("\n📊 Data Cache Summary")
        print("=" * 60)

        # Get summary from data manager
        summary = self.data_manager.get_data_summary()

        if summary.empty:
            print(f"❌ No cached data for {self.ticker}")
            return

        print(f"\nTicker: {self.ticker}")
        print("-" * 60)
        print(f"{'Timeframe':<12} {'First Date':<20} {'Last Date':<20} {'Rows':<10} {'Quality':<10}")
        print("-" * 60)

        for _, row in summary.iterrows():
            print(f"{row['timeframe']:<12} "
                  f"{row['first_date'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{row['last_date'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{row['total_rows']:<10} "
                  f"{row['data_quality']:.1f}%")

        # Get overall cache statistics
        stats = get_cache_statistics()

        print("\n📈 Overall Cache Statistics:")
        print(f"   Total rows in database: {stats['total_rows']:,}")
        print(f"   Database size: {stats['database_size_mb']:.2f} MB")
        print(f"   Total tickers cached: {len(stats['ticker_stats'])}")

        # Ask if user wants to download more data
        print("\n" + "-" * 60)
        download_more = input("\nDo you want to download/update data? (y/n): ").strip().lower()

        if download_more == 'y':
            self._download_data_interactive()

    def _download_data_interactive(self):
        """Interactive data download"""
        print("\n📥 Data Download Options")
        print("-" * 40)
        print("[1] Download all timeframes (maximum data)")
        print("[2] Download specific timeframe")
        print("[3] Update existing data only")
        print("[4] Cancel")

        choice = input("\nChoice: ").strip()

        if choice == '1':
            print("\n⏳ Downloading all timeframes (this may take a few minutes)...")
            timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
            for tf in timeframes:
                print(f"   Downloading {tf}...", end='')
                try:
                    df = self.data_manager.fetch_data(tf, max_data=True, force_refresh=True)
                    print(f" ✅ {len(df)} rows")
                except Exception as e:
                    print(f" ❌ Error: {e}")

            # Optimize cache after bulk download
            print("\n🔧 Optimizing cache...")
            self.data_manager.optimize_cache()
            print("✅ Download complete!")

        elif choice == '2':
            print("\nAvailable timeframes:")
            timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
            for i, tf in enumerate(timeframes, 1):
                print(f"[{i}] {tf}")

            tf_choice = input("\nSelect timeframe: ").strip()
            if tf_choice.isdigit() and 1 <= int(tf_choice) <= len(timeframes):
                tf = timeframes[int(tf_choice) - 1]
                print(f"\n⏳ Downloading {tf} data...")
                df = self.data_manager.fetch_data(tf, max_data=True, force_refresh=True)
                print(f"✅ Downloaded {len(df)} rows")

        elif choice == '3':
            print("\n⏳ Updating all cached data...")
            summary = self.data_manager.get_data_summary()
            for _, row in summary.iterrows():
                tf = row['timeframe']
                print(f"   Updating {tf}...", end='')
                df = self.data_manager.fetch_data(tf, max_data=False, force_refresh=False)
                print(f" ✅")
            print("✅ Update complete!")

    def _display_signal_summary(self, signal: Dict):
        """Display signal summary in console"""
        print("\n" + "=" * 60)
        print("   TRADING SIGNAL SUMMARY")
        print("=" * 60)

        signal_type = signal.get('signal', 'NEUTRAL')
        score = signal.get('score', 0)
        confidence = signal.get('confidence', 'LOW')

        # Color coding for terminal (simplified)
        if 'LONG' in signal_type:
            print(f"🟢 Signal: {signal_type}")
        elif 'SHORT' in signal_type:
            print(f"🔴 Signal: {signal_type}")
        else:
            print(f"🟡 Signal: {signal_type}")

        print(f"📊 Score: {score:.1f}/100")
        print(f"🎯 Confidence: {confidence}")

        print("\n📋 Key Evidence:")
        for evidence in signal.get('evidence', [])[:5]:
            print(f"  • {evidence}")

        if 'component_scores' in signal:
            print("\n🎯 Component Scores:")
            for component, score in signal['component_scores'].items():
                print(f"  • {component.replace('_', ' ').title()}: {score:.1f}")

    def run(self):
        """Main application loop"""
        print("\n🚀 Starting Advanced Trading Analysis System...")

        while True:
            self.display_menu()

            is_market_open = self.check_market_hours()

            if is_market_open:
                choice = input("\nEnter choice (1-9): ").strip()

                if choice == '1':
                    self.start_live_scanner()
                elif choice == '2':
                    self.generate_trading_plan(is_next_day=False)
                elif choice == '3':
                    self.generate_trading_plan(is_next_day=True)
                elif choice == '4':
                    self.view_historical_statistics()
                elif choice == '5':
                    self.train_ml_models()
                elif choice == '6':
                    self.configure_notifications()
                elif choice == '7':
                    self.view_cache_summary()
                elif choice == '8':
                    self.set_new_ticker()
                elif choice == '9':
                    print("\n👋 Thank you for using the Advanced Trading System. Good luck trading!")
                    break
                else:
                    print("\n⚠️  Invalid choice. Please try again.")
            else:
                choice = input("\nEnter choice (1-7): ").strip()

                if choice == '1':
                    self.generate_trading_plan(is_next_day=True)
                elif choice == '2':
                    self.view_historical_statistics()
                elif choice == '3':
                    self.train_ml_models()
                elif choice == '4':
                    self.configure_notifications()
                elif choice == '5':
                    self.view_cache_summary()
                elif choice == '6':
                    self.set_new_ticker()
                elif choice == '7':
                    print("\n👋 Thank you for using the Advanced Trading System. Good luck trading!")
                    break
                else:
                    print("\n⚠️  Invalid choice. Please try again.")

            if choice != '1':  # Don't pause for live scanner
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
