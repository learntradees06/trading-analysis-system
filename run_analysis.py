from main import TradingSystem

def run_analysis():
    """
    Initializes the TradingSystem and runs the trading plan generation.
    """
    print("Initializing Trading System for analysis...")
    system = TradingSystem()
    print("Generating trading plan...")
    system.generate_trading_plan(is_next_day=True)
    print("Trading plan generation complete.")

if __name__ == "__main__":
    run_analysis()
