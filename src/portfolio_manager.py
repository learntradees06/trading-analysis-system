# src/portfolio_manager.py
"""Manages multi-ticker analysis, parallel execution, and results aggregation"""

import concurrent.futures
from typing import List, Dict, Callable, Any
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console

class PortfolioManager:
    def __init__(self, analysis_function: Callable[[str], Dict]):
        """
        Initialize the PortfolioManager.

        Args:
            analysis_function: A callable function that takes a ticker string
                               and returns a dictionary of analysis results.
        """
        self.analysis_function = analysis_function
        self.results: Dict[str, Dict] = {}
        self.console = Console()

    def analyze_watchlist(self, tickers: List[str], max_workers: int = 10) -> Dict[str, Dict]:
        """
        Analyze a list of tickers in parallel using a thread pool.

        Args:
            tickers: A list of ticker symbols to analyze.
            max_workers: The maximum number of threads to use.

        Returns:
            A dictionary containing the analysis results for all tickers.
        """
        self.results = {}

        text_column = TextColumn("[progress.description]{task.description}")
        bar_column = BarColumn()
        spinner_column = SpinnerColumn()

        with Progress(spinner_column, text_column, bar_column, console=self.console) as progress:
            task = progress.add_task("[cyan]Analyzing watchlist...", total=len(tickers))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a future for each ticker analysis
                future_to_ticker = {executor.submit(self.analysis_function, ticker): ticker for ticker in tickers}

                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        self.results[ticker] = result
                    except Exception as e:
                        self.results[ticker] = {"error": str(e)}

                    progress.update(task, advance=1, description=f"[cyan]Analyzed {ticker}")

        return self.results

    def get_summary(self) -> Dict[str, Dict]:
        """
        Return the latest analysis results.
        """
        return self.results

    def get_top_signals(self, top_n: int = 5) -> List[Dict]:
        """
        Get the top N signals based on score.
        """
        if not self.results:
            return []

        # Filter out errors and sort by score
        valid_results = [
            {"ticker": t, **r['signal']} for t, r in self.results.items()
            if "signal" in r and "error" not in r
        ]

        sorted_signals = sorted(valid_results, key=lambda x: x.get('score', 0), reverse=True)

        return sorted_signals[:top_n]

if __name__ == '__main__':
    # Example Usage
    import time
    import random

    # A dummy analysis function for demonstration
    def dummy_analysis_function(ticker: str) -> Dict[str, Any]:
        print(f"Analyzing {ticker}...")
        time.sleep(random.uniform(1, 3)) # Simulate work

        # Simulate a possible error
        if ticker == "RTY=F":
            return {"error": "Failed to fetch data"}

        return {
            "signal": {
                'signal': random.choice(['LONG', 'SHORT', 'NEUTRAL']),
                'score': round(random.uniform(30, 90), 2),
                'confidence': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'evidence': ['Fake evidence point 1', 'Fake evidence point 2']
            },
            "technicals": {
                'RSI': round(random.uniform(20, 80), 2),
                'ADX': round(random.uniform(15, 50), 2)
            }
        }

    # --- Test the manager ---
    watchlist = ["ES=F", "NQ=F", "AAPL", "GOOGL", "RTY=F", "MSFT"]

    portfolio_manager = PortfolioManager(analysis_function=dummy_analysis_function)

    print("--- Starting Watchlist Analysis ---")
    final_results = portfolio_manager.analyze_watchlist(watchlist)

    print("\n--- Full Analysis Results ---")
    import json
    print(json.dumps(final_results, indent=2))

    print("\n--- Top Signals ---")
    top_signals = portfolio_manager.get_top_signals(3)
    print(json.dumps(top_signals, indent=2))
