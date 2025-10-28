# src/dashboard.py
"""Module for displaying multi-ticker analysis results in a console dashboard"""

from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
import time

class Dashboard:
    def __init__(self):
        self.console = Console()

    def display_summary_table(self, results: Dict):
        """
        Display a summary table of the latest analysis for all tickers.

        Args:
            results: A dictionary where keys are tickers and values are the signal dictionaries.
        """
        table = Table(title="Trading Plan Summary Dashboard", show_header=True, header_style="bold magenta")

        table.add_column("Ticker", style="cyan", width=12)
        table.add_column("Signal", style="white")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Key Evidence", style="white", max_width=60)
        table.add_column("RSI", justify="right", style="magenta")
        table.add_column("ADX", justify="right", style="blue")

        # Separate successful and failed analyses
        successful_results = {t: r for t, r in results.items() if 'error' not in r}
        failed_results = {t: r for t, r in results.items() if 'error' in r}

        # Sort successful results by score, descending
        sorted_tickers = sorted(successful_results.keys(), key=lambda t: successful_results[t]['signal'].get('score', 0), reverse=True)

        # Add successful results to the table
        for ticker in sorted_tickers:
            result_data = successful_results[ticker]
            signal_data = result_data.get('signal', {})
            tech_data = result_data.get('technicals', {})

            signal = signal_data.get('signal', 'N/A')
            score = f"{signal_data.get('score', 0):.1f}"
            confidence = signal_data.get('confidence', 'N/A')
            evidence = ", ".join(signal_data.get('evidence', []))
            rsi = f"{tech_data.get('RSI', 0):.1f}"
            adx = f"{tech_data.get('ADX', 0):.1f}"

            # Color code the signal
            if "LONG" in signal: signal_colored = f"[bold green]{signal}[/bold green]"
            elif "SHORT" in signal: signal_colored = f"[bold red]{signal}[/bold red]"
            else: signal_colored = f"[bold yellow]{signal}[/bold yellow]"

            table.add_row(ticker, signal_colored, score, confidence, evidence, rsi, adx)

        # Add failed results to the table
        if failed_results:
            # Add a separator for clarity
            if successful_results:
                table.add_row("---", "---", "---", "---", "---", "---", "---")

            for ticker, result_data in failed_results.items():
                error_message = result_data.get('error', 'Unknown error')
                table.add_row(
                    f"[dim]{ticker}[/dim]",
                    "[bold red]ERROR[/bold red]",
                    "-",
                    "-",
                    f"[dim red]{error_message}[/dim red]",
                    "-",
                    "-"
                )

        self.console.print(table)

    def display_alert(self, ticker: str, signal: Dict):
        """
        Display a real-time alert for a high-confidence signal.
        """
        signal_type = signal.get('signal', 'NEUTRAL')
        score = signal.get('score', 0)

        if "LONG" in signal_type:
            color = "green"
            icon = "🔼"
        elif "SHORT" in signal_type:
            color = "red"
            icon = "🔽"
        else:
            return # Don't alert for neutral signals

        self.console.print(
            f"[{color}]🔔 HIGH CONFIDENCE ALERT: {icon} {ticker} - {signal_type} (Score: {score:.1f})[/]"
        )

    def show_live_scanner_layout(self):
        """
        Creates and returns a Live object with a layout for the scanner.
        This is a placeholder for a more complex live view.
        """
        spinner = Spinner("dots", " Scanning for signals...")

        layout = Table.grid(expand=True)
        layout.add_column(justify="center")
        layout.add_row(spinner)

        return Live(layout, console=self.console, screen=False, auto_refresh=True)

if __name__ == '__main__':
    # Example Usage
    dashboard = Dashboard()

    # Example data
    example_results = {
        "ES=F": {
            "signal": {
                'signal': 'LONG', 'score': 75.2, 'confidence': 'HIGH',
                'evidence': ['Bullish Engulfing', 'RSI Oversold', 'Above 200 EMA']
            },
            "technicals": {'RSI': 28.5, 'ADX': 22.1}
        },
        "AAPL": {
            "signal": {
                'signal': 'SHORT', 'score': 68.0, 'confidence': 'MEDIUM',
                'evidence': ['Bearish Divergence', 'MACD Cross', 'Below 50 EMA']
            },
            "technicals": {'RSI': 65.2, 'ADX': 35.8}
        },
        "GOOGL": {
            "signal": {
                'signal': 'NEUTRAL', 'score': 45.0, 'confidence': 'LOW',
                'evidence': ['Choppy Market', 'ADX below 20']
            },
            "technicals": {'RSI': 51.7, 'ADX': 18.2}
        }
    }

    dashboard.display_summary_table(example_results)

    print("\n--- Testing Alerts ---")
    dashboard.display_alert("ES=F", example_results["ES=F"]["signal"])
    dashboard.display_alert("AAPL", example_results["AAPL"]["signal"])
