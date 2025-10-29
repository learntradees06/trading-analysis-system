# src/watchlist_manager.py
"""Manages loading, saving, and editing of watchlists from a JSON file."""

import json
from pathlib import Path
from typing import Dict, List, Optional

# Define the default path to the watchlists file
DEFAULT_WATCHLIST_FILE_PATH = Path(__file__).resolve().parent.parent / "config/watchlists.json"

class WatchlistManager:
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the WatchlistManager.

        Args:
            filepath (Optional[str]): Path to the watchlist JSON file.
                                     Defaults to the standard application path.
        """
        self.filepath = Path(filepath) if filepath else DEFAULT_WATCHLIST_FILE_PATH
        # Ensure the directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._watchlists = self._load()

    def _load(self) -> Dict:
        """Load watchlists from the JSON file."""
        if self.filepath.exists():
            with open(self.filepath, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {} # Return empty dict if file is corrupted
        return {}

    def _save(self):
        """Save the current watchlists to the JSON file."""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self._watchlists, f, indent=4)

    def get_all(self) -> Dict:
        """Return all watchlists."""
        return self._watchlists

    def get_watchlist(self, name: str) -> Optional[Dict]:
        """Return a specific watchlist by name."""
        return self._watchlists.get(name)

    def create_watchlist(self, name: str, description: str = "") -> bool:
        """
        Create a new, empty watchlist.

        Returns:
            True if successful, False if the watchlist already exists.
        """
        if name in self._watchlists:
            return False

        self._watchlists[name] = {
            "description": description,
            "tickers": []
        }
        self._save()
        return True

    def delete_watchlist(self, name: str) -> bool:
        """
        Delete a watchlist.

        Returns:
            True if successful, False if the watchlist does not exist.
        """
        if name not in self._watchlists:
            return False

        del self._watchlists[name]
        self._save()
        return True

    def add_ticker(self, watchlist_name: str, ticker: str) -> bool:
        """
        Add a ticker to a watchlist. Ensures no duplicates are added.

        Returns:
            True if successful, False if watchlist doesn't exist or ticker is already in it.
        """
        ticker = ticker.upper()
        if watchlist_name not in self._watchlists:
            return False

        if ticker not in self._watchlists[watchlist_name]["tickers"]:
            self._watchlists[watchlist_name]["tickers"].append(ticker)
            self._save()
            return True

        return False # Ticker already exists in the list

    def remove_ticker(self, watchlist_name: str, ticker: str) -> bool:
        """
        Remove a ticker from a watchlist.

        Returns:
            True if successful, False if watchlist or ticker does not exist.
        """
        ticker = ticker.upper()
        if watchlist_name not in self._watchlists:
            return False

        if ticker in self._watchlists[watchlist_name]["tickers"]:
            self._watchlists[watchlist_name]["tickers"].remove(ticker)
            self._save()
            return True

        return False # Ticker not found
