"""
Strategy Repository Module.

Persists passing strategies to JSON storage.
Features:
- Append-only storage
- Deduplication by strategy hash
- Timestamp tracking
- Export functionality
"""

import json
import os
from typing import List, Optional
from datetime import datetime
from models import Strategy, BacktestResult, StoredStrategy
from config import Config


class StrategyRepository:
    """
    Persistent storage for passing strategies.
    
    Stores strategies in JSON format for portability.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize repository.
        
        Args:
            storage_path: Path to JSON storage file
        """
        self.storage_path = storage_path or Config.STORAGE_PATH
        self._ensure_storage_exists()
    
    def _ensure_storage_exists(self):
        """Create storage file if it doesn't exist."""
        if not os.path.exists(self.storage_path):
            self._save([])
    
    def _load(self) -> List[dict]:
        """Load strategies from storage."""
        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save(self, data: List[dict]):
        """Save strategies to storage."""
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def store(self, strategy: Strategy, 
              backtest_result: BacktestResult) -> bool:
        """
        Store a passing strategy.
        
        Args:
            strategy: Strategy that passed filters
            backtest_result: Associated backtest results
        
        Returns:
            True if stored, False if duplicate
        """
        stored = StoredStrategy(
            strategy=strategy,
            backtest_result=backtest_result,
        )
        
        # Load existing
        data = self._load()
        
        # Check for duplicates
        existing_hashes = {item.get("hash", "") for item in data}
        if stored.hash in existing_hashes:
            print(f"⚠️ Strategy {strategy.name} is a duplicate, skipping")
            return False
        
        # Append and save
        data.append(stored.to_dict())
        self._save(data)
        
        print(f"✅ Stored strategy: {strategy.name}")
        return True
    
    def get_all(self) -> List[StoredStrategy]:
        """
        Get all stored strategies.
        
        Returns:
            List of StoredStrategy objects
        """
        data = self._load()
        return [StoredStrategy.from_dict(item) for item in data]
    
    def get_count(self) -> int:
        """Get count of stored strategies."""
        return len(self._load())
    
    def get_latest(self, n: int = 10) -> List[StoredStrategy]:
        """
        Get N most recent strategies.
        
        Args:
            n: Number of strategies to return
        
        Returns:
            List of most recent StoredStrategy objects
        """
        data = self._load()
        # Sort by stored_at descending
        data.sort(key=lambda x: x.get("stored_at", ""), reverse=True)
        return [StoredStrategy.from_dict(item) for item in data[:n]]
    
    def export_to_file(self, filepath: str) -> bool:
        """
        Export all strategies to a separate file.
        
        Args:
            filepath: Path to export file
        
        Returns:
            True if successful
        """
        try:
            data = self._load()
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
    
    def clear(self):
        """Clear all stored strategies (use with caution)."""
        self._save([])
        print("⚠️ All strategies cleared")


# Module-level instance
_repository = StrategyRepository()


def store_strategy(strategy: Strategy, 
                   backtest_result: BacktestResult) -> bool:
    """
    Convenience function to store a strategy.
    
    Args:
        strategy: Strategy to store
        backtest_result: Associated backtest results
    
    Returns:
        True if stored successfully
    """
    return _repository.store(strategy, backtest_result)


def get_stored_count() -> int:
    """Get count of stored strategies."""
    return _repository.get_count()
