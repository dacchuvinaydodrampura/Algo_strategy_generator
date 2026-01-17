"""
Intelligence Module - The Brain of the Research Engine.

Analyzes past successful strategies to extract 'DNA' patterns.
Updates DNA weights to bias the generator toward profitable concepts.
"""

import json
import os
import collections
from typing import Dict, List, Any
from config import Config

class IntelligenceModule:
    """
    Extracts and manages 'DNA' patterns from successful strategies.
    
    A 'DNA' consists of:
    - Keywords frequencies in entry rules
    - Timeframe success rates
    - Optimal R:R ranges
    """
    
    def __init__(self, memory_path: str = None):
        self.memory_path = memory_path or Config.DNA_MEMORY_PATH
        self.strategies_path = Config.STORAGE_PATH
        self.dna_weights = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        """Load state from memory or return defaults."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default Weights (Uniform)
        return {
            "keywords": {},
            "timeframes": {"1m": 1.0, "5m": 1.0, "15m": 1.0},
            "rr_focus": 2.0,
            "version": 1.0
        }

    def _save_memory(self):
        """Save current weights to disk."""
        with open(self.memory_path, "w") as f:
            json.dump(self.dna_weights, f, indent=2)

    def learn_from_repository(self):
        """
        Analyze the most recent successful strategies and update DNA weights.
        Optimized for 512MB RAM using a rolling window.
        """
        if not os.path.exists(self.strategies_path):
            return

        try:
            with open(self.strategies_path, "r") as f:
                strategies = json.load(f)
        except Exception:
            return

        if not strategies:
            return

        # 🧠 RAM Optimization: Use only the last 50 strategies for 'current' market DNA
        # This prevents the memory usage from growing over time.
        strategies = strategies[-50:]

        # 1. Analyze Keywords
        all_rules = []
        for s in strategies:
            rules = s.get("strategy", {}).get("entry_rules", [])
            all_rules.extend(rules)
        
        # Count frequencies
        counts = collections.Counter(all_rules)
        total = sum(counts.values())
        
        if total > 0:
            self.dna_weights["keywords"] = {k: round(v/total, 4) for k, v in counts.items()}

        # 2. Analyze Timeframes
        tf_counts = collections.Counter([s.get("strategy", {}).get("timeframe") for s in strategies])
        tf_total = sum(tf_counts.values())
        if tf_total > 0:
            self.dna_weights["timeframes"] = {k: round(v/tf_total, 4) for k, v in tf_counts.items()}

        # 3. Analyze R:R
        rr_vals = [s.get("strategy", {}).get("risk_reward", 2.0) for s in strategies]
        if rr_vals:
            self.dna_weights["rr_focus"] = round(sum(rr_vals) / len(rr_vals), 2)

        self._save_memory()
        
        # Explicit cleanup for 512MB RAM protection
        count = len(strategies)
        del strategies
        import gc
        gc.collect()
        
        print(f"🧠 Intelligence: Learned from {count} most recent strategies. DNA Memory updated.")

    def get_weights(self) -> Dict[str, Any]:
        """Return the calculated weights for the generator."""
        return self.dna_weights

# Singleton instance
_brain = IntelligenceModule()

def get_market_experience() -> Dict[str, Any]:
    """Get learned DNA weights."""
    return _brain.get_weights()

def train_brain():
    """Trigger a learning session."""
    _brain.learn_from_repository()
