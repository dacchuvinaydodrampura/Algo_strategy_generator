"""
AI Internet Brain Module - Powered by Google Gemini.

Provides high-level market context and strategy bias using AI 
to help the research engine adapt to real-time internet context.

Uses direct REST API calls to avoid library version issues.
"""

import os
import json
import logging
import requests
import time
from config import Config

logger = logging.getLogger(__name__)

# Gemini API Endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


class AIInternetBrain:
    """
    Interfaces with Gemini AI to provide market context.
    Uses direct HTTP requests with caching and retries.
    """
    
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        self.enabled = bool(self.api_key)
        self.last_bias = self._default_bias()
        self.last_pull_time = 0
        self.cache_duration = 900  # 15 minutes cache

    def get_market_bias(self, recent_performance: str = "") -> dict:
        """Get AI-driven market bias with caching and retries."""
        if not self.enabled:
            return self._default_bias()

        # Check Cache
        current_time = time.time()
        if (current_time - self.last_pull_time) < self.cache_duration:
            logger.debug("🧠 AI Insight: Using cached data")
            return self.last_bias

        prompt = f"""
        Act as an Advanced Institutional Trading AI. 
        Context: The research engine is currently searching for strategies on Nifty 50.
        Recent Engine Performance: {recent_performance}
        
        Task: Provide a JSON-only response for the current market bias based on your internal knowledge of global market trends as of today.
        
        Format:
        {{
            "sentiment": "BULLISH" or "BEARISH" or "NEUTRAL",
            "mode": "SCALPING" (High Vol) or "TREND" (Smooth) or "RANGE" (Chop),
            "bias_weight": 0.5 to 2.0 (Directional strength),
            "concept_focus": "INSTITUTIONAL" (SMC) or "PRICE_ACTION" or "TECHNICAL"
        }}
        """

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{GEMINI_API_URL}?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }]
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0].strip()
                    
                    bias = json.loads(text)
                    self.last_bias = bias
                    self.last_pull_time = current_time
                    logger.info(f"🧠 AI Insight: {bias.get('sentiment')} | Mode: {bias.get('mode')}")
                    return bias
                
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * 5
                    logger.warning(f"⚠️ AI Quota Hit (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"⚠️ AI Brain Error: {response.status_code} - {response.text[:100]}")
                    return self.last_bias  # Fallback to cached
                    
            except Exception as e:
                logger.error(f"⚠️ AI Brain Request Error: {e}")
                return self.last_bias

        return self.last_bias

    def _default_bias(self) -> dict:
        """Fallback when AI is disabled or fails."""
        return {
            "sentiment": "NEUTRAL",
            "mode": "TREND",
            "bias_weight": 1.0,
            "concept_focus": "INSTITUTIONAL"
        }

# Singleton instance
_ai_brain = AIInternetBrain()

def get_ai_insight(context_summary: str = "") -> dict:
    """Get summarized AI insight for the generator."""
    return _ai_brain.get_market_bias(context_summary)
