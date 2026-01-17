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
from config import Config

logger = logging.getLogger(__name__)

# Gemini API Endpoint (v1beta)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


class AIInternetBrain:
    """
    Interfaces with Gemini AI to provide market context.
    Uses direct HTTP requests for maximum compatibility.
    Optimized for 512MB RAM with concise prompting.
    """
    
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        self.enabled = bool(self.api_key)

    def get_market_bias(self, recent_performance: str = "") -> dict:
        """
        Get AI-driven market bias.
        
        Returns:
            dict: {
                "sentiment": "BULLISH/BEARISH/NEUTRAL",
                "mode": "SCALPING/TREND/RANGE",
                "bias_weight": float (0.5 to 1.5),
                "concept_focus": "SMC/PRICE_ACTION/TECHNICAL"
            }
        """
        if not self.enabled:
            return self._default_bias()

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

        try:
            # Direct REST API call
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
            
            if response.status_code != 200:
                logger.error(f"⚠️ AI Brain Error: {response.status_code} - {response.text[:200]}")
                return self._default_bias()
            
            result = response.json()
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Basic cleanup for accidental markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            bias = json.loads(text)
            logger.info(f"🧠 AI Insight: {bias.get('sentiment')} | Mode: {bias.get('mode')}")
            return bias
            
        except requests.exceptions.Timeout:
            logger.error("⚠️ AI Brain Error: Request timed out")
            return self._default_bias()
        except Exception as e:
            logger.error(f"⚠️ AI Brain Error: {e}")
            return self._default_bias()
        finally:
            # RAM Cleanup
            import gc
            gc.collect()

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
