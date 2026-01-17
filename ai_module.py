"""
AI Internet Brain Module - Powered by Google Gemini.

Provides high-level market context and strategy bias using AI 
to help the research engine adapt to real-time internet context.
"""

import os
import json
import logging
import google.generativeai as genai
from config import Config

logger = logging.getLogger(__name__)

class AIInternetBrain:
    """
    Interfaces with Gemini AI to provide market context.
    Optimized for 512MB RAM with concise prompting and memory cleanup.
    """
    
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        self.enabled = bool(self.api_key)
        if self.enabled:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

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
        if not self.enabled or not self.model:
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
            response = self.model.generate_content(prompt)
            # Safe JSON extraction
            text = response.text
            # Basic cleanup for accidental markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            bias = json.loads(text)
            logger.info(f"🧠 AI Insight: {bias.get('sentiment')} | Mode: {bias.get('mode')}")
            return bias
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
