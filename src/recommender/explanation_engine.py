"""
Explanation Engine - Simplified and Unified
--------------------------------------------
Converts recommendation signals into user-friendly explanations.
All explanation logic is centralized here.
"""

from typing import Dict, List, Any


def generate_explanation(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate explanation from recommendation signals.
    
    Args:
        signals: List of signal dicts with:
                 - 'source': 'IBCF', 'CBF', 'UBCF', 'Watchlist'
                 - 'strength': float (0-1) or 'strong'/'moderate'/'weak'
                 - 'context_items': List of movie titles
                 - 'features': List of keywords/themes
    
    Returns:
        {
            'primary_reason': str,
            'secondary_reasons': [str, ...],
            'confidence_level': 'Strong'/'Moderate'/'Weak',
            'signal_source': str
        }
    """
    if not signals:
        return {
            "primary_reason": "Recommended based on overall similarity to your preferences.",
            "secondary_reasons": [],
            "confidence_level": "Weak",
            "signal_source": "General"
        }
    
    # Sort by strength
    def strength_val(s):
        val = s.get('strength', 0)
        if isinstance(val, str):
            return {'strong': 3, 'moderate': 2, 'weak': 1}.get(val.lower(), 1)
        return val
    
    sorted_signals = sorted(signals, key=strength_val, reverse=True)
    primary_signal = sorted_signals[0]
    secondary_signals = sorted_signals[1:3]
    
    # Generate reasons
    primary_text = _format_reason(primary_signal)
    secondary_texts = [_format_reason(s) for s in secondary_signals]
    
    return {
        "primary_reason": primary_text,
        "secondary_reasons": secondary_texts,
        "confidence_level": _map_strength(primary_signal.get('strength')),
        "signal_source": primary_signal.get('source', 'Unknown')
    }


def _map_strength(raw_strength) -> str:
    """Convert numeric or string strength to standard format."""
    if isinstance(raw_strength, (int, float)):
        if raw_strength > 0.8:
            return "Strong"
        if raw_strength > 0.5:
            return "Moderate"
        return "Weak"
    return str(raw_strength).capitalize()


def _format_reason(signal: Dict[str, Any]) -> str:
    """Format a single signal into a readable explanation."""
    source = signal.get('source', '').upper()
    items = [str(i) for i in signal.get('context_items', []) if i]
    features = [str(f).lower() for f in signal.get('features', []) if f]
    
    # WATCHLIST explanations
    if source == 'WATCHLIST':
        if "direct match" in features:
            return "This movie is in your watchlist."
        if items:
            return f"Similar to '{items[0]}' which is in your watchlist."
        return "Matches the style of movies in your watchlist."
    
    # IBCF explanations (collaborative filtering)
    if source == 'IBCF':
        if items:
            return f"Users who liked '{items[0]}' also enjoyed this movie."
        return "Popular among users with similar taste to yours."
    
    # CBF explanations (content-based)
    if source == 'CBF':
        if items:
            base = f"Resembles '{items[0]}'"
            if features:
                feats = ", ".join(features[:3])
                return f"{base} due to shared themes like {feats}."
            return f"{base} in tone and style."
        
        if features:
            feats = ", ".join(features[:3])
            return f"Recommended for its focus on {feats}."
        
        return "Aligns with your typical movie preferences."
    
    # UBCF explanations
    if source == 'UBCF':
        return "Popular among users with similar viewing history."
    
    # Fallback
    return "Recommended based on overall similarity to your preferences."


# Backward compatibility: Keep class-based API
class ExplanationEngine:
    """Legacy class wrapper for backward compatibility."""
    
    @staticmethod
    def generate_explanation(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        return generate_explanation(signals)
    
    @staticmethod
    def _map_strength(raw_strength):
        return _map_strength(raw_strength)
    
    @staticmethod
    def _format_reason(signal: Dict[str, Any], is_primary: bool = True) -> str:
        return _format_reason(signal)
