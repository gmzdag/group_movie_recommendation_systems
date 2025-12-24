    def _generate_detailed_explanation(self, rec: Dict[str, Any], group_users: List[int]) -> Dict[str, Any]:
        """
        Generate detailed explanation for a recommendation.
        
        Returns a rich object with:
        - model_info: Which hybrid model was used
        - signal_breakdown: Percentage breakdown of signals (CF, Content, Watchlist)
        - user_reasons: Per-user explanations
        - group_summary: Why this movie fits the group
        """
        source_model = rec.get('source_model', 'H1')
        user_explanations = rec.get('user_explanations', {})
        
        # Model name mapping
        model_names = {
            'H1': 'Dynamic Weighted Hybrid',
            'H2': 'Switching Hybrid',
            'H3': 'Watchlist-Enhanced Hybrid',
            'AI_AGENT': 'AI-Guided Selection'
        }
        
        # Signal breakdown
        signal_counts = Counter()
        for expl in user_explanations.values():
            signal = expl.get('signal_source', 'Unknown')
            signal_counts[signal] += 1
        
        total_signals = sum(signal_counts.values())
        signal_breakdown = {}
        if total_signals > 0:
            for signal, count in signal_counts.items():
                percentage = (count / total_signals) * 100
                signal_breakdown[signal] = round(percentage, 1)
        
        # User-specific reasons
        user_reasons = []
        for uid in group_users:
            if uid in user_explanations:
                expl = user_explanations[uid]
                user_reasons.append({
                    'user_id': uid,
                    'primary_reason': expl.get('primary_reason', 'Recommended for you'),
                    'signal_source': expl.get('signal_source', 'Unknown'),
                    'confidence': expl.get('confidence', 'Medium')
                })
        
        # Group summary
        group_summary = rec.get('group_explanation', 'Recommended for the group')
        
        return {
            'model_info': {
                'model_code': source_model,
                'model_name': model_names.get(source_model, 'Hybrid Model'),
                'model_description': self._get_model_description(source_model)
            },
            'signal_breakdown': signal_breakdown,
            'user_reasons': user_reasons,
            'group_summary': group_summary,
            'recommendation_strength': rec.get('group_score', 0.0)
        }
    
    def _get_model_description(self, model_code: str) -> str:
        """Get user-friendly description of the model."""
        descriptions = {
            'H1': 'Dynamically balances collaborative filtering, content similarity, and watchlist preferences based on data availability.',
            'H2': 'Intelligently switches between different recommendation strategies based on user profile characteristics.',
            'H3': 'Emphasizes items similar to movies in your watchlists, ensuring personalized future viewing suggestions.',
            'AI_AGENT': 'Uses advanced AI to understand your natural language preferences and find the perfect match.'
        }
        return descriptions.get(model_code, 'Hybrid recommendation model combining multiple signals.')
