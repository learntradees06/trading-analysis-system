# src/statistics.py
"""Advanced Statistical Analysis Engine for Opening Type Probabilities"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

class StatisticalAnalyzer:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.key_levels = [
            'IBH', 'IBL', 'pVAH', 'pVAL', 'pPOC',
            'pHI', 'pLO', 'pMID', 'pCL'
        ]

    def calculate_opening_type_statistics(self, profiles: List[Dict]) -> Dict:
        results_by_type = defaultdict(list)
        for i in range(1, len(profiles)):
            current_profile = profiles[i]
            prior_profile = profiles[i-1]
            opening_type = current_profile.get('opening_type', 'Unknown')
            if opening_type != 'Unknown':
                day_results = self._analyze_day_levels(current_profile, prior_profile)
                results_by_type[opening_type].append(day_results)
        
        statistics = {}
        for otype in ['HOR', 'HIR', 'LIR', 'LOR']:
            day_results = results_by_type.get(otype, [])
            stats = {'close_above': {}, 'broken_during_rth': {}}
            total_days = len(day_results)
            if total_days > 0:
                for level_name in self.key_levels:
                    close_above_count = sum(d['close_above'].get(level_name, False) for d in day_results)
                    broken_count = sum(d['broken_during_rth'].get(level_name, False) for d in day_results)
                    stats['close_above'][level_name] = (close_above_count / total_days) * 100
                    stats['broken_during_rth'][level_name] = (broken_count / total_days) * 100
            statistics[otype] = {'count': total_days, 'stats': stats}
        return statistics

    def _analyze_day_levels(self, current_profile: Dict, prior_profile: Dict) -> Dict:
        session_high = current_profile.get('session_high', 0)
        session_low = current_profile.get('session_low', 0)
        session_close = current_profile.get('session_close', 0)

        levels = {
            'IBH': current_profile.get('ib_high'), 'IBL': current_profile.get('ib_low'),
            'pVAH': prior_profile.get('vah'), 'pVAL': prior_profile.get('val'),
            'pPOC': prior_profile.get('poc'), 'pHI': prior_profile.get('session_high'),
            'pLO': prior_profile.get('session_low'), 'pCL': prior_profile.get('session_close'),
            'pMID': (prior_profile.get('session_high', 0) + prior_profile.get('session_low', 0)) / 2
        }
        
        results = {'close_above': {}, 'broken_during_rth': {}}
        for name, value in levels.items():
            if value is not None:
                results['close_above'][name] = session_close >= value
                results['broken_during_rth'][name] = session_high >= value and session_low <= value
        return results

    def get_probabilities_for_day(self, opening_type: str, all_stats: Dict) -> Dict:
        if opening_type not in all_stats:
            return {}
        stats = all_stats[opening_type]['stats']
        flat_dict = {}
        for stat_type, levels in stats.items():
            for level, prob in levels.items():
                flat_dict[f"stat_{stat_type}_{level}"] = prob
        return flat_dict

    def display_statistics(self, statistics: Dict):
        print("\n" + "="*80)
        print(f"  STATISTICAL ANALYSIS FOR: {self.ticker}")
        print("="*80)
        self._format_table("CLOSE ABOVE PROBABILITY", statistics, 'close_above')
        self._format_table("BROKEN DURING RTH PROBABILITY", statistics, 'broken_during_rth')

    def _format_table(self, title: str, statistics: Dict, stat_type: str):
        print("\n" + "-"*80)
        print(f"  {title}")
        print("-" * 80)
        header = f"{'Level':<10} | {'HOR':>10} | {'HIR':>10} | {'LIR':>10} | {'LOR':>10}"
        print(header)
        counts = f"{'Days':<10} | {statistics.get('HOR', {}).get('count', 0):>10} | {statistics.get('HIR', {}).get('count', 0):>10} | {statistics.get('LIR', {}).get('count', 0):>10} | {statistics.get('LOR', {}).get('count', 0):>10}"
        print(counts)
        print("-" * 80)
        for level in self.key_levels:
            row = f"{level:<10} | "
            for otype in ['HOR', 'HIR', 'LIR', 'LOR']:
                prob = statistics.get(otype, {}).get('stats', {}).get(stat_type, {}).get(level, 0)
                row += f"{prob:>9.1f}% | "
            print(row)
        print("-" * 80)
