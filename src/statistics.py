# src/statistics.py
"""Advanced Statistical Analysis Engine for Opening Type Probabilities"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

class StatisticalAnalyzer:
    def __init__(self, ticker: str):
        """
        Initialize Statistical Analyzer
        
        Args:
            ticker: Symbol being analyzed
        """
        self.ticker = ticker
        
        # Map our opening types to standard classifications
        self.opening_type_mapping = {
            'Open Drive Up': 'HOR',  # Higher Open, Outside Range
            'Open Test Drive Up': 'HOR',  # Higher Open, Outside Range (moderate)
            'Open Auction Above POC': 'HIR',  # Higher Inside Range (above POC)
            'Open Auction In Range': 'HIR',  # Inside Range near POC
            'Open Auction Below POC': 'LIR',  # Lower Inside Range (below POC)
            'Open Test Drive Down': 'LOR',  # Lower Open, Outside Range (moderate)
            'Open Drive Down': 'LOR',  # Lower Open, Outside Range
            'Unknown': 'Unknown'
        }
        
        # Define all events to track
        self.events_to_track = [
            'IBH', 'IBL', 'IBH or IBL', 'IBH & IBL (Neutral)',
            '1.5X IBH', '1.5X IBL', '2X IBH', '2X IBL',
            'pVAH', 'pVAL', 'pPOC', 'pCL (Gap)',
            'pHI (Range Gap)', 'pLO', 'pMID', '1/2 Gap',
            'Inside Day', 'Outside Day'
        ]
    
    def calculate_opening_type_statistics(self, profiles: List[Dict], 
                                         intraday_data: Dict[datetime, pd.DataFrame]) -> Dict:
        """
        Calculate historical probabilities for each opening type
        
        Args:
            profiles: List of daily market profile dictionaries
            intraday_data: Dictionary mapping dates to intraday DataFrames
        
        Returns:
            Dictionary with statistics for each opening type
        """
        # Initialize storage for results
        results_by_type = defaultdict(list)
        
        # Process each day (starting from second day for prior comparisons)
        for i in range(1, len(profiles)):
            current_profile = profiles[i]
            prior_profile = profiles[i-1]
            
            # Skip if profiles are incomplete
            if not current_profile or not prior_profile:
                continue
            
            # Get the date and intraday data
            current_date = current_profile.get('date')
            if not current_date or current_date not in intraday_data:
                continue
            
            current_data = intraday_data[current_date]
            if current_data.empty:
                continue
            
            # Determine opening type
            opening_type_raw = current_profile.get('opening_type', 'Unknown')
            opening_type = self.opening_type_mapping.get(opening_type_raw, 'Unknown')
            
            if opening_type == 'Unknown':
                continue
            
            # Calculate all events for this day
            events = self._check_all_events(current_profile, prior_profile, current_data)
            
            # Store results
            results_by_type[opening_type].append(events)
        
        # Calculate statistics for each opening type
        statistics = {}
        
        for opening_type in ['HOR', 'HIR', 'LIR', 'LOR']:
            if opening_type in results_by_type and results_by_type[opening_type]:
                day_results = results_by_type[opening_type]
                stats = {}
                
                for event in self.events_to_track:
                    # Calculate percentage of days this event occurred
                    occurrences = sum(1 for day in day_results if day.get(event, False))
                    total_days = len(day_results)
                    percentage = (occurrences / total_days * 100) if total_days > 0 else 0
                    stats[event] = round(percentage, 2)
                
                statistics[opening_type] = {
                    'count': len(day_results),
                    'stats': stats
                }
                # Add a warning for small sample sizes
                MIN_SAMPLES = 5 # Or get this from config
                if len(day_results) < MIN_SAMPLES:
                    # You could add a 'warning' key to the results dictionary
                    statistics[opening_type]['warning'] = f"Warning: Statistics based on only {len(day_results)} samples."
            else:
                # No data for this opening type
                statistics[opening_type] = {
                    'count': 0,
                    'stats': {event: 0.0 for event in self.events_to_track}
                }
        
        # Add summary statistics
        statistics['summary'] = self._calculate_summary_statistics(statistics)
        
        return statistics
    
    def _check_all_events(self, current_profile: Dict, prior_profile: Dict, 
                         current_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Check all statistical events for a single day
        
        Args:
            current_profile: Current day's market profile
            prior_profile: Previous day's market profile
            current_data: Current day's intraday data
        
        Returns:
            Dictionary with boolean results for each event
        """
        events = {}
        
        # Get key values
        session_high = current_data['High'].max()
        session_low = current_data['Low'].min()
        session_open = current_data['Open'].iloc[0] if not current_data.empty else None
        
        # IB values
        ib_high = current_profile.get('ib_high', 0)
        ib_low = current_profile.get('ib_low', 0)
        ib_range = ib_high - ib_low if ib_high and ib_low else 0
        
        # Prior day values
        prior_vah = prior_profile.get('vah', 0)
        prior_val = prior_profile.get('val', 0)
        prior_poc = prior_profile.get('poc', 0)
        prior_close = prior_profile.get('session_close', 0)
        prior_high = prior_profile.get('session_high', 0)
        prior_low = prior_profile.get('session_low', 0)
        prior_mid = (prior_high + prior_low) / 2 if prior_high and prior_low else 0
        
        # Check each event
        
        # Initial Balance events
        events['IBH'] = session_high >= ib_high if ib_high else False
        events['IBL'] = session_low <= ib_low if ib_low else False
        events['IBH or IBL'] = events['IBH'] or events['IBL']
        events['IBH & IBL (Neutral)'] = events['IBH'] and events['IBL']
        
        # IB Extension events
        if ib_range > 0:
            events['1.5X IBH'] = session_high >= (ib_high + 0.5 * ib_range)
            events['1.5X IBL'] = session_low <= (ib_low - 0.5 * ib_range)
            events['2X IBH'] = session_high >= (ib_high + 1.0 * ib_range)
            events['2X IBL'] = session_low <= (ib_low - 1.0 * ib_range)
        else:
            events['1.5X IBH'] = False
            events['1.5X IBL'] = False
            events['2X IBH'] = False
            events['2X IBL'] = False
        
        # Prior Value Area events
        events['pVAH'] = session_high >= prior_vah if prior_vah else False
        events['pVAL'] = session_low <= prior_val if prior_val else False
        events['pPOC'] = (session_high >= prior_poc and session_low <= prior_poc) if prior_poc else False
        
        # Prior Close and Gap events
        events['pCL (Gap)'] = (session_high >= prior_close and session_low <= prior_close) if prior_close else False
        
        # Prior Range events
        events['pHI (Range Gap)'] = session_high >= prior_high if prior_high else False
        events['pLO'] = session_low <= prior_low if prior_low else False
        events['pMID'] = (session_high >= prior_mid and session_low <= prior_mid) if prior_mid else False
        
        # Gap fill events
        if session_open and prior_close:
            gap_fill = (prior_close + session_open) / 2
            events['1/2 Gap'] = session_high >= gap_fill and session_low <= gap_fill
        else:
            events['1/2 Gap'] = False
        
        # Day type events
        events['Inside Day'] = (session_high < prior_high and session_low > prior_low) if prior_high and prior_low else False
        events['Outside Day'] = (session_high > prior_high and session_low < prior_low) if prior_high and prior_low else False
        
        return events
    
    def _calculate_summary_statistics(self, statistics: Dict) -> Dict:
        """
        Calculate summary statistics across all opening types
        
        Args:
            statistics: Dictionary with statistics for each opening type
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_days_analyzed': sum(stats['count'] for stats in statistics.values() if 'count' in stats),
            'most_common_opening': None,
            'highest_ib_break_probability': {'type': None, 'value': 0},
            'key_observations': []
        }
        
        # Find most common opening type
        opening_counts = {otype: stats['count'] for otype, stats in statistics.items() if otype != 'summary'}
        if opening_counts:
            summary['most_common_opening'] = max(opening_counts, key=opening_counts.get)
        
        # Find highest IB break probability
        for otype, stats in statistics.items():
            if otype != 'summary' and 'stats' in stats:
                ib_break_prob = stats['stats'].get('IBH or IBL', 0)
                if ib_break_prob > summary['highest_ib_break_probability']['value']:
                    summary['highest_ib_break_probability'] = {
                        'type': otype,
                        'value': ib_break_prob
                    }
        
        # Generate key observations
        for otype, stats in statistics.items():
            if otype != 'summary' and 'stats' in stats and stats['count'] > 0:
                # High probability events (>70%)
                high_prob_events = [
                    event for event, prob in stats['stats'].items() 
                    if prob > 70
                ]
                if high_prob_events:
                    summary['key_observations'].append(
                        f"{otype}: High probability (>70%) for {', '.join(high_prob_events[:3])}"
                    )
        
        return summary
    
    def format_statistics_table(self, statistics: Dict) -> str:
        """
        Format statistics as a beautiful ASCII table for console output
        
        Args:
            statistics: Dictionary with statistics
        
        Returns:
            Formatted string table
        """
        # Create header
        output = "\n" + "=" * 100 + "\n"
        output += f"  OPENING TYPE STATISTICAL ANALYSIS - {self.ticker}\n"
        output += "=" * 100 + "\n\n"
        
        # Create table header
        output += f"{'Event':<25} {'HOR':>12} {'HIR':>12} {'LIR':>12} {'LOR':>12}\n"
        output += f"{'':.<25} {'(n=' + str(statistics.get('HOR', {}).get('count', 0)) + ')':>12} "
        output += f"{'(n=' + str(statistics.get('HIR', {}).get('count', 0)) + ')':>12} "
        output += f"{'(n=' + str(statistics.get('LIR', {}).get('count', 0)) + ')':>12} "
        output += f"{'(n=' + str(statistics.get('LOR', {}).get('count', 0)) + ')':>12}\n"
        output += "-" * 100 + "\n"
        
        # Format each event row
        for event in self.events_to_track:
            row = f"{event:<25}"
            for otype in ['HOR', 'HIR', 'LIR', 'LOR']:
                if otype in statistics and 'stats' in statistics[otype]:
                    value = statistics[otype]['stats'].get(event, 0)
                    # Color code based on probability
                    if value >= 70:
                        row += f"  ▲ {value:>6.1f}%"
                    elif value >= 50:
                        row += f"  ● {value:>6.1f}%"
                    else:
                        row += f"    {value:>6.1f}%"
                else:
                    row += f"    {'--':>6}"
            output += row + "\n"
        
        output += "-" * 100 + "\n"
        
        # Add summary
        if 'summary' in statistics:
            summary = statistics['summary']
            output += f"\n📊 SUMMARY:\n"
            output += f"  • Total Days Analyzed: {summary['total_days_analyzed']}\n"
            output += f"  • Most Common Opening: {summary['most_common_opening']}\n"
            output += f"  • Highest IB Break Probability: {summary['highest_ib_break_probability']['type']} "
            output += f"({summary['highest_ib_break_probability']['value']:.1f}%)\n"
            
            if summary['key_observations']:
                output += f"\n🔍 KEY OBSERVATIONS:\n"
                for obs in summary['key_observations'][:5]:
                    output += f"  • {obs}\n"
        
        output += "\n" + "=" * 100 + "\n"
        
        return output
    
    def get_event_descriptions(self) -> Dict[str, str]:
        """
        Get human-readable descriptions for each event
        
        Returns:
            Dictionary mapping event names to descriptions
        """
        return {
            'IBH': 'Initial Balance High touched',
            'IBL': 'Initial Balance Low touched',
            'IBH or IBL': 'Either IB High or Low touched',
            'IBH & IBL (Neutral)': 'Both IB High and Low touched',
            '1.5X IBH': '1.5x IB range extension above IBH',
            '1.5X IBL': '1.5x IB range extension below IBL',
            '2X IBH': '2x IB range extension above IBH',
            '2X IBL': '2x IB range extension below IBL',
            'pVAH': 'Prior day Value Area High touched',
            'pVAL': 'Prior day Value Area Low touched',
            'pPOC': 'Prior day Point of Control touched',
            'pCL (Gap)': 'Prior day Close (gap fill)',
            'pHI (Range Gap)': 'Prior day High exceeded',
            'pLO': 'Prior day Low broken',
            'pMID': 'Prior day midpoint touched',
            '1/2 Gap': 'Half gap fill achieved',
            'Inside Day': 'Range inside prior day range',
            'Outside Day': 'Range outside prior day range'
        }