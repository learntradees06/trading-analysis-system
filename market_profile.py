# Market profile module
# src/market_profile.py
"""Market Profile Module - TPO, Value Area, and Opening Type Analysis"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time
import pytz

class MarketProfile:
    def __init__(self, ticker: str, tick_size: float):
        """
        Initialize Market Profile analyzer
        
        Args:
            ticker: Symbol being analyzed
            tick_size: Minimum price movement for the instrument
        """
        self.ticker = ticker
        self.tick_size = tick_size
        self.tpo_size = tick_size * 2  # Use 2x tick size for TPO blocks
        
    def calculate_tpo_profile(self, df: pd.DataFrame, session_date: datetime) -> Dict:
        """
        Calculate TPO (Time Price Opportunity) profile for a session
        
        Args:
            df: DataFrame with RTH-only intraday data (30min recommended)
            session_date: Date of the session to analyze
        
        Returns:
            Dictionary with TPO profile data
        """
        if df.empty:
            return {}
        
        # Filter to specific session
        session_data = df[df.index.date == session_date.date()]
        
        if session_data.empty:
            return {}
        
        # Create price levels based on tick size
        price_min = session_data['Low'].min()
        price_max = session_data['High'].max()
        
        # Round to nearest tick
        price_min = np.floor(price_min / self.tpo_size) * self.tpo_size
        price_max = np.ceil(price_max / self.tpo_size) * self.tpo_size
        
        # Create price levels
        price_levels = np.arange(price_min, price_max + self.tpo_size, self.tpo_size)
        
        # Initialize TPO counts
        tpo_counts = {price: [] for price in price_levels}
        
        # Assign TPO letters (A, B, C, etc.) for each 30-min period
        tpo_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        for idx, (timestamp, row) in enumerate(session_data.iterrows()):
            if idx >= len(tpo_letters):
                break
                
            letter = tpo_letters[idx]
            period_low = row['Low']
            period_high = row['High']
            
            # Mark all prices touched in this period
            for price in price_levels:
                if period_low <= price <= period_high:
                    tpo_counts[price].append(letter)
        
        # Calculate TPO statistics
        tpo_profile = []
        total_tpos = 0
        
        for price, letters in tpo_counts.items():
            if letters:
                count = len(letters)
                total_tpos += count
                tpo_profile.append({
                    'price': price,
                    'tpo_count': count,
                    'tpo_letters': ''.join(letters)
                })
        
        if not tpo_profile:
            return {}
        
        # Sort by TPO count to find POC
        tpo_profile.sort(key=lambda x: x['tpo_count'], reverse=True)
        poc_price = tpo_profile[0]['price']
        
        # Calculate Value Area (70% of volume/TPOs)
        value_area_tpos = int(total_tpos * 0.70)
        
        # Start from POC and expand outward
        va_prices = [poc_price]
        va_tpo_sum = tpo_profile[0]['tpo_count']
        
        # Get all prices and their counts
        price_tpo_map = {item['price']: item['tpo_count'] for item in tpo_profile}
        
        while va_tpo_sum < value_area_tpos:
            # Find next price above and below current VA
            current_min = min(va_prices)
            current_max = max(va_prices)
            
            price_above = current_max + self.tpo_size
            price_below = current_min - self.tpo_size
            
            tpos_above = price_tpo_map.get(price_above, 0)
            tpos_below = price_tpo_map.get(price_below, 0)
            
            if tpos_above == 0 and tpos_below == 0:
                break
            
            if tpos_above >= tpos_below:
                if price_above in price_tpo_map:
                    va_prices.append(price_above)
                    va_tpo_sum += tpos_above
            else:
                if price_below in price_tpo_map:
                    va_prices.append(price_below)
                    va_tpo_sum += tpos_below
        
        vah = max(va_prices)
        val = min(va_prices)
        
        # Calculate Initial Balance (first hour)
        ib_data = session_data.iloc[:2] if len(session_data) >= 2 else session_data
        ib_high = ib_data['High'].max()
        ib_low = ib_data['Low'].min()
        ib_range = ib_high - ib_low
        
        return {
            'date': session_date,
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'va_width': vah - val,
            'ib_high': ib_high,
            'ib_low': ib_low,
            'ib_range': ib_range,
            'session_high': session_data['High'].max(),
            'session_low': session_data['Low'].min(),
            'session_open': session_data['Open'].iloc[0] if not session_data.empty else None,
            'session_close': session_data['Close'].iloc[-1] if not session_data.empty else None,
            'tpo_profile': tpo_profile,
            'total_tpos': total_tpos
        }
    
    def classify_opening_type(self, current_open: float, prior_va: Dict) -> str:
        """
        Classify the opening type based on prior day's value area
        
        Args:
            current_open: Today's opening price
            prior_va: Dictionary with prior day's VAH, VAL, POC
        
        Returns:
            Opening type classification
        """
        if not prior_va or 'vah' not in prior_va or 'val' not in prior_va:
            return 'Unknown'
        
        prior_vah = prior_va['vah']
        prior_val = prior_va['val']
        prior_poc = prior_va['poc']
        prior_range = prior_vah - prior_val
        
        # Define thresholds
        above_va = current_open > prior_vah
        below_va = current_open < prior_val
        in_va = prior_val <= current_open <= prior_vah
        
        # Distance from value area
        if above_va:
            distance = current_open - prior_vah
            if distance > prior_range * 0.5:
                return 'Open Drive Up'
            else:
                return 'Open Test Drive Up'
        elif below_va:
            distance = prior_val - current_open
            if distance > prior_range * 0.5:
                return 'Open Drive Down'
            else:
                return 'Open Test Drive Down'
        else:
            # Inside value area
            if abs(current_open - prior_poc) < self.tick_size * 2:
                return 'Open Auction In Range'
            elif current_open > prior_poc:
                return 'Open Auction Above POC'
            else:
                return 'Open Auction Below POC'
    
    def analyze_day_type(self, profile: Dict) -> str:
        """
        Analyze and classify the day type based on profile shape
        
        Args:
            profile: Market profile dictionary
        
        Returns:
            Day type classification
        """
        if not profile or 'ib_range' not in profile:
            return 'Unknown'
        
        ib_range = profile['ib_range']
        session_range = profile['session_high'] - profile['session_low']
        
        # Range extension analysis
        ib_high = profile['ib_high']
        ib_low = profile['ib_low']
        session_high = profile['session_high']
        session_low = profile['session_low']
        
        extension_up = max(0, session_high - ib_high)
        extension_down = max(0, ib_low - session_low)
        
        # Classify based on range extension
        if session_range <= ib_range * 1.15:
            return 'Normal Day'
        elif extension_up > ib_range * 0.5 and extension_down < ib_range * 0.15:
            return 'Trend Day Up'
        elif extension_down > ib_range * 0.5 and extension_up < ib_range * 0.15:
            return 'Trend Day Down'
        elif extension_up > ib_range * 0.3 and extension_down > ib_range * 0.3:
            return 'Expanded Range Day'
        else:
            return 'Normal Variation Day'
    
    def calculate_market_internals(self, profile: Dict, prior_profile: Optional[Dict] = None) -> Dict:
        """
        Calculate market internal statistics
        
        Args:
            profile: Current session profile
            prior_profile: Previous session profile
        
        Returns:
            Dictionary with market internal metrics
        """
        internals = {}
        
        if not profile:
            return internals
        
        # TPO distribution analysis
        if 'tpo_profile' in profile and profile['tpo_profile']:
            tpo_counts = [item['tpo_count'] for item in profile['tpo_profile']]
            internals['tpo_skew'] = self._calculate_skew(tpo_counts)
            internals['tpo_concentration'] = max(tpo_counts) / sum(tpo_counts) if sum(tpo_counts) > 0 else 0
        
        # Value area metrics
        if 'va_width' in profile and 'ib_range' in profile:
            internals['va_to_ib_ratio'] = profile['va_width'] / profile['ib_range'] if profile['ib_range'] > 0 else 0
        
        # Prior day relationships
        if prior_profile:
            # Overlap analysis
            if all(k in prior_profile for k in ['vah', 'val']) and all(k in profile for k in ['vah', 'val']):
                overlap_high = min(profile['vah'], prior_profile['vah'])
                overlap_low = max(profile['val'], prior_profile['val'])
                
                if overlap_high > overlap_low:
                    overlap = overlap_high - overlap_low
                    avg_va = (profile['va_width'] + prior_profile['va_width']) / 2
                    internals['va_overlap_percent'] = (overlap / avg_va) * 100 if avg_va > 0 else 0
                else:
                    internals['va_overlap_percent'] = 0
            
            # POC migration
            if 'poc' in prior_profile and 'poc' in profile:
                internals['poc_migration'] = profile['poc'] - prior_profile['poc']
                internals['poc_migration_ticks'] = internals['poc_migration'] / self.tick_size
        
        return internals
    
    def _calculate_skew(self, values: List[float]) -> float:
        """Calculate skewness of a distribution"""
        if not values or len(values) < 3:
            return 0
        
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std == 0:
            return 0
        
        return np.mean(((arr - mean) / std) ** 3)
    
    def get_composite_profile(self, profiles: List[Dict], days: int = 20) -> Dict:
        """
        Create composite profile from multiple days
        
        Args:
            profiles: List of daily profiles
            days: Number of days to include
        
        Returns:
            Composite profile statistics
        """
        if not profiles:
            return {}
        
        # Use most recent N days
        recent_profiles = profiles[-days:] if len(profiles) > days else profiles
        
        # Aggregate all prices and TPOs
        composite_tpos = {}
        
        for profile in recent_profiles:
            if 'tpo_profile' not in profile:
                continue
                
            for item in profile['tpo_profile']:
                price = item['price']
                count = item['tpo_count']
                
                if price not in composite_tpos:
                    composite_tpos[price] = 0
                composite_tpos[price] += count
        
        if not composite_tpos:
            return {}
        
        # Find composite POC
        composite_poc = max(composite_tpos.items(), key=lambda x: x[1])[0]
        
        # Calculate composite value area
        total_tpos = sum(composite_tpos.values())
        va_tpos_needed = int(total_tpos * 0.70)
        
        # Sort prices by TPO count
        sorted_prices = sorted(composite_tpos.items(), key=lambda x: x[1], reverse=True)
        
        va_tpos_sum = 0
        va_prices = []
        
        for price, count in sorted_prices:
            va_prices.append(price)
            va_tpos_sum += count
            if va_tpos_sum >= va_tpos_needed:
                break
        
        composite_vah = max(va_prices) if va_prices else composite_poc
        composite_val = min(va_prices) if va_prices else composite_poc
        
        # Calculate developing value area
        developing_va_width = composite_vah - composite_val
        
        # High volume nodes (HVN) and low volume nodes (LVN)
        avg_tpo = total_tpos / len(composite_tpos) if composite_tpos else 0
        hvn_threshold = avg_tpo * 1.5
        lvn_threshold = avg_tpo * 0.5
        
        hvn_levels = [price for price, count in composite_tpos.items() if count > hvn_threshold]
        lvn_levels = [price for price, count in composite_tpos.items() if count < lvn_threshold]
        
        return {
            'composite_poc': composite_poc,
            'composite_vah': composite_vah,
            'composite_val': composite_val,
            'composite_va_width': developing_va_width,
            'hvn_levels': sorted(hvn_levels),
            'lvn_levels': sorted(lvn_levels),
            'total_days': len(recent_profiles),
            'price_acceptance_zones': self._identify_acceptance_zones(composite_tpos)
        }
    
    def _identify_acceptance_zones(self, tpo_distribution: Dict[float, int]) -> List[Dict]:
        """Identify zones of price acceptance based on TPO concentration"""
        if not tpo_distribution:
            return []
        
        zones = []
        sorted_prices = sorted(tpo_distribution.keys())
        
        if not sorted_prices:
            return []
        
        # Group consecutive high-TPO prices into zones
        avg_tpo = sum(tpo_distribution.values()) / len(tpo_distribution)
        threshold = avg_tpo * 1.2
        
        current_zone = None
        
        for price in sorted_prices:
            tpo_count = tpo_distribution[price]
            
            if tpo_count >= threshold:
                if current_zone is None:
                    current_zone = {
                        'start': price,
                        'end': price,
                        'total_tpos': tpo_count,
                        'prices': [price]
                    }
                elif price - current_zone['end'] <= self.tpo_size * 2:
                    # Extend current zone
                    current_zone['end'] = price
                    current_zone['total_tpos'] += tpo_count
                    current_zone['prices'].append(price)
                else:
                    # Save current zone and start new one
                    zones.append(current_zone)
                    current_zone = {
                        'start': price,
                        'end': price,
                        'total_tpos': tpo_count,
                        'prices': [price]
                    }
            elif current_zone is not None:
                # End current zone
                zones.append(current_zone)
                current_zone = None
        
        # Don't forget last zone
        if current_zone is not None:
            zones.append(current_zone)
        
        # Calculate zone strength
        for zone in zones:
            zone['strength'] = zone['total_tpos'] / sum(tpo_distribution.values())
            zone['midpoint'] = (zone['start'] + zone['end']) / 2
        
        return zones