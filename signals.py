# Signal generation module
# src/signals.py
"""Signal Generation Module with Weighted Confluence Score"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class SignalGenerator:
    def __init__(self, ticker: str):
        """
        Initialize Signal Generator
        
        Args:
            ticker: Symbol being analyzed
        """
        self.ticker = ticker
        
        # Weights for different components (must sum to 100%)
        self.weights = {
            'market_profile': 0.30,
            'technical': 0.25,
            'sr_levels': 0.20,
            'ml_prediction': 0.25
        }
    
    def generate_signal(self, 
                       market_profile: Dict,
                       technical_data: pd.Series,
                       sr_analysis: Dict,
                       ml_predictions: Dict,
                       statistics: Optional[Dict] = None) -> Dict:
        """
        Generate trading signal with weighted confluence score
        
        Args:
            market_profile: Current market profile data
            technical_data: Current technical indicators
            sr_analysis: Support/resistance analysis
            ml_predictions: ML model predictions
            statistics: Historical statistics for context
        
        Returns:
            Dictionary with signal, confidence, and evidence
        """
        scores = {}
        evidence = []
        
        # 1. Market Profile Score (30% weight)
        mp_score, mp_evidence = self._calculate_market_profile_score(market_profile, statistics)
        scores['market_profile'] = mp_score
        evidence.extend(mp_evidence)
        
        # 2. Technical Score (25% weight)
        tech_score, tech_evidence = self._calculate_technical_score(technical_data)
        scores['technical'] = tech_score
        evidence.extend(tech_evidence)
        
        # 3. S/R Score (20% weight)
        sr_score, sr_evidence = self._calculate_sr_score(sr_analysis, technical_data)
        scores['sr_levels'] = sr_score
        evidence.extend(sr_evidence)
        
        # 4. ML Score (25% weight)
        ml_score, ml_evidence = self._calculate_ml_score(ml_predictions)
        scores['ml_prediction'] = ml_score
        evidence.extend(ml_evidence)
        
        # Calculate weighted total score
        total_score = sum(scores[component] * self.weights[component] 
                         for component in self.weights.keys())
        
        # Determine signal
        if total_score >= 70:
            signal = 'STRONG LONG'
            confidence = 'HIGH'
        elif total_score >= 55:
            signal = 'LONG'
            confidence = 'MEDIUM'
        elif total_score <= 30:
            signal = 'STRONG SHORT'
            confidence = 'HIGH'
        elif total_score <= 45:
            signal = 'SHORT'
            confidence = 'MEDIUM'
        else:
            signal = 'NEUTRAL'
            confidence = 'LOW'
        
        return {
            'signal': signal,
            'score': round(total_score, 2),
            'confidence': confidence,
            'component_scores': scores,
            'evidence': evidence,
            'timestamp': pd.Timestamp.now()
        }
    
    def _calculate_market_profile_score(self, profile: Dict, statistics: Optional[Dict]) -> Tuple[float, List[str]]:
        """Calculate score based on Market Profile context"""
        score = 50  # Start neutral
        evidence = []
        
        if not profile:
            return score, evidence
        
        # Opening type analysis
        opening_type = profile.get('opening_type', 'Unknown')
        
        # Bullish opening types
        if 'Drive Up' in opening_type:
            score += 20
            evidence.append(f"Bullish opening type: {opening_type}")
        elif 'Above POC' in opening_type:
            score += 10
            evidence.append(f"Opening above POC")
        elif 'Below POC' in opening_type:
            score -= 10
            evidence.append(f"Opening below POC")
        elif 'Drive Down' in opening_type:
            score -= 20
            evidence.append(f"Bearish opening type: {opening_type}")
        
        # Value area position
        current_price = profile.get('session_close', 0)
        vah = profile.get('vah', 0)
        val = profile.get('val', 0)
        poc = profile.get('poc', 0)
        
        if current_price and vah and val:
            if current_price > vah:
                score += 15
                evidence.append("Price above Value Area High")
            elif current_price < val:
                score -= 15
                evidence.append("Price below Value Area Low")
            elif current_price > poc:
                score += 5
                evidence.append("Price above POC")
            else:
                score -= 5
                evidence.append("Price below POC")
        
        # Add statistical context if available
        if statistics and opening_type != 'Unknown':
            # Map opening type to statistical categories
            opening_type_mapping = {
                'Open Drive Up': 'HOR',
                'Open Test Drive Up': 'HOR',
                'Open Auction Above POC': 'HIR',
                'Open Auction In Range': 'HIR',
                'Open Auction Below POC': 'LIR',
                'Open Test Drive Down': 'LOR',
                'Open Drive Down': 'LOR'
            }
            
            stat_type = opening_type_mapping.get(opening_type)
            if stat_type and stat_type in statistics:
                stats = statistics[stat_type].get('stats', {})
                
                # High probability events
                if stats.get('IBH', 0) > 70:
                    score += 10
                    evidence.append(f"High probability ({stats['IBH']:.1f}%) of IBH break")
                elif stats.get('IBL', 0) > 70:
                    score -= 10
                    evidence.append(f"High probability ({stats['IBL']:.1f}%) of IBL break")
        
        # Day type influence
        day_type = profile.get('day_type', 'Unknown')
        if 'Trend Day Up' in day_type:
            score += 15
            evidence.append("Trend Day Up detected")
        elif 'Trend Day Down' in day_type:
            score -= 15
            evidence.append("Trend Day Down detected")
        
        return max(0, min(100, score)), evidence
    
    def _calculate_technical_score(self, technical_data: pd.Series) -> Tuple[float, List[str]]:
        """Calculate score based on technical indicators"""
        score = 50  # Start neutral
        evidence = []
        
        if technical_data is None or (isinstance(technical_data, pd.Series) and technical_data.empty):
            return score, evidence
        
        # RSI analysis
        rsi = technical_data.get('RSI', 50)
        if rsi > 70:
            score -= 10
            evidence.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60:
            score += 10
            evidence.append(f"RSI bullish ({rsi:.1f})")
        elif rsi < 30:
            score += 10
            evidence.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            score -= 10
            evidence.append(f"RSI bearish ({rsi:.1f})")
        
        # ADX trend strength
        adx = technical_data.get('ADX', 25)
        if adx > 25:
            # Strong trend
            if technical_data.get('Plus_DI', 0) > technical_data.get('Minus_DI', 0):
                score += 15
                evidence.append(f"Strong uptrend (ADX: {adx:.1f})")
            else:
                score -= 15
                evidence.append(f"Strong downtrend (ADX: {adx:.1f})")
        else:
            evidence.append(f"Weak trend (ADX: {adx:.1f})")
        
        # MACD
        macd_histogram = technical_data.get('MACD_Histogram', 0)
        if macd_histogram > 0:
            score += 10
            evidence.append("MACD positive")
        else:
            score -= 10
            evidence.append("MACD negative")
        
        # Bollinger Bands
        bb_percent = technical_data.get('BB_PercentB', 0.5)
        if bb_percent > 1:
            score -= 5
            evidence.append("Price above upper BB")
        elif bb_percent > 0.8:
            score += 5
            evidence.append("Price near upper BB")
        elif bb_percent < 0:
            score += 5
            evidence.append("Price below lower BB")
        elif bb_percent < 0.2:
            score -= 5
            evidence.append("Price near lower BB")
        
        # Stochastic
        stoch_k = technical_data.get('Stoch_K', 50)
        if stoch_k > 80:
            score -= 5
            evidence.append(f"Stochastic overbought ({stoch_k:.1f})")
        elif stoch_k < 20:
            score += 5
            evidence.append(f"Stochastic oversold ({stoch_k:.1f})")
        
        # EMA Alignment
        if 'EMA_9' in technical_data and 'EMA_21' in technical_data:
            if technical_data['EMA_9'] > technical_data['EMA_21']:
                score += 5
                evidence.append("Short-term EMAs bullish (9 > 21)")
            else:
                score -= 5
                evidence.append("Short-term EMAs bearish (9 < 21)")
        
        if 'EMA_50' in technical_data and 'EMA_200' in technical_data:
            if technical_data['EMA_50'] > technical_data['EMA_200']:
                score += 5
                evidence.append("Long-term EMAs bullish (50 > 200)")
            else:
                score -= 5
                evidence.append("Long-term EMAs bearish (50 < 200)")
        
        # Volume analysis
        if 'Volume' in technical_data:
            # This would need volume average from the full dataframe
            # For now, just note if volume is present
            if technical_data['Volume'] > 0:
                evidence.append(f"Volume: {technical_data['Volume']:,.0f}")
        
        return max(0, min(100, score)), evidence
    
    def _calculate_sr_score(self, sr_analysis: Dict, technical_data: pd.Series) -> Tuple[float, List[str]]:
        """Calculate score based on S/R levels"""
        score = 50  # Start neutral
        evidence = []
        
        if not sr_analysis or technical_data is None or (isinstance(technical_data, pd.Series) and technical_data.empty):
            return score, evidence
        
        current_price = technical_data.get('Close', 0)
        if not current_price:
            return score, evidence
        
        # Get nearest levels
        key_support = sr_analysis.get('key_support', [])
        key_resistance = sr_analysis.get('key_resistance', [])
        
        # Distance to nearest support
        if key_support and len(key_support) > 0:
            nearest_support = key_support[0]
            support_distance = (current_price - nearest_support['zone_center']) / current_price
            
            if support_distance < 0.01:  # Very close to support
                score += 20
                evidence.append(f"At strong support ({nearest_support['confluence_score']} confluences)")
            elif support_distance < 0.02:
                score += 10
                evidence.append(f"Near support (distance: {support_distance*100:.1f}%)")
        
        # Distance to nearest resistance
        if key_resistance and len(key_resistance) > 0:
            nearest_resistance = key_resistance[0]
            resistance_distance = (nearest_resistance['zone_center'] - current_price) / current_price
            
            if resistance_distance < 0.01:  # Very close to resistance
                score -= 20
                evidence.append(f"At strong resistance ({nearest_resistance['confluence_score']} confluences)")
            elif resistance_distance < 0.02:
                score -= 10
                evidence.append(f"Near resistance (distance: {resistance_distance*100:.1f}%)")
        
        # Room to move analysis
        if key_support and key_resistance and len(key_support) > 0 and len(key_resistance) > 0:
            support_room = (current_price - key_support[0]['zone_center']) / current_price
            resistance_room = (key_resistance[0]['zone_center'] - current_price) / current_price
            
            if resistance_room > support_room * 2:
                score += 10
                evidence.append("More room to upside")
            elif support_room > resistance_room * 2:
                score -= 10
                evidence.append("More room to downside")
        
        return max(0, min(100, score)), evidence
    
    def _calculate_ml_score(self, ml_predictions: Dict) -> Tuple[float, List[str]]:
        """Calculate score based on ML predictions"""
        score = 50  # Start neutral
        evidence = []
        
        if not ml_predictions:
            return score, evidence
        
        # IB break predictions
        if 'target_broke_ibh' in ml_predictions:
            pred = ml_predictions['target_broke_ibh']
            if pred and 'prediction' in pred:
                if pred['prediction'] == 1 and pred.get('confidence', 0) > 0.7:
                    score += 15
                    evidence.append(f"ML predicts IBH break (conf: {pred['confidence']:.1%})")
                elif pred['prediction'] == 0 and pred.get('confidence', 0) > 0.7:
                    score -= 5
                    evidence.append(f"ML predicts no IBH break (conf: {pred['confidence']:.1%})")
        
        if 'target_broke_ibl' in ml_predictions:
            pred = ml_predictions['target_broke_ibl']
            if pred and 'prediction' in pred:
                if pred['prediction'] == 1 and pred.get('confidence', 0) > 0.7:
                    score -= 15
                    evidence.append(f"ML predicts IBL break (conf: {pred['confidence']:.1%})")
                elif pred['prediction'] == 0 and pred.get('confidence', 0) > 0.7:
                    score += 5
                    evidence.append(f"ML predicts no IBL break (conf: {pred['confidence']:.1%})")
        
        # Next day direction prediction
        if 'target_next_day_direction' in ml_predictions:
            pred = ml_predictions['target_next_day_direction']
            if pred and 'prediction' in pred:
                direction = pred['prediction']
                confidence = pred.get('confidence', 0)
                
                if direction == 'Strong Up' and confidence > 0.6:
                    score += 25
                    evidence.append(f"ML predicts Strong Up (conf: {confidence:.1%})")
                elif direction == 'Mod Up' and confidence > 0.6:
                    score += 15
                    evidence.append(f"ML predicts Moderate Up (conf: {confidence:.1%})")
                elif direction == 'Strong Down' and confidence > 0.6:
                    score -= 25
                    evidence.append(f"ML predicts Strong Down (conf: {confidence:.1%})")
                elif direction == 'Mod Down' and confidence > 0.6:
                    score -= 15
                    evidence.append(f"ML predicts Moderate Down (conf: {confidence:.1%})")
                else:
                    evidence.append(f"ML predicts {direction} (conf: {confidence:.1%})")
        
        return max(0, min(100, score)), evidence
    
    def calculate_risk_management(self, 
                                 current_price: float,
                                 atr: float,
                                 signal: str,
                                 account_balance: float = 100000) -> Dict:
        """
        Calculate risk management parameters
        
        Args:
            current_price: Current market price
            atr: Average True Range
            signal: Trading signal (LONG/SHORT/NEUTRAL)
            account_balance: Account balance for position sizing
        
        Returns:
            Dictionary with stop loss, take profit, and position size
        """
        if signal in ['NEUTRAL'] or atr <= 0:
            return {
                'stop_loss': None,
                'take_profit': None,
                'position_size': 0,
                'risk_amount': 0
            }
        
        # Calculate stop loss and take profit based on ATR
        if 'LONG' in signal:
            stop_loss = current_price - (atr * 2.0)  # 2 ATR stop
            take_profit = current_price + (atr * 3.0)  # 3 ATR target
        else:  # SHORT
            stop_loss = current_price + (atr * 2.0)
            take_profit = current_price - (atr * 3.0)
        
        # Calculate position size (1% risk per trade)
        risk_percent = 0.01
        risk_amount = account_balance * risk_percent
        price_risk = abs(current_price - stop_loss)
        
        if price_risk > 0:
            position_size = risk_amount / price_risk
        else:
            position_size = 0
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'position_size': int(position_size),
            'risk_amount': round(risk_amount, 2),
            'risk_reward_ratio': 1.5  # 3 ATR target / 2 ATR stop
        }