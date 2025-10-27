# Machine learning models module
# src/ml_models.py
"""Machine Learning Models Module - Fusion of Market Profile and Technical Analysis"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    def __init__(self, ticker: str, models_dir: Path):
        """
        Initialize ML Predictor with fusion features

        Args:
            ticker: Symbol being analyzed
            models_dir: Directory to save/load models
        """
        self.ticker = ticker
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}

        # Feature columns for the fusion model
        self.market_profile_features = [
            'opening_type_encoded',
            'ib_range_normalized',
            'value_area_width_normalized',
            'prior_close_vs_va',
            'poc_migration_normalized',
            'va_overlap_percent',
            'tpo_skew',
            'day_type_encoded'
        ]

        self.technical_features = [
            'rsi_14',
            'adx_14',
            'is_trending',
            'macd_histogram',
            'bb_percent_b',
            'bb_width_normalized',
            'stoch_k',
            'mfi_14',
            'cci_20',
            'roc_10',
            'lr_slope',
            'lr_r2',
            'ema_cross_score',
            'volume_ratio'
        ]

        self.model_configs = {
            'target_broke_ibh': {
                'type': 'binary',
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5
            },
            'target_broke_ibl': {
                'type': 'binary',
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5
            },
            'target_next_day_direction': {
                'type': 'multiclass',
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 5,
                'classes': ['Strong Down', 'Mod Down', 'Chop', 'Mod Up', 'Strong Up']
            }
        }

    def create_fusion_features(self, market_profiles: List[Dict],
                              technical_data: pd.DataFrame,
                              sr_analysis: Dict) -> pd.DataFrame:
        """
        Create feature matrix by fusing Market Profile and Technical Analysis

        Args:
            market_profiles: List of daily market profile dictionaries
            technical_data: DataFrame with technical indicators
            sr_analysis: Support/Resistance analysis

        Returns:
            DataFrame with fusion features
        """
        features_list = []

        # Ensure technical_data has a DatetimeIndex
        if not isinstance(technical_data.index, pd.DatetimeIndex):
            technical_data.index = pd.to_datetime(technical_data.index, utc=True)

        for i in range(1, len(market_profiles)):
            current_profile = market_profiles[i]
            prior_profile = market_profiles[i-1]

            if not current_profile or not prior_profile:
                continue

            date = current_profile.get('date')
            if date is None:
                continue

            # Find corresponding technical data
            tech_row = technical_data[technical_data.index.date == date.date()]
            if tech_row.empty:
                continue

            tech_row = tech_row.iloc[-1]  # Use last row of the day

            features = {}

            # MARKET PROFILE FEATURES
            # Opening type (will be encoded)
            features['opening_type'] = current_profile.get('opening_type', 'Unknown')

            # IB range normalized by ATR
            if 'ATR' in tech_row:
                features['ib_range_normalized'] = current_profile.get('ib_range', 0) / tech_row['ATR'] if tech_row['ATR'] > 0 else 0
            else:
                features['ib_range_normalized'] = 0

            # Value area width normalized by ATR
            if 'ATR' in tech_row:
                features['value_area_width_normalized'] = current_profile.get('va_width', 0) / tech_row['ATR'] if tech_row['ATR'] > 0 else 0
            else:
                features['value_area_width_normalized'] = 0

            # Prior close location vs value area
            prior_close = prior_profile.get('session_close', 0)
            if prior_close:
                if prior_close < current_profile.get('val', prior_close):
                    features['prior_close_vs_va'] = -1
                elif prior_close > current_profile.get('vah', prior_close):
                    features['prior_close_vs_va'] = 1
                else:
                    features['prior_close_vs_va'] = 0
            else:
                features['prior_close_vs_va'] = 0

            # POC migration
            poc_migration = current_profile.get('poc', 0) - prior_profile.get('poc', 0)
            if 'ATR' in tech_row:
                features['poc_migration_normalized'] = poc_migration / tech_row['ATR'] if tech_row['ATR'] > 0 else 0
            else:
                features['poc_migration_normalized'] = 0

            # Value area overlap
            features['va_overlap_percent'] = self._calculate_va_overlap(current_profile, prior_profile)

            # TPO skew
            features['tpo_skew'] = current_profile.get('tpo_skew', 0)

            # Day type
            features['day_type'] = current_profile.get('day_type', 'Unknown')

            # TECHNICAL ANALYSIS FEATURES
            features['rsi_14'] = tech_row.get('RSI', 50)
            features['adx_14'] = tech_row.get('ADX', 25)
            features['is_trending'] = 1 if tech_row.get('ADX', 25) > 25 else 0
            features['macd_histogram'] = tech_row.get('MACD_Histogram', 0)
            features['bb_percent_b'] = tech_row.get('BB_PercentB', 0.5)

            # Normalized BB width
            if 'BB_Width' in tech_row and 'Close' in tech_row:
                features['bb_width_normalized'] = tech_row['BB_Width'] / tech_row['Close'] if tech_row['Close'] > 0 else 0
            else:
                features['bb_width_normalized'] = 0

            features['stoch_k'] = tech_row.get('Stoch_K', 50)
            features['mfi_14'] = tech_row.get('MFI', 50)
            features['cci_20'] = tech_row.get('CCI', 0)
            features['roc_10'] = tech_row.get('ROC', 0)
            features['lr_slope'] = tech_row.get('LR_Slope', 0)
            features['lr_r2'] = tech_row.get('LR_R2', 0)

            # EMA cross score
            ema_score = 0
            if 'EMA_9' in tech_row and 'EMA_21' in tech_row:
                if tech_row['EMA_9'] > tech_row['EMA_21']:
                    ema_score += 1
            if 'EMA_50' in tech_row and 'EMA_200' in tech_row:
                if tech_row['EMA_50'] > tech_row['EMA_200']:
                    ema_score += 2
            features['ema_cross_score'] = ema_score

            # Volume ratio
            if 'Volume' in tech_row:
                vol_ma = technical_data['Volume'].rolling(20).mean()
                if not vol_ma.empty and vol_ma.iloc[-1] > 0:
                    features['volume_ratio'] = tech_row['Volume'] / vol_ma.iloc[-1]
                else:
                    features['volume_ratio'] = 1
            else:
                features['volume_ratio'] = 1

            # S/R FEATURES
            if sr_analysis and 'key_support' in sr_analysis and 'key_resistance' in sr_analysis:
                current_price = tech_row.get('Close', 0)

                # Distance to nearest support
                if sr_analysis['key_support']:
                    nearest_support = sr_analysis['key_support'][0]['zone_center']
                    features['distance_to_support'] = (current_price - nearest_support) / current_price if current_price > 0 else 0
                else:
                    features['distance_to_support'] = 0

                # Distance to nearest resistance
                if sr_analysis['key_resistance']:
                    nearest_resistance = sr_analysis['key_resistance'][0]['zone_center']
                    features['distance_to_resistance'] = (nearest_resistance - current_price) / current_price if current_price > 0 else 0
                else:
                    features['distance_to_resistance'] = 0
            else:
                features['distance_to_support'] = 0
                features['distance_to_resistance'] = 0

            # TARGET VARIABLES
            # Did price break Initial Balance High?
            if current_profile.get('session_high', 0) > current_profile.get('ib_high', 0):
                features['target_broke_ibh'] = 1
            else:
                features['target_broke_ibh'] = 0

            # Did price break Initial Balance Low?
            if current_profile.get('session_low', float('inf')) < current_profile.get('ib_low', float('inf')):
                features['target_broke_ibl'] = 1
            else:
                features['target_broke_ibl'] = 0

            # Next day direction (need next day's data)
            if i < len(market_profiles) - 1:
                next_profile = market_profiles[i + 1]
                if next_profile:
                    next_return = (next_profile.get('session_close', 0) - current_profile.get('session_close', 0)) / current_profile.get('session_close', 1) if current_profile.get('session_close', 0) != 0 else 0

                    # Classify return into categories
                    if next_return < -0.02:
                        features['target_next_day_direction'] = 'Strong Down'
                    elif next_return < -0.005:
                        features['target_next_day_direction'] = 'Mod Down'
                    elif next_return < 0.005:
                        features['target_next_day_direction'] = 'Chop'
                    elif next_return < 0.02:
                        features['target_next_day_direction'] = 'Mod Up'
                    else:
                        features['target_next_day_direction'] = 'Strong Up'
                else:
                    features['target_next_day_direction'] = 'Chop'

            features['date'] = date
            features_list.append(features)

        return pd.DataFrame(features_list)

    def _calculate_va_overlap(self, current_profile: Dict, prior_profile: Dict) -> float:
        """Calculate value area overlap percentage"""
        if not all(k in current_profile for k in ['vah', 'val']) or \
           not all(k in prior_profile for k in ['vah', 'val']):
            return 0

        overlap_high = min(current_profile['vah'], prior_profile['vah'])
        overlap_low = max(current_profile['val'], prior_profile['val'])

        if overlap_high > overlap_low:
            overlap = overlap_high - overlap_low
            avg_va = (current_profile['va_width'] + prior_profile['va_width']) / 2
            return (overlap / avg_va) * 100 if avg_va > 0 else 0

        return 0

    def prepare_features(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training/prediction

        Args:
            df: Raw feature dataframe
            target: Target variable name

        Returns:
            Tuple of (feature matrix, target vector)
        """
        df = df.copy()

        # Encode categorical variables
        if 'opening_type' in df.columns:
            if 'opening_type' not in self.label_encoders:
                self.label_encoders['opening_type'] = LabelEncoder()
                df['opening_type_encoded'] = self.label_encoders['opening_type'].fit_transform(df['opening_type'].fillna('Unknown'))
            else:
                df['opening_type_encoded'] = self.label_encoders['opening_type'].transform(df['opening_type'].fillna('Unknown'))

        if 'day_type' in df.columns:
            if 'day_type' not in self.label_encoders:
                self.label_encoders['day_type'] = LabelEncoder()
                df['day_type_encoded'] = self.label_encoders['day_type'].fit_transform(df['day_type'].fillna('Unknown'))
            else:
                df['day_type_encoded'] = self.label_encoders['day_type'].transform(df['day_type'].fillna('Unknown'))

        # Select features
        feature_cols = self.market_profile_features + self.technical_features
        feature_cols = [col for col in feature_cols if col in df.columns]

        X = df[feature_cols].fillna(0)

        # Get target
        y = None
        if target in df.columns:
            if self.model_configs[target]['type'] == 'multiclass':
                if target not in self.label_encoders:
                    self.label_encoders[target] = LabelEncoder()
                    y = self.label_encoders[target].fit_transform(df[target])
                else:
                    y = self.label_encoders[target].transform(df[target])
            else:
                y = df[target]

        return X, y

    def train_models(self, features_df: pd.DataFrame) -> Dict:
        """
        Train all prediction models

        Args:
            features_df: DataFrame with all features and targets

        Returns:
            Dictionary with training results
        """
        results = {}

        for target, config in self.model_configs.items():
            if target not in features_df.columns:
                continue

            # Remove rows with missing target
            train_df = features_df.dropna(subset=[target])

            if len(train_df) < 20:  # Need minimum samples
                results[target] = {'status': 'insufficient_data'}
                continue

            # Prepare features
            X, y = self.prepare_features(train_df, target)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            if target not in self.scalers:
                self.scalers[target] = StandardScaler()
                X_train_scaled = self.scalers[target].fit_transform(X_train)
            else:
                X_train_scaled = self.scalers[target].transform(X_train)

            X_test_scaled = self.scalers[target].transform(X_test)

            # Train model
            model = RandomForestClassifier(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train_scaled, y_train)
            self.models[target] = model

            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            results[target] = {
                'status': 'success',
                'train_score': train_score,
                'test_score': test_score,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'feature_importance': feature_importance,
                'n_samples': len(train_df)
            }

            # Save model
            self.save_model(target)

        return results

    def predict(self, features_df: pd.DataFrame) -> Dict:
        """
        Make predictions using trained models

        Args:
            features_df: DataFrame with features

        Returns:
            Dictionary with predictions and probabilities
        """
        predictions = {}

        for target, model in self.models.items():
            if model is None:
                continue

            # Prepare features
            X, _ = self.prepare_features(features_df, target)

            if target in self.scalers:
                X_scaled = self.scalers[target].transform(X)
            else:
                X_scaled = X

            # Make predictions
            pred = model.predict(X_scaled)
            pred_proba = model.predict_proba(X_scaled)

            # Decode if necessary
            if target in self.label_encoders and self.model_configs[target]['type'] == 'multiclass':
                pred_decoded = self.label_encoders[target].inverse_transform(pred)
            else:
                pred_decoded = pred

            predictions[target] = {
                'prediction': pred_decoded[-1] if len(pred_decoded) > 0 else None,
                'probabilities': pred_proba[-1] if len(pred_proba) > 0 else None,
                'confidence': max(pred_proba[-1]) if len(pred_proba) > 0 else 0
            }

        return predictions

    def model_exists(self, target: str) -> bool:
        """Check if a model file exists for the given target."""
        model_path = self.models_dir / f"{self.ticker}_{target}_model.pkl"
        return model_path.exists()

    def save_model(self, target: str):
        """Save model and associated objects"""
        model_path = self.models_dir / f"{self.ticker}_{target}_model.pkl"
        scaler_path = self.models_dir / f"{self.ticker}_{target}_scaler.pkl"
        encoder_path = self.models_dir / f"{self.ticker}_{target}_encoders.pkl"

        if target in self.models:
            joblib.dump(self.models[target], model_path)

        if target in self.scalers:
            joblib.dump(self.scalers[target], scaler_path)

        if self.label_encoders:
            joblib.dump(self.label_encoders, encoder_path)

    def load_model(self, target: str) -> bool:
        """Load model and associated objects"""
        model_path = self.models_dir / f"{self.ticker}_{target}_model.pkl"
        scaler_path = self.models_dir / f"{self.ticker}_{target}_scaler.pkl"
        encoder_path = self.models_dir / f"{self.ticker}_{target}_encoders.pkl"

        try:
            if model_path.exists():
                self.models[target] = joblib.load(model_path)

            if scaler_path.exists():
                self.scalers[target] = joblib.load(scaler_path)

            if encoder_path.exists():
                self.label_encoders = joblib.load(encoder_path)

            return True
        except:
            return False
