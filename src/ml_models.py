# src/ml_models.py
"""Machine Learning Models Module - Prediction of Next Day Opening Type"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

from src.statistics import StatisticalAnalyzer

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self, ticker: str, models_dir: Path):
        self.ticker = ticker
        self.models_dir = models_dir
        self.model: RandomForestClassifier = None
        self.scaler: StandardScaler = None
        self.encoder: LabelEncoder = LabelEncoder()
        self.is_trained = False
        self.feature_cols = []
        # Define the classes explicitly for the encoder
        self.opening_classes = ['HOR', 'HIR', 'LIR', 'LOR']
        self.encoder.fit(self.opening_classes)
        self.stats_analyzer = StatisticalAnalyzer(ticker)


    def create_features(self, profiles: List[Dict], technical_data: pd.DataFrame, all_stats: Dict) -> pd.DataFrame:
        """
        Creates features from market profiles and technical data to predict the next day's opening type.
        """
        features_list = []
        for i in range(1, len(profiles) - 1): # Iterate up to the second to last day
            current_profile = profiles[i]
            prior_profile = profiles[i-1]
            next_day_profile = profiles[i+1] # Target profile

            tech_row = technical_data[technical_data.index.date == current_profile['date'].date()]
            if tech_row.empty:
                continue

            tech_row = tech_row.iloc[0]

            # Feature Engineering
            features = {
                'poc_migration_norm': (current_profile['poc'] - prior_profile['poc']) / tech_row.get('ATR', 1),
                'va_width_norm': current_profile['va_width'] / tech_row.get('ATR', 1),
                'ib_range_norm': current_profile['ib_range'] / tech_row.get('ATR', 1),
                'close_in_value_area': 1 if current_profile['val'] <= current_profile['session_close'] <= current_profile['vah'] else 0,
                'close_vs_poc': (current_profile['session_close'] - current_profile['poc']) / tech_row.get('ATR', 1),
                'high_vs_vah': (current_profile['session_high'] - current_profile['vah']) / tech_row.get('ATR', 1),
                'low_vs_val': (current_profile['session_low'] - current_profile['val']) / tech_row.get('ATR', 1),
                'rsi': tech_row.get('RSI', 50),
                'adx': tech_row.get('ADX', 25),
                'volume_change_pct': tech_row.get('Volume_pct_change', 0),
            }

            # Add historical probability features
            opening_type = current_profile.get('opening_type', 'Unknown')
            if opening_type != 'Unknown':
                prob_features = self.stats_analyzer.get_probabilities_for_day(opening_type, all_stats)
                features.update(prob_features)

            # Target Variable
            target_opening_type = next_day_profile.get('opening_type')
            if target_opening_type in self.opening_classes:
                features['target_opening_type'] = target_opening_type
                features_list.append(features)

        df = pd.DataFrame(features_list)
        if df.empty:
            return pd.DataFrame()

        # Define feature columns dynamically, excluding the target
        if not self.feature_cols:
            self.feature_cols = [col for col in df.columns if 'target' not in col]

        # Fill NaNs that might have been introduced by missing prob features
        df.fillna(0.5, inplace=True) # Fill with a neutral probability

        # Encode the target variable
        df['target_opening_type_encoded'] = self.encoder.transform(df['target_opening_type'])

        return df

    def create_prediction_features(self, profiles: List[Dict], technical_data: pd.DataFrame, all_stats: Dict) -> pd.DataFrame:
        """
        Creates features for the most recent day to predict the upcoming day's opening type.
        """
        if len(profiles) < 2:
            return pd.DataFrame()

        current_profile = profiles[-1]
        prior_profile = profiles[-2]

        tech_row = technical_data[technical_data.index.date == current_profile['date'].date()]
        if tech_row.empty:
            return pd.DataFrame()
        tech_row = tech_row.iloc[0]

        features = {
            'poc_migration_norm': (current_profile['poc'] - prior_profile['poc']) / tech_row.get('ATR', 1),
            'va_width_norm': current_profile['va_width'] / tech_row.get('ATR', 1),
            'ib_range_norm': current_profile['ib_range'] / tech_row.get('ATR', 1),
            'close_in_value_area': 1 if current_profile['val'] <= current_profile['session_close'] <= current_profile['vah'] else 0,
            'close_vs_poc': (current_profile['session_close'] - current_profile['poc']) / tech_row.get('ATR', 1),
            'high_vs_vah': (current_profile['session_high'] - current_profile['vah']) / tech_row.get('ATR', 1),
            'low_vs_val': (current_profile['session_low'] - current_profile['val']) / tech_row.get('ATR', 1),
            'rsi': tech_row.get('RSI', 50),
            'adx': tech_row.get('ADX', 25),
            'volume_change_pct': tech_row.get('Volume_pct_change', 0),
        }

        # Add historical probability features
        opening_type = current_profile.get('opening_type', 'Unknown')
        if opening_type != 'Unknown':
            prob_features = self.stats_analyzer.get_probabilities_for_day(opening_type, all_stats)
            features.update(prob_features)

        df = pd.DataFrame([features])

        if not self.feature_cols and self.is_trained:
             # Load feature columns from model if available
             if hasattr(self.model, 'feature_names_in_'):
                 self.feature_cols = self.model.feature_names_in_
        elif not self.feature_cols:
             self.feature_cols = list(features.keys())

        # Reorder columns to match the training set, adding any that are missing
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0.5 # Use a neutral probability for missing features
            df = df[self.feature_cols]

        return df

    def train_model(self, features_df: pd.DataFrame):
        """
        Trains a multiclass classification model to predict the opening type.
        """
        if features_df.empty or 'target_opening_type_encoded' not in features_df.columns:
            logger.warning(f"Feature DataFrame is empty or target column is missing for {self.ticker}. Skipping training.")
            return

        X = features_df[self.feature_cols]
        y = features_df['target_opening_type_encoded']

        if len(y.unique()) < 2:
            logger.warning(f"Not enough class diversity for {self.ticker} to train a model. Skipping.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            class_weight='balanced_subsample',
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)

        logger.info(f"--- Opening Type Prediction Model Report for {self.ticker} ---")
        y_pred = self.model.predict(X_test_scaled)

        class_names = self.encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        logger.info(f"\n{report}")

        self.is_trained = True
        self.save_model()


    def predict(self, latest_features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predicts the next day's opening type using the trained model.
        """
        if not self.is_trained or self.model is None:
            return {"error": "Model not trained or loaded."}

        if latest_features_df.empty:
            return {"error": "Input features for prediction are empty."}

        X = latest_features_df[self.feature_cols]
        X_scaled = self.scaler.transform(X)

        pred_encoded = self.model.predict(X_scaled)[0]
        confidence_scores = self.model.predict_proba(X_scaled)[0]

        predicted_class = self.encoder.inverse_transform([pred_encoded])[0]
        confidence = np.max(confidence_scores)

        return {
            'predicted_opening_type': predicted_class,
            'confidence': confidence,
            'probabilities': {self.encoder.classes_[i]: score for i, score in enumerate(confidence_scores)}
        }

    def save_model(self):
        """Saves the trained model, scaler, and encoder."""
        if not self.is_trained: return

        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.models_dir / f"{self.ticker}_opening_type_model.pkl")
        joblib.dump(self.scaler, self.models_dir / f"{self.ticker}_opening_type_scaler.pkl")
        joblib.dump(self.encoder, self.models_dir / f"{self.ticker}_opening_type_encoder.pkl")
        logger.info(f"Successfully saved model for {self.ticker}")

    def load_model(self) -> bool:
        """Loads a pre-trained model, scaler, and encoder."""
        model_path = self.models_dir / f"{self.ticker}_opening_type_model.pkl"
        scaler_path = self.models_dir / f"{self.ticker}_opening_type_scaler.pkl"
        encoder_path = self.models_dir / f"{self.ticker}_opening_type_encoder.pkl"

        if not model_path.exists() or not scaler_path.exists() or not encoder_path.exists():
            logger.warning(f"Model files not found for {self.ticker}")
            return False

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.encoder = joblib.load(encoder_path)
            # Set feature columns from the loaded model
            if hasattr(self.model, 'feature_names_in_'):
                 self.feature_cols = self.model.feature_names_in_
            self.is_trained = True
            logger.info(f"Successfully loaded pre-trained model for {self.ticker}")
            return True
        except Exception as e:
            logger.error(f"Error loading model for {self.ticker}: {e}")
            return False
