# src/ml_models.py
"""Machine Learning Models Module - Fusion of Market Profile and Technical Analysis"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
from typing import Dict, List, Any

from src.statistics import StatisticalAnalyzer

class MLPredictor:
    def __init__(self, ticker: str, models_dir: Path):
        self.ticker = ticker
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.stats_analyzer = StatisticalAnalyzer(ticker)
        self.model_configs = {
            'target_broke_IBH': {}, 'target_broke_IBL': {},
            'target_broke_pVAH': {}, 'target_broke_pVAL': {},
            'target_closed_in_value': {}
        }
        self.feature_cols = []

    def create_fusion_features(self, profiles: List[Dict], technical_data: pd.DataFrame, all_stats: Dict) -> pd.DataFrame:
        features_list = []
        for i in range(1, len(profiles)):
            current_profile = profiles[i]
            prior_profile = profiles[i-1]
            opening_type = current_profile.get('opening_type', 'Unknown')

            tech_row = technical_data[technical_data.index.date == current_profile['date'].date()]
            if tech_row.empty or opening_type == 'Unknown':
                continue
            tech_row = tech_row.iloc[0]

            features = {
                'opening_type_encoded': ['HOR', 'HIR', 'LIR', 'LOR'].index(opening_type),
                'poc_migration_normalized': (current_profile['poc'] - prior_profile['poc']) / tech_row.get('ATR', 1),
                'rsi': tech_row.get('RSI', 50),
                'adx': tech_row.get('ADX', 25),
            }

            prob_features = self.stats_analyzer.get_probabilities_for_day(opening_type, all_stats)
            features.update(prob_features)

            # Targets
            features['target_broke_IBH'] = 1 if current_profile['session_high'] >= current_profile['ib_high'] else 0
            features['target_broke_IBL'] = 1 if current_profile['session_low'] <= current_profile['ib_low'] else 0
            features['target_broke_pVAH'] = 1 if current_profile['session_high'] >= prior_profile['vah'] else 0
            features['target_broke_pVAL'] = 1 if current_profile['session_low'] <= prior_profile['val'] else 0
            features['target_closed_in_value'] = 1 if prior_profile['val'] <= current_profile['session_close'] <= prior_profile['vah'] else 0

            features_list.append(features)

        df = pd.DataFrame(features_list)
        if not self.feature_cols:
            self.feature_cols = [col for col in df.columns if 'target' not in col]
        return df

    def train_models(self, features_df: pd.DataFrame):
        for target in self.model_configs.keys():
            X = features_df[self.feature_cols]
            y = features_df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train_scaled, y_train)

            self.models[target] = model
            self.scalers[target] = scaler
            self.save_model(target)

            print(f"--- {target} Model ---")
            print(classification_report(y_test, model.predict(X_test_scaled)))

    def predict(self, features_df: pd.DataFrame) -> Dict:
        predictions = {}
        X = features_df[self.feature_cols]
        for target, model in self.models.items():
            X_scaled = self.scalers[target].transform(X)
            pred = model.predict(X_scaled)[0]
            confidence = np.max(model.predict_proba(X_scaled)[0])
            predictions[target] = {'prediction': pred, 'confidence': confidence}
        return predictions

    def save_model(self, target: str):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[target], self.models_dir / f"{self.ticker}_{target}_model.pkl")
        joblib.dump(self.scalers[target], self.models_dir / f"{self.ticker}_{target}_scaler.pkl")

    def load_model(self, target: str) -> bool:
        model_path = self.models_dir / f"{self.ticker}_{target}_model.pkl"
        scaler_path = self.models_dir / f"{self.ticker}_{target}_scaler.pkl"
        if not model_path.exists() or not scaler_path.exists():
            return False
        self.models[target] = joblib.load(model_path)
        self.scalers[target] = joblib.load(scaler_path)
        return True
