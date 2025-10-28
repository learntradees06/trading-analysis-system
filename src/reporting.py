# src/reporting.py
"""Interactive HTML Report Generation Module with Plotly and Jinja2"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from jinja2 import Template
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class ReportGenerator:
    def __init__(self, ticker: str, reports_dir: Path):
        self.ticker = ticker
        self.reports_dir = reports_dir
        self.html_template = self._get_html_template()

    def generate_report(self, signal: Dict, current_profile: Dict, daily_with_indicators: pd.DataFrame,
                        sr_analysis: Dict, ml_predictions: Dict, statistics: Dict, price_data: pd.DataFrame) -> str:

        template_vars = {
            'ticker': self.ticker,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': f"${daily_with_indicators['Close'].iloc[-1]:.2f}",
            'opening_type': current_profile.get('opening_type', 'Unknown'),
            'signal_section': self._generate_signal_section(signal),
            'key_levels': self._generate_key_levels_section(current_profile, sr_analysis),
            'statistics_tables': self._generate_statistics_tables(statistics, current_profile.get('opening_type')),
            'ml_predictions': self._generate_ml_predictions_section(ml_predictions),
            'chart_script': self._generate_price_chart(price_data, daily_with_indicators, current_profile, sr_analysis),
        }

        html_content = Template(self.html_template).render(template_vars)
        report_path = self.reports_dir / f"{self.ticker}_trading_plan_{datetime.now().strftime('%Y%m%d%H%M')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return str(report_path)

    def _generate_signal_section(self, signal: Dict) -> str:
        color_map = {'LONG': 'green', 'SHORT': 'red', 'NEUTRAL': 'orange'}
        evidence_html = "".join([f"<li>{item}</li>" for item in signal.get('evidence', [])])
        return f"""
            <h2 style="color:{color_map.get(signal.get('signal'), 'black')}">Signal: {signal.get('signal')}</h2>
            <p><strong>Confidence:</strong> {signal.get('confidence')}</p>
            <ul>{evidence_html}</ul>
        """

    def _generate_key_levels_section(self, profile: Dict, sr: Dict) -> str:
        levels = {
            "pVAH": profile.get('pVAH'), "pPOC": profile.get('pPOC'), "pVAL": profile.get('pVAL'),
            "IBH": profile.get('ib_high'), "IBL": profile.get('ib_low'),
            "Nearest Support": sr.get('key_support', [{}])[0].get('zone_center'),
            "Nearest Resistance": sr.get('key_resistance', [{}])[0].get('zone_center')
        }
        levels_html = "".join([f"<li><strong>{name}:</strong> {value:.2f}</li>" for name, value in levels.items() if value])
        return f"<ul>{levels_html}</ul>"

    def _generate_statistics_tables(self, statistics: Dict, opening_type: str) -> str:
        if not statistics: return "<p>No statistical data available.</p>"
        html = "<h3>Historical Probabilities for Opening Type: " f"{opening_type}" "</h3>"
        html += self._format_stat_table("Close Above Probability", statistics, 'close_above', opening_type)
        html += self._format_stat_table("Broken During RTH Probability", statistics, 'broken_during_rth', opening_type)
        return html

    def _format_stat_table(self, title: str, stats: Dict, stat_type: str, otype: str) -> str:
        table = f"<h4>{title}</h4><table><tr><th>Level</th><th>Probability</th></tr>"
        if otype in stats and stat_type in stats[otype]['stats']:
            for level, prob in stats[otype]['stats'][stat_type].items():
                table += f"<tr><td>{level}</td><td>{prob:.1f}%</td></tr>"
        table += "</table>"
        return table

    def _generate_ml_predictions_section(self, predictions: Dict) -> str:
        if not predictions: return "<p>No ML predictions available.</p>"
        pred_html = "".join([f"<li><strong>{target.replace('target_', '')}:</strong> {'Yes' if p['prediction'] == 1 else 'No'} ({p['confidence']:.1%})</li>" for target, p in predictions.items()])
        return f"<ul>{pred_html}</ul>"

    def _generate_price_chart(self, price: pd.DataFrame, tech: pd.DataFrame, profile: Dict, sr: Dict) -> str:
        fig = go.Figure(data=[go.Candlestick(x=price.index, open=price['Open'], high=price['High'], low=price['Low'], close=price['Close'])])
        levels_to_plot = {
            "pVAH": profile.get('pVAH'), "pPOC": profile.get('pPOC'), "pVAL": profile.get('pVAL'),
            "IBH": profile.get('ib_high'), "IBL": profile.get('ib_low'),
        }
        for name, val in levels_to_plot.items():
            if val: fig.add_hline(y=val, line_dash="dash", annotation_text=name)

        fig.update_layout(title_text=f"{self.ticker} Price Chart", xaxis_rangeslider_visible=False)
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    def _get_html_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ ticker }} Trading Plan</title>
            <style>
                body { font-family: sans-serif; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 50%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>{{ ticker }} Trading Plan - {{ timestamp }}</h1>
            <p><strong>Current Price:</strong> {{ current_price }}</p>
            <p><strong>Opening Type:</strong> {{ opening_type }}</p>

            <h2>Signal</h2>
            <div>{{ signal_section }}</div>

            <h2>Key Levels</h2>
            <div>{{ key_levels }}</div>

            <h2>Statistics</h2>
            <div>{{ statistics_tables }}</div>

            <h2>ML Predictions</h2>
            <div>{{ ml_predictions }}</div>

            <h2>Price Chart</h2>
            <div id='chart'>{{ chart_script }}</div>
        </body>
        </html>
        """
