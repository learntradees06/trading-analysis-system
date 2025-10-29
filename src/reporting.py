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
                        sr_analysis: Dict, ml_predictions: Dict, statistics: Dict, all_data: Dict) -> str:

        # We need the SRLevelAnalyzer to recalculate fib_levels for the report
        from src.sr_levels import SRLevelAnalyzer
        from src.config import INSTRUMENT_SETTINGS
        settings = INSTRUMENT_SETTINGS.get(self.ticker, {"tick_size": 0.01})
        sr_analyzer = SRLevelAnalyzer(self.ticker, settings['tick_size'])
        fib_levels = sr_analyzer.calculate_fibonacci_levels(all_data['1d'])

        template_vars = {
            'ticker': self.ticker,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': f"${daily_with_indicators['Close'].iloc[-1]:.2f}",
            'opening_type': current_profile.get('opening_type', 'Unknown'),
            'signal_section': self._generate_signal_section(signal),
            'key_levels': self._generate_key_levels_section(current_profile, sr_analysis),
            'statistics_tables': self._generate_statistics_tables(statistics, current_profile.get('opening_type')),
            'ml_predictions_section': self._generate_ml_predictions_section(ml_predictions, statistics),
            'chart_script': self._generate_price_chart(all_data['1d'], daily_with_indicators, current_profile, sr_analysis),
            'atr_table': self._generate_atr_table(daily_with_indicators),
            'fibonacci_table': self._generate_fibonacci_table(fib_levels),
            'sr_cluster_table': self._generate_sr_cluster_table(sr_analysis),
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

    def _generate_ml_predictions_section(self, predictions: Dict, statistics: Dict) -> str:
        if not predictions or 'error' in predictions:
            return f"<p>ML Predictions not available: {predictions.get('error', 'N/A')}</p>"

        pred = predictions.get('predicted_opening_type', 'N/A')
        conf = predictions.get('confidence', 0)

        html = f"<p><strong>Predicted Next Open Type:</strong> {pred} (Confidence: {conf:.1%})</p>"

        # Now, add the statistics tables for the predicted opening type
        if pred != 'N/A' and statistics:
            html += f"<p><strong>Historical Probabilities for Predicted '{pred}' Opening:</strong></p>"
            html += self._format_stat_table("Close Above Probability", statistics, 'close_above', pred)
            html += self._format_stat_table("Broken During RTH Probability", statistics, 'broken_during_rth', pred)

        return html

    def _generate_atr_table(self, daily_data: pd.DataFrame) -> str:
        """Generates an HTML table for ATR and ATR Projections."""
        if 'ATR' not in daily_data.columns: return ""
        current_atr = daily_data['ATR'].iloc[-1]
        current_close = daily_data['Close'].iloc[-1]

        table = "<h4>ATR Analysis</h4><table>"
        table += f"<tr><td>Current ATR (14D)</td><td>{current_atr:.2f}</td></tr>"
        table += f"<tr><td>ATR % of Price</td><td>{ (current_atr / current_close) * 100:.2f}%</td></tr>"
        table += "<tr><td colspan='2' style='text-align:center;'><strong>ATR Projections</strong></td></tr>"
        table += f"<tr><td>+1 ATR</td><td>{current_close + current_atr:.2f}</td></tr>"
        table += f"<tr><td>-1 ATR</td><td>{current_close - current_atr:.2f}</td></tr>"
        table += f"<tr><td>+2 ATR</td><td>{current_close + (current_atr * 2):.2f}</td></tr>"
        table += f"<tr><td>-2 ATR</td><td>{current_close - (current_atr * 2):.2f}</td></tr>"
        table += "</table>"
        return table

    def _generate_fibonacci_table(self, fib_levels: Dict) -> str:
        """Generates an HTML table for Fibonacci levels."""
        if not fib_levels: return ""
        table = "<h4>Fibonacci Levels</h4><table><tr><th>Level</th><th>Price</th></tr>"
        for name, level in fib_levels.items():
            # Clean up the name for display
            display_name = name.replace('fib_ext_', 'Extension ').replace('fib_', 'Retracement ').replace('_', '.')
            table += f"<tr><td>{display_name.title()}</td><td>{level:.2f}</td></tr>"
        table += "</table>"
        return table

    def _generate_sr_cluster_table(self, sr_analysis: Dict) -> str:
        """Generates an HTML table for S/R Cluster Zones."""
        if not sr_analysis or 'all_zones' not in sr_analysis: return ""

        zones = sr_analysis['all_zones']
        # Sort by strength (confluence score)
        sorted_zones = sorted(zones, key=lambda z: z.get('confluence_score', 0), reverse=True)

        table = "<h4>S/R Cluster Zones (by Strength)</h4><table>"
        table += "<tr><th>Price Range</th><th>Center</th><th>Type</th><th>Strength</th><th>Confluent Reasons</th></tr>"

        for zone in sorted_zones[:10]: # Display top 10 strongest zones
            price_range = f"{zone.get('zone_low', 0):.2f} - {zone.get('zone_high', 0):.2f}"
            center = f"{zone.get('zone_center', 0):.2f}"
            zone_type = zone.get('type', 'N/A')
            strength = zone.get('confluence_score', 0)
            reasons = ", ".join(zone.get('confluent_reasons', []))

            table += f"<tr><td>{price_range}</td><td>{center}</td><td>{zone_type}</td><td>{strength}</td><td>{reasons}</td></tr>"

        table += "</table>"
        return table

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
                body { font-family: sans-serif; margin: 0; padding: 0; background-color: #f4f4f9; }
                .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; }
                .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1, h2, h3, h4 { color: #333; }
                h1 { text-align: center; grid-column: 1 / -1; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.9em; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>{{ ticker }} Trading Plan - {{ timestamp }}</h1>
            <div class="container">
                <div class="card">
                    <h2>Signal & Key Info</h2>
                    <p><strong>Current Price:</strong> {{ current_price }}</p>
                    <p><strong>Opening Type:</strong> {{ opening_type }}</p>
                    <div>{{ signal_section }}</div>
                </div>
                <div class="card">
                    <h2>Key Levels</h2>
                    <div>{{ key_levels }}</div>
                </div>
                <div class="card">
                    <h2>ML Prediction (Next Open)</h2>
                    <div>{{ ml_predictions_section }}</div>
                </div>
                <div class="card">
                    <h2>Historical Statistics</h2>
                    <div>{{ statistics_tables }}</div>
                </div>
                <div class="card">
                    <h2>ATR & Fibonacci</h2>
                    {{ atr_table }}
                    {{ fibonacci_table }}
                </div>
                <div class="card">
                     <h2>S/R Cluster Zones</h2>
                    {{ sr_cluster_table }}
                </div>
                 <div class="card" style="grid-column: 1 / -1;">
                    <h2>Price Chart</h2>
                    <div id='chart'>{{ chart_script }}</div>
                </div>
            </div>
        </body>
        </html>
        """
