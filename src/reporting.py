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

    def generate_report(self, signal: Dict, current_profile: Dict, developing_profile: Dict, daily_with_indicators: pd.DataFrame,
                        sr_analysis: Dict, ml_predictions: Dict, statistics: Dict, all_data: Dict) -> str:

        from src.indicators import calculate_fibonacci_targets
        fib_targets = calculate_fibonacci_targets(all_data['1d'])

        template_vars = {
            'ticker': self.ticker,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': f"${daily_with_indicators['Close'].iloc[-1]:.2f}",
            'opening_type': current_profile.get('opening_type', 'Unknown'),
            'signal_section': self._generate_signal_section(signal),
            'key_levels_section': self._generate_key_levels_section(current_profile, sr_analysis),
            'developing_profile_section': self._generate_developing_profile_section(developing_profile),
            'statistics_tables': self._generate_statistics_tables(statistics, current_profile.get('opening_type')),
            'ml_predictions_section': self._generate_ml_predictions_section(ml_predictions, statistics),
            'chart_script': self._generate_price_chart(all_data['1d'], daily_with_indicators, current_profile, sr_analysis, developing_profile),
            'targets_section': self._generate_targets_section(daily_with_indicators, fib_targets),
            'sr_cluster_table': self._generate_sr_cluster_table(sr_analysis),
        }

        html_content = Template(self.html_template).render(template_vars)
        report_path = self.reports_dir / f"{self.ticker}_trading_plan_{datetime.now().strftime('%Y%m%d%H%M')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return str(report_path)

    def _generate_signal_section(self, signal: Dict) -> str:
        color_map = {'LONG': '#28a745', 'SHORT': '#dc3545', 'NEUTRAL': '#ffc107'}
        signal_text = signal.get('signal', 'NEUTRAL')
        evidence_html = "".join([f"<li>{item}</li>" for item in signal.get('evidence', [])])
        return f"""
            <h2 style="color:{color_map.get(signal_text, '#6c757d')}">Signal: {signal_text}</h2>
            <p><strong>Confidence:</strong> {signal.get('confidence')} | <strong>Confluence Score:</strong> {signal.get('score', 0)}</p>
            <h4>Evidence:</h4>
            <ul>{evidence_html}</ul>
        """

    def _generate_key_levels_section(self, profile: Dict, sr: Dict) -> str:
        levels = {
            "Prev. VAH": profile.get('vah'), "Prev. POC": profile.get('poc'), "Prev. VAL": profile.get('val'),
            "Prev. IBH": profile.get('ib_high'), "Prev. IBL": profile.get('ib_low'),
            "Nearest Support Zone": sr.get('key_support', [{}])[0].get('zone_center'),
            "Nearest Resistance Zone": sr.get('key_resistance', [{}])[0].get('zone_center')
        }
        levels_html = "".join([f"<li><strong>{name}:</strong> {value:.2f}</li>" for name, value in levels.items() if value is not None])
        return f"<ul>{levels_html}</ul>"

    def _generate_developing_profile_section(self, profile: Dict) -> str:
        if not profile or not profile.get('poc'):
            return "<h4>Today's Developing Profile</h4><p>Not yet available or market is closed.</p>"

        levels = {
            "Dev. VAH": profile.get('vah'), "Dev. POC": profile.get('poc'), "Dev. VAL": profile.get('val'),
            "Dev. IBH": profile.get('ib_high'), "Dev. IBL": profile.get('ib_low'),
        }
        levels_html = "".join([f"<li><strong>{name}:</strong> {value:.2f}</li>" for name, value in levels.items() if value is not None])
        return f"<ul>{levels_html}</ul>"

    def _generate_statistics_tables(self, statistics: Dict, opening_type: str) -> str:
        if not statistics: return ""
        html = self._format_stat_table("Close Above Probability", statistics, 'close_above', opening_type)
        html += self._format_stat_table("Broken During RTH Probability", statistics, 'broken_during_rth', opening_type)
        return html

    def _format_stat_table(self, title: str, stats: Dict, stat_type: str, otype: str) -> str:
        table = f"<h4>{title}</h4><table><tr><th>Level</th><th>Probability</th></tr>"
        if otype in stats and stat_type in stats[otype].get('stats', {}):
            for level, prob in stats[otype]['stats'][stat_type].items():
                table += f"<tr><td>{level}</td><td>{prob:.1f}%</td></tr>"
        else:
            table += "<tr><td colspan='2'>N/A</td></tr>"
        table += "</table>"
        return table

    def _generate_ml_predictions_section(self, predictions: Dict, statistics: Dict) -> str:
        if not predictions or 'error' in predictions:
            return f"<p>ML Predictions not available: {predictions.get('error', 'N/A')}</p>"

        pred = predictions.get('predicted_opening_type', 'N/A')
        conf = predictions.get('confidence', 0)

        html = f"<p><strong>Predicted Next Open Type:</strong> {pred} (Confidence: {conf:.1%})</p>"

        if pred != 'N/A' and statistics:
            html += self._format_stat_table(f"Probabilities for Predicted '{pred}' Open", statistics, 'broken_during_rth', pred)
        return html

    def _generate_targets_section(self, daily_data: pd.DataFrame, fib_targets: Dict) -> str:
        """Generates HTML for the Expected Targets section."""
        html = "<h4>ATR Projections</h4>"
        if 'ATR' in daily_data.columns:
            current_atr = daily_data['ATR'].iloc[-1]
            current_close = daily_data['Close'].iloc[-1]
            html += "<table>"
            html += f"<tr><td>Bullish Target (+1 ATR)</td><td>{current_close + current_atr:.2f}</td></tr>"
            html += f"<tr><td>Bearish Target (-1 ATR)</td><td>{current_close - current_atr:.2f}</td></tr>"
            html += "</table>"

        html += "<h4>Fibonacci Price Targets</h4>"
        if fib_targets:
            html += "<table><tr><th>Level</th><th>Price</th></tr>"
            for name, level in fib_targets.items():
                html += f"<tr><td>{name}</td><td>{level:.2f}</td></tr>"
            html += "</table>"
        else:
            html += "<p>Not available.</p>"

        return html

    def _generate_sr_cluster_table(self, sr_analysis: Dict) -> str:
        if not sr_analysis or 'all_zones' not in sr_analysis: return ""
        zones = sr_analysis['all_zones']
        sorted_zones = sorted(zones, key=lambda z: z.get('confluence_score', 0), reverse=True)
        table = "<table><tr><th>Center</th><th>Type</th><th>Strength</th><th>Reasons</th></tr>"
        for zone in sorted_zones[:10]:
            center = f"{zone.get('zone_center', 0):.2f}"
            zone_type = zone.get('type', 'N/A')
            strength = zone.get('confluence_score', 0)
            reasons = ", ".join(zone.get('confluent_reasons', []))
            table += f"<tr><td>{center}</td><td>{zone_type}</td><td>{strength}</td><td>{reasons}</td></tr>"
        table += "</table>"
        return table

    def _generate_price_chart(self, price: pd.DataFrame, tech: pd.DataFrame, prev_profile: Dict, sr: Dict, dev_profile: Dict) -> str:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])

        fig.add_trace(go.Candlestick(x=price.index, open=price['Open'], high=price['High'], low=price['Low'], close=price['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Bar(x=tech.index, y=tech['Volume'], name='Volume'), row=2, col=1)

        levels_to_plot = {
            "pVAH": (prev_profile.get('vah'), 'blue'), "pPOC": (prev_profile.get('poc'), 'blue'), "pVAL": (prev_profile.get('val'), 'blue'),
            "dVAH": (dev_profile.get('vah'), 'red'), "dPOC": (dev_profile.get('poc'), 'red'), "dVAL": (dev_profile.get('val'), 'red')
        }
        for name, (val, color) in levels_to_plot.items():
            if val: fig.add_hline(y=val, line_dash="dash", line_color=color, annotation_text=name, annotation_position="bottom right")

        fig.update_layout(title_text=f"{self.ticker} Price Chart & Key Levels", xaxis_rangeslider_visible=False, showlegend=False)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    def _get_html_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ ticker }} Trading Plan</title>
            <style>
                :root {
                    --bg-color: #f1f5f9; --text-color: #0f172a; --card-bg: #ffffff;
                    --border-color: #cbd5e1; --header-bg: #e2e8f0; --accent-color: #4f46e5;
                }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    margin: 0; padding: 2rem; background-color: var(--bg-color); color: var(--text-color);
                }
                .container {
                    max-width: 1400px; margin: 0 auto;
                }
                h1, h2, h3, h4 { color: var(--text-color); }
                h1 { text-align: center; border-bottom: 2px solid var(--border-color); padding-bottom: 1rem; margin-bottom: 2rem; }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 1.5rem;
                }
                .card {
                    background-color: var(--card-bg);
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
                    border: 1px solid var(--border-color);
                }
                .full-width { grid-column: 1 / -1; }
                table {
                    border-collapse: collapse; width: 100%; font-size: 0.9em; margin-top: 1rem;
                }
                th, td {
                    border: 1px solid var(--border-color); padding: 0.75rem; text-align: left;
                }
                th { background-color: var(--header-bg); font-weight: 600; }
                tr:nth-child(even) { background-color: #f8fafc; }
                .accordion-button {
                    background: var(--accent-color); color: white; cursor: pointer; padding: 1rem;
                    width: 100%; border: none; text-align: left; outline: none; font-size: 1.25rem;
                    font-weight: 700; border-radius: 0.5rem; margin-top: 1.5rem;
                    transition: background-color 0.3s ease;
                }
                .accordion-button:hover { background-color: #4338ca; }
                .panel {
                    padding: 0 1.5rem; background-color: var(--card-bg);
                    max-height: 0; overflow: hidden;
                    transition: max-height 0.3s ease-in-out;
                    border-radius: 0 0 0.5rem 0.5rem;
                }
            </style>
        </head>
        <body>
        <div class="container"><h1>{{ ticker }} Trading Plan - {{ timestamp }}</h1>
        <div class="grid">
        <div class="card"><h2>Signal & Context</h2><p><strong>Current Price:</strong> {{ current_price }}</p><p><strong>Prev. Day Open Type:</strong> {{ opening_type }}</p>{{ signal_section }}</div>
        <div class="card"><h2>Key Levels</h2>{{ key_levels_section }}{{ developing_profile_section }}</div>
        <div class="card"><h2>ML Prediction</h2>{{ ml_predictions_section }}</div>
        <div class="card full-width" id="chart">{{ chart_script }}</div>
        <div class="card"><h3>Expected Targets</h3>{{ targets_section }}</div>
        <div class="card"><h3>Multi-Timeframe S/R Zones</h3>{{ sr_cluster_table }}</div>
        </div>
        <button class="accordion-button">Detailed Analysis</button>
        <div class="panel">
            <div class="grid" style="padding-top: 1.5rem; padding-bottom: 1.5rem;">
                <div class="card"><h3>Historical Statistics</h3>{{ statistics_tables }}</div>
            </div>
        </div>
        </div>
        <script>
            document.querySelectorAll('.accordion-button').forEach(button => {
                button.addEventListener('click', () => {
                    const panel = button.nextElementSibling;
                    button.classList.toggle('active');
                    if (panel.style.maxHeight) {
                        panel.style.maxHeight = null;
                    } else {
                        panel.style.maxHeight = panel.scrollHeight + "px";
                    }
                });
            });
        </script>
        </body>
        </html>
        """
