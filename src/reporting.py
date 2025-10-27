# Reporting module
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
import json
from typing import Dict, List, Optional

class ReportGenerator:
    def __init__(self, ticker: str, reports_dir: Path):
        """
        Initialize Report Generator
        
        Args:
            ticker: Symbol being analyzed
            reports_dir: Directory to save reports
        """
        self.ticker = ticker
        self.reports_dir = reports_dir
        
        # HTML template
        self.html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ ticker }} - Trading Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            color: #666;
            font-size: 1.2em;
        }
        
        .report-meta {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #f0f0f0;
        }
        
        .meta-item {
            text-align: center;
        }
        
        .meta-item .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        
        .meta-item .label {
            color: #888;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .section {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }
        
        .card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        .key-levels {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .level-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .level-card .level-value {
            font-size: 1.5em;
            font-weight: bold;
            margin: 5px 0;
        }
        
        .level-card .level-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .signal-section {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .signal-section.bullish {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        
        .signal-section.bearish {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }
        
        .signal-section .signal {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .signal-section .confidence {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .evidence-list {
            list-style: none;
            margin-top: 20px;
            text-align: left;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 20px;
        }
        
        .evidence-list li {
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .evidence-list li:last-child {
            border-bottom: none;
        }
        
        .statistics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .statistics-table th, .statistics-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }
        
        .statistics-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
        }
        
        .statistics-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .high-prob {
            background-color: #d4edda !important;
            color: #155724;
            font-weight: bold;
        }
        
        .medium-prob {
            background-color: #fff3cd !important;
            color: #856404;
        }
        
        .low-prob {
            color: #6c757d;
        }
        
        .highlight-column {
            background-color: #cce5ff !important;
            font-weight: bold;
        }
        
        .sr-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .sr-table th, .sr-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .sr-table th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #667eea;
        }
        
        .support-zone {
            background-color: #d4edda;
        }
        
        .resistance-zone {
            background-color: #f8d7da;
        }
        
        .chart-container {
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: white;
            margin-top: 50px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
            
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>{{ ticker }} - Trading Analysis Report</h1>
            <div class="subtitle">Generated on {{ timestamp }}</div>
            
            <div class="report-meta">
                <div class="meta-item">
                    <div class="value">{{ current_price }}</div>
                    <div class="label">Current Price</div>
                </div>
                <div class="meta-item">
                    <div class="value">{{ opening_type }}</div>
                    <div class="label">Opening Type</div>
                </div>
                <div class="meta-item">
                    <div class="value">{{ day_type }}</div>
                    <div class="label">Day Type</div>
                </div>
                <div class="meta-item">
                    <div class="value">{{ atr }}</div>
                    <div class="label">ATR (14)</div>
                </div>
            </div>
        </div>
        
        <!-- Trading Signal -->
        {{ signal_section }}
        
        <!-- Key Levels -->
        <div class="section">
            <h2>🎯 Key Levels</h2>
            <div class="key-levels">
                {{ key_levels }}
            </div>
        </div>
        
        <!-- Market Profile Chart -->
        <div class="section">
            <h2>📊 Market Profile Analysis</h2>
            <div class="chart-container">
                <div id="market-profile-chart"></div>
            </div>
            {{ market_profile_data }}
        </div>
        
        <!-- Price Chart with Indicators -->
        <div class="section">
            <h2>📈 Price Action & Technical Analysis</h2>
            <div class="chart-container">
                <div id="price-chart"></div>
            </div>
            {{ technical_analysis }}
        </div>
        
        <!-- Historical Statistics -->
        <div class="section">
            <h2>📊 Historical Opening Type Probabilities</h2>
            {{ statistics_table }}
        </div>
        
        <!-- Support/Resistance Zones -->
        <div class="section">
            <h2>🔄 Support & Resistance Zones</h2>
            {{ sr_zones_table }}
        </div>
        
        <!-- ML Predictions -->
        <div class="section">
            <h2>🤖 Machine Learning Predictions</h2>
            <div class="grid">
                {{ ml_predictions }}
            </div>
        </div>
        
        <!-- Technical Indicators Summary -->
        <div class="section">
            <h2>📉 Technical Indicators Summary</h2>
            <div class="grid">
                {{ technical_summary }}
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>© 2024 Advanced Trading System | Data provided by Yahoo Finance</p>
            <p>⚠️ This report is for informational purposes only and should not be considered as financial advice.</p>
        </div>
    </div>
    
    <!-- Plotly Charts Scripts -->
    <script>
        {{ chart_scripts }}
    </script>
</body>
</html>
        """
    
    def generate_report(self, 
                       market_profile: Dict,
                       technical_data: pd.DataFrame,
                       sr_analysis: Dict,
                       ml_predictions: Dict,
                       signal: Dict,
                       statistics: Dict,
                       price_data: pd.DataFrame) -> str:
        """
        Generate complete HTML report
        
        Args:
            market_profile: Current market profile data
            technical_data: Technical indicators DataFrame
            sr_analysis: Support/Resistance analysis
            ml_predictions: ML model predictions
            signal: Trading signal with evidence
            statistics: Historical statistics
            price_data: Price data for charts
        
        Returns:
            Path to generated HTML report
        """
        # Prepare template variables
        template_vars = {
            'ticker': self.ticker,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': f"${technical_data['Close'].iloc[-1]:.2f}" if not technical_data.empty else 'N/A',
            'opening_type': market_profile.get('opening_type', 'Unknown'),
            'day_type': market_profile.get('day_type', 'Unknown'),
            'atr': f"{technical_data['ATR'].iloc[-1]:.2f}" if 'ATR' in technical_data.columns else 'N/A'
        }
        
        # Generate sections
        template_vars['signal_section'] = self._generate_signal_section(signal)
        template_vars['key_levels'] = self._generate_key_levels(market_profile, sr_analysis)
        template_vars['market_profile_data'] = self._generate_market_profile_section(market_profile)
        template_vars['technical_analysis'] = self._generate_technical_section(technical_data)
        template_vars['statistics_table'] = self._generate_statistics_table(statistics, market_profile.get('opening_type'))
        template_vars['sr_zones_table'] = self._generate_sr_zones_table(sr_analysis)
        template_vars['ml_predictions'] = self._generate_ml_predictions(ml_predictions)
        template_vars['technical_summary'] = self._generate_technical_summary(technical_data)
        
        # Generate interactive charts
        chart_scripts = self._generate_charts(price_data, technical_data, market_profile, sr_analysis)
        template_vars['chart_scripts'] = chart_scripts
        
        # Render template
        template = Template(self.html_template)
        html_content = template.render(**template_vars)
        
        # Save report
        report_path = self.reports_dir / f"{self.ticker}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_signal_section(self, signal: Dict) -> str:
        """Generate signal section HTML"""
        signal_type = signal.get('signal', 'NEUTRAL')
        score = signal.get('score', 50)
        confidence = signal.get('confidence', 'LOW')
        evidence = signal.get('evidence', [])
        
        # Determine CSS class based on signal
        if 'LONG' in signal_type:
            css_class = 'bullish'
        elif 'SHORT' in signal_type:
            css_class = 'bearish'
        else:
            css_class = ''
        
        evidence_html = '\n'.join([f'<li>✓ {e}</li>' for e in evidence[:10]])
        
        return f"""
        <div class="signal-section {css_class}">
            <h2>🎯 Trading Signal</h2>
            <div class="signal">{signal_type}</div>
            <div class="confidence">Confidence: {confidence} ({score:.1f}/100)</div>
            <ul class="evidence-list">
                {evidence_html}
            </ul>
        </div>
        """
    
    def _generate_key_levels(self, market_profile: Dict, sr_analysis: Dict) -> str:
        """Generate key levels HTML"""
        levels = []
        
        # Market Profile levels
        if market_profile:
            levels.append(('POC', market_profile.get('poc', 0), 'level-card'))
            levels.append(('VAH', market_profile.get('vah', 0), 'level-card'))
            levels.append(('VAL', market_profile.get('val', 0), 'level-card'))
            levels.append(('IBH', market_profile.get('ib_high', 0), 'level-card'))
            levels.append(('IBL', market_profile.get('ib_low', 0), 'level-card'))
        
        # Key S/R levels
        if sr_analysis and 'key_resistance' in sr_analysis and sr_analysis['key_resistance']:
            levels.append(('R1', sr_analysis['key_resistance'][0]['zone_center'], 'level-card resistance-zone'))
        
        if sr_analysis and 'key_support' in sr_analysis and sr_analysis['key_support']:
            levels.append(('S1', sr_analysis['key_support'][0]['zone_center'], 'level-card support-zone'))
        
        html = ''
        for label, value, css_class in levels:
            if value:
                html += f"""
                <div class="{css_class}">
                    <div class="level-label">{label}</div>
                    <div class="level-value">${value:.2f}</div>
                </div>
                """
        
        return html
    
    def _generate_market_profile_section(self, market_profile: Dict) -> str:
        """Generate market profile section HTML"""
        if not market_profile:
            return "<p>No market profile data available</p>"
        
        html = """
        <div class="grid">
            <div class="card">
                <h3>Value Area</h3>
                <p><strong>Width:</strong> ${:.2f}</p>
                <p><strong>POC Migration:</strong> {:.2f} ticks</p>
            </div>
            <div class="card">
                <h3>Initial Balance</h3>
                <p><strong>Range:</strong> ${:.2f}</p>
                <p><strong>Extension:</strong> {:.1f}%</p>
            </div>
            <div class="card">
                <h3>Session Stats</h3>
                <p><strong>Range:</strong> ${:.2f}</p>
                <p><strong>TPO Count:</strong> {}</p>
            </div>
        </div>
        """.format(
            market_profile.get('va_width', 0),
            (market_profile.get('poc', 0) - market_profile.get('prior_poc', market_profile.get('poc', 0))) / 0.25,
            market_profile.get('ib_range', 0),
            ((market_profile.get('session_high', 0) - market_profile.get('session_low', 0)) / 
             market_profile.get('ib_range', 1) - 1) * 100 if market_profile.get('ib_range', 0) > 0 else 0,
            market_profile.get('session_high', 0) - market_profile.get('session_low', 0),
            market_profile.get('total_tpos', 0)
        )
        
        return html
    
    def _generate_technical_section(self, technical_data: pd.DataFrame) -> str:
        """Generate technical analysis section HTML"""
        if technical_data.empty:
            return "<p>No technical data available</p>"
        
        last_row = technical_data.iloc[-1]
        
        indicators = [
            ('RSI', last_row.get('RSI', 0), 'Overbought' if last_row.get('RSI', 50) > 70 else 'Oversold' if last_row.get('RSI', 50) < 30 else 'Neutral'),
            ('ADX', last_row.get('ADX', 0), 'Trending' if last_row.get('ADX', 0) > 25 else 'Ranging'),
            ('MACD', last_row.get('MACD_Histogram', 0), 'Bullish' if last_row.get('MACD_Histogram', 0) > 0 else 'Bearish'),
            ('Stochastic', last_row.get('Stoch_K', 0), 'Overbought' if last_row.get('Stoch_K', 50) > 80 else 'Oversold' if last_row.get('Stoch_K', 50) < 20 else 'Neutral')
        ]
        
        html = '<div class="grid">'
        for name, value, status in indicators:
            html += f"""
            <div class="card">
                <h3>{name}</h3>
                <p><strong>Value:</strong> {value:.2f}</p>
                <p><strong>Status:</strong> {status}</p>
            </div>
            """
        html += '</div>'
        
        return html
    
    def _generate_statistics_table(self, statistics: Dict, current_opening_type: str) -> str:
        """Generate statistics table HTML"""
        if not statistics:
            return "<p>No statistical data available</p>"
        
        # Map opening type
        opening_type_mapping = {
            'Open Drive Up': 'HOR',
            'Open Test Drive Up': 'HOR',
            'Open Auction Above POC': 'HIR',
            'Open Auction In Range': 'HIR',
            'Open Auction Below POC': 'LIR',
            'Open Test Drive Down': 'LOR',
            'Open Drive Down': 'LOR'
        }
        
        current_type = opening_type_mapping.get(current_opening_type, None)
        
        events = [
            'IBH', 'IBL', 'IBH or IBL', 'IBH & IBL (Neutral)',
            '1.5X IBH', '1.5X IBL', '2X IBH', '2X IBL',
            'pVAH', 'pVAL', 'pPOC', 'pCL (Gap)',
            'pHI (Range Gap)', 'pLO', 'pMID', '1/2 Gap',
            'Inside Day', 'Outside Day'
        ]
        
        html = """
        <table class="statistics-table">
            <thead>
                <tr>
                    <th>Event</th>
        """
        
        for otype in ['HOR', 'HIR', 'LIR', 'LOR']:
            count = statistics.get(otype, {}).get('count', 0)
            highlight = 'highlight-column' if current_type == otype else ''
            html += f'<th class="{highlight}">{otype}<br><small>(n={count})</small></th>'
        
        html += """
                </tr>
            </thead>
            <tbody>
        """
        
        for event in events:
            html += f"<tr><td>{event}</td>"
            
            for otype in ['HOR', 'HIR', 'LIR', 'LOR']:
                value = statistics.get(otype, {}).get('stats', {}).get(event, 0)
                
                # Color code based on probability
                if value >= 70:
                    cell_class = 'high-prob'
                    symbol = '▲ '
                elif value >= 50:
                    cell_class = 'medium-prob'
                    symbol = '● '
                else:
                    cell_class = 'low-prob'
                    symbol = ''
                
                highlight = 'highlight-column' if current_type == otype else ''
                html += f'<td class="{cell_class} {highlight}">{symbol}{value:.1f}%</td>'
            
            html += "</tr>"
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _generate_sr_zones_table(self, sr_analysis: Dict) -> str:
        """Generate S/R zones table HTML"""
        if not sr_analysis or 'all_zones' not in sr_analysis:
            return "<p>No S/R zones identified</p>"
        
        html = """
        <table class="sr-table">
            <thead>
                <tr>
                    <th>Zone Center</th>
                    <th>Type</th>
                    <th>Strength</th>
                    <th>Confluences</th>
                    <th>Distance</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for zone in sr_analysis['all_zones'][:10]:  # Top 10 zones
            zone_class = 'support-zone' if zone.get('type') == 'Support' else 'resistance-zone'
            confluences = ', '.join(zone.get('confluent_reasons', [])[:3])
            
            html += f"""
            <tr class="{zone_class}">
                <td>${zone['zone_center']:.2f}</td>
                <td>{zone.get('type', 'Unknown')}</td>
                <td>{zone.get('confluence_score', 0)}</td>
                <td>{confluences}</td>
                <td>{zone.get('distance_percent', 0):.1f}%</td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _generate_ml_predictions(self, ml_predictions: Dict) -> str:
        """Generate ML predictions HTML"""
        if not ml_predictions:
            return "<p>No ML predictions available</p>"
        
        html = ''
        
        for target, pred in ml_predictions.items():
            if pred and 'prediction' in pred:
                html += f"""
                <div class="card">
                    <h3>{target.replace('_', ' ').title()}</h3>
                    <p><strong>Prediction:</strong> {pred['prediction']}</p>
                    <p><strong>Confidence:</strong> {pred.get('confidence', 0):.1%}</p>
                </div>
                """
        
        return html
    
    def _generate_technical_summary(self, technical_data: pd.DataFrame) -> str:
        """Generate technical summary HTML"""
        if technical_data.empty:
            return "<p>No technical data available</p>"
        
        last_row = technical_data.iloc[-1]
        
        summaries = []
        
        # Trend summary
        if 'ADX' in last_row:
            if last_row['ADX'] > 25:
                if last_row.get('Plus_DI', 0) > last_row.get('Minus_DI', 0):
                    summaries.append(('Trend', 'Strong Uptrend', 'bullish'))
                else:
                    summaries.append(('Trend', 'Strong Downtrend', 'bearish'))
            else:
                summaries.append(('Trend', 'No Clear Trend', 'neutral'))
        
        # Momentum summary
        if 'RSI' in last_row:
            rsi = last_row['RSI']
            if rsi > 70:
                summaries.append(('Momentum', f'Overbought (RSI: {rsi:.1f})', 'bearish'))
            elif rsi < 30:
                summaries.append(('Momentum', f'Oversold (RSI: {rsi:.1f})', 'bullish'))
            else:
                summaries.append(('Momentum', f'Neutral (RSI: {rsi:.1f})', 'neutral'))
        
        # Volatility summary
        if 'BB_Width' in last_row:
            bb_width = last_row['BB_Width']
            if bb_width > technical_data['BB_Width'].mean() * 1.5:
                summaries.append(('Volatility', 'High', 'warning'))
            else:
                summaries.append(('Volatility', 'Normal', 'neutral'))
        
        # Volume summary
        if 'Volume' in last_row:
            vol_ratio = last_row['Volume'] / technical_data['Volume'].mean()
            if vol_ratio > 1.5:
                summaries.append(('Volume', f'High ({vol_ratio:.1f}x avg)', 'bullish'))
            elif vol_ratio < 0.5:
                summaries.append(('Volume', f'Low ({vol_ratio:.1f}x avg)', 'bearish'))
            else:
                summaries.append(('Volume', f'Normal ({vol_ratio:.1f}x avg)', 'neutral'))
        
        html = ''
        for category, status, sentiment in summaries:
            html += f"""
            <div class="card">
                <h3>{category}</h3>
                <p class="{sentiment}">{status}</p>
            </div>
            """
        
        return html
    
    def _generate_charts(self, price_data: pd.DataFrame, technical_data: pd.DataFrame,
                        market_profile: Dict, sr_analysis: Dict) -> str:
        """Generate Plotly chart scripts"""
        scripts = []
        
        # Price chart with indicators
        if not price_data.empty:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=price_data.index,
                    open=price_data['Open'],
                    high=price_data['High'],
                    low=price_data['Low'],
                    close=price_data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add Bollinger Bands if available
            if 'BB_Upper' in technical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=technical_data.index,
                        y=technical_data['BB_Upper'],
                        name='BB Upper',
                        line=dict(color='gray', dash='dash')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=technical_data.index,
                        y=technical_data['BB_Lower'],
                        name='BB Lower',
                        line=dict(color='gray', dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Add EMAs if available
            for ema_period in [9, 21, 50, 200]:
                ema_col = f'EMA_{ema_period}'
                if ema_col in technical_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=technical_data.index,
                            y=technical_data[ema_col],
                            name=ema_col,
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )
            
            # Add horizontal lines for key levels
            if market_profile:
                levels = [
                    ('POC', market_profile.get('poc'), 'purple'),
                    ('VAH', market_profile.get('vah'), 'green'),
                    ('VAL', market_profile.get('val'), 'red'),
                    ('IBH', market_profile.get('ib_high'), 'blue'),
                    ('IBL', market_profile.get('ib_low'), 'orange')
                ]
                
                for name, level, color in levels:
                    if level:
                        fig.add_hline(
                            y=level,
                            line_dash="dash",
                            line_color=color,
                            annotation_text=name,
                            row=1, col=1
                        )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['Volume'],
                    name='Volume',
                    marker_color='blue'
                ),
                row=2, col=1
            )
            
            # RSI
            if 'RSI' in technical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=technical_data.index,
                        y=technical_data['RSI'],
                        name='RSI',
                        line=dict(color='orange')
                    ),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # Update layout
            fig.update_layout(
                title=f'{self.ticker} - Price Action Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            # Convert to HTML
            chart_html = pio.to_html(fig, include_plotlyjs=False, div_id="price-chart")
            scripts.append(chart_html.split('<script>')[1].split('</script>')[0] if '<script>' in chart_html else '')
        
        return '\n'.join(scripts)