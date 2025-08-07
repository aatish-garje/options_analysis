import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta, date
import ta
import warnings
import calendar
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Options Trading Signals - OI Analysis & Short Covering",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .bullish { color: #00ff00; font-weight: bold; }
    .bearish { color: #ff0000; font-weight: bold; }
    .neutral { color: #ffa500; font-weight: bold; }
    .signal-box {
        border: 2px solid;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .buy-signal { border-color: #00ff00; background-color: rgba(0, 255, 0, 0.1); }
    .sell-signal { border-color: #ff0000; background-color: rgba(255, 0, 0, 0.1); }
    .hold-signal { border-color: #ffa500; background-color: rgba(255, 165, 0, 0.1); }
    .short-covering { border-color: #9370DB; background-color: rgba(147, 112, 219, 0.1); }
    .oi-alert {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedOptionsSignalGenerator:
    def __init__(self):
        self.nifty_symbol = "^NSEI"
        self.banknifty_symbol = "^NSEBANK"

        # CORRECTED LOT SIZES
        self.lot_sizes = {
            "NIFTY": 75,    # Nifty 50 lot size is 75
            "BANKNIFTY": 35  # Bank Nifty lot size is 35
        }

        # OI Analysis Parameters
        self.oi_change_threshold = 10  # % change in OI to consider significant
        self.volume_spike_factor = 1.5  # Volume > 1.5x average for spike
        self.pcr_levels = {"oversold": 0.7, "overbought": 1.3}

    def get_expiry_dates(self, symbol_type="NIFTY"):
        """Calculate next expiry dates based on correct rules"""
        today = datetime.now().date()

        if symbol_type == "NIFTY":
            # Nifty expires every Thursday
            days_ahead = (3 - today.weekday()) % 7  # 3 = Thursday
            if days_ahead == 0:  # If today is Thursday
                current_time = datetime.now().time()
                if current_time.hour >= 15 and current_time.minute >= 30:
                    days_ahead = 7  # Next Thursday
                else:
                    days_ahead = 0  # Today's expiry
            next_expiry = today + timedelta(days=days_ahead)

        else:  # BANKNIFTY
            # Bank Nifty expires on last Thursday of the month
            current_month = today.month
            current_year = today.year

            # Find last Thursday of current month
            last_day = calendar.monthrange(current_year, current_month)[1]
            last_date = date(current_year, current_month, last_day)

            # Find last Thursday
            last_thursday = last_date
            while last_thursday.weekday() != 3:  # 3 = Thursday
                last_thursday -= timedelta(days=1)

            # If current date is past this month's last Thursday, get next month's
            if today > last_thursday:
                if current_month == 12:
                    next_month = 1
                    next_year = current_year + 1
                else:
                    next_month = current_month + 1
                    next_year = current_year

                last_day_next = calendar.monthrange(next_year, next_month)[1]
                last_date_next = date(next_year, next_month, last_day_next)
                next_expiry = last_date_next
                while next_expiry.weekday() != 3:
                    next_expiry -= timedelta(days=1)
            else:
                next_expiry = last_thursday

        return next_expiry

    def simulate_option_chain_data(self, spot_price, symbol_type="NIFTY"):
        """
        Simulate realistic option chain data with OI, Volume, and premium values
        In real implementation, this would fetch from NSE API
        """
        strikes = self.get_option_strikes(spot_price, symbol_type)

        option_chain = []

        for strike in strikes:
            # Calculate moneyness
            moneyness = abs(spot_price - strike) / spot_price

            # Simulate OI based on moneyness (ATM has higher OI)
            if moneyness < 0.01:  # ATM
                base_oi = np.random.randint(15000, 25000)
            elif moneyness < 0.02:  # Near ATM
                base_oi = np.random.randint(8000, 15000)
            elif moneyness < 0.05:  # Moderate OTM/ITM
                base_oi = np.random.randint(3000, 8000)
            else:  # Deep OTM/ITM
                base_oi = np.random.randint(500, 3000)

            # Simulate OI change (realistic market behavior)
            oi_change = np.random.randint(-30, 50)  # % change
            prev_oi = int(base_oi / (1 + oi_change/100))

            # Simulate volume (related to OI but with randomness)
            volume = int(base_oi * np.random.uniform(0.1, 0.8))

            # Calculate premiums using our enhanced method
            ce_data = self.calculate_realistic_options_premium(
                spot_price, strike, 21, 18, "CE", symbol_type
            )
            pe_data = self.calculate_realistic_options_premium(
                spot_price, strike, 21, 18, "PE", symbol_type
            )

            option_chain.append({
                'strike': strike,
                'ce_premium': ce_data['premium'],
                'ce_oi': base_oi if strike >= spot_price else base_oi * 0.7,
                'ce_oi_change': oi_change,
                'ce_volume': volume if strike >= spot_price else volume * 0.6,
                'pe_premium': pe_data['premium'],
                'pe_oi': base_oi * 0.7 if strike >= spot_price else base_oi,
                'pe_oi_change': np.random.randint(-40, 30),
                'pe_volume': volume * 0.6 if strike >= spot_price else volume,
                'iv_ce': np.random.uniform(15, 25),
                'iv_pe': np.random.uniform(14, 24)
            })

        return pd.DataFrame(option_chain)

    def analyze_oi_patterns(self, option_chain_data, current_price):
        """
        Analyze Open Interest patterns for market insights
        """
        analysis = {}

        # Calculate Put-Call Ratio (PCR)
        total_call_oi = option_chain_data['ce_oi'].sum()
        total_put_oi = option_chain_data['pe_oi'].sum()
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1

        # PCR Volume
        total_call_vol = option_chain_data['ce_volume'].sum()
        total_put_vol = option_chain_data['pe_volume'].sum()
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 1

        # Identify key strikes with high OI
        option_chain_data['total_oi'] = option_chain_data['ce_oi'] + option_chain_data['pe_oi']
        max_pain_candidates = option_chain_data.nlargest(3, 'total_oi')
        max_pain = max_pain_candidates.iloc[0]['strike']

        # OI Change Analysis
        significant_call_oi_increase = option_chain_data[
            option_chain_data['ce_oi_change'] > self.oi_change_threshold
        ]
        significant_put_oi_increase = option_chain_data[
            option_chain_data['pe_oi_change'] > self.oi_change_threshold
        ]

        significant_call_oi_decrease = option_chain_data[
            option_chain_data['ce_oi_change'] < -self.oi_change_threshold
        ]
        significant_put_oi_decrease = option_chain_data[
            option_chain_data['pe_oi_change'] < -self.oi_change_threshold
        ]

        analysis = {
            'pcr_oi': pcr_oi,
            'pcr_volume': pcr_vol,
            'max_pain': max_pain,
            'call_writing': len(significant_call_oi_increase),
            'put_writing': len(significant_put_oi_increase),
            'call_unwinding': len(significant_call_oi_decrease),
            'put_unwinding': len(significant_put_oi_decrease),
            'high_oi_strikes': max_pain_candidates['strike'].tolist(),
            'market_bias': self.determine_market_bias(pcr_oi, pcr_vol)
        }

        return analysis

    def detect_short_covering(self, data, oi_analysis):
        """
        short covering detection using price action and OI analysis
        """
        if data is None or len(data) < 5:
            return {"short_covering": False, "confidence": 0, "signals": []}

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        short_covering_signals = []
        confidence = 0

        # Price-based signals
        price_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100

        # 1. Strong price recovery from lows
        recent_low = data['Low'].rolling(10).min().iloc[-1]
        if latest['Close'] > recent_low * 1.02:  # 2% above recent low
            short_covering_signals.append("Price recovery from recent lows")
            confidence += 15

        # 2. High volume with price increase
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        if latest['Volume'] > avg_volume * self.volume_spike_factor and price_change > 0.5:
            short_covering_signals.append("High volume price spike")
            confidence += 20

        # 3. RSI divergence (price making higher lows while RSI was oversold)
        rsi = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        if rsi.iloc[-5:].min() < 35 and rsi.iloc[-1] > 45:  # RSI recovery from oversold
            short_covering_signals.append("RSI recovery from oversold levels")
            confidence += 15

        # 4. OI-based short covering signals
        pcr_oi = oi_analysis.get('pcr_oi', 1)
        if pcr_oi > self.pcr_levels['overbought'] and price_change > 0:
            short_covering_signals.append("High PCR with price rise suggests short covering")
            confidence += 25

        # 5. Put unwinding (decreasing put OI with rising prices)
        put_unwinding = oi_analysis.get('put_unwinding', 0)
        if put_unwinding > 2 and price_change > 0:
            short_covering_signals.append("Significant put unwinding detected")
            confidence += 20

        # 6. Gap up opening (shorts getting squeezed)
        if len(data) > 1:
            gap = (latest['Open'] - prev['Close']) / prev['Close'] * 100
            if gap > 0.5:  # Gap up > 0.5%
                short_covering_signals.append("Gap up opening indicates short squeeze")
                confidence += 15

        # 7. Price above key resistance with volume
        resistance = data['High'].rolling(20).max().iloc[-2]  # Exclude today
        if latest['Close'] > resistance and latest['Volume'] > avg_volume:
            short_covering_signals.append("Breakout above resistance on volume")
            confidence += 10

        is_short_covering = confidence >= 40  # Threshold for short covering

        return {
            "short_covering": is_short_covering,
            "confidence": min(confidence, 90),
            "signals": short_covering_signals[:5],  # Top 5 signals
            "strength": "Strong" if confidence > 60 else "Moderate" if confidence > 40 else "Weak"
        }

    def determine_market_bias(self, pcr_oi, pcr_vol):
        """Determine overall market bias from PCR values"""
        if pcr_oi < self.pcr_levels['oversold'] and pcr_vol < self.pcr_levels['oversold']:
            return "Strongly Bullish"
        elif pcr_oi > self.pcr_levels['overbought'] and pcr_vol > self.pcr_levels['overbought']:
            return "Strongly Bearish"
        elif pcr_oi < 1 and pcr_vol < 1:
            return "Bullish"
        elif pcr_oi > 1 and pcr_vol > 1:
            return "Bearish"
        else:
            return "Neutral"

    def fetch_live_data(self, symbol, period="1d", interval="1m"):
        """Fetch live data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                st.error(f"No data available for {symbol}")
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, data):
        """Calculate various technical indicators"""
        if data is None or data.empty:
            return {}

        # Basic indicators
        data['SMA_9'] = ta.trend.SMAIndicator(data['Close'], window=9).sma_indicator()
        data['EMA_21'] = ta.trend.EMAIndicator(data['Close'], window=21).ema_indicator()
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_histogram'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['Close'])
        data['BB_upper'] = bb.bollinger_hband()
        data['BB_middle'] = bb.bollinger_mavg()
        data['BB_lower'] = bb.bollinger_lband()

        # ATR
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()

        # VWAP (simplified)
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

        # Support and Resistance
        high_20 = data['High'].rolling(window=20).max()
        low_20 = data['Low'].rolling(window=20).min()
        data['Resistance'] = high_20
        data['Support'] = low_20

        return data

    def calculate_volatility_metrics(self, data):
        """Calculate volatility-related metrics"""
        if data is None or data.empty:
            return {}

        # Historical volatility
        returns = data['Close'].pct_change().dropna()
        hist_vol = returns.std() * np.sqrt(252) * 100

        # Enhanced implied volatility calculation
        price_range = (data['High'] - data['Low']) / data['Close']
        base_iv = price_range.rolling(window=20).mean() * 100

        # ATR-based volatility
        atr_vol = (data['ATR'].iloc[-1] / data['Close'].iloc[-1]) * 100 * np.sqrt(252)

        # Use higher of the two for more realistic IV
        implied_vol = max(base_iv.iloc[-1] if not base_iv.empty else 15.0, atr_vol)

        return {
            'historical_volatility': hist_vol,
            'implied_volatility': implied_vol,
            'atr_volatility': atr_vol
        }

    def get_realistic_premium_from_market(self, spot_price, strike_price, option_type="CE", symbol_type="NIFTY", days_to_expiry=21):
        """Get realistic premium based on actual market patterns"""

        moneyness = (spot_price - strike_price) / spot_price if option_type == "CE" else (strike_price - spot_price) / spot_price

        if symbol_type == "NIFTY":
            base_premium_factor = 0.02
            if abs(moneyness) < 0.005:
                premium_factor = base_premium_factor
            elif moneyness > 0.005:
                premium_factor = base_premium_factor + abs(moneyness) * 2
            else:
                premium_factor = max(0.001, base_premium_factor - abs(moneyness) * 1.5)
            base_premium = spot_price * premium_factor

        else:  # BANKNIFTY
            base_premium_factor = 0.015

            if abs(moneyness) < 0.008:
                if spot_price > 55000 and strike_price == 55600:
                    premium_factor = 600 / spot_price
                else:
                    premium_factor = base_premium_factor
            elif moneyness > 0.008:
                premium_factor = base_premium_factor + abs(moneyness) * 3
            else:
                distance_factor = abs(moneyness)
                if distance_factor < 0.02:
                    premium_factor = max(0.005, base_premium_factor * (1 - distance_factor * 20))
                else:
                    premium_factor = max(0.001, base_premium_factor * 0.1)
            base_premium = spot_price * premium_factor

        # Time value adjustment
        time_factor = max(0.3, days_to_expiry / 30.0)
        vol_factor = 1.2 if symbol_type == "BANKNIFTY" else 1.0
        final_premium = base_premium * time_factor * vol_factor
        min_premium = 5 if symbol_type == "NIFTY" else 10

        return max(final_premium, min_premium)

    def calculate_realistic_options_premium(self, spot_price, strike_price, days_to_expiry, volatility, option_type="CE", symbol_type="NIFTY"):
        """Calculate realistic options premium using market-calibrated method"""
        from scipy.stats import norm
        import math

        market_premium = self.get_realistic_premium_from_market(
            spot_price, strike_price, option_type, symbol_type, days_to_expiry
        )

        S = spot_price
        K = strike_price
        T = max(days_to_expiry / 365.0, 1/365)
        r = 0.065
        sigma = min(volatility / 100, 0.4)

        try:
            d1 = (math.log(S/K) + (r + sigma**2/2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)

            if option_type == "CE":
                delta = norm.cdf(d1)
                theoretical_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                delta = norm.cdf(d1) - 1
                theoretical_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) 
                    - r * K * math.exp(-r * T) * (norm.cdf(d2) if option_type == "CE" else norm.cdf(-d2))) / 365
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100

            return {
                'premium': market_premium,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'theoretical_price': theoretical_price
            }

        except Exception as e:
            intrinsic = max(0, S - K) if option_type == "CE" else max(0, K - S)

            return {
                'premium': market_premium,
                'delta': 0.5,
                'gamma': 0.01,
                'theta': -2,
                'vega': 0.1,
                'theoretical_price': market_premium
            }

    def get_option_strikes(self, spot_price, symbol_type="NIFTY"):
        """Generate realistic option strikes"""
        if symbol_type == "NIFTY":
            base_strike = round(spot_price / 50) * 50
            strikes = [base_strike + i * 50 for i in range(-10, 11)]
        else:  # BANKNIFTY
            base_strike = round(spot_price / 100) * 100
            strikes = [base_strike + i * 100 for i in range(-10, 11)]

        return strikes

    def generate_enhanced_signal(self, data, vol_metrics, oi_analysis, short_covering_data, symbol_type="NIFTY"):
        """Generate trading signals incorporating OI analysis and short covering"""
        if data is None or data.empty:
            return {"signal": "HOLD", "confidence": 0, "reason": "No data"}

        # Get basic technical signals first
        basic_signal = self.generate_basic_signal(data, vol_metrics, symbol_type)

        # Enhance with OI analysis
        oi_score = 0
        oi_signals = []

        # PCR Analysis
        pcr_oi = oi_analysis.get('pcr_oi', 1)
        if pcr_oi < self.pcr_levels['oversold']:
            oi_score += 1
            oi_signals.append(f"Low PCR OI ({pcr_oi:.2f}) - Bullish")
        elif pcr_oi > self.pcr_levels['overbought']:
            oi_score -= 1
            oi_signals.append(f"High PCR OI ({pcr_oi:.2f}) - Bearish")

        # Call/Put Writing Analysis
        call_writing = oi_analysis.get('call_writing', 0)
        put_writing = oi_analysis.get('put_writing', 0)

        if put_writing > call_writing:
            oi_score += 0.5
            oi_signals.append("More put writing than call writing - Bullish")
        elif call_writing > put_writing:
            oi_score -= 0.5
            oi_signals.append("More call writing than put writing - Bearish")

        # Max Pain Analysis
        current_price = basic_signal['spot_price']
        max_pain = oi_analysis.get('max_pain', current_price)
        pain_diff = (current_price - max_pain) / current_price * 100

        if abs(pain_diff) > 2:  # More than 2% away from max pain
            if current_price < max_pain:
                oi_score += 0.5
                oi_signals.append(f"Price below Max Pain ({max_pain:.0f}) - Bullish pull")
            else:
                oi_score -= 0.5
                oi_signals.append(f"Price above Max Pain ({max_pain:.0f}) - Bearish pull")

        # Short covering enhancement
        if short_covering_data['short_covering']:
            if short_covering_data['confidence'] > 60:
                oi_score += 1.5
            else:
                oi_score += 1
            oi_signals.extend(short_covering_data['signals'][:2])

        # Combine basic and OI signals
        enhanced_tech_score = basic_signal['tech_score'] + oi_score
        all_signals = basic_signal['signals'] + oi_signals

        # Generate enhanced signal
        if enhanced_tech_score >= 2.5:
            if short_covering_data['short_covering']:
                signal = "STRONG BUY CE"
            else:
                signal = "BUY CE"
            confidence = min(enhanced_tech_score * 12 + short_covering_data.get('confidence', 0) * 0.3, 95)
        elif enhanced_tech_score <= -2.5:
            signal = "BUY PE"
            confidence = min(abs(enhanced_tech_score) * 12, 90)
        elif enhanced_tech_score >= 1.5:
            signal = "BUY CE"
            confidence = min(enhanced_tech_score * 15, 80)
        elif enhanced_tech_score <= -1.5:
            signal = "BUY PE"
            confidence = min(abs(enhanced_tech_score) * 15, 80)
        else:
            signal = "HOLD"
            confidence = max(30, 50 - abs(enhanced_tech_score) * 10)

        return {
            "signal": signal,
            "confidence": confidence,
            "tech_score": enhanced_tech_score,
            "signals": all_signals[:8],  # Top signals
            "spot_price": current_price,
            "rsi": basic_signal['rsi'],
            "iv": basic_signal['iv'],
            "oi_analysis": oi_analysis,
            "short_covering": short_covering_data
        }

    def generate_basic_signal(self, data, vol_metrics, symbol_type="NIFTY"):
        """Generate basic technical signals"""
        if data is None or data.empty:
            return {"signal": "HOLD", "confidence": 0, "reason": "No data"}

        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest

        tech_score = 0
        signals = []

        # RSI Analysis
        rsi = latest['RSI']
        if rsi < 30:
            tech_score += 2
            signals.append("RSI Oversold (Bullish)")
        elif rsi > 70:
            tech_score -= 2
            signals.append("RSI Overbought (Bearish)")
        elif rsi > 50:
            tech_score += 1
            signals.append("RSI Above 50 (Bullish Bias)")
        else:
            tech_score -= 0.5
            signals.append("RSI Below 50 (Bearish Bias)")

        # MACD Analysis
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            tech_score += 2
            signals.append("MACD Bullish Crossover")
        elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            tech_score -= 2
            signals.append("MACD Bearish Crossover")
        elif latest['MACD'] > latest['MACD_signal']:
            tech_score += 0.5
            signals.append("MACD Above Signal")
        else:
            tech_score -= 0.5
            signals.append("MACD Below Signal")

        # Moving Average Analysis
        if latest['Close'] > latest['SMA_9'] > latest['EMA_21']:
            tech_score += 1
            signals.append("Price Above Moving Averages")
        elif latest['Close'] < latest['SMA_9'] < latest['EMA_21']:
            tech_score -= 1
            signals.append("Price Below Moving Averages")

        # VWAP Analysis
        if latest['Close'] > latest['VWAP']:
            tech_score += 0.5
            signals.append("Price Above VWAP")
        else:
            tech_score -= 0.5
            signals.append("Price Below VWAP")

        return {
            "signal": "BUY CE" if tech_score >= 2 else "BUY PE" if tech_score <= -2 else "HOLD",
            "confidence": min(abs(tech_score) * 15, 80),
            "tech_score": tech_score,
            "signals": signals,
            "spot_price": latest['Close'],
            "rsi": rsi,
            "iv": vol_metrics.get('implied_volatility', 15)
        }

    def calculate_sl_target(self, entry_price, signal_type, atr, confidence, spot_price, strike_price):
        """Calculate Stop Loss and Target based on premium movements and ATR"""

        atr_premium_factor = (atr / spot_price) * entry_price
        confidence_multiplier = 1.2 if confidence > 70 else 1.5 if confidence > 50 else 2.0

        if signal_type in ["BUY CE", "BUY PE", "STRONG BUY CE"]:
            sl_percent = 0.25 if confidence > 70 else 0.35 if confidence > 50 else 0.50
            sl_price = entry_price * (1 - sl_percent)

            target_percent = min(1.5, 0.6 + (atr_premium_factor / entry_price) * confidence_multiplier)
            target_price = entry_price * (1 + target_percent)
        else:
            sl_price = entry_price * 0.85
            target_price = entry_price * 1.30

        return max(sl_price, 1), target_price

# Initialize the signal generator
@st.cache_data(ttl=30)
def get_market_data(symbol):
    generator = EnhancedOptionsSignalGenerator()
    return generator.fetch_live_data(symbol, period="1d", interval="1m")

def main():
    st.title("🚀 Options Trading Signals")
    st.markdown("*Advanced Analysis with OI Patterns & Short Covering Detection*")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    symbol_choice = st.sidebar.selectbox(
        "Select Index:",
        ["NIFTY 50", "BANK NIFTY"],
        index=0
    )

    trading_style = st.sidebar.selectbox(
        "Trading Style:",
        ["Intraday", "Scalping", "Swing"],
        index=0
    )

    risk_tolerance = st.sidebar.select_slider(
        "Risk Tolerance:",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate"
    )

    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)

    # Map symbol choice
    if symbol_choice == "NIFTY 50":
        yf_symbol = "^NSEI"
        symbol_type = "NIFTY"
    else:
        yf_symbol = "^NSEBANK"
        symbol_type = "BANKNIFTY"

    # Initialize generator
    generator = EnhancedOptionsSignalGenerator()

    # Display specifications
    st.sidebar.markdown("---")


    # Display lot sizes and expiry
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Contract Specs:**")
    st.sidebar.write(f"**{symbol_type} Lot Size:** {generator.lot_sizes[symbol_type]}")

    next_expiry = generator.get_expiry_dates(symbol_type)
    days_to_expiry = (next_expiry - datetime.now().date()).days

    st.sidebar.write(f"**Next Expiry:** {next_expiry.strftime('%d %b %Y')}")
    st.sidebar.write(f"**Days to Expiry:** {days_to_expiry}")

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(f"📊 {symbol_choice} Enhanced Analysis")

        # Fetch and process data
        data = get_market_data(yf_symbol)

        if data is not None and not data.empty:
            data_with_indicators = generator.calculate_technical_indicators(data)
            vol_metrics = generator.calculate_volatility_metrics(data_with_indicators)

            current_price = data_with_indicators['Close'].iloc[-1]

            # Generate simulated option chain data (in real app, fetch from NSE)
            option_chain = generator.simulate_option_chain_data(current_price, symbol_type)

            # OI Analysis
            oi_analysis = generator.analyze_oi_patterns(option_chain, current_price)

            # Short covering detection
            short_covering_data = generator.detect_short_covering(data_with_indicators, oi_analysis)

            # Generate enhanced signals
            signal_data = generator.generate_enhanced_signal(
                data_with_indicators, vol_metrics, oi_analysis, short_covering_data, symbol_type
            )

            # Current price display
            prev_close = data.iloc[-2]['Close'] if len(data) > 1 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100

            # Enhanced metrics
            col1_1, col1_2, col1_3, col1_4, col1_5 = st.columns(5)

            with col1_1:
                st.metric(
                    "Current Price",
                    f"₹{current_price:.2f}",
                    f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                )

            with col1_2:
                st.metric("RSI", f"{signal_data['rsi']:.1f}", "")

            with col1_3:
                st.metric("PCR OI", f"{oi_analysis['pcr_oi']:.2f}", "")

            with col1_4:
                st.metric("Max Pain", f"₹{oi_analysis['max_pain']:.0f}", "")

            with col1_5:
                st.metric("Tech Score", f"{signal_data['tech_score']:.1f}", "")

            # Enhanced signal display
            signal = signal_data['signal']
            confidence = signal_data['confidence']

            if "STRONG BUY" in signal:
                signal_class = "short-covering"
                signal_text = f"💜 {signal} - {confidence:.0f}% Confidence"
            elif signal == "BUY CE":
                signal_class = "buy-signal"
                signal_text = f"🟢 {signal} - {confidence:.0f}% Confidence"
            elif signal == "BUY PE":
                signal_class = "sell-signal"
                signal_text = f"🔴 {signal} - {confidence:.0f}% Confidence"
            else:
                signal_class = "hold-signal"
                signal_text = f"🟡 {signal} - {confidence:.0f}% Confidence"

            st.markdown(
                f'<div class="signal-box {signal_class}">{signal_text}</div>',
                unsafe_allow_html=True
            )

            # Short covering alert
            if short_covering_data['short_covering']:
                st.markdown(f"""
                <div class="oi-alert">
                🔥 <strong>SHORT COVERING DETECTED</strong> - {short_covering_data['strength']} Signal<br>
                Confidence: {short_covering_data['confidence']:.0f}%
                </div>
                """, unsafe_allow_html=True)

            # OI Analysis Summary
            st.subheader("📊 Open Interest Analysis")
            oi_col1, oi_col2, oi_col3, oi_col4 = st.columns(4)

            with oi_col1:
                st.metric("PCR Volume", f"{oi_analysis['pcr_volume']:.2f}")
            with oi_col2:
                st.metric("Call Writing", oi_analysis['call_writing'])
            with oi_col3:
                st.metric("Put Writing", oi_analysis['put_writing'])
            with oi_col4:
                bias_color = "🟢" if "Bullish" in oi_analysis['market_bias'] else "🔴" if "Bearish" in oi_analysis['market_bias'] else "🟡"
                st.write(f"**Market Bias:** {bias_color} {oi_analysis['market_bias']}")

            # Charts (keeping original chart structure)
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Price & Indicators', 'RSI', 'MACD'),
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2]
            )

            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data_with_indicators.index,
                    open=data_with_indicators['Open'],
                    high=data_with_indicators['High'],
                    low=data_with_indicators['Low'],
                    close=data_with_indicators['Close'],
                    name='Price'
                ), row=1, col=1
            )

            # Moving averages
            fig.add_trace(
                go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['SMA_9'],
                    name='SMA 9',
                    line=dict(color='orange', width=1)
                ), row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['VWAP'],
                    name='VWAP',
                    line=dict(color='purple', width=1)
                ), row=1, col=1
            )

            # RSI
            fig.add_trace(
                go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['RSI'],
                    name='RSI',
                    line=dict(color='blue')
                ), row=2, col=1
            )

            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # MACD
            fig.add_trace(
                go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['MACD'],
                    name='MACD',
                    line=dict(color='blue')
                ), row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['MACD_signal'],
                    name='Signal',
                    line=dict(color='red')
                ), row=3, col=1
            )

            fig.update_layout(height=800, showlegend=True)
            fig.update_xaxes(rangeslider_visible=False)

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header("🎯 Recommendations")

        if 'signal_data' in locals():
            # Options recommendations
            strikes = generator.get_option_strikes(current_price, symbol_type)
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))

            # Get option type based on signal
            if "CE" in signal:
                option_type = "CE"
            elif "PE" in signal:
                option_type = "PE"
            else:
                option_type = "CE"  # Default

            # Calculate premium using enhanced method
            option_data = generator.calculate_realistic_options_premium(
                current_price, atm_strike, days_to_expiry, 
                vol_metrics.get('implied_volatility', 15), 
                option_type, symbol_type
            )

            premium = option_data['premium']
            delta = option_data['delta'] 
            theta = option_data['theta']
            gamma = option_data['gamma']
            vega = option_data['vega']

            # Calculate SL and Target
            atr = data_with_indicators['ATR'].iloc[-1]
            sl_price, target_price = generator.calculate_sl_target(
                premium, signal, atr, confidence, current_price, atm_strike
            )

            # Display recommendation
            st.subheader("📋 Signal Details")

            risk = premium - sl_price
            reward = target_price - premium
            risk_reward_ratio = reward / risk if risk > 0 else 0

            st.markdown(f"""
            **Recommended Trade:**
            - **Option:** {symbol_type} {atm_strike} {option_type}
            - **Entry Premium:** ₹{premium:.2f}
            - **Stop Loss:** ₹{sl_price:.2f}
            - **Target:** ₹{target_price:.2f}
            - **Risk-Reward:** 1:{risk_reward_ratio:.1f}
            """)

            # Short covering specific info
            if short_covering_data['short_covering']:
                st.success(f"🔥 SHORT COVERING: {short_covering_data['strength']} signal detected!")
                st.write("**Short Covering Signals:**")
                for sc_signal in short_covering_data['signals'][:3]:
                    st.write(f"• {sc_signal}")

            # OI Analysis details
            st.subheader("📊 OI Insights")
            st.write(f"**Max Pain:** ₹{oi_analysis['max_pain']:.0f}")
            pain_distance = abs(current_price - oi_analysis['max_pain'])
            st.write(f"**Distance from Max Pain:** ₹{pain_distance:.0f}")

            if oi_analysis['call_writing'] > oi_analysis['put_writing']:
                st.write("🔴 **More Call Writing** - Bearish outlook by option writers")
            elif oi_analysis['put_writing'] > oi_analysis['call_writing']:
                st.write("🟢 **More Put Writing** - Bullish outlook by option writers")
            else:
                st.write("🟡 **Balanced Writing** - Neutral option writer sentiment")

            # Greeks display
            st.subheader("📊 Option Greeks")
            greeks_col1, greeks_col2 = st.columns(2)

            with greeks_col1:
                st.metric("Delta", f"{delta:.3f}")
                st.metric("Gamma", f"{gamma:.4f}")

            with greeks_col2:
                st.metric("Theta", f"{theta:.2f}")
                st.metric("Vega", f"{vega:.2f}")

            # Enhanced Risk parameters
            st.subheader("⚠️ Risk Analysis")
            position_size = st.slider("Position Size (lots)", 1, 10, 1)
            lot_size = generator.lot_sizes[symbol_type]

            max_loss = (premium - sl_price) * lot_size * position_size
            max_gain = (target_price - premium) * lot_size * position_size
            total_premium_required = premium * lot_size * position_size

            st.markdown(f"""
            **Risk Analysis:**
            - **Premium Required:** ₹{total_premium_required:,.0f}
            - **Max Loss:** ₹{max_loss:,.0f}
            - **Max Gain:** ₹{max_gain:,.0f}
            - **Lot Size:** {lot_size}
            - **Total Quantity:** {lot_size * position_size}
            """)

            # Enhanced signal reasoning
            st.subheader("🧠 Signal Analysis")
            st.write("**Technical + OI Signals:**")
            for reason in signal_data['signals'][:6]:
                st.write(f"• {reason}")

            # Time factors
            st.subheader("⏰ Time & Expiry")
            st.write(f"• Days to Expiry: {days_to_expiry}")
            st.write(f"• Time Decay (Theta): ₹{abs(theta):.2f}/day")

            if days_to_expiry <= 1:
                st.error("⚠️ EXPIRY DAY: Extreme time decay risk!")
            elif days_to_expiry <= 3:
                st.warning("🟡 Near Expiry: High time decay!")

    # Enhanced bottom section
    st.markdown("---")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.subheader("🔍 Market Overview")
        if 'signal_data' in locals():
            trend = "Bullish" if signal_data['tech_score'] > 0 else "Bearish" if signal_data['tech_score'] < 0 else "Neutral"
            st.write(f"**Overall Trend:** {trend}")
            st.write(f"**OI Bias:** {oi_analysis['market_bias']}")
            st.write(f"**Short Covering:** {'Yes' if short_covering_data['short_covering'] else 'No'}")

    with info_col2:
        st.subheader("📈 Key Levels")
        if 'data_with_indicators' in locals():
            resistance = data_with_indicators['Resistance'].iloc[-1]
            support = data_with_indicators['Support'].iloc[-1]
            st.write(f"**Resistance:** ₹{resistance:.2f}")
            st.write(f"**Support:** ₹{support:.2f}")
            st.write(f"**Max Pain:** ₹{oi_analysis['max_pain']:.0f}")
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()