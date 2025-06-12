import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta, timezone
from optionlab import run_strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

st.set_page_config(page_title="Option Buy Only (6_Option_Buy_Only)", layout="centered")
st.title("미국 옵션 매수 전략 시뮬레이터 (Option Buy Only)")

st.markdown("""
#### 🇰🇷 한국 투자자 전용 안내
- 이 페이지는 **미국 옵션 매수(롱콜, 콜스프레드) 전략**만 시뮬레이션합니다.
- 국내 증권사에서는 미국 옵션 매도(숏풋, 커버드콜)가 불가하므로, 매수 전략만 제공합니다.
- 승률 산출 기준: **프리미엄(투자금) 대비 목표수익률(%) + 거래비용(%)** 이상 순수익 달성 시 '승'으로 간주합니다.
""")

ticker = st.text_input("Enter ticker (e.g. AAPL, TSLA, SPY)", value="AAPL")

try:
    data = yf.Ticker(ticker)
    price = data.history(period='1d')['Close'].iloc[-1]
    st.subheader(f"Underlying ({ticker.upper()}) Price: ${price:.2f}")
    today_utc = datetime.now(timezone.utc).date()
    min_date = today_utc + timedelta(days=2)
    expiries = [d for d in data.options if datetime.strptime(d, "%Y-%m-%d").date() >= min_date]
    if not expiries:
        st.error("오늘로부터 2일 이후 만기 옵션이 없습니다.")
        st.stop()
    expiry = st.selectbox("Select expiry", expiries)
    target_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    start_date = target_date - timedelta(days=2)
    chain = data.option_chain(expiry)
    calls = chain.calls
    puts = chain.puts
    st.write(f"Call options: {len(calls)}, Put options: {len(puts)}")
    strikes = sorted([float(s) for s in set(calls['strike']).union(set(puts['strike']))])
    st.write(f"Current price: {price:.2f}, Available strikes: {len(strikes)}")
    if strikes:
        closest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - float(price)))
        st.write(f"Default strike: {strikes[closest_idx]:.2f} (index: {closest_idx})")
    else:
        closest_idx = 0
    strike = st.selectbox("Select strike", strikes, index=closest_idx)
    call_premium = calls[calls['strike'] == strike]['lastPrice'].values
    put_premium = puts[puts['strike'] == strike]['lastPrice'].values
    call_premium_val = 2.0
    if len(call_premium) > 0 and call_premium[0] is not None and not pd.isna(call_premium[0]):
        call_premium_val = float(call_premium[0])
    put_premium_val = 1.5
    if len(put_premium) > 0 and put_premium[0] is not None and not pd.isna(put_premium[0]):
        put_premium_val = float(put_premium[0])
    st.write(f"Call premium: {call_premium_val}, Put premium: {put_premium_val}")

    st.subheader("Option Chain Premium/IV/Delta Visualization")
    call_delta = calls['delta'] if 'delta' in calls.columns else np.nan
    put_delta = puts['delta'] if 'delta' in puts.columns else np.nan
    chain_df = pd.DataFrame({
        'Strike': strikes,
        'Call Premium': [float(calls[calls['strike'] == s]['lastPrice'].values[0]) if len(calls[calls['strike'] == s]['lastPrice'].values) > 0 else np.nan for s in strikes],
        'Put Premium': [float(puts[puts['strike'] == s]['lastPrice'].values[0]) if len(puts[puts['strike'] == s]['lastPrice'].values) > 0 else np.nan for s in strikes],
        'Call IV': [float(calls[calls['strike'] == s]['impliedVolatility'].values[0]) if 'impliedVolatility' in calls.columns and len(calls[calls['strike'] == s]['impliedVolatility'].values) > 0 else np.nan for s in strikes],
        'Put IV': [float(puts[puts['strike'] == s]['impliedVolatility'].values[0]) if 'impliedVolatility' in puts.columns and len(puts[puts['strike'] == s]['impliedVolatility'].values) > 0 else np.nan for s in strikes],
        'Call Delta': [float(calls[calls['strike'] == s]['delta'].values[0]) if 'delta' in calls.columns and len(calls[calls['strike'] == s]['delta'].values) > 0 else np.nan for s in strikes],
        'Put Delta': [float(puts[puts['strike'] == s]['delta'].values[0]) if 'delta' in puts.columns and len(puts[puts['strike'] == s]['delta'].values) > 0 else np.nan for s in strikes],
    })
    st.dataframe(chain_df)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chain_df['Strike'], chain_df['Call Premium'], label='Call Premium', marker='o')
    ax.plot(chain_df['Strike'], chain_df['Put Premium'], label='Put Premium', marker='o')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Premium')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chain_df['Strike'], chain_df['Call IV'], label='Call IV', marker='o')
    ax.plot(chain_df['Strike'], chain_df['Put IV'], label='Put IV', marker='o')
    ax.set_xlabel('Strike')
    ax.set_ylabel('IV')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    if 'delta' in calls.columns or 'delta' in puts.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(chain_df['Strike'], chain_df['Call Delta'], label='Call Delta', marker='o')
        ax.plot(chain_df['Strike'], chain_df['Put Delta'], label='Put Delta', marker='o')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Delta')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    vix_vol = 0.2
    try:
        vix = yf.Ticker("^VIX")
        vix_price = vix.history(period='1d')['Close'].iloc[-1]
        st.subheader(f"S&P500 Volatility Index (VIX): {vix_price:.2f}")
        vix_vol = float(vix_price) / 100
    except Exception:
        st.info("VIX data could not be loaded.")
        vix_vol = 0.2

    strategies = ["long_call", "vertical_call_spread"]
    results = []
    col1, col2 = st.columns(2)
    with col1:
        target_pct = st.number_input("목표수익률 (%)", min_value=1.0, max_value=50.0, value=5.0, step=0.5)
    with col2:
        cost_pct = st.number_input("거래비용(수수료+슬리피지, %)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)

    st.info(f"승률 산출 기준: 프리미엄(투자금) 대비 순수익이 {target_pct}% + 거래비용 {cost_pct}% = 총 {target_pct+cost_pct}% 이상일 때만 '승'으로 간주합니다.")

    for strategy in strategies:
        input_data = {
            "stock_price": float(price),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "volatility": vix_vol,
            "interest_rate": 0.01,
            "min_stock": float(price) * 0.8,
            "max_stock": float(price) * 1.2,
        }
        if strategy == "long_call":
            input_data["strategy"] = [{
                "type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "buy"
            }]
            invest_base = call_premium_val
        elif strategy == "vertical_call_spread":
            next_strike = float(strike) + 5
            next_call_premium = calls[calls['strike'] == next_strike]['lastPrice'].values
            next_call_premium_val = float(next_call_premium[0]) if len(next_call_premium) > 0 else 1.0
            input_data["strategy"] = [
                {"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "buy"},
                {"type": "call", "strike": next_strike, "premium": next_call_premium_val, "n": 1, "action": "sell"},
            ]
            invest_base = call_premium_val - next_call_premium_val
        profit_target = invest_base * (target_pct + cost_pct) / 100
        input_data["profit_target"] = profit_target
        out = run_strategy(input_data)
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        results.append({
            "Strategy": strategy,
            "Win Rate": safe_float(out["probability_of_profit"]) if isinstance(out, dict) and "probability_of_profit" in out else safe_float(getattr(out, "probability_of_profit", None)),
            "Maximum Return": safe_float(out["maximum_return"]) if isinstance(out, dict) and "maximum_return" in out else safe_float(getattr(out, "maximum_return_in_the_domain", None)),
            "Maximum Loss": safe_float(out["minimum_return"]) if isinstance(out, dict) and "minimum_return" in out else safe_float(getattr(out, "minimum_return_in_the_domain", None)),
            "Expected Profit": safe_float(out["expected_profit"]) if isinstance(out, dict) and "expected_profit" in out else safe_float(getattr(out, "expected_profit", None)),
            "Expected Loss": safe_float(out["expected_loss"]) if isinstance(out, dict) and "expected_loss" in out else safe_float(getattr(out, "expected_loss", None)),
        })
    df = pd.DataFrame(results)
    st.subheader("Strategy Simulation Results")
    best_row = df.loc[df['Win Rate'].idxmax()]
    st.markdown(f"### 🏆 Recommended Strategy: **{best_row['Strategy']}** (Win Rate: {best_row['Win Rate']:.2%}, Expected Profit: {best_row['Expected Profit']:.2f})")
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #ffe082' if v else '' for v in is_max]
    styled_df = df.style.apply(highlight_max, subset=['Win Rate', 'Expected Profit', 'Maximum Return'], axis=0)
    st.dataframe(styled_df, use_container_width=True)
    st.subheader("Strategy Payoff Curves")
    x_prices = np.linspace(float(price) * 0.8, float(price) * 1.2, 100)
    for strategy in strategies:
        y_payoff = np.zeros_like(x_prices)
        if strategy == "long_call":
            y_payoff = np.maximum(0, x_prices - float(strike)) - call_premium_val
        elif strategy == "vertical_call_spread":
            next_strike = float(strike) + 5
            next_call_premium_vals = calls[calls['strike'] == next_strike]['lastPrice'].values
            next_call_premium_val = 1.0
            if len(next_call_premium_vals) > 0 and next_call_premium_vals[0] is not None and not pd.isna(next_call_premium_vals[0]):
                next_call_premium_val = float(next_call_premium_vals[0])
            long_call_payoff = np.maximum(0, x_prices - float(strike)) - call_premium_val
            short_call_payoff = -(np.maximum(0, x_prices - next_strike) - next_call_premium_val)
            y_payoff = long_call_payoff + short_call_payoff
        fig, ax = plt.subplots()
        ax.plot(x_prices, y_payoff, label=strategy)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(float(price), color='gray', linestyle='--', label='Current Price')
        ax.set_title(f"{strategy} Payoff Curve")
        ax.set_xlabel("Underlying Price")
        ax.set_ylabel("Profit/Loss")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
except Exception as e:
    st.error(f"Data collection or evaluation error occurred: {e}") 