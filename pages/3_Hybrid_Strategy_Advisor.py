import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from optionlab import run_strategy

st.set_page_config(page_title="하이브리드 옵션 전략 추천 (3_Hybrid_Strategy_Advisor)", layout="centered")
st.title("하이브리드 옵션 전략 추천 (Hybrid Strategy Advisor)")

st.markdown('''
- **자동 추천**: 종목만 입력하면 만기/행사가/전략 모든 조합을 계산, 최적 전략을 자동 추천합니다.
- **예측 기반 추천**: 시장 예측(상승/하락/횡보 등)과 선호 만기/행사가/전략을 입력하면, 그에 맞는 전략을 추천합니다.
''')

mode = st.radio("추천 모드 선택", ["자동 추천", "예측 기반 추천"], index=0)

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("티커 입력 (예: AAPL, TSLA, SPY)", value="TSLA")
with col2:
    today_utc = datetime.now(timezone.utc).date()
    min_date = today_utc + timedelta(days=2)

try:
    data = yf.Ticker(ticker)
    price = data.history(period='1d')['Close'].iloc[-1]
    st.subheader(f"기초자산({ticker.upper()}) 현재가: ${price:.2f}")
    expiries = [d for d in data.options if datetime.strptime(d, "%Y-%m-%d").date() >= min_date]
    if not expiries:
        st.error("오늘로부터 2일 이후 만기 옵션이 없습니다.")
        st.stop()
    chain = data.option_chain(expiries[0])
    calls = chain.calls
    puts = chain.puts
    strikes = sorted([float(s) for s in set(calls['strike']).union(set(puts['strike']))])
    # 현재가 ±20% 범위로 제한
    strikes = [s for s in strikes if price * 0.8 <= s <= price * 1.2]
except Exception as e:
    st.error(f"옵션체인 데이터 로딩 실패: {e}")
    st.stop()

if mode == "자동 추천":
    st.markdown('''
    #### ✅ 자동 추천: 종목만 입력하면 최적 전략을 자동 추천합니다.
    - 대표 전략(롱콜, 숏풋, 콜스프레드, 커버드콜)과 주요 만기/행사가 조합을 계산합니다.
    - 승률, 기대수익, 최대손실 등으로 순위화하여 추천합니다.
    ''')
    expiry = st.selectbox("만기 선택", expiries)
    target_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    start_date = target_date - timedelta(days=2)
    # 대표 전략/행사가 조합 생성
    strategies = ["롱콜", "숏풋", "콜스프레드", "커버드콜"]
    results = []
    for strike in strikes:
        call_premium = calls[calls['strike'] == strike]['lastPrice'].values
        call_premium_val = float(call_premium[0]) if len(call_premium) > 0 else 2.0
        put_premium = puts[puts['strike'] == strike]['lastPrice'].values
        put_premium_val = float(put_premium[0]) if len(put_premium) > 0 else 1.5
        iv = float(calls[calls['strike'] == strike]['impliedVolatility'].values[0]) if 'impliedVolatility' in calls.columns and len(calls[calls['strike'] == strike]['impliedVolatility'].values) > 0 else 0.3
        for strategy in strategies:
            input_data = {
                "stock_price": float(price),
                "start_date": start_date.strftime("%Y-%m-%d"),
                "target_date": target_date.strftime("%Y-%m-%d"),
                "volatility": iv,
                "interest_rate": 0.01,
                "min_stock": float(price) * 0.8,
                "max_stock": float(price) * 1.2,
            }
            if strategy == "롱콜":
                input_data["strategy"] = [{"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "buy"}]
                invest_base = call_premium_val * 100
            elif strategy == "숏풋":
                input_data["strategy"] = [{"type": "put", "strike": float(strike), "premium": put_premium_val, "n": 1, "action": "sell"}]
                invest_base = put_premium_val * 100
            elif strategy == "콜스프레드":
                next_strike = float(strike) + 5
                next_call_premium = calls[calls['strike'] == next_strike]['lastPrice'].values
                next_call_premium_val = float(next_call_premium[0]) if len(next_call_premium) > 0 else 1.0
                input_data["strategy"] = [
                    {"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "buy"},
                    {"type": "call", "strike": next_strike, "premium": next_call_premium_val, "n": 1, "action": "sell"},
                ]
                invest_base = (call_premium_val - next_call_premium_val) * 100
            elif strategy == "커버드콜":
                input_data["strategy"] = [
                    {"type": "stock", "n": 1, "action": "buy"},
                    {"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "sell"},
                ]
                invest_base = float(price) * 100 * 0.4  # 예시: 증거금 기준(40%)
            profit_target = invest_base * 0.08  # 예시: 8% (목표수익률+거래비용)
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
                "만기": expiry,
                "행사가": strike,
                "전략": strategy,
                "승률": safe_float(getattr(out, 'probability_of_profit', None)),
                "기대수익": safe_float(getattr(out, 'expected_profit', None)),
                "최대손실": safe_float(getattr(out, 'minimum_return_in_the_domain', None)),
            })
    df = pd.DataFrame(results)
    st.subheader("전략별 시뮬레이션 결과 (자동 추천)")
    st.dataframe(df.sort_values(by=["승률", "기대수익"], ascending=[False, False]).reset_index(drop=True))
    best_row = df.loc[df['승률'].idxmax()]
    st.markdown(f"### 🏆 추천 전략: **{best_row['전략']}** (만기: {best_row['만기']}, 행사가: {best_row['행사가']}, 승률: {best_row['승률']:.2%}, 기대수익: {best_row['기대수익']:.2f})")

elif mode == "예측 기반 추천":
    st.markdown('''
    #### ✅ 예측 기반 추천: 시장 예측과 선호 조건을 입력하면 맞춤 전략을 추천합니다.
    - 상승/하락/횡보 예측, 선호 만기/행사가/전략을 직접 선택할 수 있습니다.
    ''')
    market_view = st.selectbox("시장 예측(본인 뷰)", ["상승", "하락", "횡보"], index=0)
    expiry = st.selectbox("만기 선택", expiries)
    target_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    start_date = target_date - timedelta(days=2)
    strike = st.selectbox("행사가 선택", strikes)
    call_premium = calls[calls['strike'] == strike]['lastPrice'].values
    call_premium_val = float(call_premium[0]) if len(call_premium) > 0 else 2.0
    put_premium = puts[puts['strike'] == strike]['lastPrice'].values
    put_premium_val = float(put_premium[0]) if len(put_premium) > 0 else 1.5
    iv = float(calls[calls['strike'] == strike]['impliedVolatility'].values[0]) if 'impliedVolatility' in calls.columns and len(calls[calls['strike'] == strike]['impliedVolatility'].values) > 0 else 0.3
    # 예측에 따라 추천 전략 후보군 제시
    if market_view == "상승":
        candidate_strategies = ["롱콜", "콜스프레드", "커버드콜"]
    elif market_view == "하락":
        candidate_strategies = ["숏풋"]
    else:
        candidate_strategies = ["커버드콜"]
    strategy = st.selectbox("전략 선택(추천 후보)", candidate_strategies)
    input_data = {
        "stock_price": float(price),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "volatility": iv,
        "interest_rate": 0.01,
        "min_stock": float(price) * 0.8,
        "max_stock": float(price) * 1.2,
    }
    if strategy == "롱콜":
        input_data["strategy"] = [{"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "buy"}]
        invest_base = call_premium_val * 100
    elif strategy == "숏풋":
        input_data["strategy"] = [{"type": "put", "strike": float(strike), "premium": put_premium_val, "n": 1, "action": "sell"}]
        invest_base = put_premium_val * 100
    elif strategy == "콜스프레드":
        next_strike = float(strike) + 5
        next_call_premium = calls[calls['strike'] == next_strike]['lastPrice'].values
        next_call_premium_val = float(next_call_premium[0]) if len(next_call_premium) > 0 else 1.0
        input_data["strategy"] = [
            {"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "buy"},
            {"type": "call", "strike": next_strike, "premium": next_call_premium_val, "n": 1, "action": "sell"},
        ]
        invest_base = (call_premium_val - next_call_premium_val) * 100
    elif strategy == "커버드콜":
        input_data["strategy"] = [
            {"type": "stock", "n": 1, "action": "buy"},
            {"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "sell"},
        ]
        invest_base = float(price) * 100 * 0.4  # 예시: 증거금 기준(40%)
    profit_target = invest_base * 0.08  # 예시: 8% (목표수익률+거래비용)
    input_data["profit_target"] = profit_target
    out = run_strategy(input_data)
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    st.subheader("전략 시뮬레이션 결과")
    st.write(f"**승률:** {safe_float(getattr(out, 'probability_of_profit', None)):.2%}")
    st.write(f"**기대수익:** {safe_float(getattr(out, 'expected_profit', None)):.2f}")
    st.write(f"**최대손실:** {safe_float(getattr(out, 'minimum_return_in_the_domain', None)):.2f}")
    st.write(f"**최대수익:** {safe_float(getattr(out, 'maximum_return_in_the_domain', None)):.2f}")
    # payoff 곡선
    st.subheader("Payoff 곡선 (만기 주가별)")
    x_prices = np.linspace(float(price) * 0.8, float(price) * 1.2, 100)
    y_payoff = []
    for s in x_prices:
        test_input = input_data.copy()
        test_input["stock_price"] = s
        test_input["min_stock"] = s * 0.999
        test_input["max_stock"] = s * 1.001
        y = run_strategy(test_input)
        y_payoff.append(safe_float(getattr(y, 'maximum_return_in_the_domain', None)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_prices, y_payoff, label='Payoff')
    ax.axvline(float(price), color='r', ls=':', label='현재가')
    ax.set_xlabel('만기 주가')
    ax.set_ylabel('만기 손익')
    ax.legend()
    st.pyplot(fig) 