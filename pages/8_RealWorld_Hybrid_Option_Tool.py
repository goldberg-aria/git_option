import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from optionlab import run_strategy, create_price_array, BlackScholesModelInputs

st.set_page_config(page_title="실전 하이브리드 옵션 전략 도구 (8_RealWorld_Hybrid_Option_Tool)", layout="centered")
st.title("실전 하이브리드 옵션 전략 도구 (Hybrid Option Tool)")

st.markdown('''
- 실시간 옵션체인/프리미엄/IV/델타 불러오기
- 전략별 시뮬레이션: Black-Scholes 공식, MC array, FDM(향후) 중 선택
- 승률, 기대수익, 최대손실, payoff 곡선 등 실전 지표 제공
- (롱콜/콜스프레드/커버드콜 우선, 구조는 확장 가능)
''')

# 1. 입력
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("티커 입력 (예: AAPL, TSLA, SPY)", value="AAPL")
    sim_method = st.selectbox("시뮬레이션 방식", ["Black-Scholes 공식", "MC array"], index=0)
    target_pct = st.number_input("목표수익률(%)", min_value=1.0, max_value=50.0, value=5.0, step=0.5)
    cost_pct = st.number_input("거래비용(%)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
with col2:
    strategy = st.selectbox("전략 선택", ["롱콜", "콜스프레드", "커버드콜"], index=0)

try:
    data = yf.Ticker(ticker)
    price = data.history(period='1d')['Close'].iloc[-1]
    st.subheader(f"기초자산({ticker.upper()}) 현재가: ${price:.2f}")
    today_utc = datetime.now(timezone.utc).date()
    min_date = today_utc + timedelta(days=2)
    expiries = [d for d in data.options if datetime.strptime(d, "%Y-%m-%d").date() >= min_date]
    if not expiries:
        st.error("오늘로부터 2일 이후 만기 옵션이 없습니다.")
        st.stop()
    expiry = st.selectbox("만기 선택", expiries)
    target_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    start_date = target_date - timedelta(days=2)
    chain = data.option_chain(expiry)
    calls = chain.calls
    strikes = sorted([float(s) for s in set(calls['strike'])])
    closest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - float(price))) if strikes else 0
    strike = st.selectbox("행사가 선택", strikes, index=closest_idx)
    call_premium = calls[calls['strike'] == strike]['lastPrice'].values
    call_premium_val = float(call_premium[0]) if len(call_premium) > 0 else 2.0
    next_strike = float(strike) + 5
    next_call_premium = calls[calls['strike'] == next_strike]['lastPrice'].values
    next_call_premium_val = float(next_call_premium[0]) if len(next_call_premium) > 0 else 1.0
    iv = float(calls[calls['strike'] == strike]['impliedVolatility'].values[0]) if 'impliedVolatility' in calls.columns and len(calls[calls['strike'] == strike]['impliedVolatility'].values) > 0 else 0.3
except Exception as e:
    st.error(f"옵션체인 데이터 로딩 실패: {e}")
    st.stop()

st.markdown('---')

# 2. 전략별 입력/시뮬레이션 파라미터 구성
def get_input_data(strategy, sim_method):
    base = {
        "stock_price": float(price),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "volatility": iv,
        "interest_rate": 0.01,
        "min_stock": float(price) * 0.8,
        "max_stock": float(price) * 1.2,
    }
    if strategy == "롱콜":
        base["strategy"] = [{"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "buy"}]
        invest_base = call_premium_val
    elif strategy == "콜스프레드":
        base["strategy"] = [
            {"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "buy"},
            {"type": "call", "strike": next_strike, "premium": next_call_premium_val, "n": 1, "action": "sell"},
        ]
        invest_base = call_premium_val - next_call_premium_val
    elif strategy == "커버드콜":
        base["strategy"] = [
            {"type": "stock", "n": 1, "action": "buy"},
            {"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "sell"},
        ]
        invest_base = float(price)
    profit_target = invest_base * (target_pct + cost_pct) / 100
    base["profit_target"] = profit_target
    return base, invest_base

input_data, invest_base = get_input_data(strategy, sim_method)

# 3. 시뮬레이션 실행
def run_simulation(input_data, sim_method):
    if sim_method == "Black-Scholes 공식":
        out = run_strategy(input_data)
    elif sim_method == "MC array":
        arr = create_price_array(
            BlackScholesModelInputs(
                stock_price=input_data["stock_price"],
                volatility=input_data["volatility"],
                interest_rate=input_data["interest_rate"],
                years_to_target_date=(datetime.strptime(input_data["target_date"], "%Y-%m-%d") - datetime.strptime(input_data["start_date"], "%Y-%m-%d")).days / 365,
            ),
            n=100_000,
            seed=0,
        )
        input_data_arr = input_data.copy()
        input_data_arr["model"] = "array"
        input_data_arr["array"] = arr
        out = run_strategy(input_data_arr)
    else:
        out = None
    return out

out = run_simulation(input_data, sim_method)

# 4. 결과 요약/시각화
def safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

st.subheader("시뮬레이션 결과 요약")
if out is not None:
    st.write(f"**승률:** {safe_float(getattr(out, 'probability_of_profit', None)):.2%}")
    st.write(f"**기대수익:** {safe_float(getattr(out, 'expected_profit', None)):.2f}")
    st.write(f"**기대손실:** {safe_float(getattr(out, 'expected_loss', None)):.2f}")
    st.write(f"**최대수익:** {safe_float(getattr(out, 'maximum_return_in_the_domain', None)):.2f}")
    st.write(f"**최대손실:** {safe_float(getattr(out, 'minimum_return_in_the_domain', None)):.2f}")
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
    # 분포 시각화 (MC array일 때)
    if sim_method == "MC array":
        st.subheader("만기 주가 분포 (MC array)")
        arr = input_data["array"] if "array" in input_data else None
        if arr is not None:
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.hist(arr, bins=100, color='skyblue', edgecolor='k', alpha=0.7)
            ax2.set_xlabel('만기 주가')
            ax2.set_ylabel('빈도')
            st.pyplot(fig2)
else:
    st.warning("시뮬레이션 결과가 없습니다.")

st.markdown('''
#### 💡 실전 팁
- 시뮬레이션 방식(공식/MC array)과 전략을 바꿔가며 결과를 비교해보세요.
- 거래비용, 목표수익률, 변동성(IV) 등 실전 파라미터를 조정해 다양한 시나리오를 실험할 수 있습니다.
- 향후 배리어, 아메리칸, FDM 등 고급 기능도 추가 예정입니다.
''') 