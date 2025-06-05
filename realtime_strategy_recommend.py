import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from optionlab import run_strategy
import pandas as pd

st.set_page_config(page_title="실시간 옵션 전략 추천", layout="centered")
st.title("실제 옵션 체인 기반 전략 추천")

with st.form("input_form"):
    ticker = st.text_input("티커를 입력하세요 (예: AAPL, TSLA, SPY)", value="AAPL")
    submitted = st.form_submit_button("옵션 체인 불러오기")

if submitted:
    try:
        data = yf.Ticker(ticker)
        price = data.history(period='1d')['Close'][-1]
        st.write(f"현재가: {price:.2f}")
        # 1. 옵션 만기 리스트
        expiries = data.options
        expiry = st.selectbox("옵션 만기 선택", expiries)
        chain = data.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts
        st.write(f"콜옵션 {len(calls)}개, 풋옵션 {len(puts)}개")
        # 2. 행사가 리스트
        strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
        strike = st.selectbox("행사가 선택", strikes)
        # 3. 프리미엄(마지막 체결가) 선택
        call_premium = calls[calls['strike'] == strike]['lastPrice'].values
        put_premium = puts[puts['strike'] == strike]['lastPrice'].values
        call_premium = float(call_premium[0]) if len(call_premium) > 0 else None
        put_premium = float(put_premium[0]) if len(put_premium) > 0 else None
        st.write(f"콜옵션 프리미엄: {call_premium}, 풋옵션 프리미엄: {put_premium}")
        # 4. 전략 선택
        strategy = st.selectbox("전략 선택", ["long_call", "short_put", "vertical_call_spread", "covered_call"])
        # 5. 평가 버튼
        eval_btn = st.form_submit_button("전략 평가 실행")
        if eval_btn:
            start_date = datetime.today()
            target_date = datetime.strptime(expiry, "%Y-%m-%d")
            input_data = {
                "stock_price": price,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "target_date": target_date.strftime("%Y-%m-%d"),
                "volatility": 0.2,  # 실제 IV/HV 연동은 2단계에서
                "interest_rate": 0.01,
                "min_stock": price * 0.8,
                "max_stock": price * 1.2,
            }
            if strategy == "long_call":
                input_data["strategy"] = [{
                    "type": "call", "strike": strike, "premium": call_premium or 2, "n": 1, "action": "buy"
                }]
            elif strategy == "short_put":
                input_data["strategy"] = [{
                    "type": "put", "strike": strike, "premium": put_premium or 1.5, "n": 1, "action": "sell"
                }]
            elif strategy == "vertical_call_spread":
                # 스프레드: 선택 행사가 매수, +5 매도(존재할 때만)
                next_strike = strike + 5
                next_call_premium = calls[calls['strike'] == next_strike]['lastPrice'].values
                next_call_premium = float(next_call_premium[0]) if len(next_call_premium) > 0 else 1
                input_data["strategy"] = [
                    {"type": "call", "strike": strike, "premium": call_premium or 2, "n": 1, "action": "buy"},
                    {"type": "call", "strike": next_strike, "premium": next_call_premium, "n": 1, "action": "sell"},
                ]
            elif strategy == "covered_call":
                input_data["strategy"] = [
                    {"type": "stock", "n": 1, "action": "buy"},
                    {"type": "call", "strike": strike, "premium": call_premium or 2, "n": 1, "action": "sell"},
                ]
            out = run_strategy(input_data)
            st.subheader(":star: 전략 평가 결과")
            st.write(f"승률(Probability of Profit): {out.probability_of_profit:.2%}")
            st.write(f"최대수익: {out.maximum_return_in_the_domain}")
            st.write(f"최대손실: {out.minimum_return_in_the_domain}")
            st.write(f"기대수익: {out.expected_profit}")
            st.write(f"기대손실: {out.expected_loss}")
    except Exception as e:
        st.error(f"데이터 수집 또는 평가 중 오류 발생: {e}") 