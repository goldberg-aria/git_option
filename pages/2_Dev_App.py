import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from optionlab import run_strategy
import pandas as pd

st.set_page_config(page_title="실제 옵션 체인 기반 전략 추천", layout="centered")
st.title("실제 옵션 체인 기반 전략 추천 (개발 중)")

ticker = st.text_input("티커를 입력하세요 (예: AAPL, TSLA, SPY)", value="AAPL")

try:
    data = yf.Ticker(ticker)
    price = data.history(period='1d')['Close'].iloc[-1]
    st.subheader(f"기초자산({ticker.upper()}) 현재가: ${price:.2f}")
    # 옵션 만기일 필터링: 오늘 이후 만기만 선택 (오늘은 제외)
    today = datetime.today().date()
    expiries = [d for d in data.options if datetime.strptime(d, "%Y-%m-%d").date() > today]
    if not expiries:
        st.error("오늘 이후 만기 옵션이 없습니다.")
        st.stop()
    expiry = st.selectbox("옵션 만기 선택", expiries)
    chain = data.option_chain(expiry)
    calls = chain.calls
    puts = chain.puts
    st.write(f"콜옵션 {len(calls)}개, 풋옵션 {len(puts)}개")
    strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
    # 현재가에 가장 가까운 행사가 인덱스 찾기 (strikes가 비어있지 않을 때만)
    if strikes:
        closest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - price))
    else:
        closest_idx = 0
    strike = st.selectbox("행사가 선택", strikes, index=closest_idx)
    call_premium = calls[calls['strike'] == strike]['lastPrice'].values
    put_premium = puts[puts['strike'] == strike]['lastPrice'].values
    call_premium = float(call_premium[0]) if len(call_premium) > 0 else None
    put_premium = float(put_premium[0]) if len(put_premium) > 0 else None
    st.write(f"콜옵션 프리미엄: {call_premium}, 풋옵션 프리미엄: {put_premium}")

    # 전략별 시뮬레이션 결과 자동 출력
    strategies = ["long_call", "short_put", "vertical_call_spread", "covered_call"]
    results = []
    for strategy in strategies:
        start_date = datetime.today()
        target_date = datetime.strptime(expiry, "%Y-%m-%d")
        if start_date >= target_date:
            start_date = target_date - timedelta(days=1)
        input_data = {
            "stock_price": float(price),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "volatility": 0.2,
            "interest_rate": 0.01,
            "min_stock": float(price) * 0.8,
            "max_stock": float(price) * 1.2,
        }
        if strategy == "long_call":
            input_data["strategy"] = [{
                "type": "call", "strike": float(strike), "premium": float(call_premium or 2), "n": 1, "action": "buy"
            }]
        elif strategy == "short_put":
            input_data["strategy"] = [{
                "type": "put", "strike": float(strike), "premium": float(put_premium or 1.5), "n": 1, "action": "sell"
            }]
        elif strategy == "vertical_call_spread":
            next_strike = float(strike) + 5
            next_call_premium = calls[calls['strike'] == next_strike]['lastPrice'].values
            next_call_premium = float(next_call_premium[0]) if len(next_call_premium) > 0 else 1
            input_data["strategy"] = [
                {"type": "call", "strike": float(strike), "premium": float(call_premium or 2), "n": 1, "action": "buy"},
                {"type": "call", "strike": next_strike, "premium": next_call_premium, "n": 1, "action": "sell"},
            ]
        elif strategy == "covered_call":
            input_data["strategy"] = [
                {"type": "stock", "n": 1, "action": "buy"},
                {"type": "call", "strike": float(strike), "premium": float(call_premium or 2), "n": 1, "action": "sell"},
            ]
        out = run_strategy(input_data)
        results.append({
            "전략": strategy,
            "승률": float(out.probability_of_profit),
            "최대수익": float(out.maximum_return_in_the_domain),
            "최대손실": float(out.minimum_return_in_the_domain),
            "기대수익": float(out.expected_profit) if out.expected_profit is not None else None,
            "기대손실": float(out.expected_loss) if out.expected_loss is not None else None,
        })
    df = pd.DataFrame(results)
    st.subheader("전략별 시뮬레이션 결과")
    st.dataframe(df)
except Exception as e:
    st.error(f"데이터 수집 또는 평가 중 오류 발생: {e}") 