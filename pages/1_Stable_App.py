import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from optionlab import run_strategy
import pandas as pd

st.set_page_config(page_title="옵션 전략 추천(안정 버전)", layout="centered")
st.title("옵션 전략 추천 (안정 버전)")

with st.form("input_form"):
    ticker = st.text_input("티커를 입력하세요 (예: AAPL, TSLA, SPY)", value="AAPL")
    submitted = st.form_submit_button("전략 추천 실행")

if submitted:
    try:
        data = yf.Ticker(ticker)
        price = data.history(period='1d')['Close'].iloc[-1]
        hist = data.history(period='1mo')['Close']
        vol = hist.pct_change().std() * (252 ** 0.5)  # 연율화

        expiries = [7, 30, 90]
        strikes = [round(price*0.95, 2), round(price, 2), round(price*1.05, 2)]
        strategies = ["long_call", "short_put", "vertical_call_spread", "covered_call"]

        results = []
        for expiry_days in expiries:
            for strike in strikes:
                for strat in strategies:
                    start_date = datetime.today()
                    target_date = start_date + timedelta(days=expiry_days)
                    input_data = {
                        "stock_price": price,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "target_date": target_date.strftime("%Y-%m-%d"),
                        "volatility": vol,
                        "interest_rate": 0.01,
                        "min_stock": price * 0.8,
                        "max_stock": price * 1.2,
                        "profit_target": round(price * 0.05, 2),  # 5% 수익 기준
                    }
                    if strat == "long_call":
                        input_data["strategy"] = [{
                            "type": "call", "strike": strike, "premium": 2, "n": 1, "action": "buy"
                        }]
                    elif strat == "short_put":
                        input_data["strategy"] = [{
                            "type": "put", "strike": strike, "premium": 1.5, "n": 1, "action": "sell"
                        }]
                    elif strat == "vertical_call_spread":
                        input_data["strategy"] = [
                            {"type": "call", "strike": strike, "premium": 2, "n": 1, "action": "buy"},
                            {"type": "call", "strike": strike+5, "premium": 1, "n": 1, "action": "sell"},
                        ]
                    elif strat == "covered_call":
                        input_data["strategy"] = [
                            {"type": "stock", "n": 1, "action": "buy"},
                            {"type": "call", "strike": strike, "premium": 2, "n": 1, "action": "sell"},
                        ]
                    out = run_strategy(input_data)
                    results.append({
                        "만기(일)": expiry_days,
                        "행사가": strike,
                        "전략": strat,
                        "승률(Probability of Profit)": out.probability_of_profit,
                        "기대수익": out.expected_profit,
                    })
        if results:
            df = pd.DataFrame(results)
            best = df.loc[df['승률(Probability of Profit)'].idxmax()]
            st.subheader(":star: 실시간 추천 전략")
            st.table(best)
            st.subheader("전체 전략별 평가 결과")
            st.dataframe(df.sort_values('승률(Probability of Profit)', ascending=False).reset_index(drop=True))
            st.subheader("전략별 승률 Bar Chart")
            st.bar_chart(df.set_index('전략')['승률(Probability of Profit)'])
            
            st.markdown("""
---
#### ⚠️ 승률(Probability of Profit) 산출 기준 안내
- 본 페이지의 승률은 **각 전략이 만기 시점에 5% 이상의 수익을 달성할 확률**을 의미합니다.
- 기존 0.01달러(1센트) 기준은 실제 투자에서 의미가 약해, **실전 투자에서 참고할 만한 5% 수익 기준**으로 산출합니다.
- 이로 인해, 표기된 승률은 기존보다 낮게 나올 수 있습니다.
- 옵션 매수(long) 전략은 특히 5% 이상 수익 확률이 매우 낮을 수 있습니다.
""")
        else:
            st.warning("결과가 없습니다.")
    except Exception as e:
        st.error(f"데이터 수집 또는 평가 중 오류 발생: {e}") 