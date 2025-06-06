import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta, timezone
from optionlab import run_strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # streamlit 호환 백엔드 강제

st.set_page_config(page_title="실제 옵션 체인 기반 전략 추천 (5_Dev_App_Table_Recommend)", layout="centered")
st.title("실제 옵션 체인 기반 전략 추천 (5_Dev_App_Table_Recommend)")

ticker = st.text_input("티커를 입력하세요 (예: AAPL, TSLA, SPY)", value="AAPL")

try:
    data = yf.Ticker(ticker)
    price = data.history(period='1d')['Close'].iloc[-1]
    st.subheader(f"기초자산({ticker.upper()}) 현재가: ${price:.2f}")
    # 오늘로부터 최소 2일 이후 만기일만 허용
    today_utc = datetime.now(timezone.utc).date()
    min_date = today_utc + timedelta(days=2)
    expiries = [d for d in data.options if datetime.strptime(d, "%Y-%m-%d").date() >= min_date]
    if not expiries:
        st.error("오늘로부터 2일 이후 만기 옵션이 없습니다.")
        st.stop()
    expiry = st.selectbox("옵션 만기 선택", expiries)
    target_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    # start_date를 target_date보다 확실히 2일 이전으로 설정
    start_date = target_date - timedelta(days=2)
    chain = data.option_chain(expiry)
    calls = chain.calls
    puts = chain.puts
    st.write(f"콜옵션 {len(calls)}개, 풋옵션 {len(puts)}개")
    strikes = sorted([float(s) for s in set(calls['strike']).union(set(puts['strike']))])
    st.write(f"현재가: {price:.2f}, 사용 가능한 행사가 수: {len(strikes)}")
    if strikes:
        closest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - float(price)))
        st.write(f"선택된 기본 행사가: {strikes[closest_idx]:.2f} (인덱스: {closest_idx})")
    else:
        closest_idx = 0
    strike = st.selectbox("행사가 선택", strikes, index=closest_idx)
    call_premium = calls[calls['strike'] == strike]['lastPrice'].values
    put_premium = puts[puts['strike'] == strike]['lastPrice'].values
    # 프리미엄 안전 처리 (None, NaN 포함)
    call_premium_val = 2.0
    if len(call_premium) > 0 and call_premium[0] is not None and not pd.isna(call_premium[0]):
        call_premium_val = float(call_premium[0])
    put_premium_val = 1.5
    if len(put_premium) > 0 and put_premium[0] is not None and not pd.isna(put_premium[0]):
        put_premium_val = float(put_premium[0])
    st.write(f"콜옵션 프리미엄: {call_premium_val}, 풋옵션 프리미엄: {put_premium_val}")

    # 4번 기능: 옵션 체인 프리미엄/IV/델타 등 표 및 시각화
    st.subheader("옵션 체인 프리미엄/IV/델타 시각화")
    # 델타가 있으면 포함, 없으면 NaN
    call_delta = calls['delta'] if 'delta' in calls.columns else np.nan
    put_delta = puts['delta'] if 'delta' in puts.columns else np.nan
    chain_df = pd.DataFrame({
        '행사가': strikes,
        '콜 프리미엄': [float(calls[calls['strike'] == s]['lastPrice'].values[0]) if len(calls[calls['strike'] == s]['lastPrice'].values) > 0 else np.nan for s in strikes],
        '풋 프리미엄': [float(puts[puts['strike'] == s]['lastPrice'].values[0]) if len(puts[puts['strike'] == s]['lastPrice'].values) > 0 else np.nan for s in strikes],
        '콜 IV': [float(calls[calls['strike'] == s]['impliedVolatility'].values[0]) if 'impliedVolatility' in calls.columns and len(calls[calls['strike'] == s]['impliedVolatility'].values) > 0 else np.nan for s in strikes],
        '풋 IV': [float(puts[puts['strike'] == s]['impliedVolatility'].values[0]) if 'impliedVolatility' in puts.columns and len(puts[puts['strike'] == s]['impliedVolatility'].values) > 0 else np.nan for s in strikes],
        '콜 델타': [float(calls[calls['strike'] == s]['delta'].values[0]) if 'delta' in calls.columns and len(calls[calls['strike'] == s]['delta'].values) > 0 else np.nan for s in strikes],
        '풋 델타': [float(puts[puts['strike'] == s]['delta'].values[0]) if 'delta' in puts.columns and len(puts[puts['strike'] == s]['delta'].values) > 0 else np.nan for s in strikes],
    })
    st.dataframe(chain_df)
    # 프리미엄/IV/델타 시각화
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chain_df['행사가'], chain_df['콜 프리미엄'], label='콜 프리미엄', marker='o')
    ax.plot(chain_df['행사가'], chain_df['풋 프리미엄'], label='풋 프리미엄', marker='o')
    ax.set_xlabel('행사가')
    ax.set_ylabel('프리미엄')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chain_df['행사가'], chain_df['콜 IV'], label='콜 IV', marker='o')
    ax.plot(chain_df['행사가'], chain_df['풋 IV'], label='풋 IV', marker='o')
    ax.set_xlabel('행사가')
    ax.set_ylabel('IV')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    if 'delta' in calls.columns or 'delta' in puts.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(chain_df['행사가'], chain_df['콜 델타'], label='콜 델타', marker='o')
        ax.plot(chain_df['행사가'], chain_df['풋 델타'], label='풋 델타', marker='o')
        ax.set_xlabel('행사가')
        ax.set_ylabel('델타')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # 2번 기능: S&P500 VIX(변동성 지수) 표시 및 시뮬레이션에 반영
    vix_vol = 0.2
    try:
        vix = yf.Ticker("^VIX")
        vix_price = vix.history(period='1d')['Close'].iloc[-1]
        st.subheader(f"S&P500 변동성 지수(VIX): {vix_price:.2f}")
        vix_vol = float(vix_price) / 100
    except Exception:
        st.info("VIX 데이터를 불러올 수 없습니다.")
        vix_vol = 0.2

    # 전략별 시뮬레이션 결과 자동 출력
    strategies = ["long_call", "short_put", "vertical_call_spread", "covered_call"]
    results = []
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
        elif strategy == "short_put":
            input_data["strategy"] = [{
                "type": "put", "strike": float(strike), "premium": put_premium_val, "n": 1, "action": "sell"
            }]
        elif strategy == "vertical_call_spread":
            next_strike = float(strike) + 5
            next_call_premium = calls[calls['strike'] == next_strike]['lastPrice'].values
            next_call_premium_val = float(next_call_premium[0]) if len(next_call_premium) > 0 else 1.0
            input_data["strategy"] = [
                {"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "buy"},
                {"type": "call", "strike": next_strike, "premium": next_call_premium_val, "n": 1, "action": "sell"},
            ]
        elif strategy == "covered_call":
            input_data["strategy"] = [
                {"type": "stock", "n": 1, "action": "buy"},
                {"type": "call", "strike": float(strike), "premium": call_premium_val, "n": 1, "action": "sell"},
            ]
        out = run_strategy(input_data)
        
        # run_strategy 결과 안전 처리
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        
        results.append({
            "전략": strategy,
            "승률": safe_float(out["probability_of_profit"]) if isinstance(out, dict) and "probability_of_profit" in out else safe_float(getattr(out, "probability_of_profit", None)),
            "최대수익": safe_float(out["maximum_return"]) if isinstance(out, dict) and "maximum_return" in out else safe_float(getattr(out, "maximum_return_in_the_domain", None)),
            "최대손실": safe_float(out["minimum_return"]) if isinstance(out, dict) and "minimum_return" in out else safe_float(getattr(out, "minimum_return_in_the_domain", None)),
            "기대수익": safe_float(out["expected_profit"]) if isinstance(out, dict) and "expected_profit" in out else safe_float(getattr(out, "expected_profit", None)),
            "기대손실": safe_float(out["expected_loss"]) if isinstance(out, dict) and "expected_loss" in out else safe_float(getattr(out, "expected_loss", None)),
        })
    df = pd.DataFrame(results)
    # 5번 기능: 전략별 승률/기대수익/최대손실 등 지표 하이라이트 및 자동 추천
    st.subheader("전략별 시뮬레이션 결과")
    # 승률이 가장 높은 전략 추천
    best_row = df.loc[df['승률'].idxmax()]
    st.markdown(f"### 🏆 추천 전략: **{best_row['전략']}** (승률: {best_row['승률']:.2%}, 기대수익: {best_row['기대수익']:.2f})")
    # 주요 지표 하이라이트(스타일링)
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #ffe082' if v else '' for v in is_max]
    styled_df = df.style.apply(highlight_max, subset=['승률', '기대수익', '최대수익'], axis=0)
    st.dataframe(styled_df, use_container_width=True)

    # 1번: 전략별 손익곡선 시각화 (직접 계산)
    st.subheader("전략별 손익곡선(수익/손실) 그래프")

    # 손익계산을 위한 주가 범위 생성
    x_prices = np.linspace(float(price) * 0.8, float(price) * 1.2, 100)

    for strategy in strategies:
        y_payoff = np.zeros_like(x_prices)
    
        if strategy == "long_call":
            y_payoff = np.maximum(0, x_prices - float(strike)) - call_premium_val

        elif strategy == "short_put":
            y_payoff = put_premium_val - np.maximum(0, float(strike) - x_prices)
        
        elif strategy == "vertical_call_spread":
            next_strike = float(strike) + 5
            next_call_premium_vals = calls[calls['strike'] == next_strike]['lastPrice'].values
            next_call_premium_val = 1.0
            if len(next_call_premium_vals) > 0 and next_call_premium_vals[0] is not None and not pd.isna(next_call_premium_vals[0]):
                next_call_premium_val = float(next_call_premium_vals[0])
            long_call_payoff = np.maximum(0, x_prices - float(strike)) - call_premium_val
            short_call_payoff = -(np.maximum(0, x_prices - next_strike) - next_call_premium_val)
            y_payoff = long_call_payoff + short_call_payoff

        elif strategy == "covered_call":
            stock_profit = x_prices - float(price)
            short_call_payoff = -(np.maximum(0, x_prices - float(strike)) - call_premium_val)
            y_payoff = stock_profit + short_call_payoff

        fig, ax = plt.subplots()
        ax.plot(x_prices, y_payoff, label=strategy)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(float(price), color='gray', linestyle='--', label='현재가')
        ax.set_title(f"{strategy} 손익곡선")
        ax.set_xlabel("기초자산 가격")
        ax.set_ylabel("수익/손실")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)  # 리소스 정리

except Exception as e:
    st.error(f"데이터 수집 또는 평가 중 오류 발생: {e}") 