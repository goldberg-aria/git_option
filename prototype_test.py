import pandas as pd
from datetime import datetime, timedelta
from optionlab import run_strategy
import seaborn as sns
import matplotlib.pyplot as plt

# 변수 목록 확장
stock_prices = [90, 100, 110]
vols = [0.15, 0.2, 0.3]
expiries = [7, 30, 90]  # days to expiry
strikes = [95, 100, 105]  # ITM, ATM, OTM
strategies = ["long_call", "short_put", "vertical_call_spread", "covered_call"]

results = []

for stock_price in stock_prices:
    for vol in vols:
        for expiry_days in expiries:
            for strike in strikes:
                for strat in strategies:
                    start_date = datetime(2024, 7, 1)
                    target_date = start_date + timedelta(days=expiry_days)
                    input_data = {
                        "stock_price": stock_price,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "target_date": target_date.strftime("%Y-%m-%d"),
                        "volatility": vol,
                        "interest_rate": 0.01,
                        "min_stock": stock_price * 0.8,
                        "max_stock": stock_price * 1.2,
                    }
                    # 전략별 입력 생성
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
                    # 평가
                    out = run_strategy(input_data)
                    results.append({
                        "stock_price": stock_price,
                        "vol": vol,
                        "expiry_days": expiry_days,
                        "strike": strike,
                        "strategy": strat,
                        "pop": out.probability_of_profit,
                        "max_pl": out.maximum_return_in_the_domain,
                        "min_pl": out.minimum_return_in_the_domain,
                        "expected_profit": out.expected_profit,
                        "expected_loss": out.expected_loss,
                    })

# 결과 DataFrame 저장
if results:
    df = pd.DataFrame(results)
    print(df.head())

    # 1. 전략별 평균 승률 히트맵 (변동성-전략)
    pivot = df.pivot_table(index='strategy', columns='vol', values='pop', aggfunc='mean')
    sns.heatmap(pivot, annot=True, cmap='YlGnBu')
    plt.title('전략별 변동성 구간 평균 승률')
    plt.show()

    # 2. 만기별, 행사가별 승률 히트맵 (예시)
    pivot2 = df.pivot_table(index='expiry_days', columns='strike', values='pop', aggfunc='mean')
    sns.heatmap(pivot2, annot=True, cmap='YlOrRd')
    plt.title('만기-행사가별 평균 승률')
    plt.show()

    # 3. 조건별 최적 전략 추천 (승률 기준)
    group_cols = ['stock_price', 'vol', 'expiry_days', 'strike']
    idx = df.groupby(group_cols)['pop'].idxmax()
    best_strategies = df.loc[idx, group_cols + ['strategy', 'pop']]
    print("\n조건별 승률 최적 전략:")
    print(best_strategies.reset_index(drop=True))
else:
    print("No results.") 