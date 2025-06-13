import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.title('Black-Scholes 공식 vs 유한차분법(FDM) 하이브리드 옵션 가격 계산')

st.markdown('''
### ✨ 목적
- **Black-Scholes 공식(해석해)**와 **유한차분법(FDM, Explicit)** 기반 수치해를 비교/혼합하는 하이브리드 옵션 가격 계산 예제입니다.
- 유럽형 콜옵션을 예시로, 두 방식의 결과와 차이를 시각적으로 확인할 수 있습니다.
- (향후 배리어, 아메리칸, 비정상성 등 확장 가능)
''')

# 입력 파라미터
col1, col2 = st.columns(2)
with col1:
    S = st.number_input('기초자산 현재가 S', value=100.0)
    K = st.number_input('행사가 K', value=100.0)
    T = st.number_input('만기(년) T', value=1.0)
    r = st.number_input('무위험이자율 r', value=0.05)
    sigma = st.number_input('변동성 sigma', value=0.2)
with col2:
    N = st.slider('자산가격 그리드 수 N', 20, 200, 100)
    M = st.slider('시간 스텝 수 M', 20, 500, 100)
    Smax = st.number_input('자산가격 최대값 Smax', value=3.0 * S)

st.markdown('---')

# Black-Scholes 공식(해석해)
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

bs_price = bs_call_price(S, K, T, r, sigma)
st.info(f'Black-Scholes 공식 콜옵션 가격: **{bs_price:.4f}**')

# FDM(Explicit) 구현
def fdm_explicit_european_call(S, K, T, r, sigma, Smax, N, M):
    dS = Smax / N
    dt = T / M
    grid = np.zeros((N+1, M+1))
    stock = np.linspace(0, Smax, N+1)
    # 만기 페이오프
    grid[:, -1] = np.maximum(stock - K, 0)
    # 경계조건
    grid[0, :] = 0
    grid[-1, :] = Smax - K * np.exp(-r * dt * np.arange(M+1)[::-1])
    # 안정성 조건 체크
    alpha = dt * sigma**2 * (np.arange(N+1))**2 / 2 / dS**2
    if dt > 0.9 * dS**2 / (sigma**2 * N**2):
        st.warning('안정성 조건 위반: dt가 너무 큽니다. M을 늘리거나 N을 줄이세요.')
    # 역순(time-marching)
    for j in reversed(range(M)):
        for i in range(1, N):
            a = 0.5 * dt * (sigma**2 * (i**2) - r * i)
            b = 1 - dt * (sigma**2 * (i**2) + r)
            c = 0.5 * dt * (sigma**2 * (i**2) + r * i)
            grid[i, j] = a * grid[i-1, j+1] + b * grid[i, j+1] + c * grid[i+1, j+1]
    # 현재가에 가장 가까운 grid 값 반환
    idx = int(S / dS)
    return grid, stock, grid[idx, 0]

fdm_grid, stock_grid, fdm_price = fdm_explicit_european_call(S, K, T, r, sigma, Smax, N, M)
st.success(f'FDM(Explicit) 콜옵션 가격: **{fdm_price:.4f}**')

st.markdown('---')

# 결과 시각화
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# 1. 만기 페이오프와 t=0 가격
ax[0].plot(stock_grid, fdm_grid[:, 0], label='FDM t=0')
ax[0].plot(stock_grid, np.maximum(stock_grid - K, 0), '--', label='만기 페이오프')
ax[0].axvline(S, color='r', ls=':', label='현재가')
ax[0].set_title('FDM t=0 가격 vs 만기 페이오프')
ax[0].set_xlabel('자산가격')
ax[0].set_ylabel('옵션가격')
ax[0].legend()
# 2. FDM vs BS 가격 비교
ax[1].plot(stock_grid, fdm_grid[:, 0], label='FDM t=0')
bs_curve = [bs_call_price(s, K, T, r, sigma) for s in stock_grid]
ax[1].plot(stock_grid, bs_curve, '--', label='BS 공식')
ax[1].axvline(S, color='r', ls=':', label='현재가')
ax[1].set_title('FDM vs Black-Scholes 공식')
ax[1].set_xlabel('자산가격')
ax[1].set_ylabel('옵션가격')
ax[1].legend()
st.pyplot(fig)

st.markdown('''
#### 💡 해설
- **FDM(Explicit)** 방식은 grid 수(N, M)에 따라 정확도가 달라집니다. (N, M을 충분히 키우면 BS 공식과 거의 일치)
- **안정성 조건**을 반드시 확인하세요. (dt < dS² / (sigma² N²))
- 이 구조를 바탕으로 배리어, 아메리칸, 비정상성 등 다양한 옵션으로 확장 가능합니다.
''') 