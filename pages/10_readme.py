import streamlit as st

st.set_page_config(page_title="기능 및 사용법", layout="centered")
st.title("📖 기능 및 사용법")

st.markdown("""
## 프로젝트 개요

이 프로젝트는 실시간 옵션 체인 데이터를 활용하여 다양한 옵션 전략(롱콜, 숏풋, 스프레드, 커버드콜 등)의 수익성, 승률, 기대수익, 손익곡선 등을 자동으로 평가·추천하는 웹앱입니다. Streamlit 기반 멀티페이지 구조로, 실험적 기능과 안정 버전을 분리하여 제공합니다.

---

## 주요 기능
- **실시간 옵션 체인 데이터(yfinance) 기반 전략 시뮬레이션**
- **옵션 만기, 행사가, 프리미엄, IV(내재변동성), 델타 등 자동 추출 및 시각화**
- **S&P500 VIX(시장 변동성 지수) 반영 및 표시**
- **전략별 승률, 기대수익, 최대손실 등 자동 계산 및 추천**
- **전략별 손익곡선(수익/손실) 그래프 직접 계산 및 시각화**
- **각 단계별 실험/확장 기능을 별도 파일로 관리**

---

## 주요 파일 및 역할
| 파일명 | 주요 역할 |
|:---|:---|
| `pages/1_Stable_App.py` | 단순/안정 버전, 임의 프리미엄·IV로 전략별 승률·기대수익 평가 |
| `pages/2_Dev_App.py` | 실시간 옵션 체인 데이터 기반 전략 추천(기본 기능) |
| `pages/3_Dev_App_Experimental.py` | 프리미엄 None/NaN 안전처리, 손익곡선 직접 계산 등 실험적 기능 |
| `pages/4_Dev_App_IV.py` | 행사가별 IV(내재변동성) 테이블 및 시각화, VIX 반영 |
| `pages/5_Dev_App_Table_Recommend.py` | 프리미엄/IV/델타 표·그래프, 전략별 주요지표 하이라이트 및 자동 추천, UX 강화 |

---

## 사용법

1. **환경 준비**
   - Python 3.8 이상 권장
   - 필수 패키지 설치:
     ```bash
     pip install -r requirements.txt
     ```
2. **앱 실행**
   - 프로젝트 루트에서 아래 명령어 실행:
     ```bash
     streamlit run Home.py
     ```
   - 또는 원하는 페이지 직접 실행:
     ```bash
     streamlit run pages/5_Dev_App_Table_Recommend.py
     ```
3. **기능별 페이지 이동**
   - 좌측 사이드바에서 각 기능별 페이지 선택 가능

---

## 문의 및 기여
- 개선 아이디어, 버그 제보, 기능 제안 등은 이슈 또는 PR로 남겨주세요.

""") 