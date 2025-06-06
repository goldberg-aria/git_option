import streamlit as st

st.set_page_config(page_title="옵션 전략 추천 시스템", layout="centered")
st.title("옵션 전략 추천 시스템")
st.write("아래에서 원하는 버전을 선택하세요.")

st.page_link("stable_app.py", label="안정 버전 (최초 커밋)")
st.page_link("realtime_strategy_recommend.py", label="개발 중 최신 버전") 