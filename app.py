import streamlit as st
from datetime import datetime
from skyfield.api import load, Topos
import numpy as np
import cv2

# 사용자 입력
st.title("ND 필터 진단 소프트웨어 (2차 버전)")
lat = st.number_input("위도", value=35.6351)
lon = st.number_input("경도", value=127.4263)
date = st.date_input("날짜", value=datetime(2025, 6, 18))
time = st.time_input("시각", value=datetime.strptime("14:00", "%H:%M").time())
r_corr = st.number_input("R_corr 값", value=1.0)

# 고도 계산
ts = load.timescale()
eph = load('de421.bsp')
t_obs = ts.utc(datetime.combine(date, time))
observer = Topos(latitude_degrees=lat, longitude_degrees=lon)
sun = eph["sun"]
earth = eph["earth"]
alt, _, _ = (earth + observer).at(t_obs).observe(sun).apparent().altaz()

secZ = 1 / np.cos(np.radians(90 - alt.degrees))
predicted_dm = 0.25 * secZ

st.write(f"🌞 태양 고도: **{alt.degrees:.2f}°**, 예상 Δm = **{predicted_dm:.3f}**")

# R_corr 판별
if r_corr < 0.95:
    st.error("⚠️ R_corr 값이 낮습니다. 필터 감쇠 또는 손상 의심.")
else:
    st.success("✅ R_corr 양호")

# 이미지 분석 (선택적)
uploaded_img = st.file_uploader("태양 이미지 업로드", type=["jpg", "png"])
if uploaded_img:
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="업로드된 태양 이미지", use_column_width=True)
