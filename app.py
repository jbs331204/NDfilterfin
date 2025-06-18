import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
from datetime import datetime, timezone
from skyfield.api import load, Topos

# 폴더 경로
frame_dir = "/content/extracted_frames"
image_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg") or f.endswith(".png")])

# 사용자 입력
st.title("ND 필터 자동 진단 소프트웨어 (2차 버전)")

lat = st.number_input("위도", value=35.6351)
lon = st.number_input("경도", value=127.4263)
date = st.date_input("관찰 날짜", value=datetime(2025, 6, 18).date())
time = st.time_input("관찰 시간", value=datetime.strptime("14:00", "%H:%M").time())
r_corr = st.number_input("R_corr (복정 R값)", value=1.0)

# Skyfield 구현복수형 적용
try:
    ts = load.timescale()
    eph = load('de421.bsp')
    dt_naive = datetime.combine(date, time)
    dt_aware = dt_naive.replace(tzinfo=timezone.utc)
    t_obs = ts.utc(dt_aware)

    observer = Topos(latitude_degrees=lat, longitude_degrees=lon)
    sun = eph["sun"]
    earth = eph["earth"]
    alt, _, _ = (earth + observer).at(t_obs).observe(sun).apparent().altaz()
    secZ = 1 / np.cos(np.radians(90 - alt.degrees))
    k = 0.25
    predicted_dm = k * secZ

    st.write(f"태양 고도: **{alt.degrees:.2f}°**, secZ = {secZ:.3f}")
    st.write(f"예상 밝기 감소량 (∆m): **{predicted_dm:.3f}** (k={k} 기준)")
except Exception as e:
    st.error(f"Skyfield 계산 오류: {e}")

# R_corr 감소 경고 메시지
st.subheader("R_corr R_corr \uc진단")
if r_corr < 0.95:
    st.error("⚠️ R_corr 값이 기준치보다 낮습니다. R 체널 감소 또는 필터 손상 가능성이 있습니다.")
else:
    st.success("✅ R_corr 값이 정상 범위에 있습니다.")

# 사용자 업로드 이미지 및 발과
uploaded_img = st.file_uploader("태양 이미지 업로드 (jpg/png)", type=["jpg", "png"])
if uploaded_img:
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.image(img, caption="업로드된 태양 이미지", use_column_width=True)
        st.write("중심 및 주부 밝기 반응을 통해 추가 진단이 가능합니다.")
    else:
        st.error("이미지를 보내는 데 실패했습니다.")
