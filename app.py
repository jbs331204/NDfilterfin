import streamlit as st
from datetime import datetime
from skyfield.api import load, Topos
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 🌐 Streamlit 기본 설정
st.set_page_config(page_title="ND 필터 자동 진단 소프트웨어", layout="centered")
st.title("☀️ ND 필터 자동 진단 소프트웨어 (2차 버전)")

# 📌 사용자 입력
st.header("1. 관측 조건 입력")
lat = st.number_input("위도 (Latitude)", value=35.6351)
lon = st.number_input("경도 (Longitude)", value=127.4263)
date = st.date_input("관측 날짜", value=datetime(2025, 6, 18))
time = st.time_input("관측 시각", value=datetime.strptime("14:00", "%H:%M").time())
r_corr = st.number_input("R_corr (보정된 R 채널값)", value=1.00)

# 🌤 태양 고도 및 소광 계산
st.header("2. 대기 소광 자동 계산")
try:
    # Skyfield 시간 객체 생성 (주의: naive datetime → 직접 분해)
    t_combined = datetime.combine(date, time)
    ts = load.timescale()
    eph = load('de421.bsp')
    observer = Topos(latitude_degrees=lat, longitude_degrees=lon)
    sun = eph["sun"]
    earth = eph["earth"]
    t_obs = ts.utc(t_combined.year, t_combined.month, t_combined.day,
                   t_combined.hour, t_combined.minute)

    # 고도 계산
    alt, _, _ = (earth + observer).at(t_obs).observe(sun).apparent().altaz()
    secZ = 1 / np.cos(np.radians(90 - alt.degrees))
    k_assumed = 0.25
    predicted_dm = k_assumed * secZ

    st.write(f"☀️ **태양 고도**: `{alt.degrees:.2f}°`")
    st.write(f"🔍 **sec(Z)**: `{secZ:.3f}`")
    st.write(f"🧮 이론 밝기 변화량 (Δm): `{predicted_dm:.3f}` (k={k_assumed})")

except Exception as e:
    st.error(f"❌ Skyfield 계산 오류: {e}")

# 🔎 R_corr 진단
st.header("3. 색상 감쇠 진단")
if r_corr < 0.95:
    st.error("⚠️ R_corr 값이 기준치보다 낮습니다. R 채널 감쇠 또는 필터 열화 가능성이 있습니다.")
else:
    st.success("✅ R_corr 값이 정상 범위입니다.")

# 📸 이미지 업로드
st.header("4. 태양 이미지 업로드")
uploaded_img = st.file_uploader("태양 영상 이미지 업로드 (JPG/PNG)", type=["jpg", "png"])

if uploaded_img:
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # 이미지 처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.1)

    center = gray[cy - r:cy + r, cx - r:cx + r]
    ring = gray[cy - 2*r:cy + 2*r, cx - 2*r:cx + 2*r]
    center_mean = np.mean(center)
    ring_mean = np.mean(ring)
    scattering_index = ring_mean / center_mean

    st.image(img, caption="업로드된 태양 이미지", use_column_width=True)
    st.markdown(f"📊 **Scattering Index**: `{scattering_index:.3f}`")

    # RGB 비율 분석
    center_rgb = img[cy - r:cy + r, cx - r:cx + r]
    center_r = np.mean(center_rgb[:, :, 2])
    center_g = np.mean(center_rgb[:, :, 1])
    center_b = np.mean(center_rgb[:, :, 0])

    rg_ratio = center_r / center_g
    gb_ratio = center_g / center_b
    rb_ratio = center_r / center_b

    st.markdown("### 🎨 중심부 RGB 채널 비율")
    st.write(f"🔴 R/G: `{rg_ratio:.3f}`")
    st.write(f"🟢 G/B: `{gb_ratio:.3f}`")
    st.write(f"🔵 R/B: `{rb_ratio:.3f}`")

    # 간단 시각화
    st.header("📈 RGB 비율 시각화")
    fig, ax = plt.subplots()
    ratios = [rg_ratio, gb_ratio, rb_ratio]
    labels = ["R/G", "G/B", "R/B"]
    colors = ["red", "green", "blue"]
    ax.bar(labels, ratios, color=colors)
    ax.set_ylim([0, 2])
    ax.set_ylabel("비율")
    ax.set_title("RGB 채널 비율")
    st.pyplot(fig)
