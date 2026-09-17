import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model

# ==========================================
# 0. TÙY CHỈNH CSS (Giữ nguyên font chữ to, rõ)
# ==========================================
st.markdown(
    """
    <style>
    h3 {
        font-size: 26px !important;
        font-weight: 600 !important;
        color: #4da6ff !important;
    }
    .stSlider label p, .stSelectbox label p {
        font-size: 24px !important;
    }
    .stSlider, .stSelectbox {
        margin-top: -10px !important;
    }
    [data-baseweb="slider"] div, [data-baseweb="slider"] span {
        font-size: 22px !important;
        font-weight: 600 !important;
    }
    ul[data-baseweb="menu"] li span {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="ECC-Slab Predictor (Optimized)", layout="wide")
st.title("Prediction of Punching Shear Capacity of ECC-Strengthened Flat Slabs")
st.markdown("##### **A Product Developed by:** Dr. Cong-Luyen Nguyen | Ngoc Han Nguyen & Duc Nhan Hoang")
st.markdown("---")

# ==========================================
# 2. LOAD MODEL AND SCALER PARAMS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'ann_model.h5')
    scaler_path = os.path.join(BASE_DIR, 'scaler_params.npz')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Cannot find: {scaler_path}")

    model = load_model(model_path, compile=False)
    scaler_params = np.load(scaler_path)

    X_min = scaler_params['X_min']
    X_max = scaler_params['X_max']
    y_min = scaler_params['y_min']
    y_max = scaler_params['y_max']

    return model, X_min, X_max, y_min, y_max

try:
    model, X_min, X_max, y_min, y_max = load_assets()
except Exception as e:
    st.error(f"INITIALIZATION ERROR: {e}")
    st.stop()

# ==========================================
# 3. GIAO DIỆN NHẬP LIỆU (Chỉ gồm 9 biến độc lập)
# ==========================================
with st.expander("**⚙ INPUT PARAMETERS (Click to Expand / Collapse)**", expanded=True):
    col1, col2, col3 = st.columns(3)

    # CỘT 1: Độ dày & Đặc tính ECC
    with col1:
        st.subheader("1. Thickness Setup")
        tc = st.slider("Concrete thickness - tc (mm)", 100, 160, 100)
        tECC = st.slider("ECC thickness - tECC (mm)", 30, 90, 30)
        
        st.subheader("4. ECC Properties")
        fc_ECC = st.slider("ECC compressive strength - f'c,ECC (MPa)", 31, 60, 31)

    # CỘT 2: Cấu hình hình học
    with col2:
        st.subheader("2. Geometry Configuration")
        c1 = st.slider("Column's short side dimension - c1 (mm)", 150, 250, 150)
        c2_c1 = st.slider("Long-to-short side dimension ratio - c2/c1", 1.0, 1.67, 1.0)
        
        # 2: Corner, 3: Edge, 4: Interior
        alpha_s = st.selectbox("Loading location - αs (2: Corner, 3: Edge, 4: Interior)", [2.0, 3.0, 4.0], index=2)

    # CỘT 3: Đặc tính Bê tông & Thép
    with col3:
        st.subheader("3. Normal Concrete (NC)")
        fc_c = st.slider("Concrete compressive strength - f'c,c (MPa)", 30, 60, 30)
        
        st.subheader("5. Reinforcement Details")
        fy = st.slider("Rebar yield strength - fy (MPa)", 456, 750, 456)
        mu = st.slider("Reinforcement ratio - μ (%)", 1.227, 2.454, 1.227, step=0.001)

st.markdown("---")
run_button = st.button("🚀 RUN PREDICTION", use_container_width=True)

# ==========================================
# 4. XỬ LÝ DỮ LIỆU & DỰ ĐOÁN
# ==========================================
# LƯU Ý QUAN TRỌNG: Thứ tự các biến trong mảng input_data dưới đây 
# PHẢI KHỚP TUYỆT ĐỐI với thứ tự cột trong DataFrame lúc bạn train trên Colab.
# Giả định thứ tự là: tc, tECC, fc_c, fc_ECC, c1, c2_c1, alpha_s, fy, mu
input_data = np.array([[c1, c2_c1, L/d, alpha_s, fc_c, fc_ECC, tc, tECC, fy, mu]])

if run_button:
    with st.spinner("Analyzing data..."):
        # Chuẩn hóa đầu vào (Min-Max)
        input_scaled = (input_data - X_min) / (X_max - X_min)
        
        # Dự đoán
        prediction_norm = model.predict(input_scaled)
        
        # Giải chuẩn hóa đầu ra
        prediction_real = prediction_norm[0][0] * (y_max - y_min) + y_min
        
        st.success("Prediction Completed!")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            # Bạn có thể cập nhật lại giá trị MAE theo kết quả đánh giá mô hình mới nhất trên tập Test
            MAE_error = 8.59 
            st.markdown("<p style='font-size: 24px; font-weight: bold; margin-bottom: 0px;'>Predicted Punching Shear Capacity (Vp)</p>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='display: flex; align-items: baseline; gap: 8px; margin-bottom: 0px;'>"
                f"<span style='font-size: 40px; font-weight: bold;'>{prediction_real:.2f} kN</span>"
                f"<span style='font-size: 16px; color: #A5A5A5;'>± {MAE_error} kN Expected Error (MAE)</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            # Cập nhật R2 theo mô hình mới
            st.markdown("<p style='font-size: 14px; color: #09AB3B; margin-top: 5px;'>↑ ANN Model (R² = 0.99)</p>", unsafe_allow_html=True)
        
        with col_res2:
            st.info(f"**Total Slab Thickness:** {tc + tECC:.1f} mm")

st.markdown("---")
