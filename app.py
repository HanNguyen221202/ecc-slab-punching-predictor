import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model

st.set_page_config(page_title="ECC-Slab Predictor", layout="wide")

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

st.title("Prediction of Punching Shear Capacity of ECC-Strengthened Flat (ESRC) Slabs")
st.markdown("##### **Developed by:** Dr. Cong-Luyen Nguyen | Ngoc Han Nguyen & Duc Nhan Hoang")
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
# 3. GIAO DIỆN NHẬP LIỆU (10 biến)
# ==========================================
with st.expander("**INPUT PARAMETERS (Click to Expand / Collapse)**", expanded=True):
    col1, col2, col3 = st.columns(3)

    # CỘT 1: Độ dày & Đặc tính ECC
    with col1:
        st.subheader("1. Thickness & Effective Depth:")
        tc = st.slider("Concrete thickness - tc (mm)", 100, 160, 100)
        tECC = st.slider("ECC thickness - tECC (mm)", 30, 90, 30)
        
        # Tự động tính d dựa trên quy định tiêu chuẩn về lớp bảo vệ mặc định
        cover = 15
        d = tc - cover
        st.info(f"Slab's effective depth (d) auto-calculated: **{d} mm**")

        st.subheader("4. ECC Layer Property:")
        fc_ECC = st.slider("ECC compressive strength - fc,ECC (MPa)", 31, 60, 31)

    # CỘT 2: Cấu hình hình học
    with col2:
        st.subheader("2. Geometry Configuration:")
        c1 = st.slider("Column's short side dimension - c1 (mm)", 150, 250, 150)
        c2_c1 = st.slider("Long-to-short side dimension ratio - c2/c1", 1.00, 1.67, 1.0)
        
        # 2: Corner, 3: Edge, 4: Interior
        alpha_s = st.selectbox("Loading location - αs (2: Corner, 3: Edge, 4: Interior)", [2.0, 3.0, 4.0], index=2)

    # CỘT 3: Đặc tính Bê tông & Thép
    with col3:
        st.subheader("3. Normal Concrete (NC) Property:")
        fc_c = st.slider("Concrete compressive strength - fc,c (MPa)", 30, 60, 30)
        
        st.subheader("5. Reinforcement Details:")
        fy = st.slider("Rebar yield strength - fy (MPa)", 456, 750, 456)
        mu = st.slider("Reinforcement ratio - μ (%)", 1.227, 2.454, 1.227, step=0.001, format="%.3f")

st.markdown("---")
run_button = st.button("RUN PREDICTION", use_container_width=True)

# ==========================================
# 4. XỬ LÝ DỮ LIỆU & DỰ ĐOÁN (Chỉ giữ 1 khối duy nhất)
# ==========================================
# Mảng 10 biến nạp vào mô hình theo đúng thứ tự cột trong file CSV:
input_data = np.array([[c1, c2_c1, alpha_s, fc_c, fc_ECC, tc, tECC, fy, mu]])

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
            MAE_error = 8.59 
            st.markdown(
                f"""
                <div style="background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 6px 16px rgba(0,0,0,0.08); border-left: 8px solid #4da6ff; display: flex; flex-direction: column; justify-content: center; height: 100%;">
                    <div style="font-size: 15px; font-weight: 700; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px;">
                        Predicted Punching Shear Capacity (Vp)
                    </div>
                    <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 12px;">
                        <span style="font-size: 52px; font-weight: 900; background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{prediction_real:.2f}</span>
                        <span style="font-size: 26px; font-weight: 800; color: #203a43;">kN</span>
                        <span style="font-size: 15px; font-weight: 500; color: #95a5a6; margin-left: 4px;">± {MAE_error} (MAE)</span>
                    </div>
                    <div>
                        <span style="background-color: #e8f8f5; color: #1abc9c; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: 0 2px 5px rgba(26, 188, 156, 0.15);">
                            ANN Model (R² = 0.99)
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Đảm bảo khối này thụt lề 8 dấu cách (nằm trọn vẹn trong if run_button:)
        with col_res2:
            st.markdown(
                f"<div style='background-color: #e6f3ff; padding: 18px; border-radius: 8px; border-left: 6px solid #0088cc; display: flex; align-items: center; height: 100%;'>"
                f"<span style='font-size: 24px; color: #003366;'><b>Total Slab Thickness:</b> {tc + tECC:.1f} mm</span>"
                f"</div>",
                unsafe_allow_html=True
            )
