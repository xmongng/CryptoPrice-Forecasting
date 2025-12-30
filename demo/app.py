import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import os

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN (UI/UX)
# ==========================================
st.set_page_config(layout="wide", page_title="AI Crypto Analytics 2025", page_icon="📈")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    .stAlert { border-radius: 10px; }
    .css-1kyx0rg { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOGIC XỬ LÝ DỮ LIỆU & MODEL
# ==========================================
@st.cache_data
def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        for col in ['Open', 'High', 'Low', 'Price']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
        df_filtered = df[df['Date'].dt.year == 2025]
        return df_filtered
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return pd.DataFrame()

class PredictionModel:
    def __init__(self, lags=10):
        self.lags = lags
        self.model = None
        self.scaler = None
        self.is_ready = False

        try:
            model_path = "/Users/mong/Documents/ComputerScience/AI4SE/code-close-price/mlp_model.pkl"
            scaler_path = "mlp_scaler.pkl"
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_ready = True
        except:
            pass

    def predict(self, price_series):
        if not self.is_ready:
            last_p = price_series[-1]
            return last_p * (1 + np.random.uniform(-0.02, 0.02))

        try:
            price_series = np.array(price_series).reshape(-1, 1)
            scaled = self.scaler.transform(price_series)
            X = scaled[-self.lags:].reshape(1, -1)
            pred_scaled = self.model.predict(X)
            pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            return pred
        except:
            return price_series[-1]

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
def main():
    c1, c2 = st.columns([1, 4])
    with c1:
        st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=80)
    with c2:
        st.title("AI Crypto Trend Analytics - Model 2025")
        st.caption("Hệ thống dự báo giá đóng cửa (Close Price) sử dụng mạng nơ-ron")

    csv_filename = "/Users/mong/Documents/ComputerScience/AI4SE/bitcoin2025.csv"

    if not os.path.exists(csv_filename):
        st.error(f"Không tìm thấy file '{csv_filename}'.")
        return

    df = load_data(csv_filename)
    if df.empty:
        st.warning("File không có dữ liệu phù hợp.")
        return

    # SIDEBAR
    st.sidebar.header("Bảng điều khiển")
    mode = st.sidebar.selectbox(
        "Chế độ phân tích:",
        ("Dự báo Tương lai (Next Day)", "Kiểm tra Model (Backtest)")
    )

    view_days = st.sidebar.slider("Số ngày hiển thị trên biểu đồ", 20, len(df), 60)

    # Thêm chọn ngày để xem dự đoán cho ngày kế tiếp
    selected_date = st.sidebar.date_input(
        "Chọn ngày để dự đoán ngày tiếp theo",
        value=df['Date'].max().date(),
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date()
    )
    selected_date = pd.to_datetime(selected_date)

    st.sidebar.divider()
    if not os.path.exists("mlp_scaler.pkl"):
        st.sidebar.warning("Đang chạy chế độ DEMO (Không tìm thấy file model)")
    else:
        st.sidebar.success("Đã nạp Model thực tế thành công")

    model_handler = PredictionModel(lags=10)

    # Lấy dữ liệu đến ngày được chọn (bao gồm ngày đó)
    df_up_to_selected = df[df['Date'] <= selected_date].copy()
    
    if len(df_up_to_selected) < model_handler.lags:
        st.error(f"Không đủ dữ liệu (cần ít nhất {model_handler.lags} ngày).")
        return

    last_date = pd.to_datetime(df_up_to_selected['Date'].iloc[-1])
    next_date = last_date + timedelta(days=1)

    current_price = df_up_to_selected['Price'].iloc[-1]
    input_series = df_up_to_selected['Price'].tail(model_handler.lags).values
    predicted_price = model_handler.predict(input_series)

    real_next_price = None
    if next_date in df['Date'].values:
        real_next_price = df[df['Date'] == next_date]['Price'].values[0]

    # Khu vực hiển thị kết quả
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    delta = predicted_price - current_price
    trend = "Tăng" if delta > 0 else "Giảm"

    if mode == "🔮 Dự báo Tương lai (Next Day)":
        pct = (delta / current_price) * 100 if current_price != 0 else 0

        with res_col1: st.metric("Giá hiện tại", f"${current_price:,.1f}")
        with res_col2: st.metric("Dự báo ngày mai", f"${predicted_price:,.1f}", f"{pct:+.2f}%")
        with res_col3:
            signal = "MUA (BUY)" if delta > 0 else "BÁN (SELL)"
            st.markdown(f"Tín hiệu: <b style='color:{'#00ffcc' if delta > 0 else '#ff4b4b'}'>{signal}</b>", unsafe_allow_html=True)
        with res_col4: st.write(f"Ngày dự báo: {next_date.strftime('%d/%m/%Y')}")

        plot_df = df_up_to_selected.tail(view_days)
        point_input_date = last_date
        point_input_price = current_price
        point_predict_date = next_date
        point_predict_price = predicted_price
        real_target_price = None

    else:  # Backtest
        if real_next_price is None:
            st.warning("Không có dữ liệu thực tế cho ngày tiếp theo. Hãy chọn ngày sớm hơn để backtest.")
            real_next_price = predicted_price  # fallback để vẫn vẽ được
            show_real_point = False
        else:
            show_real_point = True

        with res_col1: st.metric("Giá thực tế ngày mai", f"${real_next_price:,.1f}")
        with res_col2: st.metric("AI Dự đoán", f"${predicted_price:,.1f}")
        with res_col3:
            st.markdown(f"Xu hướng dự đoán: <b style='color:{'#00ffcc' if trend == 'Tăng' else '#ff4b4b'}'>{trend}</b>", unsafe_allow_html=True)
        with res_col4: st.write(f"Ngày Backtest: {next_date.strftime('%d/%m/%Y')}")

        plot_df = df[df['Date'] <= next_date].tail(view_days) if real_next_price is not None else df_up_to_selected.tail(view_days)
        point_input_date = last_date
        point_input_price = current_price
        point_predict_date = next_date
        point_predict_price = predicted_price
        real_target_price = real_next_price if show_real_point else None

    # BIỂU ĐỒ
    st.subheader("📊 Biểu đồ phân tích kỹ thuật")
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=plot_df['Date'],
        open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Price'],
        name="Dữ liệu giá",
        increasing_line_color='#00ffcc', decreasing_line_color='#ff4b4b'
    ))

    pred_color = "#00ffcc" if point_predict_price > point_input_price else "#ff4b4b"
    fig.add_trace(go.Scatter(
        x=[point_input_date, point_predict_date],
        y=[point_input_price, point_predict_price],
        mode='lines+markers',
        line=dict(color=pred_color, width=3, dash='dot'),
        marker=dict(size=10, symbol='diamond'),
        name="Dự báo của AI"
    ))

    if real_target_price is not None:
        fig.add_trace(go.Scatter(
            x=[point_predict_date], y=[real_target_price],
            mode='markers',
            marker=dict(color='white', size=12, symbol='circle-open', line=dict(width=2)),
            name="Giá thực tế"
        ))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

    # BẢNG DỮ LIỆU
    with st.expander("Xem dữ liệu chi tiết"):
        st.dataframe(df.sort_values(by='Date', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()