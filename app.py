import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# --- ページ設定 ---
st.set_page_config(page_title="AI資産予測くん Premium", layout="wide", page_icon="📈")

# --- カスタムCSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI資産予測くん Premium")
st.markdown("Meta社開発のAIエンジン『Prophet』が、あなたの資産の未来を描き出します。")

# --- サイドバー：入力エリア ---
with st.sidebar:
    st.header("🛠 設定パネル")
    
    ticker_dict = {
        "ビットコイン (BTC-USD)": "BTC-USD",
        "イーサリアム (ETH-USD)": "ETH-USD",
        "S&P 500 (^GSPC)": "^GSPC",
        "ナスダック100 (^NDX)": "^NDX",
        "トヨタ自動車 (7203.T)": "7203.T",
        "ソニーグループ (6758.T)": "6758.T",
        "任天堂 (7974.T)": "7974.T",
        "米ドル/円 (JPY=X)": "JPY=X",
        "Apple (AAPL)": "AAPL",
        "NVIDIA (NVDA)": "NVDA",
    }
    
    selected_name = st.selectbox("銘柄を選択", list(ticker_dict.keys()))
    ticker = ticker_dict[selected_name]
    
    use_manual = st.checkbox("ティッカーを直接入力する")
    if use_manual:
        ticker = st.text_input("ティッカーシンボル", value="BTC-USD")
    
    st.markdown("---")
    years = st.slider("予測期間（年）", 1, 5, 2)
    investment_amount = st.number_input("投資予定額 (万円)", value=10, step=1)
    
    st.markdown("---")
    st.caption("※ 過去データに基づく統計的シミュレーションです。")

# --- データ取得と整形（ここを修正） ---
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache_data
def load_data(ticker):
    # データをダウンロード
    data = yf.download(ticker, start=START, end=TODAY, progress=False)
    
    # 【修正ポイント】データが空の場合はエラー
    if data.empty:
        return None

    # 【修正ポイント】MultiIndex（多層カラム）の処理
    # 最近のyfinanceはカラムが ('Close', 'BTC-USD') のようになる場合があるため、それを平坦化
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data.reset_index(inplace=True)
    
    # Prophet用にカラム名が正しいか確認
    # Dateカラムなどがindexに入っている場合や名前が違う場合に対応
    if 'Date' not in data.columns and 'Datetime' not in data.columns:
        # indexが日付の場合
        data.reset_index(inplace=True)
        
    return data

# --- メイン処理 ---
data = load_data(ticker)

if data is None or data.empty:
    st.error("データの取得に失敗しました。銘柄コードが正しいか確認してください。")
    st.stop()

try:
    with st.spinner('データを取得し、AIモデルを構築中...'):
        
        # 直近データの計算
        # カラム名の揺らぎを吸収（'Close' または 'Adj Close' を使用）
        target_col = 'Close'
        if 'Close' not in data.columns:
            if 'Adj Close' in data.columns:
                target_col = 'Adj Close'
            else:
                st.error("価格データ(Close)が見つかりませんでした。")
                st.stop()

        latest_close = float(data[target_col].iloc[-1])
        prev_close = float(data[target_col].iloc[-2])
        price_change = latest_close - prev_close
        price_change_pct = (price_change / prev_close) * 100

        currency = "JPY" if ".T" in ticker or "JPY=X" in ticker else "USD"
        currency_symbol = "¥" if currency == "JPY" else "$"

        # --- AI学習用データ作成 ---
        # Prophetに必要な 'ds' と 'y' の2列だけにする
        # 日付カラムの特定
        date_col = 'Date' if 'Date' in data.columns else 'Datetime'
        
        df_train = data[[date_col, target_col]].copy()
        df_train = df_train.rename(columns={date_col: "ds", target_col: "y"})
        
        # 時パース（タイムゾーン情報を削除してProphetのエラーを防ぐ）
        df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None)

        # モデル作成
        m = Prophet()
        m.fit(df_train)
        
        # 未来の枠を作成
        future = m.make_future_dataframe(periods=years * 365)
        forecast = m.predict(future)

        # 予測値の取得
        future_price = forecast['yhat'].iloc[-1]
        future_price_upper = forecast['yhat_upper'].iloc[-1]
        future_price_lower = forecast['yhat_lower'].iloc[-1]
        
        roi = ((future_price - latest_close) / latest_close) * 100

    # --- 結果表示 UI ---
    st.subheader(f"📊 {selected_name if not use_manual else ticker} の現状")
    c1, c2, c3 = st.columns(3)
    c1.metric("現在価格", f"{currency_symbol}{latest_close:,.2f}")
    c2.metric("前日比", f"{price_change:,.2f}", f"{price_change_pct:.2f}%")
    c3.metric("AI予測トレンド", "上昇傾向 📈" if roi > 0 else "下落傾向 📉", delta_color="normal")

    st.markdown("---")

    st.subheader(f"💰 もし今、{investment_amount}万円 投資していたら？ ({years}年後)")
    
    multiplier = future_price / latest_close
    expected_amount = investment_amount * multiplier
    best_case = investment_amount * (future_price_upper / latest_close)
    worst_case = investment_amount * (future_price_lower / latest_close)

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    
    with col_sim1:
        st.info("📉 最悪のシナリオ")
        st.markdown(f"### {currency_symbol}{worst_case:,.1f}万円")
        st.caption("予測レンジの下限値")
        
    with col_sim2:
        st.success("🎯 AIの予測中心値")
        st.markdown(f"## {currency_symbol}{expected_amount:,.1f}万円")
        diff = expected_amount - investment_amount
        st.caption(f"利益予想: {'+' if diff > 0 else ''}{currency_symbol}{diff:,.1f}万円 ({roi:+.1f}%)")

    with col_sim3:
        st.warning("🚀 最高のシナリオ")
        st.markdown(f"### {currency_symbol}{best_case:,.1f}万円")
        st.caption("予測レンジの上限値")

    st.markdown("---")
    st.subheader("📈 未来予測チャート")
    
    tab1, tab2 = st.tabs(["予測グラフ", "トレンド分析"])
    
    with tab1:
        st.markdown("青い線が予測値、水色の帯が予測の振れ幅です。")
        fig1 = plot_plotly(m, forecast)
        fig1.update_layout(height=500, xaxis_title="日付", yaxis_title="価格")
        st.plotly_chart(fig1, use_container_width=True)
        
    with tab2:
        st.markdown("曜日や季節ごとの傾向を分析します。")
        fig2 = m.plot_components(forecast)
        st.write(fig2)

except Exception as e:
    st.error(f"予期せぬエラーが発生しました: {e}")
    st.write("もしこのエラーが続く場合は、銘柄を変更してみてください。")