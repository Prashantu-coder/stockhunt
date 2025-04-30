import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.parse import quote

# --- Basic Setup ---
st.set_page_config(layout="wide")
st.title("💎 SMART MONEY SIGNALS")

# --- Load Data ---
SHEET_ID = "1_pmG2oMSEk8VciNm2uqcshyvPPZBbjf-oKV59chgT1w"
SHEET_NAME = "Daily Price"
GSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(SHEET_NAME)}"

try:
    df = pd.read_csv(GSHEET_URL)
    df.columns = [col.lower().strip() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''))
    df = df.dropna()
except:
    st.error("DATA LOADING FAILED! CHECK YOUR SHEET.")
    st.stop()

# --- Simple Signal Detection ---
def find_signals(df):
    df = df.copy()
    df['signal'] = ''
    avg_vol = df['volume'].rolling(20).mean()
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # BIG GREEN CANDLE
        if current['close'] > current['open'] and current['volume'] > avg_vol.iloc[i]*1.5:
            df.at[i, 'signal'] = 'BUYER'
        
        # BIG RED CANDLE
        elif current['open'] > current['close'] and current['volume'] > avg_vol.iloc[i]*1.5:
            df.at[i, 'signal'] = 'SELLER'
    
    return df

# --- BIG VISIBLE CHART ---
symbol = st.selectbox("SELECT STOCK", df['symbol'].unique())
stock_df = df[df['symbol'] == symbol].copy()
signals_df = find_signals(stock_df)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=signals_df['date'],
    y=signals_df['close'],
    mode='lines+markers',
    name=f'{symbol} PRICE',
    line=dict(color='black', width=3)
))

# GIANT MARKERS
signals = signals_df[signals_df['signal'] != '']
if not signals.empty:
    fig.add_trace(go.Scatter(
        x=signals['date'],
        y=signals['close'],
        mode='markers',
        name='SIGNALS',
        marker=dict(
            color=['green' if s == 'BUYER' else 'red' for s in signals['signal']],
            size=15,
            symbol='diamond',
            line=dict(width=2, color='white')
    ))

fig.update_layout(
    title=f"🚨 {symbol} SIGNALS 🚨",
    height=600,
    font=dict(size=18)
st.plotly_chart(fig, use_container_width=True)

# --- SIMPLE TABLE ---
if not signals.empty:
    st.subheader("🔥 DETECTED SIGNALS")
    st.dataframe(
        signals[['date', 'close', 'volume', 'signal']],
        column_config={
            "date": "DATE",
            "close": "PRICE",
            "volume": "VOLUME",
            "signal": "SIGNAL"
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.warning("NO SIGNALS FOUND FOR THIS STOCK")