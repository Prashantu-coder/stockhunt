import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.parse import quote

# --- App Setup ---
st.set_page_config(layout="wide")
st.title("💰 CRYSTAL CLEAR SMART MONEY SIGNALS")

# --- Data Loading with Debug ---
SHEET_ID = "1_pmG2oMSEk8VciNm2uqcshyvPPZBbjf-oKV59chgT1w"
SHEET_NAME = "Daily Price"
GSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(SHEET_NAME)}"

def load_data():
    try:
        st.write("🔍 Attempting to load data from Google Sheets...")
        df = pd.read_csv(GSHEET_URL)
        
        st.write("✅ Raw data loaded. First 3 rows:")
        st.write(df.head(3))
        
        # Check for required columns
        required_cols = {'date', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        missing_cols = required_cols - set(col.lower().strip() for col in df.columns)
        
        if missing_cols:
            st.error(f"❌ MISSING COLUMNS: {missing_cols}")
            return None
        
        # Clean data
        df.columns = [col.lower().strip() for col in df.columns]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        df = df.dropna(subset=['date', 'symbol'])
        
        st.write("🧹 Cleaned data sample:")
        st.write(df.head(3))
        
        return df
    
    except Exception as e:
        st.error(f"🔥 CRITICAL ERROR: {str(e)}")
        st.error("Please check:")
        st.error("1. Google Sheet sharing settings (must be 'Anyone with link can view')")
        st.error("2. Column names in your sheet (need: date, symbol, open, high, low, close, volume)")
        st.error("3. Internet connection")
        return None

# --- Simple Signal Detection ---
def find_signals(df):
    df = df.copy()
    df['signal'] = ''
    df['avg_volume'] = df['volume'].rolling(20, min_periods=1).mean()
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # Big Green Candle with High Volume
        if row['close'] > row['open'] and row['volume'] > row['avg_volume']*1.5:
            df.at[i, 'signal'] = 'BUY'
        
        # Big Red Candle with High Volume
        elif row['open'] > row['close'] and row['volume'] > row['avg_volume']*1.5:
            df.at[i, 'signal'] = 'SELL'
    
    return df

# --- Main App ---
df = load_data()

if df is not None:
    st.success("✅ DATA LOADED SUCCESSFULLY!")
    
    symbol = st.selectbox("SELECT STOCK SYMBOL", sorted(df['symbol'].unique()))
    stock_df = df[df['symbol'] == symbol].copy()
    
    if not stock_df.empty:
        signals_df = find_signals(stock_df)
        
        # --- BIG VISIBLE CHART ---
        fig = go.Figure()
        
        # Price Line
        fig.add_trace(go.Scatter(
            x=signals_df['date'],
            y=signals_df['close'],
            mode='lines',
            name=f'{symbol} PRICE',
            line=dict(color='black', width=3)
        ))
        
        # Signals
        signals = signals_df[signals_df['signal'] != '']
        if not signals.empty:
            fig.add_trace(go.Scatter(
                x=signals['date'],
                y=signals['close'],
                mode='markers',
                name='SIGNALS',
                marker=dict(
                    size=20,
                    color=['green' if s == 'BUY' else 'red' for s in signals['signal']],
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                text=signals['signal'],
                hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Price: %{y}"
            ))
        
        fig.update_layout(
            title=f"📈 {symbol} - SMART MONEY SIGNALS",
            height=600,
            font=dict(size=16),
            hoverlabel=dict(font_size=16)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- SIMPLE TABLE ---
        if not signals.empty:
            st.subheader("📋 DETECTED SIGNALS")
            st.dataframe(
                signals[['date', 'close', 'volume', 'signal']].sort_values('date', ascending=False),
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
            st.warning(f"No signals detected for {symbol} (need big moves with high volume)")
    else:
        st.error(f"No data found for symbol: {symbol}")
else:
    st.error("Cannot proceed without data. Please fix the errors above.")