import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.parse import quote

# --- Simple Configuration ---
st.set_page_config(layout="wide")
st.title("Smart Money Signals")

# --- Clean Data Loading ---
SHEET_ID = "1_pmG2oMSEk8VciNm2uqcshyvPPZBbjf-oKV59chgT1w"
SHEET_NAME = "Daily Price"
GSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(SHEET_NAME)}"

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv(GSHEET_URL)
        
        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Convert and clean data
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        return df.dropna(subset=['date', 'symbol']).sort_values('date')
    except:
        return pd.DataFrame()

# --- Signal Detection ---
def detect_signals(df):
    df = df.copy()
    df['signal'] = ''
    df['avg_volume'] = df['volume'].rolling(20, min_periods=1).mean()
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        body = abs(row['close'] - row['open'])
        range_ = row['high'] - row['low']
        
        # Breakouts
        if row['high'] > df['high'].iloc[max(0,i-3):i].max() and row['volume'] > row['avg_volume']:
            df.at[i, 'signal'] = 'Bullish POR'
        elif row['low'] < df['low'].iloc[max(0,i-3):i].min() and row['volume'] > row['avg_volume']:
            df.at[i, 'signal'] = 'Bearish POR'
        
        # Aggressive Participants
        elif (row['close'] > row['open'] and row['close'] >= row['high'] - 0.1*range_ and row['volume'] > row['avg_volume']*1.5):
            df.at[i, 'signal'] = 'Aggressive Buyer'
        elif (row['open'] > row['close'] and row['close'] <= row['low'] + 0.1*range_ and row['volume'] > row['avg_volume']*1.5):
            df.at[i, 'signal'] = 'Aggressive Seller'
        
        # Absorption
        elif (row['high'] > prev['high'] and row['close'] < prev['close'] and (row['high'] - row['close']) > body and row['volume'] > row['avg_volume']):
            df.at[i, 'signal'] = 'Buyer Absorption'
        elif (row['low'] < prev['low'] and row['close'] > prev['close'] and (row['close'] - row['low']) > body and row['volume'] > row['avg_volume']):
            df.at[i, 'signal'] = 'Seller Absorption'
    
    return df

# --- Clean Visualization ---
def create_chart(df, symbol):
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))
    
    # Signals
    signals = df[df['signal'] != '']
    if not signals.empty:
        fig.add_trace(go.Scatter(
            x=signals['date'],
            y=signals['close'],
            mode='markers',
            name='Signals',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-up',
                line=dict(width=1, color='black')
            ),
            text=signals['signal'],
            hovertemplate="%{text}<br>Price: %{y}"
        ))
    
    fig.update_layout(
        title=f"{symbol} Price with Signals",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    return fig

# --- App Execution ---
df = load_data()

if not df.empty:
    symbol = st.selectbox("Select Symbol", sorted(df['symbol'].unique()))
    
    if symbol:
        analyzed_df = detect_signals(df[df['symbol'] == symbol])
        st.plotly_chart(create_chart(analyzed_df, symbol), use_container_width=True)
        
        # Show signals table if any exist
        signals = analyzed_df[analyzed_df['signal'] != '']
        if not signals.empty:
            st.write("Detected Signals:")
            st.dataframe(
                signals[['date', 'open', 'high', 'low', 'close', 'volume', 'signal']],
                use_container_width=True
            )
        else:
            st.info("No signals detected for this symbol")
else:
    st.error("Failed to load data. Please check your Google Sheet.")