import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.parse import quote

# --- Configuration ---
st.set_page_config(layout="wide")
st.title("Smart Money Signal Visualizer")

# --- Data Loading ---
SHEET_ID = "1_pmG2oMSEk8VciNm2uqcshyvPPZBbjf-oKV59chgT1w"
SHEET_NAME = "Daily Price"
GSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(SHEET_NAME)}"

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv(GSHEET_URL)
        # Clean and standardize column names
        df.columns = [col.lower().strip() for col in df.columns]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df.dropna(subset=['date']).sort_values('date')
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

# --- Signal Detection ---
def detect_signals(df):
    """Detects all Smart Money patterns"""
    df = df.copy()
    df['signal'] = ''
    avg_volume = df['volume'].rolling(20).mean().fillna(df['volume'].mean())
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        body = abs(row['close'] - row['open'])
        range_ = row['high'] - row['low']
        
        # 1. Breakouts (POR)
        if row['high'] > max(df['high'].iloc[max(0,i-3):i]) and row['volume'] > avg_volume.iloc[i]:
            df.at[i, 'signal'] = 'Bullish POR'
        elif row['low'] < min(df['low'].iloc[max(0,i-3):i]) and row['volume'] > avg_volume.iloc[i]:
            df.at[i, 'signal'] = 'Bearish POR'
        
        # 2. Aggressive Participants
        elif (row['close'] > row['open'] and 
              row['close'] >= row['high'] - 0.1*range_ and 
              row['volume'] > avg_volume.iloc[i]*1.5):
            df.at[i, 'signal'] = 'Aggressive Buyer'
        elif (row['open'] > row['close'] and 
              row['close'] <= row['low'] + 0.1*range_ and 
              row['volume'] > avg_volume.iloc[i]*1.5):
            df.at[i, 'signal'] = 'Aggressive Seller'
        
        # 3. Absorption Patterns
        elif (row['high'] > prev['high'] and 
              row['close'] < prev['close'] and 
              (row['high'] - row['close']) > body and 
              row['volume'] > avg_volume.iloc[i]):
            df.at[i, 'signal'] = 'Buyer Absorption'
        elif (row['low'] < prev['low'] and 
              row['close'] > prev['close'] and 
              (row['close'] - row['low']) > body and 
              row['volume'] > avg_volume.iloc[i]):
            df.at[i, 'signal'] = 'Seller Absorption'
    
    return df

# --- Visualization ---
def create_line_chart(df, symbol):
    fig = go.Figure()
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate="Date: %{x}<br>Price: %{y}<extra></extra>"
    ))
    
    # Signal markers
    signal_config = {
        'Bullish POR': {'color': 'green', 'symbol': 'triangle-up'},
        'Bearish POR': {'color': 'red', 'symbol': 'triangle-down'},
        'Aggressive Buyer': {'color': 'lime', 'symbol': 'circle'},
        'Aggressive Seller': {'color': 'crimson', 'symbol': 'circle'},
        'Buyer Absorption': {'color': 'blue', 'symbol': 'square'},
        'Seller Absorption': {'color': 'orange', 'symbol': 'square'}
    }
    
    for signal, style in signal_config.items():
        signal_df = df[df['signal'] == signal]
        if not signal_df.empty:
            fig.add_trace(go.Scatter(
                x=signal_df['date'],
                y=signal_df['close'],
                mode='markers',
                name=signal,
                marker=dict(
                    color=style['color'],
                    size=10,
                    symbol=style['symbol'],
                    line=dict(width=1, color='black')
                ),
                hovertemplate=f"{signal}<br>Price: %{{y}}<extra></extra>"
            ))
    
    fig.update_layout(
        title=f"{symbol} - Smart Money Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig

# --- App Execution ---
df = load_data()

if not df.empty:
    # Symbol selection
    symbol = st.selectbox("Select Symbol", sorted(df['symbol'].astype(str).unique()))
    
    if symbol:
        symbol_df = df[df['symbol'] == symbol].copy()
        analyzed_df = detect_signals(symbol_df)
        
        # Display chart
        st.plotly_chart(create_line_chart(analyzed_df, symbol), use_container_width=True)
        
        # Show signals table (only if signals exist)
        if not analyzed_df['signal'].empty:
            st.subheader("Detected Signals")
            st.dataframe(
                analyzed_df[['date', 'open', 'high', 'low', 'close', 'volume', 'signal']]
                .loc[analyzed_df['signal'] != ''],
                use_container_width=True,
                hide_index=True
            )