import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.parse import quote

# --- App Setup ---
st.set_page_config(layout="wide")
st.title("💰 Smart Money Signals Dashboard")

# --- Data Loading ---
SHEET_ID = "1_pmG2oMSEk8VciNm2uqcshyvPPZBbjf-oKV59chgT1w"
SHEET_NAME = "Daily Price"
GSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(SHEET_NAME)}"

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv(GSHEET_URL)
        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]
        # Convert data types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        return df.dropna(subset=['date', 'symbol']).sort_values('date')
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
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
        
        # Breakout Signals
        if row['high'] > df['high'].iloc[max(0,i-3):i].max() and row['volume'] > row['avg_volume']:
            df.at[i, 'signal'] = 'Bullish POR'
        elif row['low'] < df['low'].iloc[max(0,i-3):i].min() and row['volume'] > row['avg_volume']:
            df.at[i, 'signal'] = 'Bearish POR'
        
        # Aggressive Participants
        elif (row['close'] > row['open'] and row['close'] >= row['high'] - 0.1*range_ and row['volume'] > row['avg_volume']*1.5):
            df.at[i, 'signal'] = 'Aggressive Buyer'
        elif (row['open'] > row['close'] and row['close'] <= row['low'] + 0.1*range_ and row['volume'] > row['avg_volume']*1.5):
            df.at[i, 'signal'] = 'Aggressive Seller'
        
        # Absorption Patterns
        elif (row['high'] > prev['high'] and row['close'] < prev['close'] and (row['high'] - row['close']) > body and row['volume'] > row['avg_volume']):
            df.at[i, 'signal'] = 'Buyer Absorption'
        elif (row['low'] < prev['low'] and row['close'] > prev['close'] and (row['close'] - row['low']) > body and row['volume'] > row['avg_volume']):
            df.at[i, 'signal'] = 'Seller Absorption'
    
    return df

# --- Enhanced Visualization ---
def create_chart(df, symbol):
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name=f'{symbol} Price',
        line=dict(color='#2c3e50', width=2),
        hovertemplate="Date: %{x}<br>Price: %{y}<extra></extra>"
    ))
    
    # Signal markers
    signal_colors = {
        'Bullish POR': '#27ae60',
        'Bearish POR': '#e74c3c',
        'Aggressive Buyer': '#2ecc71',
        'Aggressive Seller': '#c0392b',
        'Buyer Absorption': '#3498db',
        'Seller Absorption': '#e67e22'
    }
    
    for signal, color in signal_colors.items():
        signal_df = df[df['signal'] == signal]
        if not signal_df.empty:
            fig.add_trace(go.Scatter(
                x=signal_df['date'],
                y=signal_df['close'],
                mode='markers',
                name=signal,
                marker=dict(
                    color=color,
                    size=10,
                    symbol='diamond' if 'POR' in signal else 'circle',
                    line=dict(width=1, color='white')
                ),
                hovertemplate=f"{signal}<br>Date: %{{x}}<br>Price: %{{y}}<extra></extra>"
            ))
    
    fig.update_layout(
        title=f"{symbol} - Smart Money Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# --- Main App ---
df = load_data()

if not df.empty:
    # Symbol selection
    symbol = st.selectbox("Select Symbol", sorted(df['symbol'].astype(str).unique()))
    
    if symbol:
        # Filter and analyze data
        symbol_df = df[df['symbol'] == symbol].copy()
        analyzed_df = detect_signals(symbol_df)
        
        # Display chart
        st.plotly_chart(create_chart(analyzed_df, symbol), use_container_width=True)
        
        # Display signals table
        signals_df = analyzed_df[analyzed_df['signal'] != '']
        if not signals_df.empty:
            st.subheader("📋 Detected Signals")
            st.dataframe(
                signals_df[['date', 'open', 'high', 'low', 'close', 'volume', 'signal']]
                .sort_values('date', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No signals detected for this symbol")
else:
    st.error("Failed to load data. Please check your Google Sheet configuration.")