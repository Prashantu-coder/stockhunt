import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.parse import quote

# --- Configuration ---
st.set_page_config(layout="wide")
st.title("📊 Complete Smart Money Signals")

# --- Data Loading ---
SHEET_ID = "1_pmG2oMSEk8VciNm2uqcshyvPPZBbjf-oKV59chgT1w"
SHEET_NAME = "Daily Price"
GSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(SHEET_NAME)}"

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv(GSHEET_URL)
        df.columns = [col.lower().strip() for col in df.columns]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean numeric columns
        num_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        return df.dropna(subset=['date', 'symbol']).sort_values('date')
    except Exception as e:
        st.error(f"🚨 Data Error: {str(e)}")
        return pd.DataFrame()

# --- Complete Signal Detection ---
def detect_all_signals(df):
    df = df.copy()
    df['signal'] = ''
    df['avg_volume'] = df['volume'].rolling(20, min_periods=1).mean()
    
    for i in range(1, len(df)):
        try:
            row = df.iloc[i]
            prev = df.iloc[i-1]
            body = abs(row['close'] - row['open'])
            range_ = row['high'] - row['low']
            
            # 1. Breakouts (POR)
            if row['high'] > df['high'].iloc[max(0,i-3):i].max() and row['volume'] > row['avg_volume']:
                df.at[i, 'signal'] = 'Bullish POR'
            elif row['low'] < df['low'].iloc[max(0,i-3):i].min() and row['volume'] > row['avg_volume']:
                df.at[i, 'signal'] = 'Bearish POR'
            
            # 2. Aggressive Participants
            elif (row['close'] > row['open'] and 
                  (row['close'] >= row['high'] - 0.1*range_) and 
                  row['volume'] > row['avg_volume']*1.5):
                df.at[i, 'signal'] = 'Aggressive Buyer'
            elif (row['open'] > row['close'] and 
                  (row['close'] <= row['low'] + 0.1*range_) and 
                  row['volume'] > row['avg_volume']*1.5):
                df.at[i, 'signal'] = 'Aggressive Seller'
            
            # 3. Absorption Patterns
            elif (row['high'] > prev['high'] and 
                  row['close'] < prev['close'] and 
                  (row['high'] - row['close']) > body and 
                  row['volume'] > row['avg_volume']):
                df.at[i, 'signal'] = 'Buyer Absorption'
            elif (row['low'] < prev['low'] and 
                  row['close'] > prev['close'] and 
                  (row['close'] - row['low']) > body and 
                  row['volume'] > row['avg_volume']):
                df.at[i, 'signal'] = 'Seller Absorption'
                
        except:
            continue
            
    return df

# --- Enhanced Visualization ---
def create_complete_chart(df, symbol):
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='Price',
        line=dict(color='#636EFA', width=2),
        hovertemplate="Date: %{x}<br>Price: %{y}<extra></extra>"
    ))
    
    # All signal configurations
    signal_config = {
        'Bullish POR': {'color': '#00CC96', 'symbol': 'triangle-up', 'size': 12},
        'Bearish POR': {'color': '#EF553B', 'symbol': 'triangle-down', 'size': 12},
        'Aggressive Buyer': {'color': '#00FF00', 'symbol': 'circle', 'size': 10},
        'Aggressive Seller': {'color': '#FF0000', 'symbol': 'circle', 'size': 10},
        'Buyer Absorption': {'color': '#1E90FF', 'symbol': 'square', 'size': 10},
        'Seller Absorption': {'color': '#FFA500', 'symbol': 'square', 'size': 10}
    }
    
    # Add all signals
    for signal, style in signal_config.items():
        sig_df = df[df['signal'] == signal]
        if not sig_df.empty:
            fig.add_trace(go.Scatter(
                x=sig_df['date'],
                y=sig_df['close'],
                mode='markers',
                name=signal,
                marker=dict(
                    color=style['color'],
                    size=style['size'],
                    symbol=style['symbol'],
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                hovertemplate=f"{signal}<br>Date: %{{x}}<br>Price: %{{y}}<extra></extra>"
            ))
    
    fig.update_layout(
        title=f"{symbol} - Complete Smart Money Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

# --- App Execution ---
df = load_data()

if not df.empty:
    symbol = st.selectbox("Select Symbol", sorted(df['symbol'].astype(str).unique()))
    
    if symbol:
        analyzed_df = detect_all_signals(df[df['symbol'] == symbol])
        
        # Display complete chart
        st.plotly_chart(create_complete_chart(analyzed_df, symbol), use_container_width=True)
        
        # Signals table
        signals_df = analyzed_df[analyzed_df['signal'] != ''][
            ['date', 'open', 'high', 'low', 'close', 'volume', 'signal']
        ]
        if not signals_df.empty:
            st.subheader("📋 Detected Signals")
            st.dataframe(
                signals_df.sort_values('date', ascending=False),
                use_container_width=True,
                hide_index=True
            )