import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from urllib.parse import quote

# Google Sheet Configuration (Updated for your columns)
SHEET_ID = "1_pmG2oMSEk8VciNm2uqcshyvPPZBbjf-oKV59chgT1w"
SHEET_NAME = "Daily Price"
GSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(SHEET_NAME)}"

# Signal Detection Parameters
VOLUME_THRESHOLD_MULTIPLIER = 1.5  # Times average volume
WICK_TO_BODY_RATIO = 1.5  # For absorption detection
MIN_POR_MOVE = 0.03  # 3% price move for POR detection

def detect_signals(df):
    """Detect Smart Money patterns in OHLCV data"""
    df = df.copy()
    df['Signal'] = np.nan
    df['Signal_Type'] = np.nan
    df['Body'] = abs(df['CLOSE'] - df['OPEN'])
    df['Upper_Wick'] = df['HIGH'] - df[['OPEN', 'CLOSE']].max(axis=1)
    df['Lower_Wick'] = df[['OPEN', 'CLOSE']].min(axis=1) - df['LOW']
    avg_volume = df['VOLUME'].rolling(20).mean().fillna(df['VOLUME'].mean())
    
    for i in range(1, len(df)):
        # Bullish POR
        if (df['CLOSE'].iloc[i] > df['HIGH'].iloc[i-1] * (1 + MIN_POR_MOVE) and \
           (df['VOLUME'].iloc[i] > avg_volume.iloc[i]):
            df.loc[df.index[i], 'Signal'] = df['HIGH'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Bullish POR'
        
        # Bearish POR
        elif (df['CLOSE'].iloc[i] < df['LOW'].iloc[i-1] * (1 - MIN_POR_MOVE)) and \
             (df['VOLUME'].iloc[i] > avg_volume.iloc[i]):
            df.loc[df.index[i], 'Signal'] = df['LOW'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Bearish POR'
        
        # Buyers Absorption (Long lower wick + green candle + high volume)
        elif (df['Lower_Wick'].iloc[i] > df['Body'].iloc[i] * WICK_TO_BODY_RATIO) and \
             (df['CLOSE'].iloc[i] > df['OPEN'].iloc[i]) and \
             (df['VOLUME'].iloc[i] > avg_volume.iloc[i] * VOLUME_THRESHOLD_MULTIPLIER):
            df.loc[df.index[i], 'Signal'] = df['LOW'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Buyers Absorption'
        
        # Sellers Absorption (Long upper wick + red candle + high volume)
        elif (df['Upper_Wick'].iloc[i] > df['Body'].iloc[i] * WICK_TO_BODY_RATIO) and \
             (df['CLOSE'].iloc[i] < df['OPEN'].iloc[i]) and \
             (df['VOLUME'].iloc[i] > avg_volume.iloc[i] * VOLUME_THRESHOLD_MULTIPLIER):
            df.loc[df.index[i], 'Signal'] = df['HIGH'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Sellers Absorption'
    
    # POI Detection (Swing Highs/Lows)
    df['Swing_High'] = (df['HIGH'] > df['HIGH'].shift(1)) & (df['HIGH'] > df['HIGH'].shift(-1))
    df['Swing_Low'] = (df['LOW'] < df['LOW'].shift(1)) & (df['LOW'] < df['LOW'].shift(-1))
    
    return df

def create_chart(df, symbol):
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['DATE'],
        open=df['OPEN'],
        high=df['HIGH'],
        low=df['LOW'],
        close=df['CLOSE'],
        name='Price'
    ))
    
    # Add signals
    signal_colors = {
        'Bullish POR': 'green',
        'Bearish POR': 'red',
        'Buyers Absorption': 'blue',
        'Sellers Absorption': 'orange'
    }
    
    for sig_type, color in signal_colors.items():
        sig_df = df[df['Signal_Type'] == sig_type]
        if not sig_df.empty:
            fig.add_trace(go.Scatter(
                x=sig_df['DATE'],
                y=sig_df['Signal'],
                mode='markers',
                name=sig_type,
                marker=dict(color=color, size=12, symbol='diamond'),
                hovertemplate=f'{sig_type}<br>Price: %{{y}}<extra></extra>'
            ))
    
    # Add POI markers
    for _, row in df[df['Swing_High']].iterrows():
        fig.add_annotation(
            x=row['DATE'],
            y=row['HIGH'],
            text="POI (Bear)",
            showarrow=True,
            arrowhead=1,
            bgcolor="red"
        )
    
    for _, row in df[df['Swing_Low']].iterrows():
        fig.add_annotation(
            x=row['DATE'],
            y=row['LOW'],
            text="POI (Bull)",
            showarrow=True,
            arrowhead=1,
            bgcolor="green"
        )
    
    fig.update_layout(
        title=f"{symbol} - Smart Money Analysis",
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    
    return fig

# Streamlit App
st.set_page_config(layout="wide")
st.title("📊 Smart Money Dashboard")

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv(GSHEET_URL)
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df.sort_values('DATE')
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    symbol = st.selectbox("Select Symbol", sorted(df['SYMBOL'].unique()))
    
    if symbol:
        symbol_df = df[df['SYMBOL'] == symbol].copy()
        analyzed_df = detect_signals(symbol_df)
        
        st.plotly_chart(create_chart(analyzed_df, symbol), use_container_width=True)
        
        st.subheader("Detected Signals")
        st.dataframe(
            analyzed_df[
                ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'Signal_Type']
            ].dropna(subset=['Signal_Type']),
            use_container_width=True
        )