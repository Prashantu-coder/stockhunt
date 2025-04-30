import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from urllib.parse import quote

# Google Sheet Configuration
SHEET_ID = "1_pmG2oMSEk8VciNm2uqcshyvPPZBbjf-oKV59chgT1w"
SHEET_NAME = "Daily Price"
GSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(SHEET_NAME)}"

def clean_data(df):
    """Clean and convert raw dataframe"""
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Convert date - handle Excel serial numbers and string dates
    if df['DATE'].dtype == 'int64':
        df['DATE'] = pd.to_datetime(df['DATE'], unit='D', origin='1899-12-30')
    else:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    
    # Clean numeric columns
    num_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    for col in num_cols:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Drop rows with invalid dates or missing values
    df = df.dropna(subset=['DATE', 'SYMBOL'])
    return df.sort_values('DATE')

def detect_signals(df):
    """Detect Smart Money patterns"""
    df = df.copy()
    df['Signal'] = np.nan
    df['Signal_Type'] = np.nan
    
    # Calculate technical metrics
    df['Body'] = abs(df['CLOSE'] - df['OPEN'])
    df['Upper_Wick'] = df['HIGH'] - df[['OPEN', 'CLOSE']].max(axis=1)
    df['Lower_Wick'] = df[['OPEN', 'CLOSE']].min(axis=1) - df['LOW']
    avg_volume = df['VOLUME'].rolling(20).mean().fillna(df['VOLUME'].mean())
    
    # Signal detection parameters
    VOLUME_THRESHOLD = 1.5
    WICK_RATIO = 1.5
    
    for i in range(1, len(df)):
        # Bullish Breakout
        if (df['CLOSE'].iloc[i] > df['HIGH'].iloc[i-1]) and \
           (df['VOLUME'].iloc[i] > avg_volume.iloc[i] * VOLUME_THRESHOLD):
            df.loc[df.index[i], 'Signal'] = df['HIGH'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Bullish POR'
        
        # Bearish Breakout
        elif (df['CLOSE'].iloc[i] < df['LOW'].iloc[i-1]) and \
             (df['VOLUME'].iloc[i] > avg_volume.iloc[i] * VOLUME_THRESHOLD):
            df.loc[df.index[i], 'Signal'] = df['LOW'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Bearish POR'
        
        # Buyers Absorption
        elif (df['Lower_Wick'].iloc[i] > df['Body'].iloc[i] * WICK_RATIO) and \
             (df['CLOSE'].iloc[i] > df['OPEN'].iloc[i]) and \
             (df['VOLUME'].iloc[i] > avg_volume.iloc[i]):
            df.loc[df.index[i], 'Signal'] = df['LOW'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Buyers Absorption'
        
        # Sellers Absorption
        elif (df['Upper_Wick'].iloc[i] > df['Body'].iloc[i] * WICK_RATIO) and \
             (df['CLOSE'].iloc[i] < df['OPEN'].iloc[i]) and \
             (df['VOLUME'].iloc[i] > avg_volume.iloc[i]):
            df.loc[df.index[i], 'Signal'] = df['HIGH'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Sellers Absorption'
    
    return df

def create_chart(df, symbol):
    """Create interactive visualization"""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['DATE'],
        open=df['OPEN'],
        high=df['HIGH'],
        low=df['LOW'],
        close=df['CLOSE'],
        name='Price'
    ))
    
    # Add signals
    signals = df[df['Signal_Type'].notna()]
    for _, row in signals.iterrows():
        fig.add_annotation(
            x=row['DATE'],
            y=row['Signal'],
            text=row['Signal_Type'],
            showarrow=True,
            arrowhead=1,
            bgcolor="green" if "Bullish" in row['Signal_Type'] else "red"
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
        return clean_data(df)
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    st.subheader("Data Preview")
    st.dataframe(df.head(3))
    
    symbol = st.selectbox("Select Symbol", sorted(df['SYMBOL'].astype(str).unique()))
    
    if symbol:
        symbol_df = df[df['SYMBOL'] == symbol].copy()
        analyzed_df = detect_signals(symbol_df)
        
        st.plotly_chart(create_chart(analyzed_df, symbol), use_container_width=True)
        
        st.subheader("Detected Signals")
        st.dataframe(
            analyzed_df[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'Signal_Type']]
            .dropna(subset=['Signal_Type']),
            use_container_width=True
        )