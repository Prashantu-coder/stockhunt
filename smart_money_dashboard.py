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
    """Clean and prepare data without showing preview"""
    # Convert date (handles Excel serial numbers and string dates)
    if df['DATE'].dtype == 'int64':
        df['DATE'] = pd.to_datetime(df['DATE'], unit='D', origin='1899-12-30')
    else:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    
    # Clean numeric columns (remove commas)
    num_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    for col in num_cols:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    return df.dropna(subset=['DATE', 'SYMBOL']).sort_values('DATE')

def detect_signals(df):
    """Smart Money signal detection"""
    df = df.copy()
    df['Signal'] = np.nan
    df['Signal_Type'] = np.nan
    
    # Technical calculations
    df['Body'] = abs(df['CLOSE'] - df['OPEN'])
    df['Upper_Wick'] = df['HIGH'] - df[['OPEN', 'CLOSE']].max(axis=1)
    df['Lower_Wick'] = df[['OPEN', 'CLOSE']].min(axis=1) - df['LOW']
    avg_volume = df['VOLUME'].rolling(20).mean().fillna(df['VOLUME'].mean())
    
    # Signal detection
    for i in range(1, len(df)):
        if (df['CLOSE'].iloc[i] > df['HIGH'].iloc[i-1]) and (df['VOLUME'].iloc[i] > avg_volume.iloc[i] * 1.5):
            df.loc[df.index[i], ['Signal', 'Signal_Type']] = [df['HIGH'].iloc[i], 'Bullish POR']
        elif (df['CLOSE'].iloc[i] < df['LOW'].iloc[i-1]) and (df['VOLUME'].iloc[i] > avg_volume.iloc[i] * 1.5):
            df.loc[df.index[i], ['Signal', 'Signal_Type']] = [df['LOW'].iloc[i], 'Bearish POR']
    
    return df

def create_chart(df, symbol):
    """Create clean visualization without preview clutter"""
    fig = go.Figure(go.Candlestick(
        x=df['DATE'],
        open=df['OPEN'],
        high=df['HIGH'],
        low=df['LOW'],
        close=df['CLOSE'],
        name='Price'
    ))
    
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
        title=f"{symbol} Analysis",
        xaxis_rangeslider_visible=False
    )
    return fig

# Streamlit UI (Minimal Version)
st.set_page_config(layout="wide")
st.title("Smart Money Dashboard")

@st.cache_data(ttl=3600)
def load_data():
    try:
        return clean_data(pd.read_csv(GSHEET_URL))
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

# Main App Flow
df = load_data()
if not df.empty:
    symbol = st.selectbox("Select Symbol", sorted(df['SYMBOL'].unique()))
    if symbol:
        analyzed_df = detect_signals(df[df['SYMBOL'] == symbol])
        st.plotly_chart(create_chart(analyzed_df, symbol), use_container_width=True)
        
        # Optional: Show signals table only if signals exist
        if not analyzed_df['Signal_Type'].isna().all():
            st.subheader("Signals")
            st.dataframe(
                analyzed_df[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'Signal_Type']]
                .dropna(subset=['Signal_Type']),
                hide_index=True
            )