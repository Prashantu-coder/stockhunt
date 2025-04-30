import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.parse import quote

# --- Configuration ---
st.set_page_config(layout="wide")
st.title("🔍 Debugged Smart Money Signals")

# --- Debugging Setup ---
debug = st.checkbox("Show debug info", True)

# --- Data Loading with Validation ---
SHEET_ID = "1_pmG2oMSEk8VciNm2uqcshyvPPZBbjf-oKV59chgT1w"
SHEET_NAME = "Daily Price"
GSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(SHEET_NAME)}"

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv(GSHEET_URL)
        if debug:
            st.write("🔧 Raw columns from sheet:", df.columns.tolist())
        
        # Standardize column names (case insensitive, strip whitespace)
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Handle common column name variations
        column_mapping = {
            'symbol': ['symbol', 'ticker', 'company'],
            'date': ['date', 'timestamp'],
            'open': ['open', 'opening'],
            'high': ['high', 'highprice'],
            'low': ['low', 'lowprice'],
            'close': ['close', 'closing', 'last'],
            'volume': ['volume', 'vol', 'quantity']
        }
        
        # Find matching columns
        matched_cols = {}
        for standard_name, alternatives in column_mapping.items():
            for alt in alternatives:
                if alt in df.columns:
                    matched_cols[standard_name] = alt
                    break
        
        if debug:
            st.write("🔍 Matched columns:", matched_cols)
        
        # Check if we have all required columns
        required = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in matched_cols]
        
        if missing:
            st.error(f"❌ Missing columns: {missing}")
            return pd.DataFrame()
        
        # Rename columns to standard names
        df = df.rename(columns={v:k for k,v in matched_cols.items()})
        
        # Convert and clean data
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Drop invalid rows
        clean_df = df.dropna(subset=['date', 'symbol']).sort_values('date')
        
        if debug:
            st.write("✅ Cleaned data sample:", clean_df.head(3))
            st.write("📊 Data types:", clean_df.dtypes)
            st.write("📅 Date range:", clean_df['date'].min(), "to", clean_df['date'].max())
            st.write("📌 Unique symbols:", clean_df['symbol'].unique())
        
        return clean_df
    
    except Exception as e:
        st.error(f"🚨 Critical Error: {str(e)}")
        return pd.DataFrame()

# --- Signal Detection ---
def detect_signals(df):
    if df.empty:
        return df
    
    df = df.copy()
    df['signal'] = ''
    df['avg_volume'] = df['volume'].rolling(20, min_periods=1).mean()
    
    for i in range(1, len(df)):
        try:
            row = df.iloc[i]
            prev = df.iloc[i-1]
            body = abs(row['close'] - row['open'])
            range_ = row['high'] - row['low']
            
            # Signal Conditions
            if row['high'] > df['high'].iloc[max(0,i-3):i].max() and row['volume'] > row['avg_volume']:
                df.at[i, 'signal'] = 'Bullish POR'
            elif row['low'] < df['low'].iloc[max(0,i-3):i].min() and row['volume'] > row['avg_volume']:
                df.at[i, 'signal'] = 'Bearish POR'
            elif (row['close'] > row['open'] and row['close'] >= row['high'] - 0.1*range_ and row['volume'] > row['avg_volume']*1.5):
                df.at[i, 'signal'] = 'Aggressive Buyer'
            elif (row['open'] > row['close'] and row['close'] <= row['low'] + 0.1*range_ and row['volume'] > row['avg_volume']*1.5):
                df.at[i, 'signal'] = 'Aggressive Seller'
            elif (row['high'] > prev['high'] and row['close'] < prev['close'] and (row['high'] - row['close']) > body and row['volume'] > row['avg_volume']):
                df.at[i, 'signal'] = 'Buyer Absorption'
            elif (row['low'] < prev['low'] and row['close'] > prev['close'] and (row['close'] - row['low']) > body and row['volume'] > row['avg_volume']):
                df.at[i, 'signal'] = 'Seller Absorption'
                
        except Exception as e:
            if debug:
                st.write(f"⚠️ Error processing row {i}: {str(e)}")
            continue
    
    if debug:
        st.write("🔍 Signals detected:", df['signal'].value_counts())
    
    return df

# --- Main App ---
df = load_data()

if not df.empty:
    symbol = st.selectbox("Select Symbol", sorted(df['symbol'].astype(str).unique()))
    
    if symbol:
        symbol_df = df[df['symbol'] == symbol].copy()
        analyzed_df = detect_signals(symbol_df)
        
        if not analyzed_df.empty:
            # Display chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=analyzed_df['date'],
                y=analyzed_df['close'],
                mode='lines',
                name='Price',
                line=dict(color='#636EFA', width=2)
            ))
            
            # Add all signals
            signals = analyzed_df[analyzed_df['signal'] != '']
            if not signals.empty:
                fig.add_trace(go.Scatter(
                    x=signals['date'],
                    y=signals['close'],
                    mode='markers',
                    name='Signals',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='triangle-up'
                    ),
                    text=signals['signal'],
                    hovertemplate="%{text}<br>Date: %{x}<br>Price: %{y}"
                ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show signals table
            st.subheader("📋 Detected Signals")
            st.dataframe(
                signals[['date', 'open', 'high', 'low', 'close', 'volume', 'signal']],
                use_container_width=True
            )
        else:
            st.warning("No signals detected for this symbol")
    else:
        st.error("No symbol selected")
else:
    st.error("No valid data loaded")