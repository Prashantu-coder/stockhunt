import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Google Sheet Configuration
SHEET_ID = "1_pmG2oMSEk8VciNm2uqcshyvPPZBbjf-oKV59chgT1w"
SHEET_NAME = "Daily Price"
GSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

# Signal Detection Parameters
VOLUME_THRESHOLD_MULTIPLIER = 1.5  # Times average volume
WICK_TO_BODY_RATIO = 1.5  # For absorption detection
MIN_POR_MOVE = 0.03  # 3% price move for POR detection

def detect_signals(df):
    """Detect Smart Money patterns in OHLCV data"""
    df = df.copy()
    df['Signal'] = np.nan
    df['Signal_Type'] = np.nan
    
    # Calculate metrics
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    avg_volume = df['Volume'].rolling(20).mean().fillna(df['Volume'].mean())
    
    # Detect patterns
    for i in range(1, len(df)):
        # Bullish POR (Breakout with high volume)
        if (df['Close'].iloc[i] > df['High'].iloc[i-1] * (1 + MIN_POR_MOVE) and
            df['Volume'].iloc[i] > avg_volume.iloc[i]):
            df.loc[df.index[i], 'Signal'] = df['High'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Bullish POR'
        
        # Bearish POR (Breakdown with high volume)
        elif (df['Close'].iloc[i] < df['Low'].iloc[i-1] * (1 - MIN_POR_MOVE) and
              df['Volume'].iloc[i] > avg_volume.iloc[i]):
            df.loc[df.index[i], 'Signal'] = df['Low'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Bearish POR'
        
        # Buyers Absorption
        elif (df['Lower_Wick'].iloc[i] > df['Body'].iloc[i] * WICK_TO_BODY_RATIO and
              df['Close'].iloc[i] > df['Open'].iloc[i] and
              df['Volume'].iloc[i] > avg_volume.iloc[i] * VOLUME_THRESHOLD_MULTIPLIER):
            df.loc[df.index[i], 'Signal'] = df['Low'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Buyers Absorption'
        
        # Sellers Absorption
        elif (df['Upper_Wick'].iloc[i] > df['Body'].iloc[i] * WICK_TO_BODY_RATIO and
              df['Close'].iloc[i] < df['Open'].iloc[i] and
              df['Volume'].iloc[i] > avg_volume.iloc[i] * VOLUME_THRESHOLD_MULTIPLIER):
            df.loc[df.index[i], 'Signal'] = df['High'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Sellers Absorption'
        
        # Aggressive Buyers
        elif (df['Close'].iloc[i] > df['Open'].iloc[i] and
              df['Body'].iloc[i] > df['Upper_Wick'].iloc[i] and
              df['Volume'].iloc[i] > avg_volume.iloc[i] * VOLUME_THRESHOLD_MULTIPLIER):
            df.loc[df.index[i], 'Signal'] = df['Close'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Aggressive Buyer'
        
        # Aggressive Sellers
        elif (df['Close'].iloc[i] < df['Open'].iloc[i] and
              df['Body'].iloc[i] > df['Lower_Wick'].iloc[i] and
              df['Volume'].iloc[i] > avg_volume.iloc[i] * VOLUME_THRESHOLD_MULTIPLIER):
            df.loc[df.index[i], 'Signal'] = df['Close'].iloc[i]
            df.loc[df.index[i], 'Signal_Type'] = 'Aggressive Seller'
    
    # Detect POI (Points of Interest) - Swing Highs/Lows
    df['Swing_High'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['Swing_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    
    return df

def create_chart(df, company):
    """Create interactive Plotly chart with signals"""
    fig = go.Figure()
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add signals
    signal_colors = {
        'Bullish POR': 'green',
        'Bearish POR': 'red',
        'Buyers Absorption': 'blue',
        'Sellers Absorption': 'orange',
        'Aggressive Buyer': 'lime',
        'Aggressive Seller': 'crimson'
    }
    
    for signal_type, color in signal_colors.items():
        signal_df = df[df['Signal_Type'] == signal_type]
        if not signal_df.empty:
            fig.add_trace(go.Scatter(
                x=signal_df['Date'],
                y=signal_df['Signal'],
                mode='markers',
                name=signal_type,
                marker=dict(color=color, size=10, symbol='diamond'),
                hovertemplate=f'{signal_type}<br>Price: %{{y}}<extra></extra>'
            ))
    
    # Add POI markers
    poi_df = df[df['Swing_High'] | df['Swing_Low']]
    for _, row in poi_df.iterrows():
        if row['Swing_High']:
            fig.add_annotation(
                x=row['Date'],
                y=row['High'],
                text="POI (Bear)",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                bgcolor="red"
            )
        else:
            fig.add_annotation(
                x=row['Date'],
                y=row['Low'],
                text="POI (Bull)",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=40,
                bgcolor="green"
            )
    
    fig.update_layout(
        title=f"{company} - Smart Money Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        showlegend=True
    )
    
    return fig

# Streamlit App
st.set_page_config(layout="wide", page_title="Smart Money Dashboard")
st.title("📊 Smart Money Analysis Dashboard")

# Load data
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(GSHEET_URL)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

try:
    full_df = load_data()
    
    # Company search
    company = st.text_input("🔍 Search Company", placeholder="Enter company name (e.g., NTC)")
    
    if company:
        # Filter data
        company_df = full_df[full_df['Company'].str.strip().str.lower() == company.strip().lower()]
        
        if company_df.empty:
            st.warning(f"No data found for '{company}'")
        else:
            # Detect signals
            analyzed_df = detect_signals(company_df)
            
            # Display chart
            st.plotly_chart(create_chart(analyzed_df, company), use_container_width=True)
            
            # Show raw data with signals
            st.subheader("Signal Details")
            st.dataframe(
                analyzed_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Signal_Type']]
                .dropna(subset=['Signal_Type']),
                use_container_width=True
            )
except Exception as e:
    st.error(f"Error loading data: {str(e)}")