import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- Upload data ---
st.title("Smart Money Visualizer")
uploaded_file = st.file_uploader("Upload Daily OHLCV CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.lower() for col in df.columns]
    required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}

    if required_cols.issubset(set(df.columns)):
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # --- Tagging logic ---
        df['tag'] = ''
        avg_volume = df['volume'].rolling(window=10).mean()

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            body = abs(row['close'] - row['open'])
            prev_body = abs(prev['close'] - prev['open'])

            if row['close'] > row['open'] and row['close'] >= row['high'] - (row['high'] - row['low']) * 0.1 and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = 'Aggressive Buyer'
            elif row['open'] > row['close'] and row['close'] <= row['low'] + (row['high'] - row['low']) * 0.1 and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = 'Aggressive Seller'
            elif row['high'] > prev['high'] and row['close'] < prev['close'] and (row['high'] - row['close']) > (row['close'] - row['open']) and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = 'Buyer Absorption'
            elif row['low'] < prev['low'] and row['close'] > prev['close'] and (row['close'] - row['low']) > (row['open'] - row['close']) and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = 'Seller Absorption'
            elif row['close'] > row['open'] and body < 0.3 * prev_body and row['volume'] < avg_volume[i]:
                df.at[i, 'tag'] = 'Bullish Weak Leg'
            elif row['open'] > row['close'] and body < 0.3 * prev_body and row['volume'] < avg_volume[i]:
                df.at[i, 'tag'] = 'Bearish Weak Leg'
            elif row['high'] > max(df['high'].iloc[max(0, i-3):i]) and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = 'Bullish PoR'
            elif row['low'] < min(df['low'].iloc[max(0, i-3):i]) and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = 'Bearish PoR'

        # --- Plot chart ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Close Price'))

        for tag in df['tag'].unique():
            if tag:
                subset = df[df['tag'] == tag]
                fig.add_trace(go.Scatter(
                    x=subset['date'],
                    y=subset['close'],
                    mode='markers',
                    name=tag,
                    marker=dict(size=8),
                    text=tag
                ))

        st.plotly_chart(fig, use_container_width=True)
        st.write(df[['date', 'open', 'high', 'low', 'close', 'volume', 'tag']].tail(30))

    else:
        st.error("Missing required columns: date, open, high, low, close, volume")
