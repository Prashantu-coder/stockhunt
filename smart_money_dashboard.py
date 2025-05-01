import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# --- Page setup ---
st.set_page_config(page_title="Smart Money Visualizer", layout="wide")
st.title("💸 Smart Money Visualizer")

# --- Upload data ---
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
                df.at[i, 'tag'] = '🟢 Aggressive buyers'
            elif row['open'] > row['close'] and row['close'] <= row['low'] + (row['high'] - row['low']) * 0.1 and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = '🔴 Aggressive sellers'
            elif row['high'] > prev['high'] and row['close'] < prev['close'] and (row['high'] - row['close']) > (row['close'] - row['open']) and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = '⛔Buyer absorption'
            elif row['low'] < prev['low'] and row['close'] > prev['close'] and (row['close'] - row['low']) > (row['open'] - row['close']) and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = '🚀 Seller absorption'
            elif row['close'] > row['open'] and body < 0.3 * prev_body and row['volume'] < avg_volume[i]:
                df.at[i, 'tag'] = '📉Bullish weak legs'
            elif row['open'] > row['close'] and body < 0.3 * prev_body and row['volume'] < avg_volume[i]:
                df.at[i, 'tag'] = '📈Bearish weak legs'
            elif row['high'] > max(df['high'].iloc[max(0, i-3):i]) and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = '💥 Bullish POR'
            elif row['low'] < min(df['low'].iloc[max(0, i-3):i]) and row['volume'] > avg_volume[i]:
                df.at[i, 'tag'] = '💣Bearish POR'
            # --- POI logic
            elif row['close'] > row['open'] and (row['close'] - row['open']) > (row['high'] - row['low']) * 0.7 and row['volume'] > avg_volume[i]*1.5:
                df.at[i, 'tag'] = '🐂 Bullish POI'
            elif row['open'] > row['close'] and (row['open'] - row['close']) > (row['high'] - row['low']) * 0.7 and row['volume'] > avg_volume[i]*1.5:
                df.at[i, 'tag'] = '🐻 Bearish POI'

        # --- Dropdown Multiselect ---
        tags_available = df['tag'].unique()
        tags_available = [tag for tag in tags_available if tag]

        selected_tags = st.multiselect(
            "Select Signal(s) to View", 
            options=tags_available, 
            default=tags_available
        )

        # --- Plotting Chart ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['close'], 
            mode='lines', 
            name='Close Price',
            line=dict(color='lightblue')
        ))

        for tag in selected_tags:
            subset = df[df['tag'] == tag]
            
            text_color = "white"
            if '🐂' in tag:
                text_color = "lime"
            elif '🐻' in tag:
                text_color = "red"
            
            fig.add_trace(go.Scatter(
                x=subset['date'],
                y=subset['close'],
                mode='text',
                name=tag,
                text=[tag]*len(subset),
                textposition="top center",
                textfont=dict(size=16, color=text_color),
            ))

        fig.update_layout(
            height=800,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            legend=dict(font=dict(size=12)),
            title="Smart Money Signals Chart"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Table for last 1 month signals ---
        st.subheader("📋 Recent 1 Month Signal Observed")

        last_date = df['date'].max()
        one_month_ago = last_date - timedelta(days=30)
        recent_df = df[(df['date'] >= one_month_ago) & (df['tag'] != '')]

        st.dataframe(recent_df[['date', 'open', 'high', 'low', 'close', 'volume', 'tag']].sort_values('date', ascending=False))

    else:
        st.error("❌ Missing required columns: date, open, high, low, close, volume")
