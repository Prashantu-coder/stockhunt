import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")

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

        # --- Define emojis for each tag ---
        tag_emojis = {
            'Bullish PoR': '💥 Bullish PoR',
            'Bearish PoR': '💣 Bearish PoR',
            'Aggressive Buyer': '🟢 Aggressive Buyer',
            'Aggressive Seller': '🔴 Aggressive Seller',
            'Buyer Absorption': '⛔ Buyer Absorption',
            'Seller Absorption': '🚀 Seller Absorption',
            'Bullish Weak Leg': '📉 Bullish Weak Leg',
            'Bearish Weak Leg': '📈 Bearish Weak Leg',
            'Bullish POI': '🐂 Bullish POI',
            'Bearish POI': '🐻 Bearish POI'
        }

        # --- Define marker symbol for each tag ---
        tag_symbols = {
            'Bullish PoR': 'star',
            'Bearish PoR': 'star-diamond',
            'Aggressive Buyer': 'triangle-up',
            'Aggressive Seller': 'triangle-down',
            'Buyer Absorption': 'x',
            'Seller Absorption': 'x-open',
            'Bullish Weak Leg': 'diamond',
            'Bearish Weak Leg': 'diamond-open',
            'Bullish POI': 'square',
            'Bearish POI': 'square-open'
        }

        # --- Define marker color for each tag ---
        tag_colors = {
            'Bullish PoR': 'green',
            'Bearish PoR': 'red',
            'Aggressive Buyer': 'lime',
            'Aggressive Seller': 'orangered',
            'Buyer Absorption': 'blue',
            'Seller Absorption': 'purple',
            'Bullish Weak Leg': 'deepskyblue',
            'Bearish Weak Leg': 'hotpink',
            'Bullish POI': 'darkgreen',
            'Bearish POI': 'darkred'
        }

        # --- Plot chart ---
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['close'], 
            mode='lines', 
            name='Close Price', 
            line=dict(color='lightgrey')
        ))

        for tag in df['tag'].unique():
            if tag:
                subset = df[df['tag'] == tag]
                display_name = tag_emojis.get(tag, tag)
                symbol = tag_symbols.get(tag, 'circle')
                color = tag_colors.get(tag, 'black')
                fig.add_trace(go.Scatter(
                    x=subset['date'],
                    y=subset['close'],
                    mode='markers',
                    name=display_name,
                    marker=dict(
                        size=12,
                        symbol=symbol,
                        color=color,
                        line=dict(width=1, color='black')
                    ),
                    text=[display_name]*len(subset),
                    hoverinfo='text'
                ))

        fig.update_layout(
            height=800,
            plot_bgcolor="black",
            legend=dict(font=dict(size=12)),
            title='Smart Money Signals Chart'
        )

        st.plotly_chart(fig, use_container_width=True)
        st.write(df[['date', 'open', 'high', 'low', 'close', 'volume', 'tag']].tail(30))

    else:
        st.error("Missing required columns: date, open, high, low, close, volume")
