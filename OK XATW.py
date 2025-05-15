import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import io
from datetime import timedelta

# --- Page setup ---
st.set_page_config(page_title="Quantexo", layout="wide")
st.title("Advanced Insights for Bold Trades")

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

        # --- Signal Tagging ---
        df['tag'] = ''
        avg_volume = df['volume'].rolling(window=10).mean()

        for i in range(min(3, len(df)-1), len(df)):
                row = df.iloc[i]
                prev = df.iloc[i - 1]
                next_candles = df.iloc[i + 1:min(i + 6, len(df))]
                is_last_candle = (i == len(df) - 1)  # Flag for last candle

                body = abs(row['close'] - row['open'])
                prev_body = abs(prev['close'] - prev['open'])
                recent_tags = df['tag'].iloc[max(0, i - 4):i]

                # --- Signals that DON'T require future data (always checked) ---
                if (
                    row['close'] > row['open'] and
                    row['close'] >= row['high'] - (row['high'] - row['low']) * 0.1 and
                    row['volume'] > avg_volume[i] * 1.5 and
                    body > prev_body and
                    '🟢' not in recent_tags.values
                ):
                    df.at[i, 'tag'] = '🟢'

                elif (
                    row['open'] > row['close'] and
                    row['close'] <= row['low'] + (row['high'] - row['low']) * 0.1 and
                    row['volume'] > avg_volume[i] * 1.5 and
                    body > prev_body and
                    '🔴' not in recent_tags.values
                ):
                    df.at[i, 'tag'] = '🔴'

                elif (
                    i >= 10 and
                    row['high'] > max(df['high'].iloc[i - 10:i]) and
                    row['volume'] > avg_volume[i] * 1.8
                ):
                    if not (df['tag'].iloc[i - 3:i] == '💥').any():
                        df.at[i, 'tag'] = '💥'

                elif (
                    i >= 10 and
                    row['low'] < min(df['low'].iloc[i - 10:i]) and
                    row['volume'] > avg_volume[i] * 1.8
                ):
                    if not (df['tag'].iloc[i - 3:i] == '💣').any():
                        df.at[i, 'tag'] = '💣'

                elif (
                    row['close'] > row['open'] and
                    body > (row['high'] - row['low']) * 0.7 and
                    row['volume'] > avg_volume[i] * 2
                ):
                    df.at[i, 'tag'] = '🐂'

                elif (
                    row['open'] > row['close'] and
                    body > (row['high'] - row['low']) * 0.7 and
                    row['volume'] > avg_volume[i] * 2
                ):
                    df.at[i, 'tag'] = '🐻'

                # --- Signals that normally require future data ---
                # Modified to work on last candle with adjusted conditions
                if is_last_candle:
                    # For ⛔ (Buyer Absorption): Check if last candle is bullish with high volume
                    if (
                        row['close'] > row['open'] and
                        row['volume'] > avg_volume[i] * 1.5
                    ):
                        df.at[i, 'tag'] = '⛔ (Potential)'

                    # For 🚀 (Seller Absorption): Check if last candle is bearish with high volume
                    elif (
                        row['open'] > row['close'] and
                        row['volume'] > avg_volume[i] * 1.5
                    ):
                        df.at[i, 'tag'] = '🚀 (Potential)'
                else:
                    # Original future-dependent logic for non-last candles
                    if (
                        row['close'] > row['open'] and
                        row['volume'] > avg_volume[i] * 1.2
                    ):
                        df.loc[df['tag'] == '⛔', 'tag'] = ''
                        for j, candle in next_candles.iterrows():
                            if candle['close'] < row['open']:
                                df.at[j, 'tag'] = '⛔'
                                break

                    elif (
                        row['open'] > row['close'] and
                        row['volume'] > avg_volume[i] * 1.2
                    ):
                        df.loc[df['tag'] == '🚀', 'tag'] = ''
                        for j, candle in next_candles.iterrows():
                            if candle['close'] > row['open']:
                                df.at[j, 'tag'] = '🚀'
                                break

        # --- Filter tags ---
        tags_available = [tag for tag in df['tag'].unique() if tag]
        selected_tags = st.multiselect("Select Signal(s) to View", options=tags_available, default=tags_available)

        # --- Plotting Chart ---
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['date'], y=df['close'],
            mode='lines', name='Close Price',
            line=dict(color='lightblue', width=2),
            hovertext=df['close'],
            hoverinfo="x+y+text"
        ))

        # Tag descriptions
        tag_labels = {
            '🟢': '🟢 Aggressive Buyers',
            '🔴': '🔴 Aggressive Sellers',
            '⛔': '⛔ Buyer Absorption',
            '🚀': '🚀 Seller Absorption',
            '💥': '💥 Bullish POR',
            '💣': '💣 Bearish POR',
            '🐂': '🐂 Bullish POI',
            '🐻': '🐻 Bearish POI',
        }

        for tag in selected_tags:
            subset = df[df['tag'] == tag]
            fig.add_trace(go.Scatter(
                x=subset['date'], y=subset['close'],
                mode='markers+text',
                name=tag_labels.get(tag, tag),
                text=[tag] * len(subset),
                textposition='top center',
                textfont=dict(size=20),
                marker=dict(size=14, symbol="circle", color='white'),
                customdata=subset[['open', 'high', 'low', 'close']].values,
                hovertemplate=(
                    "📅 Date: %{x|%Y-%m-%d}<br>" +
                    "🟢 Open: %{customdata[0]:.2f}<br>" +
                    "📈 High: %{customdata[1]:.2f}<br>" +
                    "📉 Low: %{customdata[2]:.2f}<br>" +
                    "🔚 Close: %{customdata[3]:.2f}<br>" +
                    f"{tag_labels.get(tag, tag)}<extra></extra>"
                )
            ))

        fig.update_layout(
            height=800,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            legend=dict(font=dict(size=14)),
            title="Smart Money Signals Chart",
            xaxis=dict(
                title="Date",
                tickangle=-45,
                showgrid=False
            ),
            yaxis=dict(
                title="Price",
                showgrid=True,
                gridcolor="gray",
                zeroline=True,
                zerolinecolor="gray",
            ),
            margin=dict(l=50, r=50, b=150, t=50),
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Table for last 1 month signals ---
        st.subheader("📋 Recent 1 Month Signal Observed")
        last_date = df['date'].max()
        one_month_ago = last_date - timedelta(days=30)
        recent_df = df[(df['date'] >= one_month_ago) & (df['tag'] != '')]

        st.dataframe(recent_df[['date', 'open', 'high', 'low', 'close',  'volume', 'tag']].sort_values('date', ascending=False))

        # --- Download Excel ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            recent_df[['date', 'open', 'high', 'low', 'close', 'volume', 'tag']].to_excel(writer, index=False, sheet_name='Signals')
        processed_data = output.getvalue()

        st.download_button(
            label="📥 Download 1 Month Signals as Excel",
            data=processed_data,
            file_name='recent_1_month_signals.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    else:
        st.error("❌ Missing required columns: date, open, high, low, close, volume")  