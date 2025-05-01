import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import io
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

        # --- Signal Tagging ---
        df['tag'] = ''
        avg_volume = df['volume'].rolling(window=10).mean()

        for i in range(3, len(df)-2):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            next1 = df.iloc[i + 1]
            next2 = df.iloc[i + 2]
            body = abs(row['close'] - row['open'])
            prev_body = abs(prev['close'] - prev['open'])

            # Refined signals with stricter filters
            if (
                row['close'] > row['open']
                and row['close'] >= row['high'] - (row['high'] - row['low']) * 0.1
                and row['volume'] > avg_volume[i] * 1
                and body > prev_body
            ):
                df.at[i, 'tag'] = '🟢'
            elif (
                row['open'] > row['close']
                and row['close'] <= row['low'] + (row['high'] - row['low']) * 0.1
                and row['volume'] > avg_volume[i] * 1.5
                and body > prev_body
            ):
                df.at[i, 'tag'] = '🔴'
            # Updated Buyer Absorption ⛔
            elif (
                row['high'] > prev['high']
                and row['close'] < prev['close']
                and (row['high'] - row['close']) > body
                and row['volume'] > avg_volume[i] * 1.5
                and next1['close'] < row['open']
                and next2['close'] < row['open']
            ):
                df.at[i, 'tag'] = '⛔'

            # Updated Seller Absorption 🚀
            elif (
                row['low'] < prev['low']
                and row['close'] > prev['close']
                and (row['close'] - row['low']) > body
                and row['volume'] > avg_volume[i] * 1.5
                and next1['close'] > row['close']
                and next2['close'] > row['close']
            ):
                df.at[i, 'tag'] = '🚀'
            # Detecting buyer absorption across multiple previous bullish candles (loop until condition is met)
            elif (
                row['close'] < df.iloc[i - 1]['open']  # Close below the previous bullish candle's open
                and row['volume'] > avg_volume[i] * 1.5
                and row['high'] > df.iloc[i - 1]['high']  # Current high exceeds previous bullish candle's high
                and next1['close'] < row['open']  # Check if next candle closes below current open
            ):
                df.at[i, 'tag'] = '⛔'  # Mark as Buyer Absorption

            # Detecting seller absorption across multiple previous bearish candles (loop until condition is met)
            elif (
                row['close'] > df.iloc[i - 1]['open']  # Close above the previous bearish candle's open
                and row['volume'] > avg_volume[i] * 1.5
                and row['low'] < df.iloc[i - 1]['low']  # Current low is below previous bearish candle's low
                and next1['close'] > row['close']  # Check if next candle closes above current close
            ):
                df.at[i, 'tag'] = '🚀'  # Mark as Seller Absorption

            elif (
                row['high'] > max(df['high'].iloc[i - 3:i])
                and row['volume'] > avg_volume[i] * 1.8
            ):
                if not (df['tag'].iloc[i - 3:i] == '💥').any():
                    df.at[i, 'tag'] = '💥'
            elif (
                row['low'] < min(df['low'].iloc[i - 3:i])
                and row['volume'] > avg_volume[i] * 1.8
            ):
                if not (df['tag'].iloc[i - 3:i] == '💣').any():
                    df.at[i, 'tag'] = '💣'
            elif (
                row['close'] > row['open']
                and body > (row['high'] - row['low']) * 0.7
                and row['volume'] > avg_volume[i] * 2
            ):
                df.at[i, 'tag'] = '🐂'
            elif (
                row['open'] > row['close']
                and body > (row['high'] - row['low']) * 0.7
                and row['volume'] > avg_volume[i] * 2
            ):
                df.at[i, 'tag'] = '🐻'
            elif (
                row['close'] > row['open']
                and body < 0.3 * prev_body
                and row['volume'] < avg_volume[i] * 1
            ):
                df.at[i, 'tag'] = '📉'   
            elif (
                row['open'] > row['close']
                and body < 0.3 * prev_body
                and row['volume'] < avg_volume[i] * 1.5
            ):
                df.at[i, 'tag'] = '📈'

        # --- Filter tags ---
        tags_available = df['tag'].unique()
        tags_available = [tag for tag in tags_available if tag]

        selected_tags = st.multiselect(
            "Select Signal(s) to View",
            options=tags_available,
            default=tags_available
        )

        # --- Plotting Chart ---
        fig = go.Figure()

        # Create the chart with dynamic X-axis based on price
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='lightblue', width=2),
            hovertext=df['close'],  # Show price on hover
            hoverinfo="x+y+text"  # Correct hover info format
        ))

        # Define full tag names
        tag_labels = {
            '🟢': '🟢 Aggressive Buyers',
            '🔴': '🔴 Aggressive Sellers',
            '⛔': '⛔ Buyer Absorption',
            '🚀': '🚀 Seller Absorption',
            '💥': '💥 Bullish POR',
            '💣': '💣 Bearish POR',
            '🐂': '🐂 Bullish POI',
            '🐻': '🐻 Bearish POI',
            '📉': '📉 Bullish weak legs',
            '📈': '📈 Bearish weak legs'
        }

        for tag in selected_tags:
            subset = df[df['tag'] == tag]

            fig.add_trace(go.Scatter(
                x=subset['date'],
                y=subset['close'],
                mode='markers+text',
                name=tag_labels.get(tag, tag),  # Show full name in legend
                text=[tag]*len(subset),
                textposition='top center',
                textfont=dict(size=20),
                marker=dict(size=14, symbol="circle", color='white'),
                customdata=subset[['open', 'high', 'low', 'close']].values,
                hovertemplate=(
                    "📅 Date: %{x|%Y-%m-%d}<br>" +
                    "🟢 Open: %{customdata[0]}<br>" +
                    "📈 High: %{customdata[1]}<br>" +
                    "📉 Low: %{customdata[2]}<br>" +
                    "🔚 Close: %{customdata[3]}<br>" +
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
                tickmode="array",
                tickvals=df['date'][::int(len(df['date']) / 10)],  # Reduce number of ticks
                ticktext=[f"{date.strftime('%Y-%m-%d')}" for date in df['date'][::int(len(df['date']) / 10)]],
                tickangle=-45,  # Rotate tick labels for better readability
                showgrid=False
            ),
            yaxis=dict(
                title="Price",
                showgrid=True,
                gridcolor="gray",
                zeroline=True,
                zerolinecolor="gray",
                ticks="outside",
                ticklen=5,
                tickwidth=2
            ),
            margin=dict(l=50, r=50, b=150, t=50),  # Make space for x-axis labels
        )
        fig.update_layout(
            height=800,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            legend=dict(font=dict(size=14)),
            title="Smart Money Signals Chart",
            xaxis=dict(
                title="Date",
                tickmode="array",
                tickvals=df['date'][::int(len(df['date']) / 10)],  # Reduce number of ticks
                ticktext=[f"{date.strftime('%Y-%m-%d')}" for date in df['date'][::int(len(df['date']) / 10)]],
                tickangle=-45,  # Rotate tick labels for better readability
                showgrid=False
            ),
            yaxis=dict(
                title="Price",
                showgrid=False,
                zeroline=True,
                zerolinecolor="gray",
                ticks="outside",
                ticklen=5,
                tickwidth=2,
                dtick=20  # Set the y-axis interval to 20
            ),
            margin=dict(l=50, r=50, b=150, t=50),  # Make space for x-axis labels
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Table for last 1 month signals ---
        st.subheader("📋 Recent 1 Month Signal Observed")
        last_date = df['date'].max()
        one_month_ago = last_date - timedelta(days=30)
        recent_df = df[(df['date'] >= one_month_ago) & (df['tag'] != '')]

        st.dataframe(recent_df[['date', 'open', 'high', 'low', 'close', 'volume', 'tag']].sort_values('date', ascending=False))

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
