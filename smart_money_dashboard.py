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

    if required_cols.issubset(df.columns):
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # --- Signal Tagging Logic (add your logic here)
        # Example placeholder: remove or replace with real logic
        if 'tag' not in df.columns:
            df['tag'] = ''  # Add empty tag column to avoid errors

        # --- Tag selection UI ---
        if 'tag' in df.columns:
            tags_available = df['tag'].dropna().unique()
            tags_available = [tag for tag in tags_available if tag]

            with st.container():
                st.subheader("📌 Select Signal(s) to View")
                selected_tags = st.multiselect(
                    "Choose from available signals",
                    options=tags_available,
                    default=tags_available,
                    key="tag_selector"
                )

            # --- Plotting Chart ---
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df['date'], y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='lightblue', width=2),
                hovertext=df['close'],
                hoverinfo="x+y+text"
            ))

            tag_labels = {
                '🟢': '🟢 Aggressive Buyers',
                '🔴': '🔴 Aggressive Sellers',
                '⛔': '⛔ Buyer Absorption',
                '🚀': '🚀 Seller Absorption',
                '💥': '💥 Bullish POR',
                '💣': '💣 Bearish POR',
                '🐂': '🐂 Bullish POI',
                '🐻': '🐻 Bearish POI',
                '📉': '📉 Bullish Weak Legs',
                '📈': '📈 Bearish Weak Legs'
            }

            for tag in selected_tags:
                subset = df[df['tag'] == tag]
                fig.add_trace(go.Scatter(
                    x=subset['date'],
                    y=subset['close'],
                    mode='markers+text',
                    name=tag_labels.get(tag, tag),
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
                legend=dict(font=dict(size=14), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                title="Smart Money Signals Chart",
                xaxis=dict(
                    title="Date",
                    tickmode="auto",
                    tickangle=-45,
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
                    dtick=20
                ),
                margin=dict(l=50, r=50, b=150, t=50)
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- Table of last 1-month signals ---
            st.subheader("📋 Signals from the Last 1 Month")
            last_date = df['date'].max()
            one_month_ago = last_date - timedelta(days=30)
            recent_df = df[(df['date'] >= one_month_ago) & (df['tag'] != '')]

            st.dataframe(
                recent_df[['date', 'open', 'high', 'low', 'close', 'volume', 'tag']].sort_values('date', ascending=False),
                use_container_width=True
            )

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
            st.warning("⚠️ No 'tag' column found in the data. Please upload a file that includes tagged signals.")
    else:
        st.error("❌ Missing required columns: date, open, high, low, close, volume")
