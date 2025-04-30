import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from google.oauth2 import service_account
from gsheetsdb import connect

# --- Google Sheets Setup ---
# Create a connection object
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=["https://www.googleapis.com/auth/spreadsheets"],
)
conn = connect(credentials=credentials)

# --- Streamlit App ---
st.title("📈 Smart Money Visualizer - Google Sheets Data")

# --- User Inputs ---
st.sidebar.header("Google Sheets Configuration")
sheet_id = st.sidebar.text_input("Enter Google Sheet ID", 
                                help="The long ID in your Google Sheet URL (docs.google.com/spreadsheets/d/[THIS_IS_YOUR_SHEET_ID]/edit)")
sheet_name = st.sidebar.text_input("Enter Sheet Name", 
                                  help="The exact name of the worksheet/tab in your Google Sheet")
company_name = st.sidebar.text_input("Search for Company", 
                                    help="Enter the company name/ticker as it appears in your data")

if sheet_id and sheet_name and company_name:
    # --- Load data from Google Sheet ---
    @st.cache_data(ttl=600)
    def load_company_data(sheet_id, sheet_name, company):
        try:
            # Use the @st.cache_data decorator to cache the data
            query = f"""
                SELECT date, open, high, low, close, volume 
                FROM "https://docs.google.com/spreadsheets/d/{sheet_id}/edit#gid=0".{sheet_name}
                WHERE LOWER(company) = LOWER('{company}')
                ORDER BY date
            """
            rows = conn.execute(query, headers=1)
            df = pd.DataFrame(rows)
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    with st.spinner(f"Loading data for {company_name}..."):
        df = load_company_data(sheet_id, sheet_name, company_name)
    
    if df is not None:
        if not df.empty:
            # --- Data Cleaning ---
            try:
                # Convert columns to appropriate types
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                
                # Drop rows with invalid dates or missing values
                df = df.dropna(subset=['date'] + numeric_cols)
                df = df.sort_values('date').reset_index(drop=True)
                
                # --- Tagging logic ---
                df['tag'] = ''
                avg_volume = df['volume'].rolling(window=10).mean().fillna(0)

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
                fig.add_trace(go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC'
                ))

                # Add markers for tags
                for tag in df['tag'].unique():
                    if tag:
                        subset = df[df['tag'] == tag]
                        fig.add_trace(go.Scatter(
                            x=subset['date'],
                            y=subset['close'],
                            mode='markers',
                            name=tag,
                            marker=dict(
                                size=10,
                                symbol='diamond',
                                line=dict(width=2)
                            ),
                            text=tag,
                            hoverinfo='text+y'
                        ))

                fig.update_layout(
                    title=f"{company_name} - Smart Money Analysis",
                    xaxis_rangeslider_visible=False,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show recent data
                st.subheader("Recent Data with Tags")
                st.dataframe(df[['date', 'open', 'high', 'low', 'close', 'volume', 'tag']].tail(30))

            except Exception as e:
                st.error(f"Error processing data: {e}")
        else:
            st.warning(f"No data found for company: {company_name} in sheet '{sheet_name}'")

# --- Instructions ---
st.sidebar.markdown("""
### Instructions:
1. Enter your Google Sheet ID (from the URL)
2. Enter the exact Sheet/Tab name
3. Enter the company name/ticker to search
4. The app will fetch and analyze the data

### Google Sheet Requirements:
- Must have columns: date, open, high, low, close, volume, company
- Share the sheet with your service account email
- First row should contain headers

### Example Sheet URL:
`https://docs.google.com/spreadsheets/d/[SHEET_ID]/edit#gid=0`
""")