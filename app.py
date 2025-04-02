import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Streamlit App Title
st.title("📈 Stock Market Data Visualization & Prediction")

# Sidebar - User Input for Stock Symbol
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL, TSLA)", "AAPL")

# Date Selection
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
if not df.empty:
    # Fix MultiIndex Columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # Remove second level index

    st.write(f"### Stock Data for {stock_symbol}")
    st.dataframe(df.head())

end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-03-31"))

# Fetch Stock Data
if stock_symbol:
    df = yf.download(stock_symbol, start=start_date, end=end_date)

    if not df.empty:
        # 🔥 Fix Multi-Index Issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)  # Remove second-level column names

        st.write(f"### Stock Data for {stock_symbol}")
        st.dataframe(df.head())

        # Line Chart - Stock Price Over Time
        st.write("### Closing Price Over Time")
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close Price", color="blue")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

        # Interactive Candlestick Chart (Fixed!)
        st.write("### Candlestick Chart")
        fig = px.line(df, x=df.index, y=['Open', 'Close', 'High', 'Low'], title=f"{stock_symbol} Stock Prices")
        st.plotly_chart(fig)

        # Volume Chart
        st.write("### Trading Volume Over Time")
        fig, ax = plt.subplots()
        ax.bar(df.index, df["Volume"], color="green", alpha=0.6)
        ax.set_xlabel("Date")
        ax.set_ylabel("Volume")
        st.pyplot(fig)

        # Moving Averages
        df["20-Day MA"] = df["Close"].rolling(window=20).mean()
        df["50-Day MA"] = df["Close"].rolling(window=50).mean()

        st.write("### Moving Averages (20-Day & 50-Day)")
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close Price", color="blue", alpha=0.6)
        ax.plot(df.index, df["20-Day MA"], label="20-Day MA", color="red")
        ax.plot(df.index, df["50-Day MA"], label="50-Day MA", color="green")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

        # Machine Learning - Stock Price Prediction
        st.write("### Stock Price Prediction")
        df["Days"] = (df.index - df.index.min()).days
        X = df[["Days"]]
        y = df["Close"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        future_days = st.slider("Predict Future Price for Days Ahead", 1, 30, 5)
        future_price = model.predict([[df["Days"].max() + future_days]])

        st.write(f"📈 Predicted Price after {future_days} days: **${future_price[0]:.2f}**")

    else:
        st.warning("No data found! Please enter a valid stock symbol.")
    # Flatten MultiIndex Columns
df.columns = [col[0] for col in df.columns]  # Only keep the first level (Open, Close, etc.)

