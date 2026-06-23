import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Indian Stock ARIMA Forecaster", layout="wide")
st.title("📈 Indian Stock Price Forecasting Dashboard (ARIMA)")
st.write("This application downloads the last 5 years of historical data from Yahoo Finance and forecasts stock prices up to **June 2027**.")

# 1. Indian Stock Selector Layout
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter NSE Stock Ticker (e.g., RELIANCE, TCS, INFYS, HDFCBANK):", value="RELIANCE").upper().strip()

# Handle standard NSE ticker suffix formatting for Yahoo Finance
if ticker_input and not ticker_input.endswith(".NS"):
    ticker = f"{ticker_input}.NS"
else:
    ticker = ticker_input

# Define Timeline boundaries
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

if ticker:
    st.markdown(f"## Fetching data for: **{ticker_input}**")
    
    with st.spinner("Downloading 5 years of historical data from Yahoo Finance..."):
        # Fetch data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
    if stock_data.empty:
        st.error(f"Could not find data for ticker '{ticker_input}'. Please make sure it is a valid NSE stock ticker symbol.")
    else:
        # Preprocess data to get a clean weekly or monthly series to speed up cloud processing and improve ARIMA convergence
        # We will use Weekly Resampling ('W-MON') of the Closing price
        df_close = stock_data['Close'].resample('W-MON').mean().dropna()
        
        st.success("Data successfully downloaded!")
        
        # Display layout split
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("💡 Historical Trend (Last 5 Years)")
            st.line_chart(df_close)
            
        with col2:
            st.subheader("📋 Recent Data Snapshot")
            st.dataframe(stock_data['Close'].tail(10), use_container_width=True)
            
        # 2. ARIMA Modeling & Forecasting Logic
        st.markdown("---")
        st.subheader("🤖 ARIMA Forecasting Model execution")
        
        with st.spinner("Running Auto-ARIMA optimization algorithms... (This may take a moment)"):
            try:
                # Find optimal (p,d,q) parameters automatically
                model = auto_arima(df_close, seasonal=False, error_action='ignore', suppress_warnings=True)
                fitted_model = model.fit(df_close)
                
                # Calculate steps needed to reach June 30, 2027
                last_date = df_close.index[-1]
                target_date = datetime(2027, 6, 30)
                
                # Calculate required weekly steps between last data point and June 2027
                weeks_diff = int((target_date - last_date).days / 7) + 1
                
                # Generate Forecast
                forecast_values, conf_int = fitted_model.predict(n_periods=weeks_diff, return_conf_int=True)
                
                # Construct forecast index timeline
                forecast_index = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=weeks_diff, freq='W-MON')
                forecast_series = pd.Series(forecast_values, index=forecast_index)
                
                # Isolate target June 2027 data frame structures
                june_2027_forecast = forecast_series[forecast_series.index.month == 6]
                june_2027_forecast = june_2027_forecast[june_2027_forecast.index.year == 2027]
                
                # Plotting execution
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_close.index, df_close.values, label="Historical Data", color="blue")
                ax.plot(forecast_series.index, forecast_series.values, label="ARIMA Forecast Track", color="red", linestyle="--")
                ax.fill_between(forecast_series.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
                ax.set_title(f"{ticker_input} Price Projection up to June 2027 (ARIMA Order: {model.order})", fontsize=14)
                ax.set_xlabel("Timeline Year")
                ax.set_ylabel("Stock Price (INR)")
                ax.legend(loc="upper left")
                ax.grid(True, linestyle=":", alpha=0.6)
                
                # Render Graphic
                st.pyplot(fig)
                plt.close()
                
                # 3. Numeric Output Display
                st.markdown("---")
                st.subheader("🎯 Forecast Numbers for June 2027")
                
                if not june_2027_forecast.empty:
                    # Match confidence intervals explicitly for June 2027 indices
                    june_indices = [forecast_series.index.get_loc(idx) for idx in june_2027_forecast.index]
                    
                    display_df = pd.DataFrame({
                        'Date': june_2027_forecast.index.strftime('%Y-%m-%d'),
                        'Forecasted Price (INR)': june_2027_forecast.values.round(2),
                        'Lower Bound Range': conf_int[june_indices, 0].round(2),
                        'Upper Bound Range': conf_int[june_indices, 1].round(2)
                    }).set_index('Date')
                    
                    st.table(display_df)
                else:
                    st.info("No specific June 2027 calculation data frames could slice correctly. Displaying terminal forecast parameters:")
                    st.dataframe(forecast_series.tail(10))
                    
            except Exception as ex:
                st.error(f"Mathematical processing failed to converge constraints: {ex}")
