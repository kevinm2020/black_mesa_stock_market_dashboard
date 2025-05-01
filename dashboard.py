
#libraries
import streamlit as st          #interactive web application
import yfinance as yf           #python lin for historical market data
import plotly.express as px     #lib for data vizulization
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage
from datetime import datetime
import time
import requests #for the news feature
import sqlite3          #databse alerts?
from streamlit_autorefresh import st_autorefresh

from fredapi import Fred    #for economic and social indicators
from ta.momentum import RSIIndicator
from twilio.rest import Client
import threading
import random


conn = sqlite3.connect("alerts.db", check_same_thread=False)
cursor = conn.cursor()

#cursor.execute("DROP TABLE IF EXISTS alerts")
#cursor.execute("""
#CREATE TABLE alerts (
 #   id INTEGER PRIMARY KEY AUTOINCREMENT,
  #  ticker TEXT,
   # type TEXT,
    #threshold REAL,
    #direction TEXT,
    #email TEXT,
    #phone TEXT,
    #timestamp TEXT,
    #status TEXT DEFAULT 'pending'
#)
#""")
#conn.commit()
#print("✅ alerts table created.")

cursor.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    type TEXT,
    threshold REAL,
    direction TEXT,
    email TEXT,
    phone TEXT,
    timestamp TEXT,
    status TEXT DEFAULT 'pending'
)
""")



####------------------------------------------------------Send Alerts Function------------------------------

#--------------------------------------------------------Send Email Function--------------
def send_email_notification(to_email, ticker, alert_type, current_value, threshold):
        print(f"📧 Sending email to {to_email}")

        msg = EmailMessage()
        msg["Subject"] = f"📈 {ticker} Alert Triggered!"
        msg["From"] = "black.mesa.softwar3@gmail.com"
        msg["To"] = to_email

        msg.set_content(
            f"""
            Hello,

            Your alert for {ticker} has been triggered.

            Alert Type: {alert_type}
            Current Value: {current_value}
            Threshold: {threshold}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            Visit your dashboard for more info.

            - Black Mesa Alert System
            """
        )

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login("black.mesa.softwar3@gmail.com", "opdj licp cgia ohnw")
                smtp.send_message(msg)
                print(f"✅ Email sent to {to_email}")
        except Exception as e:
            print(f"❌ Email failed: {e}")

#-------------------------------------------Send SMS Function--------------------------------------

account_sid = "ACccee4f046cca24527a11e18fbd652507"
auth_token = "5d18d25f474eb892d414698206767c02"
twilio_number = "+18887732224"

def send_sms_notification(to_phone, ticker, alert_type, current_value, threshold):
    print(f"!!!!!!!  SENDING SMS TO {to_phone}")
    try:
        client = Client(account_sid, auth_token)
        body = "f{ticker} ALERT: {alert_type} triggered.\nValue: {current_value}\nThreshold: {threshold}"
        message = client.messages.create(
            body="Your stock alert has been triggered!",
            from_= twilio_number,
            to=to_phone
        )
        print(f"!!!SMS Sent: {message.sid}")
    except Exception as e:
        print(f"!!!SMS Failed: {e}")



#-----------------------------------------End SMS Function------------------------------------------

#https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key=4de30e46287d5d259fd7e0901ef91c59&file_type=json

#BACKEND OFFICES


#auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="datarefresh")
#limit=None refresh indefinitely, key is used to refresh identity


#FRONT END OFFICES

#with style
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Page mode toggle
st.sidebar.title("🔀 Navigation")
mode = st.sidebar.selectbox("Select Mode:", ["User", "Admin"])


#-----------------------------------------------------------------user mode----------

if mode == "User":

    st.title("Stock Market Dashboard")
    st.write("by BLACK MESA SOFTWAR3")
    st.write("Last Fix: May 1 at 01:08PM CT")
   

#--------------------------------------------------------------------Economic Indicators Start-------------------------------
    FRED_API_KEY = "4de30e46287d5d259fd7e0901ef91c59"
    fred = Fred(api_key=FRED_API_KEY)

    # Safe data fetching
    try:
        unemployment = fred.get_series("UNRATE")
        df_unemp = unemployment.reset_index()
    except Exception as e:
        st.error(f"Failed to fetch unemployment data: {e}")

    try:
        retail_sales = fred.get_series("RSAFS")
    except Exception as e:
        st.error(f"Failed to fetch retail sales data: {e}")

    try:
        car_sales = fred.get_series("TOTALSA")
    except Exception as e:
        st.error(f"Failed to fetch car sales data: {e}")

    try:
        gas_price = fred.get_series("GASREGW")
    except Exception as e:
        st.error(f"Failed to fetch gas price data: {e}")

    #convert to dataDrames
    df_unemp = unemployment.reset_index()
    df_unemp.columns = ['Date', 'Unemployment Rate']

    df_retail = retail_sales.reset_index()
    df_retail.columns = ['Date', 'Retail Sales']

    df_car = car_sales.reset_index()
    df_car.columns = ['Date', 'Car Sales']

    df_gas = gas_price.reset_index()
    df_gas.columns = ['Date', 'Gas Price']


    #--Get latest & previous values

    def get_metric_change(df, column):
        latest = df.iloc[-1][column]
        previous = df.iloc[-2][column]
        delta = latest - previous
        return latest, delta
    
    #calculate changes
    unemp_val, unemp_delta = get_metric_change(df_unemp, "Unemployment Rate")
    retail_val, retail_delta = get_metric_change(df_retail, "Retail Sales")
    gas_val, gas_delta = get_metric_change(df_gas, "Gas Price")
    car_val, car_delta = get_metric_change(df_car, "Car Sales")

    st.title("🇺🇸 U.S Economic Idicators")
    st.write("Source: Federal Reserve Back of St.Louis (Past 2 Months)")

    #display metrics in one row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📉 Unemployment Rate", f"{unemp_val:.1f}%", f"{unemp_delta:+.2f}%")

    with col2:
        st.metric("🛍️ Retail Sales", f"${retail_val:.0f}M", f"{retail_delta:+,.0f}")

    with col3:
        st.metric("⛽ Avg Gas Price", f"${gas_val:.2f}/gal", f"{gas_delta:+.2f}")

    with col4:
        st.metric("🚗 Car Sales", f"{car_val:.2f}M units", f"{car_delta:+.2f}")

    # Optional: full chart section
    with st.expander("📈 See Historical Trends"):
        st.line_chart(df_unemp.set_index("Date")["Unemployment Rate"])
        st.line_chart(df_retail.set_index("Date")["Retail Sales"])
        st.line_chart(df_gas.set_index("Date")["Gas Price"])
        st.line_chart(df_car.set_index("Date")["Car Sales"])


#--------------------------------------------------------------------End Economic Indicators End-------------------------------

#--------------------------------------------------------------------Sector Overview Feature---------------------------------------------------------

    #--------------------------------------------------------------------Sector Overview Feature---------------------------------------------------------

   
    # Utility: robust batch download with retries
    def robust_download(tickers, period, max_retries=3, backoff=2):
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers,
                    period=period,
                    group_by='ticker',
                    threads=True,
                    progress=False,
                    timeout=10
                )
                return data
            except YFRateLimitError:
                wait = backoff ** attempt
                st.warning(f"Rate limit hit, retrying in {wait}s…")
                time.sleep(wait)
            except Exception as e:
                st.error(f"Download error: {e}")
                break
        st.error("Failed to download data after multiple attempts.")
        return pd.DataFrame()

    # === Sector Overview ===
    @st.cache_data(ttl=600)
    def get_sector_data(all_tickers, period):
        return robust_download(all_tickers, period)

    def sector_overview():

        st.title("📊 SECTORS OVERVIEW")
        st.write("Watch the sectors")

        sector_tickers = {
            "Technology": ["AAPL", "MSFT", "NVDA"],
            "Consumer Discretionary": ["TSLA", "AMZN"],
            "Financials": ["JPM", "BAC"],
            "Energy": ["XOM", "CVX"],
            "Healthcare": ["JNJ", "PFE"],
            "Industrials": ["BA", "CAT"]
        }

        st.subheader("Sector Performance")
        period = st.selectbox("Select time range:", ["1d", "5d", "1mo"], index=1)
        all_tickers = [t for lst in sector_tickers.values() for t in lst]

        with st.spinner("Loading sector data..."):
            all_data = get_sector_data(all_tickers, period)

        # calculate avg change
        performance = []
        for sector, tickers in sector_tickers.items():
            changes = []
            for t in tickers:
                df = all_data.get(t)
                if df is None or df.empty:
                    st.warning(f"No data for {t}")
                    continue
                o, c = df["Open"].iloc[0], df["Close"].iloc[-1]
                changes.append((c - o) / o * 100)
            if changes:
                performance.append({"Sector": sector, "Change": round(sum(changes)/len(changes),2)})

        cols = st.columns(max(len(performance),1))
        st.subheader(f"Sector Performance Summary ({period})")
        for i, row in enumerate(performance):
            emoji = "📈" if row["Change"]>0 else "📉"
            val = f"{row['Change']:+.2f}%"
            cols[i].metric(f"{emoji} {row['Sector']}", val, val)

        dfp = pd.DataFrame(performance)
        if not dfp.empty and "Change" in dfp.columns:
            sorted_df = dfp.sort_values("Change", ascending=False)
            gainers, losers = sorted_df.head(3), sorted_df.tail(3)
            st.subheader("Top Sector Movers")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("### 📈 Gainers")
                for _,r in gainers.iterrows():
                    st.markdown(f"- **{r.Sector}**: {r.Change:+.2f}%")
            with c2:
                st.markdown("### 📉 Losers")
                for _,r in losers.iterrows():
                    st.markdown(f"- **{r.Sector}**: {r.Change:+.2f}%")
        else:
            st.warning("No sector performance data available.")

        st.subheader("Sector Trend Comparison")
        trends = pd.DataFrame()
        for sector, tickers in sector_tickers.items():
            df_close = pd.DataFrame()
            for t in tickers:
                df = all_data.get(t)
                if df is not None and not df.empty:
                    df_close[t] = df["Close"]
            if not df_close.empty:
                norm = df_close/df_close.iloc[0]*100
                trends[sector] = norm.mean(axis=1)

        if not trends.empty:
            fig = px.line(trends, x=trends.index, y=trends.columns,
                        title=f"Sector Performance Over {period.upper()}",
                        labels={"value":"Normalized Price","index":"Date"})
            fig.update_layout(template="plotly_white", legend_title_text="Sector")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No sector trend data available.")

    sector_overview()

    # === Top 10 Stocks ===
    @st.cache_data(ttl=600)
    def get_top10(tickers):
        data = robust_download(tickers, "1d")
        result = []
        for t in tickers:
            df = data.get(t)
            if df is None or df.empty:
                st.warning(f"No data for {t}")
                continue
            o,c = df["Open"].iloc[0], df["Close"].iloc[-1]
            result.append({"Ticker": t, "Change": round((c-o)/o*100,2)})
        return result

    st.title("🏅 Top 10 Best Performing Stocks Today")
    st.write("Here are today’s top 10 by percent change:")
    tickers = ['AAPL','TSLA','AMZN','GOOGL','MSFT','SPY','NFLX','NVDA','META','AMD']
    with st.spinner("Loading top stocks..."):
        perf = get_top10(tickers)

    if perf:
        df10 = pd.DataFrame(sorted(perf, key=lambda x: x["Change"], reverse=True)[:10])
        fig2 = px.bar(df10, x="Ticker", y="Change",
                    title="Top 10 Stocks Today",
                    labels={"Change":"% Change"},
                    color="Change", color_continuous_scale="Viridis")
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("No stock performance data available.")

    st.caption("Source: Yahoo Finance")


    # === Single Ticker Evaluation ===
    @st.cache_data(ttl=600)
    def load_ticker(ticker):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period="1y")
            info = tk.info
            return df, info
        except:
            return pd.DataFrame(), {}

    st.title("📈 Single Ticker Evaluation")
    user = st.text_input("Enter ticker:", "AAPL")
    user_ticker = user
    if user:
        with st.spinner(f"Loading {user} data..."):
            df_t, info = load_ticker(user)

        if df_t.empty:
            st.error(f"Could not fetch data for {user}. Please try again later.")
        else:
            st.subheader(f"{user} Price & Volume")
            fig3 = make_subplots(rows=2,cols=1,shared_xaxes=True,
                                vertical_spacing=0.1,
                                subplot_titles=("Closing Price","Volume"))
            fig3.add_trace(go.Scatter(x=df_t.index,y=df_t["Close"],name="Price"),row=1,col=1)
            fig3.add_trace(go.Bar(x=df_t.index,y=df_t["Volume"],name="Volume"),row=2,col=1)
            fig3.update_layout(height=700,template="plotly_white")
            c1,c2 = st.columns([2,1])
            with c1:
                st.plotly_chart(fig3,use_container_width=True)
            with c2:
                st.subheader("Key Indicators")
                try:
                    price = df_t["Close"].iloc[-1]
                    rsi = RSIIndicator(df_t["Close"]).rsi().iloc[-1]
                    vol = df_t["Volume"].iloc[-1]
                    ma50 = df_t["Close"].rolling(50).mean().iloc[-1]
                    ma200= df_t["Close"].rolling(200).mean().iloc[-1]
                    pe = info.get("trailingPE","N/A")
                    beta= info.get("beta","N/A")
                    st.metric("Price",f"${price:.2f}")
                    st.metric("RSI",f"{rsi:.2f}")
                    st.metric("Volume",f"{vol:,}")
                    st.metric("50-Day MA",f"${ma50:.2f}")
                    st.metric("200-Day MA",f"${ma200:.2f}")
                    st.metric("P/E Ratio",pe)
                    st.metric("Beta",beta)
                except Exception as e:
                    st.warning(f"Indicator error: {e}")
            st.caption("Source: Yahoo Finance via yfinance")
#---------------------------------------------------Volume Feature end-----------------------------------------

#------------------------------------------------------------------------News Feature Start-----------------------------------


    # 🔒 Replace this with your actual Finnhub API key
    FINNHUB_API_KEY = "cvrb3r1r01qp88cpcn7gcvrb3r1r01qp88cpcn80"

    # ---------------- News Page ----------------
    st.title("📰 Stock News")

    # Let user pick a ticker or reuse one they've searched
    news_ticker = user_ticker

    if news_ticker:
        st.write(f"Latest news for **{news_ticker.upper()}**")

        # Construct Finnhub API request
        url = f"https://finnhub.io/api/v1/company-news?symbol={news_ticker.upper()}&from=2024-01-01&to=2025-12-31&token=cvrb3r1r01qp88cpcn7gcvrb3r1r01qp88cpcn80"

        
        response = requests.get(url)
        
        if response.status_code == 200:
            news_data = response.json()

            if news_data:
                st.subheader("🗞️ Top News Headlines")

                # Create 3 horizontal columns
                col1, col2, col3 = st.columns(3)

                # Loop through first 3 articles and assign to each column
                for i, article in enumerate(news_data[:3]):
                    with [col1, col2, col3][i]:
                        st.markdown(f"**{article['headline']}**")
                        st.markdown(f"🕒 {datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d %H:%M')}")
                        st.markdown(f"📎 [Read more]({article['url']})")
            else:
                st.info("No news articles found.")
        else:
            st.error("Failed to fetch news. Please check your API key or ticker.")



    #------------------------------------------------------------------------End news feature-----------------------------------------

    #-----------------------------------------------------------------Start News Sentiment Feature-----------------------------------------

    #feature needs to be paid for

    #ticker = ticker
    #url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token=cvrb3r1r01qp88cpcn7gcvrb3r1r01qp88cpcn80"
    #res = requests.get(url)

    #if res.status_code == 200:
        #data = res.json()
        #if "sentiment" in data and "bullishPercent" in data["sentiment"]:
            #bullish = data['sentiment']['bullishPercent']
            #bearish = data['sentiment']['bearishPercent']
            #buzz = data['buzz']['articlesInLastWeek']

            #st.subheader(f"🧠 Sentiment for {ticker.upper()}")
            #st.write(f"**Bullish Sentiment:** {bullish}%")
            #st.write(f"**Bearish Sentiment:** {bearish}%")
            #st.write(f"**Buzz Score:** {buzz} articles this week")
        #else:
            #st.warning("⚠️ Sentiment data not available for this ticker.")
    #else:
        #st.error(f"Failed to fetch sentiment data. Status code: {res.status_code}")


    #-----------------------------------------------------------------End News Sentiment Feature -----------------------------------------
    
    


    #-------------------------------------------------Analysts Concensus------------------------------

    st.write("                                    ")
    st.write("                                    ")
    st.title("🔎 Analyst Concensus")
    st.write("Buy? Hold? Sell? - See what the experts are saying")
    ticker = user_ticker

    FINNHUB_API_KEY = "cvrb3r1r01qp88cpcn7gcvrb3r1r01qp88cpcn80"

    url =  f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token=cvrb3r1r01qp88cpcn7gcvrb3r1r01qp88cpcn80" 
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data:
            latest = data[0]
            st.subheader(f"🧭 Analyst Ratings for {ticker.upper()} ({latest['period']})")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("📈 Buy", latest['buy'])

            with col2:
                st.metric("📊 Hold", latest['hold'])

            with col3:
                st.metric("📉 Sell", latest['sell'])

            with col4:
                st.metric("✅ Strong Buy", latest['strongBuy'])

            with col5:
                st.metric("❌ Strong Sell", latest['strongSell'])


            import plotly.graph_objects as go
            
            score = latest['strongBuy'] * 2 + latest['buy'] - latest['sell'] - latest['strongSell']

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                title = {'text': f"{ticker.upper()} Sentiment Score"},
                gauge = {
                    'axis': {'range': [-10, 10]},
                    'bar': {'color': "green" if score > 0 else "red"},
                    'steps': [
                        {'range': [-10, -5], 'color': "maroon"},
                        {'range': [-5, 0], 'color': "orange"},
                        {'range': [0, 5], 'color': "lightgreen"},
                        {'range': [5, 10], 'color': "green"}
                    ]
                }
            ))

            st.plotly_chart(fig)

            # Interpret the sentiment score
            if score >= 40:
                sentiment_label = "🔰 Strong Buy — Analysts are highly bullish."
            elif score >= 20:
                sentiment_label = "🟢 Buy — Most analysts lean positive."
            elif score > -20:
                sentiment_label = "🟡 Hold — Mixed sentiment or cautious outlook."
            elif score > -40:
                sentiment_label = "🔴 Sell — Analysts see downside risk."
            else:
                sentiment_label = "⚠️ Strong Sell — Broad consensus to avoid or exit."

            # Display interpretation
            st.markdown(f"### Analyst Consensus: **{sentiment_label}**")

            st.write("Source: Finhub")

        else:
            st.info("No analyst recommendations available.")
    else:
        st.error("Error fetching recommendation data.")
        
        

    

    #-------------------------------------------------Analysyt Consensus Feature End------------------------------
    

    #----------------------------------------------------Price Alert Notification-------------------------------------
    
    #-------------Alert UI-------------------
    st.write("                                    ")
    st.write("                                    ")
    st.header("🔔 Create a Stock Alert")
    ticker = st.text_input("Enter Ticker:", "AAPL").upper()

    alert_type = st.selectbox("Alert Type",[
        "Price Threshold",
        "R.S.I",
        "Volume Spike",
        "Moving Average Crossover",
        "Price/Earnings Ratio"    
                              ])
    
    alert_info = {
    "Price Threshold": "Alerts you when the stock price crosses a specific value.",
    "R.S.I": "Tracks Relative Strength Index, alerts when cross a specific value: 0-30 oversold (too cold - might rise soon) |  30-70 Neutal/Stable | 70-100 Overbought (too hot - might drop soon) ",
    "Volume Spike": "Detects when trading volume sharply increases : Current Volume > 2 x 30dayAvgVolume ",
    "Moving Average Crossover": "Checks when short-term trends cross long-term ones : 50dayMA > 200dayMA",
    "Price/Earnings Ratio": "Monitors if the Price-to-Earnings crosses a specific value: <0 (Possible trouble or no profit) | 10-15 (undevalued/slow company growth) | 30-50+ (strong future growth (e.g tech))"
    }

    if alert_type:
        st.info(f"ℹ️ {alert_info.get(alert_type)}")
    
    direction = None
    threshold = None
    
    #show threshold/direction field only for certain types of alerts
    if alert_type in ["Price Threshold", "R.S.I", "Price/Earnings Ratio"]:
        direction = st.selectbox("Alert when value is:", ["Above", "Below"])
        threshold = st.number_input("Alert if value crosses:", min_value=0.0)
    
    #show email and sms for all alert types
    email = st.text_input("Your Email:")
    phone = st.text_input("Your Phone Number (optional):")
    
    #submit alert
    if st.button("Add Alert"):
        timestamp = datetime.now().isoformat()

        cursor.execute("""
        INSERT INTO alerts (ticker, type, threshold, direction, email, phone, timestamp, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, alert_type, threshold, direction, email, phone, timestamp, "pending"))

        conn.commit()
        st.success(f"{alert_type} alert set for {ticker} ({direction or 'N/A'})")


    st.write("                                    ")
    st.write("                                    ")


    #-------------Alert UI-------------------
    #------------Poll & Trigger Alerts--------

    
    def check_alerts():

        print(">>> check_alerts() is executing")

        # Open a NEW connection for this thread
        conn = sqlite3.connect("alerts.db")
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM alerts WHERE status='pending'")
            alerts = cursor.fetchall()
            print(f"📋 {len(alerts)} pending alerts found.")
        except Exception as e:
            print(f"⚠️ DB query error: {e}")
            conn.close()
            return

        for alert in alerts:
            id, ticker, alert_type, threshold, direction, email, phone, status, timestamp = alert
            triggered = False
            current_value = None
            message = ""

            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d", interval="1m")

                if data.empty:
                    print(f"⚠️ No data found for {ticker}")
                    continue

                if alert_type == "Price Threshold":
                    current_price = data["Close"].iloc[-1]
                    current_value = current_price
                    if (direction == "Above" and current_value >= threshold) or \
                    (direction == "Below" and current_value <= threshold):
                        triggered = True
                        message = f"{ticker} hit ${current_value:.2f} ({direction} {threshold})"

                elif alert_type == "RSI":
                    rsi = RSIIndicator(data["Close"]).rsi().iloc[-1]
                    current_value = rsi
                    if (direction == "Above" and rsi >= threshold) or \
                    (direction == "Below" and rsi <= threshold):
                        triggered = True
                        message = f"{ticker} RSI is {rsi:.2f} ({direction} {threshold})"

                elif alert_type == "Volume Spike":
                    curr_vol = data["Volume"].iloc[-1]
                    avg_vol = data["Volume"].rolling(window=30).mean().iloc[-2]
                    current_value = curr_vol
                    if curr_vol > 2 * avg_vol:
                        triggered = True
                        message = f"{ticker} volume spike: {curr_vol:.0f} vs avg {avg_vol:.0f}"

                elif alert_type == "Moving Average Crossover":
                    ma50 = data["Close"].rolling(window=50).mean().iloc[-1]
                    ma200 = data["Close"].rolling(window=200).mean().iloc[-1]
                    current_value = ma50
                    if ma50 > ma200:
                        triggered = True
                        message = f"{ticker}: 50-day MA ({ma50:.2f}) crossed above 200-day MA ({ma200:.2f})"

                elif alert_type == "P/E Ratio":
                    pe = stock.info.get("trailingPE")
                    current_value = pe
                    if pe and ((direction == "Above" and pe > threshold) or \
                            (direction == "Below" and pe < threshold)):
                        triggered = True
                        message = f"{ticker} P/E is {pe:.2f} ({direction} {threshold})"

                # 🔔 If alert triggered
                if triggered:
                    print(f"✅ Alert triggered for {ticker}: {alert_type} - {message}")

                    if email:
                        print(f"📧 Sending email to: {email}")
                        send_email_notification(email, ticker, alert_type, current_value, f"{direction} {threshold}")

                    if phone:
                        print(f"📱 Sending SMS to: {phone}")
                        send_sms_notification(phone, ticker, alert_type, current_value, f"{direction} {threshold}")

                    cursor.execute("UPDATE alerts SET status='triggered' WHERE id=?", (id,))
                    conn.commit()

                    # Optional UI notification (Streamlit toast if in Streamlit)
                    try:
                        import streamlit as st
                        st.toast(f"Alert sent for {ticker} ({alert_type})")
                    except:
                        pass

            except Exception as e:
                print(f"⚠️ Error processing {ticker}: {e}")

            # 🕒 Rate limit delay
            time.sleep(random.uniform(1.5, 3.0))

        conn.close()  


#-------------------------------button-------------------------------------------------------------------

    st.subheader("Ready to take the next step in Financial Planning: 401(k), Brokerage, Tax Mitigation, Retirment?")
    st.write("                   ")
    if st.button("📅 Book a Free Financial Advisor Consultation"):
        st.markdown("[Click here to schedule a 15-minute call](https://calendly.com/black-mesa-softwar3)", unsafe_allow_html=True)


#-----------------------------Credits------------------------------------------------

    st.write("                                    ")
    st.write("                                    ")
    st.subheader("Credits")
    st.write("Devloped by Kevin Martinez - 2025")
    st.write("Contact Email : black.mesa.softwar3@gmail.com")
    st.write("Last Fix: April 30/25 11:09PM CT")
    
#----------------------------------------end user mode-------------------------------------------

#---------------------------------------Start Admin Mode------------------------------------
if mode == "Admin":
    st.title("🔐 Admin Panel")

    admin_key = st.text_input("Enter Admin Access Key:", type="password")
    if admin_key == "blackmesa42":

        st.success("Access Granted ✔️")

        cursor.execute("SELECT * FROM alerts")
        rows = cursor.fetchall()

        columns = ["ID", "Ticker", "Type", "Threshold", "Direction", "Email", "Phone", "Timestamp", "Status"]

        df = pd.DataFrame(rows, columns=columns)

        st.dataframe(df)

        # Download CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Alerts CSV", data=csv_data, file_name="alerts.csv", mime="text/csv")
#-------------------end Admin mode---------------------------

def run_alert_checker():
    while True:
        check_alerts()
        time.sleep(60)

if 'alert_thread' not in st.session_state:
    st.session_state.alert_thread = threading.Thread(target=run_alert_checker)
    st.session_state.alert_thread.daemon = True
    st.session_state.alert_thread.start()

#Terminal Commands
#activate virtual enviroment: source venv/bin/activate
#run the program: streamlit run dashboard.py
#testing logic python dashboard.py

#push to github
#git add .
#git commit -m "message"
#git push

"""

Versiion 1.0 
Last Update: May 1 - 2025  at 09:57AM CT  

"""