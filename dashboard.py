
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
import numpy as np
from fredapi import Fred    #for economic and social indicators
from ta.momentum import RSIIndicator
from twilio.rest import Client
import threading
import random
import requests


conn = sqlite3.connect("alerts.db", check_same_thread=False)
cursor = conn.cursor()

#cursor.execute("DROP TABLE IF EXISTS alerts")

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

#-------------------------------------------Credentials Section--------------------------------
account_sid = "ACccee4f046cca24527a11e18fbd652507"
auth_token = "5d18d25f474eb892d414698206767c02"
twilio_number = "+18887732224"
FMP_API_KEY = "ZaYQNqtexbx4FAkxSBTvrDegpa25FFv1"
BASE_URL = "https://financialmodelingprep.com/api/v3"
NEWS_API =  "79153b02c1e94dc0ad074dae21eb9aa2"
SPORTS_URL = f"https://newsapi.org/v2/top-headlines?category=sports&pageSize=5&language=en&apiKey={NEWS_API}"
MOVIE_NEWS_URL = (
    "https://newsapi.org/v2/everything?"
    "q=movie OR film OR box office OR streaming OR Netflix OR HBO OR Hulu OR Prime Video&"
    "language=en&"
    "pageSize=15&"
    "sortBy=publishedAt&"
    f"apiKey={NEWS_API}"
)

ENTERTAINMENT_KEYWORDS = [
    "movie", "film", "box office", "streaming", "netflix", "hulu", "hbo", "prime video", "cinema", "tv series", "trailer"
]

#-------------------------------------------End Credentials Section-------------------

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

if "mode" not in st.session_state:
    st.session_state.mode = "Home"

# Function to change mode
def set_mode(new_mode):
    st.session_state.mode = new_mode

with st.sidebar:
    st.markdown("Directory")
    if st.button("Home"):
        set_mode("Home")
    if st.button("401(k) Strategies"):
        set_mode("401(k) Strategies")
    if st.button("Culture and Capital"):
        set_mode("Culture and Capital")
    if st.button("Stock Database"):
        set_mode("Stock Database")
    if st.button("Market Insights"):
        set_mode("Market Insights")
    if st.button("Sector Performance"):
        set_mode("Sector Performance")
    if st.button("Stock Alerts"):
        set_mode("Stock Alerts")
    if st.button("Admin"):
        set_mode("Admin")

#---------------------------------Helper Functions---------------------



def get_company_profile(ticker):
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200 and response.json():
        return response.json()[0]
    return None

def get_historical_data(ticker):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries=200&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200 and response.json():
        return pd.DataFrame(response.json()["historical"])
    return pd.DataFrame()

def get_sector_performance():
    url = f"https://financialmodelingprep.com/api/v3/stock/sectors-performance?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data["sectorPerformance"])
    return pd.DataFrame()

def calculate_rsi(data, period=14):
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=600)
def get_top_gainers(limit: int = 10) -> pd.DataFrame:
    """
    Fetch today's top gainers from Financial Modeling Prep.
    
    Parameters:
      limit: number of tickers to return (default=10)
    
    Returns:
      DataFrame with columns: ['Ticker', 'Price', 'Price Change', 'Percent Change']
    """
    url = f"{BASE_URL}/gainers?apikey={FMP_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Build DataFrame and rename/format columns
        df = pd.DataFrame(data[:limit])
        df = df.rename(columns={
            "ticker": "Ticker",
            "price": "Price",
            "changes": "Price Change",
            "changesPercentage": "Percent Change"
        })
        # Convert "(+3.45%)" → 3.45
        df["Percent Change"] = (
            df["Percent Change"]
              .str.strip("()%+")
              .astype(float)
        )
        return df[["Ticker", "Price", "Price Change", "Percent Change"]]
    except Exception as e:
        st.error(f"Failed to load top gainers: {e}")
        return pd.DataFrame()


def fetch_sports_articles():
    response = requests.get(SPORTS_URL)
    if response.status_code == 200:
        data = response.json()
        return data.get("articles", [])
    else:
        return []


def getStockPrice(*tickers):
    FMP_API_KEY = "ZaYQNqtexbx4FAkxSBTvrDegpa25FFv1"
    tickers_str = ",".join(tickers)
    url = f"https://financialmodelingprep.com/api/v3/quote/{tickers_str}?apikey={FMP_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        stock_data = []
        for stock in data:
            stock_data.append({
                "Name": stock.get("name", "N/A"),
                "Ticker": stock.get("symbol", "N/A"),
                "Price": f"${stock.get('price', 'N/A'):.2f}" if stock.get("price") else "N/A"
            })
        return stock_data

    except requests.exceptions.RequestException as e:
        print("Error fetching stock data:", e)
        return []


def display_billboard_chart(data):
    """
    Displays a Plotly bar chart comparing current vs last week's Billboard ranks.
    
    Parameters:
    - data (list of dict): List of song dictionaries with 'Rank', 'Title', 'Artist',
      'Last Week', and optionally 'Peak Position'.
    """


    df = pd.DataFrame(music_data)

    # Create a bar chart
    fig = px.bar(df, x="Title", y="Rank", color="Rank", 
                title="Billboard Top 5 Music Hits", 
                labels={"Title": "Song Title", "Rank": "Chart Rank"},
                color_continuous_scale="Viridis")

    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        xaxis_title='Song Title',
        yaxis_title='Rank',
        xaxis_tickangle=-45,
        font=dict(family='Arial', size=14, color='white')
    )

    # Show the chart
    st.plotly_chart(fig)


def is_entertainment_article(article):
    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
    return any(keyword in text for keyword in ENTERTAINMENT_KEYWORDS)

def fetch_entertainment_articles():
    response = requests.get(MOVIE_NEWS_URL)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        curated = [a for a in articles if is_entertainment_article(a)]
        return curated[:5] if len(curated) >= 3 else curated + articles[:5 - len(curated)]
    return []



#---------------------------------End Helper Functions--------------------------------


#-------------------------------------------------------------Start---------------------
# Main router
mode = st.session_state.mode
#-----------------------------------------------------------------Home mode----------

if mode == "Home":

    st.image("images/austin_syline.jpg", use_container_width=True)  # This stretches the image to the container width
    st.title("Welcome to Team Austin Market Dashboard")
    st.subheader("Helping Austin locals and Texans take control of their financial future — from retirement to tax strategies and everything in between.")
    
    
    st.subheader("Read our Financial Insights on the sidebar. Our tech foward insights give your account an edge and clarity over a typical plan provider advisor. Contact a Fidicuary Financial Advisor today for a complementary consulation.")
    st.subheader("Today, Tomorrow, Togehter.")
    st.write("                   ")
    st.write("                   ")

    #button
    st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://calendly.com/black-mesa-softwar3" target="_blank">
            <button style="padding: 0.75em 1.5em; font-size: 1em; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">
                📅 Book a Free Financial Advisor Consultation
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True

    )

    st.subheader("Why Me?")
    st.write("I am not just pushing product. I built the tools, insights, and a platform that shows real transparency and care for my clients. I want to prove my value with this website before asking for a meeting. I want to give people the power to make smarter decisions today.")

#-----------------------------Credits------------------------------------------------

    st.subheader("Credits")
    st.write("Devloped by Kevin Martinez - 2025 for Texas Finanancial Advisors")
    st.write("Contact Email : kevin.martinez@texasfa.com")
    
#----------------------------------------end home mode-------------------------------------------

#---------------------------------------Start Admin Mode------------------------------------
elif mode == "Admin":
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


elif mode == "Stock Database":

    st.image("images/stocks.jpg", use_container_width=True)  # This stretches the image to the container width
    st.subheader("Welcome to the Stock Database")
    st.divider()
    st.write("Enter any Ticker and get Key Indicators as well as Trending Company News")
    st.write("Information is Power at Team Austin")
    st.divider()

    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", value="AAPL")
    if ticker:
        profile = get_company_profile(ticker)
        hist_data = get_historical_data(ticker)

        if profile:
            hist_data["date"] = pd.to_datetime(hist_data["date"])
            hist_data.sort_values("date", inplace=True)
            hist_data.set_index("date", inplace=True)

            # Display basic company info
            st.subheader(f"{profile['companyName']} ({profile['symbol']})")
            st.write(f"**Industry:** {profile['industry']}")
            st.write(f"**Website:** [{profile['website']}]({profile['website']})")
            st.write(f"**Description:** {profile['description']}")
            st.write(f"**Price:** ${profile['price']}")
            st.write(f"**Exchange:** {profile['exchangeShortName']}")

            # Calculate technical indicators
            hist_data["50_MA"] = hist_data["close"].rolling(window=50).mean()
            hist_data["200_MA"] = hist_data["close"].rolling(window=200).mean()
            hist_data["RSI"] = calculate_rsi(hist_data)

            # Key metrics
            latest = hist_data.iloc[-1]
            st.markdown("### 📊 Key Indicators")
            st.metric("Price", f"${latest['close']:.2f}")
            st.metric("Volume", f"{int(latest['volume']):,}")
            st.metric("RSI (14)", f"{latest['RSI']:.2f}")
            st.metric("50-day MA", f"{latest['50_MA']:.2f}")
            st.metric("200-day MA", f"{latest['200_MA']:.2f}")
            st.metric("P/E Ratio", profile.get("pe", "N/A"))
            st.metric("Beta", profile.get("beta", "N/A"))

            st.markdown("### 📈 Price & Moving Averages")
            st.line_chart(hist_data[["close", "50_MA", "200_MA"]])

            st.markdown("### 📊 Volume Activity")
            st.bar_chart(hist_data["volume"])
        else:
             st.warning("No historical data found.")
    else:
        st.error("Ticker not found or invalid.")

    st.subheader("Company Trending News")

    #------------------------------------------------------------------------News Feature Start-----------------------------------


    # 🔒 Replace this with your actual Finnhub API key
    FINNHUB_API_KEY = "cvrb3r1r01qp88cpcn7gcvrb3r1r01qp88cpcn80"

    # ---------------- News Page ----------------
    st.title("📰 Stock News")

    # Let user pick a ticker or reuse one they've searched
    news_ticker = ticker

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

elif mode == "Sector Performance":
    st.image("images/sectors.jpg", use_container_width=True)  # This stretches the image to the container width
    st.subheader("Welcome to the Sector Performance")
    st.subheader("Track Sector Performance Like a Pro")
    st.write("Get the macro view of how each sector is moving — and why it matters.")
    st.write("Data since last business day close")

    df_sectors = get_sector_performance()
    if not df_sectors.empty:
        df_sectors.columns = ["Sector", "Changes"]
        st.dataframe(df_sectors)
        st.bar_chart(df_sectors.set_index("Sector"))
    else:
        st.warning("Failed to fetch sector data.")


    #--------------------------------------------------------------------Economic Indicators Start-------------------------------
    st.subheader("MacroEconomics")
    st.write("Federal Reserve Bank of St.Louis")
   
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
        st.write("Why it matters: A signal of workforce strength and corporate health")

    with col2:
        st.metric("🛍️ Retail Sales", f"${retail_val:.0f}M", f"{retail_delta:+,.0f}")
        st.write("Why it matters: A direct gauge of consumer demand and spending power")

    with col3:
        st.metric("⛽ Avg Gas Price", f"${gas_val:.2f}/gal", f"{gas_delta:+.2f}")
        st.write("Why it matters: Impacts shipping, travel, and inflation")

    with col4:
        st.metric("🚗 Car Sales", f"{car_val:.2f}M units", f"{car_delta:+.2f}")
        st.write("Why it matters: Confidence indicator of big-ticket spending and credit access")

    # Optional: full chart section
    with st.expander("📈 See Historical Trends"):
        st.line_chart(df_unemp.set_index("Date")["Unemployment Rate"])
        st.line_chart(df_retail.set_index("Date")["Retail Sales"])
        st.line_chart(df_gas.set_index("Date")["Gas Price"])
        st.line_chart(df_car.set_index("Date")["Car Sales"])


    #--------------------------------------------------------------------End Economic Indicators End-------------------------------

elif mode == "Market Insights":

    st.subheader("Welcome to Market Insights")
    st.subheader("Sharp analysis. Big-picture trends. No fluff. Know where the market is — and where it might be headed.")

    st.subheader("🏅 Top 10 Gainers Today")
    with st.spinner("Loading top gainers..."):
        top_gain = get_top_gainers(limit=10)

    if not top_gain.empty:
        st.dataframe(top_gain.set_index("Ticker"))       # Table view
        st.bar_chart(top_gain.set_index("Ticker")["Percent Change"])  # Chart
    else:
        st.warning("No gainers data available at the moment.")

    st.divider()

    st.subheader("Market Concensus - Buy? Hold? Sell?")
    st.write("See what other's are saying about a particular Stock?")
    st.write("Source: Finhub")

    #-------------------------------------------------Analysts Concensus------------------------------

    #ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", value="AAPL")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", value="AAPL")

    FINNHUB_API_KEY = "cvrb3r1r01qp88cpcn7gcvrb3r1r01qp88cpcn80"
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={FINNHUB_API_KEY}" 
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data:
            # Convert JSON to DataFrame
            df = pd.DataFrame(data)
            df['period'] = pd.to_datetime(df['period'])
            df = df.sort_values('period')  # Oldest to newest

            # Optional: convert to human-readable string
            df['Period Label'] = df['period'].dt.strftime("%b %Y")

            st.subheader(f"🧭 Analyst Ratings for {ticker.upper()} (Historical Trend)")

            # Line chart of ratings
            fig = px.line(
                df,
                x='Period Label',
                y=['strongBuy', 'buy', 'hold', 'sell', 'strongSell'],
                labels={
                    'value': 'Number of Ratings',
                    'Period Label': 'Period'
                },
                title=f"Analyst Recommendation Trends for {ticker.upper()}",
                markers=True
            )
            fig.update_layout(legend_title_text='Rating Type')
            st.plotly_chart(fig)

            # Show latest values as metrics (optional)
            latest = df.iloc[-1]
            pretty_period = latest['period'].strftime("%B %Y")
            st.markdown(f"### 📊 Latest Consensus Snapshot ({pretty_period})")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("✅ Strong Buy", latest['strongBuy'])
            with col2: st.metric("📈 Buy", latest['buy'])
            with col3: st.metric("📊 Hold", latest['hold'])
            with col4: st.metric("📉 Sell", latest['sell'])
            with col5: st.metric("❌ Strong Sell", latest['strongSell'])

        else:
            st.info("No analyst recommendations available.")
    else:
        st.error("Error fetching recommendation data.")
        
        

    

    #-------------------------------------------------Analysyt Consensus Feature End------------------------------

elif mode == "Stock Alerts":
    st.subheader("Welcome to Stock Alerts")
    st.subheader("Get instant SMS/Email Notifications on Stock breakthroughs")

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
                        #send_sms_notification(phone, ticker, alert_type, current_value, f"{direction} {threshold}")

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

elif mode == "Culture and Capital":
    st.subheader("Welcome to Culture and Capital")
    st.write("  To be an Austinite is to be effortlessly cool — and plugged in.")
    st.write("This section tracks what the tastemakers are watching, listening to, and betting on in music, media, entertainment, and sports — because the culture is the economy.")
    st.write("  Team Austin articles updated Weekly")

    st.subheader("Music")
    st.image("images/music.jpg", use_container_width=True)  # This stretches the image to the container width
    

    st.divider()


    col1, col2 = st.columns([2.3,1.7])

    with col1:
        music_data = [{'Rank': 1, 'Title': 'Luther', 'Artist': 'Kendrick Lamar & SZA', 'Last Week': 1, 'Peak Position': 1}, {'Rank': 2, 'Title': 'Ordinary', 'Artist': 'Alex Warren', 'Last Week': 3, 'Peak Position': 2}, {'Rank': 3, 'Title': 'Die With A Smile', 'Artist': 'Lady Gaga & Bruno Mars', 'Last Week': 2, 'Peak Position': 1}, {'Rank': 4, 'Title': 'Nokia', 'Artist': 'Drake', 'Last Week': 4, 'Peak Position': 2}, {'Rank': 5, 'Title': 'A Bar Song (Tipsy)', 'Artist': 'Shaboozey', 'Last Week': 5, 'Peak Position': 1}]
        display_billboard_chart(music_data)

    with col2:
        st.subheader("Music Finance")
        st.divider()
        music_stock = ["SPOT", "LYV", "SONY"]
        music_stock_market_data = getStockPrice(*music_stock)

        if music_stock_market_data:
            st.table(music_stock_market_data)
        else:
            st.error("Failed to load music stock prices.")

    st.subheader("On Music by Team Austin MUSIC Coverage")
    st.write("Article Coming Soon")
    st.divider()    


    st.subheader("Entertainment : Film, TV, Celebraties, Gossip")
    st.image("images/film.jpg", use_container_width=True)  # This stretches the image to the container width
    st.divider()

    st.subheader("🎥 Top 5 Trending Entertainment Articles")

    col1, col2 = st.columns([2,2])

    with col1:

        articles = fetch_entertainment_articles()

        if articles:
            for article in articles:
                st.markdown(f"### [{article['title']}]({article['url']})")
                st.markdown(f"*{article['source']['name']}* — {article['publishedAt'][:10]}")
                if article.get('urlToImage'):
                    st.image(article['urlToImage'], use_container_width=True)
                st.markdown(f"{article['description']}")
                st.markdown("---")
        else:
            st.error("Couldn't load trending entertainment articles.")

    
    with col2:
        st.subheader("Entertainment Finance")
        
        sport_stock = ["TTWO", "NFLX", "DIS", "WBD", "PARA", "CMCSA", "AMCX", "IMAX"]
        sport_stock_market_data = getStockPrice(*sport_stock)

        if sport_stock_market_data:
            st.table(sport_stock_market_data)
        else:
            st.error("Failed to load sport stock prices.")

    
    st.subheader("On FILM by Team Austin ENTERTAINMENT Coverage")
    st.write("Article Coming Soon")
    st.divider()  



    st.subheader("Sports")
    st.image("images/sports.jpg", use_container_width=True)  # This stretches the image to the container width
    st.divider()
    

    col1, col2 = st.columns([2,2])

    with col1:
        st.subheader("Top 5 Trending Sports Articles")
        articles = fetch_sports_articles()

        if articles:
            for article in articles:
                st.markdown(f"### [{article['title']}]({article['url']})")
                st.markdown(f"*{article['source']['name']}* — {article['publishedAt'][:10]}")
                st.markdown(f"{article['description']}")
                st.markdown("---")
        else:
            st.error("Unable to fetch sports articles at this time.")

    
    with col2:
        st.subheader("Sport Finance")
        st.divider()
        sport_stock = ["DIS", "CMCSA", "FOXA", "PARA", "DKNG", "NKE", "ADDYY"]
        sport_stock_market_data = getStockPrice(*sport_stock)

        if sport_stock_market_data:
            st.table(sport_stock_market_data)
        else:
            st.error("Failed to load sport stock prices.")


    st.divider()

    st.subheader("On SPORTS by Team Austin SPORTS Coverage")
    st.write("Article Coming Soon")
    st.divider()

elif mode == "401(k) Strategies":
    st.image("images/american.jpg", use_container_width=True)  # This stretches the image to the container width

    def get_etf_data(ticker):
        url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_API_KEY}"
        response = requests.get(url)
        data = response.json()
        if data:
            return data[0]
        return None

    # Fetch the data for ETFs
    def get_optimized_funds():
        # Define the list of low-cost ETFs
        etf_tickers = {
            "Equity": "VTI",               # Total Market ETF
            "International": "VXUS",       # International ETF
            "Bond": "BND",                 # Bond ETF
            "Cash": "CASHX",               # Stable Value Fund
        }
        
        # Fetch the latest data for each ETF
        optimized_funds = {}
        for asset_class, ticker in etf_tickers.items():
            data = get_etf_data(ticker)
            if data:
                optimized_funds[asset_class] = {
                    "Fund": data['name'],
                    "Ticker": data['symbol'],
                    "Expense Ratio": data['price'],  # Assume we are using price as an approximation for expense ratio
                    "Current Price": data['price']
                }
            else:
                optimized_funds[asset_class] = {
                    "Fund": f"{asset_class} ETF",
                    "Ticker": ticker,
                    "Expense Ratio": 0.05,  # Default value if data is not available
                    "Current Price": 0
                }
        
        return optimized_funds

    # Get the optimized funds based on FMP API data
    optimized_funds = get_optimized_funds()
    st.subheader("Take Back Control of Your 401(k): Break Free from Generic Funds, Cut Hidden Fees, and Unlock Higher Returns.")
    st.subheader("Your 401(k) is part of the American Dream — don’t let it get lost in the system. " \
    "Too many hard-working people are stuck in generic target date funds, paying high fees for mediocre results. " \
    "That’s not freedom — that’s missed opportunity. ")
    st.subheader(
    "You’ve worked too hard to hand your future over to a cookie-cutter plan. " \
    "With the right strategy, you can cut costs, boost performance, and take real ownership of your retirement. " \
    "Let’s put your money back to work — for you.")

    st.divider()
    st.write("                            ")
    st.write("                            ")

    st.subheader("Meet Ellie")
    st.subheader("Ellie, like most Americans was stuck in a underperforming, untailored, default 401(k) plan")
    st.write("                            ")
    st.write("                            ")

    # --- Step 1: Load CSV directly ---
    st.header("Ellie's Previous 401(k) Holdings")

    # Replace this path with the actual path to your CSV file
    uploaded_file = "401(k)_optomizer/401(k)_csv/sample_CSV.csv"  

    try:
        fund_df = pd.read_csv(uploaded_file)
        #st.success("401(k) holdings loaded successfully.")
        #st.dataframe(fund_df)
    except FileNotFoundError:
        st.error("CSV file not found. Please make sure the file exists at the specified path.")

    
    if uploaded_file:
        st.write("Ellie's Previous Portfolio", fund_df)

        # --- Automate classification of funds ---                  #basic keyword matching to categorize the type of fund
        def classify_asset_class(name):
            name = name.lower()
            if any(x in name for x in ["equity", "stock", "s&p", "index", "growth", "value"]):
                return "Equity"
            elif any(x in name for x in ["bond", "income", "fixed"]):
                return "Bond"
            elif any(x in name for x in ["intl", "international", "global", "emerging"]):
                return "International"
            elif any(x in name for x in ["cash", "stable", "money"]):
                return "Cash"
            return "Other"                  #catch all i fnothing matches
        
        #new column to the datafram classifying each fund
        fund_df["Asset Class"] = fund_df["Fund Name"].apply(classify_asset_class)

    #step 2: Get Expense Ratio

    #calculate total weighted expense ratio of the uploaded portfolio
    total_expense = sum((row["% Allocation"] / 100) * row["Expense Ratio"] for idx, row in fund_df.iterrows())
    st.metric("💸 Total Weighted Expense Ratio", f"{total_expense:.2%}")


    st.write("Total Weighted Expense Ratio is calculated by expense ratio of each fund, weighted by their allocation.")
    st.subheader("Ellie's Portfolio Earnings were being cut by 6.30% !")

    st.write("                            ")
    st.write("                            ")
    st.write("                            ")

    # --- Risk Profile Inputs ---
    st.subheader("Play with Ellie's Portfolio, Enter your Personalized Risk Inputs too see what it could mean for your 401(k)")
    age = st.slider("Your Age", 20, 70, 45)     #user's age
    retirement_age = st.slider("Desired Retirement Age", 50, 75, 65)  #desired retirment
    years_to_retirement = retirement_age - age
    balance = st.number_input("Current 401(k) Balance ($)", min_value=1000, value=10000, step=1000)     #users current 401(k) balance
    st.subheader("Change Risk Level to Run different Simulations")
    risk_level = st.selectbox("Risk Preference", ["Conservative", "Balanced", "Aggressive"])  #user's risk tolerance

    st.write("Risk based asset allocation is based of classic porfolio theory")
    st.write("Conservative: Equity: 40, International: 10, Bond: 45, Cash: 5")
    st.write("Balanced: Equity: 55, International: 15, Bond: 25, Cash: 5")
    st.write("Agressive: Equity: 70, International: 20, Bond: 10, Cash: 0")

    # Recommended equity allocation (based on risk)         follow classic portfolio theory
    risk_profiles = {
        "Conservative": {"Equity": 40, "International": 10, "Bond": 45, "Cash": 5},
        "Balanced":     {"Equity": 55, "International": 15, "Bond": 25, "Cash": 5},
        "Aggressive":   {"Equity": 70, "International": 20, "Bond": 10, "Cash": 0},
    }
    target_allocation = risk_profiles[risk_level]

    # --- Step 3: Optimized Recommendations --- 
   # Build optimized portfolio DataFrame based on profile (low-cost ETF recommendations by asset class)
    opt_data = []
    for asset_class, weight in target_allocation.items():
        fund_data = optimized_funds.get(asset_class, {})
        opt_data.append({
            "Fund": fund_data["Fund"],
            "Ticker": fund_data["Ticker"],
            "Allocation": weight,
            "Expense Ratio": fund_data["Expense Ratio"],
            "Current Price": fund_data["Current Price"]
        })
    opt_df = pd.DataFrame(opt_data)

    # Display the optimized portfolio
    st.subheader("Ellie's Optimized Portfolio")
    st.dataframe(opt_df)

    # --- Allocation Pie Charts ---

    st.subheader("Portfolio Allocation Comparison: Previous Vs Optomized")
    col1, col2 = st.columns([1.5,2.5])
    with col1:
        st.write("**Previous Allocation**")
        current_pie = fund_df.groupby("Asset Class")["% Allocation"].sum()
        st.plotly_chart(go.Figure(data=[go.Pie(labels=current_pie.index, values=current_pie.values)]))
    with col2:
        st.write("**Optimized Allocation**")
        st.plotly_chart(go.Figure(data=[go.Pie(labels=opt_df["Fund"], values=opt_df["Allocation"])]))

    st.markdown("---")

    # --- Growth projection chart ---
    st.subheader("Projected Growth Over 25 Years")
    years = np.arange(0, 26)        # range

    # Calculate the weighted return for the current portfolio
    fund_df["Expected Return"] = fund_df["Asset Class"].map({
        "Equity": 0.07,
        "International": 0.06,
        "Bond": 0.035,
        "Cash": 0.015
    })
    weighted_return_current = (fund_df["% Allocation"] / 100 * fund_df["Expected Return"]).sum()

    # Calculate the weighted return for the optimized portfolio
    opt_df["Expected Return"] = opt_df["Fund"].map({
        "Total Market ETF": 0.07,
        "International ETF": 0.06,
        "Bond ETF": 0.035,
        "Stable Value Fund": 0.015
    })

    # Adjusting for average expense ratio
    weighted_return_opt = (opt_df["Allocation"] / 100 * opt_df["Expected Return"]).sum()

    # Calculate net growth rate for both portfolios
    net_rate_current = weighted_return_current - total_expense  # Current portfolio net rate
    net_rate_opt = weighted_return_opt - (opt_df["Expense Ratio"].mean() / 100)  # Optimized portfolio net rate

    # Prevent a very small or negative growth rate by ensuring minimum growth
    net_rate_current = max(net_rate_current, 0.01)  # Minimum 1% growth for current portfolio
    net_rate_opt = max(net_rate_opt, 0.02)          # Minimum 2% growth for optimized portfolio

    # Project growth over time (25 years)
    curr_growth = balance * (1 + net_rate_current) ** years
    opt_growth = balance * (1 + net_rate_opt) ** years

    # Display the projections
    st.line_chart(pd.DataFrame({
        "Current (net)": curr_growth,
        "Optimized (net)": opt_growth
    }, index=years))  # Display both projections

    st.subheader("Ellie's Portfolio after Enhacment: Lower expense ratio, risk-level personalization, and higher returns")
    st.subheader("Now Ellie's Portfolio is supercharged for Retirment")
    st.write("                            ")
    st.write("                            ")
    st.subheader("Book a meeting to take back control of your 401(k)!")
    #button
    st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://calendly.com/black-mesa-softwar3" target="_blank">
            <button style="padding: 0.75em 1.5em; font-size: 1em; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">
                📅 Book a Free Financial Advisor Consultation
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True

    )


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
#install dependencies: pip install -r requirements.txt


#push to github
#git add .
#git commit -m "message"
#git push

"""

Version 1.7 
Last Update: May 7 - 2025  at 01:23PM CT  

"""