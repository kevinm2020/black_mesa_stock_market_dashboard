#imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import requests

FMP_API_KEY = "ZaYQNqtexbx4FAkxSBTvrDegpa25FFv1"

# Function to fetch ETF data from the FMP API
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

#Streamlit Page Config
st.set_page_config(page_title="401(k) Optimizer", layout="wide")
st.title("Black Mesa 401(k) Optimizer")

# --- Step 1: User Inputs ---
st.header("Step 1: Enter Your Current 401(k) Holdings")
uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])         #user uploads csv

if uploaded_file:
    fund_df = pd.read_csv(uploaded_file)        #read csv
    st.write("### Your Uploaded Portfolio", fund_df)

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




    # --- Step 2: Fee Analysis & Risk Evaluation ---
    st.header("Step 2: Fee Analysis + Risk Evaluator")

    st.markdown("""
    Many 401(k) plans offer only **pre-selected mutual funds**, which often come with:
    - ⚠️ **High Expense Ratios**
    - 📉 **Undeperforming fund managers***
    - 🔒 **Limited Investment Choices**
    - 🕵️ Hidden transactions or plan fees
                
    These can **erode your retirment returns** over time without you realizing it.
                """)

    #calculate total weighted expense ratio of the uploaded portfolio
    total_expense = sum((row["% Allocation"] / 100) * row["Expense Ratio"] for idx, row in fund_df.iterrows())
    st.metric("💸 Total Weighted Expense Ratio", f"{total_expense:.2%}")


    #gauge visualization for expense ratio (speedometer)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_expense * 100,
        title={'text': "Expense Ratio Gauge"},
        gauge={
            'axis': {'range': [0, 1]},
            'steps': [
                {'range': [0, 0.2], 'color': "lightgreen"},
                {'range': [0.2, 0.6], 'color': "gold"},
                {'range': [0.6, 1], 'color': "tomato"},
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': total_expense * 100}
        }
    ))
    st.plotly_chart(fig)

    st.write("Total Weighted Expense Ratio is calculated by : expense ratio of each user provided fund, weighted by their allocation.")
    st.write("What this means is an overall expense ratio reflecting the entirety of the portfolio .")

    # --- Risk Profile Inputs ---
    st.subheader("Personalized Risk Inputs")
    age = st.slider("Your Age", 20, 70, 45)     #user's age
    retirement_age = st.slider("Desired Retirement Age", 50, 75, 65)  #desired retirment
    years_to_retirement = retirement_age - age
    balance = st.number_input("Current 401(k) Balance ($)", min_value=1000, value=10000, step=1000)     #users current 401(k) balance

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


    #does current allocation align with risk profile?
    #equity exposure
    user_stock_pct = fund_df[fund_df["Asset Class"] == "Equity"]["% Allocation"].sum()
    #reccomended stock percentage
    recommended_stock_pct = target_allocation["Equity"]
    risk_alignment = abs(user_stock_pct - recommended_stock_pct)

    color = "🟢 Aligned" if risk_alignment < 10 else "🟡 Slightly Off" if risk_alignment < 20 else "🔴 Misaligned"
    st.metric("🎯 Risk Alignment", f"{user_stock_pct:.0f}% equity vs {recommended_stock_pct:.0f}% recommended", delta=color)

    st.markdown("---")

    st.write("Risk Alingment is calculated on how close user's portfolio mimics classic portfolio theory risk based allocation ")



    # --- Step 3: Optimized Recommendations --- 
    st.header("Step 3: Optimized Recommendations")

    st.write("Approach: Move user from mutual funds to low-cost ETF (universe: Vanguard)")

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
    st.dataframe(opt_df)



   # --- Growth projection chart ---
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



    # --- Allocation Pie Charts ---

    st.subheader("What is the Brokerage Link Tool?")
    st.markdown("""
    BrokerageLink is a powerful tool offered inside some 401(k) plans that unlocks:
    - ✅ Full access to ETFs, index funds, and stocks
    - 🛠️ Greater custimization of your portfolio
    - 📊 Better transperancy and control
    """)

    # Comparison chart
    comparison_df = pd.DataFrame({
        "Feature": ["Fund Choices", "Fees", "Transparency", "Control"],
        "Traditional 401(k)": ["Limited mutual funds", "High", "Low", "Low"],
        "With BrokerageLink": ["Thousands of ETFs/stocks", "Low", "High", "Full"]
    })

    st.table(comparison_df)

    st.subheader("Portfolio Allocation Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Current Allocation by Asset Class**")
        current_pie = fund_df.groupby("Asset Class")["% Allocation"].sum()
        st.plotly_chart(go.Figure(data=[go.Pie(labels=current_pie.index, values=current_pie.values)]))
    with col2:
        st.write("**Optimized Allocation**")
        st.plotly_chart(go.Figure(data=[go.Pie(labels=opt_df["Fund"], values=opt_df["Allocation"])]))

    st.markdown("---")



    # --- Step 4: Report Placeholder ---
    st.header("Step 4: Generate Report")
    st.info("📥 PDF Report Generator Coming Soon! Will include: Fee summary, growth projections, optimized allocation, and BrokerageLink tips.")

 

    # CTA button
    st.markdown("---")
    st.markdown("### Ready to take control of your retirement?")
    if st.button("📞 Connect with a Trusted Fidicuary Financial Advisor Today"):
        st.markdown("[Schedule a Call](https://yourcalendly.com/link)")

else:
    st.warning("Please upload your portfolio CSV file to begin.")



    #terminal
    #switch directory: cd "/Users/edwinmartinez/Desktop/black_mesa/401(k)_optomizer"
    #deploy: streamlit run "black_mesa_401(k)_optomizer_BETA.py"  