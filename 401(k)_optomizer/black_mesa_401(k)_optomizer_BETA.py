#imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

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



    # --- Risk Profile Inputs ---
    st.subheader("Personalized Risk Inputs")
    age = st.slider("Your Age", 20, 70, 45)     #user's age
    retirement_age = st.slider("Desired Retirement Age", 50, 75, 65)  #desired retirment
    years_to_retirement = retirement_age - age
    balance = st.number_input("Current 401(k) Balance ($)", min_value=1000, value=10000, step=1000)     #users current 401(k) balance

    risk_level = st.selectbox("Risk Preference", ["Conservative", "Balanced", "Aggressive"])  #user's risk tolerance

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




    # --- Step 3: Optimized Recommendations --- 
    st.header("Step 3: Optimized Recommendations")

    #this is how much they will be able to save in fees
    st.markdown("""
    <div style="background-color:#f0f9ff;padding:20px;border-radius:10px;border:2px solid #00bfff;">
    <h2>✅ You could be saving up to <span style="color:green;">$117,000</span> in hidden fees over 25 years.</h2>
    <p>Here's what we recommend, based on your risk profile and available funds:</p>
    </div>
    """, unsafe_allow_html=True)

    #user saves in fees by reccomending low-cost ETFs

    # Build optimized portfolio DataFrame based on profile (low cost ETF recommendatiosn by asset class)
    low_cost_funds = {
        "Equity": ("Total Market ETF", "VTI", 0.03),
        "International": ("International ETF", "VXUS", 0.07),
        "Bond": ("Bond ETF", "BND", 0.04),
        "Cash": ("Stable Value Fund", "CASHX", 0.01),
    }

    opt_data = []
    for asset_class, weight in target_allocation.items():
        name, ticker, fee = low_cost_funds[asset_class]
        opt_data.append({"Fund": name, "Ticker": ticker, "Allocation": weight, "Expense Ratio": fee})
    opt_df = pd.DataFrame(opt_data)

    #display
    st.dataframe(opt_df)



    # --- Growth projection chart ---
    st.subheader("Projected Growth Over 25 Years")
    years = np.arange(0, 26)        #range


    curr_growth = balance * (1 + 0.05 - total_expense) ** years                                         #5% assumed return - expenses
    opt_growth = balance * (1 + 0.06 - opt_df["Expense Ratio"].mean() / 100) ** years                   #slightly better returns and lower fees
    st.line_chart(pd.DataFrame({"Current": curr_growth, "Optimized": opt_growth}, index=years))         #display both projections


    # --- Allocation Pie Charts ---

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

else:
    st.warning("Please upload your portfolio CSV file to begin.")



    #terminal
    #switch directory: cd "/Users/edwinmartinez/Desktop/black_mesa/401(k)_optomizer"
    #deploy: streamlit run "black_mesa_401(k)_optomizer_BETA.py"  