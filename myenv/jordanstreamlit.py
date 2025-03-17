import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import timedelta, datetime
import random

#############################################
# 1. Data Loading and Preparation
#############################################
@st.cache_data
def load_data():
    # Load your pre-scraped CSV file ("final_jordan_data.csv")
    df = pd.read_csv("final_jordan_data.csv")
    
    # Attempt to parse the sell_date column (expected format "Apr 10, 2023")
    df["sell_date"] = df["sell_date"].fillna("").astype(str).str.strip()
    df["sell_date_dt"] = pd.to_datetime(df["sell_date"], format="%b %d, %Y", errors="coerce")
    
    # If no valid sell_date_dt exists (all NaT), fabricate fake sell dates.
    if df["sell_date_dt"].isna().all():
        st.write("No valid sell dates found – assigning fake sell dates (evenly spaced over the last 90 days).")
        new_df = pd.DataFrame()
        for model, group in df.groupby("model", as_index=False):
            group = group.copy()
            n = len(group)
            start_date = pd.Timestamp.today() - pd.Timedelta(days=90)
            # Evenly spaced dates over the 90-day period.
            if n > 1:
                fake_dates = pd.date_range(start=start_date, periods=n, freq="D")
            else:
                fake_dates = [pd.Timestamp.today()]
            group["sell_date_dt"] = fake_dates.strftime("%b %d, %Y")
            new_df = pd.concat([new_df, group], ignore_index=True)
        df = new_df
        df["sell_date_dt"] = pd.to_datetime(df["sell_date_dt"], format="%b %d, %Y", errors="coerce")
    
    # Create a numeric price column if not already present.
    def parse_price(price_str):
        match = re.search(r"\$([\d.,]+)", str(price_str))
        if match:
            return float(match.group(1).replace(",", ""))
        return None
    if "price_numeric" not in df.columns:
        df["price_numeric"] = df["price"].apply(parse_price)
    
    # Create additional columns:
    # ISO week number and daily date
    df["week"] = df["sell_date_dt"].dt.isocalendar().week
    df["date"] = df["sell_date_dt"].dt.date
    
    # Categorize models: "Retro" if model name contains "Retro", otherwise "OG"
    df["category"] = df["model"].apply(lambda x: "Retro" if "Retro" in x else "OG")
    
    # Create a 'limited' flag based on keywords
    def is_limited(model):
        keywords = ["Limited", "Rare", "Off White", "Travis Scott"]
        return int(any(kw.lower() in model.lower() for kw in keywords))
    df["limited"] = df["model"].apply(is_limited)
    
    return df

df = load_data()

#############################################
# 2. Sidebar Filters and Comparison Options
#############################################
st.title("Jordan Sales Data Analysis (Last 90 Days)")

st.sidebar.header("Filters and Comparison Options")
unique_models = sorted(df["model"].unique())
model1 = st.sidebar.selectbox("Select first Jordan model", unique_models, index=0)
model2 = st.sidebar.selectbox("Select second Jordan model", unique_models, index=1)

# Date range filter (using the sell_date_dt range)
if df["sell_date_dt"].notna().any():
    min_date = df["sell_date_dt"].min().date()
    max_date = df["sell_date_dt"].max().date()
else:
    min_date = datetime.today().date() - timedelta(days=90)
    max_date = datetime.today().date()
date_range = st.sidebar.date_input("Select date range", [min_date, max_date])

mask = (df["sell_date_dt"] >= pd.to_datetime(date_range[0])) & (df["sell_date_dt"] <= pd.to_datetime(date_range[1]))
filtered_df = df.loc[mask]
st.write(f"Data contains {len(filtered_df)} records in the selected date range.")

#############################################
# 3. Analytical Visualizations and Tools
#############################################

# 3A. Time-Series Line Chart: Weekly Average & Median Sale Price for Two Models
def aggregate_weekly(model_name):
    model_df = filtered_df[filtered_df["model"] == model_name]
    weekly = model_df.groupby("week")["price_numeric"].agg(["mean", "median"]).reset_index()
    weekly["model"] = model_name
    return weekly

weekly_model1 = aggregate_weekly(model1)
weekly_model2 = aggregate_weekly(model2)
weekly_df = pd.concat([weekly_model1, weekly_model2])

fig_line = go.Figure()
for m, subdf in weekly_df.groupby("model"):
    fig_line.add_trace(go.Scatter(x=subdf["week"], y=subdf["mean"],
                                  mode="lines+markers", name=f"{m} Avg"))
    fig_line.add_trace(go.Scatter(x=subdf["week"], y=subdf["median"],
                                  mode="lines+markers", name=f"{m} Median", line=dict(dash="dash")))
fig_line.update_layout(title="Weekly Average and Median Sale Price",
                       xaxis_title="Week of the Year", yaxis_title="Price ($)")
st.plotly_chart(fig_line, use_container_width=True)

# 3B. Box Plot: Distribution of Sale Prices by Category (OG vs Retro)
fig_box = px.box(filtered_df, x="category", y="price_numeric", color="category",
                 title="Distribution of Sale Prices by Category (OG vs Retro)",
                 labels={"price_numeric": "Sale Price ($)", "category": "Category"})
st.plotly_chart(fig_box, use_container_width=True)

# 3C. Overall Daily Trend: Line Chart of Daily Average and Median Sale Price Across All Models
daily_trend = filtered_df.groupby("date")["price_numeric"].agg(["mean", "median"]).reset_index()
fig_daily = go.Figure()
fig_daily.add_trace(go.Scatter(x=daily_trend["date"], y=daily_trend["mean"],
                               mode="lines+markers", name="Daily Avg"))
fig_daily.add_trace(go.Scatter(x=daily_trend["date"], y=daily_trend["median"],
                               mode="lines+markers", name="Daily Median", line=dict(dash="dash")))
fig_daily.update_layout(title="Overall Daily Average and Median Sale Price (All Models)",
                        xaxis_title="Date", yaxis_title="Price ($)")
st.plotly_chart(fig_daily, use_container_width=True)

# 3D. Annotated Event Plot: Overlay Fake Cultural Events on the Weekly Trend for Model1
model1_weekly = aggregate_weekly(model1)
# Fake events (customize these as needed)
events = [(10, "Documentary Release"), (20, "Movie Premiere"), (35, "Anniversary")]
fig_event = px.line(model1_weekly, x="week", y="mean",
                    title=f"{model1} Weekly Average Price with Events",
                    labels={"week": "Week", "mean": "Average Price ($)"})
for wk, desc in events:
    fig_event.add_vline(x=wk, line_dash="dash", annotation_text=desc, annotation_position="top left")
st.plotly_chart(fig_event, use_container_width=True)

# 3E. Overall Trend Analysis for All Shoes: Compare Weekly Averages Across Models
overall_weekly = filtered_df.groupby(["model", "week"])["price_numeric"].mean().reset_index()
fig_overall = px.line(overall_weekly, x="week", y="price_numeric", color="model",
                      title="Weekly Average Sale Price Across All Models",
                      labels={"price_numeric": "Average Price ($)", "week": "Week of the Year"})
st.plotly_chart(fig_overall, use_container_width=True)

#############################################
# 4. Data Display and Save Options
#############################################
st.subheader("Raw Data Sample")
st.dataframe(filtered_df.head(20))

st.subheader("Summary Statistics by Model")
summary_stats = filtered_df.groupby("model")["price_numeric"].describe().reset_index()
st.dataframe(summary_stats)

if st.button("Save Final Data as CSV"):
    final_df = df.copy()  # Save the full dataset, or use filtered_df if preferred.
    final_df.to_csv("final_jordan_data_updated.csv", index=False)
    st.success("Data saved as final_jordan_data_updated.csv")
