import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date
import matplotlib.pyplot as plt
import random

# --- Page and App Configuration ---
st.set_page_config(layout="wide")
st.title("Electricity Tariff Comparison Dashboard")
st.markdown("Analyze your consumption, compare flexible vs. static tariffs, and understand your usage patterns.")

# --- Constants and Configuration ---
LOCAL_TIMEZONE = "Europe/Vienna"  # Timezone for your consumption data
FLEX_COLOR = "#fd690d"
FLEX_COLOR_LIGHT = "#f7be44"
STATIC_COLOR = "#989898"

# --- Data Loading and Caching ---

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_spot_data(start: date, end: date) -> pd.DataFrame:
    """Fetches spot market price data from the aWATTar API for a given date range."""
    base_url = "https://api.awattar.de/v1/marketdata"
    start_dt, end_dt = datetime.combine(start, datetime.min.time()), datetime.combine(end + pd.Timedelta(days=1), datetime.min.time())
    params = {"start": int(start_dt.timestamp() * 1000), "end": int(end_dt.timestamp() * 1000)}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        df = pd.DataFrame(response.json()["data"])
        df["timestamp"] = pd.to_datetime(df["start_timestamp"], unit="ms", utc=True)
        df["spot_price_eur_kwh"] = df["marketprice"] / 1000
        return df[["timestamp", "spot_price_eur_kwh"]]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch spot price data: {e}")
    except (KeyError, IndexError, TypeError):
        st.error("Received unexpected data from the spot price API.")
    return pd.DataFrame()

@st.cache_data(ttl=3600) # Cache for 1 hour
def process_consumption_data(uploaded_file) -> pd.DataFrame:
    """Loads and processes the user"s consumption CSV, returning hourly UTC data."""
    if uploaded_file is None: return pd.DataFrame()
    try:
        df = pd.read_csv(uploaded_file, sep=";", decimal=",", encoding="utf-8", dayfirst=True, parse_dates=["Datum"])
        consumption_col = next((col for col in df.columns if "Verbrauch [kWh]" in col), None)
        if not consumption_col:
            st.error("A 'Verbrauch [kWh]' column was not found.")
            return pd.DataFrame()
        
        df_meas = df[["Datum", "Zeit von", consumption_col]].copy().dropna()
        df_meas.columns = ["date", "time_str", "consumption_kwh"]
        df_meas["timestamp_local"] = pd.to_datetime(df_meas["date"].dt.strftime("%Y-%m-%d") + " " + df_meas["time_str"])
        df_meas["timestamp"] = df_meas["timestamp_local"].dt.tz_localize(LOCAL_TIMEZONE, ambiguous="infer").dt.tz_convert("UTC")
        return df_meas.set_index("timestamp")["consumption_kwh"].resample("h").sum().dropna().reset_index()
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return pd.DataFrame()

# --- Core Analysis Functions ---
def classify_usage(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Classifies hourly consumption into Base, Peak, and Regular load."""
    if df.empty:
        return df, 0.0, 0.0
    
    df_c = df.copy()
    
    # --- Base Load Calculation ---
    # Find the most stable, lowest consumption period for each day to define the base load.
    # This is more robust than assuming base load occurs at fixed night hours.
    df_local = df_c.copy()
    df_local["timestamp_local"] = df_local["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
    df_local["date"] = df_local["timestamp_local"].dt.date

    # Calculate rolling metrics over a 3-hour window to find stable periods.
    indexed_consumption = df_local.set_index("timestamp_local")["consumption_kwh"]
    df_local["rolling_std"] = indexed_consumption.rolling(window=3, center=True).std().values
    df_local["rolling_mean"] = indexed_consumption.rolling(window=3, center=True).mean().values
    
    # For each day, find the point with the minimum rolling std dev (stability).
    # In case of a tie, choose the one with the lower rolling mean (consumption).
    stable_periods = df_local.dropna(subset=["rolling_std", "rolling_mean"]).sort_values(by=["rolling_std", "rolling_mean"])
    daily_base_load_points = stable_periods.groupby("date").first()
    base_load_threshold = daily_base_load_points["rolling_mean"].mean() if not daily_base_load_points.empty else 0.0

    # Calculate Peak Load threshold from consumption spikes
    df_c["consumption_diff"] = df_c["consumption_kwh"].diff().fillna(0)
    peak_threshold = df_c[df_c["consumption_diff"] > 0]["consumption_diff"].std() * 1.5 # Difference larger than 1.5x standard deviation
    peak_threshold = 0.0 if pd.isna(peak_threshold) else peak_threshold
    
    # Classify loads
    df_c["base_load_kwh"] = df_c["consumption_kwh"].clip(upper=base_load_threshold)
    influenceable_load = df_c["consumption_kwh"] - df_c["base_load_kwh"]
    is_peak = (df_c["consumption_diff"] > peak_threshold) & (peak_threshold > 0)
    df_c["peak_load_kwh"] = influenceable_load.where(is_peak, 0).clip(lower=0)
    df_c["regular_load_kwh"] = influenceable_load.where(~is_peak, 0).clip(lower=0)
        
    return df_c.drop(columns=["consumption_diff"]), base_load_threshold, peak_threshold

def run_cost_analysis(df: pd.DataFrame, flex_on_top: float, flex_fee: float, static_price: float, static_fee: float) -> pd.DataFrame:
    """Calculates flexible and static costs for the given dataframe."""
    df_costs = df.copy()
    df_costs["month"] = df_costs["timestamp"].dt.to_period("M")
    df_costs["days_in_month"] = df_costs["timestamp"].dt.days_in_month
    
    # Calculate total cost per hour for both tariffs
    flex_total_kwh_price = df_costs["spot_price_eur_kwh"] + flex_on_top
    hourly_flex_fee = (flex_fee / df_costs["days_in_month"]) / 24
    hourly_static_fee = (static_fee / df_costs["days_in_month"]) / 24
    
    df_costs["total_cost_flexible"] = (df_costs["consumption_kwh"] * flex_total_kwh_price) + hourly_flex_fee
    df_costs["total_cost_static"] = (df_costs["consumption_kwh"] * static_price) + hourly_static_fee
    return df_costs

def simulate_peak_shifting(df: pd.DataFrame, shift_percentage: float) -> pd.DataFrame:
    """
    Simulates shifting a percentage of each peak load to the cheapest hour
    within a +/- 2-hour window. This provides a more realistic model of
    short-term load shifting (e.g., delaying an appliance by an hour or two).
    """
    if shift_percentage == 0:
        return df

    df_sim = df.copy()
    
    # This series will store the load that is moved TO a specific hour.
    # This avoids modifying the dataframe while iterating over it.
    shifted_load_additions = pd.Series(0.0, index=df_sim.index)

    # Identify rows with peak load to iterate over them
    peak_indices = df_sim[df_sim['peak_load_kwh'] > 0.001].index

    for peak_idx in peak_indices:
        # 1. Define the 5-hour window (current hour +/- 2 hours) for each peak
        current_timestamp = df_sim.loc[peak_idx, 'timestamp']
        window_df = df_sim[
            (df_sim['timestamp'] >= current_timestamp - pd.Timedelta(hours=2)) &
            (df_sim['timestamp'] <= current_timestamp + pd.Timedelta(hours=2))
        ]
        if window_df.empty: continue

        # 2. Find the hour with the minimum price in this local window
        cheapest_hour_in_window = window_df.loc[window_df['spot_price_eur_kwh'].idxmin()]
        
        # 3. If a cheaper hour exists, shift the load
        if cheapest_hour_in_window['spot_price_eur_kwh'] < df_sim.loc[peak_idx, 'spot_price_eur_kwh']:
            kwh_to_shift = df_sim.loc[peak_idx, 'peak_load_kwh'] * (shift_percentage / 100.0)
            df_sim.loc[peak_idx, 'peak_load_kwh'] -= kwh_to_shift
            shifted_load_additions.loc[cheapest_hour_in_window.name] += kwh_to_shift

    # 4. Add the accumulated shifted loads to the regular load of the target hours
    df_sim['regular_load_kwh'] += shifted_load_additions
    
    # 6. Recalculate total consumption based on the new distribution
    df_sim["consumption_kwh"] = df_sim["base_load_kwh"] + df_sim["regular_load_kwh"] + df_sim["peak_load_kwh"]
    
    return df_sim

# --- UI and Rendering Functions ---

def get_sidebar_inputs(df_consumption: pd.DataFrame):
    """Renders all sidebar inputs and returns the configuration values."""
    with st.sidebar:
        st.header("Configuration")
        
        # 1. Date Range
        st.subheader("1. Analysis Period")
        min_date = df_consumption["timestamp"].min().date()
        max_date = df_consumption["timestamp"].max().date()
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = col2.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # 2. Tariffs
        st.subheader("2. Tariff Plans")
        with st.expander("Flexible (Spot Price) Plan", expanded=True):
            flex_on_top = st.number_input("On-Top Price (€/kWh)", value=0.0215, min_value=0.0, step=0.001, format="%.4f")
            flex_fee = st.number_input("Monthly Fee (€)", value=2.40, min_value=0.0, step=0.1, key="flex_fee")
        with st.expander("Static (Fixed Price) Plan"):
            static_price = st.number_input("Fixed Price (€/kWh)", value=0.14, min_value=0.0, step=0.01)
            static_fee = st.number_input("Monthly Fee (€)", value=2.99, min_value=0.0, step=0.1)

        # 3. Cost Simulation
        st.subheader("3. Cost Simulation")
        shift_percentage = st.slider(
            "Shift Peak Load (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Simulate shifting a percentage of your peak consumption to a cheaper hour within a +/- 2-hour window. This models realistic, short-term adjustments."
        )

        return start_date, end_date, flex_on_top, flex_fee, static_price, static_fee, shift_percentage

def render_recommendation(df: pd.DataFrame):
    """Displays the final tariff recommendation based on calculated savings."""
    st.subheader("Tariff Recommendation")
    savings = df["total_cost_static"].sum() - df["total_cost_flexible"].sum()
    
    df["price_quantile"] = df.groupby(df["timestamp"].dt.to_period("M"))["spot_price_eur_kwh"].transform(lambda x: pd.qcut(x, 4, labels=False, duplicates="drop"))
    peak_total_kwh = df["peak_load_kwh"].sum()
    peak_cheap_kwh = df[df["price_quantile"] == 0]["peak_load_kwh"].sum()
    peak_ratio = peak_cheap_kwh / peak_total_kwh if peak_total_kwh > 0 else 0

    if savings > 0:
        st.success(f"✅ Flexible Plan Recommended: You could have saved €{savings:.2f}")
        if peak_ratio > 0.3:
            st.write(f"This is a great fit. You align **{peak_ratio:.0%}** of your peak usage with the cheapest market prices.")
        else:
            st.write("You could save even more by shifting high-consumption activities to times with lower spot prices.")
    else:
        st.warning(f"⚠️ Static Plan Recommended: The flexible plan would have cost €{-savings:.2f} more.")
        st.write("A fixed price offers better cost stability for your current usage pattern.")

def render_cost_comparison_tab(df: pd.DataFrame):
    """Renders the content for the "Cost Comparison" tab."""
    st.subheader("Cost Breakdown by Period")
    resolution = st.radio("Select Time Resolution", ("Daily", "Weekly", "Monthly"), horizontal=True, key="res")
    freq_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "ME"}
    grouper = pd.Grouper(key="timestamp", freq=freq_map[resolution])

    # Table to compare flex vs spot prices
    df_summary = df.groupby(grouper).agg(
        **{
            "Total Consumption": ("consumption_kwh", "sum"),
            "Total Flexible Cost": ("total_cost_flexible", "sum"),
            "Total Static Cost": ("total_cost_static", "sum")
        }
    ).reset_index()
    df_summary = df_summary[df_summary["Total Consumption"] > 0].copy()
    
    if not df_summary.empty:
        df_summary["Period"] = df_summary["timestamp"].dt.strftime("%Y-%m-%d" if resolution != "Monthly" else "%Y-%m")
        st.subheader("Total Cost Comparison")
        st.line_chart(df_summary.set_index("Period"), y=["Total Flexible Cost", "Total Static Cost"], y_label="Total Cost (€)", color=[FLEX_COLOR, STATIC_COLOR])
        
        if resolution != "Daily":
            df_summary['Avg. Flexible Price (€/kWh)'] = df_summary['Total Flexible Cost'] / df_summary['Total Consumption']
            df_summary['Avg. Static Price (€/kWh)'] = df_summary['Total Static Cost'] / df_summary['Total Consumption']
            st.subheader("Average Price Comparison")
            st.line_chart(df_summary.set_index('Period'), y=['Avg. Flexible Price (€/kWh)', 'Avg. Static Price (€/kWh)'], y_label="Average Price (€/kWh)", color=[FLEX_COLOR, STATIC_COLOR])
            
        df_summary["Difference (€)"] = df_summary["Total Static Cost"] - df_summary["Total Flexible Cost"]
        styler = df_summary[["Period", "Total Consumption", "Total Flexible Cost", "Total Static Cost", "Difference (€)"]].style.format({
            "Total Consumption": "{:.2f} kWh", "Total Flexible Cost": "€{:.2f}", "Total Static Cost": "€{:.2f}", "Difference (€)": "€{:.2f}"
        }).map(lambda v: "color: #5fba7d" if v > 0 else "color: #d65f5f", subset=["Difference (€)"])
        st.dataframe(styler, hide_index=True, use_container_width=True)

def render_usage_pattern_tab(df: pd.DataFrame, base_threshold: float, peak_threshold: float):
    """Renders the content for the "Usage Pattern Analysis" tab, including the new combined chart."""
    st.subheader("Analyze Your Consumption Profile")
    df["day_type"] = df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
    day_filter = st.radio("Filter data by:", ("All Days", "Weekdays", "Weekends"), horizontal=True)
    
    df_pattern = df[df["day_type"] == day_filter[:-1]] if day_filter != "All Days" else df
    
    if df_pattern.empty:
        st.warning(f"No data available for {day_filter.lower()}.")
        return

    # --- COMBINED CHART ---
    st.subheader("Consumption & Cost Profile")
    st.text("The following chart depicts the proportion of each usage pattern on the x-axis, and the average price per kWh on the y-axis.\nIf peak usage costs align with regular usage, significant cost-saving potential is indicated.")
    load_types = ["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]
    profile_data = []
    total_kwh = df_pattern["consumption_kwh"].sum()

    for load in load_types:
        kwh = df_pattern[load].sum()
        if kwh > 0:
            avg_price = (df_pattern[load] * df_pattern["spot_price_eur_kwh"]).sum() / kwh
            proportion = kwh / total_kwh if total_kwh > 0 else 0
            profile_data.append({"Profile": load.replace("_kwh", "").replace("_", " ").title(), "kwh": kwh, "avg_price": avg_price, "proportion": proportion})
    
    if profile_data:
        df_plot = pd.DataFrame(profile_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create variable-width bar chart
        cumulative_width = 0
        for _, row in df_plot.iterrows():
            ax.bar(cumulative_width, row["avg_price"], width=row["proportion"], align="edge", color=FLEX_COLOR, edgecolor="white")
            # Add annotations
            annotation_text = f"{row["Profile"]}\n{row["proportion"]:.1%} of kWh\n€{row["avg_price"]:.3f}/kWh"
            ax.text(cumulative_width + row["proportion"]/2, row["avg_price"]/2, annotation_text, ha="center", va="center", color="white", fontsize=10, weight="bold")
            cumulative_width += row["proportion"]
            
        ax.set_ylabel("Average Spot Price (€/kWh)")
        ax.set_xlabel("Proportion of Total Consumption")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
        st.pyplot(fig)
    
    # --- Example Day Breakdown ---
    st.subheader("Example Day Breakdown")
    st.markdown("This chart shows a random day from your selection, breaking down each hour's consumption into its classified components: **Base**, **Regular**, and **Peak** load. This helps visualize when your high-consumption activities occur.")

    available_dates = df_pattern["timestamp"].dropna().dt.tz_convert(LOCAL_TIMEZONE).dt.date.unique()

    if len(available_dates) > 0:
        # Use session state to keep the random day consistent across reruns
        if 'random_day' not in st.session_state or st.session_state.random_day not in available_dates:
            st.session_state.random_day = random.choice(available_dates)

        col1_day, col2_day = st.columns([3, 1])
        with col1_day:
            st.caption(f"Displaying data for {st.session_state.random_day.strftime('%A, %Y-%m-%d')}")
        with col2_day:
            if st.button("Show a Different Day"):
                st.session_state.random_day = random.choice(available_dates)
                st.rerun()

        # Filter data for the selected random day
        df_day = df_pattern[df_pattern["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.date == st.session_state.random_day].copy()
        
        if not df_day.empty:
            df_day["hour"] = df_day["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.hour
            df_plot_day = df_day.set_index("hour")[["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]]
            
            # Rename for clearer legend
            df_plot_day.columns = ["Base Load", "Regular Load", "Peak Load"]
            
            st.bar_chart(
                df_plot_day,
                color=[STATIC_COLOR, FLEX_COLOR_LIGHT, FLEX_COLOR],
                y_label="Consumption (kWh)"
            )
    else:
        st.info("Not enough data in the selection to display an example day.")
    
    col1, col2 = st.columns(2)
    col1.metric("Base Load Threshold (Continuous Usage)", f"{base_threshold:.3f} kWh")
    col2.metric("Peak Detection Threshold (Sharp Increase)", f"{peak_threshold:.3f} kWh")
        
def render_yearly_summary_tab(df: pd.DataFrame):
    """Renders the content for the "Yearly Summary" tab."""
    st.subheader("Yearly Summary")
    df_yearly = df.copy()
    df_yearly["Year"] = df_yearly["timestamp"].dt.year
    yearly_agg = df_yearly.groupby("Year").agg(**{
        "Total Consumption": ("consumption_kwh", "sum"),
        "Total Flexible Cost":("total_cost_flexible", "sum"),
        "Total Static Cost": ("total_cost_static", "sum")}
    ).reset_index()
    
    if not yearly_agg.empty and yearly_agg["Total Consumption"].sum() > 0:
        yearly_agg["Avg. Flex Price (€/kWh)"] = yearly_agg["Total Flexible Cost"] / yearly_agg["Total Consumption"]
        yearly_agg["Avg. Static Price (€/kWh)"] = yearly_agg["Total Static Cost"] / yearly_agg["Total Consumption"]
        st.dataframe(yearly_agg.style.format(
            {"Total Consumption": "{:,.2f} kWh", 
            "Total Flexible Cost": "€{:.2f}",
            "Total Static Cost": "€{:.2f}",
            "Avg. Flex Price (€/kWh)": "€{:.2f}", 
            "Avg. Static Price (€/kWh)": "€{:.3f}"}), hide_index=True, use_container_width=True)

# --- Main Application Execution ---
uploaded_file = st.sidebar.file_uploader("Upload Your Consumption CSV", type=["csv"])

if not uploaded_file:
    st.info("👋 Welcome! Please upload your consumption data to begin.")
else:
    df_consumption = process_consumption_data(uploaded_file)
    if not df_consumption.empty:
        # Get all user inputs from the sidebar
        start_date, end_date, flex_on_top, flex_fee, static_price, static_fee, shift_percentage = get_sidebar_inputs(df_consumption)
        
        # Load and merge data based on date range
        df_spot_prices = fetch_spot_data(start_date, end_date)
        df_merged = pd.merge(df_consumption, df_spot_prices, on="timestamp", how="inner")
        df_merged = df_merged[(df_merged.timestamp.dt.date >= start_date) & (df_merged.timestamp.dt.date <= end_date)].dropna()

        if not df_merged.empty:
            # Classify usage to get thresholds (needed for absence detection)
            df_classified, base_threshold, peak_threshold = classify_usage(df_merged)
            
            # Conditionally render absence handling in sidebar
            with st.sidebar:
                df_classified["date_col"] = df_classified["timestamp"].dt.date
                daily_consumption = df_classified.groupby("date_col")["consumption_kwh"].sum()
                absence_days = daily_consumption[daily_consumption < (base_threshold * 24 * 0.8)].index.tolist()
                excluded_days = []
                if absence_days:
                    st.subheader("4. Absence Handling")
                    if st.checkbox("Select/Deselect All Absence Days"):
                        excluded_days = st.multiselect("Exclude days?", options=absence_days, default=absence_days)
                    else:
                        excluded_days = st.multiselect("Exclude days?", options=absence_days, default=[])

            # Filter out excluded days for final analysis
            df_analysis = df_classified[~df_classified["date_col"].isin(excluded_days)].copy()
            
            # Apply simulation if requested
            df_simulated = simulate_peak_shifting(df_analysis, shift_percentage)
            
            # Run final cost analysis on the clean data
            df_final = run_cost_analysis(df_simulated, flex_on_top, flex_fee, static_price, static_fee)
            
            # Render all UI components
            render_recommendation(df_final)
            tab1, tab2, tab3 = st.tabs(["**Cost Comparison**", "**Usage Pattern Analysis**", "**Yearly Summary**"])
            with tab1:
                render_cost_comparison_tab(df_final)
            with tab2:
                render_usage_pattern_tab(df_final, base_threshold, peak_threshold)
            with tab3:
                render_yearly_summary_tab(df_final)
        else:
            st.warning("No overlapping data found for the selected period.")