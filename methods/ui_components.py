import streamlit as st
import pandas as pd
import random
from datetime import datetime, date

from methods.config import *
from methods.tariffs import Tariff, TariffManager
from methods.utils import get_min_max_date, to_excel, get_intervals_per_day, get_aggregation_config, calculate_granular_data
import methods.data_loader as data_loader
import methods.charts as charts

# --- Sidebar and Input Controls ---

@st.cache_data(ttl=60*10)
def _compare_all_tariffs(df_consumption: pd.DataFrame, _tariff_manager: TariffManager, country: str) -> tuple[Tariff | None, Tariff | None]:
    """Finds the cheapest flex and static tariffs from the predefined lists."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Calculating cheapest tariff comparison")
    
    flex_options = _tariff_manager.get_flex_tariffs_with_custom()
    static_options = _tariff_manager.get_static_tariffs_with_custom()
    total_costs = {}
    
    # Calculate total costs for all non-custom tariffs.
    # First, ensure we have spot price data for flexible tariff calculations.
    df = df_consumption.copy()
    if "spot_price_eur_kwh" not in df.columns:
        min_date, max_date = get_min_max_date(df)
        df_spot_prices = data_loader.get_spot_data(country, min_date, max_date)
        df = data_loader.merge_consumption_with_prices(df, df_spot_prices)

    for name, tariff in flex_options.items():
        if name == "Custom": continue
        total_costs[("flex", name)] = _tariff_manager._calculate_flexible_cost(df, tariff).sum()
    
    for name, tariff  in static_options.items():
        if name == "Custom": continue
        total_costs[("static", name)] = _tariff_manager._calculate_static_cost(df, tariff).sum()
    
    # Identify the cheapest tariffs based on the calculated costs robustly
    final_flex_tariff = None
    flex_keys = [k for k in total_costs if k[0] == "flex"]
    if flex_keys:
        cheapest_flex_key = min(flex_keys, key=total_costs.get) #type: ignore
        final_flex_tariff = flex_options.get(cheapest_flex_key[1])
    else:
        st.warning("No predefined flexible tariffs found to compare.")

    final_static_tariff = None
    static_keys = [k for k in total_costs if k[0] == "static"]
    if static_keys:
        cheapest_static_key = min(static_keys, key=total_costs.get) #type: ignore
        final_static_tariff = static_options.get(cheapest_static_key[1])
    else:
        st.warning("No predefined static tariffs found to compare.")
    
    return final_flex_tariff, final_static_tariff

def _return_tariff_selection(_tariff_manager: TariffManager) -> tuple[Tariff, Tariff]:
    """Renders tariff selection expanders in the UI for user customization."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Tariff Selection")
    
    final_tariffs = []
    options = [
        ("Flexible (Spot Price) Plan", _tariff_manager.get_flex_tariffs_with_custom()),
        ("Static (Fixed Price) Plan", _tariff_manager.get_static_tariffs_with_custom())
    ]

    for title, tariff_options in options:
        with st.expander(title, expanded=True):
            tariff_type = title.split(" ")[0]
            # Let user select a predefined tariff or 'Custom'
            selected_name = st.selectbox(f"Select {tariff_type} Tariff", options=list(tariff_options.keys()), index=len(tariff_options) - 1)
            selected_tariff = tariff_options[selected_name]
            
            # Display input fields pre-filled with the selected tariff's data
            price_kwh = st.number_input("On-Top Price (€/kWh)", value=selected_tariff.price_kwh, min_value=0.0, step=0.001, format="%.4f", key=f"{tariff_type}_price")
            price_kwh_pct = 0.0
            if tariff_type == "Flexible":
                price_kwh_pct = st.number_input("Variable Price (% of EPEX)", value=selected_tariff.price_kwh_pct, min_value=0.0, max_value=100.0, step=1.0, format="%.1f", key=f"{tariff_type}_pct")
            
            monthly_fee = st.number_input("Monthly Fee (€)", value=selected_tariff.monthly_fee, min_value=0.0, step=1.0, format="%.2f", key=f"{tariff_type}_fee")
            
            # Create a new Tariff object with the potentially modified values
            final_tariffs.append(
                Tariff(name=selected_name, type=selected_tariff.type, price_kwh=price_kwh, monthly_fee=monthly_fee, price_kwh_pct=price_kwh_pct)
            )
    return tuple(final_tariffs)

def render_sidebar_inputs(df: pd.DataFrame, tariff_manager: TariffManager) -> tuple[str, date, date, Tariff, Tariff, float]:
    """Renders all sidebar inputs and returns the configuration values."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Sidebar")
    with st.sidebar:
        st.header("Configuration")
        
        # 1. Country Selection for EPEX
        country_select = {"Austria": "at", "Germany": "de"}
        
        st.subheader("1. Country")
        selected_country = st.selectbox(label="Select country for EPEX spot prices.", options=country_select.keys(), index=0)
        awattar_country = country_select[selected_country]
        
        # 2. Analysis Period Selection
        st.subheader("2. Analysis Period")
        min_date, max_date = get_min_max_date(df)
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, format="DD.MM.YYYY")
        end_date = col2.date_input("End Date", max_date, min_value=min_date, max_value=max_date, format="DD.MM.YYYY")
        
        # 3. Tariff Plan Selection
        st.subheader("3. Tariff Plans")
        st.text("Choose a tariff or adjust parameters for a custom comparison.")
        
        compare_cheapest = st.checkbox("Compare cheapest tariffs", value=False, help="Automatically selects the most economical predefined tariffs based on your data.")
        if compare_cheapest:
            final_flex_tariff, final_static_tariff = _compare_all_tariffs(df, tariff_manager, awattar_country)
            flex_info = f"Cheapest Flex Tariff:\n\n**{final_flex_tariff.name}**" if final_flex_tariff else "No flexible tariff found."
            static_info = f"Cheapest Static Tariff:\n\n**{final_static_tariff.name}**" if final_static_tariff else "No static tariff found."
            st.info(f"{flex_info}\n\n{static_info}")

        else:
            final_flex_tariff, final_static_tariff = _return_tariff_selection(tariff_manager)
            
        # 4. Load Shifting Simulation
        st.subheader("4. Cost Simulation")
        st.markdown("Simulate shifting a percentage of your peak consumption to a cheaper hour within a +/- 2-hour window.", help="This shows the potential savings if you can be flexible with high-power activities like EV charging, dish washer, washing machine or running a heat pump.")
        shift_percentage = st.slider("Shift Peak Load (%)", min_value=0, max_value=100, value=0, step=5)

        return awattar_country, start_date, end_date, final_flex_tariff, final_static_tariff, shift_percentage

# --- Main Page Components ---

@st.cache_data(ttl=3600)
def _compute_absence_data(df: pd.DataFrame, base_threshold: float, absence_threshold: float) -> list:
    """Computes and caches the days of absence based on low daily consumption."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Absence Data")
    if 'date' not in df.columns:
        df['date'] = df['timestamp'].dt.date
    daily_consumption = df.groupby('date')["consumption_kwh"].sum()
    intervals_per_day = get_intervals_per_day(df)
    # Identify days where total consumption is below a fraction of the typical daily base load
    absence_days = daily_consumption[daily_consumption < (base_threshold * intervals_per_day * absence_threshold)].index.tolist()
    return absence_days

def render_absence_days(df: pd.DataFrame, base_threshold: float) -> pd.DataFrame:
    """Adds a sidebar option to remove days with very low consumption."""
    with st.sidebar:
        absence_days = _compute_absence_data(df, base_threshold, ABSENCE_THRESHOLD)
        if absence_days:
            st.subheader("5. Absence Handling", help=f"Remove days with consumption below {ABSENCE_THRESHOLD:.0%} of the typical base load. This can provide a more accurate analysis of your normal usage.")
            select_all = st.checkbox(f"Exclude all {len(absence_days)} detected absence days", value=False)
            default_selection = absence_days if select_all else []
            excluded_days = st.multiselect("Select specific days to exclude:", options=absence_days, default=default_selection)
            if excluded_days:
                # Filter out the selected absence days
                return df[~df["date"].isin(excluded_days)]
    return df

@st.cache_data(ttl=3600)
def render_recommendation(df: pd.DataFrame):
    """Displays the final tariff recommendation based on calculated savings."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Recommendation")

    is_granular_data = calculate_granular_data(df)
    if not is_granular_data:
        st.warning(f"⚠️ Static Plan Recommended: A flexible plan is only recommended for hourly or 15-minute data.")
        return

    savings = df["total_cost_static"].sum() - df["total_cost_flexible"].sum()

    # Calculate the proportion of peak consumption that occurs during the cheapest 25% of hours
    df["price_quantile"] = df.groupby(pd.Grouper(key="timestamp", freq="MS"))["spot_price_eur_kwh"].transform(
        lambda x: pd.qcut(x, 4, labels=False, duplicates="drop"))
    peak_total_kwh = df["peak_load_kwh"].sum()
    peak_cheap_kwh = df[df["price_quantile"] == 0]["peak_load_kwh"].sum()
    peak_ratio = peak_cheap_kwh / peak_total_kwh if peak_total_kwh > 0 else 0

    # Display the appropriate recommendation message
    if savings > 0:
        if peak_ratio > 0.4:
            additional_text = f"This is a great fit. You align **{peak_ratio:.0%}** of your peak usage with the cheapest market prices."
        else:
            additional_text = "You could save even more by shifting high-consumption activities to times with lower spot prices."
            
        st.success(f"✅ Flexible Plan Recommended: You could have saved €{savings:.2f}\n\n{additional_text}")
    else:
        st.warning(f"⚠️ Static Plan Recommended: The flexible plan would have cost €{-savings:.2f} more.\n\nA fixed price offers better cost stability for your current usage pattern.")

# --- Tab: Spot Price Analysis ---

@st.cache_data(ttl=3600)
def _compute_price_distribution_data(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Computes and caches the quartile price data for the selected resolution."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Price Distribution Data")
    price_agg_dict = {"spot_price_eur_kwh": [("q1", lambda x: x.quantile(0.25)), ("median", "median"), ("q3", lambda x: x.quantile(0.75))]}
    
    config = get_aggregation_config(df, resolution)
    df_price = df.groupby(config["grouper"]).agg(price_agg_dict).dropna()
    df_price.columns = ["Spot Price Q1", "Spot Price Median", "Spot Price Q3"]
    df_price.index = df_price.index.map(config["x_axis_map"])
    df_price.index.name = config["name"]
    df_price = df_price.reindex(config["x_axis_map"].values())
    return df_price

@st.cache_data(ttl=3600)
def _compute_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the data needed for the price heatmap."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Heatmap Data")
    return df.pivot_table(values="spot_price_eur_kwh", index=df["timestamp"].dt.month, columns=df["timestamp"].dt.hour, aggfunc="mean")

def render_price_analysis_tab(df: pd.DataFrame, static_tariff: Tariff):
    """Renders the interactive analysis of electricity spot prices."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Price Analysis Tab")
    
    resolution = st.radio("Select Time Resolution", ("Hourly", "Weekly", "Monthly"), horizontal=True, key="price_res")
    
    # Quartile Distribution Chart
    st.subheader("Distribution Over Time")
    st.markdown("This chart shows the median spot price (solid line) and the 25th to 75th percentile range (dotted lines) compared to the static tariff price. 50% of the spot prices fall in the orange shaded price range.")
    df_price = _compute_price_distribution_data(df, resolution)
    price_fig = charts.get_price_chart(df_price, pd.Series([static_tariff.price_kwh]*len(df.index)))
    st.plotly_chart(price_fig, use_container_width=True)

    # Heatmap Analysis
    st.subheader("Average Price Heatmap (Month vs. Hour)")
    st.markdown("This heatmap visualizes the average spot price for each hour across the months, helping to identify recurring daily and seasonal patterns.")
    heatmap_data = _compute_heatmap_data(df)
    heatmap_fig = charts.get_heatmap(heatmap_data)
    st.plotly_chart(heatmap_fig, use_container_width=True)

# --- Tab: Cost Comparison ---

@st.cache_data(ttl=3600)
def _compute_cost_comparison_data(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Computes and caches aggregated cost data for comparison."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Cost Comparison Data")
    freq_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "ME"}
    grouper = pd.Grouper(key="timestamp", freq=freq_map[resolution])

    summary_agg_dict: dict = {
        "Total Consumption": ("consumption_kwh", "sum"),
        "Total Flexible Cost": ("total_cost_flexible", "sum"),
        "Total Static Cost": ("total_cost_static", "sum")
    }
    df_summary = df.groupby(grouper).agg(**summary_agg_dict).reset_index()
    df_summary = df_summary[df_summary["Total Consumption"] > 0.01] # Filter out empty periods
    
    df_summary["Difference (€)"] = df_summary["Total Static Cost"] - df_summary["Total Flexible Cost"]
    df_summary["Period"] = df_summary["timestamp"].dt.strftime("%Y-%m-%d" if resolution == "Daily" else "%Y-%U" if resolution == "Weekly" else "%Y-%m")
    return df_summary

def render_cost_comparison_tab(df: pd.DataFrame):
    """Renders the content for the 'Cost Comparison' tab."""

    print(f"{datetime.now().strftime(DATE_FORMAT)}: Rendering Cost Comparison Tab")

    is_granular_data = calculate_granular_data(df)
    if not is_granular_data:
        st.info("Flexible cost comparison is only meaningful for hourly or 15-minute consumption data. The flexible tariff results are hidden.")

    resolution = st.radio("Select Time Resolution", ("Daily", "Weekly", "Monthly"), horizontal=True, key="summary_res")
    
    df_summary = _compute_cost_comparison_data(df, resolution)
    if df_summary.empty:
        st.warning("No data to display for the selected period and resolution.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Costs per Period")
        st.markdown("Compare the total energy bill for both tariff types over time, including energy usage costs and monthly fees.")
        y_cols_total = ["Total Flexible Cost", "Total Static Cost"] if is_granular_data else ["Total Static Cost"]
        colors_total = [FLEX_COLOR, STATIC_COLOR] if is_granular_data else [STATIC_COLOR]
        st.line_chart(df_summary.set_index("Period"), y=y_cols_total, y_label="Total Cost (€)", color=colors_total)

    with col2:
        st.subheader("Average Price per kWh")
        st.markdown("See the effective price per kWh after accounting for both variable costs and fixed fees.")
        df_summary["Avg. Static Price (€/kWh)"] = df_summary["Total Static Cost"] / df_summary["Total Consumption"]
        if is_granular_data:
            df_summary["Avg. Flexible Price (€/kWh)"] = df_summary["Total Flexible Cost"] / df_summary["Total Consumption"]
        
        y_cols_avg = ["Avg. Flexible Price (€/kWh)", "Avg. Static Price (€/kWh)"] if is_granular_data else ["Avg. Static Price (€/kWh)"]
        colors_avg = [FLEX_COLOR, STATIC_COLOR] if is_granular_data else [STATIC_COLOR]
        st.line_chart(df_summary.set_index("Period"), y=y_cols_avg, y_label="Average Price (€/kWh)", color=colors_avg)
            
    st.subheader("Detailed Comparison Table")
    st.text("Review the costs and savings for each period.", help="Note: For Austria, this table does not include 'Strompreisbremse' in the years before 2025, nor 'Energieabgabe' or network fees!")
    if is_granular_data:
        cols_to_show = ["Period", "Total Consumption", "Total Flexible Cost", "Total Static Cost", "Difference (€)"]
        styler = df_summary[cols_to_show].style
        styler = styler.map(lambda v: f"color: {GREEN}" if v > 0 else f"color: {RED}", subset=["Difference (€)"]) #type: ignore
        styler = styler.format({"Total Consumption": "{:.2f} kWh", "Total Flexible Cost": "€{:.2f}", "Total Static Cost": "€{:.2f}", "Difference (€)": "€{:.2f}"})
    else:
        cols_to_show = ["Period", "Total Consumption", "Total Static Cost"]
        styler = df_summary[cols_to_show].style
        styler = styler.format({"Total Consumption": "{:.2f} kWh", "Total Static Cost": "€{:.2f}"})
    st.dataframe(styler, hide_index=True, use_container_width=True)

# --- Tab: Usage Pattern Analysis ---

@st.cache_data(ttl=3600)
def _compute_usage_profile_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the proportion and average cost for each usage profile."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Usage Profile Data")
    
    load_types = ["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]
    profile_data = []
    total_kwh = df["consumption_kwh"].sum()
    intervals_per_day = get_intervals_per_day(df)
    
    for load in load_types:
        kwh = df[load].sum()
        if kwh > 0.01:
            avg_price = (df[load] * df["spot_price_eur_kwh"]).sum() / kwh
            proportion = kwh / total_kwh
            mean_daily_kwh = df[load].mean() * intervals_per_day
            profile_data.append({
                "Profile": load.replace("_kwh", "").replace("_", " ").title(),
                "kwh": kwh,
                "kwh_mean": mean_daily_kwh,
                "avg_price": avg_price,
                "proportion": proportion
            })
    return pd.DataFrame(profile_data)


@st.cache_data(ttl=3600)
def _compute_consumption_quartiles(df: pd.DataFrame, intervals_per_day: int) -> pd.DataFrame:
    """Computes and caches the usage data for the selected resolution."""
    
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Consumption Data")
    
    # Define the aggregation logic once to be reused.
    consumption_agg_dict = {
        "consumption_kwh": [
            ("q1", lambda x: x.quantile(0.25)),
            ("median", lambda x: x.quantile(0.50)),
            ("q3", lambda x: x.quantile(0.75))
        ]
    }
    
    if intervals_per_day == 24:
        resolution = "Monthly"
    else:
        resolution = "Hourly"
    
    config = get_aggregation_config(df, resolution)
    df_consumption_quartiles = df.groupby(config["grouper"]).agg(consumption_agg_dict)
    df_consumption_quartiles.columns = ["Consumption Q1", "Consumption Median", "Consumption Q3"]
    df_consumption_quartiles.index.name = config["name"]        
    df_consumption_quartiles.index = df_consumption_quartiles.index.map(config["x_axis_map"])
    df_consumption_quartiles = df_consumption_quartiles.reindex(config["x_axis_map"].values())

    return df_consumption_quartiles

@st.cache_data(ttl=3600)
def _compute_example_day(df: pd.DataFrame, random_day, group: bool = False) -> pd.DataFrame:
    """Selects a random day and return the data for plotting as well as the DataFrame itself."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Example Day")

    df_hour = df[df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.date == random_day].copy()
    
    if not df_hour.empty:
        df_hour["hour"] = df_hour["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.hour

        # Group by hour and sum the kWh columns
        if group:
            df_hour = (
                df_hour.groupby("hour")[["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]]
                .sum())
        else:
            df_hour = df_hour.set_index("timestamp")[["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]]
        
        #df_hour = df_hour[["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]]
        
        df_hour = df_hour.rename(columns={
                "base_load_kwh": "Base Load",
                "regular_load_kwh": "Regular Load",
                "peak_load_kwh": "Peak Load"
            })
    
        return df_hour
    else:
        return pd.DataFrame()


def render_usage_pattern_tab(df: pd.DataFrame, base_threshold: float, peak_threshold: float):
    """Renders the content for the 'Usage Pattern Analysis' tab."""
    
    # Allow filtering by day type
    df_filtered = df[df["consumption_kwh"] > 0].copy()
    day_filter = st.radio("Filter data by:", ("All Days", "Weekdays", "Weekends"), horizontal=True)
    if day_filter != "All Days":
        is_weekend = df_filtered["timestamp"].dt.dayofweek >= 5
        df_filtered = df_filtered[is_weekend if day_filter == "Weekends" else ~is_weekend]

    if df_filtered.empty:
        st.warning(f"No data available for {day_filter.lower()}.")
        return
        
    intervals = get_intervals_per_day(df)
    
    # Consumption Over Time
    df_consumption_day = _compute_consumption_quartiles(df_filtered, intervals)
    
    st.subheader("Consumption Over Time")
    st.markdown(
        "This chart illustrates the statistical distribution of your consumption for the selected time resolution. "
        "The solid line represents the **median (50th percentile)** consumption, while the dotted lines show the first and third Quartile."
    )

    consumption_fig = charts.get_consumption_chart(df_consumption_day)
    st.plotly_chart(consumption_fig, use_container_width=True)
    
    # Only show the detailed analysis when consumption data includes 15 minutes intervals.
    if intervals <= 24: # Hourly data or less
        st.info("Please provide data with 15-minute intervals for a more detailed usage profile analysis.")
        return

    # Consumption & Cost Profile (Marimekko Chart)
    st.subheader("Consumption & Cost Profile")
    st.markdown("This chart shows how much each consumption type (Base, Regular, Peak) contributes to your total usage, and the average spot price you paid for each.")
    profile_data = _compute_usage_profile_data(df_filtered)

    # Display Thresholds
    col1, col2 = st.columns(2)
    col1.metric("Base Load Threshold", f"{base_threshold:.3f} kWh", help="Continuous Usage", width="stretch")
    
    # The peak_threshold passed is the influenceable part. Add base for the absolute value.
    absolute_peak_threshold = base_threshold + peak_threshold
    col2.metric("Peak Sustain Threshold", f"{absolute_peak_threshold:.3f} kWh", help="A peak event starts with a sharp increase and continues for every hour consumption stays above this level.", width="stretch")

    if not profile_data.empty:
        marimekko_fig = charts.get_marimekko_chart(profile_data)      
        st.plotly_chart(marimekko_fig, use_container_width=True)

    # Example Day Breakdown
    st.subheader("Example Day Breakdown")
    st.markdown("This chart shows the classification for a random day from your data to provide a better intuition.")
    
    available_dates = df_filtered["date"].unique().tolist()
    if available_dates:
        if "random_day" not in st.session_state or st.session_state.random_day not in available_dates:
            st.session_state.random_day = random.choice(available_dates)
        
        if st.button("Show a Different Day"):
            st.session_state.random_day = random.choice(available_dates)
            st.rerun()
            
        df_day= _compute_example_day(df_filtered, st.session_state.random_day, group=False)

        st.caption(f"Displaying data for {st.session_state.random_day.strftime('%A, %Y-%m-%d')} (Total: {df_day.sum().sum():.2f} kWh)")
        st.bar_chart(df_day, color=[BASE_COLOR, PEAK_COLOR, REGULAR_COLOR], x_label="Hour of Day", y_label="Consumption (kWh)")

# --- Tab: Yearly Summary ---

@st.cache_data(ttl=3600)
def _compute_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the yearly summary of the data."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Yearly Summary")
    df["Year"] = df["timestamp"].dt.year
    yearly_summary_agg_dict = {
        "Total Consumption": ("consumption_kwh", "sum"),
        "Total Flexible Cost": ("total_cost_flexible", "sum"),
        "Total Static Cost": ("total_cost_static", "sum")
    }
    yearly_agg = df.groupby("Year").agg(**yearly_summary_agg_dict).reset_index() # type: ignore
    
    if not yearly_agg.empty and yearly_agg["Total Consumption"].sum() > 0:
        yearly_agg["Avg. Flex Price (€/kWh)"] = yearly_agg["Total Flexible Cost"] / yearly_agg["Total Consumption"]
        yearly_agg["Avg. Static Price (€/kWh)"] = yearly_agg["Total Static Cost"] / yearly_agg["Total Consumption"]
    return yearly_agg

def render_yearly_summary_tab(df: pd.DataFrame):
    """Renders the content for the 'Yearly Summary' tab."""
    st.subheader("Yearly Summary")
    yearly_agg = _compute_yearly_summary(df)
    if yearly_agg.empty:
        st.warning("No data available to generate a yearly summary.")
        return
        
    style_format: dict = {
        "Total Consumption": "{:,.2f} kWh",
        "Total Flexible Cost": "€{:,.2f}",
        "Total Static Cost": "€{:,.2f}",
        "Avg. Flex Price (€/kWh)": "€{:.4f}",
        "Avg. Static Price (€/kWh)": "€{:.4f}"
    }
    st.dataframe(yearly_agg.style.format(style_format), hide_index=True, use_container_width=True)

# --- Tab: Download Data ---

@st.cache_data(ttl=3600)
def _compute_download_data(df: pd.DataFrame) -> tuple[bytes, bytes]:
    """Prepares and caches the Excel file bytes for download."""
    print(f"{datetime.now().strftime(DATE_FORMAT)}: Computing Download Data")
    
    # Prepare data for spot price-only download (hourly resolution)
    excel_spot_data_df = df.set_index("timestamp").resample("h").first().reset_index()[["timestamp", "spot_price_eur_kwh"]].dropna()
    excel_spot_bytes = to_excel(excel_spot_data_df)
    
    # Prepare full analysis data for download
    excel_full_bytes = to_excel(df)

    return excel_full_bytes, excel_spot_bytes

def render_download_tab(df: pd.DataFrame, start_date: date, end_date: date):
    """Renders the content for the Download tab."""
    excel_full_data, excel_spot_data = _compute_download_data(df)
    
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.markdown("Download the full, detailed analysis including consumption classification and cost calculations for both tariff types.")
        st.download_button(
            label="Download Full Analysis (XLSX)",
            data=excel_full_data,
            file_name=f"electricity_analysis_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        st.markdown("Download only the hourly EPEX spot prices for the selected period.")
        st.download_button(
            label="Download Spot Prices (XLSX)",
            data=excel_spot_data,
            file_name=f"spot_prices_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# --- Footer ---
@st.cache_data
def render_footer():
    """Renders the footer with information about the project and further links."""
    st.markdown("\n\n---")
    st.markdown("""Developed by [__albortino__](https://github.com/albortino). This tool builds upon the great work of [awattar backtesting](https://awattar-backtesting.github.io), which provides further analyses.""")