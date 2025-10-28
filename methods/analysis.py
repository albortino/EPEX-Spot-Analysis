import pandas as pd
import streamlit as st
import numpy as np
from prophet import Prophet

from methods.utils import get_intervals_per_day, get_min_max_date, get_aggregation_config
from methods.config import NEGLIGABLE_KWH, BASE_QUANTILE_THRESHOLD, PEAK_QUANTILE_THRESHOLD, STD_MULTIPLE, THRESHOLD_STABLE_TREND, TODAY_IS_MAX_DATE, LOCAL_TIMEZONE
from methods.tariffs import Tariff, TariffManager
import methods.data_loader as data_loader
from methods.logger import logger


# --- Usage Classification ---

def classify_usage(df: pd.DataFrame, local_timezone: str) -> tuple[pd.DataFrame, float, float]:
    """
    Classifies hourly consumption into Base, Peak, and Regular load using a stateful approach.
    A peak event starts with a sharp increase in consumption (trigger) and continues
    as long as consumption remains above a high-usage sustain threshold.
    Base Usage is always classified. However, in every time resolution, only either peak or regular usage
    can appear. This is intentionally done instead of taking the mean regular usage before and after a peak
    as this would further bias the analysis.
    """

    if df.empty:
        return df, 0.0, 0.0

    df_c = df.copy()

    # --- Base Load Calculation ---
    # Find the most stable, lowest consumption period for each day to define the base load.
    # This is more robust than assuming base load occurs at fixed night hours.
    df_local = df.copy()
    df_local["timestamp_local"] = df_local["timestamp"].dt.tz_convert(local_timezone)
    indexed_consumption = df_local.set_index("timestamp_local")["consumption_kwh"]
    
    # Calculate rolling metrics over a 4-hour window to find stable periods.
    # Define rolling window as 4h. In case data logs every 15 minutes multiply it with a factor.
    intervals_per_day = get_intervals_per_day(df_local)
    rolling_window = 4 * intervals_per_day // 24
     
    df_local["rolling_std"] = indexed_consumption.rolling(window=rolling_window, center=True).std().values
    df_local["rolling_mean"] = indexed_consumption.rolling(window=rolling_window, center=True).quantile(BASE_QUANTILE_THRESHOLD).values
    
    # For each day, find the point with the minimum rolling std dev (stability).
    # In case of a tie, choose the one with the lower rolling mean (consumption).
    stable_periods = df_local.dropna(subset=["rolling_std", "rolling_mean"]).sort_values(by=["rolling_std", "rolling_mean"])
    daily_base_load_points = stable_periods.groupby(df_local['timestamp_local'].dt.date).first()
    base_load_threshold = daily_base_load_points["rolling_mean"].mean() if not daily_base_load_points.empty else 0.0

    # --- Peak Load Calculation ---
    # Influencable load is everything above the base load.
    influenceable_load = (df_c["consumption_kwh"] - base_load_threshold).clip(lower=0)
    consumption_diff = df_c["consumption_kwh"].diff().fillna(0).astype(float)
    
    # Sustain Threshold: A high level of consumption. We use a quantile on the *influenceable* load.
    # This represents consumption significantly above the base.
    peak_sustain_threshold_influenceable = influenceable_load[influenceable_load > NEGLIGABLE_KWH].quantile(PEAK_QUANTILE_THRESHOLD)
    peak_sustain_threshold_influenceable = 0.0 if pd.isna(peak_sustain_threshold_influenceable) else peak_sustain_threshold_influenceable
    peak_sustain_threshold_absolute = base_load_threshold + peak_sustain_threshold_influenceable

    # Trigger Threshold: A sharp increase from the previous hour (based on std dev of positive changes).
    positive_diffs = consumption_diff[consumption_diff > NEGLIGABLE_KWH]
    peak_trigger_threshold = positive_diffs.std() * STD_MULTIPLE if not positive_diffs.empty else 0.0
    peak_trigger_threshold = 0.0 if pd.isna(peak_trigger_threshold) else peak_trigger_threshold

    # If sustain threshold is negligible, classify all influenceable load as "regular".
    if peak_sustain_threshold_influenceable < NEGLIGABLE_KWH:
        df_c["base_load_kwh"] = df_c["consumption_kwh"].clip(upper=base_load_threshold)
        df_c["peak_load_kwh"] = 0.0
        df_c["regular_load_kwh"] = influenceable_load
        return df_c, base_load_threshold, peak_sustain_threshold_influenceable

    # --- Classification Loop ---
    is_peak_list = []
    in_peak_state = False
    for i in range(len(df_c)):
        is_trigger = consumption_diff.iloc[i] > peak_trigger_threshold
        is_sustained = df_c["consumption_kwh"].iloc[i] > peak_sustain_threshold_absolute
        
        # Start a new peak if a sharp increase pushes consumption above the sustain level
        if not in_peak_state and is_trigger and is_sustained:
            in_peak_state = True
        # Already in a peak. End the peak if consumption falls below the sustain level
        elif in_peak_state and not is_sustained:
            in_peak_state = False
        is_peak_list.append(in_peak_state)
    
    is_peak = pd.Series(is_peak_list, index=df_c.index)

    #  Assign load types based on classification
    df_c["base_load_kwh"] = df_c["consumption_kwh"].clip(upper=base_load_threshold)
    df_c["regular_load_kwh"] = influenceable_load.where(~is_peak, 0)
    df_c["peak_load_kwh"] = influenceable_load.where(is_peak, 0)

    # --- Refinement Step ---
    # Re-assign a portion of the peak load back to regular load.
    # This reflects that even during a peak, there's underlying regular consumption.
    # The amount re-assigned is the average regular load from non-peak intervals of that day.
    if is_peak.any():
        df_c['date'] = df_c['timestamp'].dt.date
        # Calculate the average regular load for non-peak intervals on each day.
        daily_avg_regular_load = df_c[~is_peak & (df_c['regular_load_kwh'] > 0)].groupby('date')['regular_load_kwh'].mean()

        # Map the daily average back to each row and fill days with no regular load with 0.
        df_c['avg_regular_for_day'] = df_c['date'].map(daily_avg_regular_load).fillna(0)

        # Shift the calculated average amount from peak to regular load for peak intervals.
        df_c.loc[is_peak, 'regular_load_kwh'] = df_c['avg_regular_for_day']
        df_c.loc[is_peak, 'peak_load_kwh'] = (df_c['peak_load_kwh'] - df_c['avg_regular_for_day']).clip(lower=0)
        df_c = df_c.drop(columns=['date', 'avg_regular_for_day'])

    return df_c, base_load_threshold, peak_sustain_threshold_influenceable

# --- Peak Shifting Simulation ---
@st.cache_data(ttl=60*10)
def simulate_peak_shifting(df: pd.DataFrame, shift_percentage: float) -> pd.DataFrame:
    """
    Simulates shifting a percentage of peak load from the most expensive times
    to the cheapest hour within a +/- 2-hour window.
    """
    if shift_percentage == 0:
        return df

    df_sim = df.copy()
    total_peak_kwh = df_sim["peak_load_kwh"].sum()
    kwh_to_shift_total = total_peak_kwh * (shift_percentage / 100.0)
    if kwh_to_shift_total <= 0: return df_sim

    peaks = df_sim[df_sim["peak_load_kwh"] > 0.001].copy()
    if peaks.empty: return df_sim
    
    peaks["peak_cost"] = peaks["peak_load_kwh"] * peaks["spot_price_eur_kwh"]
    sorted_peaks = peaks.sort_values(by="peak_cost", ascending=False)

    shifted_load_additions = pd.Series(0.0, index=df_sim.index)

    for peak_idx, peak_row in sorted_peaks.iterrows():
        if kwh_to_shift_total <= 0: break
        
        current_timestamp = peak_row["timestamp"]
        window_df = df_sim[
            (df_sim["timestamp"] >= current_timestamp - pd.Timedelta(hours=2)) & 
            (df_sim["timestamp"] <= current_timestamp + pd.Timedelta(hours=2))
        ]
        if window_df.empty: continue
        
        cheapest_hour_in_window = window_df.loc[window_df["spot_price_eur_kwh"].idxmin()]
        
        if cheapest_hour_in_window["spot_price_eur_kwh"] < peak_row["spot_price_eur_kwh"]:
            kwh_in_this_peak = df_sim.at[peak_idx, "peak_load_kwh"]
            kwh_to_shift_now = min(kwh_in_this_peak, kwh_to_shift_total)
            
            df_sim.at[peak_idx, "peak_load_kwh"] -= kwh_to_shift_now
            shifted_load_additions.loc[cheapest_hour_in_window.name] += kwh_to_shift_now
            kwh_to_shift_total -= kwh_to_shift_now

    df_sim["regular_load_kwh"] += shifted_load_additions
    df_sim["consumption_kwh"] = df_sim["base_load_kwh"] + df_sim["regular_load_kwh"] + df_sim["peak_load_kwh"]
    
    return df_sim

# --- Data Computation for UI Components ---

@st.cache_data(ttl=60*10)
def compare_all_tariffs(_tariff_manager: TariffManager, df_consumption: pd.DataFrame, country: str) -> tuple[Tariff | None, Tariff | None]:
    """Finds the cheapest flex and static tariffs from the predefined lists."""
    logger.log("Calculating cheapest tariff comparison")
    
    flex_options = _tariff_manager.get_flex_tariffs_with_custom()
    static_options = _tariff_manager.get_static_tariffs_with_custom()
    total_costs = {}
    
    # Calculate total costs for all non-custom tariffs.
    # First, ensure we have spot price data for flexible tariff calculations.
    df = df_consumption.copy()
    if "spot_price_eur_kwh" not in df.columns:
        min_date, max_date = get_min_max_date(df, today_as_max=TODAY_IS_MAX_DATE)
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

    final_static_tariff = None
    static_keys = [k for k in total_costs if k[0] == "static"]
    if static_keys:
        cheapest_static_key = min(static_keys, key=total_costs.get) #type: ignore
        final_static_tariff = static_options.get(cheapest_static_key[1])
    
    return final_flex_tariff, final_static_tariff

@st.cache_data(ttl=3600)
def compute_absence_data(df: pd.DataFrame, base_threshold: float, absence_threshold: float) -> list:
    """Computes and caches the days of absence based on low daily consumption."""
    logger.log("Computing Absence Data")
    if 'date' not in df.columns:
        df['date'] = df['timestamp'].dt.date
    daily_consumption = df.groupby('date')["consumption_kwh"].sum()
    intervals_per_day = get_intervals_per_day(df)
    # Identify days where total consumption is below a fraction of the typical daily base load
    absence_days = daily_consumption[daily_consumption < (base_threshold * intervals_per_day * absence_threshold)].index.tolist()
    return absence_days

@st.cache_data(ttl=3600)
def compute_price_distribution_data(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Computes and caches the quartile price data for the selected resolution."""
    logger.log("Computing Price Distribution Data")
    price_agg_dict = {"spot_price_eur_kwh": [("q1", lambda x: x.quantile(0.25)), ("median", "median"), ("mean", "mean"), ("q3", lambda x: x.quantile(0.75))]}
    
    config = get_aggregation_config(df, resolution)
    # Drop NA before aggregation to avoid issues with empty groups
    df_price = df.dropna(subset=["spot_price_eur_kwh"]).groupby(config["grouper"]).agg(price_agg_dict)
    df_price.columns = ["Spot Price Q1", "Spot Price Median", "Spot Price Mean", "Spot Price Q3"]
    df_price.index = df_price.index.map(config["x_axis_map"])
    df_price.index.name = config["name"]
    # Reindex to ensure correct chronological order (e.g., Jan, Feb, Mar...)
    # Then, drop any rows that are all NA, which happens for months with no data.
    df_price = df_price.reindex(config["x_axis_map"].values()).dropna(how="all")
    return df_price

@st.cache_data(ttl=3600)
def compute_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the data needed for the price heatmap."""
    logger.log("Computing Heatmap Data")
    df_pvt = df.pivot_table(values="spot_price_eur_kwh", index=df["timestamp"].dt.month, columns=df["timestamp"].dt.hour, aggfunc="mean")
    
    # Create a copy and convert integer column names to strings for plotly compatibility.
    df_pvt = df_pvt.copy()
    df_pvt.columns = df_pvt.columns.map(str)
    
    return df_pvt

@st.cache_data(ttl=3600)
def compute_cost_comparison_data(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Computes and caches aggregated cost data for comparison."""
    logger.log("Computing Cost Comparison Data")
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
    df_summary["Period"] = df_summary["timestamp"].dt.strftime("%Y-%m-%d" if resolution == "Daily" else "%G-W%V" if resolution == "Weekly" else "%Y-%m")
    return df_summary

@st.cache_data(ttl=3600)
def compute_cumulative_savings_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes cumulative savings over time."""
    logger.log("Computing Cumulative Savings Data")
    df_savings = df[["timestamp", "total_cost_static", "total_cost_flexible"]].copy()
    df_savings = df_savings.sort_values("timestamp")
    df_savings["savings"] = df_savings["total_cost_static"] - df_savings["total_cost_flexible"]
    df_savings["cumulative_savings"] = df_savings["savings"].cumsum()
    return df_savings

@st.cache_data(ttl=3600)
def compute_usage_profile_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the proportion and average cost for each usage profile."""
    logger.log("Computing Usage Profile Data")
    
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
def compute_consumption_quartiles(df: pd.DataFrame, intervals_per_day: int) -> pd.DataFrame:
    """Computes and caches the usage data for the selected resolution."""
    logger.log("Computing Consumption Data")
    
    consumption_agg_dict = { "consumption_kwh": [ ("q1", lambda x: x.quantile(0.25)), ("median", "median"), ("q3", lambda x: x.quantile(0.75)) ] }
    resolution = "Hourly" if intervals_per_day > 1 else "Daily" # Simplified logic
    
    config = get_aggregation_config(df, resolution)
    df_consumption_quartiles = df.dropna(subset=["consumption_kwh"]).groupby(config["grouper"]).agg(consumption_agg_dict)
    df_consumption_quartiles.columns = ["Consumption Q1", "Consumption Median", "Consumption Q3"]
    df_consumption_quartiles.index.name = config["name"]        
    df_consumption_quartiles.index = df_consumption_quartiles.index.map(config["x_axis_map"])
    df_consumption_quartiles = df_consumption_quartiles.reindex(config["x_axis_map"].values()).dropna(how="all")
    
    return df_consumption_quartiles

@st.cache_data(ttl=3600)
def compute_example_day(df: pd.DataFrame, random_day, group: bool = False) -> pd.DataFrame:
    """Selects a random day and return the data for plotting."""
    logger.log("Computing Example Day")
    df_hour = df[df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.date == random_day].copy()
    
    if not df_hour.empty:
        df_hour["hour"] = df_hour["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.hour
        if group:
            df_hour = df_hour.groupby("hour")[["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]].sum()
        else:
            df_hour = df_hour.set_index("timestamp")[["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]]
                
        df_hour = df_hour.rename(columns={"base_load_kwh": "Base Load", "regular_load_kwh": "Regular Load", "peak_load_kwh": "Peak Load"})
        return df_hour
    return pd.DataFrame()

@st.cache_resource(ttl=3600, show_spinner=True)
def fit_forecast_model(df: pd.DataFrame) -> tuple[Prophet|None, pd.DataFrame]:
    """Fits the Prophet model on provided data."""
    if "timestamp" not in df.columns: return None, pd.DataFrame()
    
    df_daily = df.resample("D", on="timestamp")["consumption_kwh"].sum().reset_index()
    df_daily = df_daily.rename(columns={"timestamp": "ds", "consumption_kwh": "y"})
    df_daily["ds"] = df_daily["ds"].dt.tz_localize(None)
    df_daily.loc[df_daily["y"] == 0, "y"] = pd.NA

    if len(df_daily) < 30: return None, pd.DataFrame()

    holidays = pd.DataFrame({'holiday': 'absence', 'ds': df_daily[df_daily["y"].isna()]["ds"], 'lower_window': 0, 'upper_window': 0})
    
    use_yearly = len(df_daily) >= 365
    model = Prophet(holidays=holidays, yearly_seasonality=use_yearly)
    model.add_country_holidays(country_name="AT")
    if not use_yearly: model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    
    model.fit(df_daily.dropna())
    return model, df_daily

@st.cache_data(ttl=3600)
def compute_consumption_trend_and_forecast(df: pd.DataFrame, forcast_periods: int = 90):
    """Analyzes and forecasts daily consumption using Prophet."""
    logger.log("Computing Consumption Trend and Forecast with Prophet")
    
    model, df_daily = fit_forecast_model(df)
    if model is None: return None

    future = model.make_future_dataframe(periods=forcast_periods, freq="D")
    forecast = model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper", "trend"]:
        if col in forecast.columns: forecast[col] = forecast[col].clip(0)
            
    historical_trend = forecast[forecast['ds'].isin(df_daily['ds'])]['trend']
    slope, _ = np.polyfit(np.arange(len(historical_trend)), historical_trend.values, 1)
    total_change = slope * len(historical_trend)
    avg_consumption = df_daily['y'].mean()
    percent_change = (total_change / avg_consumption) * 100 if avg_consumption > 0 else 0

    if abs(percent_change) < THRESHOLD_STABLE_TREND: trend_description = "Stable"
    elif percent_change > 0: trend_description = "Increasing"
    else: trend_description = "Decreasing"
        
    return df_daily, forecast, trend_description, percent_change

@st.cache_data(ttl=3600)
def compute_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the yearly summary of the data."""
    logger.log("Computing Yearly Summary")
    df["Year"] = df["timestamp"].dt.year
    summary_agg = { "Total Consumption": ("consumption_kwh", "sum"), "Total Static Cost": ("total_cost_static", "sum") }
    
    is_granular = "total_cost_flexible" in df.columns
    if is_granular: summary_agg["Total Flexible Cost"] = ("total_cost_flexible", "sum")
        
    yearly_agg = df.groupby("Year").agg(**summary_agg).reset_index()
    
    if not yearly_agg.empty and yearly_agg["Total Consumption"].sum() > 0:
        yearly_agg["Avg Static Price"] = yearly_agg["Total Static Cost"] / yearly_agg["Total Consumption"]
        if is_granular: yearly_agg["Avg. Flex Price"] = yearly_agg["Total Flexible Cost"] / yearly_agg["Total Consumption"]

    return yearly_agg
