import pandas as pd
import streamlit as st
import numpy as np
from prophet import Prophet

from methods.utils import get_intervals_per_day, get_min_max_date, get_aggregation_config
from methods.config import NEGLIGABLE_KWH, BASE_QUANTILE_THRESHOLD, PEAK_QUANTILE_THRESHOLD, STD_MULTIPLE, THRESHOLD_STABLE_TREND, TODAY_IS_MAX_DATE, LOCAL_TIMEZONE, FFT_BASE_HARMONICS, OVERNIGHT_HOURS, PEAK_SUSTAIN_INTERVALS
from methods.tariffs import Tariff, TariffManager
import methods.data_loader as data_loader
from methods.logger import logger


# --- Usage Classification ---

def classify_usage(df: pd.DataFrame, local_timezone: str) -> tuple[pd.DataFrame, float, float]:
    """
    Classifies consumption into Base, Regular, and Peak load using signal decomposition.

    Base Load is extracted per day via FFT: the DC component plus a small number of
    low-frequency harmonics represent the slow, always-on background draw (fridge,
    modem, standby). The per-day signals are then anchored to a global overnight
    baseline so the base is comparable across days.

    After subtracting base, the residual (influenceable load) is split into Regular
    and Peak using two complementary conditions OR'd together:
      A) Rapid-onset trigger: sharp derivative spike that pushes the residual above
         the high-usage sustain threshold (catches kettles, ovens switching on).
      B) Sustained amplitude: residual stays above the sustain threshold for at least
         PEAK_SUSTAIN_INTERVALS consecutive intervals without a trigger (catches EV
         chargers, oven bake sessions with gradual onset).

    During peak intervals the existing re-attribution step carves back an estimate of
    the underlying regular load so all three buckets are always consistent.
    """

    if df.empty:
        return df, 0.0, 0.0

    df_c = df.copy()
    df_local = df_c.copy()
    df_local["timestamp_local"] = df_local["timestamp"].dt.tz_convert(local_timezone)
    intervals_per_day = get_intervals_per_day(df_local)

    # --- Step 1: Per-day FFT base signal extraction ---
    # For each calendar day, keep only the DC component + FFT_BASE_HARMONICS low
    # harmonics. This reconstructs the slow always-on floor for that day.
    df_local["date"] = df_local["timestamp_local"].dt.date
    base_signal_parts = []

    for day, group in df_local.groupby("date", sort=True):
        y = group["consumption_kwh"].values.astype(float)
        n = len(y)
        if n < 2:
            base_signal_parts.append(pd.Series(y, index=group.index))
            continue

        coeffs = np.fft.rfft(y)
        # Zero out every component above DC + FFT_BASE_HARMONICS
        cutoff = 1 + FFT_BASE_HARMONICS
        coeffs[cutoff:] = 0.0
        base_day = np.fft.irfft(coeffs, n=n)
        # Clip negatives and values above the actual consumption
        base_day = np.clip(base_day, 0, y.max() if y.max() > 0 else 0)
        base_signal_parts.append(pd.Series(base_day, index=group.index))

    base_signal = pd.concat(base_signal_parts).reindex(df_c.index)

    # --- Step 2: Global overnight anchor ---
    # Scale the per-day base signals so the base level is globally stable and
    # comparable across days. Anchor = 0.95 quantile of raw overnight consumption.
    oh_start, oh_end = OVERNIGHT_HOURS
    overnight_mask = (
        df_local["timestamp_local"].dt.hour >= oh_start
    ) & (
        df_local["timestamp_local"].dt.hour < oh_end
    )
    overnight_vals = df_c.loc[overnight_mask, "consumption_kwh"]
    overnight_anchor = overnight_vals.quantile(0.95) if not overnight_vals.empty else base_signal.mean()
    overnight_anchor = float(overnight_anchor) if not pd.isna(overnight_anchor) else 0.0

    # Rescale each day's FFT base so its mean equals overnight_anchor.
    # Days with a near-zero reconstructed base get overnight_anchor directly.
    rescaled_parts = []
    for day, group in df_local.groupby("date", sort=True):
        seg = base_signal.loc[group.index]
        seg_mean = seg.mean()
        if seg_mean > NEGLIGABLE_KWH:
            seg = seg * (overnight_anchor / seg_mean)
        else:
            seg = pd.Series(overnight_anchor, index=group.index)
        # Never let base exceed actual consumption
        seg = seg.clip(upper=df_c.loc[group.index, "consumption_kwh"])
        rescaled_parts.append(seg)

    base_signal = pd.concat(rescaled_parts).reindex(df_c.index)

    # --- Step 3: Residual (influenceable load) ---
    residual = (df_c["consumption_kwh"] - base_signal).clip(lower=0)

    # --- Step 4: Peak detection on the residual ---
    # Global sustain threshold: high-usage quantile of the non-negligible residual.
    significant_residual = residual[residual > NEGLIGABLE_KWH]
    peak_sustain_threshold = significant_residual.quantile(PEAK_QUANTILE_THRESHOLD) if not significant_residual.empty else 0.0
    peak_sustain_threshold = 0.0 if pd.isna(peak_sustain_threshold) else float(peak_sustain_threshold)

    # If the residual spread is negligible, nothing is peak — all is regular.
    if peak_sustain_threshold < NEGLIGABLE_KWH:
        df_c["base_load_kwh"] = base_signal
        df_c["regular_load_kwh"] = residual
        df_c["peak_load_kwh"] = 0.0
        return df_c, overnight_anchor, peak_sustain_threshold

    residual_diff = residual.diff().fillna(0).astype(float)
    positive_diffs = residual_diff[residual_diff > NEGLIGABLE_KWH]
    trigger_threshold = positive_diffs.std() * STD_MULTIPLE if not positive_diffs.empty else 0.0
    trigger_threshold = 0.0 if pd.isna(trigger_threshold) else float(trigger_threshold)

    # Condition A — rapid-onset trigger + sustain (stateful)
    is_peak_trigger = []
    in_peak_state = False
    for i in range(len(df_c)):
        is_trigger = residual_diff.iloc[i] > trigger_threshold
        is_sustained = residual.iloc[i] > peak_sustain_threshold
        if not in_peak_state and is_trigger and is_sustained:
            in_peak_state = True
        elif in_peak_state and not is_sustained:
            in_peak_state = False
        is_peak_trigger.append(in_peak_state)
    is_peak_trigger = pd.Series(is_peak_trigger, index=df_c.index)

    # Condition B — sustained amplitude: residual above threshold for >= PEAK_SUSTAIN_INTERVALS
    # consecutive intervals (no trigger required). Captures gradual-onset high draws.
    above_threshold = residual > peak_sustain_threshold
    is_peak_sustained = pd.Series(False, index=df_c.index)
    if PEAK_SUSTAIN_INTERVALS <= 1:
        # Every interval above threshold qualifies immediately
        is_peak_sustained = above_threshold
    else:
        # Run-length: mark a run as peak only once it has lasted long enough
        run_len = above_threshold.groupby((above_threshold != above_threshold.shift()).cumsum()).transform("sum")
        is_peak_sustained = above_threshold & (run_len >= PEAK_SUSTAIN_INTERVALS)

    is_peak = is_peak_trigger | is_peak_sustained

    # --- Step 5: Assign load columns ---
    df_c["base_load_kwh"] = base_signal
    df_c["regular_load_kwh"] = residual.where(~is_peak, 0)
    df_c["peak_load_kwh"] = residual.where(is_peak, 0)

    # --- Step 6: Refinement — re-attribute base + regular from within peak intervals ---
    # During a peak event the underlying regular load is still running. Estimate it
    # from surrounding non-peak intervals and shift that portion out of peak_load_kwh.
    if is_peak.any():
        df_c["_date"] = df_c["timestamp"].dt.date
        daily_avg_regular = df_c[~is_peak & (df_c["regular_load_kwh"] > 0)].groupby("_date")["regular_load_kwh"].mean()
        df_c["_avg_regular"] = df_c["_date"].map(daily_avg_regular).fillna(0)
        df_c.loc[is_peak, "regular_load_kwh"] = df_c["_avg_regular"]
        df_c.loc[is_peak, "peak_load_kwh"] = (df_c["peak_load_kwh"] - df_c["_avg_regular"]).clip(lower=0)
        df_c = df_c.drop(columns=["_date", "_avg_regular"])

    return df_c, overnight_anchor, peak_sustain_threshold

# --- Peak Shifting Simulation ---
@st.cache_data(ttl=60*10)
def simulate_peak_shifting(df: pd.DataFrame, shift_percentage: float, window_hours: int = 2) -> pd.DataFrame:
    """
    Simulates shifting a percentage of peak load from the most expensive times
    to a cheaper available interval within a symmetric time window. The receiving
    interval is capped at the source peak energy, preventing implausible spikes.
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
            (df_sim["timestamp"] >= current_timestamp - pd.Timedelta(hours=window_hours)) &
            (df_sim["timestamp"] <= current_timestamp + pd.Timedelta(hours=window_hours))
        ].copy()
        window_df = window_df[window_df.index != peak_idx]
        if window_df.empty: continue
        
        cheapest_hour_in_window = window_df.loc[window_df["spot_price_eur_kwh"].idxmin()]
        
        if cheapest_hour_in_window["spot_price_eur_kwh"] < peak_row["spot_price_eur_kwh"]:
            kwh_in_this_peak = df_sim.at[peak_idx, "peak_load_kwh"]
            receiving_capacity = peak_row["peak_load_kwh"]
            kwh_to_shift_now = min(kwh_in_this_peak, kwh_to_shift_total, receiving_capacity)
            
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
    price_agg_dict = {
        "spot_price_eur_kwh": [
            ("q1", lambda x: x.quantile(0.25)),
            ("median", "median"),
            ("mean", "mean"),
            ("q3", lambda x: x.quantile(0.75))
        ]
    }
    
    df_valid = df.dropna(subset=["spot_price_eur_kwh"]).copy()
    if df_valid.empty:
        return pd.DataFrame(columns=["Spot Price Q1", "Spot Price Median", "Spot Price Mean", "Spot Price Q3"])

    if resolution == "Hourly":
        config = get_aggregation_config(df_valid, resolution)
        df_price = df_valid.groupby(config["grouper"]).agg(price_agg_dict)
        df_price.columns = ["Spot Price Q1", "Spot Price Median", "Spot Price Mean", "Spot Price Q3"]
        df_price.index = df_price.index.map(config["x_axis_map"])
        df_price.index.name = config["name"]
        df_price = df_price.reindex(config["x_axis_map"].values()).dropna(how="all")
    else:
        freq_map = {"Weekly": "W-MON", "Monthly": "ME"}
        date_format = "%G-W%V" if resolution == "Weekly" else "%Y-%m"
        axis_name = "Week" if resolution == "Weekly" else "Month"

        grouper = pd.Grouper(key="timestamp", freq=freq_map.get(resolution, "ME"))
        df_price = df_valid.groupby(grouper).agg(price_agg_dict)
        df_price.columns = ["Spot Price Q1", "Spot Price Median", "Spot Price Mean", "Spot Price Q3"]
        df_price = df_price.dropna(how="all")
        df_price.index = df_price.index.strftime(date_format)
        df_price.index.name = axis_name

    return df_price

@st.cache_data(ttl=3600)
def compute_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the data needed for the price heatmap."""
    logger.log("Computing Heatmap Data")
    # Use YYYY-MM string as index so rows are ordered chronologically across years.
    year_month = df["timestamp"].dt.strftime("%Y-%m")
    df_pvt = df.pivot_table(
        values="spot_price_eur_kwh",
        index=year_month,
        columns=df["timestamp"].dt.hour,
        aggfunc="mean"
    )
    df_pvt = df_pvt.copy()
    df_pvt.columns = df_pvt.columns.map(str)
    # Ensure chronological row order (string comparison on YYYY-MM is correct).
    df_pvt = df_pvt.sort_index()
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

    # Calculate average prices
    df_summary["Avg. Static Price"] = df_summary["Total Static Cost"] / df_summary["Total Consumption"]
    if "Total Flexible Cost" in df_summary.columns:
        df_summary["Avg. Flex Price"] = df_summary["Total Flexible Cost"] / df_summary["Total Consumption"]

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
    df_agg = df.copy()

    # If data is more granular than hourly, resample to hourly sums first.
    if intervals_per_day > 24:
        df_agg = df_agg.set_index('timestamp').resample('h').agg({
            'consumption_kwh': 'sum'
        }).reset_index()
    
    consumption_agg_dict = { "consumption_kwh": [ ("q1", lambda x: x.quantile(0.25)), ("median", "median"), ("q3", lambda x: x.quantile(0.75)) ] }
    resolution = "Hourly" if intervals_per_day > 1 else "Daily" # Simplified logic
    
    config = get_aggregation_config(df_agg, resolution)
    df_consumption_quartiles = df_agg.dropna(subset=["consumption_kwh"]).groupby(config["grouper"]).agg(consumption_agg_dict)
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
        if is_granular:
            yearly_agg["Difference (€)"] = yearly_agg["Total Static Cost"] - yearly_agg["Total Flexible Cost"]
        yearly_agg["Avg. Static Price"] = yearly_agg["Total Static Cost"] / yearly_agg["Total Consumption"]
        if is_granular: yearly_agg["Avg. Flex Price"] = yearly_agg["Total Flexible Cost"] / yearly_agg["Total Consumption"]

    return yearly_agg


@st.cache_data(ttl=3600)
def compute_peak_timing_score(df: pd.DataFrame) -> float:
    """
    Fraction of peak load that falls in the cheapest 25% of spot-price hours
    (bottom Q1 quantile, computed per calendar month).

    Returns a value in [0.0, 1.0]. Returns 0.0 when there is no peak load or
    no spot-price data — e.g. for coarse daily data without load classification.
    """
    if "peak_load_kwh" not in df.columns or "spot_price_eur_kwh" not in df.columns:
        return 0.0
    df = df.copy()
    df["_price_q"] = df.groupby(pd.Grouper(key="timestamp", freq="MS"))["spot_price_eur_kwh"].transform(
        lambda x: pd.qcut(x, 4, labels=False, duplicates="drop")
    )
    peak_total = df["peak_load_kwh"].sum()
    if peak_total <= 0:
        return 0.0
    peak_cheap = df[df["_price_q"] == 0]["peak_load_kwh"].sum()
    return float(np.clip(peak_cheap / peak_total, 0.0, 1.0))
