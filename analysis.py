# analysis.py

import pandas as pd
from utils import get_intervals_per_day

def classify_usage(df: pd.DataFrame, local_timezone: str) -> tuple[pd.DataFrame, float, float]:
    """
    Classifies hourly consumption into Base, Peak, and Regular load using a stateful approach.
    A peak event starts with a sharp increase in consumption (trigger) and continues
    as long as consumption remains above a high-usage sustain threshold.
    Prefers data in 15-minute intervals but returns clusters for every hour.
    """
    NEGLIGABLE_KWH = 0.05
    BASE_QUANTILE_THRESHOLD = 0.9
    PEAK_QUANTILE_THRESHOLD = 0.75
    STD_MULTIPLE = 1.25

    if df.empty:
        return df, 0.0, 0.0

    df_c = df.copy()

    # --- Base Load Calculation ---
    df_local = df.copy()
    df_local["timestamp_local"] = df_local["timestamp"].dt.tz_convert(local_timezone)
    indexed_consumption = df_local.set_index("timestamp_local")["consumption_kwh"]
    
    intervals_per_day = get_intervals_per_day(df_local)
    rolling_window = 4 * intervals_per_day // 24
     
    df_local["rolling_std"] = indexed_consumption.rolling(window=rolling_window, center=True).std().values
    df_local["rolling_mean"] = indexed_consumption.rolling(window=rolling_window, center=True).quantile(BASE_QUANTILE_THRESHOLD).values
    
    stable_periods = df_local.dropna(subset=["rolling_std", "rolling_mean"]).sort_values(by=["rolling_std", "rolling_mean"])
    daily_base_load_points = stable_periods.groupby(df_local['timestamp_local'].dt.date).first()
    base_load_threshold = daily_base_load_points["rolling_mean"].mean() if not daily_base_load_points.empty else 0.0

    # --- Peak Load Calculation ---
    influenceable_load = (df_c["consumption_kwh"] - base_load_threshold).clip(lower=0)
    consumption_diff = df_c["consumption_kwh"].diff().fillna(0)
    
    peak_sustain_threshold_influenceable = influenceable_load[influenceable_load > NEGLIGABLE_KWH].quantile(PEAK_QUANTILE_THRESHOLD)
    peak_sustain_threshold_influenceable = 0.0 if pd.isna(peak_sustain_threshold_influenceable) else peak_sustain_threshold_influenceable
    peak_sustain_threshold_absolute = base_load_threshold + peak_sustain_threshold_influenceable

    positive_diffs = consumption_diff[consumption_diff > NEGLIGABLE_KWH]
    peak_trigger_threshold = positive_diffs.std() * STD_MULTIPLE if not positive_diffs.empty else 0.0
    peak_trigger_threshold = 0.0 if pd.isna(peak_trigger_threshold) else peak_trigger_threshold

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
        if not in_peak_state and is_trigger and is_sustained:
            in_peak_state = True
        elif in_peak_state and not is_sustained:
            in_peak_state = False
        is_peak_list.append(in_peak_state)
    
    is_peak = pd.Series(is_peak_list, index=df_c.index)

    df_c["base_load_kwh"] = df_c["consumption_kwh"].clip(upper=base_load_threshold)
    df_c["peak_load_kwh"] = influenceable_load.where(is_peak, 0)
    df_c["regular_load_kwh"] = influenceable_load.where(~is_peak, 0)
        
    return df_c, base_load_threshold, peak_sustain_threshold_influenceable

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
            kwh_in_this_peak = df_sim.loc[peak_idx, "peak_load_kwh"]
            kwh_to_shift_now = min(kwh_in_this_peak, kwh_to_shift_total)
            
            df_sim.loc[peak_idx, "peak_load_kwh"] -= kwh_to_shift_now
            shifted_load_additions.loc[cheapest_hour_in_window.name] += kwh_to_shift_now
            kwh_to_shift_total -= kwh_to_shift_now

    df_sim["regular_load_kwh"] += shifted_load_additions
    df_sim["consumption_kwh"] = df_sim["base_load_kwh"] + df_sim["regular_load_kwh"] + df_sim["peak_load_kwh"]
    
    return df_sim