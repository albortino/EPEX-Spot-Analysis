import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date

# --- Page and App Configuration ---
st.set_page_config(layout="wide")
st.title("Electricity Tariff Comparison Dashboard")
st.markdown("Analyze your consumption, compare flexible vs. static tariffs, and understand your usage patterns.")


# --- Constants and Configuration ---
LOCAL_TIMEZONE = "Europe/Vienna" # Timezone for your consumption data, used to convert to UTC

# --- Data Fetching and Processing Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_spot_data(start: date, end: date) -> pd.DataFrame:
    """Fetches electricity market data from the aWATTar API for a given date range."""
    base_url = "https://api.awattar.de/v1/marketdata"
    
    # Add one day to the end date because the API's 'end' parameter is exclusive
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end + pd.Timedelta(days=1), datetime.min.time())
    
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    params = {'start': start_ts, 'end': end_ts}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['start_timestamp'], unit='ms', utc=True)
        df['spot_price_eur_kwh'] = df['marketprice'] / 1000
        return df[['timestamp', 'spot_price_eur_kwh']]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch spot price data: {e}")
    except (KeyError, IndexError, TypeError):
        st.error("Received unexpected data from the spot price API. It may be down or have no data for the selected range.")
    return pd.DataFrame()


def process_consumption_data(uploaded_file) -> pd.DataFrame:
    """Loads and processes the user's consumption CSV, converting it to hourly UTC."""
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        # Load the CSV data from the user
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8', dayfirst=True, parse_dates=['Datum'])
        consumption_col = next((col for col in df.columns if 'Verbrauch [kWh]' in col), None)
        
        if not consumption_col:
            st.error("A consumption column containing 'Verbrauch [kWh]' was not found in the file.")
            return pd.DataFrame()

        df_meas = df[['Datum', 'Zeit von', consumption_col]].copy()
        df_meas.columns = ['date', 'time_str', 'consumption_kwh']
        
        # Combine date and time to create a local timestamp
        df_meas['timestamp_local'] = pd.to_datetime(df_meas['date'].dt.strftime('%Y-%m-%d') + ' ' + df_meas['time_str'])
        
        # Localize the timestamp and convert to UTC for merging with spot price data
        df_meas['timestamp'] = df_meas['timestamp_local'].dt.tz_localize(LOCAL_TIMEZONE, ambiguous='infer').dt.tz_convert('UTC')

        df_meas = df_meas.dropna(subset=['consumption_kwh'])
        
        # Aggregate 15-minute data to hourly sums
        df_hourly = df_meas.set_index('timestamp').resample('h')['consumption_kwh'].sum().reset_index()
        return df_hourly
    
    except Exception as e:
        st.error(f"Error processing CSV file: {e}. Please ensure it matches the required format.")
        return pd.DataFrame()

def get_base_load_threshold(df: pd.DataFrame) -> float:
    """Calculates the base load from average consumption during nighttime hours."""
    df_local = df.copy()
    df_local['timestamp'] = df_local['timestamp'].dt.tz_convert(LOCAL_TIMEZONE)
    night_hours = df_local[df_local['timestamp'].dt.hour.isin([2, 3, 4, 5])]
    return night_hours['consumption_kwh'].mean() if not night_hours.empty else 0

def classify_usage(df: pd.DataFrame, base_load_threshold: float) -> pd.DataFrame:
    """Classifies hourly consumption into Base, Peak, and Regular load."""
    if df.empty:
        return df
    df_c = df.copy()
    
    df_c['base_load_kwh'] = df_c['consumption_kwh'].clip(upper=base_load_threshold)
    
    df_c['consumption_diff'] = df_c['consumption_kwh'].diff().fillna(0)
    peak_threshold = df_c[df_c['consumption_diff'] > 0]['consumption_diff'].std() * 1.5
    df_c['is_peak_hour'] = (df_c['consumption_diff'] > peak_threshold) & (peak_threshold > 0)
    
    influenceable_load = df_c['consumption_kwh'] - df_c['base_load_kwh']
    df_c['peak_load_kwh'] = influenceable_load.where(df_c['is_peak_hour'], 0)
    df_c['regular_load_kwh'] = influenceable_load.where(~df_c['is_peak_hour'], 0)

    for col in ['base_load_kwh', 'peak_load_kwh', 'regular_load_kwh']:
        df_c[col] = df_c[col].clip(lower=0)
        
    return df_c.drop(columns=['consumption_diff', 'is_peak_hour'])

# --- Main Application Logic ---
# Initial setup and sidebar definition
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("1. Upload Your Consumption CSV", type=['csv'])

# Load data first to dynamically set defaults
if not uploaded_file:
    st.info("Welcome! Please upload your consumption data to begin.")
else:
    df_consumption = process_consumption_data(uploaded_file)
    
    if not df_consumption.empty:
        # Dynamically set date range for the sidebar based on uploaded file
        min_date_available = df_consumption['timestamp'].min().date()
        max_date_available = df_consumption['timestamp'].max().date()

        with st.sidebar:
            st.subheader("2. Select Analysis Period")
            col1, col2 = st.columns(2)
            start_date = col1.date_input("Start Date", min_date_available, min_value=min_date_available, max_value=max_date_available)
            end_date = col2.date_input("End Date", max_date_available, min_value=min_date_available, max_value=max_date_available)

            st.subheader("3. Configure Tariff Plans")
            # user provides on-top/kWh and monthly fee 
            with st.expander("Flexible (Spot Price) Plan", expanded=True):
                flex_on_top_price = st.number_input("On-Top Price (€/kWh)", value=0.0215, min_value=0.0, step=0.001, format="%.4f", help="Your provider's margin on top of the spot price.")
                flex_monthly_fee = st.number_input("Monthly Fee (€)", value=2.40, min_value=0.0, step=0.1, key="flex_fee")
            with st.expander("Static (Fixed Price) Plan"):
                static_kwh_price = st.number_input("Fixed Price (€/kWh)", value=0.14, min_value=0.0, step=0.01, key="static_price")
                static_monthly_fee = st.number_input("Monthly Fee (€)", value=2.99, min_value=0.0, step=0.1, key="static_fee")

        # Fetch spot price data for the selected range
        df_spot_prices = fetch_spot_data(start_date, end_date)

        if not df_spot_prices.empty and start_date < end_date:
            df_merged = pd.merge(df_consumption, df_spot_prices, on='timestamp', how='inner')
            df_merged = df_merged.dropna()

            if df_merged.empty:
                st.warning("No overlapping data found for the selected period.")
            else:
                # --- Analysis Execution ---
                base_load_threshold = get_base_load_threshold(df_merged)
                
                with st.sidebar:
                    st.info(f"Calculated Base Load: **{base_load_threshold:.3f} kWh/hour**")
                    df_merged['date_col'] = df_merged['timestamp'].dt.date
                    daily_consumption = df_merged.groupby('date_col')['consumption_kwh'].sum()
                    absence_threshold = base_load_threshold * 24 * 0.8
                    potential_absence_days = daily_consumption[daily_consumption < absence_threshold].index.tolist()

                    excluded_days = []
                    if potential_absence_days:
                        st.subheader("4. Absence Handling")
                        select_all = st.checkbox("Select/Deselect All Absence Days")
                        default_selection = potential_absence_days if select_all else []
                        excluded_days = st.multiselect(
                            "Exclude days with very low consumption?",
                            options=potential_absence_days,
                            default=default_selection,
                            format_func=lambda d: d.strftime('%Y-%m-%d')
                        )
                
                df_analysis = df_merged[~df_merged['date_col'].isin(excluded_days)].copy() if excluded_days else df_merged.copy()

                # Classify usage patterns and calculate costs on the filtered dataframe
                df_classified = classify_usage(df_analysis, base_load_threshold)
                df_classified['month'] = df_classified['timestamp'].dt.to_period('M')
                df_classified['days_in_month'] = df_classified['timestamp'].dt.days_in_month
                
                # Use the new "on-top" price for flexible cost calculation
                flex_total_price_kwh = df_classified['spot_price_eur_kwh'] + flex_on_top_price
                df_classified['total_cost_flexible'] = (df_classified['consumption_kwh'] * flex_total_price_kwh) + ((flex_monthly_fee / df_classified['days_in_month']) / 24)
                df_classified['total_cost_static'] = (df_classified['consumption_kwh'] * static_kwh_price) + ((static_monthly_fee / df_classified['days_in_month']) / 24)

                # --- Dashboard Display ---
                st.header("Recommendation & Summary")
                total_cost_flex = df_classified['total_cost_flexible'].sum()
                total_cost_static = df_classified['total_cost_static'].sum()
                savings = total_cost_static - total_cost_flex

                rec_col1, rec_col2 = st.columns(2)
                with rec_col1:
                    df_classified['price_quantile'] = df_classified.groupby('month')['spot_price_eur_kwh'].transform(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop'))
                    peak_consumption_cheap = df_classified[df_classified['price_quantile'] == 0]['peak_load_kwh'].sum()
                    peak_total = df_classified['peak_load_kwh'].sum()
                    peak_ratio = peak_consumption_cheap / peak_total if peak_total > 0 else 0
                    
                    if savings > 0 and peak_ratio > 0.3:
                        st.subheader("✅ Flexible Plan Recommended")
                        st.write(f"The flexible plan is better because you align **{peak_ratio:.0%}** of your peak usage with the cheapest 25% of market prices.")
                    else:
                        st.subheader("✅ Static Plan Recommended")
                        st.write("The recommendation is based on your overall consumption pattern versus market prices during the selected period.")
                
                with rec_col2:
                    if savings > 0:
                        st.metric(label="Potential Savings", value=f"€{savings:.2f}")
                    else:
                        st.metric(label="Potential Savings", value=f"€{-savings:.2f}")
                        
                    
                # --- Detailed Analysis Tabs ---
                tab1, tab2 = st.tabs(["Cost Comparison", "Usage Pattern Analysis"])

                with tab1:
                    st.subheader("Cost Breakdown by Period")
                    resolution = st.radio("Select Time Resolution", ('Daily', 'Weekly', 'Monthly'), horizontal=True, key="res")
                    freq_map = {'Daily': 'D', 'Weekly': 'W-MON', 'Monthly': 'M'}
                    grouper = pd.Grouper(key='timestamp', freq=freq_map[resolution])

                    # Table to compare flex vs spot prices
                    df_summary = df_classified.groupby(grouper).agg(
                        total_consumption_kwh=('consumption_kwh', 'sum'),
                        total_cost_flexible=('total_cost_flexible', 'sum'),
                        total_cost_static=('total_cost_static', 'sum')
                    ).reset_index()
                    
                    df_summary['Difference (€)'] = df_summary['total_cost_static'] - df_summary['total_cost_flexible']
                    df_summary['Period'] = df_summary['timestamp'].dt.strftime('%Y-%m-%d' if resolution != 'Monthly' else '%Y-%m')
                    
                    st.bar_chart(df_summary.set_index('Period'), y=['total_cost_flexible', 'total_cost_static'], y_label="Total Cost (€)")
                    
                    st.subheader("Cost Summary Table")
                    st.dataframe(df_summary[['Period', 'total_consumption_kwh', 'total_cost_flexible', 'total_cost_static', 'Difference (€)']].style.format({
                        'total_consumption_kwh': '{:.2f} kWh',
                        'total_cost_flexible': '€{:.2f}',
                        'total_cost_static': '€{:.2f}',
                        'Difference (€)': '€{:.2f}'
                    }).bar(subset=['Difference (€)'], align='zero', color=['#d65f5f', '#5fba7d']))

                with tab2:
                    st.subheader("Analyze Your Consumption Profile")
                    
                    # Add a filter for day type
                    df_classified['day_type'] = df_classified['timestamp'].dt.tz_convert(LOCAL_TIMEZONE).dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
                    day_filter = st.radio("Filter data by:", ("All Days", "Weekdays", "Weekends"), horizontal=True, key="day_filter")
                    
                    if day_filter == "Weekdays":
                        df_pattern = df_classified[df_classified['day_type'] == 'Weekday']
                    
                    elif day_filter == "Weekends":
                        df_pattern = df_classified[df_classified['day_type'] == 'Weekend']
                    
                    else:
                        df_pattern = df_classified
                    
                    if df_pattern.empty:
                        st.warning(f"No data available for {day_filter.lower()}.")
                    else:
                        # Display thresholds
                        st.metric(label="Base Load Threshold", value=f"{base_load_threshold:.3f} kWh")
                        peak_display_threshold = df_pattern[df_pattern['consumption_diff'] > 0]['consumption_diff'].std() * 1.5
                        st.metric(label="Peak Detection Threshold (Sharp Increase)", value=f"{peak_display_threshold:.3f} kWh")

                        pat_col1, pat_col2 = st.columns(2)
                        with pat_col1:
                            st.subheader("Consumption Proportions")
                            load_sums = df_pattern[['base_load_kwh', 'regular_load_kwh', 'peak_load_kwh']].sum()
                            load_sums.index = ['Base Load', 'Regular Load', 'Peak Load']
                            st.bar_chart(load_sums, y_label="Total Consumption (kWh)")

                        with pat_col2:
                            st.subheader("Average Spot Price per Profile")
                            # Calculate weighted average price for each load type
                            avg_prices = {}
                            for load_type in ['base_load_kwh', 'regular_load_kwh', 'peak_load_kwh']:
                                total_kwh = df_pattern[load_type].sum()
                                if total_kwh > 0:
                                    weighted_price_sum = (df_pattern[load_type] * df_pattern['spot_price_eur_kwh']).sum()
                                    avg_prices[load_type.replace('_kwh', '').replace('_', ' ').title()] = weighted_price_sum / total_kwh
                                else:
                                    avg_prices[load_type.replace('_kwh', '').replace('_', ' ').title()] = 0
                            
                            df_avg_prices = pd.DataFrame.from_dict(avg_prices, orient='index', columns=['Average Price (€/kWh)'])
                            st.bar_chart(df_avg_prices, y_label="Average Price (€/kWh)")