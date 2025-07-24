import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date

# --- Page and App Configuration ---
st.set_page_config(layout="wide")
st.title("💡 Electricity Tariff Comparison Dashboard")
st.write("Analyze your electricity consumption and compare a flexible (spot market) plan with a static (fixed-rate) plan.")

# --- Constants and Configuration ---
AWATTAR_SERVICE_FEE = 0.0179  # €/kWh
AWATTAR_ORIGIN_FEE = 0.0036   # €/kWh
LOCAL_TIMEZONE = "Europe/Vienna" # Timezone for consumption data

# --- Data Fetching and Processing Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_awattar_data(start: date, end: date) -> pd.DataFrame:
    """Fetches electricity market data from the aWATTar API for a given date range."""
    base_url = "https://api.awattar.de/v1/marketdata"
    
    # Convert dates to UTC datetime objects for the API query
    # Add one day to the end date because the API's 'end' parameter is exclusive
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=datetime.now().astimezone().tzinfo).astimezone(tz=None)
    end_dt = datetime.combine(end + pd.Timedelta(days=1), datetime.min.time(), tzinfo=datetime.now().astimezone().tzinfo).astimezone(tz=None)
    
    # Get timestamps in milliseconds
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    params = {'start': start_ts, 'end': end_ts}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()['data']
        df = pd.DataFrame(data)
        
        # Create a timezone-aware timestamp (UTC) from the data.
        df['timestamp'] = pd.to_datetime(df['start_timestamp'], unit='ms', utc=True)
        
        # Convert price from EUR/MWh to EUR/kWh
        df['spot_price_eur_kwh'] = df['marketprice'] / 1000
        return df[['timestamp', 'spot_price_eur_kwh']]
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch aWATTar data: {e}")
        
    except (KeyError, IndexError, TypeError):
        st.error("Received unexpected or empty data from aWATTar API. The API might be down or have no data for the selected range.")
    
    return pd.DataFrame()


def process_consumption_data(uploaded_file) -> pd.DataFrame:
    """Loads and processes the user-uploaded consumption CSV, converting it to hourly UTC."""
    
    if uploaded_file is None:
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8')
        
        # Wiener Netze writes the Zählernummer as a header but included Verbrauch too
        consumption_col = next((col for col in df.columns if 'Verbrauch [kWh]' in col), None)
        
        if not consumption_col:
            st.error("Consumption column 'Verbrauch [kWh]' not found in the uploaded file.")
            return pd.DataFrame()

        df_meas = df[['Datum', 'Zeit von', consumption_col]].copy()
        df_meas.columns = ['date_str', 'time_str', 'consumption_kwh']
        
        # Convert the data from date and time into a timestamp
        df_meas['timestamp_local'] = pd.to_datetime(df_meas['date_str'] + ' ' + df_meas['time_str'], format='%d.%m.%Y %H:%M:%S')
        
        # Localize timestamp to the user's timezone, then convert to UTC.
        # This is critical for correctly merging with UTC-based spot price data.
        df_meas['timestamp'] = df_meas['timestamp_local'].dt.tz_localize(LOCAL_TIMEZONE, ambiguous='infer').dt.tz_convert('UTC')

        df_meas = df_meas.dropna(subset=['consumption_kwh'])
        df_hourly = df_meas.set_index('timestamp').resample('h')['consumption_kwh'].sum().reset_index()
        
        return df_hourly
    
    except Exception as e:
        st.error(f"Error processing CSV file: {e}. Please ensure it matches the required format (dd.mm.yy, semicolon-separated).")
        return pd.DataFrame()

def get_base_load_threshold(df: pd.DataFrame) -> float:
    """Calculates the base load threshold based on average consumption during nighttime hours (e.g. fridge, wifi, stand-by power)."""
    df_local = df.copy()
    df_local['timestamp'] = df_local['timestamp'].dt.tz_convert(LOCAL_TIMEZONE)
    night_hours = df_local[df_local['timestamp'].dt.hour.isin([2, 3, 4, 5])]
    return night_hours['consumption_kwh'].mean() if not night_hours.empty else 0


def classify_usage(df: pd.DataFrame, base_load_threshold: float) -> pd.DataFrame:
    """Classifies hourly consumption into Base, Peak, and Regular load."""
    if df.empty:
        return df

    df_c = df.copy()
    
    # 1. Base Load: Consumption below the calculated threshold
    df_c['base_load_kwh'] = df_c['consumption_kwh'].clip(upper=base_load_threshold)
    
    # 2. Peak Load: A sharp, significant increase from the previous hour
    df_c['consumption_diff'] = df_c['consumption_kwh'].diff().fillna(0)
    peak_threshold = df_c[df_c['consumption_diff'] > 0]['consumption_diff'].std() * 1.5
    df_c['is_peak_hour'] = (df_c['consumption_diff'] > peak_threshold) & (peak_threshold > 0)
    
    # 3. Regular Load: The remaining influenceable consumption
    influenceable_load = df_c['consumption_kwh'] - df_c['base_load_kwh']
    df_c['peak_load_kwh'] = influenceable_load.where(df_c['is_peak_hour'], 0)
    df_c['regular_load_kwh'] = influenceable_load.where(~df_c['is_peak_hour'], 0)

    # Ensure no negative values from floating point inaccuracies
    for col in ['base_load_kwh', 'peak_load_kwh', 'regular_load_kwh']:
        df_c[col] = df_c[col].clip(lower=0)
        
    return df_c.drop(columns=['consumption_diff', 'is_peak_hour'])


# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("1. Upload Your Consumption CSV", type=['csv'])
    st.info("CSV must be semi-colon separated with columns 'Datum' (dd.mm.yy), 'Zeit von', and 'Verbrauch [kWh]'.")

    st.subheader("2. Select Analysis Period")
    # Get the min/max dates from the uploaded file if available
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", df["timestam"].min().date())
    end_date = col2.date_input("End Date",  df["timestam"].max().date())

    st.subheader("3. Configure Tariff Plans")
    with st.expander("Flexible Plan Details"):
        flex_monthly_fee = st.number_input("Monthly Fee (€)", value=5.99, min_value=0.0, step=0.1, key="flex_fee")
    with st.expander("Static Plan Details"):
        static_kwh_price = st.number_input("Fixed Price (€/kWh)", value=0.14, min_value=0.0, step=0.01, key="static_price")
        static_monthly_fee = st.number_input("Monthly Fee (€)", value=5.99, min_value=0.0, step=0.1, key="static_fee")

# --- Main Application Logic ---
if not uploaded_file:
    st.info("Welcome! Please upload your consumption data and set your parameters in the sidebar to begin.")
elif start_date >= end_date:
    st.error("Error: The start date must be before the end date.")
else:
    df_consumption = process_consumption_data(uploaded_file)
    df_awattar = fetch_awattar_data(start_date, end_date)

    if not df_consumption.empty and not df_awattar.empty:
        df_merged = pd.merge(df_consumption, df_awattar, on='timestamp', how='inner')
        df_merged = df_merged.dropna()

        if df_merged.empty:
            st.warning("No overlapping data found for the selected period. Please check your dates and the contents of your CSV file.")
        else:
            # --- Analysis Execution ---

            # 1. Absence Detection
            base_load_threshold = get_base_load_threshold(df_merged)
            st.sidebar.info(f"Calculated Base Load: {base_load_threshold:.3f} kWh/hour")
            
            df_merged['date'] = df_merged['timestamp'].dt.date
            daily_consumption = df_merged.groupby('date')['consumption_kwh'].sum()
            absence_threshold = base_load_threshold * 24 * 0.8 # Heuristic: Day's consumption is <80% of constant base load
            potential_absence_days = daily_consumption[daily_consumption < absence_threshold].index.tolist()

            excluded_days = []
            if potential_absence_days:
                with st.sidebar:
                    st.subheader("Absence Handling")
                    excluded_days = st.multiselect(
                        "We detected days with very low consumption (possible absence). Exclude them from the analysis?",
                        options=potential_absence_days,
                        default=[]
                    )
            
            if excluded_days:
                df_analysis = df_merged[~df_merged['date'].isin(excluded_days)].copy()
                st.success(f"Recalculating with {len(excluded_days)} day(s) excluded.")
            else:
                df_analysis = df_merged.copy()

            # 2. Classify Usage Patterns on the filtered data
            df_classified = classify_usage(df_analysis, base_load_threshold)

            # 3. Cost Calculation
            df_classified['month'] = df_classified['timestamp'].dt.to_period('M')
            df_classified['days_in_month'] = df_classified['timestamp'].dt.daysinmonth
            
            flex_price_kwh = df_classified['spot_price_eur_kwh'] + AWATTAR_SERVICE_FEE + AWATTAR_ORIGIN_FEE
            df_classified['cost_flexible_hourly'] = df_classified['consumption_kwh'] * flex_price_kwh
            df_classified['cost_static_hourly'] = df_classified['consumption_kwh'] * static_kwh_price

            df_classified['fee_flex_hourly'] = (flex_monthly_fee / df_classified['days_in_month']) / 24
            df_classified['fee_static_hourly'] = (static_monthly_fee / df_classified['days_in_month']) / 24

            df_classified['total_cost_flexible'] = df_classified['cost_flexible_hourly'] + df_classified['fee_flex_hourly']
            df_classified['total_cost_static'] = df_classified['cost_static_hourly'] + df_classified['fee_static_hourly']

            # --- Dashboard Display ---
            st.header("🏁 Recommendation & Summary")
            total_cost_flex = df_classified['total_cost_flexible'].sum()
            total_cost_static = df_classified['total_cost_static'].sum()
            savings = total_cost_static - total_cost_flex

            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                if savings > 0:
                    st.subheader("✅ The Flexible Plan is recommended")
                    st.metric(label="Potential Savings over Static Plan", value=f"€{savings:.2f}")
                else:
                    st.subheader("✅ The Static Plan is recommended")
                    st.metric(label="Potential Savings over Flexible Plan", value=f"€{-savings:.2f}")
            
            with rec_col2:
                # Automated Justification
                df_classified['price_quantile'] = df_classified.groupby('month')['spot_price_eur_kwh'].transform(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop'))
                df_classified['is_cheap_hour'] = df_classified['price_quantile'] == 0
                
                peak_consumption = df_classified['peak_load_kwh'].sum()
                peak_in_cheap_hours = df_classified[df_classified['is_cheap_hour']]['peak_load_kwh'].sum()
                
                justification = "The recommendation is based on your overall consumption pattern compared to market prices."
                if peak_consumption > 0:
                    peak_in_cheap_ratio = peak_in_cheap_hours / peak_consumption
                    if savings > 0 and peak_in_cheap_ratio > 0.3:
                        justification = f"The flexible plan is better because you managed to align **{peak_in_cheap_ratio:.0%}** of your peak usage with cheap price hours."
                    elif savings <= 0 and peak_in_cheap_ratio < 0.2:
                        justification = f"The static plan is safer because your peak usage often occurred outside of cheap price hours. A fixed rate protects you from this volatility."

                st.markdown("**Justification**")
                st.write(justification)

            # --- Detailed Analysis Tabs ---
            tab1, tab2, tab3 = st.tabs(["Cost Comparison", "Weekday vs. Weekend", "Usage Pattern"])

            with tab1:
                resolution = st.radio("Select Time Resolution", ('Daily', 'Weekly', 'Monthly'), horizontal=True, key="res")
                
                freq_map = {'Daily': 'D', 'Weekly': 'W-MON', 'Monthly': 'M'}
                grouper = pd.Grouper(key='timestamp', freq=freq_map[resolution])

                df_summary = df_classified.groupby(grouper).agg(
                    total_consumption_kwh=('consumption_kwh', 'sum'),
                    total_cost_flexible=('total_cost_flexible', 'sum'),
                    total_cost_static=('total_cost_static', 'sum')
                ).reset_index()

                df_summary['Period'] = df_summary['timestamp'].dt.strftime('%Y-%m-%d' if resolution != 'Monthly' else '%Y-%m')
                
                st.line_chart(df_summary.set_index('Period'), y=['total_cost_flexible', 'total_cost_static'])

                st.dataframe(df_summary[['Period', 'total_consumption_kwh', 'total_cost_flexible', 'total_cost_static']].style.format({
                    'total_consumption_kwh': '{:.2f} kWh',
                    'total_cost_flexible': '€{:.2f}',
                    'total_cost_static': '€{:.2f}'
                }))

            with tab2:
                df_classified['day_type'] = df_classified['timestamp'].dt.tz_convert(LOCAL_TIMEZONE).dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
                df_weekday_summary = df_classified.groupby('day_type').agg(
                    avg_hourly_consumption_kwh=('consumption_kwh', 'mean'),
                    avg_hourly_cost_flexible=('total_cost_flexible', 'mean')
                ).reset_index()
                
                st.bar_chart(df_weekday_summary.set_index('day_type'), y='avg_hourly_consumption_kwh', y_label="Avg. Hourly Consumption (kWh)")

            with tab3:
                df_load_summary = df_classified.groupby(grouper)[['base_load_kwh', 'regular_load_kwh', 'peak_load_kwh']].sum().reset_index()
                df_load_summary['Period'] = df_load_summary['timestamp'].dt.strftime('%Y-%m-%d' if resolution != 'Monthly' else '%Y-%m')
                
                st.write(f"Consumption Profile by {resolution}")
                st.line_chart(df_load_summary.set_index('Period'), y=['base_load_kwh', 'regular_load_kwh', 'peak_load_kwh'])