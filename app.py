# app.py

import streamlit as st
from datetime import datetime
import config
import data_loader
import analysis
import ui_components
from tariffs import TariffManager

# --- Page and App Configuration ---
st.set_page_config(layout="wide")
st.title("Electricity Tariff Comparison Dashboard")
st.markdown("Analyze your consumption, compare flexible vs. static tariffs, and understand your usage patterns.")

# --- Main Application ---
def main():
    # Instantiate managers once
    tariff_manager = TariffManager("flex_tariffs.json", "static_tariffs.json")

    # --- File Upload and Initial Data Processing ---
    uploaded_file = st.sidebar.file_uploader("Upload Your Consumption CSV", type=["csv"])
    if not uploaded_file:
        st.info("👋 Welcome! Please upload your consumption data to begin.")
        return

    df_consumption = data_loader.process_consumption_data(uploaded_file)
    if df_consumption.empty:
        return

    # --- Data Loading and Merging ---
    min_date, max_date = ui_components.get_min_max_date(df_consumption)
    df_spot_prices = data_loader.get_spot_data(min_date, max_date)
    df_merged = data_loader.merge_consumption_with_prices(df_consumption, df_spot_prices)

    if df_merged.empty:
        st.warning("No overlapping data found for the selected period. Please check your file's date range.")
        return

    # --- User Inputs and Configuration ---
    start_date, end_date, flex_tariff, static_tariff, shift_percentage = ui_components.get_sidebar_inputs(df_merged, tariff_manager)

    # --- Core Analysis Pipeline ---
    print(f"{datetime.now().strftime(config.DATE_FORMAT)}: Performing Full Analysis")
    df_classified, base_threshold, peak_threshold = analysis.classify_usage(df_merged, config.LOCAL_TIMEZONE)
    df_with_shifting = analysis.simulate_peak_shifting(df_classified, shift_percentage)
    df_analysis = tariff_manager.run_cost_analysis(df_with_shifting, flex_tariff, static_tariff)

    # --- Final Filtering and Rendering ---
    df_analysis = ui_components.get_absence_days(df_analysis, base_threshold)
    
    ui_components.render_recommendation(df_analysis)

    # Render the analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "**Spot Price Analysis**", 
        "**Cost Comparison**", 
        "**Usage Pattern Analysis**", 
        "**Yearly Summary**", 
        "**Download Data**"
    ])
    with tab1:
        ui_components.render_price_analysis_tab(df_analysis)
    with tab2:
        ui_components.render_cost_comparison_tab(df_analysis)
    with tab3:
        ui_components.render_usage_pattern_tab(df_analysis, base_threshold, peak_threshold)
    with tab4:
        ui_components.render_yearly_summary_tab(df_analysis)
    with tab5:
        ui_components.render_download_tab(df_analysis, start_date, end_date)

if __name__ == "__main__":
    main()