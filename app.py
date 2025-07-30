import streamlit as st
from datetime import datetime
import config
import data_loader
import analysis
import ui_components
from tariffs import TariffManager
from config import AWATTAR_COUNTRY

# --- Page and App Configuration ---
st.set_page_config(layout="wide")
st.title("Electricity Tariff Comparison Dashboard")
st.markdown("Analyze your consumption, compare flexible vs. static tariffs, and understand your usage patterns.")

# --- Main Application ---
def main(country: str):
    
    # Instantiate managers once
    tariff_manager = TariffManager("flex_tariffs.json", "static_tariffs.json")

    # --- File Upload and Initial Data Processing ---
    uploaded_file = st.sidebar.file_uploader("Upload Your Consumption CSV", type=["csv"], help="Please see https://awattar-backtesting.github.io/ for more tipps and tricks how to get the consumption data from your network provider.")
    if not uploaded_file:
        st.info("👋 Welcome! Please upload your consumption data to begin.")
        st.markdown("This project was influenced by https://awattar-backtesting.github.io/, which provides a simple and effective overview.\n\n"
                   "This tool provides further insights into your consumption behavior and help you to choose the most economic tariff plan.")
        return

    df_consumption = data_loader.process_consumption_data(uploaded_file)
    if df_consumption.empty:
        return

    # --- Sidebar and Input Controls ---
    country, start_date, end_date, flex_tariff, static_tariff, shift_percentage = ui_components.render_sidebar_inputs(df_consumption, tariff_manager, country)
    
    # --- Data Loading and Merging ---
    min_date, max_date = ui_components.get_min_max_date(df_consumption)
    df_spot_prices = data_loader.get_spot_data(country, min_date, max_date)
    df_merged = data_loader.merge_consumption_with_prices(df_consumption, df_spot_prices)

    if df_merged.empty:
        st.warning("No overlapping data found for the selected period. Please check your file's date range.")
        return
    
    # --- Core Analysis Pipeline ---
    print(f"{datetime.now().strftime(config.DATE_FORMAT)}: Performing Full Analysis")
    df_classified, base_threshold, peak_threshold = analysis.classify_usage(df_merged, config.LOCAL_TIMEZONE)
    df_with_shifting = analysis.simulate_peak_shifting(df_classified, shift_percentage)
    df_analysis = tariff_manager.run_cost_analysis(df_with_shifting, flex_tariff, static_tariff)

    # --- Final Filtering and Rendering ---
    df_analysis = ui_components.render_absence_days(df_analysis, base_threshold)
    
    ui_components.render_recommendation(df_analysis)

    # Define the "tabs"
    tab_options = [
        "Spot Price Analysis", 
        "Cost Comparison", 
        "Usage Pattern Analysis", 
        "Yearly Summary", 
        "Download Data"
    ]
    
    # # Render the analysis tabs
    # tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_options)
    # with tab1:
    #     ui_components.render_price_analysis_tab(df_analysis, static_tariff)
    # with tab2:
    #     ui_components.render_cost_comparison_tab(df_analysis)
    # with tab3:
    #     ui_components.render_usage_pattern_tab(df_analysis, base_threshold, peak_threshold)
    # with tab4:
    #     ui_components.render_yearly_summary_tab(df_analysis)
    # with tab5:
    #     ui_components.render_download_tab(df_analysis, start_date, end_date)

   

    # Create control panel instead of tabs for better control and performance gains.
    st.markdown("---")

    st.segmented_control(
        "Select a view:",
        options=tab_options,
        default=tab_options[0],
        selection_mode="single",
        format_func=lambda x: f"**{x}**",
        key="active_tab",  # This key is crucial for statefulness
        width="stretch",
        label_visibility="collapsed"
    )
        
    # If/elif blocks render only the selected view's content and keep the user on the same "tab" after an interaction.
    if st.session_state.active_tab == "Spot Price Analysis":
        ui_components.render_price_analysis_tab(df_analysis, static_tariff)
    
    elif st.session_state.active_tab == "Cost Comparison":
        ui_components.render_cost_comparison_tab(df_analysis)
    
    elif st.session_state.active_tab == "Usage Pattern Analysis":
        ui_components.render_usage_pattern_tab(df_analysis, base_threshold, peak_threshold)
    
    elif st.session_state.active_tab == "Yearly Summary":
        ui_components.render_yearly_summary_tab(df_analysis)
    
    elif st.session_state.active_tab == "Download Data":
        ui_components.render_download_tab(df_analysis, start_date, end_date)

    ui_components.render_footer()
if __name__ == "__main__":
    main(AWATTAR_COUNTRY)