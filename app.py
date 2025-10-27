import streamlit as st
import methods.config as config
import methods.data_loader as data_loader
import methods.analysis as analysis
import methods.ui_components as ui_components
from methods.tariffs import TariffManager
from methods.logger import logger
from methods.utils import filter_dataframe

# --- Page and App Configuration ---
st.set_page_config(layout="wide")

# --- Main Application ---
def main():
    
    # Instantiate managers once
    tariff_manager = TariffManager("resources/tariffs_flexible.json", "resources/tariffs_static.json")

    # --- File Upload and Initial Data Processing ---
    uploaded_file = ui_components.render_upload_file()
    ui_components.render_language_selection()

    if not uploaded_file:
        ui_components.render_intro()
        ui_components.render_footer()
        return

    df_consumption = data_loader.process_consumption_data(uploaded_file)
    if df_consumption.empty:
        return

    # --- Sidebar and Input Controls ---
    mode, country, start_date, end_date, shift_percentage = ui_components.render_sidebar_inputs(df_consumption)
    
    # --- Data Loading and Merging ---
    df_consumption = filter_dataframe(df_consumption, start_date, end_date)
    min_date, max_date = ui_components.get_min_max_date(df_consumption, config.TODAY_IS_MAX_DATE)
    df_spot_prices = data_loader.get_spot_data(country, min_date, max_date)
    df_merged = data_loader.merge_consumption_with_prices(df_consumption, df_spot_prices)
    
    if df_merged.empty:
        st.warning("No overlapping data found for the selected period. Please check your file's date range.")
        return
    
    # --- Tariff Selection (Main Page) ---
    flex_tariff, static_tariff = ui_components.render_tariff_selection_header(df_merged, tariff_manager, country)

    # --- Core Analysis Pipeline ---
    df_classified, base_threshold, peak_threshold = analysis.classify_usage(df_merged, config.LOCAL_TIMEZONE)
    df_with_shifting = analysis.simulate_peak_shifting(df_classified, shift_percentage)
    df_analysis = tariff_manager.run_cost_analysis(df_with_shifting, flex_tariff, static_tariff)
    
    # --- Final Filtering and Rendering ---
    df_analysis = ui_components.render_absence_days(df_analysis, base_threshold)
    
    ui_components.render_recommendation(df_analysis, flex_tariff, static_tariff)

    # --- Tab Definitions based on Mode ---
    if mode == "Expert":
        tab_options = [
            "Spot Price Analysis", 
            "Cost Comparison", 
            "Usage Patterns", 
            "Yearly Summary", 
            "Download Data",
            "FAQ & Help"
        ]
    else: # Basic Mode
        tab_options = [
            "Dashboard",
            "Yearly Summary", 
            "Download Data",
            "FAQ & Help"
        ]
    
    # Create control panel instead of tabs for better control and performance gains.
    st.markdown("---")
    st.segmented_control(
        "Select a view:",
        options=tab_options,
        default=tab_options[0],
        key="active_tab",  # This key is crucial for statefulness!
        label_visibility="collapsed" # Completely hides the label as it's not relevant.
    )
        
    # If/elif blocks render only the selected view's content and keep the user on the same "tab" after an interaction.
    active_tab = st.session_state.get("active_tab", tab_options[0])

    if active_tab == "Dashboard": # Basic Mode Main Tab
        ui_components.render_basic_dashboard_tab(df_analysis, static_tariff, base_threshold, peak_threshold)

    elif active_tab == "Spot Price Analysis":
        ui_components.render_price_analysis_tab(df_analysis.copy(), static_tariff)
    
    elif active_tab == "Cost Comparison":
        ui_components.render_cost_comparison_tab(df_analysis)
    
    elif active_tab == "Usage Patterns":
        ui_components.render_usage_pattern_tab(df_analysis, base_threshold, peak_threshold)
    
    elif active_tab == "Yearly Summary":
        ui_components.render_yearly_summary_tab(df_analysis)
    
    elif active_tab == "Download Data":
        ui_components.render_download_tab(df_analysis, start_date, end_date)

    elif active_tab == "FAQ & Help":
        ui_components.render_faq_tab()

    ui_components.render_footer()
    
if __name__ == "__main__":
    main()