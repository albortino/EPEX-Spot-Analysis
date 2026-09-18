import streamlit as st
import src.config as config
import src.data_loader as data_loader
import src.analysis as analysis
import src.ui_components as ui
from src.tariffs import TariffManager
from src.utils import filter_dataframe, filter_by_quarter, inspect_consumption, inspect_price_coverage
from src.i18n import t


# --- Page and App Configuration ---
st.set_page_config(layout="wide")

# --- Main Application ---
def main():
   
    # Instantiate managers once
    tariff_manager = TariffManager("resources/tariffs_spot.json", "resources/tariffs_fixed.json")

    # --- File Upload and Initial Data Processing ---
    uploaded_file = ui.render_upload_file()
    ui.render_language_selection()

    if not uploaded_file:
        ui.render_intro()
        ui.render_footer()
        return

    df_consumption = data_loader.process_consumption_data(uploaded_file)
    if df_consumption.empty:
        return
 
    # --- Sidebar and Input Controls ---
    mode, country, start_date, end_date, selected_quarter, shift_percentage = ui.render_sidebar_inputs(df_consumption)

    # --- Data Loading and Merging ---
    df_consumption = filter_dataframe(df_consumption, start_date, end_date)

    # Apply quarter filtering
    df_consumption = filter_by_quarter(df_consumption, selected_quarter)
    quality = inspect_consumption(df_consumption)
    if not quality.usable:
        ui.render_data_quality(quality)
        st.error("Analysis is unavailable until the data-quality issues above are fixed.")
        return

    min_date, max_date = ui.get_min_max_date(df_consumption, config.TODAY_IS_MAX_DATE)
    df_spot_prices = data_loader.get_spot_data(country, min_date, max_date)
    df_merged = data_loader.merge_consumption_with_prices(df_consumption, df_spot_prices)
    coverage = inspect_price_coverage(df_merged)
    ui.render_data_quality(quality, coverage)
    if df_merged.empty or coverage < 0.99:
        st.warning("No overlapping data found for the selected period. Please check your file's date range.")
        return


    # --- Perform initial analysis once and render sidebar components that modify the dataframe ---
    df_classified, base_threshold, peak_threshold = analysis.classify_usage(df_merged, config.LOCAL_TIMEZONE)
    df_analysis_base = ui.render_absence_days(df_classified, base_threshold, config.ABSENCE_THRESHOLD)
  
    # --- Perform main analysis pipeline once, outside the tab loop ---
    flex_tariff, variable_tariff, static_tariff = ui.render_tariff_selection_header(df_analysis_base, tariff_manager, country)
    df_with_shifting = analysis.simulate_peak_shifting(df_analysis_base, shift_percentage)
    df_analysis = tariff_manager.run_cost_analysis(df_with_shifting, flex_tariff, variable_tariff, static_tariff)
    ui.render_recommendation(df_analysis, flex_tariff, static_tariff, variable_tariff)

    # --- Tab Definitions based on Mode ---
    if mode == "Expert":
        tab_options = [
            t("tab_spot_price_analysis"), 
            t("tab_cost_comparison"), 
            t("tab_usage_patterns"),
            t("tab_download"),
            t("tab_faq"),
            t("tab_about")
        ]
    else: # Basic Mode
        tab_options = [
            t("tab_dashboard"),
            t("tab_download"),
            t("tab_faq"),
            t("tab_about")
        ]
    
    tabs = st.tabs(tab_options)  

    for i, tab_name in enumerate(tab_options):
        with tabs[i]:
            if tab_name == t("tab_dashboard"):
                ui.render_basic_dashboard_tab(df_analysis, static_tariff, base_threshold, peak_threshold)
            elif tab_name == t("tab_spot_price_analysis"):
                ui.render_price_analysis_tab(df_analysis, static_tariff)
            elif tab_name == t("tab_cost_comparison"):
                ui.render_cost_comparison_tab(df_analysis)
            elif tab_name == t("tab_usage_patterns"):
                ui.render_usage_pattern_tab(df_analysis, base_threshold, peak_threshold)
            elif tab_name == t("tab_download"):
                 # Use base analysis data
                ui.render_download_tab(df_analysis_base, flex_tariff, start_date, end_date)
            elif tab_name == t("tab_faq"):
                ui.render_faq_tab()
            elif tab_name == t("tab_about"):
                ui.render_about_tab()

    ui.render_footer()


if __name__ == "__main__":
    main()
