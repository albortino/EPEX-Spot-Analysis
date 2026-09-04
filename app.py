import streamlit as st
import methods.config as config
import methods.data_loader as data_loader
import methods.analysis as analysis
import methods.ui_components as ui_components
from methods.tariffs import TariffManager
from methods.logger import logger
from methods.utils import filter_dataframe, filter_by_quarter
from methods.validation import inspect_consumption, inspect_price_coverage


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
    mode, country, start_date, end_date, selected_quarter, shift_percentage = ui_components.render_sidebar_inputs(df_consumption)
    
    # --- Data Loading and Merging ---
    df_consumption = filter_dataframe(df_consumption, start_date, end_date)
    
    # Apply quarter filtering
    df_consumption = filter_by_quarter(df_consumption, selected_quarter)
    quality = inspect_consumption(df_consumption)
    if not quality.usable:
        ui_components.render_data_quality(quality)
        st.error("Analysis is unavailable until the data-quality issues above are fixed.")
        return
    
    min_date, max_date = ui_components.get_min_max_date(df_consumption, config.TODAY_IS_MAX_DATE)
    df_spot_prices = data_loader.get_spot_data(country, min_date, max_date)
    df_merged = data_loader.merge_consumption_with_prices(df_consumption, df_spot_prices)
    coverage = inspect_price_coverage(df_merged)
    ui_components.render_data_quality(quality, coverage)
    if df_merged.empty or coverage < 0.99:
        st.warning("No overlapping data found for the selected period. Please check your file's date range.")
        return

    
    # --- Perform initial analysis and render sidebar components that modify the dataframe ---
    # This should be done once, before the tab loop.
    df_classified, base_threshold, peak_threshold = analysis.classify_usage(df_merged, config.LOCAL_TIMEZONE)
    df_analysis_base = ui_components.render_absence_days(df_classified, base_threshold, config.ABSENCE_THRESHOLD)
    
    # --- Perform main analysis pipeline once, outside the tab loop ---
    # This is a major efficiency gain, as these expensive operations are not re-run for each tab.
    flex_tariff, static_tariff = ui_components.render_tariff_selection_header(df_merged, tariff_manager, country)
    if not flex_tariff or not static_tariff:
        st.error("Tariff comparison is unavailable because no valid tariff is configured.")
        return
    df_with_shifting = analysis.simulate_peak_shifting(df_analysis_base, shift_percentage)
    df_analysis = tariff_manager.run_cost_analysis(df_with_shifting, flex_tariff, static_tariff)
    df_baseline = tariff_manager.run_cost_analysis(df_analysis_base.copy(), flex_tariff, static_tariff)
    ui_components.render_recommendation(df_analysis, flex_tariff, static_tariff)
    ui_components.render_energy_cost_notice()


     # --- Tab Definitions based on Mode ---
    tab_options = ["🏠 Your result", "⚡ Improve timing", "⬇️ Download", "❓ FAQ"]
    if mode == "Expert":
        tab_options[1:1] = ["💰 Compare tariffs", "📊 Explore data"]
    
    tabs = st.tabs(tab_options)
    

    for i, tab_name in enumerate(tab_options):
        with tabs[i]:
            # Icon-stripped name for matching
            clean_tab_name = " ".join(tab_name.split(" ")[1:])
            
            # Render tabs using the pre-computed df_analysis
            if clean_tab_name == "Your result":
                ui_components.render_basic_dashboard_tab(df_analysis, static_tariff, base_threshold, peak_threshold)
            elif clean_tab_name == "Compare tariffs":
                ui_components.render_cost_comparison_tab(df_analysis)
            elif clean_tab_name == "Improve timing":
                st.subheader("Potential savings under this scenario")
                st.caption("Peak energy is moved only to a cheaper interval before or after the original event within the selected window.")
                shifting_savings = df_baseline["total_cost_flexible"].sum() - df_analysis["total_cost_flexible"].sum()
                shifted_kwh = (df_analysis["regular_load_kwh"] - df_baseline["regular_load_kwh"]).clip(lower=0).sum()
                col1, col2 = st.columns(2)
                col1.metric("Estimated flexible-cost saving", f"€{shifting_savings:.2f}")
                col2.metric("Peak energy shifted", f"{shifted_kwh:.2f} kWh")
            elif clean_tab_name == "Explore data":
                ui_components.render_price_analysis_tab(df_analysis, static_tariff)
                ui_components.render_usage_pattern_tab(df_analysis, base_threshold, peak_threshold)
            elif clean_tab_name == "Download":
                ui_components.render_download_tab(df_analysis_base, flex_tariff, start_date, end_date) # Use base analysis data
            elif clean_tab_name == "FAQ":
                ui_components.render_faq_tab()

    ui_components.render_footer()
    
if __name__ == "__main__":
    main()
