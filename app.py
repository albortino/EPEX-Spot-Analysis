import streamlit as st
import methods.config as config
import methods.data_loader as data_loader
import methods.analysis as analysis
import methods.ui_components as ui_components
from methods.tariffs import TariffManager
from methods.logger import logger
from methods.utils import filter_dataframe, filter_by_quarter


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
    
    min_date, max_date = ui_components.get_min_max_date(df_consumption, config.TODAY_IS_MAX_DATE)
    df_spot_prices = data_loader.get_spot_data(country, min_date, max_date)
    df_merged = data_loader.merge_consumption_with_prices(df_consumption, df_spot_prices)
    
    if df_merged.empty:
        st.warning("No overlapping data found for the selected period. Please check your file's date range.")
        return
    
        # --- Tab Definitions based on Mode ---
    if mode == "Expert":
        tab_options = [
            "📊 Spot Price Analysis", 
            "💰 Cost Comparison", 
            "📈 Usage Patterns", 
            "🗓️ Yearly Summary", 
            "⬇️ Download",
            "❓ FAQ",
            "ℹ️ About"
        ]
    else: # Basic Mode
        tab_options = [
            "🏠 Dashboard",
            "🗓️ Yearly Summary", 
            "⬇️ Download",
            "❓ FAQ",
            "ℹ️ About"
        ]
    
    tabs = st.tabs(tab_options)
    
    # --- Perform initial analysis and render sidebar components that modify the dataframe ---
    # This should be done once, before the tab loop.
    df_classified, base_threshold, peak_threshold = analysis.classify_usage(df_merged, config.LOCAL_TIMEZONE)
    df_analysis_base = ui_components.render_absence_days(df_classified, base_threshold, config.ABSENCE_THRESHOLD)

    for i, tab_name in enumerate(tab_options):
        with tabs[i]:
            # Icon-stripped name for matching
            clean_tab_name = " ".join(tab_name.split(" ")[1:])
            tabs_with_tariff_selection = ["Dashboard", "Spot Price Analysis", "Cost Comparison", "Usage Patterns"]
            
            if clean_tab_name in tabs_with_tariff_selection:
                # --- Tariff Selection (Main Page) ---
                flex_tariff, static_tariff = ui_components.render_tariff_selection_header(df_merged, tariff_manager, country, key_prefix=clean_tab_name)

                # --- Continue analysis pipeline with potentially filtered data ---
                df_with_shifting = analysis.simulate_peak_shifting(df_analysis_base, shift_percentage)
                df_analysis = tariff_manager.run_cost_analysis(df_with_shifting, flex_tariff, static_tariff)
                ui_components.render_recommendation(df_analysis, flex_tariff, static_tariff)
                
                if clean_tab_name == "Dashboard":
                    ui_components.render_basic_dashboard_tab(df_analysis, static_tariff, base_threshold, peak_threshold)
                elif clean_tab_name == "Spot Price Analysis":
                    ui_components.render_price_analysis_tab(df_analysis.copy(), static_tariff)
                elif clean_tab_name == "Cost Comparison":
                    ui_components.render_cost_comparison_tab(df_analysis)
                elif clean_tab_name == "Usage Patterns":
                    ui_components.render_usage_pattern_tab(df_analysis, base_threshold, peak_threshold)

            # Render other tabs that don't need the tariff selection
            elif clean_tab_name == "Yearly Summary":
                ui_components.render_yearly_summary_tab(df_analysis_base) # Use base analysis data
            elif clean_tab_name == "Download":
                ui_components.render_download_tab(df_analysis_base, start_date, end_date) # Use base analysis data
            elif clean_tab_name == "FAQ":
                ui_components.render_faq_tab()
            elif clean_tab_name == "About":
                ui_components.render_about_tab()

    ui_components.render_footer()
    
if __name__ == "__main__":
    main()