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
    tariff_manager = TariffManager("tariffs_flexible.json", "tariffs_static.json")

    # --- File Upload and Initial Data Processing ---
    uploaded_file = None
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader(
            "First upload consumption data from your network provider.",
            type=["csv"],
            help="See the 'FAQ & Help' tab for tips on getting your data from your network provider.",
            key="file_uploader"
        )
        if not uploaded_file:
            st.caption("For best results, your data should have 15-minute or hourly intervals.")
        
    if not uploaded_file:
        st.markdown("## Electricity Tariff Comparison Dashboard\n\nAnalyze your consumption, compare flexible vs. static tariffs, and understand your usage patterns.")
        st.markdown("### Introduction\nThis project was influenced by https://awattar-backtesting.github.io/, which provides a simple and effective overview.\n\n"
                   "This tool provides further insights into your consumption behavior and help you to choose the most economic tariff plan.\n\n"
                   "For a detailed description and explanation please refer to this project's [Read Me](https://github.com/albortino/EPEX-Spot-Analysis/blob/main/readme.md).\n\n"
                   "IMPORTANT: File uploads are in fact uploaded to a server when the app is running on Steamlit Cloud. Even though no file is stored by this script, others might have access to it. Therefore, please anonymize any personal data first, but don't delete the columns.")
        st.info("👋 Welcome! Please upload your consumption data to begin.")
        ui_components.render_footer()
        return

    df_consumption = data_loader.process_consumption_data(uploaded_file)
    if df_consumption.empty:
        return

    # --- Sidebar and Input Controls ---
    country, start_date, end_date, flex_tariff, static_tariff, shift_percentage = ui_components.render_sidebar_inputs(df_consumption, tariff_manager)
    
    # --- Data Loading and Merging ---
    df_consumption = filter_dataframe(df_consumption, start_date, end_date)
    min_date, max_date = ui_components.get_min_max_date(df_consumption)
    df_spot_prices = data_loader.get_spot_data(country, min_date, max_date)
    df_merged = data_loader.merge_consumption_with_prices(df_consumption, df_spot_prices)

    if df_merged.empty:
        st.warning("No overlapping data found for the selected period. Please check your file's date range.")
        return
    
    # --- Core Analysis Pipeline ---

    df_classified, base_threshold, peak_threshold = analysis.classify_usage(df_merged, config.LOCAL_TIMEZONE)
    df_with_shifting = analysis.simulate_peak_shifting(df_classified, shift_percentage)
    df_analysis = tariff_manager.run_cost_analysis(df_with_shifting, flex_tariff, static_tariff)

    # --- Final Filtering and Rendering ---
    df_analysis = ui_components.render_absence_days(df_analysis, base_threshold)
    
    ui_components.render_recommendation(df_analysis, flex_tariff, static_tariff)

    # Define the "tabs"
    tab_options = [
        "Spot Price Analysis", 
        "Cost Comparison", 
        "Usage Patterns", 
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
        selection_mode="single",
        format_func=lambda x: f"**{x}**", # Makes only text bold without influencing post-processing.
        key="active_tab",  # This key is crucial for statefulness!
        width="stretch",
        label_visibility="collapsed" # Completely hides the label as it's not relevant.
    )
        
    # If/elif blocks render only the selected view's content and keep the user on the same "tab" after an interaction.
    if st.session_state.active_tab == "Spot Price Analysis":
        ui_components.render_price_analysis_tab(df_analysis.copy(), static_tariff)
    
    elif st.session_state.active_tab == "Cost Comparison":
        ui_components.render_cost_comparison_tab(df_analysis)
    
    elif st.session_state.active_tab == "Usage Patterns":
        ui_components.render_usage_pattern_tab(df_analysis, base_threshold, peak_threshold)
    
    elif st.session_state.active_tab == "Yearly Summary":
        ui_components.render_yearly_summary_tab(df_analysis)
    
    elif st.session_state.active_tab == "Download Data":
        ui_components.render_download_tab(df_analysis, start_date, end_date)

    elif st.session_state.active_tab == "FAQ & Help":
        ui_components.render_faq_tab()

    ui_components.render_footer()
    
if __name__ == "__main__":
    main()