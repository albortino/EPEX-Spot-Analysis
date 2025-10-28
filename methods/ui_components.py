import streamlit as st
import pandas as pd
import random
from datetime import date
import numpy as np
import plotly.graph_objects as go
import io

from methods.i18n import t
from methods.config import *
from methods.tariffs import Tariff, TariffManager, TariffType
from methods.utils import to_excel, get_intervals_per_day, get_aggregation_config, calculate_granular_data, get_min_max_date
import methods.charts as charts
from methods.logger import logger

# --- Introduction ---
def render_intro():
    st.markdown(f"## {t('intro_title')}\n\n{t('intro_subtitle')}")
    st.info(t('intro_welcome_message'))
    st.markdown(f"### {t('intro_introduction_header')}\n{t('intro_introduction_text')}\n\n"
                f"**{t('intro_important_notice')}**")


# --- Sidebar and Input Controls ---
def render_language_selection():
    """ Renders the language select widget. """
    with st.sidebar:
        # Language Selector
        lang_map = {"de": "Deutsch", "en": "English"}
        lang_options = list(lang_map.keys())
        selected_lang_key = st.selectbox(
            "Language / Sprache", 
            options=lang_options, 
            format_func=lambda x: lang_map[x],
            index=lang_options.index(st.session_state.get("lang", "de"))
        )
        if st.session_state.get("lang") != selected_lang_key:
            st.session_state["lang"] = selected_lang_key
            st.rerun()


def render_upload_file():
    """Renders the file upload widget and returns the uploaded file."""

    with st.sidebar:
        st.header(t("upload_data"))
        
        if DEBUG:
            if st.button(t("load_example_data")):
                try:

                    # Read the example data file from disk
                    with open("resources/EXAMPLE-DATA-15M.csv", "r") as f:
                        example_data_content = f.read()
                    
                    # Create a BytesIO object from the file content and ensure the content is encoded to bytes.
                    example_data_io = io.BytesIO(example_data_content.encode('utf-8'))
                    
                    # Directly update the session state for the file_uploader key
                    st.session_state["file_uploader"] = example_data_io
                    
                    # Trigger a rerun of the Streamlit app.
                    # On the rerun, the st.file_uploader widget will now read the BytesIO object
                    # from st.session_state["file_uploader"], and its return value will be populated.
                    st.rerun()
                    
                except FileNotFoundError:
                    st.error(t("example_data_not_found"))
                    
                except Exception as e:
                    st.error(t("error_loading_example_data", e=e))
                        

        uploaded_file_widget = st.file_uploader(
            t("upload_prompt"),
            type=["csv"],
            help=t("upload_help"))
        
        if not uploaded_file_widget:
            st.caption(t("upload_caption"))
        else:
            st.session_state["file_uploader"] = uploaded_file_widget

    return st.session_state.get("file_uploader")

def _render_tariff_selection_widgets(_tariff_manager: TariffManager, expanded: bool = True, key_prefix: str = "") -> tuple[Tariff, Tariff]:
    """Renders tariff selection expanders in the UI for user customization."""
    logger.log("Rendering Tariff Selection")
    
    options = [
        (t("flexible_plan_title"), _tariff_manager.get_flex_tariffs_with_custom()),
        (t("static_plan_title"), _tariff_manager.get_static_tariffs_with_custom())
    ]

    col1, col2 = st.columns(2)
    final_tariffs = {}

    # Helper function to render a single tariff selector
    def render_selector(column, title, tariff_options, tariff_key):
        with column:
            with st.expander(title, expanded=expanded):
                tariff_type_str = title.split(" ")[0]
                selected_name = st.selectbox(
                    t("select_tariff_type", tariff_type=tariff_type_str),
                    options=list(tariff_options.keys()),
                    index=len(tariff_options) - 1,
                    key=f"{key_prefix}_select_{tariff_type_str}"
                )
                selected_tariff = tariff_options[selected_name]
                
                price_label = t("on_top_price") if selected_tariff.type == TariffType.FLEXIBLE else t("price_per_kwh")
                
                price_kwh = st.number_input(price_label, value=selected_tariff.price_kwh, min_value=0.0, step=0.001, format="%.4f", key=f"{key_prefix}_{tariff_type_str}_price")
                
                price_kwh_pct = 0.0
                if selected_tariff.type == TariffType.FLEXIBLE:
                    price_kwh_pct = st.number_input(t("variable_price_pct"), value=selected_tariff.price_kwh_pct, min_value=0.0, max_value=100.0, step=1.0, format="%.1f", key=f"{key_prefix}_{tariff_type_str}_pct")
                monthly_fee = st.number_input(t("monthly_fee"), value=selected_tariff.monthly_fee, min_value=0.0, step=1.0, format="%.2f", key=f"{key_prefix}_{tariff_type_str}_fee")
                usage_tax = st.checkbox(t("include_usage_fee"), value=False, key=f"{key_prefix}_{tariff_type_str}_usage_tax")
                
                final_tariffs[tariff_key] = Tariff(
                    name=selected_name, 
                    type=selected_tariff.type, 
                    price_kwh=price_kwh, 
                    monthly_fee=monthly_fee, 
                    price_kwh_pct=price_kwh_pct,
                    usage_tax=usage_tax)

    # Render selectors in columns
    render_selector(col1, options[0][0], options[0][1], "flex")
    render_selector(col2, options[1][0], options[1][1], "static")

    return final_tariffs["flex"], final_tariffs["static"]

def render_sidebar_inputs(df: pd.DataFrame) -> tuple[str, str, date, date, str, float]:
    """Renders all sidebar inputs and returns the configuration values."""
    logger.log("Rendering Sidebar")
    with st.sidebar:
        st.header(t("configuration"))

        # Mode Selection
        is_expert_mode = st.toggle(
            "Expert Mode",
            value=False,
            help="Enable for in-depth analysis and more configuration options."
        )
        mode = "Expert" if is_expert_mode else "Basic"
        
        # 1. Country Selection for EPEX
        country_select = {"Austria": "at", "Germany": "de"}
        
        with st.expander(t("select_country"), expanded=False):
            selected_country = st.selectbox(label=t("select_country_label"), options=country_select.keys(), index=0)
            awattar_country = country_select[selected_country]

        # 2. Analysis Period Selection (with Reset button)
        with st.expander(t("select_analysis_period"), expanded=True):
            min_date, max_date = get_min_max_date(df, today_as_max=TODAY_IS_MAX_DATE)
            
            if st.button(t("reset_to_default")):
                st.session_state.date_range_selector = (min_date, max_date)
                st.rerun()

            # The `key` parameter is crucial. It links the widget's state to st.session_state
            selected_range = st.date_input(
                t("date_input_label"),
                value=(min_date, max_date),  # This sets the default on the first run
                min_value=min_date,
                max_value=max_date,
                format="DD.MM.YYYY",
                key="date_range_selector",  # The link to session state
                label_visibility="collapsed"
            )

            # Quarter selection
            quarter_options = ["All", "Q1", "Q2", "Q3", "Q4"]
            selected_quarter = st.selectbox(
                t("select_quarter_label"),
                options=quarter_options,
                index=0, # Default to "All"
                key="quarter_selector"
            )

        # Split into start and end dates
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            # Fallback for the rare case where only one date is returned
            start_date, end_date = selected_range[0], max_date
        
        # 4. Load Shifting Simulation
        with st.expander(t("simulate_consumption_shifting"), expanded=False):
            st.markdown(t("simulate_shifting_markdown"), help=t("simulate_shifting_help"))
            shift_percentage = st.slider(t("shift_peak_load_slider"), min_value=0, max_value=100, value=0, step=5)

        return mode, awattar_country, start_date, end_date, selected_quarter, shift_percentage

def render_tariff_selection_header(df: pd.DataFrame, tariff_manager: TariffManager, country: str, key_prefix: str = "") -> tuple[Tariff, Tariff]:
    """Renders the main tariff selection UI on the main page."""
    with st.expander(t("select_tariff_plan"), expanded=True):
        with st.container(border=False):

            compare_cheapest = st.checkbox(t("compare_cheapest_tariffs"), value=True, help=t("compare_cheapest_tariffs_help"), key=f"{key_prefix}_compare_cheapest")
            
            if compare_cheapest:
                # Lazily import to avoid circular dependency
                from methods.analysis import compare_all_tariffs
                final_flex_tariff, final_static_tariff = compare_all_tariffs(tariff_manager, df, country)
                
                col1, col2 = st.columns(2)
                with col1:
                    if final_flex_tariff:
                        flex_info = t("cheapest_flex_tariff_info", tariff_name=final_flex_tariff.name)
                    else:
                        flex_info = t("no_predefined_flex_tariffs")
                        # Create a dummy tariff to avoid errors downstream
                        final_flex_tariff = Tariff(name="Dummy", type=TariffType.FLEXIBLE, price_kwh=0, monthly_fee=0)
                    st.info(flex_info)
                with col2:
                    static_info = t("cheapest_static_tariff_info", tariff_name=final_static_tariff.name) if final_static_tariff else t("no_predefined_static_tariffs")
                    st.info(static_info)
                
                return final_flex_tariff, final_static_tariff
            else:
                final_flex_tariff, final_static_tariff = _render_tariff_selection_widgets(tariff_manager, expanded=True, key_prefix=key_prefix)
                return final_flex_tariff, final_static_tariff

def render_absence_days(df: pd.DataFrame, base_threshold: float, absence_threshold: float) -> pd.DataFrame:
    """Adds a sidebar option to remove days with very low consumption."""
    # Lazily import to avoid circular dependency
    from methods.analysis import compute_absence_data

    # --- Main Page Components ---

    # Ensure the 'date' column exists before any processing.
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df['date'] = df['timestamp'].dt.date

    with st.sidebar:
        absence_days = compute_absence_data(df, base_threshold, absence_threshold)
        if absence_days:
            with st.expander(t("remove_absence_days"), expanded=False):
                st.text(t("remove_absence_days_help", count=len(absence_days)), help=t("remove_absence_days_long_help", threshold=absence_threshold))
                select_all = st.checkbox(t("exclude_all_days_checkbox"), value=False, key="absence_select_all")
                default_selection = absence_days if select_all else []
                excluded_days = st.multiselect(t("multiselect_excluded_days"), options=absence_days, default=default_selection, key="absence_multiselect")
                
            if excluded_days:
                # Filter out the selected absence days
                return df[~df["date"].isin(excluded_days)]
    return df

@st.cache_data(ttl=3600)
def render_recommendation(df: pd.DataFrame, flex_tariff: Tariff, static_tariff: Tariff):
    """Displays the final tariff recommendation based on calculated savings."""
    logger.log("Rendering Recommendation")

    is_granular_data = calculate_granular_data(df)
    if not is_granular_data:
        st.warning(t("recommendation_only_for_granular_data"))
        return

    savings = df["total_cost_static"].sum() - df["total_cost_flexible"].sum()

    # Calculate the proportion of peak consumption that occurs during the cheapest 25% of hours
    df["price_quantile"] = df.groupby(pd.Grouper(key="timestamp", freq="MS"))["spot_price_eur_kwh"].transform(
        lambda x: pd.qcut(x, 4, labels=False, duplicates="drop"))
    peak_total_kwh = df["peak_load_kwh"].sum()
    peak_cheap_kwh = df[df["price_quantile"] == 0]["peak_load_kwh"].sum()
    peak_ratio = peak_cheap_kwh / peak_total_kwh if peak_total_kwh > 0 else 0
    

    # Display the appropriate recommendation message
    if savings > 0:
        if peak_ratio > 0.4:
            additional_text = t("peak_ratio_good_fit", peak_ratio=peak_ratio)
        else:
            additional_text = t("peak_ratio_potential")
        
        link_text = t("check_cheapest_offer", link=flex_tariff.link) if flex_tariff.link else ""
        st.success(t("flex_plan_recommended", savings=savings, additional_text=additional_text, link_text=link_text), icon="✅")
    elif savings < 0: # Only show warning if there are actual losses
        link_text = t("check_cheapest_offer", link=static_tariff.link) if static_tariff.link else ""
        st.warning(t("static_plan_recommended", abs_savings=-savings, link_text=link_text), icon="⚠️")
    # If savings is 0, no recommendation is strictly needed, or a neutral message could be added.

# --- Tab: Spot Price Analysis ---

def render_price_analysis_tab(df: pd.DataFrame, static_tariff: Tariff):
    """Renders the interactive analysis of electricity spot prices."""
    logger.log("Rendering Price Analysis Tab")
    # Lazily import to avoid circular dependency
    from methods.analysis import compute_price_distribution_data, compute_heatmap_data
    
    # Quartile Price Chart
    st.subheader(t("price_over_time_header"))
    st.markdown(t("price_over_time_markdown"))

    resolution = st.radio(t("price_analysis_resolution_label"), ("Monthly", "Weekly", "Hourly"), horizontal=True, key="price_res")

    df_price = compute_price_distribution_data(df, resolution)
    price_fig = charts.get_price_chart(df_price, static_tariff.price_kwh)
    st.plotly_chart(price_fig, config={"width": "stretch"}, key="price_analysis_chart")

    # Heatmap Analysis
    st.subheader(t("heatmap_header"))
    st.markdown(t("heatmap_markdown"))
    heatmap_data = compute_heatmap_data(df)
    heatmap_fig = charts.get_heatmap(heatmap_data)
    st.plotly_chart(heatmap_fig, config={"width": "stretch"}, key="price_heatmap_chart")

def render_basic_dashboard_tab(df: pd.DataFrame, static_tariff: Tariff, base_threshold: float, peak_threshold: float):
    """Renders the content for the 'Basic Dashboard' tab."""
    logger.log("Rendering Basic Dashboard Tab")
    # Lazily import to avoid circular dependency
    from methods.analysis import compute_price_distribution_data, compute_cost_comparison_data, compute_consumption_quartiles, compute_usage_profile_data

    # 1. Price Chart (Monthly)
    st.subheader(t("price_over_time_header"))
    st.markdown(t("price_over_time_markdown"))
    
    resolution = st.radio(t("price_analysis_resolution_label"), ("Monthly", "Weekly", "Hourly"), horizontal=True, key="basic_res")

    df_price = compute_price_distribution_data(df, resolution)
    price_fig = charts.get_price_chart(df_price, static_tariff.price_kwh)
    st.plotly_chart(price_fig, config={"width": "stretch"}, key="basic_price_chart")

    # 2. Average Price per kWh
    st.subheader(t("avg_price_per_kwh_header"))
    st.markdown(t("avg_price_per_kwh_markdown"))
    df_summary = compute_cost_comparison_data(df, "Monthly") # Always use monthly for this overview chart
    is_granular_data = calculate_granular_data(df)
    if not df_summary.empty:
        df_summary["Avg Static Price"] = df_summary["Total Static Cost"] / df_summary["Total Consumption"]
        if is_granular_data:
            df_summary["Avg. Flexible Price"] = df_summary["Total Flexible Cost"] / df_summary["Total Consumption"]
        
        avg_price_fig = charts.get_avg_price_chart(df_summary, is_granular_data)
        st.plotly_chart(avg_price_fig, config={"width": "stretch"}, key="basic_avg_price_chart")

    # 3. Daily Usage
    st.subheader(t("daily_consumption_header"))
    st.markdown(t("daily_consumption_markdown"))
    intervals = get_intervals_per_day(df)
    df_median_spot = compute_price_distribution_data(df, "Hourly")
    df_consumption_day = compute_consumption_quartiles(df, intervals)
    if not df_consumption_day.empty:
        consumption_fig = charts.get_consumption_chart(df_consumption_day, intervals, df_median_spot)
        st.plotly_chart(consumption_fig, config={"width": "stretch"}, key="basic_consumption_chart")

    # 4. Usage Profile
    if intervals > 24: # Only for granular data
        st.subheader(t("usage_profile_header"))
        st.markdown(t("usage_profile_markdown"))
        
        # Display Thresholds
        col1, col2, _, _ = st.columns(4)
        col1.metric(t("base_load_threshold_metric"), f"{base_threshold:.3f} kWh", help=t("base_load_threshold_help"))
        # The peak_threshold passed is the influenceable part. Add base for the absolute value.
        absolute_peak_threshold = base_threshold + peak_threshold
        col2.metric(t("peak_sustain_threshold_metric"), f"{absolute_peak_threshold:.3f} kWh", help=t("peak_sustain_threshold_help"))

        profile_data = compute_usage_profile_data(df)
        if not profile_data.empty:
            marimekko_fig = charts.get_marimekko_chart(profile_data)      
            st.plotly_chart(marimekko_fig, config={"width": "stretch"}, key="basic_marimekko_chart")

    # 5. Comparison Table
    render_cost_comparison_tab(df, mode="basic")
    
# --- Tab: Cost Comparison ---
def _display_summary_table(df_summary: pd.DataFrame, is_granular_data: bool):
    """Helper to display and style the main summary dataframe."""
    difference_formatter = lambda v: f"color: {GREEN}" if v > 0 else f"color: {RED}"
    granular_col_format = {"Total Consumption": "{:.2f} kWh", "Total Flexible Cost": "€{:.2f}", "Total Static Cost": "€{:.2f}", "Difference (€)": "€{:.2f}"}
    regular_col_format = {"Total Consumption": "{:,.2f} kWh", "Total Static Cost": "€{:.2f}"}

    if is_granular_data:
        cols_to_show = ["Period", "Total Consumption", "Total Flexible Cost", "Total Static Cost", "Difference (€)"]
        styler = df_summary[cols_to_show].style
        styler = styler.map(difference_formatter, subset=["Difference (€)"]) #type: ignore
        styler = styler.format(granular_col_format)
    else:
        cols_to_show = ["Period", "Total Consumption", "Total Static Cost"]
        styler = df_summary[cols_to_show].style
        styler = styler.format(regular_col_format)
        
    st.dataframe(styler, hide_index=True, width="stretch")
    return cols_to_show, granular_col_format, regular_col_format, difference_formatter

def _compute_col_vals(df: pd.DataFrame, is_granular_data: bool, func, func_name: str) -> pd.DataFrame:
    """Computes aggregated column values using the provided function. """
    result = {"Period": func_name}
    
    # Always include Total Consumption
    result["Total Consumption"] = func(df["Total Consumption"])

    # Add conditional columns based on data granularity
    if is_granular_data:
        result["Total Flexible Cost"] = func(df["Total Flexible Cost"])
        result["Total Static Cost"] = func(df["Total Static Cost"])
        result["Difference (€)"] = func(df["Difference (€)"])
    else:
        result["Total Static Cost"] = func(df["Total Static Cost"])
    
    return pd.DataFrame([result])


def render_cost_comparison_tab(df: pd.DataFrame, mode: str = "expert"):
    """Renders the content for the 'Cost Comparison' tab."""
    # Lazily import to avoid circular dependency
    from methods.analysis import compute_cost_comparison_data, compute_cumulative_savings_data
    
    logger.log("Rendering Cost Comparison Tab")

    is_granular_data = calculate_granular_data(df)
    if not is_granular_data:
        st.info(t("granular_data_info"))

    # In Basic mode, some charts are already shown. Avoid duplication.
    if mode == "expert":
        resolution = st.radio(t("cost_comparison_resolution_label"), ("Monthly","Weekly", "Daily"), horizontal=True, key="summary_res")
        
        df_summary = compute_cost_comparison_data(df, resolution)
        if df_summary.empty:
            st.warning(t("no_data_for_period"))
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(t("total_costs_per_period_header"))
            st.markdown(t("total_costs_per_period_markdown"))
            total_cost_fig = charts.get_total_cost_chart(df_summary, is_granular_data)
            st.plotly_chart(total_cost_fig, config={"width": "stretch"})
        with col2:
            st.subheader(t("avg_price_per_kwh_header"))
            st.markdown(t("avg_price_per_kwh_markdown"))
            df_summary["Avg Static Price"] = df_summary["Total Static Cost"] / df_summary["Total Consumption"]
            
            if is_granular_data:
                df_summary["Avg. Flexible Price"] = df_summary["Total Flexible Cost"] / df_summary["Total Consumption"]
            
            avg_price_fig = charts.get_avg_price_chart(df_summary, is_granular_data)
            st.plotly_chart(avg_price_fig, config={"width": "stretch"})

        # Cumulative Savings
        st.subheader(t("cumulative_savings_header"))
        st.markdown(t("cumulative_savings_markdown"))
        df_cumulative_savings = compute_cumulative_savings_data(df)
        if not df_cumulative_savings.empty:
            cumulative_savings_fig = charts.get_cumulative_savings_chart(df_cumulative_savings)
            st.plotly_chart(cumulative_savings_fig, config={"width": "stretch"})
            
    st.subheader(t("detailed_comparison_table_header"))
    st.text(t("detailed_comparison_table_markdown"), help=t("detailed_comparison_table_help"))
    
    # Main DataFrame
    df_summary = compute_cost_comparison_data(df, "Monthly") # Default to monthly for the table
    cols_to_show, granular_format, regular_format, diff_formatter = _display_summary_table(df_summary, is_granular_data)

    # Calculate totals
    df_totals = _compute_col_vals(df_summary, is_granular_data, sum, "Total")
    df_means = _compute_col_vals(df_summary, is_granular_data, lambda x: x.mean(), "Average")
    df_display = pd.concat([df_totals, df_means], ignore_index=True)

    # Style and display the totals DataFrame
    if is_granular_data:
        totals_styler = df_display[cols_to_show].style
        totals_styler = totals_styler.map(diff_formatter, subset=["Difference (€)"]) #type: ignore
        totals_styler = totals_styler.format(granular_format)
    else:
        totals_styler = df_display[cols_to_show].style
        totals_styler = totals_styler.format(regular_format)

    st.text(t("total_and_average_values_text"))
    st.dataframe(totals_styler.hide(axis="index"), hide_index=True, width="stretch")

# --- Tab: Usage Patterns ---

def render_usage_pattern_tab(df: pd.DataFrame, base_threshold: float, peak_threshold: float):
    """Renders the content for the 'Usage Patterns' tab."""
    # Lazily import to avoid circular dependency
    from methods.analysis import (
        compute_consumption_quartiles, compute_price_distribution_data,
        compute_consumption_trend_and_forecast, fit_forecast_model,
        compute_usage_profile_data, compute_example_day
    )
    
    # Allow filtering by day type
    df_filtered = df[df["consumption_kwh"] > 0].copy()
    day_filter_options = {"All Days": t("all_days"), "Weekdays": t("weekdays"), "Weekends": t("weekends")}
    day_filter = st.radio(t("filter_by_day_type"), list(day_filter_options.keys()), format_func=lambda x: day_filter_options[x], horizontal=True)

    if day_filter != "All Days":
        is_weekend = df_filtered["timestamp"].dt.dayofweek >= 5
        df_filtered = df_filtered[is_weekend if day_filter == "Weekends" else ~is_weekend]

    if df_filtered.empty:
        st.warning(t("no_data_for_filter", day_filter=day_filter.lower()))
        return
        
    intervals = get_intervals_per_day(df)
    
    # Consumption Over Time
    df_consumption_day = compute_consumption_quartiles(df_filtered, intervals)
    
    st.subheader(t("consumption_over_time_header"))
    
    # Daily Consumption with Quartiles
    st.markdown(f"#### {t('daily_consumption_header')}\n{t('daily_consumption_markdown')}")

    if not df_consumption_day.empty:
        df_median_spot = compute_price_distribution_data(df_filtered, "Hourly")
        consumption_fig = charts.get_consumption_chart(df_consumption_day, intervals, df_median_spot)
        st.plotly_chart(consumption_fig, config={"width": "stretch"})
        
    # Trend Visualization and Forecast
    try:
        st.subheader(t("consumption_trend_forecast_header"))
        if day_filter != "All Days":
            st.info("Forecasting is only available when 'All Days' are selected.")
        else:
            st.markdown(t("consumption_trend_forecast_markdown"))
            col1, _, _, col2 = st.columns(4) # Use 4 columns such that the metric is on the very right of the screen.

            # Configuration for the forecast.
            with col1:
                days_in_dataset = (df_filtered["timestamp"].max() - df_filtered["timestamp"].min()).days
                
                # Set the default value for the slider based on the number of days in the dataset
                max_value = 365
                if days_in_dataset < 90: 
                    value = 30
                    
                elif days_in_dataset < 365:
                    value = 90
                else:
                    value = 180
                    max_value = 365*2
                
                forecast_days = st.slider(t("forecast_slider_label"), min_value=30, max_value=max_value, value=value, step=10, key="forecast_days", width=300)

            # Calculation the trend data. The fitted model is cached such that only the forecast is recomputed (efficient!).
            trend_data = compute_consumption_trend_and_forecast(df_filtered, forecast_days)

            # Display Results
            if trend_data:
                df_daily_trend, df_forecast, trend_description, trend_metric = trend_data
                model, _ = fit_forecast_model(df_filtered) # We need the model for the components

                # Display the summary metric first, as a key insight
                with col2:
                    st.metric(label=t("underlying_consumption_trend_label"),
                        value=trend_description,
                        delta=f"{trend_metric:.1f}% change over period",
                        delta_color=("inverse" if trend_metric < 0 else "normal"),
                        label_visibility="hidden"
                    )

                # Display the detailed chart
                trend_fig = charts.get_trend_chart(df_daily_trend, df_forecast)
                st.plotly_chart(trend_fig, config={"width": "stretch"})

                # Expander for seasonality components
                with st.expander(t("show_seasonality_details")):
                    st.markdown(t("seasonality_details_markdown"))
                    if model:
                        seasonality_charts = charts.get_seasonality_charts(model, df_forecast)
                        st.plotly_chart(seasonality_charts, config={"width": "stretch"}, key=f"forecast_details")

            else:
                st.info(t("no_trend_info"))

    except AssertionError as e:
        logger.log(f"Error in Forecasting: {e}", severity=1)

    except (KeyError, ValueError) as e:
        logger.log(f"Error in Forecasting: {e}", severity=1)
        st.error(t("forecasting_error"))

            
    # Only show the detailed analysis when consumption data includes 15 minutes intervals.
    if intervals <= 24: # Hourly data or less
        st.info(t("granular_data_needed_for_profile"))
        return

    # Consumption & Usage Profile (Marimekko Chart)
    st.subheader(t("usage_profile_header"))
    st.markdown(t("usage_profile_markdown"))
    profile_data = compute_usage_profile_data(df_filtered)

    # Define columns here so they are always available
    col1, col2, _, _ = st.columns(4)

    # Display Thresholds
    col1.metric(t("base_load_threshold_metric"), f"{base_threshold:.3f} kWh", help=t("base_load_threshold_help"))
    
    # The peak_threshold passed is the influenceable part. Add base for the absolute value.
    absolute_peak_threshold = base_threshold + peak_threshold
    col2.metric(t("peak_sustain_threshold_metric"), f"{absolute_peak_threshold:.3f} kWh", help=t("peak_sustain_threshold_help"))

    if not profile_data.empty:
        marimekko_fig = charts.get_marimekko_chart(profile_data)      
        st.plotly_chart(marimekko_fig, config={"width": "stretch"})

    # Example Day Breakdown
    st.subheader(t("example_day_breakdown_header"))
    st.markdown(t("example_day_breakdown_markdown"))
    
    available_dates = df_filtered["date"].unique().tolist()
    if available_dates:
        if "random_day" not in st.session_state or st.session_state.random_day not in available_dates:
            st.session_state.random_day = random.choice(available_dates)
        
        if st.button(t("show_different_day_button")):
            st.session_state.random_day = random.choice(available_dates)
            st.rerun()
        
        intervals = get_intervals_per_day(df_filtered)
        df_day = compute_example_day(df_filtered, st.session_state.random_day, group=False)
        # Ensure correct stacking order for the bar chart: Base (bottom), Regular, Peak (top).
        df_day = df_day[["Base Load", "Regular Load", "Peak Load"]]
        day_str = st.session_state.random_day.strftime('%A, %Y-%m-%d')
        st.caption(t("example_day_caption", day=day_str, total_kwh=df_day.to_numpy().sum()))
        example_day_fig = charts.get_example_day_chart(df_day, intervals)
        st.plotly_chart(example_day_fig, config={"width": "stretch"})

# --- Tab: Yearly Summary ---

def render_yearly_summary_tab(df: pd.DataFrame):
    """Renders the content for the 'Yearly Summary' tab."""
    # Lazily import to avoid circular dependency
    from methods.analysis import compute_yearly_summary

    st.subheader(t("yearly_summary_header"))
    st.text(t("yearly_summary_text"))

    yearly_agg = compute_yearly_summary(df)
    if yearly_agg.empty:
        st.warning(t("no_yearly_summary_data"))
        return
    
    # Rename columns for a more descriptive display in the table
    display_df = yearly_agg.rename(columns={
        "Avg. Flex Price": "Avg. Flex Price (€/kWh)",
        "Avg. Static Price": "Avg. Static Price (€/kWh)"
    })
        
    style_format: dict = {
        "Total Consumption": "{:,.2f} kWh",
        "Total Flexible Cost": "€{:,.2f}",
        "Total Static Cost": "€{:,.2f}",
        "Avg. Flex Price (€/kWh)": "€{:.4f}",
        "Avg. Static Price (€/kWh)": "€{:.4f}"
    }
    st.dataframe(display_df.style.format(style_format), hide_index=True, width="stretch")

# --- Tab: Download Data ---

@st.cache_data(ttl=3600)
def _compute_download_data(df: pd.DataFrame) -> tuple[bytes, bytes]:
    """Prepares and caches the Excel file bytes for download."""
    logger.log("Computing Download Data")
    
    # Prepare data for spot price-only download (hourly resolution)
    excel_spot_data_df = df.set_index("timestamp").resample("h").first().reset_index()[["timestamp", "spot_price_eur_kwh"]].dropna()
    excel_spot_bytes = to_excel(excel_spot_data_df)
    
    # Prepare full analysis data for download
    excel_full_bytes = to_excel(df.drop(columns=["date"], errors="ignore"))

    return excel_full_bytes, excel_spot_bytes

def render_download_tab(df: pd.DataFrame, start_date: date, end_date: date):
    """Renders the content for the Download tab."""
    excel_full_data, excel_spot_data = _compute_download_data(df)
    
    st.subheader(t("download_header"))
    
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.markdown(t("download_full_analysis_markdown"))
        st.download_button(
            label=t("download_full_analysis_label"),
            data=excel_full_data,
            file_name=f"electricity_analysis_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        st.markdown(t("download_spot_prices_markdown"))
        st.download_button(
            label="Download Spot Prices (XLSX)",
            data=excel_spot_data,
            file_name=f"spot_prices_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# --- FAQ & Help Tab ---
def render_faq_tab():
    """Renders the content for the 'FAQ & Help' tab."""
    st.subheader(t("faq_header"))

    with st.expander(t("faq_what_does_it_do_q"), expanded=True):
        st.markdown(t("faq_what_does_it_do_a"))
    
    with st.expander(t("faq_price_differences_q")):
        st.markdown(t("faq_price_differences_a"))

    with st.expander(t("faq_how_to_use_q")):
        st.markdown(t("faq_how_to_use_a"))

    with st.expander(t("faq_data_file_q")):
        st.markdown(t("faq_data_file_a"))

    with st.expander(t("faq_load_types_q")):
        st.markdown(t("faq_load_types_a"))
    
    with st.expander(t("faq_peak_shifting_q")):
        st.markdown(t("faq_peak_shifting_a"))
        
    st.info(t("faq_footer_info"))

# --- About Tab ---
def render_about_tab():
    """Renders the content for the 'About' tab by displaying the readme.md file."""
    st.header("About This Project")
    try:
        with open("readme.md", "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("readme.md file not found.")
# --- Footer ---
@st.cache_data
def render_footer():
    """Renders the footer with information about the project and further links."""
    
    #st.markdown(footer_css, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f'<div class="footer"><p>{t("footer_text")}</p></div>', unsafe_allow_html=True)