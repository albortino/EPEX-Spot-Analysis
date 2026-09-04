import streamlit as st
import pandas as pd
import random
from datetime import date
from typing import Optional
import numpy as np
import plotly.graph_objects as go
import io

from src.i18n import t
from src.config import *
from src.tariffs import Tariff, TariffManager, TariffType, SpotTariff, FixedTariff, TimeVariableTariff
from src.utils import to_excel, get_intervals_per_day, get_aggregation_config, has_granular_resolution, get_min_max_date, DataQuality
import src.charts as charts
from src.logger import logger

# --- Introduction ---
def render_intro():
    """Renders a modern, card-based introduction for the user."""
    #st.title(t('intro_title'))
    st.markdown(f"<h2 style='text-align: center;'>{t('intro_subtitle')}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h6 style='text-align: center;color: #808080;'>{t('intro_subtitle_detail')}</h6>", unsafe_allow_html=True)
    
    # Spacing
    st.container(height=50, border=False)

    
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(f"<div style='text-align: center;'><h3>{t('intro_step1_header')}</h3><p style='font-size: 2em; margin-bottom: -10px;'>📤</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'>{t('intro_step1_text')}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='text-align: center;'><h3>{t('intro_step2_header')}</h3><p style='font-size: 2em; margin-bottom: -10px;'>🔍</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'>{t('intro_step2_text')}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='text-align: center;'><h3>{t('intro_step3_header')}</h3><p style='font-size: 2em; margin-bottom: -10px;'>💡</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'>{t('intro_step3_text')}</div>", unsafe_allow_html=True)
    
    # Spacing
    st.container(height=75, border=False)
    
    st.info(t('intro_welcome_message'), icon="👋")
    
    st.warning(f"**{t('intro_important_notice')}**", icon="🔒")

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

def _render_tariff_selection_widgets(_tariff_manager: TariffManager, expanded: bool = True, key_prefix: str = "") -> tuple[Tariff, Tariff, Tariff]:
    """Renders tariff selection expanders in the UI for user customization across 3 tariff types."""
    logger.log("Rendering Tariff Selection")
    
    col1, col2, col3 = st.columns(3)
    final_tariffs = {}

    # 1. Spot Tariff Selector
    with col1:
        with st.expander(t("flexible_plan_title"), expanded=expanded):
            spot_options = _tariff_manager.get_flex_tariffs_with_custom()
            selected_name = st.selectbox(
                t("select_tariff_type", tariff_type=t("flexible_plan_title")),
                options=list(spot_options.keys()),
                index=len(spot_options) - 1,
                key=f"{key_prefix}_select_spot"
            )
            selected_tariff = spot_options[selected_name]
            price_kwh = st.number_input(t("on_top_price"), value=selected_tariff.price_kwh, min_value=0.0, step=0.001, format="%.4f", key=f"{key_prefix}_spot_price")
            price_kwh_pct = st.number_input(t("variable_price_pct"), value=selected_tariff.price_kwh_pct, min_value=0.0, max_value=100.0, step=1.0, format="%.1f", key=f"{key_prefix}_spot_pct")
            monthly_fee = st.number_input(t("monthly_fee"), value=selected_tariff.monthly_fee, min_value=0.0, step=1.0, format="%.2f", key=f"{key_prefix}_spot_fee")
            usage_tax = st.checkbox(t("include_usage_fee"), value=False, key=f"{key_prefix}_spot_usage_tax")
            final_tariffs["spot"] = SpotTariff(
                name=selected_name,
                price_kwh=price_kwh,
                price_kwh_pct=price_kwh_pct,
                monthly_fee=monthly_fee,
                usage_tax=usage_tax,
                link=selected_tariff.link
            )

    # 2. Time-Variable Tariff Selector
    with col2:
        with st.expander(t("variable_plan_title"), expanded=expanded):
            var_options = _tariff_manager.get_variable_tariffs_with_custom()
            selected_name = st.selectbox(
                t("select_tariff_type", tariff_type=t("variable_plan_title")),
                options=list(var_options.keys()),
                index=len(var_options) - 1,
                key=f"{key_prefix}_select_var"
            )
            selected_tariff = var_options[selected_name]
            summer_sun_price = st.number_input(t("summer_sun_price"), value=selected_tariff.summer_sun_price if selected_tariff.summer_sun_price is not None else 0.05, min_value=0.0, step=0.001, format="%.4f", key=f"{key_prefix}_var_summer_sun")
            has_winter_sun = st.checkbox("WinterSonne (10-16h)", value=(selected_tariff.winter_sun_price is not None), key=f"{key_prefix}_var_has_winter_sun")
            winter_sun_price = None
            if has_winter_sun:
                default_w = selected_tariff.winter_sun_price if selected_tariff.winter_sun_price is not None else 0.10
                winter_sun_price = st.number_input(t("winter_sun_price"), value=default_w, min_value=0.0, step=0.001, format="%.4f", key=f"{key_prefix}_var_winter_sun")
            summer_regular_price = st.number_input(t("summer_regular_price"), value=selected_tariff.summer_regular_price, min_value=0.0, step=0.001, format="%.4f", key=f"{key_prefix}_var_summer_regular")
            winter_regular_price = st.number_input(t("winter_regular_price"), value=selected_tariff.winter_regular_price, min_value=0.0, step=0.001, format="%.4f", key=f"{key_prefix}_var_winter_regular")
            monthly_fee = st.number_input(t("monthly_fee"), value=selected_tariff.monthly_fee, min_value=0.0, step=1.0, format="%.2f", key=f"{key_prefix}_var_fee")
            usage_tax = st.checkbox(t("include_usage_fee"), value=False, key=f"{key_prefix}_var_usage_tax")
            final_tariffs["variable"] = TimeVariableTariff(
                name=selected_name,
                summer_sun_price=summer_sun_price,
                winter_sun_price=winter_sun_price,
                summer_regular_price=summer_regular_price,
                winter_regular_price=winter_regular_price,
                monthly_fee=monthly_fee,
                usage_tax=usage_tax,
                link=selected_tariff.link
            )

    # 3. Fixed Tariff Selector
    with col3:
        with st.expander(t("static_plan_title"), expanded=expanded):
            fixed_options = _tariff_manager.get_static_tariffs_with_custom()
            selected_name = st.selectbox(
                t("select_tariff_type", tariff_type=t("static_plan_title")),
                options=list(fixed_options.keys()),
                index=len(fixed_options) - 1,
                key=f"{key_prefix}_select_fixed"
            )
            selected_tariff = fixed_options[selected_name]
            price_kwh = st.number_input(t("price_per_kwh"), value=selected_tariff.price_kwh, min_value=0.0, step=0.001, format="%.4f", key=f"{key_prefix}_fixed_price")
            monthly_fee = st.number_input(t("monthly_fee"), value=selected_tariff.monthly_fee, min_value=0.0, step=1.0, format="%.2f", key=f"{key_prefix}_fixed_fee")
            usage_tax = st.checkbox(t("include_usage_fee"), value=False, key=f"{key_prefix}_fixed_usage_tax")
            final_tariffs["fixed"] = FixedTariff(
                name=selected_name,
                price_kwh=price_kwh,
                monthly_fee=monthly_fee,
                usage_tax=usage_tax,
                link=selected_tariff.link
            )

    return final_tariffs["spot"], final_tariffs["variable"], final_tariffs["fixed"]

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

            curr_range = st.session_state.get("date_range_selector")
            if isinstance(curr_range, (tuple, list)) and len(curr_range) == 2:
                if curr_range[0] < min_date or curr_range[1] > max_date or curr_range[0] > max_date:
                    st.session_state.date_range_selector = (min_date, max_date)

            selected_range = st.date_input(
                t("date_input_label"),
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                format="DD.MM.YYYY",
                key="date_range_selector",
                label_visibility="collapsed"
            )

            # Quarter selection
            quarter_options = ["All", "Q1", "Q2", "Q3", "Q4"]
            selected_quarter = st.selectbox(
                t("select_quarter_label"),
                options=quarter_options,
                index=0,
                key="quarter_selector"
            )

        # Split into start and end dates
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date, end_date = selected_range[0], max_date
        
        # 4. Load Shifting Simulation
        with st.expander(t("simulate_consumption_shifting"), expanded=False):
            st.markdown(t("simulate_shifting_markdown"), help=t("simulate_shifting_help"))
            shift_percentage = st.slider(t("shift_peak_load_slider"), min_value=0, max_value=100, value=0, step=5)

        return mode, awattar_country, start_date, end_date, selected_quarter, shift_percentage

def render_tariff_selection_header(df: pd.DataFrame, tariff_manager: TariffManager, country: str, key_prefix: str = "") -> tuple[Tariff, Tariff, Tariff]:
    """Renders the main tariff selection UI on the main page."""
    with st.expander(t("select_tariff_plan"), expanded=True):
        with st.container(border=False):
            compare_cheapest = st.checkbox(t("compare_cheapest_tariffs"), value=True, help=t("compare_cheapest_tariffs_help"), key=f"{key_prefix}_compare_cheapest")
            
            if compare_cheapest:
                from src.analysis import compare_all_tariffs
                final_flex_tariff, final_var_tariff, final_static_tariff = compare_all_tariffs(tariff_manager, df, country)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if final_flex_tariff:
                        flex_info = t("cheapest_flex_tariff_info", tariff_name=final_flex_tariff.name)
                    else:
                        flex_info = t("no_predefined_flex_tariffs")
                        final_flex_tariff = SpotTariff(name="Dummy", price_kwh=0, monthly_fee=0)
                    st.info(flex_info)
                with col2:
                    if final_var_tariff:
                        var_info = t("cheapest_variable_tariff_info", tariff_name=final_var_tariff.name)
                    else:
                        var_info = t("no_predefined_variable_tariffs")
                        final_var_tariff = TimeVariableTariff(name="Dummy", regular_price=0, monthly_fee=0)
                    st.info(var_info)
                with col3:
                    static_info = t("cheapest_static_tariff_info", tariff_name=final_static_tariff.name) if final_static_tariff else t("no_predefined_static_tariffs")
                    if not final_static_tariff:
                        final_static_tariff = FixedTariff(name="Dummy", price_kwh=0, monthly_fee=0)
                    st.info(static_info)
                
                return final_flex_tariff, final_var_tariff, final_static_tariff
            else:
                return _render_tariff_selection_widgets(tariff_manager, expanded=True, key_prefix=key_prefix)

def render_absence_days(df: pd.DataFrame, base_threshold: float, absence_threshold: float) -> pd.DataFrame:
    """Adds a sidebar option to remove days with very low consumption."""
    # Lazily import to avoid circular dependency
    from src.analysis import compute_absence_data

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
def render_recommendation(
    df: pd.DataFrame,
    flex_tariff: Tariff,
    static_tariff: Tariff,
    variable_tariff: Optional[Tariff] = None
):
    """Displays the final tariff recommendation comparing spot, variable, and fixed plans."""
    logger.log("Rendering Recommendation")
    from src.analysis import compute_peak_timing_score

    is_granular = has_granular_resolution(df)
    if not is_granular:
        st.warning(t("recommendation_only_for_granular_data"))
        return

    tariffs_dict = {
        "spot": (flex_tariff, df["total_cost_flexible"].sum() if "total_cost_flexible" in df.columns else df["total_cost_spot"].sum()),
        "fixed": (static_tariff, df["total_cost_static"].sum() if "total_cost_static" in df.columns else df["total_cost_fixed"].sum())
    }
    if variable_tariff is not None and "total_cost_variable" in df.columns:
        tariffs_dict["variable"] = (variable_tariff, df["total_cost_variable"].sum())

    costs = {k: v[1] for k, v in tariffs_dict.items()}
    cheapest_type = min(costs, key=costs.get)
    most_expensive_type = max(costs, key=costs.get)
    savings = costs[most_expensive_type] - costs[cheapest_type]
    winning_tariff, _ = tariffs_dict[cheapest_type]
    link_text = t("check_cheapest_offer", link=winning_tariff.link) if winning_tariff.link else ""

    peak_ratio = compute_peak_timing_score(df)
    if peak_ratio > 0.4:
        additional_text = t("peak_ratio_good_fit", peak_ratio=peak_ratio)
    else:
        additional_text = t("peak_ratio_potential")

    type_name_map = {
        "spot": t("tariff_type_spot"),
        "variable": t("tariff_type_variable"),
        "fixed": t("tariff_type_fixed"),
    }
    expensive_option = type_name_map.get(most_expensive_type, most_expensive_type.title())

    if cheapest_type == "spot":
        st.success(t("flex_plan_recommended", savings=savings, expensive_option=expensive_option, additional_text=additional_text, link_text=link_text), icon="✅")
    elif cheapest_type == "variable":
        st.success(t("variable_plan_recommended", savings=savings, expensive_option=expensive_option, additional_text="", link_text=link_text), icon="✅")
    else:
        st.info(t("static_plan_recommended", savings=savings, expensive_option=expensive_option, abs_savings=savings, link_text=link_text), icon="ℹ️")

    # Spacing
    st.container(height=50, border=False)

# --- Tab: Spot Price Analysis ---

def render_price_analysis_tab(df: pd.DataFrame, static_tariff: Tariff):
    """Renders the interactive analysis of electricity spot prices."""
    logger.log("Rendering Price Analysis Tab")
    # Lazily import to avoid circular dependency
    from src.analysis import compute_price_distribution_data, compute_heatmap_data
    
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

def _render_usage_profile_section(df: pd.DataFrame, base_threshold: float, peak_threshold: float, key_prefix: str = "basic"):
    """Renders the Usage Profile section with Marimekko chart and toggle between spot and variable prices."""
    from src.analysis import compute_usage_profile_data

    st.subheader(t("usage_profile_header"))

    price_mode = "spot"
    if "variable_price_eur_kwh" in df.columns:
        selected_mode = st.segmented_control(
            t("marimekko_price_basis_label"),
            options=["spot", "variable"],
            format_func=lambda x: t("flexible_plan_title") if x == "spot" else t("variable_plan_title"),
            default="spot",
            key=f"{key_prefix}_mekko_price_mode",
            label_visibility="collapsed"
        )
        price_mode = selected_mode if selected_mode else "spot"

    markdown_text = t("usage_profile_markdown_variable") if price_mode == "variable" else t("usage_profile_markdown_spot")
    st.markdown(markdown_text)

    col1, col2, _, _ = st.columns(4)
    col1.metric(t("base_load_threshold_metric"), f"{base_threshold:.3f} kWh", help=t("base_load_threshold_help"))
    absolute_peak_threshold = base_threshold + peak_threshold
    col2.metric(t("peak_sustain_threshold_metric"), f"{absolute_peak_threshold:.3f} kWh", help=t("peak_sustain_threshold_help"))

    profile_data = compute_usage_profile_data(df, price_type=price_mode)
    if not profile_data.empty:
        marimekko_fig = charts.get_marimekko_chart(profile_data, price_type=price_mode)
        st.plotly_chart(marimekko_fig, config={"width": "stretch"}, key=f"{key_prefix}_marimekko_chart")

def render_basic_dashboard_tab(df: pd.DataFrame, static_tariff: Tariff, base_threshold: float, peak_threshold: float):
    """Renders the content for the 'Basic Dashboard' tab."""
    logger.log("Rendering Basic Dashboard Tab")
    # Lazily import to avoid circular dependency
    from src.analysis import compute_price_distribution_data, compute_cost_comparison_data, compute_consumption_quartiles, compute_usage_profile_data

    # Consumption Summary Metrics
    total_kwh = df["consumption_kwh"].sum()
    days_count = max((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400, 1)
    avg_month_kwh = total_kwh / (days_count / 30.4375)
    est_year_kwh = total_kwh / (days_count / 365.25)

    from src.analysis import compute_peak_timing_score
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(t("total_consumption_metric"), f"{total_kwh:,.1f} kWh")
    col2.metric(t("days_analyzed_metric"), f"{days_count:.0f}")
    col3.metric(t("avg_consumption_per_month_metric"), f"{avg_month_kwh:,.1f} kWh")
    col4.metric(t("estimated_consumption_per_year_metric"), f"{int(round(est_year_kwh)):,} kWh")

    if has_granular_resolution(df):
        score = compute_peak_timing_score(df)
        col5.metric(t("peak_timing_score_metric"), f"{score:.0%}", help=t("peak_timing_score_help"))

    # 1. Price Chart (Monthly)
    st.subheader(t("price_over_time_header"))
    st.markdown(t("price_over_time_markdown"))
    
    resolution = st.radio(t("price_analysis_resolution_label"), ("Monthly", "Weekly", "Hourly"), horizontal=True, key="basic_res")

    df_price = compute_price_distribution_data(df, resolution)
    price_fig = charts.get_price_chart(df_price, static_tariff.price_kwh)
    st.plotly_chart(price_fig, config={"width": "stretch"}, key="basic_price_chart")

    # 2. Average Price per kWh (only for granular flexible comparison)
    is_granular = has_granular_resolution(df) and get_intervals_per_day(df) > 1
    if is_granular:
        st.subheader(t("avg_price_per_kwh_header"))
        st.markdown(t("avg_price_per_kwh_markdown"))
        df_summary = compute_cost_comparison_data(df, "Monthly") # Always use monthly for this overview chart
        if not df_summary.empty:
            df_summary["Avg Static Price"] = df_summary["Total Static Cost"] / df_summary["Total Consumption"]
            df_summary["Avg. Flexible Price"] = df_summary["Total Flexible Cost"] / df_summary["Total Consumption"]
            avg_price_fig = charts.get_avg_price_chart(df_summary, is_granular)
            st.plotly_chart(avg_price_fig, config={"width": "stretch"}, key="basic_avg_price_chart")

    # 3. Daily Usage
    st.subheader(t("daily_consumption_header"))
    intervals = get_intervals_per_day(df)
    if is_granular:
        st.markdown(t("daily_consumption_markdown"))
        df_median_spot = compute_price_distribution_data(df, "Hourly")
        df_consumption_day = compute_consumption_quartiles(df, intervals)
        if not df_consumption_day.empty:
            consumption_fig = charts.get_consumption_chart(df_consumption_day, intervals, df_median_spot)
            st.plotly_chart(consumption_fig, config={"width": "stretch"}, key="basic_consumption_chart")
    else:
        st.markdown(t("daily_consumption_by_type_markdown"))
        daily_cons_fig = charts.get_daily_consumption_chart(df)
        st.plotly_chart(daily_cons_fig, config={"width": "stretch"}, key="basic_consumption_chart")

    # 4. Usage Profile
    if intervals > 24: # Only for granular data
        _render_usage_profile_section(df, base_threshold, peak_threshold, key_prefix="basic")

    # 5. Comparison Table
    render_cost_comparison_tab(df, mode="basic")
    
# --- Tab: Cost Comparison ---
def _display_summary_table(df_summary: pd.DataFrame, is_granular: bool):
    """Helper to display and style the main summary DataFrame with internationalized headers."""
    difference_formatter = lambda v: f"color: {GREEN}" if v > 0 else f"color: {RED}"

    # Define column names using translations
    col_names = {
        "Period": t("col_period"),
        "Total Consumption": t("col_total_consumption"),
        "Total Flexible Cost": t("col_total_flex_cost"),
        "Total Variable Cost": t("col_total_variable_cost"),
        "Total Static Cost": t("col_total_static_cost"),
        "Difference (€)": t("col_difference"),
        "Avg. Flex Price": t("col_avg_flex_price"),
        "Avg. Variable Price": t("col_avg_variable_price"),
        "Avg. Static Price": t("col_avg_static_price")
    }

    # Define formatting for the columns
    style_format = {
        col_names["Total Consumption"]: "{:,.2f} kWh",
        col_names["Total Flexible Cost"]: "€{:,.2f}",
        col_names["Total Static Cost"]: "€{:,.2f}",
        col_names["Difference (€)"]: "€{:.2f}",
        col_names["Avg. Flex Price"]: "€{:.4f}",
        col_names["Avg. Static Price"]: "€{:.4f}"
    }
    if col_names["Total Variable Cost"] in df_summary.columns or "Total Variable Cost" in df_summary.columns:
        style_format[col_names["Total Variable Cost"]] = "€{:,.2f}"
    if col_names["Avg. Variable Price"] in df_summary.columns or "Avg. Variable Price" in df_summary.columns:
        style_format[col_names["Avg. Variable Price"]] = "€{:.4f}"

    # Select and rename columns
    df_display = df_summary.rename(columns=col_names)
    
    if is_granular:
        candidate_cols = ["Period", "Total Consumption", "Total Flexible Cost"]
        if "Total Variable Cost" in df_summary.columns:
            candidate_cols.append("Total Variable Cost")
        candidate_cols.extend(["Total Static Cost", "Difference (€)", "Avg. Flex Price"])
        if "Avg. Variable Price" in df_summary.columns:
            candidate_cols.append("Avg. Variable Price")
        candidate_cols.append("Avg. Static Price")
        cols_to_show = [col_names[c] for c in candidate_cols if c in col_names and col_names[c] in df_display.columns]
        styler = df_display[cols_to_show].style
        if col_names["Difference (€)"] in cols_to_show:
            styler = styler.map(difference_formatter, subset=[col_names["Difference (€)"]])

        # Bold lowest total cost for each row/date
        cost_cols = [col_names[c] for c in ["Total Flexible Cost", "Total Variable Cost", "Total Static Cost"] if c in col_names]
        active_cost_cols = [c for c in cost_cols if c in cols_to_show]
        if len(active_cost_cols) >= 2:
            def highlight_lowest_cost(row):
                styles = [''] * len(row)
                vals = row[active_cost_cols]
                min_val = vals.min()
                for i, col in enumerate(row.index):
                    if col in active_cost_cols and row[col] == min_val:
                        styles[i] = 'font-weight: bold;'
                return styles
            styler = styler.apply(highlight_lowest_cost, axis=1)
    else:
        cols_to_show = [col_names[c] for c in ["Period", "Total Consumption", "Total Static Cost", "Avg. Static Price"] if c in col_names]
        styler = df_display[cols_to_show].style

    styler = styler.format(style_format)
    st.dataframe(styler, hide_index=True, width="stretch")
    return cols_to_show, style_format, difference_formatter

def _compute_col_vals(df: pd.DataFrame, is_granular: bool, func, func_name: str) -> pd.DataFrame:
    """Computes aggregated column values using the provided function. """
    result = {"Period": func_name}
    
    # Always include Total Consumption
    result["Total Consumption"] = func(df["Total Consumption"])

    # Add conditional columns based on data granularity
    if is_granular:
        if "Total Flexible Cost" in df.columns:
            result["Total Flexible Cost"] = func(df["Total Flexible Cost"])
        if "Total Variable Cost" in df.columns:
            result["Total Variable Cost"] = func(df["Total Variable Cost"])
        if "Total Static Cost" in df.columns:
            result["Total Static Cost"] = func(df["Total Static Cost"])
        if "Difference (€)" in df.columns:
            result["Difference (€)"] = func(df["Difference (€)"])
        # For averages, we need to recalculate from totals, not average the averages
        if func_name == "Average":
            total_consumption = df["Total Consumption"].sum()
            if "Total Flexible Cost" in df.columns:
                result["Avg. Flex Price"] = df["Total Flexible Cost"].sum() / total_consumption if total_consumption > 0 else 0
            if "Total Variable Cost" in df.columns:
                result["Avg. Variable Price"] = df["Total Variable Cost"].sum() / total_consumption if total_consumption > 0 else 0
            if "Total Static Cost" in df.columns:
                result["Avg. Static Price"] = df["Total Static Cost"].sum() / total_consumption if total_consumption > 0 else 0
    else:
        result["Total Static Cost"] = func(df["Total Static Cost"])
    
    return pd.DataFrame([result])


def render_cost_comparison_tab(df: pd.DataFrame, mode: str = "expert"):
    """Renders the content for the 'Cost Comparison' tab."""
    # Lazily import to avoid circular dependency
    from src.analysis import compute_cost_comparison_data, compute_cumulative_savings_data
    
    logger.log("Rendering Cost Comparison Tab")

    is_granular = has_granular_resolution(df)
    if not is_granular:
        st.info(t("granular_data_info"))

    # In Basic mode, some charts are already shown. Avoid duplication.
    if mode == "expert":
        resolution = st.radio(t("cost_comparison_resolution_label"), ("Monthly","Weekly", "Daily"), horizontal=True, key="summary_res")
        
        df_summary = compute_cost_comparison_data(df, resolution)
        if df_summary.empty:
            st.warning(t("no_data_for_period"))
            return

        if is_granular:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(t("total_costs_per_period_header"))
                st.markdown(t("total_costs_per_period_markdown"))
                total_cost_fig = charts.get_total_cost_chart(df_summary, is_granular)
                st.plotly_chart(total_cost_fig, config={"width": "stretch"})
            with col2:
                st.subheader(t("avg_price_per_kwh_header"))
                st.markdown(t("avg_price_per_kwh_markdown"))
                df_summary["Avg Static Price"] = df_summary["Total Static Cost"] / df_summary["Total Consumption"]
                df_summary["Avg. Flexible Price"] = df_summary["Total Flexible Cost"] / df_summary["Total Consumption"]
                avg_price_fig = charts.get_avg_price_chart(df_summary, is_granular)
                st.plotly_chart(avg_price_fig, config={"width": "stretch"})
        else:
            st.subheader(t("total_costs_per_period_header"))
            st.markdown(t("total_costs_per_period_markdown"))
            total_cost_fig = charts.get_total_cost_chart(df_summary, is_granular)
            st.plotly_chart(total_cost_fig, config={"width": "stretch"})

        # Cumulative Savings
        st.subheader(t("cumulative_savings_header"))
        st.markdown(t("cumulative_savings_markdown"))
        df_cumulative_savings = compute_cumulative_savings_data(df)
        if not df_cumulative_savings.empty:
            cumulative_savings_fig = charts.get_cumulative_savings_chart(df_cumulative_savings)
            st.plotly_chart(cumulative_savings_fig, config={"width": "stretch"})
            
    # Main DataFrame
    df_summary = compute_cost_comparison_data(df, "Monthly") # Default to monthly for the table
    st.subheader(t("detailed_comparison_table_header"))
    
    saving_info = ""
    cost_cols = [c for c in ["Total Flexible Cost", "Total Variable Cost", "Total Static Cost"] if c in df_summary.columns]
    if is_granular and len(cost_cols) >= 2 and "Difference (€)" in df_summary.columns:
        total_saving_val = df_summary["Difference (€)"].sum()
        saving_info = f" **{t('total_saving_potential_label')}: €{total_saving_val:,.2f}**."
    st.markdown(f"{t('detailed_comparison_table_markdown')}{saving_info}", help=t("detailed_comparison_table_help"))
    
    cols_to_show, style_format, diff_formatter = _display_summary_table(df_summary, is_granular)

    # Calculate totals
    df_totals = _compute_col_vals(df_summary, is_granular, sum, t("total_row_label"))
    df_means = _compute_col_vals(df_summary, is_granular, lambda x: x.mean(), t("average_row_label"))
    df_display = pd.concat([df_totals, df_means], ignore_index=True)

    # Style and display the totals DataFrame
    st.text(t("total_and_average_values_text"))
    df_totals_display = df_display.rename(columns={col: t(f"col_{col.lower().replace(' (€)', '').replace(' ', '_')}") for col in df_display.columns})
    totals_styler = df_totals_display.style
    if is_granular:
        totals_styler = totals_styler.map(diff_formatter, subset=[t("col_difference")])
        totals_cost_cols = [t("col_total_flex_cost"), t("col_total_variable_cost"), t("col_total_static_cost")]
        active_totals_cost_cols = [c for c in totals_cost_cols if c in df_totals_display.columns]
        if len(active_totals_cost_cols) >= 2:
            def highlight_totals_min(row):
                styles = [''] * len(row)
                vals = row[active_totals_cost_cols]
                min_val = vals.min()
                for i, col in enumerate(row.index):
                    if col in active_totals_cost_cols and row[col] == min_val:
                        styles[i] = 'font-weight: bold;'
                return styles
            totals_styler = totals_styler.apply(highlight_totals_min, axis=1)

    totals_styler = totals_styler.format(style_format)
    st.dataframe(totals_styler.hide(axis="index"), hide_index=True, width="stretch")
    
    # --- Yearly Summary ---
    # Lazily import to avoid circular dependency
    from src.analysis import compute_yearly_summary

    df_yearly = compute_yearly_summary(df)
    if not df_yearly.empty:
        st.subheader(t("yearly_summary_header"))
        st.text(t("yearly_summary_text"))

        # Rename 'Year' to 'Period' to match the display function
        df_yearly = df_yearly.rename(columns={"Year": "Period"})
        df_yearly["Period"] = df_yearly["Period"].astype(str)

        # Display the yearly table using the same helper
        _display_summary_table(df_yearly, is_granular)


# --- Tab: Usage Patterns ---

def render_usage_pattern_tab(df: pd.DataFrame, base_threshold: float, peak_threshold: float):
    """Renders the content for the 'Usage Patterns' tab."""
    # Lazily import to avoid circular dependency
    from src.analysis import (
        compute_consumption_quartiles, compute_price_distribution_data,
        compute_consumption_trend_and_forecast, fit_forecast_model,
        compute_usage_profile_data, compute_example_day
    )
    
    intervals = get_intervals_per_day(df)
    is_granular = has_granular_resolution(df)

    if is_granular:
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

        st.subheader(t("consumption_over_time_header"))
        st.markdown(f"#### {t('daily_consumption_header')}\n{t('daily_consumption_markdown')}")

        df_consumption_day = compute_consumption_quartiles(df_filtered, intervals)
        if not df_consumption_day.empty:
            df_median_spot = compute_price_distribution_data(df_filtered, "Hourly")
            consumption_fig = charts.get_consumption_chart(df_consumption_day, intervals, df_median_spot)
            st.plotly_chart(consumption_fig, config={"width": "stretch"})
    else:
        df_filtered = df[df["consumption_kwh"] > 0].copy()
        day_filter = "All Days"

        st.subheader(t("consumption_over_time_header"))
        st.markdown(f"#### {t('daily_consumption_header')}\n{t('daily_consumption_by_type_markdown')}")
        daily_cons_fig = charts.get_daily_consumption_chart(df)
        st.plotly_chart(daily_cons_fig, config={"width": "stretch"})
        
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
    _render_usage_profile_section(df_filtered, base_threshold, peak_threshold, key_prefix="expert")

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

# --- Tab: Download Data ---

@st.cache_data(ttl=3600)
def _compute_download_data(df: pd.DataFrame, flex_tariff: Tariff) -> tuple[bytes, bytes]:
    """Prepares and caches the Excel file bytes for download."""
    logger.log("Computing Download Data")
    
    df_download = df.copy()

    # Calculate the flexible price per kWh based on the selected tariff
    if "spot_price_eur_kwh" in df_download.columns and flex_tariff:
        flex_price = df_download["spot_price_eur_kwh"] * (1 + flex_tariff.price_kwh_pct / 100) + flex_tariff.price_kwh
        if flex_tariff.usage_tax:
            flex_price *= 1.06
        df_download["flex_price_eur_kwh"] = flex_price

    # Prepare data for spot price-only download (hourly resolution)
    spot_cols = ["timestamp", "spot_price_eur_kwh"]
    if "flex_price_eur_kwh" in df_download.columns:
        spot_cols.append("flex_price_eur_kwh")
        
    excel_spot_data_df = df_download.set_index("timestamp").resample("h").first().reset_index()[spot_cols].dropna()
    excel_spot_bytes = to_excel(excel_spot_data_df)
    
    # Prepare full analysis data for download
    excel_full_bytes = to_excel(df_download.drop(columns=["date"], errors="ignore"))

    return excel_full_bytes, excel_spot_bytes

def render_download_tab(df: pd.DataFrame, flex_tariff: Tariff, start_date: date, end_date: date):
    """Renders the content for the Download tab."""
    excel_full_data, excel_spot_data = _compute_download_data(df, flex_tariff)
    
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
    st.container(height=200, border=False)
    st.markdown(f'<div class="footer"><p style="text-align: center;">{t("footer_text")}</p></div>', unsafe_allow_html=True)

# --- Data Quality ---
def render_data_quality(quality: DataQuality, coverage: float | None = None) -> None:
    """Show the evidence behind a result before presenting a tariff recommendation."""
    with st.expander(t("dataquality"), expanded=not quality.usable):
        st.caption(quality.message)
        cols = st.columns(4)
        cols[0].metric("Rows", f"{quality.rows:,}")
        cols[1].metric("Resolution", f"{quality.resolution_minutes or '–'} min")
        cols[2].metric("Duplicates", quality.duplicates)
        cols[3].metric("Gaps", quality.gaps)
        if coverage is not None:
            st.metric("Spot-price coverage", f"{coverage:.0%}")

