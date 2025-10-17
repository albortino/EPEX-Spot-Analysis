import streamlit as st
import pandas as pd
import random
from datetime import date
import json
import numpy as np
import io

from prophet import Prophet

from methods.config import *
from methods.tariffs import Tariff, TariffManager
from methods.utils import get_min_max_date, to_excel, get_intervals_per_day, get_aggregation_config, calculate_granular_data
import methods.data_loader as data_loader
import methods.charts as charts
from methods.logger import logger

# --- Internationalization (i18n) ---

@st.cache_data(ttl=3600)
def _load_translations(language: str) -> dict:
    """Loads the translation file for the selected language."""
    try:
        with open(f"locales/{language}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.log(f"Translation file for language '{language}' not found. Defaulting to English.", severity=1)
        with open("locales/de.json", "r", encoding="utf-8") as f:
            return json.load(f)

def t(key: str, **kwargs):
    """Returns the translated string for a given key."""
    lang = st.session_state.get("lang", "de")
    translations = _load_translations(lang)
    return translations.get(key, key).format(**kwargs)

# --- Introduction ---
def render_intro():
    st.markdown(f"## {t('intro_title')}\n\n{t('intro_subtitle')}")
    st.markdown(f"### {t('intro_introduction_header')}\n{t('intro_introduction_text')}\n\n"
                f"**{t('intro_important_notice')}**")
    st.info(t('intro_welcome_message'))


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


@st.cache_data(ttl=60*10)
def _compare_all_tariffs(df_consumption: pd.DataFrame, _tariff_manager: TariffManager, country: str) -> tuple[Tariff | None, Tariff | None]:
    """Finds the cheapest flex and static tariffs from the predefined lists."""
    logger.log("Calculating cheapest tariff comparison", )
    
    flex_options = _tariff_manager.get_flex_tariffs_with_custom()
    static_options = _tariff_manager.get_static_tariffs_with_custom()
    total_costs = {}
    
    # Calculate total costs for all non-custom tariffs.
    # First, ensure we have spot price data for flexible tariff calculations.
    df = df_consumption.copy()
    if "spot_price_eur_kwh" not in df.columns:
        min_date, max_date = get_min_max_date(df, today_as_max=TODAY_IS_MAX_DATE)
        df_spot_prices = data_loader.get_spot_data(country, min_date, max_date)
        df = data_loader.merge_consumption_with_prices(df, df_spot_prices)

    for name, tariff in flex_options.items():
        if name == "Custom": continue
        total_costs[("flex", name)] = _tariff_manager._calculate_flexible_cost(df, tariff).sum()
    
    for name, tariff  in static_options.items():
        if name == "Custom": continue
        total_costs[("static", name)] = _tariff_manager._calculate_static_cost(df, tariff).sum()
    
    # Identify the cheapest tariffs based on the calculated costs robustly
    final_flex_tariff = None
    flex_keys = [k for k in total_costs if k[0] == "flex"]
    if flex_keys:
        cheapest_flex_key = min(flex_keys, key=total_costs.get) #type: ignore
        final_flex_tariff = flex_options.get(cheapest_flex_key[1])
    else:
        st.warning(t("no_predefined_flex_tariffs"))

    final_static_tariff = None
    static_keys = [k for k in total_costs if k[0] == "static"]
    if static_keys:
        cheapest_static_key = min(static_keys, key=total_costs.get) #type: ignore
        final_static_tariff = static_options.get(cheapest_static_key[1])
    else:
        st.warning(t("no_predefined_static_tariffs"))
    
    return final_flex_tariff, final_static_tariff

def _return_tariff_selection(_tariff_manager: TariffManager, expanded: bool = True) -> tuple[Tariff, Tariff]:
    """Renders tariff selection expanders in the UI for user customization."""
    logger.log("Rendering Tariff Selection")
    
    final_tariffs = []
    options = [
        (t("flexible_plan_title"), _tariff_manager.get_flex_tariffs_with_custom()),
        (t("static_plan_title"), _tariff_manager.get_static_tariffs_with_custom())
    ]

    for title, tariff_options in options:
        with st.expander(title, expanded=expanded):
            tariff_type = title.split(" ")[0]
            # Let user select a predefined tariff or 'Custom'
            selected_name = st.selectbox(t("select_tariff_type", tariff_type=tariff_type), options=list(tariff_options.keys()), index=len(tariff_options) - 1)
            selected_tariff = tariff_options[selected_name]
            
            # Display input fields pre-filled with the selected tariff's data
            price_kwh = st.number_input(t("on_top_price"), value=selected_tariff.price_kwh, min_value=0.0, step=0.001, format="%.4f", key=f"{tariff_type}_price")
            price_kwh_pct = 0.0
            if tariff_type == "Flexible":
                price_kwh_pct = st.number_input(t("variable_price_pct"), value=selected_tariff.price_kwh_pct, min_value=0.0, max_value=100.0, step=1.0, format="%.1f", key=f"{tariff_type}_pct")
            
            monthly_fee = st.number_input(t("monthly_fee"), value=selected_tariff.monthly_fee, min_value=0.0, step=1.0, format="%.2f", key=f"{tariff_type}_fee")
            
            # Create a new Tariff object with the potentially modified values
            final_tariffs.append(
                Tariff(name=selected_name, type=selected_tariff.type, price_kwh=price_kwh, monthly_fee=monthly_fee, price_kwh_pct=price_kwh_pct)
            )
    return tuple(final_tariffs)

def render_sidebar_inputs(df: pd.DataFrame, tariff_manager: TariffManager) -> tuple[str, date, date, Tariff, Tariff, float]:
    """Renders all sidebar inputs and returns the configuration values."""
    logger.log("Rendering Sidebar")
    with st.sidebar:
        st.header(t("configuration"))
        
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

        # Split into start and end dates
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            # Fallback for the rare case where only one date is returned
            start_date, end_date = selected_range[0], max_date
        
        # 3. Tariff Plan Selection
        with st.expander(t("select_tariff_plan"), expanded=True):
            st.text(t("tariff_plan_help"))
            
            compare_cheapest = st.checkbox(t("compare_cheapest_tariffs"), value=False, help=t("compare_cheapest_tariffs_help"))
            if compare_cheapest:
                final_flex_tariff, final_static_tariff = _compare_all_tariffs(df, tariff_manager, awattar_country)
                flex_info = t("cheapest_flex_tariff_info", tariff_name=final_flex_tariff.name) if final_flex_tariff else t("no_predefined_flex_tariffs")
                static_info = t("cheapest_static_tariff_info", tariff_name=final_static_tariff.name) if final_static_tariff else t("no_predefined_static_tariffs")
                st.info(f"{flex_info}\n\n{static_info}")

            else:
                final_flex_tariff, final_static_tariff = _return_tariff_selection(tariff_manager, expanded=False)
            
        # 4. Load Shifting Simulation
        with st.expander(t("simulate_consumption_shifting"), expanded=False):
            st.markdown(t("simulate_shifting_markdown"), help=t("simulate_shifting_help"))
            shift_percentage = st.slider(t("shift_peak_load_slider"), min_value=0, max_value=100, value=0, step=5)

        return awattar_country, start_date, end_date, final_flex_tariff, final_static_tariff, shift_percentage

# --- Main Page Components ---

@st.cache_data(ttl=3600)
def _compute_absence_data(df: pd.DataFrame, base_threshold: float, absence_threshold: float) -> list:
    """Computes and caches the days of absence based on low daily consumption."""
    logger.log("Computing Absence Data")
    if 'date' not in df.columns:
        df['date'] = df['timestamp'].dt.date
    daily_consumption = df.groupby('date')["consumption_kwh"].sum()
    intervals_per_day = get_intervals_per_day(df)
    # Identify days where total consumption is below a fraction of the typical daily base load
    absence_days = daily_consumption[daily_consumption < (base_threshold * intervals_per_day * absence_threshold)].index.tolist()
    return absence_days

def render_absence_days(df: pd.DataFrame, base_threshold: float) -> pd.DataFrame:
    """Adds a sidebar option to remove days with very low consumption."""
    with st.sidebar:
        absence_days = _compute_absence_data(df, base_threshold, ABSENCE_THRESHOLD)
        if absence_days:
            with st.expander(t("remove_absence_days"), expanded=False):
                st.text(t("remove_absence_days_help", count=len(absence_days)), help=t("remove_absence_days_long_help", threshold=ABSENCE_THRESHOLD))
                select_all = st.checkbox(t("exclude_all_days_checkbox"), value=False)
                default_selection = absence_days if select_all else []
                excluded_days = st.multiselect(t("multiselect_excluded_days"), options=absence_days, default=default_selection)
                
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
        st.success(t("flex_plan_recommended", savings=savings, additional_text=additional_text, link_text=link_text))
    else:
        link_text = t("check_cheapest_offer", link=static_tariff.link) if static_tariff.link else ""
        st.warning(t("static_plan_recommended", abs_savings=-savings, link_text=link_text))

# --- Tab: Spot Price Analysis ---

@st.cache_data(ttl=3600)
def _compute_price_distribution_data(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Computes and caches the quartile price data for the selected resolution."""
    logger.log("Computing Price Distribution Data")
    price_agg_dict = {"spot_price_eur_kwh": [("q1", lambda x: x.quantile(0.25)), ("median", "median"), ("mean", "mean"), ("q3", lambda x: x.quantile(0.75))]}
    
    config = get_aggregation_config(df, resolution)
    # Drop NA before aggregation to avoid issues with empty groups
    df_price = df.dropna(subset=["spot_price_eur_kwh"]).groupby(config["grouper"]).agg(price_agg_dict)
    df_price.columns = ["Spot Price Q1", "Spot Price Median", "Spot Price Mean", "Spot Price Q3"]
    df_price.index = df_price.index.map(config["x_axis_map"])
    df_price.index.name = config["name"]
    # Reindex to ensure correct chronological order (e.g., Jan, Feb, Mar...)
    # Then, drop any rows that are all NA, which happens for months with no data.
    df_price = df_price.reindex(config["x_axis_map"].values()).dropna(how="all")
    return df_price

@st.cache_data(ttl=3600)
def _compute_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the data needed for the price heatmap."""
    logger.log("Computing Heatmap Data")
    df_pvt = df.pivot_table(values="spot_price_eur_kwh", index=df["timestamp"].dt.month, columns=df["timestamp"].dt.hour, aggfunc="mean")
    
    # Create a copy and convert integer column names to strings for plotly compatibility.
    df_pvt = df_pvt.copy()
    df_pvt.columns = df_pvt.columns.map(str)
    
    return df_pvt

def render_price_analysis_tab(df: pd.DataFrame, static_tariff: Tariff):
    """Renders the interactive analysis of electricity spot prices."""
    logger.log("Rendering Price Analysis Tab")
    
    resolution = st.radio(t("price_analysis_resolution_label"), ("Monthly", "Weekly", "Hourly"), horizontal=True, key="price_res")
    
    # Quartile Price Chart
    st.subheader(t("price_over_time_header"))
    st.markdown(t("price_over_time_markdown"))
    df_price = _compute_price_distribution_data(df, resolution)
    price_fig = charts.get_price_chart(df_price, pd.Series([static_tariff.price_kwh]*len(df.index)))
    st.plotly_chart(price_fig, use_container_width=True)

    # Heatmap Analysis
    st.subheader(t("heatmap_header"))
    st.markdown(t("heatmap_markdown"))
    heatmap_data = _compute_heatmap_data(df)
    heatmap_fig = charts.get_heatmap(heatmap_data)
    st.plotly_chart(heatmap_fig, use_container_width=True)

# --- Tab: Cost Comparison ---

@st.cache_data(ttl=3600)
def _compute_cost_comparison_data(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Computes and caches aggregated cost data for comparison."""
    logger.log("Computing Cost Comparison Data")
    freq_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "ME"}
    grouper = pd.Grouper(key="timestamp", freq=freq_map[resolution])

    summary_agg_dict: dict = {
        "Total Consumption": ("consumption_kwh", "sum"),
        "Total Flexible Cost": ("total_cost_flexible", "sum"),
        "Total Static Cost": ("total_cost_static", "sum")
    }
    df_summary = df.groupby(grouper).agg(**summary_agg_dict).reset_index()
    df_summary = df_summary[df_summary["Total Consumption"] > 0.01] # Filter out empty periods
    
    df_summary["Difference (€)"] = df_summary["Total Static Cost"] - df_summary["Total Flexible Cost"]
    df_summary["Period"] = df_summary["timestamp"].dt.strftime("%Y-%m-%d" if resolution == "Daily" else "%Y-%U" if resolution == "Weekly" else "%Y-%m")
    return df_summary

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


def render_cost_comparison_tab(df: pd.DataFrame):
    """Renders the content for the 'Cost Comparison' tab."""
    
    # Lambda function to format the difference for all tables
    difference_formatter = lambda v: f"color: {GREEN}" if v > 0 else f"color: {RED}"
    granular_col_format = {"Total Consumption": "{:.2f} kWh", "Total Flexible Cost": "€{:.2f}", "Total Static Cost": "€{:.2f}", "Difference (€)": "€{:.2f}"}
    regular_col_format = {"Total Consumption": "{:,.2f} kWh", "Total Static Cost": "€{:,.2f}"}

    logger.log("Rendering Cost Comparison Tab")

    is_granular_data = calculate_granular_data(df)
    if not is_granular_data:
        st.info(t("granular_data_info"))

    resolution = st.radio(t("cost_comparison_resolution_label"), ("Monthly","Weekly", "Daily"), horizontal=True, key="summary_res")
    
    df_summary = _compute_cost_comparison_data(df, resolution)
    if df_summary.empty:
        st.warning(t("no_data_for_period"))
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(t("total_costs_per_period_header"))
        st.markdown(t("total_costs_per_period_markdown"))
        y_cols_total = ["Total Flexible Cost", "Total Static Cost"] if is_granular_data else ["Total Static Cost"]
        colors_total = [FLEX_COLOR, STATIC_COLOR] if is_granular_data else [STATIC_COLOR]
        st.line_chart(df_summary.set_index("Period"), y=y_cols_total, y_label="Total Cost (€)", color=colors_total)

    with col2:
        st.subheader(t("avg_price_per_kwh_header"))
        st.markdown(t("avg_price_per_kwh_markdown"))
        df_summary["Avg Static Price"] = df_summary["Total Static Cost"] / df_summary["Total Consumption"]
        
        if is_granular_data:
            df_summary["Avg. Flexible Price"] = df_summary["Total Flexible Cost"] / df_summary["Total Consumption"]
        
        y_cols_avg = ["Avg Static Price", "Avg. Flexible Price"] if is_granular_data else ["Avg Static Price"]
        colors_avg = [STATIC_COLOR, FLEX_COLOR] if is_granular_data else [STATIC_COLOR]
        st.line_chart(df_summary.set_index("Period"), y=y_cols_avg, y_label="Average Price (€/kWh)", color=colors_avg)
            
    st.subheader(t("detailed_comparison_table_header"))
    st.text(t("detailed_comparison_table_markdown"), help=t("detailed_comparison_table_help"))
    
    # Main DataFrame
    if is_granular_data:
        cols_to_show = ["Period", "Total Consumption", "Total Flexible Cost", "Total Static Cost", "Difference (€)"]
        styler = df_summary[cols_to_show].style
        styler = styler.map(difference_formatter, subset=["Difference (€)"]) #type: ignore
        styler = styler.format(granular_col_format)
    else:
        cols_to_show = ["Period", "Total Consumption", "Total Static Cost"]
        styler = df_summary[cols_to_show].style
        styler = styler.format(regular_col_format)
        
    st.dataframe(styler, hide_index=True, use_container_width=True)

    # Calculate totals
    df_totals = _compute_col_vals(df_summary, is_granular_data, sum, "Total")
    df_means = _compute_col_vals(df_summary, is_granular_data, lambda x: x.mean(), "Average")
    df_display = pd.concat([df_totals, df_means], ignore_index=True)

    # Style the totals DataFrame
    if is_granular_data:
        totals_styler = df_display[cols_to_show].style
        totals_styler = totals_styler.map(difference_formatter, subset=["Difference (€)"]) #type: ignore
        totals_styler = totals_styler.format(granular_col_format)
    else:
        totals_styler = df_display[cols_to_show].style
        totals_styler = totals_styler.format(regular_col_format)
    
    # Convert to HTML and render without header
    totals_html = totals_styler.hide(axis="index")
    st.text(t("total_and_average_values_text"))
    st.dataframe(totals_html, hide_index=True, use_container_width=True)

# --- Tab: Usage Patterns ---

@st.cache_data(ttl=3600)
def _compute_usage_profile_data(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the proportion and average cost for each usage profile."""
    logger.log("Computing Usage Profile Data")
    
    load_types = ["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]
    profile_data = []
    total_kwh = df["consumption_kwh"].sum()
    intervals_per_day = get_intervals_per_day(df)
    
    for load in load_types:
        kwh = df[load].sum()
        if kwh > 0.01:
            avg_price = (df[load] * df["spot_price_eur_kwh"]).sum() / kwh
            proportion = kwh / total_kwh
            mean_daily_kwh = df[load].mean() * intervals_per_day
            profile_data.append({
                "Profile": load.replace("_kwh", "").replace("_", " ").title(),
                "kwh": kwh,
                "kwh_mean": mean_daily_kwh,
                "avg_price": avg_price,
                "proportion": proportion
            })
    return pd.DataFrame(profile_data)


@st.cache_data(ttl=3600)
def _compute_consumption_quartiles(df: pd.DataFrame, intervals_per_day: int) -> pd.DataFrame:
    """Computes and caches the usage data for the selected resolution."""
    
    logger.log("Computing Consumption Data")
    
    # Define the aggregation logic once to be reused.
    consumption_agg_dict = {
        "consumption_kwh": [
            ("q1", lambda x: x.quantile(0.25)),
            ("median", lambda x: x.quantile(0.50)),
            ("q3", lambda x: x.quantile(0.75))
        ]
    }
    
    if intervals_per_day == 24:
        resolution = "Monthly"
    else:
        resolution = "Hourly"
    
    config = get_aggregation_config(df, resolution)
    # Drop NA before aggregation to avoid issues with empty groups
    df_consumption_quartiles = df.dropna(subset=["consumption_kwh"]).groupby(config["grouper"]).agg(consumption_agg_dict)
    df_consumption_quartiles.columns = ["Consumption Q1", "Consumption Median", "Consumption Q3"]
    df_consumption_quartiles.index.name = config["name"]        
    df_consumption_quartiles.index = df_consumption_quartiles.index.map(config["x_axis_map"])
    # Reindex to ensure correct chronological order (e.g., Jan, Feb, Mar...)
    # Then, drop any rows that are all NA, which happens for months with no data.
    df_consumption_quartiles = df_consumption_quartiles.reindex(config["x_axis_map"].values()).dropna(how="all")
    
    return df_consumption_quartiles

@st.cache_data(ttl=3600)
def _compute_example_day(df: pd.DataFrame, random_day, group: bool = False) -> pd.DataFrame:
    """Selects a random day and return the data for plotting as well as the DataFrame itself."""
    logger.log("Computing Example Day")

    df_hour = df[df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.date == random_day].copy()
    
    if not df_hour.empty:
        df_hour["hour"] = df_hour["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.hour

        # Group by hour and sum the kWh columns
        if group:
            df_hour = (
                df_hour.groupby("hour")[["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]]
                .sum())
        else:
            df_hour = df_hour.set_index("timestamp")[["base_load_kwh", "regular_load_kwh", "peak_load_kwh"]]
                
        df_hour = df_hour.rename(columns={
                "base_load_kwh": "Base Load",
                "regular_load_kwh": "Regular Load",
                "peak_load_kwh": "Peak Load"
            })
    
        return df_hour
    else:
        return pd.DataFrame()
    
st.cache_resource(ttl=3600, show_spinner=True)
def _fit_forecast_model(df: pd.DataFrame) -> tuple[Prophet|None, pd.DataFrame]:
    """Fits the Prophet model on provided data. Prophet doc: https://facebook.github.io/prophet/."""
    
    if "timestamp" not in df.columns:
        return None, pd.DataFrame()
    
    # Resample to daily data and format for Prophet ("ds", "y")
    df_daily = df.resample("D", on="timestamp")["consumption_kwh"].sum().reset_index().copy()
    df_daily["timestamp"] = df_daily["timestamp"].dt.tz_localize(None)
    df_daily = df_daily.rename(columns={"timestamp": "ds", "consumption_kwh": "y"})
    df_daily.loc[df_daily["y"] == 0, "y"] = pd.NA

    # Prophet requires a minimum of 2 data points, but more is needed for seasonality.
    if len(df_daily) < 30:
        return None, pd.DataFrame()

    # Outliers
    absence_days = pd.DataFrame({
        "holiday": "absence",
        "ds": df_daily[df_daily["y"] == 0]["ds"],
        "lower_window": 0, # No days to extend
        "upper_window": 0,
        }
    )
    
    holidays = pd.concat([absence_days], ignore_index=True)
    
    # Configure and train the Prophet model.
    # Check for at least a year for yearly seasonality, otherwise disable it.
    use_yearly_seasonality = len(df_daily) >= 365
    
    model = Prophet(holidays=holidays, yearly_seasonality=use_yearly_seasonality)
    model.add_country_holidays(country_name="AT")
    
    if not use_yearly_seasonality:
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    
    #model.add_regressor("spot_price_eur_kwh")
    model.fit(df_daily)
    
    return model, df_daily

@st.cache_data(ttl=3600)
def _compute_percent_change(df_daily: pd.DataFrame, forecast: pd.DataFrame) -> float:
    """Computes the percentage change of a DataFrame with daily values by the incline of a linear regression model. """
    logger.log("Computing Percent Change")
    # Analyze the underlying trend from the model"s components. Isolate the trend component within the historical data range
    historical_trend = forecast[forecast['ds'].isin(df_daily['ds'])]['trend']
    
    # Prepare data for regression: x is time, y is the trend value
    x_values = np.arange(len(historical_trend))
    y_values = historical_trend.values
    
    # Fit a 1st-degree polynomial (linear regression) and get the slope which represents the average daily change in the trend.
    slope, _ = np.polyfit(x_values, y_values, 1)
    
    # Calculate the total change over the period based on this average slope
    total_change_over_period = slope * len(historical_trend)

    # Calculate percentage change relative to the average consumption for a stable metric
    avg_consumption = df_daily['y'].mean()
    if avg_consumption > 0:
        percent_change = (total_change_over_period / avg_consumption) * 100
    else:
        percent_change = 0
    
    return percent_change


@st.cache_data(ttl=3600)
def _compute_consumption_trend_and_forecast(df: pd.DataFrame, forcast_periods: int = 90):
    """ Analyzes and forecasts daily consumption using Prophet to account for seasonality.  """
    logger.log("Computing Consumption Trend and Forecast with Prophet")
    
    model, df_daily = _fit_forecast_model(df)
    if model is None:
        return None

    # Create a future dataframe and make a forecast
    future = model.make_future_dataframe(periods=forcast_periods, freq="D")
    forecast = model.predict(future)

    # Clean the forecast data by ensuring that consumption predictions do not fall below zero.
    for col in ["yhat", "yhat_lower", "yhat_upper", "trend"]:
        if col in forecast.columns:
            forecast[col] = forecast[col].clip(0)
            
    percent_change = _compute_percent_change(df_daily, forecast)

    # Classify the trend
    if abs(percent_change) < THRESHOLD_STABLE_TREND:
        trend_description = "Stable"
    elif percent_change > 0:
        trend_description = "Increasing"
    else:
        trend_description = "Decreasing"
        
    return df_daily, forecast, trend_description, percent_change


def render_usage_pattern_tab(df: pd.DataFrame, base_threshold: float, peak_threshold: float):
    """Renders the content for the 'Usage Patterns' tab."""
    
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
    df_consumption_day = _compute_consumption_quartiles(df_filtered, intervals)
    
    st.subheader(t("consumption_over_time_header"))
    
    # Daily Consumption with Quartiles
    st.markdown(f"#### {t('daily_consumption_header')}\n{t('daily_consumption_markdown')}")

    if not df_consumption_day.empty:
        consumption_fig = charts.get_consumption_chart(df_consumption_day, intervals)
        st.plotly_chart(consumption_fig, use_container_width=True)
        
    # Trend Visualization and Forecast
    try:
        if day_filter != "All Days":
            # Forecasting is complex with filtered days, so we skip it.
            # A log message is already present from the original code.
            pass
        else:
            st.subheader(t("consumption_trend_forecast_header"))
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
        trend_data = _compute_consumption_trend_and_forecast(df_filtered, forecast_days)

        # Display Results
        if trend_data:
            df_daily_trend, df_forecast, trend_description, trend_metric = trend_data
            model, _ = _fit_forecast_model(df_filtered) # We need the model for the components

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
            st.plotly_chart(trend_fig, use_container_width=True)

            # Expander for seasonality components
            with st.expander(t("show_seasonality_details")):
                st.markdown(t("seasonality_details_markdown"))
                if model:
                    seasonality_charts = charts.get_seasonality_charts(model, df_forecast)
                    st.plotly_chart(seasonality_charts, use_container_width=True, key=f"forecast_details")

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
    profile_data = _compute_usage_profile_data(df_filtered)

    # Display Thresholds
    col1.metric(t("base_load_threshold_metric"), f"{base_threshold:.3f} kWh", help=t("base_load_threshold_help"))
    
    # The peak_threshold passed is the influenceable part. Add base for the absolute value.
    absolute_peak_threshold = base_threshold + peak_threshold
    col2.metric(t("peak_sustain_threshold_metric"), f"{absolute_peak_threshold:.3f} kWh", help=t("peak_sustain_threshold_help"))

    if not profile_data.empty:
        marimekko_fig = charts.get_marimekko_chart(profile_data)      
        st.plotly_chart(marimekko_fig, use_container_width=True)

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
        df_day = _compute_example_day(df_filtered, st.session_state.random_day, group=False)
        # Ensure correct stacking order for the bar chart: Base (bottom), Regular, Peak (top).
        df_day = df_day[["Base Load", "Regular Load", "Peak Load"]]
        day_str = st.session_state.random_day.strftime('%A, %Y-%m-%d')
        st.caption(t("example_day_caption", day=day_str, total_kwh=df_day.sum().sum()))
        example_day_fig = charts.plot_example_day(df_day, intervals, BASE_COLOR, REGULAR_COLOR, PEAK_COLOR)
        st.plotly_chart(example_day_fig, use_container_width=True)
        #st.bar_chart(df_day, color=[BASE_COLOR, REGULAR_COLOR, PEAK_COLOR], x_label="Hour of Day", y_label="Consumption (kWh)")

# --- Tab: Yearly Summary ---

@st.cache_data(ttl=3600)
def _compute_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and caches the yearly summary of the data."""
    logger.log("Computing Yearly Summary")
    df["Year"] = df["timestamp"].dt.year
    yearly_summary_agg_dict = {
        "Total Consumption": ("consumption_kwh", "sum"),
        "Total Static Cost": ("total_cost_static", "sum")
    }
    
    is_granular_data = calculate_granular_data(df)
    if is_granular_data:
        yearly_summary_agg_dict["Total Flexible Cost"] = ("total_cost_flexible", "sum")
        
    yearly_agg = df.groupby("Year").agg(**yearly_summary_agg_dict).reset_index() # type: ignore
    
    if not yearly_agg.empty and yearly_agg["Total Consumption"].sum() > 0:
        yearly_agg["Avg Static Price"] = yearly_agg["Total Static Cost"] / yearly_agg["Total Consumption"]
        if is_granular_data:
            yearly_agg["Avg. Flex Price"] = yearly_agg["Total Flexible Cost"] / yearly_agg["Total Consumption"]

    return yearly_agg

def render_yearly_summary_tab(df: pd.DataFrame):
    """Renders the content for the 'Yearly Summary' tab."""
    st.subheader(t("yearly_summary_header"))
    yearly_agg = _compute_yearly_summary(df)
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
    st.dataframe(display_df.style.format(style_format), hide_index=True, use_container_width=True)

# --- Tab: Download Data ---

@st.cache_data(ttl=3600)
def _compute_download_data(df: pd.DataFrame) -> tuple[bytes, bytes]:
    """Prepares and caches the Excel file bytes for download."""
    logger.log("Computing Download Data")
    
    # Prepare data for spot price-only download (hourly resolution)
    excel_spot_data_df = df.set_index("timestamp").resample("h").first().reset_index()[["timestamp", "spot_price_eur_kwh"]].dropna()
    excel_spot_bytes = to_excel(excel_spot_data_df)
    
    # Prepare full analysis data for download
    excel_full_bytes = to_excel(df.drop(columns=["date", "days_in_month"]))

    return excel_full_bytes, excel_spot_bytes

def render_download_tab(df: pd.DataFrame, start_date: date, end_date: date):
    """Renders the content for the Download tab."""
    excel_full_data, excel_spot_data = _compute_download_data(df)
    
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.markdown("Download the full, detailed analysis including consumption classification and cost calculations for both tariff types.")
        st.download_button(
            label="Download Full Analysis (XLSX)",
            data=excel_full_data,
            file_name=f"electricity_analysis_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        st.markdown("Download only the hourly EPEX spot prices for the selected period.")
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

    with st.expander(t("faq_how_to_use_q")):
        st.markdown(t("faq_how_to_use_a"))

    with st.expander(t("faq_data_file_q")):
        st.markdown(t("faq_data_file_a"))

    with st.expander(t("faq_load_types_q")):
        st.markdown(t("faq_load_types_a"))
    
    with st.expander(t("faq_peak_shifting_q")):
        st.markdown(t("faq_peak_shifting_a"))
        
    st.info(t("faq_footer_info"))

# --- Footer ---
@st.cache_data
def render_footer():
    """Renders the footer with information about the project and further links."""
    st.markdown("\n\n---")
    st.markdown(t("footer_text"))