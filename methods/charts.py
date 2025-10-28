import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from methods.config import FLEX_COLOR, MEKKO_BORDER, FLEX_COLOR_LIGHT, STATIC_COLOR, FLEX_COLOR_SHADE, FORECAST_PREDICTED_COLOR, FORECAST_ACTUAL_COLOR, FORECAST_UNCERTAINTY_COLOR, PERSONAL_DATA_COLOR, PERSONAL_DATA_COLOR_SHADE, PERSONAL_DATA_COLOR_LIGHT, BASE_COLOR, REGULAR_COLOR, PEAK_COLOR
import calendar
from prophet import Prophet
from methods.i18n import t

def _get_interval_text(intervals_per_day: int, t) -> str:

    # Determine y-axis label based on data granularity
    if intervals_per_day == 24:
        interval_text = t("interval_per_hour")
    elif intervals_per_day > 24 and (1440 % intervals_per_day == 0):
        minutes = 1440 // intervals_per_day
        interval_text = t("interval_per_minute", minutes=minutes)
    else:
        interval_text = "" # Fallback for daily data or other resolutions
        
    return interval_text

def get_price_chart(df: pd.DataFrame, static_price: float) -> go.Figure:
    fig = go.Figure()

    # First trace: Q1 (lower bound of the fill)
    fig.add_trace(go.Scatter(x=df.index, y=df["Spot Price Q1"], mode="lines", line=dict(width=0), name="Q1–Q3 Range", showlegend=False))

    # Second trace: Q3 (upper bound), filled to previous (Q1)
    fig.add_trace(go.Scatter( x=df.index, y=df["Spot Price Q3"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=FLEX_COLOR_SHADE, name="Q1–Q3 Range", showlegend=False))

    # Q3 and Q1 Dotted Lines with Mean in between
    fig.add_trace(go.Scatter(x=df.index, y=df["Spot Price Q3"], mode="lines", line=dict(dash="dot", color=FLEX_COLOR_LIGHT), name="3rd Quartile (Q3)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Spot Price Median"], mode="lines", line=dict(color=FLEX_COLOR, width=3), name="Median Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Spot Price Q1"], mode="lines", line=dict(dash="dot", color=FLEX_COLOR_LIGHT), name="1st Quartile (Q1)"))

    # Static Price
    fig.add_hline(y=static_price, line=dict(color=STATIC_COLOR, width=2), name="Static Tariff")

    fig.update_layout(xaxis_title=df.index.name, yaxis_title=t("spot_price_kwh_y_axis"), legend_title_text=t("legend_metrics"), hovermode="x unified")

    return fig


def get_heatmap(df: pd.DataFrame) -> go.Figure:

    fig = px.imshow(
        df,
        labels=dict(x="Hour of Day", y="Month", color="Avg Spot Price (€/kWh)"),
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    
    # Assume df.index contains month numbers in appearance order
    month_numbers = df.index.tolist()
    month_names = [calendar.month_name[m] for m in month_numbers]

    fig.update_yaxes(
        tickvals=month_numbers,
        ticktext=month_names
    )

    return fig

def get_consumption_price_heatmap(df: pd.DataFrame) -> go.Figure:
    """Creates a 2D histogram showing consumption volume at different hours and price levels."""
    fig = px.density_heatmap(
        df,
        x="hour",
        y="price_bin",
        z="consumption_kwh",
        histfunc="sum",
        labels={
            "hour": t("consumption_price_heatmap_hour_label"),
            "price_bin": t("consumption_price_heatmap_price_bin_label"),
            "color": "Total Consumption (kWh)"
        },
        color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_title=t("consumption_price_heatmap_hour_label"), yaxis_title=t("spot_price_kwh_y_axis"))
    # Make y-axis labels more readable
    fig.update_yaxes(categoryorder='category ascending', tickformat=".3f")
    return fig

def get_consumption_chart(df: pd.DataFrame, intervals_per_day: int, df_median_spot_price: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    df_plot = df.copy()
    idx = df_plot.index

    # If data is more granular than hourly (e.g., 15-min), scale it to an hourly equivalent for plotting.
    # This makes the y-axis more intuitive ("per hour").
    if intervals_per_day > 24:
        scaling_factor = intervals_per_day / 24
        for col in df_plot.columns:
            if "Consumption" in col:
                df_plot[col] *= scaling_factor
        intervals_per_day = 24 # Treat as hourly for label generation

    # First trace: Q1 (lower bound of the fill)
    fig.add_trace(go.Scatter(x=idx, y=df_plot["Consumption Q1"], mode="lines", line=dict(width=0), name="Q1–Q3 Consumption Range", showlegend=False))

    # Second trace: Q3 (upper bound), filled to previous (Q1)
    fig.add_trace(go.Scatter( x=idx, y=df_plot["Consumption Q3"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=PERSONAL_DATA_COLOR_SHADE, name="Q1–Q3 Consumption Range", showlegend=False))

    # Add the visible lines on top of the fill area.
    fig.add_trace(go.Scatter(x=idx, y=df_plot["Consumption Q3"], mode="lines", line=dict(dash="dot", color=PERSONAL_DATA_COLOR_LIGHT), name="3rd Quartile Consumption (Q3)"))
    fig.add_trace(go.Scatter(x=idx, y=df_plot["Consumption Median"], mode="lines", line=dict(color=PERSONAL_DATA_COLOR, width=3), name="Median Consumption"))
    fig.add_trace(go.Scatter(x=idx, y=df_plot["Consumption Q1"], mode="lines", line=dict(dash="dot", color=PERSONAL_DATA_COLOR_LIGHT), name="1st Quartile Consumption (Q1)"))

    # Add Median Spot Price on a secondary y-axis
    fig.add_trace(go.Scatter(
        x=df_median_spot_price.index,
        y=df_median_spot_price["Spot Price Median"],
        name="Median Spot Price",
        mode="lines",
        line=dict(color=FLEX_COLOR, width=2),
        yaxis="y2"
    ))

    yaxis_title = t("consumption_kwh_y_axis", interval_text=_get_interval_text(intervals_per_day, t)).strip()

    fig.update_layout(xaxis_title=idx.name, yaxis_title=yaxis_title, legend_title_text=t("legend_metrics"), hovermode="x unified",
                      yaxis2=dict(title=t("spot_price_kwh_y_axis"), overlaying="y", side="right", showgrid=False, zeroline=False))

    return fig

def get_marimekko_chart(df: pd.DataFrame, border_color: str = "#FFFFFF") -> go.Figure:
 
    bars = []
    cumulative_width = 0
    annotations = []

    for _, row in df.iterrows():
        width = row["proportion"]
        price = row["avg_price"]
        label = row["Profile"]
        
        # Background bar (border)
        bars.append(
            go.Bar(
        x=[cumulative_width-MEKKO_BORDER],
        y=[price+MEKKO_BORDER/2],
        width=[width],
        marker=dict(color=border_color),
        offset=-MEKKO_BORDER/2,
        hoverinfo="skip",
        showlegend=False
            )
        )
        
        # Foreground bar (actual data)
        bars.append(
            go.Bar(
        x=[cumulative_width - MEKKO_BORDER/2],
        y=[price],
        width=[width - 2*MEKKO_BORDER],
        name=label,
        marker=dict(color=PERSONAL_DATA_COLOR),
        offset=0,
        hovertemplate=(
            f"<b>{label}</b><br>"
            f"Proportion: {width:.1%}<br>"
            f"Avg Price: €{price:.3f}/kWh<br>"
            "<extra></extra>"
        ),
        showlegend=False
            )
        )

        annotations.append(dict(
            x=cumulative_width + width / 2,
            y=price / 2,
            text=f"{label}<br>{width:.1%}<br>€{price:.3f}/kWh",
            showarrow=False,
            font=dict(color="white", size=11)
        ))

        cumulative_width += width

        # Create the figure
        fig = go.Figure(data=bars)
        fig.update_layout(
        height=400,
        barmode="overlay",  # important: overlay to simulate border effect
        xaxis=dict(
            range=[0, 1],
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            title=t("mekko_x_axis"),
            showgrid=False
        ),
        yaxis=dict(
            title=t("mekko_y_axis"),
            gridcolor="rgba(0,0,0,0.1)"
        ),
        annotations=annotations,
        margin=dict(l=40, r=10, t=30, b=40)
            )

    return fig


def get_trend_chart(df_history: pd.DataFrame, df_forecast: pd.DataFrame) -> go.Figure:
    """Creates a chart to visualize the Prophet forecast, including historical data, the forecast line, and the uncertainty interval. """
    fig = go.Figure()

    # Add the uncertainty interval (shaded area)
    fig.add_trace(go.Scatter(
        x=df_forecast["ds"],
        y=df_forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        fillcolor=FORECAST_UNCERTAINTY_COLOR,
        name="Uncertainty Interval Low",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_forecast["ds"],
        y=df_forecast["yhat_lower"],
        mode="lines",
        line=dict(width=0),
        fillcolor=FORECAST_UNCERTAINTY_COLOR,
        fill="tonexty",
        name="Uncertainty Interval",
    ))
    
    # Add the historical actual consumption
    fig.add_trace(go.Scatter(
        x=df_history["ds"], 
        y=df_history["y"],
        mode="lines",
        line=dict(width=2, color=FORECAST_ACTUAL_COLOR),
        name="Daily Consumption (Actual)"
    ))

    # Add the main forecast line
    fig.add_trace(go.Scatter(
        x=df_forecast["ds"],
        y=df_forecast["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color=FORECAST_PREDICTED_COLOR, width=3)
    ))


    fig.update_layout(
        xaxis_title=t("trend_x_axis"),
        yaxis_title=t("trend_y_axis"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True
    )
    
    return fig

def get_seasonality_charts(model: Prophet, forecast: pd.DataFrame) -> go.Figure:
    """Generates individual Plotly charts for each seasonality component in the Prophet model."""
    from prophet.plot import plot_components_plotly
    fig = plot_components_plotly(model, forecast, uncertainty=False)
    
    for trace in fig.data:
        
        # Update the line color for the main line trace
        if trace.mode == 'lines':
            trace.line.color = PERSONAL_DATA_COLOR_LIGHT
        
        # If there are uncertainty bands, you might want to adjust their color too
        # Prophet's plot_components_plotly uses fill for uncertainty
        elif trace.fill == 'tonexty': # Uncertainty bands usually have fill='tonexty'
            trace.fillcolor = FORECAST_UNCERTAINTY_COLOR + '40' # Adding transparency
                
    return fig

def get_example_day_chart(df_day: pd.DataFrame, intervals_per_day: int) -> go.Figure:
    fig = go.Figure()
    hours = df_day.index.astype(str)  # e.g., "0", "1", ..., "23"

    fig.add_trace(go.Bar(
        x=hours,
        y=df_day["Base Load"],
        name="Base Load",
        marker_color=BASE_COLOR
    ))
    fig.add_trace(go.Bar(
        x=hours,
        y=df_day["Regular Load"],
        name="Regular Load",
        marker_color=REGULAR_COLOR
    ))
    fig.add_trace(go.Bar(
        x=hours,
        y=df_day["Peak Load"],
        name="Peak Load",
        marker_color=PEAK_COLOR
    ))

    fig.update_layout(
        barmode="stack",
        xaxis_title=t("example_day_x_axis"),
        yaxis_title=t("consumption_kwh_y_axis", interval_text=_get_interval_text(intervals_per_day, t)).strip(),
        legend_title=t("example_day_load_type"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=400
    )
    return fig

def get_avg_price_chart(df_summary: pd.DataFrame, is_granular: bool) -> go.Figure:
    """Creates a line chart for comparing average prices per kWh."""
    fig = go.Figure()

    if is_granular:
        fig.add_trace(go.Scatter(x=df_summary["Period"], y=df_summary["Avg. Flexible Price"],
                                 mode='lines', name="Avg. Flexible Price",
                                 line=dict(color=FLEX_COLOR)))

    fig.add_trace(go.Scatter(x=df_summary["Period"], y=df_summary["Avg Static Price"],
                             mode='lines', name="Avg Static Price",
                             line=dict(color=STATIC_COLOR)))

    fig.update_layout(
        yaxis_title=t("avg_price_per_kwh_header").split(" (")[0], # Get base label
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1)
    )
    return fig

def get_total_cost_chart(df_summary: pd.DataFrame, is_granular: bool) -> go.Figure:
    """Creates a line chart for comparing total costs."""
    fig = go.Figure()
    df_plot = df_summary.set_index("Period")

    if is_granular:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Total Flexible Cost"],
                                 mode='lines', name="Total Flexible Cost",
                                 line=dict(color=FLEX_COLOR)))

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Total Static Cost"],
                             mode='lines', name="Total Static Cost",
                             line=dict(color=STATIC_COLOR)))

    fig.update_layout(
        xaxis_title="Period",
        yaxis_title="Total Cost (€)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def get_cumulative_savings_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a line chart showing the cumulative savings over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["cumulative_savings"],
        mode='lines',
        name=t("cumulative_savings_trace_name"),
        line=dict(color=FLEX_COLOR)
    ))
    fig.update_layout(xaxis_title=t("trend_x_axis"), yaxis_title=t("cumulative_savings_y_axis"),
                      hovermode="x unified")
    return fig