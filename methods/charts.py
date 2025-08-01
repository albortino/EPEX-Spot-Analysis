import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from methods.config import FLEX_COLOR, MEKKO_BORDER, FLEX_COLOR_LIGHT, STATIC_COLOR, FLEX_COLOR_SHADE, FORECAST_PREDICTED_COLOR, FORECAST_ACTUAL_COLOR, FORECAST_UNCERTAINTY_COLOR
import calendar
from prophet import Prophet

def get_price_chart(df: pd.DataFrame, static_price: pd.Series) -> go.Figure:
    fig = go.Figure()

    # First trace: Q1 (lower bound of the fill)
    fig.add_trace(go.Scatter(x=df.index, y=df["Spot Price Q1"], mode="lines", line=dict(width=0), name="Q1–Q3 Range", showlegend=False))

    # Second trace: Q3 (upper bound), filled to previous (Q1)
    fig.add_trace(go.Scatter( x=df.index, y=df["Spot Price Q3"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=FLEX_COLOR_SHADE, name="Q1–Q3 Range", showlegend=False))

    # Q3 and Q1 Dotted Lines with Median in between
    fig.add_trace(go.Scatter(x=df.index, y=df["Spot Price Q3"], mode="lines", line=dict(dash="dot", color=FLEX_COLOR_LIGHT), name="3rd Quartile (Q3)"))
    fig.add_trace(go.Scatter( x=df.index, y=df["Spot Price Median"], mode="lines", line=dict(color=FLEX_COLOR, width=3), name="Median Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Spot Price Q1"], mode="lines", line=dict(dash="dot", color=FLEX_COLOR_LIGHT), name="1st Quartile (Q1)"))

    # Static Price
    fig.add_trace(go.Scatter(x=df.index, y=static_price, mode="lines", line=dict(color=STATIC_COLOR, width=2), name="Static Tariff"))

    fig.update_layout(xaxis_title=df.index.name, yaxis_title="Spot Price (€/kWh)", legend_title_text="Metrics", hovermode="x unified")

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

def get_consumption_chart(df: pd.DataFrame, intervals_per_day: int) -> go.Figure:
    fig = go.Figure()
    idx = df.index

    # First trace: Q1 (lower bound of the fill)
    fig.add_trace(go.Scatter(x=idx, y=df["Consumption Q1"], mode="lines", line=dict(width=0), name="Q1–Q3 Range", showlegend=False))

    # Second trace: Q3 (upper bound), filled to previous (Q1)
    fig.add_trace(go.Scatter( x=idx, y=df["Consumption Q3"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=FLEX_COLOR_SHADE, name="Q1–Q3 Range", showlegend=False))

    # Add the visible lines on top of the fill area.
    fig.add_trace(go.Scatter(x=idx, y=df["Consumption Q3"], mode="lines", line=dict(dash="dot", color=FLEX_COLOR), name="3rd Quartile (Q3)"))
    fig.add_trace(go.Scatter(x=idx, y=df["Consumption Median"], mode="lines", line=dict(color=FLEX_COLOR, width=3), name="Median Price"))
    fig.add_trace(go.Scatter(x=idx, y=df["Consumption Q1"], mode="lines", line=dict(dash="dot", color=FLEX_COLOR), name="1st Quartile (Q1)"))

    # Determine y-axis label based on data granularity
    if intervals_per_day == 24:
        interval_text = "per hour"
    elif intervals_per_day > 24 and (1440 % intervals_per_day == 0):
        minutes = 1440 // intervals_per_day
        interval_text = f"per {minutes} min"
    else:
        interval_text = "" # Fallback for daily data or other resolutions
    yaxis_title = f"Consumption (kWh {interval_text})".strip()

    fig.update_layout(xaxis_title=idx.name, yaxis_title=yaxis_title, legend_title_text="Metrics", hovermode="x unified")

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
        marker=dict(color=FLEX_COLOR),
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
            title="Proportion of Total Consumption",
            showgrid=False
        ),
        yaxis=dict(
            title="Average Spot Price (€/kWh)",
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
        mode="markers",
        marker=dict(size=4, color=FORECAST_ACTUAL_COLOR),
        name="Daily Consumption (Actual)"
    ))
    
    # Add the main forecast line
    fig.add_trace(go.Scatter(
        x=df_forecast["ds"],
        y=df_forecast["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color=FLEX_COLOR, width=3)
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Consumption (kWh)",
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
            trace.line.color = FLEX_COLOR_LIGHT
        
        # If there are uncertainty bands, you might want to adjust their color too
        # Prophet's plot_components_plotly uses fill for uncertainty
        elif trace.fill == 'tonexty': # Uncertainty bands usually have fill='tonexty'
            trace.fillcolor = FORECAST_UNCERTAINTY_COLOR + '40' # Adding transparency
                
    return fig

def plot_example_day(df_day: pd.DataFrame, base_color: str, regular_color: str, peak_color: str) -> go.Figure:
    fig = go.Figure()
    hours = df_day.index.astype(str)  # e.g., "0", "1", ..., "23"

    fig.add_trace(go.Bar(
        x=hours,
        y=df_day["Base Load"],
        name="Base Load",
        marker_color=base_color
    ))
    fig.add_trace(go.Bar(
        x=hours,
        y=df_day["Regular Load"],
        name="Regular Load",
        marker_color=regular_color
    ))
    fig.add_trace(go.Bar(
        x=hours,
        y=df_day["Peak Load"],
        name="Peak Load",
        marker_color=peak_color
    ))

    fig.update_layout(
        barmode="stack",
        xaxis_title="Hour of Day",
        yaxis_title="Consumption (kWh)",
        legend_title="Load Type",
        margin=dict(l=40, r=20, t=40, b=40),
        height=400
    )
    return fig