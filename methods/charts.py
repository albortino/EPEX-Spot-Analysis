import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from methods.config import FLEX_COLOR, MEKKO_BORDER, FLEX_COLOR_LIGHT, STATIC_COLOR, FLEX_COLOR_SHADE
import calendar

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
        tickvals=list(range(len(month_numbers))),
        ticktext=month_names
    )

    return fig

def get_consumption_chart(df: pd.DataFrame) -> go.Figure:
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
    fig.update_layout(xaxis_title=idx.name, yaxis_title="Consumption (kWh)", legend_title_text="Metrics", hovermode="x unified")

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


def get_trend_chart(df_daily_trend: pd.DataFrame, df_forecast: pd.DataFrame) -> go.Figure:
    """Creates a line chart showing daily consumption, its trend, and a forecast."""
    
    fig = go.Figure()

    # Add historical daily consumption
    fig.add_trace(go.Scatter(
        x=df_daily_trend['timestamp'], 
        y=df_daily_trend['consumption_kwh'],
        mode='lines',
        name='Daily Consumption',
        line=dict(color='#1f77b4', width=1.5),
        opacity=0.7
    ))

    # Add the calculated trendline
    fig.add_trace(go.Scatter(
        x=df_daily_trend['timestamp'],
        y=df_daily_trend['trend'],
        mode='lines',
        name='Trendline',
        line=dict(color='#ff7f0e', width=3)
    ))
    
    # Add the forecast line
    fig.add_trace(go.Scatter(
        x=df_forecast['timestamp'],
        y=df_forecast['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='#d62728', width=3, dash='dash')
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Consumption (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig