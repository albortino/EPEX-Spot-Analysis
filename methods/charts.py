import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from methods.config import FLEX_COLOR, MEKKO_BORDER
import streamlit as st


def get_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = px.imshow(df, labels=dict(x="Hour of Day", y="Month", color="Avg Spot Price (€/kWh)"), aspect="auto", color_continuous_scale="Viridis")
    return fig

def get_consumption_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    idx = df.index
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
            text=f'{label}<br>{width:.1%}<br>€{price:.3f}/kWh',
            showarrow=False,
            font=dict(color="white", size=11)
        ))

        cumulative_width += width

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