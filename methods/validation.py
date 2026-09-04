"""Validation for untrusted household consumption uploads."""

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class DataQuality:
    usable: bool
    rows: int
    start: object | None
    end: object | None
    resolution_minutes: int | None
    duplicates: int
    gaps: int
    non_positive: int
    message: str


def inspect_consumption(df: pd.DataFrame) -> DataQuality:
    """Return a conservative readiness result before pricing an upload."""
    required = {"timestamp", "consumption_kwh"}
    if df.empty or not required.issubset(df.columns):
        return DataQuality(False, len(df), None, None, None, 0, 0, 0, "No usable timestamp and consumption columns were found.")

    data = df.sort_values("timestamp")
    duplicates = int(data["timestamp"].duplicated().sum())
    non_positive = int((data["consumption_kwh"] < 0).sum())
    deltas = data["timestamp"].diff().dropna().dt.total_seconds().div(60)
    resolution = int(deltas[deltas > 0].mode().iloc[0]) if (deltas > 0).any() else None
    gaps = int((deltas > (resolution or 60) * 1.5).sum()) if resolution else 0
    usable = duplicates == 0 and non_positive == 0 and resolution is not None
    message = "Ready for analysis." if usable else "Fix duplicate timestamps, negative consumption, or timestamp resolution before analysis."
    return DataQuality(usable, len(data), data["timestamp"].min(), data["timestamp"].max(), resolution, duplicates, gaps, non_positive, message)


def inspect_price_coverage(df: pd.DataFrame) -> float:
    """Return matched consumption-row coverage; values below 100% are not comparable."""
    if df.empty or "spot_price_eur_kwh" not in df:
        return 0.0
    return float(df["spot_price_eur_kwh"].notna().mean())


import streamlit as st


def render_data_quality(quality: DataQuality, coverage: float | None = None) -> None:
    """Show the evidence behind a result before presenting a tariff recommendation."""
    label = "Data quality" if st.session_state.get("lang", "de") == "en" else "Datenqualität"
    with st.expander(label, expanded=not quality.usable):
        st.caption(quality.message)
        cols = st.columns(4)
        cols[0].metric("Rows", f"{quality.rows:,}")
        cols[1].metric("Resolution", f"{quality.resolution_minutes or '–'} min")
        cols[2].metric("Duplicates", quality.duplicates)
        cols[3].metric("Gaps", quality.gaps)
        if coverage is not None:
            st.metric("Spot-price coverage", f"{coverage:.0%}")

