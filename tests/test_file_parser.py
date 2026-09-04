import os
import io
import pytest
import pandas as pd

from methods.file_parser import JavaScriptNetzbetreiberParser, ConsumptionDataParser


def test_javascript_parser_object_literal():
    """Verify that JavaScriptNetzbetreiberParser extracts properties from JS object literals."""
    js_sample = """
    export const SampleNetz = new Netzbetreiber({
        name: "TestNetz",
        descriptorUsage: "Gemessener Verbrauch (kWh)",
        descriptorTimestamp: "Messzeitpunkt",
        dateFormatString: "dd.MM.yyyy HH:mm",
        usageParser: parseGermanFloat,
        otherFields: ["Ersatzwert"],
        fixupTimestamp: true,
        feedin: false
    });
    """
    parser = JavaScriptNetzbetreiberParser()
    configs = parser.parse_js_file(js_sample)
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.name == "TestNetz"
    assert cfg.usage_col == "Gemessener Verbrauch (kWh)"
    assert cfg.timestamp_col == "Messzeitpunkt"
    assert cfg.date_format == "%d.%m.%Y %H:%M"
    assert cfg.other_cols == ["Ersatzwert"]
    assert cfg.fixup_timestamp is True
    assert cfg.feedin is False


@pytest.fixture
def parser():
    return ConsumptionDataParser()


def test_parse_empty_or_invalid_file(parser):
    """Empty or unparseable input must return an empty DataFrame without raising an exception."""
    assert parser.parse_file(None).empty
    empty_buffer = io.BytesIO(b"")
    assert parser.parse_file(empty_buffer).empty
    invalid_buffer = io.BytesIO(b"random text without headers\n1;2;3\n")
    assert parser.parse_file(invalid_buffer).empty


def test_parse_example_data_15m(parser):
    """Ensure standard 15-minute sample data parses into UTC timestamps and positive consumption."""
    file_path = "resources/EXAMPLE-DATA-15M.csv"
    if not os.path.exists(file_path):
        pytest.skip(f"{file_path} not found")

    with open(file_path, "rb") as f:
        df = parser.parse_file(f)

    assert not df.empty
    assert list(df.columns) == ["timestamp", "consumption_kwh"]
    assert df["timestamp"].dt.tz is not None
    assert (df["consumption_kwh"] >= 0).all()
    assert len(df) > 50000


def test_parse_linzag_daily(parser):
    """Ensure daily Linz AG format parses and aggregates properly."""
    file_path = "uploads/Strombezug_2024_linzag.csv"
    if not os.path.exists(file_path):
        pytest.skip(f"{file_path} not found")

    with open(file_path, "rb") as f:
        df = parser.parse_file(f)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "consumption_kwh" in df.columns
    assert df["timestamp"].dt.tz is not None
    assert len(df) > 1000


def test_parse_wienernetze(parser):
    """Ensure Wiener Netze 15-minute smart meter export parses properly."""
    file_path = "uploads/VIERTELSTUNDENWERTE-20250724_wienernetze.csv"
    if not os.path.exists(file_path):
        pytest.skip(f"{file_path} not found")

    with open(file_path, "rb") as f:
        df = parser.parse_file(f)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "consumption_kwh" in df.columns
    assert len(df) > 50000


def test_parse_econtrol_iso8601(parser):
    """Ensure E-Control export with ISO8601 offset and 15-min period end parses properly."""
    file_path = "uploads/VIERTELSTUNDENWERTE_ECONTROL-20241007-bis-20251017.csv"
    if not os.path.exists(file_path):
        pytest.skip(f"{file_path} not found")

    with open(file_path, "rb") as f:
        df = parser.parse_file(f)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "consumption_kwh" in df.columns
    assert df["timestamp"].dt.tz is not None
    assert len(df) > 30000


def test_parse_smart_meter_daily(parser):
    """Ensure Smart Meter daily export format parses and aggregates properly."""
    file_path = "uploads/Tageswerte-20260101-bis-20260903.csv"
    if not os.path.exists(file_path):
        pytest.skip(f"{file_path} not found")

    with open(file_path, "rb") as f:
        df = parser.parse_file(f)

    assert not df.empty
    assert list(df.columns) == ["timestamp", "consumption_kwh"]
    assert df["timestamp"].dt.tz is not None
    assert (df["consumption_kwh"] >= 0).all()
    assert len(df) > 1000


def test_parse_at_format_quarter_hourly(parser):
    """Ensure Austrian EDA AT-Format 15-minute export parses properly."""
    file_path = "uploads/AT-Format-AT0010000000000000001000000941581-20260101-20260903-QH.csv"
    if not os.path.exists(file_path):
        pytest.skip(f"{file_path} not found")

    with open(file_path, "rb") as f:
        df = parser.parse_file(f)

    assert not df.empty
    assert list(df.columns) == ["timestamp", "consumption_kwh"]
    assert df["timestamp"].dt.tz is not None
    assert (df["consumption_kwh"] >= 0).all()
    assert len(df) > 20000


def test_parse_at_format_daily(parser):
    """Ensure Austrian EDA AT-Format daily export parses and aggregates properly."""
    file_path = "uploads/AT-Format-AT0010000000000000001000000941581-20260101-20260903-D.csv"
    if not os.path.exists(file_path):
        pytest.skip(f"{file_path} not found")

    with open(file_path, "rb") as f:
        df = parser.parse_file(f)

    assert not df.empty
    assert list(df.columns) == ["timestamp", "consumption_kwh"]
    assert df["timestamp"].dt.tz is not None
    assert (df["consumption_kwh"] >= 0).all()
    assert len(df) > 1000

