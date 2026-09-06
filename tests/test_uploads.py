# python -m pytest tests/test_uploads.py

import os
import glob
import pytest

from src.file_parser import ConsumptionDataParser


@pytest.fixture(scope="module")
def parser():
    """Shared parser instance across upload tests."""
    return ConsumptionDataParser()


CSV_FILES = sorted(glob.glob("uploads/*.csv"))


@pytest.mark.parametrize("file_path", CSV_FILES, ids=lambda p: os.path.basename(p))
def test_parse_upload_csv(parser, file_path):
    """Verify that every CSV export in uploads/ is parsed to standardized UTC time series."""
    with open(file_path, "rb") as f:
        df = parser.parse_file(f)

    assert not df.empty, f"Failed to parse {file_path}"
    assert list(df.columns) == ["timestamp", "consumption_kwh"]
    assert df["timestamp"].dt.tz is not None
    assert (df["consumption_kwh"] >= 0).all()
    assert len(df) > 0


def test_parse_upload_xlsx_graceful_handling(parser):
    """Excel formats without installed excel parser should gracefully return empty dataframe."""
    xlsx_files = sorted(glob.glob("uploads/*.xlsx"))
    for file_path in xlsx_files:
        with open(file_path, "rb") as f:
            df = parser.parse_file(f)
        assert df.empty, f"Expected empty dataframe for unsupported format: {file_path}"
