"""Download and load the UCI Bank Marketing dataset."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_FILENAME = "bank-additional-full.csv"


def download_and_extract(raw_dir: str | Path, timeout: int = 60) -> Path:
    """Download the official UCI archive and return the full additional CSV path."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / RAW_FILENAME
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    request = Request(UCI_DATASET_URL, headers={"User-Agent": "AI-Pioneers-Internship-Week1/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            archive_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Could not download UCI data. Check connectivity or download the official "
            f"archive and place {RAW_FILENAME} in {raw_dir}. Original error: {exc}"
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next((name for name in archive.namelist() if name.endswith(RAW_FILENAME)), None)
            if member is None:
                raise FileNotFoundError(f"{RAW_FILENAME} was not found in the UCI archive")
            destination.write_bytes(archive.read(member))
    except zipfile.BadZipFile as exc:
        raise RuntimeError("The downloaded UCI file is not a valid ZIP archive") from exc
    return destination


def load_bank_marketing(csv_path: str | Path) -> pd.DataFrame:
    """Load the semicolon-delimited UCI CSV and validate the target column."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    frame = pd.read_csv(csv_path, sep=";", low_memory=False)
    if frame.empty:
        raise ValueError("The selected dataset is empty")
    if "y" not in frame.columns:
        raise ValueError("Expected target column 'y' is missing; use bank-additional-full.csv")
    return frame
