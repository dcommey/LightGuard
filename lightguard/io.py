from __future__ import annotations

import logging
import re
import zipfile
import csv
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_dataset(url: str, save_path: Path) -> bool:
    """Download dataset from URL."""

    logger.info(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            for data in tqdm(
                response.iter_content(block_size),
                total=total_size // block_size if total_size else None,
                unit="KB",
                desc="Downloading",
            ):
                f.write(data)

        logger.info(f"Dataset saved to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        if save_path.exists():
            save_path.unlink()
        return False


def download_files_from_apache_index(
    index_url: str,
    dest_dir: Path,
    *,
    allowed_extensions: tuple[str, ...] = (".csv", ".CSV", ".zip", ".ZIP"),
) -> list[Path]:
    """Download files linked from an Apache 'Index of' directory listing.

    This is intentionally lightweight (no HTML parser dependency).
    """

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(index_url, timeout=30)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logger.error(f"Error fetching index page {index_url}: {e}")
        return []

    hrefs = re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)
    downloaded: list[Path] = []

    for href in hrefs:
        if href in {"../", "./"}:
            continue
        if href.endswith("/"):
            continue

        if allowed_extensions and not href.lower().endswith(tuple(ext.lower() for ext in allowed_extensions)):
            continue

        file_url = urljoin(index_url, href)
        dest_path = dest_dir / Path(href).name
        if dest_path.exists():
            downloaded.append(dest_path)
            continue

        ok = download_dataset(file_url, dest_path)
        if ok:
            downloaded.append(dest_path)

    return downloaded


def prepare_iscx_vpn_nonvpn_2016(
    index_url: str,
    work_dir: Path,
) -> Path | None:
    """Download ISCX-VPN-NonVPN-2016 ARFF zips and convert them to CSV.

    Returns the directory containing converted CSVs (to be used as --data-path).
    """

    raw_dir = work_dir / "raw"
    extracted_dir = work_dir / "extracted"
    csv_dir = work_dir / "csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    zip_files = download_files_from_apache_index(index_url, raw_dir, allowed_extensions=(".zip", ".ZIP"))
    if not zip_files:
        logger.error("No zip files were downloaded; cannot prepare ISCX-VPN dataset.")
        return None

    try:
        import numpy as np
        import pandas as pd
    except Exception as e:
        logger.error(f"Missing dependencies to convert ARFF to CSV: {e}")
        return None

    def _arff_to_dataframe(arff_path: Path) -> "pd.DataFrame":
        """Parse CIC's ARFF export format (tolerates trailing commas and blank comma-only lines)."""

        attribute_names: list[str] = []
        rows: list[list[str]] = []
        in_data = False

        with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                # Skip lines that are effectively empty but contain commas
                if set(line) <= {","}:
                    continue

                # CIC mirror files often contain lots of trailing commas
                line = line.rstrip(",")

                upper = line.upper()
                if upper.startswith("@ATTRIBUTE"):
                    # Format: @ATTRIBUTE <name> <type>
                    parts = line.split()
                    if len(parts) >= 3:
                        attribute_names.append(parts[1])
                    continue

                if upper.startswith("@DATA"):
                    in_data = True
                    continue

                if not in_data:
                    continue

                if upper.startswith("@"):
                    continue

                # Data line
                fields = next(csv.reader([line]))
                while fields and fields[-1] == "":
                    fields.pop()

                # Keep rows aligned to attribute count; tolerate occasional junk lines.
                if attribute_names:
                    if len(fields) < len(attribute_names):
                        fields = fields + [""] * (len(attribute_names) - len(fields))
                    elif len(fields) > len(attribute_names):
                        fields = fields[: len(attribute_names)]

                rows.append(fields)

        df = pd.DataFrame(rows, columns=attribute_names if attribute_names else None)
        df.replace({"": np.nan}, inplace=True)
        return df

    arff_paths: list[Path] = []
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extracted_dir / zip_path.stem)
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            continue

    for p in extracted_dir.rglob("*.arff"):
        arff_paths.append(p)

    if not arff_paths:
        logger.error("No .arff files found after extracting zips.")
        return None

    for arff_path in sorted(arff_paths):
        try:
            df = _arff_to_dataframe(arff_path)

            out_name = arff_path.stem + ".csv"
            out_path = csv_dir / out_name
            df.to_csv(out_path, index=False)
        except Exception as e:
            logger.error(f"Failed to convert {arff_path} to CSV: {e}")

    csv_files = [p for p in csv_dir.glob("*.csv") if p.is_file()]
    if not csv_files:
        logger.error("ARFF conversion produced no CSV files.")
        return None

    return csv_dir
