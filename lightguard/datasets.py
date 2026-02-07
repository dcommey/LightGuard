from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    default_filename: str
    url: Optional[str] = None
    notes: Optional[str] = None


DATASETS: dict[str, DatasetSpec] = {
    # CIC-Darknet-2020 (single CSV, matches our expected schema)
    "darknet": DatasetSpec(
        key="darknet",
        display_name="CIC-Darknet-2020 (Darknet.csv)",
        default_filename="Darknet.csv",
        url="http://205.174.165.80/CICDataset/CICDarknet2020/Dataset/Darknet.CSV",
        notes="Single CSV download; expected CICFlowMeter-like columns.",
    ),
    # CICIDS2017 often comes as multiple CSV files; we support any single CSV file
    # that contains the core CICFlowMeter columns.
    "cicids2017": DatasetSpec(
        key="cicids2017",
        display_name="CICIDS2017 (single CSV)",
        default_filename="CICIDS2017.csv",
        url=None,
        notes=(
            "CICIDS2017 is typically distributed as multiple CSVs. Provide a path to a single CSV "
            "export (or merge them yourself). Column aliases like 'Source IP' will be normalized."
        ),
    ),

    # ISCX VPN/NonVPN 2016: published as zipped ARFF scenarios under the CIC mirror.
    # We download + convert ARFF to CSV automatically.
    "iscx_vpn_nonvpn_2016": DatasetSpec(
        key="iscx_vpn_nonvpn_2016",
        display_name="ISCX-VPN-NonVPN-2016 (ARFF zips → CSV)",
        default_filename="iscx_vpn_nonvpn_2016",  # directory name under data/
        url="http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/CSVs/",
        notes=(
            "Downloads Scenario *-ARFF.zip files and converts contained .arff files to CSV. "
            "The analyzer can read a directory of CSVs."
        ),
    ),

    # CSE-CIC-IDS2018: official distribution is an AWS S3 public bucket (not a single CSV URL).
    "cse_cic_ids2018": DatasetSpec(
        key="cse_cic_ids2018",
        display_name="CSE-CIC-IDS2018 (AWS S3 public bucket)",
        default_filename="cse_cic_ids2018",  # directory name under data/
        url=None,
        notes=(
            "Official download via AWS CLI (no AWS account): "
            "aws s3 sync --no-sign-request --region ca-central-1 \"s3://cse-cic-ids2018/\" <dest-dir>. "
            "See https://registry.opendata.aws/cse-cic-ids2018/."
        ),
    ),
}


def get_dataset_spec(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError as e:
        valid = ", ".join(sorted(DATASETS.keys()))
        raise KeyError(f"Unknown dataset key '{key}'. Valid options: {valid}") from e
