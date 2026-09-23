
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MAX_RUL = 125

# Official NASA C-MAPSS dataset archive.
DATASET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.%20Turbofan%20Engine%20Degradation%20Simulation%20Data%20Set.zip"
)

DATASET_ZIP = DEFAULT_DATA_DIR / "CMAPSSData.zip"

OPERATIONAL_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]

SENSOR_INFO = {
    "sensor_1": {
        "symbol": "T2",
        "description": "Total temperature at fan inlet",
        "unit": "°R",
    },
    "sensor_2": {
        "symbol": "T24",
        "description": "Total temperature at LPC outlet",
        "unit": "°R",
    },
    "sensor_3": {
        "symbol": "T30",
        "description": "Total temperature at HPC outlet",
        "unit": "°R",
    },
    "sensor_4": {
        "symbol": "T50",
        "description": "Total temperature at LPT outlet",
        "unit": "°R",
    },
    "sensor_5": {
        "symbol": "P2",
        "description": "Pressure at fan inlet",
        "unit": "psia",
    },
    "sensor_6": {
        "symbol": "P15",
        "description": "Total pressure in bypass duct",
        "unit": "psia",
    },
    "sensor_7": {
        "symbol": "P30",
        "description": "Total pressure at HPC outlet",
        "unit": "psia",
    },
    "sensor_8": {
        "symbol": "Nf",
        "description": "Physical fan speed",
        "unit": "rpm",
    },
    "sensor_9": {
        "symbol": "Nc",
        "description": "Physical core speed",
        "unit": "rpm",
    },
    "sensor_10": {
        "symbol": "epr",
        "description": "Engine pressure ratio (P50/P2)",
        "unit": "-",
    },
    "sensor_11": {
        "symbol": "Ps30",
        "description": "Static pressure at HPC outlet",
        "unit": "psia",
    },
    "sensor_12": {
        "symbol": "phi",
        "description": "Ratio of fuel flow to Ps30",
        "unit": "pps/psi",
    },
    "sensor_13": {
        "symbol": "NRf",
        "description": "Corrected fan speed",
        "unit": "rpm",
    },
    "sensor_14": {
        "symbol": "NRc",
        "description": "Corrected core speed",
        "unit": "rpm",
    },
    "sensor_15": {
        "symbol": "BPR",
        "description": "Bypass ratio",
        "unit": "-",
    },
    "sensor_16": {
        "symbol": "farB",
        "description": "Burner fuel-air ratio",
        "unit": "-",
    },
    "sensor_17": {
        "symbol": "htBleed",
        "description": "Bleed enthalpy",
        "unit": "-",
    },
    "sensor_18": {
        "symbol": "Nf_dmd",
        "description": "Demanded fan speed",
        "unit": "rpm",
    },
    "sensor_19": {
        "symbol": "PCNfR_dmd",
        "description": "Demanded corrected fan speed",
        "unit": "rpm",
    },
    "sensor_20": {
        "symbol": "W31",
        "description": "HPT coolant bleed",
        "unit": "lbm/s",
    },
    "sensor_21": {
        "symbol": "W32",
        "description": "LPT coolant bleed",
        "unit": "lbm/s",
    },
}

COLUMNS = [
    "unit_number",
    "cycle",
    *OPERATIONAL_COLUMNS,
    *[f"sensor_{i}" for i in range(1, 22)],
]


def _extract_expected_files(
    zip_ref: zipfile.ZipFile,
    data_dir: Path,
) -> set[str]:
    """
    Extract the three FD001 files from a ZIP archive.

    The files may appear inside a folder such as CMAPSSData/,
    so only the filename itself is checked.
    """
    expected = {
        "train_FD001.txt",
        "test_FD001.txt",
        "RUL_FD001.txt",
    }

    found: set[str] = set()

    for member in zip_ref.infolist():
        filename = Path(member.filename).name

        if filename not in expected:
            continue

        target = data_dir / filename

        with zip_ref.open(member) as source:
            with target.open("wb") as destination:
                destination.write(source.read())

        found.add(filename)

    return found


def ensure_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """
    Ensure that the NASA C-MAPSS FD001 dataset is available.

    If the three FD001 files already exist, they are reused.

    Otherwise, the official NASA archive is downloaded automatically.
    The downloader supports both:
      1. FD001 files directly inside the downloaded ZIP.
      2. A nested CMAPSSData.zip containing the FD001 files.

    This makes a fresh GitHub download reproducible without requiring
    the user to manually download the dataset first.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_FD001.txt"
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"

    required_files = (train_path, test_path, rul_path)

    # Dataset is already available.
    if all(path.exists() for path in required_files):
        return data_dir

    archive = data_dir / "CMAPSSData.zip"

    # Download the official archive if necessary.
    if not archive.exists():
        print("NASA C-MAPSS archive not found. Downloading it now...")

        try:
            urllib.request.urlretrieve(DATASET_URL, archive)
        except Exception as exc:
            # Remove a potentially incomplete download.
            if archive.exists():
                archive.unlink()

            raise RuntimeError(
                "Could not download the NASA C-MAPSS dataset automatically.\n"
                f"URL: {DATASET_URL}\n"
                "Check your internet connection and try again."
            ) from exc

    try:
        with zipfile.ZipFile(archive, "r") as outer_zip:

            # Case 1:
            # The required FD001 files are directly inside the archive.
            found = _extract_expected_files(
                outer_zip,
                data_dir,
            )

            # Case 2:
            # The downloaded archive contains another ZIP, commonly
            # CMAPSSData.zip, which contains the actual FD001 files.
            if len(found) < 3:

                for member in outer_zip.infolist():

                    if not member.filename.lower().endswith(".zip"):
                        continue

                    try:
                        nested_bytes = outer_zip.read(member)

                        with zipfile.ZipFile(
                            io.BytesIO(nested_bytes),
                            "r",
                        ) as nested_zip:

                            found.update(
                                _extract_expected_files(
                                    nested_zip,
                                    data_dir,
                                )
                            )

                    except zipfile.BadZipFile:
                        # Ignore unrelated ZIP-like files and continue
                        # searching the outer archive.
                        continue

                    if len(found) == 3:
                        break

    except zipfile.BadZipFile as exc:
        # Delete the corrupt archive so the next run can download
        # a clean copy.
        if archive.exists():
            archive.unlink()

        raise RuntimeError(
            "The downloaded NASA C-MAPSS archive is corrupted or "
            "is not a valid ZIP file. The incomplete archive was "
            "removed. Please run the command again."
        ) from exc

    # Verify that all three required files now exist.
    missing = [
        path.name
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "The NASA C-MAPSS archive was downloaded, but the required "
            "FD001 files could not be found:\n"
            + "\n".join(f"  - {filename}" for filename in missing)
            + "\n\n"
            "Expected files:\n"
            "  - train_FD001.txt\n"
            "  - test_FD001.txt\n"
            "  - RUL_FD001.txt"
        )

    print("NASA C-MAPSS FD001 dataset is ready.")

    return data_dir


def read_cmapss_file(path: Path) -> pd.DataFrame:
    """Read one raw C-MAPSS space-delimited file."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )

    return df


def add_train_rul(
    train_df: pd.DataFrame,
    clip_max: int = MAX_RUL,
) -> pd.DataFrame:
    """Calculate run-to-failure RUL for every training row."""
    df = train_df.copy()

    max_cycle = (
        df.groupby("unit_number")["cycle"]
        .transform("max")
    )

    df["RUL"] = max_cycle - df["cycle"]

    if clip_max is not None:
        df["RUL"] = df["RUL"].clip(
            upper=clip_max
        )

    return df


def add_test_rul(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    clip_max: int | None = None,
) -> pd.DataFrame:
    """
    Reconstruct row-level test RUL from the final observed cycle
    and the NASA RUL file.
    """
    df = test_df.copy()

    truth = rul_df.rename(
        columns={
            rul_df.columns[0]: "final_RUL"
        }
    ).copy()

    truth["unit_number"] = np.arange(
        1,
        len(truth) + 1,
    )

    last_cycle = (
        df.groupby("unit_number")["cycle"]
        .max()
        .rename("max_test_cycle")
        .reset_index()
    )

    df = df.merge(
        last_cycle,
        on="unit_number",
        how="left",
    )

    df = df.merge(
        truth,
        on="unit_number",
        how="left",
    )

    df["RUL"] = (
        df["max_test_cycle"]
        - df["cycle"]
        + df["final_RUL"]
    )

    df = df.drop(
        columns=[
            "max_test_cycle",
            "final_RUL",
        ]
    )

    if clip_max is not None:
        df["RUL"] = df["RUL"].clip(
            upper=clip_max
        )

    return df


def split_train_validation(
    df: pd.DataFrame,
    validation_fraction: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split whole engines rather than random rows to reduce
    temporal/entity leakage.
    """
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )

    train_idx, valid_idx = next(
        splitter.split(
            df,
            groups=df["unit_number"],
        )
    )

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[valid_idx].reset_index(drop=True),
    )


def get_informative_features(
    train_df: pd.DataFrame,
    near_zero_variance: float = 1e-12,
) -> list[str]:
    """Return features whose training-set variance is meaningfully non-zero."""
    candidate_features = [
        column
        for column in COLUMNS
        if column != "unit_number"
    ]

    variances = train_df[
        candidate_features
    ].var(
        numeric_only=True
    )

    return [
        column
        for column in candidate_features
        if variances[column] > near_zero_variance
    ]


def load_fd001(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download/load FD001 and return prepared train and test dataframes."""
    data_dir = ensure_dataset(data_dir)

    train_raw = read_cmapss_file(
        data_dir / "train_FD001.txt"
    )

    test_raw = read_cmapss_file(
        data_dir / "test_FD001.txt"
    )

    rul_raw = pd.read_csv(
        data_dir / "RUL_FD001.txt",
        sep=r"\s+",
        header=None,
    )

    train = add_train_rul(
        train_raw,
        clip_max=MAX_RUL,
    )

    test = add_test_rul(
        test_raw,
        rul_raw,
        clip_max=None,
    )

    return train, test


def main() -> None:
    """Download/load FD001 and print a compact dataset summary."""
    data_dir = ensure_dataset()

    train, test = load_fd001(
        data_dir
    )

    print(
        f"FD001 data ready in: {data_dir}"
    )

    print(
        f"Training rows: {len(train):,}; "
        f"engines: {train['unit_number'].nunique()}"
    )

    print(
        f"Test rows: {len(test):,}; "
        f"engines: {test['unit_number'].nunique()}"
    )

    print(
        f"Clipped training RUL range: "
        f"{train['RUL'].min():.0f} to "
        f"{train['RUL'].max():.0f} cycles"
    )


if __name__ == "__main__":
    main()

