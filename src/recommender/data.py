
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset.csv"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the Netflix dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    df = pd.read_csv(path)

    # Remove duplicate records and normalize column names.
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates()

    # Clean text fields.
    text_columns = [
        "show_id", "type", "title", "director",
        "country", "rating", "duration", "listed_in"
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()

    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(
            df["release_year"], errors="coerce"
        )

    return df.reset_index(drop=True)


if __name__ == "__main__":
    netflix_df = load_data()

    print("Dataset loaded successfully!")
    print("Dataset shape:", netflix_df.shape)
    print("\nColumns:")
    print(netflix_df.columns.tolist())
    print("\nMissing values:")
    print(netflix_df.isnull().sum())
    print("\nFirst five records:")
    print(netflix_df.head())
