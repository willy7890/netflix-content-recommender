
import pandas as pd


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare text features for content-based recommendations."""
    data = df.copy()

    # Normalize column names.
    data.columns = data.columns.str.strip().str.lower()

    # Remove records without usable titles.
    if "title" not in data.columns:
        raise ValueError("Dataset must contain a 'title' column.")

    data["title"] = data["title"].fillna("").astype(str).str.strip()
    data = data[data["title"] != ""].copy()

    # Use only features available in the dataset.
    feature_columns = [
        column
        for column in [
            "listed_in",
            "director",
            "type",
            "country",
            "rating",
        ]
        if column in data.columns
    ]

    if not feature_columns:
        raise ValueError("No supported content feature columns were found.")

    # Normalize feature values.
    for column in feature_columns:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

    # Combine features into one text field for TF-IDF.
    data["content_features"] = data[feature_columns].agg(" ".join, axis=1)
    data["content_features"] = (
        data["content_features"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return data.reset_index(drop=True)
