import pandas as pd

# -------------------------------
# Column name normalisation
# -------------------------------
def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.lower()
                  .str.strip()
                  .str.replace(" ", "_")
                  .str.replace(r"[^a-z0-9_]", "", regex=True)
                  .str.replace(r"_+", "_", regex=True)
    )
    return df


# -------------------------------
# Standardise missing values
# -------------------------------
MISSING_STRINGS = [
    "na", "n/a", "n\\a", "nan", "<na>", "none", "null", "nil",
    "", " ", "  ", "-", "--", "N/A", "NaN", "NA", "NULL", "None"
]

def standardise_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(MISSING_STRINGS, pd.NA)
    df = df.replace(r"^\s+$", pd.NA, regex=True)
    return df


# -------------------------------
# Drop irrelevant columns
# -------------------------------
def drop_columns(df: pd.DataFrame, col_list) -> pd.DataFrame:
    return df.drop(columns=col_list, errors="ignore")


# -------------------------------
# Clean categorical string columns
# -------------------------------
def clean_string_cols(df: pd.DataFrame, cols) -> None:
    for col in cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )


# -------------------------------
# Extract postcode + district
# -------------------------------
def add_postcode_district(df: pd.DataFrame, address_col: str) -> pd.DataFrame:
    df = df.copy()

    df["postcode"] = df[address_col].str.extract(
        r"([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})",
        expand=False
    )

    df["postcode"] = (
        df["postcode"]
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )

    df["postcode_district"] = df["postcode"].str[:-3]

    return df.drop(columns=[address_col, "postcode"])