import re

def clean_input_general(input_dict: dict) -> dict:
    cleaned = {}

    CLEAN_CATEGORICAL = {
        # Appeals
        "type_of_casework",
        "lpa_name",
        "procedure",
        "development_type",
        "reason_for_the_appeal",
        "type_detail",

        "application_type",
        "ward",
        "conservation_areas",
        "neighbourhood_areas",
    }

    TEXT_COLUMNS = {
        "appeal_type_reason",
        "development_description"
    }

    for key, value in input_dict.items():

        # numeric or binary → nothing to clean
        if isinstance(value, (int, float)) or value is None:
            cleaned[key] = value
            continue

        # -------------------------
        # postcode_district
        # -------------------------
        if key == "postcode_district":
            v = value.replace(" ", "").upper()
            # If full postcode → cut district
            if len(v) > 4:
                v = v[:-3]
            cleaned[key] = v
            continue

        # -------------------------
        # text fields (TF-IDF)
        # -------------------------
        if key in TEXT_COLUMNS:
            cleaned[key] = value.strip().lower()
            continue

        # -------------------------
        # categorical features
        # -------------------------
        if key in CLEAN_CATEGORICAL:
            v = (
                value.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("&", "and")
            )
            v = re.sub(r"[^a-z0-9_]", "", v)

            cleaned[key] = v
            continue

        cleaned[key] = value.strip().lower()

    return cleaned
