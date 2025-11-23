from fastapi import FastAPI
import os
from pydantic import BaseModel
import joblib
import pandas as pd
from utils import clean_input_general
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Planning ML API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

models_path = f'{os.getcwd()}/src/models'

# --------------------------------------------------------------
# Generic loader
# --------------------------------------------------------------
def load_artifacts(prefix, model_name):
    model = joblib.load(f"{models_path}/{prefix}_{model_name}.joblib")
    threshold = joblib.load(f"{models_path}/{prefix}_{model_name}_threshold.pkl")
    metadata = joblib.load(f"{models_path}/{prefix}_{model_name}_metadata.pkl")
    return model, threshold, metadata

appeals_model, appeals_threshold, appeals_meta = load_artifacts(
    prefix="appeals",
    model_name="XGBoost"
)

applications_model, applications_threshold, applications_meta = load_artifacts(
    prefix="camden_pa",
    model_name="RandomForest"
)


# --------------------------------------------------------------
# Shared prediction helper
# --------------------------------------------------------------
def predict(pipeline, threshold, input_dict):
    cleaned = clean_input_general(input_dict)
    df = pd.DataFrame([cleaned])
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    return int(preds[0]), float(probs[0])


# --------------------------------------------------------------
# Pydantic schema for APPEALS
# --------------------------------------------------------------
class AppealsInput(BaseModel):
    # binary
    site_green_belt: int
    agricultural_holding: int
    development_affect_setting_of_listed_building: int
    historic_building_grant_made: int
    in_ca_relates_to_ca: int
    is_flooding_an_issue: int
    is_the_site_within_an_aonb: int
    is_site_within_an_sssi: int

    # numeric
    area_of_site_in_hectares: float
    floor_space_in_square_metres: float
    number_of_residences: float

    # onehot categoricals
    procedure: str
    development_type: str
    reason_for_the_appeal: str
    type_detail: str
    type_of_casework: str

    # target encoding cols
    lpa_name: str
    postcode_district: str

    # text
    appeal_type_reason: str


# --------------------------------------------------------------
# Pydantic schema for PLANNING APPLICATIONS
# --------------------------------------------------------------
class PlanningApplicationInput(BaseModel):
    # binary
    in_conservation_area: int
    in_neighbourhood_area: int

    # onehot categoricals
    application_type: str
    ward: str
    conservation_areas: str
    neighbourhood_areas: str

    # target encoding cols
    postcode_district: str

    # text
    development_description: str


# --------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------

@app.post("/predict/appeals")
def predict_appeals(data: AppealsInput):
    pred, prob = predict(appeals_model, appeals_threshold, data.model_dump())
    return {
        "dataset": "appeals",
        "prediction": pred,
        "probability": prob,
        "threshold": float(appeals_threshold),
        "model": appeals_meta["model_name"],
        "metadata": appeals_meta
    }


@app.post("/predict/applications")
def predict_applications(data: PlanningApplicationInput):
    pred, prob = predict(applications_model, applications_threshold, data.model_dump())
    return {
        "dataset": "planning_applications",
        "prediction": pred,
        "probability": prob,
        "threshold": float(applications_threshold),
        "model": applications_meta["model_name"],
        "metadata": applications_meta
    }


@app.get("/health")
def health():
    print(models_path)
    return {"status": "ok"}
