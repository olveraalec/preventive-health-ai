"""Notebook 02 cells 9–10, versioned causal Sepsis hourly features.

Windows and delta lags count observed records; ages use ICU hours. No resampling,
unit conversion, backfill, learned imputation, labels, or cross-patient filling.
"""

import numpy as np
import pandas as pd

FEATURE_SCHEMA = "sepsis2019.notebook02.temporal.v1"
PREPROCESSING_VERSION = "1"
PIPELINE_ID = "notebook02.cells9-10"
NOTEBOOK_TRANSFORM_SHA256 = (
    "46b9a84a5d38188de7c5ddc0423bf7fb65e1e95865c419456411c901044b8527"
)

VITALS = ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp")
ADMIN = ("Age", "Gender", "HospAdmTime", "Unit1", "Unit2")
RAW_REQUIRED = (*VITALS, *ADMIN)
# Standard Challenge columns not used by this version of Notebook 02's builder.
UNUSED_CHALLENGE_COLUMNS = (
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
)


def feature_sources() -> dict[str, list[str]]:
    """Source-column lineage, including the time source used for measurement ages."""
    sources = {name: [name] for name in ("ICULOS", *ADMIN)}
    for col in VITALS:
        for suffix in ("current", "observed", "last", "mean6", "std6", "delta3"):
            sources[f"{col}_{suffix}"] = [col]
        sources[f"{col}_age"] = [col, "ICULOS"]
    sources.update(
        {
            "HR_x_MAP": ["HR", "MAP"],
            "Resp_x_O2Sat": ["Resp", "O2Sat"],
            "HRtrend_x_MAPtrend": ["HR", "MAP"],
            "Respstd_x_Respdelta": ["Resp"],
            "ICULOS_sin24": ["ICULOS"],
            "ICULOS_cos24": ["ICULOS"],
        }
    )
    return sources


def validate_raw_patient(frame: pd.DataFrame) -> None:
    """Require a single patient and already ordered, positive integral ICU hours.

    A missing column is an error; an explicitly missing measurement is allowed.
    Extra source columns, including evaluation columns, never enter the features.
    """
    required = {"patient_id", "ICULOS", *RAW_REQUIRED}
    if required - set(frame.columns):
        raise ValueError(
            f"Missing raw source columns: {sorted(required - set(frame.columns))}"
        )
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Expected nonempty raw history with unique columns")
    ids = frame["patient_id"]
    if (
        ids.isna().any()
        or ids.nunique() != 1
        or not ids.map(lambda v: isinstance(v, str) and bool(v)).all()
    ):
        raise ValueError("Expected one nonempty patient identifier")
    hours = frame["ICULOS"].to_numpy(dtype=float)
    if (
        not np.isfinite(hours).all()
        or np.any(hours <= 0)
        or np.any(hours != np.floor(hours))
        or np.any(np.diff(hours) <= 0)
    ):
        raise ValueError(
            "ICU hours must be finite positive integers, strictly increasing and unique"
        )
    values = frame[list(RAW_REQUIRED)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite raw measurements are invalid")


def build_sepsis_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one patient's features with exactly the Notebook 02 transformations.

    Input units are retained. NaNs remain until model-owned preprocessing.
    No label, split, patient outcome, or inferred onset appears in the output.
    """
    validate_raw_patient(frame)
    df = frame.reset_index(drop=True).copy()
    # JSON null and pandas NaN share the notebook's numeric missing semantics.
    # Gender is integral in the processed schema; retain it when fully observed.
    for col in ADMIN:
        if col != "Gender" or not pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
    # Preserve source administrative dtypes, including the notebook's integer Gender.
    temporal = df[["patient_id", "ICULOS", *ADMIN]].copy()
    for col in VITALS:
        df[col] = df[col].astype(float)
        observed = df[col].notna()
        temporal[f"{col}_current"] = df[col]
        temporal[f"{col}_observed"] = observed.astype("int8")
        last = df.groupby("patient_id")[col].ffill()
        temporal[f"{col}_last"] = last
        observed_hour = df["ICULOS"].where(observed)
        last_seen_hour = observed_hour.groupby(df["patient_id"]).ffill()
        temporal[f"{col}_age"] = (df["ICULOS"] - last_seen_hour).astype(float)
        grouped = last.groupby(df["patient_id"])
        temporal[f"{col}_mean6"] = grouped.transform(
            lambda s: s.rolling(6, min_periods=2).mean()
        )
        temporal[f"{col}_std6"] = grouped.transform(
            lambda s: s.rolling(6, min_periods=2).std()
        )
        temporal[f"{col}_delta3"] = last - grouped.shift(3)
    temporal["HR_x_MAP"] = temporal["HR_last"] * temporal["MAP_last"]
    temporal["Resp_x_O2Sat"] = temporal["Resp_last"] * temporal["O2Sat_last"]
    temporal["HRtrend_x_MAPtrend"] = temporal["HR_delta3"] * temporal["MAP_delta3"]
    temporal["Respstd_x_Respdelta"] = temporal["Resp_std6"] * temporal["Resp_delta3"]
    temporal["ICULOS_sin24"] = np.sin(2 * np.pi * temporal["ICULOS"] / 24)
    temporal["ICULOS_cos24"] = np.cos(2 * np.pi * temporal["ICULOS"] / 24)
    return temporal


class IncrementalSepsisFeatures:
    """Patient-local reference stream; recomputes each prefix for numerical parity.

    Keeps the supplied history; O(n²) work over n updates. This deliberately shares
    batch rolling arithmetic rather than introducing a second statistical formula.
    Create a new instance for each episode. Failed updates do not mutate history.
    """

    def __init__(self, patient_id: str):
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("patient_id is required")
        self.patient_id = patient_id
        self._rows: list[dict] = []

    def update(self, observation: dict) -> pd.DataFrame:
        row = dict(observation)
        if row.get("patient_id") != self.patient_id:
            raise ValueError("Mixed patient identifiers")
        validate_raw_patient(pd.DataFrame([row]))
        rows = [*self._rows, row]
        features = build_sepsis_features(pd.DataFrame(rows))
        self._rows = rows
        return features.iloc[[-1]].copy()
