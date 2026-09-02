import os, json, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler

ART_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
SCHEMA_PATH = os.path.join(ART_DIR, 'feature_schema.json')

_default = {
  "ALL_FEATURES": [
    'Age','Gender','Ethnicity','EducationLevel','BMI','Smoking',
    'AlcoholConsumption','PhysicalActivity','DietQuality','SleepQuality',
    'SystolicBP','DiastolicBP','CholesterolTotal','CholesterolLDL','CholesterolHDL',
    'CholesterolTriglycerides','MMSE','FunctionalAssessment','ADL',
    'MemoryComplaints','BehavioralProblems','Confusion','Disorientation',
    'PersonalityChanges','DifficultyCompletingTasks','Forgetfulness'
  ],
  "NUMERIC_COLS": [
    'Age','BMI','AlcoholConsumption','PhysicalActivity','DietQuality','SleepQuality',
    'SystolicBP','DiastolicBP','CholesterolTotal','CholesterolLDL','CholesterolHDL',
    'CholesterolTriglycerides','MMSE','FunctionalAssessment','ADL'
  ],
  "CATEGORICAL_COLS": [
    'Gender','Ethnicity','EducationLevel','Smoking','MemoryComplaints','BehavioralProblems',
    'Confusion','Disorientation','PersonalityChanges','DifficultyCompletingTasks','Forgetfulness'
  ],
  "TARGET_COL": "Diagnosis"
}

if os.path.exists(SCHEMA_PATH):
    with open(SCHEMA_PATH,'r',encoding='utf-8') as f:
        _schema = json.load(f)
else:
    _schema = _default

ALL_FEATURES = _schema['ALL_FEATURES']
NUMERIC_COLS = _schema['NUMERIC_COLS']
CATEGORICAL_COLS = _schema['CATEGORICAL_COLS']
TARGET_COL = _schema['TARGET_COL']

def fit_scaler(df: pd.DataFrame, path: str):
    scaler = StandardScaler()
    in_cols = [c for c in NUMERIC_COLS if c in df.columns]
    if in_cols:
        scaler.fit(df[in_cols])
    joblib.dump(scaler, path)
    return scaler

def transform_features(df: pd.DataFrame, scaler: StandardScaler):
    df = df.copy()
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0
    in_cols = [c for c in NUMERIC_COLS if c in df.columns]
    if in_cols:
        df[in_cols] = scaler.transform(df[in_cols])
    return df[ALL_FEATURES]
