from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "fake_job_postings.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
REPORTS_DIR = ROOT_DIR / "reports"

TARGET_COL = "fraudulent"

TEXT_COLS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

CATEGORICAL_COLS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location",
    "department",
]

BINARY_COLS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

ALL_INPUT_COLS = [
    "job_id",
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]
