"""prep.py
- Loads tourism.csv from Hugging Face dataset repo
- Cleans data (fixes typos, unifies labels)
- Preprocesses features (encoding + scaling via sklearn Pipeline)
- Splits into train (80%) and test (20%) with stratification
- Saves processed splits back to Hugging Face dataset repo
Triggered as the second job (prepare-data) in the GitHub Actions pipeline.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from huggingface_hub import hf_hub_download, HfApi

# ── CONFIG ────────────────────────────────────────────────────────────────────
HF_USERNAME   = os.getenv("HF_USERNAME", "vivekkumar-hf")
DATASET_REPO  = f"{HF_USERNAME}/tourism-data"
HF_TOKEN      = os.getenv("HF_TOKEN")
RANDOM_STATE  = 42
TEST_SIZE     = 0.20

# ── FEATURE GROUPS ───────────────────────────────────────────────────────────
DROP_COLS = ["Unnamed: 0", "CustomerID"]
TARGET    = "ProdTaken"

NUM_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CAT_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# ── STEP 1: LOAD ─────────────────────────────────────────────────────────────
print("Loading dataset from Hugging Face...")
# Download the tourism dataset from Hugging Face repository
local_path = hf_hub_download(
    repo_id=DATASET_REPO,  # Repository identifier defined elsewhere
    filename="tourism.csv", # Target file to download
    repo_type="dataset",    # Type of repository (dataset vs model)
    token=HF_TOKEN,        # Authentication token for accessing private repos
)
# Read the downloaded CSV file into a pandas DataFrame
df = pd.read_csv(local_path)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── STEP 2: CLEAN ─────────────────────────────────────────────────────────────
print("\nCleaning data...")
# Fix inconsistent gender labeling by replacing 'Fe Male' with 'Female'
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Standardize marital status terminology by replacing 'Unmarried' with 'Single'
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Remove columns that aren't needed for analysis (defined in DROP_COLS constant)
df = df.drop(columns=DROP_COLS, errors="ignore")
print(f"After cleaning: {df.shape[0]} rows × {df.shape[1]} columns")

# ── STEP 3: SPLIT ─────────────────────────────────────────────────────────────
# Separate features (X) from the target variable (y)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split data into training and testing sets with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE,       # Proportion of data to use for testing
    random_state=RANDOM_STATE, # Set seed for reproducibility
    stratify=y                 # Ensure class proportions are maintained in both splits
)
print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")
print(f"Train class distribution:\n{y_train.value_counts(normalize=True).round(3)}")

# ── STEP 4: BUILD PREPROCESSOR ───────────────────────────────────────────────
# Create a preprocessing pipeline using ColumnTransformer
# - Numeric features are standardized (mean=0, std=1)
# - Categorical features are one-hot encoded
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ]
)

# Apply preprocessing to train and test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

# Extract feature names after one-hot encoding
ohe_features = preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES).tolist()
all_features = NUM_FEATURES + ohe_features

# Convert processed arrays to DataFrames with proper column names
X_train_df = pd.DataFrame(X_train_processed, columns=all_features)
X_test_df  = pd.DataFrame(X_test_processed,  columns=all_features)
y_train_df = y_train.reset_index(drop=True)
y_test_df  = y_test.reset_index(drop=True)

# ── STEP 5: SAVE LOCALLY ──────────────────────────────────────────────────────
# Create directory for storing processed data
os.makedirs("tourism/data", exist_ok=True)

# Save processed datasets and preprocessor to local files
X_train_df.to_csv("tourism/data/X_train.csv", index=False)
X_test_df.to_csv("tourism/data/X_test.csv",   index=False)
y_train_df.to_csv("tourism/data/y_train.csv", index=False)
y_test_df.to_csv("tourism/data/y_test.csv",   index=False)
joblib.dump(preprocessor, "tourism/data/preprocessor.joblib")

print("\nSaved processed splits and preprocessor locally.")

# ── STEP 6: UPLOAD TO HUGGING FACE ───────────────────────────────────────────
# Initialize Hugging Face API with authentication token
api = HfApi(token=HF_TOKEN)
# Upload each file to the specified Hugging Face dataset repository
for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "preprocessor.joblib"]:
    api.upload_file(
        path_or_fileobj=f"tourism/data/{fname}",
        path_in_repo=fname,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {fname} → {DATASET_REPO}")

print("\nData preparation complete.")
