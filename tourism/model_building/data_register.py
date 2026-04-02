"""data_register.py
Creates a Hugging Face dataset repository and uploads tourism.csv to it.
Triggered as the first job in the GitHub Actions CI/CD pipeline.
"""
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Define configuration variables for the Hugging Face repository
HF_USERNAME = os.getenv("HF_USERNAME", "vivekkumar-hf")  # Get username from env var or use default
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-data"  # Full repository ID
REPO_TYPE = "dataset"  # Type of repository (dataset vs model)
DATA_FILE = "tourism.csv"  # Local file to be uploaded

# ── INITIALISE API ─────────────────────────────────────────────────────────────
# Create API client with authentication token from environment variables
os.environ["HF_TOKEN"] = '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'
api = HfApi(token=os.getenv("HF_TOKEN"))
print(os.getenv("HF_TOKEN"))

# ── CREATE REPO IF NEEDED ─────────────────────────────────────────────────────
try:
    # Check if repository already exists
    api.repo_info(repo_id=DATASET_REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{DATASET_REPO_ID}' already exists.")
except RepositoryNotFoundError:
    # Create new repository if it doesn't exist
    create_repo(repo_id=DATASET_REPO_ID, repo_type=REPO_TYPE,
                private=False, token=os.getenv("HF_TOKEN"))
    print(f"Created dataset repo: {DATASET_REPO_ID}")

# ── UPLOAD DATA ───────────────────────────────────────────────────────────────
# Upload the tourism data file to the Hugging Face repository
api.upload_file(
    path_or_fileobj=DATA_FILE,  # Local file to upload
    path_in_repo="tourism.csv",  # Destination path in the repository
    repo_id=DATASET_REPO_ID,    # Target repository
    repo_type=REPO_TYPE,        # Repository type (dataset)
)
print(f"Uploaded '{DATA_FILE}' to '{DATASET_REPO_ID}'.")
