"""hosting.py
Uploads the deployment folder (Dockerfile, app.py, requirements.txt) to a
Hugging Face Space running the app via Docker SDK.
Triggered as the fourth job (deploy) in the GitHub Actions pipeline.
"""
from huggingface_hub import HfApi, create_repo
import os
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Set up Hugging Face username and repository information
# Uses environment variables or defaults to "vivekkumar-hf" if not provided
HF_USERNAME = os.getenv("HF_USERNAME", "vivekkumar-hf")
SPACE_REPO  = f"{HF_USERNAME}/tourism_predictor"
HF_TOKEN    = os.getenv("HF_TOKEN")  # Authentication token for Hugging Face API

# Initialize the Hugging Face API client with our authentication token
api = HfApi(token=HF_TOKEN)

# ── DELETE SPACE IF IT EXISTS (avoids 400 from repo_info on broken spaces) ────
# Always attempt delete; ignore all errors (space may not exist yet).
try:
    # Delete the existing space if it exists to start fresh
    api.delete_repo(repo_id=SPACE_REPO, repo_type="space")
    print(f"Deleted existing Space '{SPACE_REPO}'. Waiting for HF to settle...")
    time.sleep(8)  # Wait for Hugging Face servers to process the deletion
except Exception as e:
    print(f"Space not found or could not be deleted (will create fresh): {e}")

# ── CREATE SPACE FRESH ────────────────────────────────────────────────────────
# Create a new Hugging Face Space with Docker SDK configuration
create_repo(
    repo_id=SPACE_REPO,      # The name of the repository
    repo_type="space",       # Type is "space" for Hugging Face Spaces
    space_sdk="docker",      # Use Docker for deployment
    private=False,           # Make the space publicly accessible
    token=HF_TOKEN,          # Authentication token
)
print(f"Created Hugging Face Space: {SPACE_REPO}")

# ── UPLOAD DEPLOYMENT FOLDER ──────────────────────────────────────────────────
# Upload all files from the deployment folder to the Hugging Face Space
api.upload_folder(
    folder_path="tourism/deployment",  # Local folder containing deployment files
    repo_id=SPACE_REPO,               # Target repository
    repo_type="space",                # Type is "space" for Hugging Face Spaces
    commit_message="Deploy tourism wellness app",  # Git commit message
)
print(f"\nDeployment files uploaded to Space: {SPACE_REPO}")
print(f"App will be live at: https://huggingface.co/spaces/{SPACE_REPO}")
