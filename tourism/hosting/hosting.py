"""hosting.py
Uploads the deployment folder (Dockerfile, app.py, requirements.txt) to a
Hugging Face Space running the app via Docker SDK.
Triggered as the fourth job (deploy) in the GitHub Actions pipeline.
"""
from huggingface_hub import HfApi, create_repo
import os
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
HF_USERNAME = os.getenv("HF_USERNAME", "vivekkumar-hf")
SPACE_REPO  = f"{HF_USERNAME}/tourism_predictor"
HF_TOKEN    = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# ── DELETE SPACE IF IT EXISTS (avoids 400 from repo_info on broken spaces) ────
# Always attempt delete; ignore all errors (space may not exist yet).
try:
    api.delete_repo(repo_id=SPACE_REPO, repo_type="space")
    print(f"Deleted existing Space '{SPACE_REPO}'. Waiting for HF to settle...")
    time.sleep(8)
except Exception as e:
    print(f"Space not found or could not be deleted (will create fresh): {e}")

# ── CREATE SPACE FRESH ────────────────────────────────────────────────────────
create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    space_sdk="docker",
    private=False,
    token=HF_TOKEN,
)
print(f"Created Hugging Face Space: {SPACE_REPO}")

# ── UPLOAD DEPLOYMENT FOLDER ──────────────────────────────────────────────────
api.upload_folder(
    folder_path="tourism/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    commit_message="Deploy tourism wellness app",
)
print(f"\nDeployment files uploaded to Space: {SPACE_REPO}")
print(f"App will be live at: https://huggingface.co/spaces/{SPACE_REPO}")
