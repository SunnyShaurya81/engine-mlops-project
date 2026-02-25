from huggingface_hub import HfApi, login
import os

# Get HF_TOKEN from environment variables
hf_token = os.environ.get("HF_TOKEN")

if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running the script.")

login(token=hf_token)

api = HfApi()

repo_id = "SunnyShaurya1981/engine-condition-app2"

files = [
    "app.py",
    "best_model.pkl",
    "requirements.txt",
    "Dockerfile"
]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=repo_id,
        repo_type="space"
    )

print("✅ Deployment files uploaded successfully!")
