from huggingface_hub import HfApi
import os
from pathlib import Path

# Set your model details
model_path = r"C:\Users\HP\Documents\Hackathon\whisper-small_Akan_non_standardspeech\whisper-small_Akan_non_standardspeech"
repo_id = "Saintdannyyy/kasayie-asr"
commit_message = "Upload whisper-small model fine-tuned for Akan non-standard speech"

# Initialize the Hugging Face API
api = HfApi()

# Create the repository (set private=False if you want a public repository)
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

# Upload the model files
api.upload_folder(
    folder_path=model_path,
    repo_id=repo_id,
    repo_type="model",
    commit_message=commit_message
)

print(f"Model successfully uploaded to {repo_id}")