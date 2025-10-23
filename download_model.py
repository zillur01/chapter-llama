from huggingface_hub import snapshot_download
import os

# Define the model ID and the local directory to save the model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
local_dir = "./checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct"

print(f"Starting download of model '{model_id}' to '{local_dir}'...")
try:
    # Download the model
    snapshot_download(repo_id=model_id, local_dir=local_dir)
    print("Download complete!")
except Exception as e:
    print(f"An error occurred during download: {e}")
    print("Please ensure you have been granted access to the model on Hugging Face Hub.")

