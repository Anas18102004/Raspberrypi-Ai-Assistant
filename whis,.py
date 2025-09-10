import os
import requests

# Set up paths
home_dir = os.path.expanduser("~")
model_dir = os.path.join(home_dir, "whisper", "models")
os.makedirs(model_dir, exist_ok=True)

# URL for the base Whisper model
model_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
model_path = os.path.join(model_dir, "ggml-base.bin")  # Keep filename consistent

# Download the model
print(f"Downloading model to {model_path} ...")
response = requests.get(model_url, stream=True)

if response.status_code == 200:
    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
            f.write(chunk)
    print("✅ Model downloaded successfully!")
else:
    print(f"❌ Failed to download model, status code: {response.status_code}")
