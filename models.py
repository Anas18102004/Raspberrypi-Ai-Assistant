import os
import requests

# ---------------- MODEL DOWNLOAD CONFIG ---------------- #
HOME_DIR = os.path.expanduser("~")

# Directories for models
WHISPER_DIR = os.path.join(HOME_DIR, "whisper", "models")
TINYLLAMA_DIR = os.path.join(HOME_DIR, "tinyllama", "models")
os.makedirs(WHISPER_DIR, exist_ok=True)
os.makedirs(TINYLLAMA_DIR, exist_ok=True)

# ---------------- MODEL URLS ---------------- #
WHISPER_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
WHISPER_MODEL_PATH = os.path.join(WHISPER_DIR, "ggml-base.bin")

TINYLLAMA_MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
TINYLLAMA_MODEL_PATH = os.path.join(TINYLLAMA_DIR, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Hugging Face Token
HF_TOKEN = 
# ---------------- HELPER FUNCTION ---------------- #
def download_model(url, path, token=None):
    if os.path.exists(path):
        print(f"✅ Model already exists: {path}")
        return
    print(f"⬇️ Downloading model from {url} ...")
    
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    done = int(50 * downloaded / total_size) if total_size else 0
                    print(f"\r[{'█' * done}{'.' * (50-done)}] "
                          f"{downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB", end="")
        print(f"\n✅ Download complete: {path}")
    else:
        print(f"❌ Failed to download {url} - status code: {response.status_code}")

# ---------------- DOWNLOAD MODELS ---------------- #
download_model(WHISPER_MODEL_URL, WHISPER_MODEL_PATH)  # no token needed
download_model(TINYLLAMA_MODEL_URL, TINYLLAMA_MODEL_PATH, token=HF_TOKEN)  # token needed

print("\nAll models are ready!")
