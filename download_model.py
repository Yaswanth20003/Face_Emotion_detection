import gdown
import os

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# File path where the model will be saved
output = "static/model.h5"

# Google Drive shareable link (this is your model link)
url = "https://drive.google.com/uc?id=1MPql4BPEmMBw9y0XJP66kk8SFxaTxG7j"

# Download only if the file doesn't exist
if not os.path.exists(output):
    print("Downloading model.h5 from Google Drive...")
    gdown.download(url, output, quiet=False)
else:
    print("model.h5 already exists. Skipping download.")
