from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
from torchvision import transforms
import torchvision.models as models

import torch
import torch.nn as nn
import numpy as np
import joblib

from groq import Groq

import os
import io
import urllib.request


# ============================================================
# APP
# ============================================================

app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIGURATION
# ============================================================

device = torch.device("cpu")

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# HUGGING FACE MODEL URLs
# ============================================================

HF_BASE_URL = (
    "https://huggingface.co/varshu13/"
    "groundnut-leaf-disease-models/resolve/main/"
)

MODEL_URLS = {
    "convnext_model.pth":
        HF_BASE_URL + "convnext_model.pth",

    "efficientnet_model.pth":
        HF_BASE_URL + "efficientnet_model.pth",

    "ensemble_model.pkl":
        HF_BASE_URL + "ensemble_model.pkl",
}


# ============================================================
# DOWNLOAD MODEL FILES
# ============================================================

def download_model(filename):
    local_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(local_path):
        print(f"{filename} already exists.")
        return local_path

    url = MODEL_URLS[filename]

    print(f"Downloading {filename}...")
    print(f"From: {url}")

    try:
        urllib.request.urlretrieve(url, local_path)

        print(f"Downloaded {filename} successfully.")

        return local_path

    except Exception as e:
        print(f"Failed to download {filename}: {e}")

        if os.path.exists(local_path):
            os.remove(local_path)

        raise


# Download all required models
convnext_path = download_model("convnext_model.pth")
efficientnet_path = download_model("efficientnet_model.pth")
ensemble_path = download_model("ensemble_model.pkl")


# ============================================================
# GROQ SETUP
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq API configured.")
else:
    print("WARNING: GROQ_API_KEY is not set.")


# ============================================================
# CACHE
# ============================================================

disease_cache = {}


# ============================================================
# CLASSES
# ============================================================

classes = [
    "curl",
    "early_spot",
    "healthy",
    "late_spot",
    "rosette",
    "rust",
    "wormbite",
]


# ============================================================
# LOAD EFFICIENTNET
# ============================================================

print("Loading EfficientNetV2...")

eff_model = models.efficientnet_v2_s(weights=None)

num_features = eff_model.classifier[1].in_features

eff_model.classifier[1] = nn.Linear(
    num_features,
    7
)

eff_model.load_state_dict(
    torch.load(
        efficientnet_path,
        map_location=device
    )
)

eff_model.to(device)
eff_model.eval()

print("EfficientNetV2 loaded.")


# ============================================================
# LOAD CONVNEXT
# ============================================================

print("Loading ConvNeXt...")

conv_model = models.convnext_base(weights=None)

num_features = conv_model.classifier[2].in_features

conv_model.classifier[2] = nn.Linear(
    num_features,
    7
)

conv_model.load_state_dict(
    torch.load(
        convnext_path,
        map_location=device
    )
)

conv_model.to(device)
conv_model.eval()

print("ConvNeXt loaded.")


# ============================================================
# LOAD ENSEMBLE MODEL
# ============================================================

print("Loading ensemble model...")

meta_model = joblib.load(ensemble_path)

print("Ensemble model loaded.")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ============================================================
# GROQ SUGGESTIONS
# ============================================================

def get_groq_suggestions(disease: str) -> str:

    # Return cached result if available
    if disease in disease_cache:
        return disease_cache[disease]

    # If Groq API key is not configured
    if groq_client is None:
        return (
            "AI suggestions are currently unavailable. "
            "Please configure the GROQ_API_KEY."
        )

    prompt = f"""
A crop has been diagnosed with: {disease}

Please provide the following in simple language for farmers:

- Cause: Why does this disease occur?
- Prevention: How to prevent it?
- Treatment: How to treat it?
- Advice: Any additional tips?

Keep it short and easy to understand.
"""

    try:

        response = groq_client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an agricultural expert helping "
                        "farmers understand crop diseases."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            max_tokens=512,
            temperature=0.7,
        )

        result = response.choices[0].message.content

        disease_cache[disease] = result

        return result

    except Exception as e:

        print(f"Groq error: {e}")

        return (
            "Unable to generate AI suggestions at the moment."
        )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Groundnut Leaf Disease Recognition API",
        "status": "running",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read uploaded image
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Transform image
        img = transform(image).unsqueeze(0)

        img = img.to(device)

        # Model prediction
        with torch.no_grad():

            eff_out = torch.softmax(
                eff_model(img),
                dim=1
            ).cpu().numpy()

            conv_out = torch.softmax(
                conv_model(img),
                dim=1
            ).cpu().numpy()

        # Combine model outputs
        X_meta = np.concatenate(
            [eff_out, conv_out],
            axis=1
        )

        # Ensemble prediction
        pred = meta_model.predict(X_meta)[0]

        # Convert prediction to Python integer
        pred = int(pred)

        # Confidence
        confidence = float(
            np.max(X_meta)
        )

        # Disease name
        disease_name = classes[pred]

        # Groq suggestions
        suggestion_text = get_groq_suggestions(
            disease_name
        )

        return {
            "disease": disease_name,
            "confidence": round(
                confidence * 100,
                2
            ),
            "suggestions": suggestion_text,
        }

    except Exception as e:

        print(f"Prediction error: {e}")

        return {
            "error": str(e)
        }


# ============================================================
# STARTUP MESSAGE
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("----------------------------------------")
    print("Groundnut Leaf Disease API started")
    print("Models loaded successfully")
    print("----------------------------------------")