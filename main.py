import os
import io
import urllib.request

# Limit CPU/thread memory usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

import numpy as np
import joblib

from groq import Groq


# ============================================================
# FASTAPI
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
# DEVICE
# ============================================================

device = torch.device("cpu")

# Prevent PyTorch from creating many CPU threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# HUGGING FACE
# ============================================================

HF_BASE_URL = (
    "https://huggingface.co/varshu13/"
    "groundnut-leaf-disease-models/resolve/main/"
)

MODEL_URLS = {
    "efficientnet_model.pth":
        HF_BASE_URL + "efficientnet_model.pth",

    "convnext_model.pth":
        HF_BASE_URL + "convnext_model.pth",

    "ensemble_model.pkl":
        HF_BASE_URL + "ensemble_model.pkl",
}


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def download_model(filename):

    local_path = os.path.join(
        MODEL_DIR,
        filename
    )

    if os.path.exists(local_path):
        print(
            f"{filename} already exists."
        )
        return local_path

    url = MODEL_URLS[filename]

    print(
        f"Downloading {filename}..."
    )

    try:

        urllib.request.urlretrieve(
            url,
            local_path
        )

        print(
            f"{filename} downloaded."
        )

        return local_path

    except Exception as e:

        print(
            f"Error downloading {filename}: {e}"
        )

        if os.path.exists(local_path):
            os.remove(local_path)

        raise


# ============================================================
# DOWNLOAD MODELS
# ============================================================

efficientnet_path = download_model(
    "efficientnet_model.pth"
)

convnext_path = download_model(
    "convnext_model.pth"
)

ensemble_path = download_model(
    "ensemble_model.pkl"
)


# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

groq_client = None

if GROQ_API_KEY:

    groq_client = Groq(
        api_key=GROQ_API_KEY
    )

    print(
        "Groq API configured."
    )

else:

    print(
        "WARNING: GROQ_API_KEY is not set."
    )


# ============================================================
# DISEASE CLASSES
# ============================================================

classes = [
    "curl",
    "early_spot",
    "healthy",
    "late_spot",
    "rosette",
    "rust",
    "wormbite"
]


# ============================================================
# LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = [
    "en",
    "kn",
    "hi"
]


# ============================================================
# CACHE
# ============================================================

disease_cache = {}


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor()
])


# ============================================================
# LOAD EFFICIENTNET
# ============================================================

print(
    "Loading EfficientNetV2..."
)

eff_model = models.efficientnet_v2_s(
    weights=None
)

eff_features = (
    eff_model
    .classifier[1]
    .in_features
)

eff_model.classifier[1] = nn.Linear(
    eff_features,
    7
)

# Load weights
eff_state = torch.load(
    efficientnet_path,
    map_location="cpu"
)

eff_model.load_state_dict(
    eff_state
)

# Free temporary state dictionary
del eff_state

eff_model.to(device)
eff_model.eval()

print(
    "EfficientNetV2 loaded."
)


# ============================================================
# LOAD CONVNEXT
# ============================================================

print(
    "Loading ConvNeXt..."
)

conv_model = models.convnext_base(
    weights=None
)

conv_features = (
    conv_model
    .classifier[2]
    .in_features
)

conv_model.classifier[2] = nn.Linear(
    conv_features,
    7
)

# Load weights
conv_state = torch.load(
    convnext_path,
    map_location="cpu"
)

conv_model.load_state_dict(
    conv_state
)

# Free temporary state dictionary
del conv_state

conv_model.to(device)
conv_model.eval()

print(
    "ConvNeXt loaded."
)


# ============================================================
# LOAD ENSEMBLE
# ============================================================

print(
    "Loading ensemble model..."
)

meta_model = joblib.load(
    ensemble_path
)

print(
    "Ensemble model loaded."
)


# ============================================================
# GROQ SUGGESTIONS
# ============================================================

def get_groq_suggestions(
    disease,
    language="en"
):

    if language not in SUPPORTED_LANGUAGES:
        language = "en"

    cache_key = (
        disease,
        language
    )

    if cache_key in disease_cache:

        return disease_cache[
            cache_key
        ]

    if language == "kn":

        language_instruction = """
Respond completely in Kannada (ಕನ್ನಡ).

Use Kannada script for the explanation.

Keep these headings EXACTLY:

Cause:
Prevention:
Treatment:
Advice:

Only the text after the headings should be in Kannada.
"""

    elif language == "hi":

        language_instruction = """
Respond completely in Hindi using Devanagari script.

Use Hindi for the explanation.

Keep these headings EXACTLY:

Cause:
Prevention:
Treatment:
Advice:

Only the text after the headings should be in Hindi.
"""

    else:

        language_instruction = """
Respond completely in simple English.

Keep these headings EXACTLY:

Cause:
Prevention:
Treatment:
Advice:
"""

    prompt = f"""
A groundnut crop has been diagnosed with:

{disease}

{language_instruction}

Follow this exact format:

Cause: [short explanation]

Prevention: [short explanation]

Treatment: [short explanation]

Advice: [short explanation]

Rules:

- Keep the explanation simple.
- Use language farmers can understand.
- Do not use markdown.
- Do not add extra headings.
- Keep each section short.
"""

    if groq_client is None:

        return (
            "Cause: Information unavailable.\n"
            "Prevention: Information unavailable.\n"
            "Treatment: Information unavailable.\n"
            "Advice: Information unavailable."
        )

    try:

        response = (
            groq_client
            .chat
            .completions
            .create(
                model="openai/gpt-oss-20b",

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an agricultural "
                            "expert helping "
                            "groundnut farmers."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                max_tokens=400,
                temperature=0.2
            )
        )

        result = (
            response
            .choices[0]
            .message
            .content
        )

        if not result:
            return (
                "Cause: Information unavailable.\n"
                "Prevention: Information unavailable.\n"
                "Treatment: Information unavailable.\n"
                "Advice: Information unavailable."
            )

        result = result.strip()

        disease_cache[
            cache_key
        ] = result

        return result

    except Exception as e:

        print(
            "Groq error:",
            str(e)
        )

        return (
            "Cause: Information unavailable.\n"
            "Prevention: Information unavailable.\n"
            "Treatment: Information unavailable.\n"
            "Advice: Information unavailable."
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "Groundnut Leaf Disease Recognition API",
        "status":
            "running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    language: str = Query(
        default="en"
    )
):

    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

        # Release original bytes
        del image_bytes

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        image_tensor = transform(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            device
        )

        # ----------------------------------------------------
        # EfficientNet
        # ----------------------------------------------------

        with torch.inference_mode():

            eff_output = eff_model(
                image_tensor
            )

            eff_output = torch.softmax(
                eff_output,
                dim=1
            )

            eff_output = (
                eff_output
                .cpu()
                .numpy()
            )

        # ----------------------------------------------------
        # ConvNeXt
        # ----------------------------------------------------

        with torch.inference_mode():

            conv_output = conv_model(
                image_tensor
            )

            conv_output = torch.softmax(
                conv_output,
                dim=1
            )

            conv_output = (
                conv_output
                .cpu()
                .numpy()
            )

        # Release tensor
        del image_tensor

        # ----------------------------------------------------
        # Meta features
        # ----------------------------------------------------

        X_meta = np.concatenate(
            [
                eff_output,
                conv_output
            ],
            axis=1
        )

        # ----------------------------------------------------
        # Ensemble prediction
        # ----------------------------------------------------

        pred = meta_model.predict(
            X_meta
        )[0]

        pred = int(pred)

        disease_name = classes[
            pred
        ]

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            np.max(X_meta)
        )

        # ----------------------------------------------------
        # AI suggestions
        # ----------------------------------------------------

        suggestions = (
            get_groq_suggestions(
                disease_name,
                language
            )
        )

        # Release temporary arrays
        del eff_output
        del conv_output
        del X_meta

        return {

            "disease":
                disease_name,

            "confidence":
                round(
                    confidence * 100,
                    2
                ),

            "suggestions":
                suggestions
        }

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return {

            "disease":
                "Unknown",

            "confidence":
                0,

            "suggestions":
                "Unable to process image."
        }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )