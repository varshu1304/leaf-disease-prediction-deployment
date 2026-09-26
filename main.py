import os
import io
import gc
import urllib.request

# ============================================================
# MEMORY / CPU OPTIMIZATION
# ============================================================

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

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hugging Face repository
HF_BASE_URL = (
    "https://huggingface.co/varshu13/"
    "groundnut-leaf-disease-models/resolve/main/"
)

efficientnet_path = os.path.join(
    MODEL_DIR,
    "efficientnet_model.pth"
)

convnext_path = os.path.join(
    MODEL_DIR,
    "convnext_tiny_model.pth"
)

ensemble_path = os.path.join(
    MODEL_DIR,
    "ensemble_tiny_model.pkl"
)


def download_model_if_missing(file_path, file_name):
    if os.path.exists(file_path):
        print(f"{file_name} already exists.")
        return

    print(f"Downloading {file_name} from Hugging Face...")

    url = HF_BASE_URL + file_name

    urllib.request.urlretrieve(
        url,
        file_path
    )

    print(f"{file_name} downloaded successfully.")


# Download models only when they are missing
download_model_if_missing(
    efficientnet_path,
    "efficientnet_model.pth"
)

download_model_if_missing(
    convnext_path,
    "convnext_tiny_model.pth"
)

download_model_if_missing(
    ensemble_path,
    "ensemble_tiny_model.pkl"
)

# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
# LOAD EFFICIENTNET ONLY WHEN NEEDED
# ============================================================

def load_efficientnet():

    print(
        "Loading EfficientNetV2..."
    )

    model = models.efficientnet_v2_s(
        weights=None
    )

    features = (
        model
        .classifier[1]
        .in_features
    )

    model.classifier[1] = nn.Linear(
        features,
        7
    )

    # mmap=True helps reduce peak memory while
    # loading a large PyTorch checkpoint.
    try:

        state = torch.load(
            efficientnet_path,
            map_location="cpu",
            weights_only=True,
            mmap=True
        )

    except TypeError:

        state = torch.load(
            efficientnet_path,
            map_location="cpu"
        )

    # assign=True avoids an unnecessary
    # parameter copy when supported.
    try:

        model.load_state_dict(
            state,
            assign=True
        )

    except TypeError:

        model.load_state_dict(
            state
        )

    del state

    model.to(device)
    model.eval()

    print(
        "EfficientNetV2 loaded."
    )

    return model


# ============================================================
# LOAD CONVNEXT ONLY WHEN NEEDED
# ============================================================

def load_convnext():

    print(
        "Loading ConvNeXt..."
    )

    model = models.convnext_tiny(
        weights=None
    )

    features = (
        model
        .classifier[2]
        .in_features
    )

    model.classifier[2] = nn.Linear(
        features,
        7
    )

    try:

        state = torch.load(
            convnext_path,
            map_location="cpu",
            weights_only=True,
            mmap=True
        )

    except TypeError:

        state = torch.load(
            convnext_path,
            map_location="cpu"
        )

    try:

        model.load_state_dict(
            state,
            assign=True
        )

    except TypeError:

        model.load_state_dict(
            state
        )

    del state

    model.to(device)
    model.eval()

    print(
        "ConvNeXt loaded."
    )

    return model


# ============================================================
# LOAD ENSEMBLE MODEL
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

        # ====================================================
        # READ IMAGE
        # ====================================================

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

        del image_bytes

        # ====================================================
        # TRANSFORM IMAGE
        # ====================================================

        image_tensor = transform(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            device
        )

        # ====================================================
        # EFFICIENTNET
        # ====================================================

        eff_model = load_efficientnet()

        with torch.inference_mode():

            eff_output = eff_model(
                image_tensor
            )

            eff_output = torch.softmax(
                eff_output,
                dim=1
            )

            # Convert immediately to small
            # NumPy array.
            eff_output = (
                eff_output
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

        # ====================================================
        # UNLOAD EFFICIENTNET
        # ====================================================

        del eff_model

        gc.collect()

        # ====================================================
        # CONVNEXT
        # ====================================================

        conv_model = load_convnext()

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
                .astype(
                    np.float32
                )
            )

        # ====================================================
        # UNLOAD CONVNEXT
        # ====================================================

        del conv_model

        gc.collect()

        # ====================================================
        # RELEASE IMAGE TENSOR
        # ====================================================

        del image_tensor

        gc.collect()

        # ====================================================
        # COMBINE MODEL OUTPUTS
        # ====================================================

        X_meta = np.concatenate(
            [
                eff_output,
                conv_output
            ],
            axis=1
        )

        # ====================================================
        # ENSEMBLE PREDICTION
        # ====================================================

        pred = meta_model.predict(
            X_meta
        )[0]

        pred = int(pred)

        disease_name = classes[
            pred
        ]

        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = float(
            np.max(X_meta)
        )

        # ====================================================
        # GROQ SUGGESTIONS
        # ====================================================

        suggestions = (
            get_groq_suggestions(
                disease_name,
                language
            )
        )

        # ====================================================
        # RELEASE ARRAYS
        # ====================================================

        del eff_output
        del conv_output
        del X_meta

        gc.collect()

        # ====================================================
        # RESPONSE
        # ====================================================

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

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )