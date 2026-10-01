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
import timm
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

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# HUGGING FACE
# ============================================================

HF_BASE_URL = (
    "https://huggingface.co/varshu13/"
    "groundnut-leaf-disease-models/resolve/main/"
)


# ============================================================
# CURRENT MODEL FILENAMES
# ============================================================

efficientnet_path = os.path.join(
    MODEL_DIR,
    "efficientnet_v2_b0_model.pth"
)

convnext_path = os.path.join(
    MODEL_DIR,
    "convnext_tiny_model.pth"
)

ensemble_path = os.path.join(
    MODEL_DIR,
    "ensemble_b0_tiny_model.pkl"
)


# ============================================================
# DOWNLOAD MODEL IF MISSING
# ============================================================

def download_model_if_missing(
    file_path,
    file_name
):

    if os.path.exists(file_path):

        print(
            f"{file_name} already exists."
        )

        return

    print(
        f"Downloading {file_name} "
        "from Hugging Face..."
    )

    url = HF_BASE_URL + file_name

    urllib.request.urlretrieve(
        url,
        file_path
    )

    print(
        f"{file_name} downloaded successfully."
    )


# ============================================================
# DOWNLOAD CURRENT MODELS
# ============================================================

download_model_if_missing(
    efficientnet_path,
    "efficientnet_v2_b0_model.pth"
)

download_model_if_missing(
    convnext_path,
    "convnext_tiny_model.pth"
)

download_model_if_missing(
    ensemble_path,
    "ensemble_b0_tiny_model.pkl"
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
# LOAD EFFICIENTNET V2 B0
# ============================================================

def load_efficientnet():

    print(
        "Loading EfficientNetV2-B0 using timm..."
    )

    model = timm.create_model(
        "tf_efficientnetv2_b0",
        pretrained=False,
        num_classes=7
    )

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
        "EfficientNetV2-B0 loaded successfully."
    )

    return model


# ============================================================
# LOAD CONVNEXT TINY
# ============================================================

def load_convnext():

    print(
        "Loading ConvNeXt Tiny..."
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
        "ConvNeXt Tiny loaded."
    )

    return model


# ============================================================
# ENSEMBLE
# ============================================================

# IMPORTANT:
# Do NOT load the Logistic Regression model here.
# It is loaded only during prediction to reduce
# Render startup memory usage.

meta_model = None


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
        # EFFICIENTNET V2 B0
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
        # CONVNEXT TINY
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

        del image

        gc.collect()


        # ====================================================
        # COMBINE CNN OUTPUTS
        # ====================================================

        X_meta = np.concatenate(
            [
                eff_output,
                conv_output
            ],
            axis=1
        )


        # ====================================================
        # LOAD ENSEMBLE ONLY WHEN NEEDED
        # ====================================================

        print(
            "Loading B0 + ConvNeXt Tiny ensemble..."
        )

        meta_model = joblib.load(
            ensemble_path
        )

        print(
            "B0 + ConvNeXt Tiny ensemble loaded."
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
        # ENSEMBLE CONFIDENCE
        # ====================================================

        try:

            ensemble_probabilities = (
                meta_model
                .predict_proba(
                    X_meta
                )[0]
            )

            confidence = float(
                np.max(
                    ensemble_probabilities
                )
            )

        except Exception:

            # Fallback if the loaded
            # meta-model does not support
            # predict_proba.

            confidence = float(
                max(
                    np.max(eff_output),
                    np.max(conv_output)
                )
            )


        # ====================================================
        # UNLOAD ENSEMBLE
        # ====================================================

        del meta_model

        gc.collect()


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

        # TEMPORARY DEBUG FIELD
        # Remove "error" after everything works.

        return {

            "disease":
                "Unknown",

            "confidence":
                0,

            "suggestions":
                "Unable to process image.",

            "error":
                str(e)
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

