# ============================================================
# main.py
# Groundnut Leaf Disease Recognition API
# ============================================================

import os
import io
import gc
import urllib.request
import threading

# ============================================================
# MEMORY / CPU OPTIMIZATION
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ============================================================
# IMPORTS
# ============================================================

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Query,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from groq import Groq


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Groundnut Leaf Disease Recognition API",
    version="1.0"
)


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
# GLOBAL MODEL
# ============================================================

# The model is loaded ONCE when the server starts.
# It is NOT loaded/deleted for every prediction.

model = None

# Prevent multiple CPU inference requests from running
# simultaneously.

model_lock = threading.Lock()


# ============================================================
# MODEL DIRECTORY
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
# EFFICIENTNET V2-S MODEL
# ============================================================

efficientnet_filename = "efficientnet_model.pth"

efficientnet_path = os.path.join(
    MODEL_DIR,
    efficientnet_filename
)


# ============================================================
# DOWNLOAD MODEL IF MISSING
# ============================================================

def download_model_if_missing():

    if os.path.exists(efficientnet_path):

        print(
            "EfficientNetV2-S model already exists."
        )

        return

    print(
        "Downloading EfficientNetV2-S model "
        "from Hugging Face..."
    )

    url = (
        HF_BASE_URL +
        efficientnet_filename
    )

    try:

        urllib.request.urlretrieve(
            url,
            efficientnet_path
        )

        print(
            "EfficientNetV2-S model "
            "downloaded successfully."
        )

    except Exception as e:

        print(
            "Error downloading model:",
            str(e)
        )

        if os.path.exists(
            efficientnet_path
        ):

            os.remove(
                efficientnet_path
            )

        raise


# ============================================================
# DOWNLOAD MODEL
# ============================================================

download_model_if_missing()


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
# SUPPORTED LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = [
    "en",
    "kn",
    "hi"
]


# ============================================================
# GROQ CACHE
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
# LOAD EFFICIENTNET V2-S
# ============================================================

def load_efficientnet():

    print(
        "Loading EfficientNetV2-S..."
    )

    # --------------------------------------------------------
    # CREATE MODEL ARCHITECTURE
    # --------------------------------------------------------

    loaded_model = models.efficientnet_v2_s(
        weights=None
    )

    # --------------------------------------------------------
    # CHANGE CLASSIFIER
    # --------------------------------------------------------

    num_features = (
        loaded_model
        .classifier[1]
        .in_features
    )

    loaded_model.classifier[1] = nn.Linear(
        num_features,
        7
    )

    # --------------------------------------------------------
    # LOAD TRAINED WEIGHTS
    # --------------------------------------------------------

    try:

        state = torch.load(
            efficientnet_path,
            map_location="cpu",
            weights_only=True,
            mmap=True
        )

    except TypeError:

        # Compatibility with older PyTorch

        state = torch.load(
            efficientnet_path,
            map_location="cpu"
        )

    # --------------------------------------------------------
    # LOAD STATE DICT
    # --------------------------------------------------------

    try:

        loaded_model.load_state_dict(
            state,
            assign=True
        )

    except TypeError:

        loaded_model.load_state_dict(
            state
        )

    # --------------------------------------------------------
    # RELEASE CHECKPOINT MEMORY
    # --------------------------------------------------------

    del state

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    loaded_model.to(device)

    # --------------------------------------------------------
    # EVALUATION MODE
    # --------------------------------------------------------

    loaded_model.eval()

    print(
        "EfficientNetV2-S loaded successfully."
    )

    return loaded_model


# ============================================================
# FASTAPI STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global model

    print(
        "============================================"
    )

    print(
        "Starting Groundnut Disease API..."
    )

    print(
        "============================================"
    )

    print(
        "Initializing EfficientNetV2-S model..."
    )

    model = load_efficientnet()

    print(
        "Model ready for predictions."
    )

    print(
        "============================================"
    )


# ============================================================
# GROQ SUGGESTIONS
# ============================================================

def get_groq_suggestions(
    disease,
    language="en"
):

    # --------------------------------------------------------
    # VALIDATE LANGUAGE
    # --------------------------------------------------------

    if language not in SUPPORTED_LANGUAGES:

        language = "en"


    # --------------------------------------------------------
    # CACHE KEY
    # --------------------------------------------------------

    cache_key = (
        disease,
        language
    )


    # --------------------------------------------------------
    # CHECK CACHE
    # --------------------------------------------------------

    if cache_key in disease_cache:

        print(
            f"Using cached Groq response: "
            f"{disease} / {language}"
        )

        return disease_cache[
            cache_key
        ]


    # --------------------------------------------------------
    # LANGUAGE INSTRUCTIONS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # GROQ NOT CONFIGURED
    # --------------------------------------------------------

    if groq_client is None:

        return (
            "Cause: Information unavailable.\n"
            "Prevention: Information unavailable.\n"
            "Treatment: Information unavailable.\n"
            "Advice: Information unavailable."
        )


    # --------------------------------------------------------
    # CALL GROQ
    # --------------------------------------------------------

    try:

        print(
            f"Calling Groq for "
            f"{disease} / {language}..."
        )

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


        # ----------------------------------------------------
        # CACHE SUCCESSFUL RESPONSE
        # ----------------------------------------------------

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
            "running",

        "model":
            "EfficientNetV2-S"

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "model":
            "EfficientNetV2-S",

        "model_loaded":
            model is not None

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

    global model

    image = None

    image_tensor = None

    try:

        # ====================================================
        # CHECK MODEL
        # ====================================================

        if model is None:

            raise Exception(
                "Model is not loaded."
            )


        # ====================================================
        # VALIDATE LANGUAGE
        # ====================================================

        if language not in SUPPORTED_LANGUAGES:

            language = "en"


        print(
            "--------------------------------------------"
        )

        print(
            "New prediction request"
        )

        print(
            f"Language: {language}"
        )

        print(
            f"Filename: {file.filename}"
        )


        # ====================================================
        # READ IMAGE
        # ====================================================

        image_bytes = await file.read()

        if not image_bytes:

            raise Exception(
                "Uploaded file is empty."
            )


        print(
            f"Image size: "
            f"{len(image_bytes)} bytes"
        )


        # ====================================================
        # OPEN IMAGE
        # ====================================================

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")


        # Release raw bytes

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
        # MODEL PREDICTION
        # ====================================================

        print(
            "Running EfficientNetV2-S inference..."
        )


        with model_lock:

            with torch.inference_mode():

                output = model(
                    image_tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1
                )[0]


                predicted_class = torch.argmax(
                    probabilities
                ).item()


                confidence = probabilities[
                    predicted_class
                ].item()


        # ====================================================
        # DISEASE NAME
        # ====================================================

        disease_name = classes[
            predicted_class
        ]


        print(
            f"Prediction: {disease_name}"
        )

        print(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )


        # ====================================================
        # RELEASE IMAGE MEMORY
        # ====================================================

        del image_tensor

        image_tensor = None

        del image

        image = None

        gc.collect()


        # ====================================================
        # GROQ SUGGESTIONS
        # ====================================================

        print(
            "Getting disease information..."
        )


        suggestions = get_groq_suggestions(
            disease_name,
            language
        )


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        print(
            "Prediction request completed."
        )

        print(
            "--------------------------------------------"
        )


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


        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        if image_tensor is not None:

            del image_tensor

        if image is not None:

            del image

        gc.collect()


        # ----------------------------------------------------
        # RETURN REAL HTTP ERROR
        # ----------------------------------------------------

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# RUN SERVER LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8000

    )