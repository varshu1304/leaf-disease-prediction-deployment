# ============================================================
# GROUNDNUT LEAF DISEASE RECOGNITION - FASTAPI BACKEND
# ============================================================

import os
import io
import gc
import urllib.request
import threading

# ============================================================
# LIMIT CPU THREADS
# Important for Render CPU deployment
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
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Groundnut Leaf Disease Recognition API",
    version="1.0",
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
# MODEL VARIABLES
# ============================================================

model = None

# Lock prevents multiple CPU predictions from running
# at exactly the same time.
model_lock = threading.Lock()


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)


# ============================================================
# HUGGING FACE MODEL URL
# ============================================================

HF_BASE_URL = (
    "https://huggingface.co/varshu13/"
    "groundnut-leaf-disease-models/resolve/main/"
)


# ============================================================
# MODEL FILE
# ============================================================

efficientnet_filename = "efficientnet_model.pth"

efficientnet_path = os.path.join(
    MODEL_DIR,
    efficientnet_filename,
)


# ============================================================
# DOWNLOAD MODEL IF NOT PRESENT
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
        HF_BASE_URL
        + efficientnet_filename
    )

    try:

        urllib.request.urlretrieve(
            url,
            efficientnet_path,
        )

        print(
            "EfficientNetV2-S model "
            "downloaded successfully."
        )

    except Exception as e:

        print(
            "Error downloading model:",
            str(e),
        )

        if os.path.exists(efficientnet_path):

            os.remove(efficientnet_path)

        raise


# Download before loading
download_model_if_missing()


# ============================================================
# GROQ CONFIGURATION
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
        "Groq API configured successfully."
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
    "wormbite",
]


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = [
    "en",
    "kn",
    "hi",
]


# ============================================================
# GROQ CACHE
#
# Important:
# Disease + language are used as the cache key.
#
# Example:
# ("early_spot", "en")
# ("early_spot", "kn")
# ("early_spot", "hi")
#
# These are treated as three different responses.
# ============================================================

disease_cache = {}


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
    ]
)


# ============================================================
# LOAD EFFICIENTNETV2-S
# ============================================================

def load_efficientnet():

    print(
        "Loading EfficientNetV2-S..."
    )

    # Create architecture
    model_instance = models.efficientnet_v2_s(
        weights=None
    )

    # Number of input features
    num_features = (
        model_instance
        .classifier[1]
        .in_features
    )

    # Replace classifier
    model_instance.classifier[1] = nn.Linear(
        num_features,
        7,
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    try:

        state = torch.load(
            efficientnet_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )

    except TypeError:

        # Compatibility for older PyTorch
        state = torch.load(
            efficientnet_path,
            map_location="cpu",
        )

    # --------------------------------------------------------
    # Load state dictionary
    # --------------------------------------------------------

    try:

        model_instance.load_state_dict(
            state,
            assign=True,
        )

    except TypeError:

        model_instance.load_state_dict(
            state
        )

    del state

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    model_instance.to(device)

    # Evaluation mode
    model_instance.eval()

    print(
        "EfficientNetV2-S loaded successfully."
    )

    return model_instance


# ============================================================
# FASTAPI STARTUP
#
# IMPORTANT:
# Model is loaded ONCE.
#
# Previously the model was loaded for every /predict request.
# That could cause:
#
# - slow second prediction
# - memory problems
# - Render restart
# - Failed fetch in Flutter
#
# Now it stays loaded in memory.
# ============================================================

@app.on_event("startup")
def startup_event():

    global model

    print(
        "=========================================="
    )

    print(
        "Initializing Groundnut Disease Model..."
    )

    model = load_efficientnet()

    print(
        "Model ready for predictions."
    )

    print(
        "=========================================="
    )


# ============================================================
# GROQ SUGGESTIONS
# ============================================================

def get_groq_suggestions(
    disease,
    language="en",
):

    # --------------------------------------------------------
    # Validate language
    # --------------------------------------------------------

    if language not in SUPPORTED_LANGUAGES:

        print(
            f"Unsupported language received: {language}"
        )

        language = "en"


    # --------------------------------------------------------
    # CACHE KEY
    # --------------------------------------------------------

    cache_key = (
        disease,
        language,
    )


    # --------------------------------------------------------
    # CHECK CACHE
    # --------------------------------------------------------

    if cache_key in disease_cache:

        print(
            f"Using cached Groq response: "
            f"{cache_key}"
        )

        return disease_cache[
            cache_key
        ]


    # ========================================================
    # LANGUAGE INSTRUCTIONS
    # ========================================================

    if language == "kn":

        language_instruction = """
You MUST write the explanation entirely in Kannada.

Use Kannada script (ಕನ್ನಡ ಲಿಪಿ).

DO NOT write the explanation in English.

DO NOT translate only a few words.

Every explanation sentence must be in Kannada.

The four labels MUST remain exactly as:

Cause:
Prevention:
Treatment:
Advice:

Only the content AFTER these labels must be in Kannada.
"""

    elif language == "hi":

        language_instruction = """
You MUST write the explanation entirely in Hindi.

Use Devanagari script (हिन्दी).

DO NOT write the explanation in English.

DO NOT translate only a few words.

Every explanation sentence must be in Hindi.

The four labels MUST remain exactly as:

Cause:
Prevention:
Treatment:
Advice:

Only the content AFTER these labels must be in Hindi.
"""

    else:

        language_instruction = """
You MUST write the explanation entirely in simple English.

Use simple English that farmers can understand.

The four labels MUST remain exactly as:

Cause:
Prevention:
Treatment:
Advice:
"""


    # ========================================================
    # GROQ PROMPT
    # ========================================================

    prompt = f"""
You are an agricultural expert helping
groundnut farmers.

The detected groundnut leaf disease is:

{disease}

The requested response language is:

{language}

{language_instruction}

Follow this EXACT format:

Cause: [explanation]

Prevention: [explanation]

Treatment: [explanation]

Advice: [explanation]

IMPORTANT RULES:

1. The content must be written completely
   in the selected language.

2. If the selected language is Kannada (kn),
   use Kannada script.

3. If the selected language is Hindi (hi),
   use Devanagari script.

4. If the selected language is English (en),
   use simple English.

5. Do NOT use English sentences when
   Kannada or Hindi is selected.

6. Keep these four labels exactly:

   Cause:
   Prevention:
   Treatment:
   Advice:

7. Do not add additional headings.

8. Do not use Markdown.

9. Keep the information simple.

10. Make the information useful for
    groundnut farmers.

11. Keep each section short.

12. Do not mention the language
    in your response.

13. Do not provide a translation.

14. Directly answer in the requested language.
"""


    # ========================================================
    # GROQ NOT CONFIGURED
    # ========================================================

    if groq_client is None:

        print(
            "Groq client is not configured."
        )

        return (
            "Cause: Information unavailable.\n"
            "Prevention: Information unavailable.\n"
            "Treatment: Information unavailable.\n"
            "Advice: Information unavailable."
        )


    # ========================================================
    # CALL GROQ
    # ========================================================

    try:

        print(
            "------------------------------------------"
        )

        print(
            "Calling Groq..."
        )

        print(
            f"Disease: {disease}"
        )

        print(
            f"Language: {language}"
        )

        print(
            "------------------------------------------"
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
                            "expert helping groundnut "
                            "farmers. "
                            "You MUST follow the "
                            "requested language exactly. "
                            "Never ignore the requested "
                            "language."
                        ),
                    },

                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                max_tokens=400,

                temperature=0.2,
            )
        )


        # ====================================================
        # GET RESPONSE
        # ====================================================

        result = (
            response
            .choices[0]
            .message
            .content
        )


        if not result:

            raise Exception(
                "Groq returned an empty response."
            )


        result = result.strip()


        # ====================================================
        # PRINT RESPONSE FOR RENDER DEBUGGING
        # ====================================================

        print(
            "Groq response:"
        )

        print(
            result
        )

        print(
            "------------------------------------------"
        )


        # ====================================================
        # CACHE RESPONSE
        # ====================================================

        disease_cache[
            cache_key
        ] = result


        return result


    except Exception as e:

        print(
            "Groq error:",
            str(e),
        )

        return (
            "Cause: Information unavailable.\n"
            "Prevention: Information unavailable.\n"
            "Treatment: Information unavailable.\n"
            "Advice: Information unavailable."
        )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "Groundnut Leaf Disease Recognition API",

        "status":
            "running",

        "model":
            "EfficientNetV2-S",

        "model_loaded":
            model is not None,

        "supported_languages":
            SUPPORTED_LANGUAGES,
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "model":
            "EfficientNetV2-S",

        "model_loaded":
            model is not None,

        "groq_configured":
            groq_client is not None,

        "supported_languages":
            SUPPORTED_LANGUAGES,
    }


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(

    file: UploadFile = File(...),

    language: str = Query(
        default="en"
    ),

):

    global model


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet.",
        )


    # ========================================================
    # PRINT REQUEST INFORMATION
    # ========================================================

    print(
        "=========================================="
    )

    print(
        "New prediction request"
    )

    print(
        f"Filename: {file.filename}"
    )

    print(
        f"Language received: {language}"
    )

    print(
        "=========================================="
    )


    # ========================================================
    # VALIDATE LANGUAGE
    # ========================================================

    if language not in SUPPORTED_LANGUAGES:

        print(
            f"Invalid language '{language}'. "
            "Using English."
        )

        language = "en"


    image = None
    image_tensor = None


    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        image_bytes = await file.read()


        if not image_bytes:

            raise Exception(
                "Uploaded file is empty."
            )


        print(
            f"Received image: "
            f"{len(image_bytes)} bytes"
        )


        # ====================================================
        # OPEN IMAGE
        # ====================================================

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        # We no longer need raw bytes
        del image_bytes


        print(
            "Image opened successfully."
        )


        # ====================================================
        # PREPROCESS
        # ====================================================

        image_tensor = (
            transform(image)
            .unsqueeze(0)
            .to(device)
        )


        print(
            "Image preprocessing complete."
        )


        # ====================================================
        # MODEL INFERENCE
        #
        # IMPORTANT:
        # Model is NOT loaded here.
        #
        # It was already loaded at startup.
        # ====================================================

        print(
            "Starting model inference..."
        )


        with model_lock:

            with torch.inference_mode():

                output = model(
                    image_tensor
                )

                probabilities = (
                    torch.softmax(
                        output,
                        dim=1,
                    )[0]
                )

                predicted_class = (
                    torch.argmax(
                        probabilities
                    ).item()
                )

                confidence = (
                    probabilities[
                        predicted_class
                    ].item()
                )


        # ====================================================
        # GET DISEASE NAME
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
        # FREE IMAGE MEMORY
        # ====================================================

        del image_tensor
        image_tensor = None

        del image
        image = None


        gc.collect()


        # ====================================================
        # GROQ
        # ====================================================

        print(
            "Requesting disease information..."
        )

        print(
            f"Groq language: {language}"
        )


        suggestions = get_groq_suggestions(
            disease_name,
            language,
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        print(
            "Prediction completed successfully."
        )

        print(
            "=========================================="
        )


        return {

            "disease":
                disease_name,

            "confidence":
                round(
                    confidence * 100,
                    2,
                ),

            "suggestions":
                suggestions,

        }


    # ========================================================
    # ERROR
    # ========================================================

    except Exception as e:

        print(
            "=========================================="
        )

        print(
            "Prediction error:",
            str(e),
        )

        print(
            "=========================================="
        )


        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


    # ========================================================
    # FINAL MEMORY CLEANUP
    # ========================================================

    finally:

        if image_tensor is not None:

            del image_tensor


        if image is not None:

            del image


        gc.collect()


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )