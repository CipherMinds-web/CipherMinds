from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import warnings
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # directory where apps.py is located
MODEL_DIR = os.path.join(BASE_DIR, "models")

TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LOGREG_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.joblib")

# Define the input data schema for the API
class TextIn(BaseModel):
    text: str

# Create the FastAPI app
app = FastAPI()

# Tell FastAPI where your 'static' folder is
app.mount("/static", StaticFiles(directory="static"), name="static")
# --- ADD THIS CODE BLOCK ---
# This defines the homepage route
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
# Add CORS middleware to allow requests from your frontend
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A dictionary to hold your loaded models and pipeline
models = {}


# Define the preprocessing function
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(cleaned_tokens)

# Load your trained models and NLTK data when the application starts
@app.on_event("startup")
async def load_models_and_nltk():
    """Load the models and create a pipeline."""
    try:        
        warnings.filterwarnings("ignore", category=UserWarning)

        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        global lemmatizer, stop_words
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        if not os.path.exists(TFIDF_PATH) or not os.path.exists(LOGREG_PATH):
             raise FileNotFoundError("Misinformation model files not found.")
        models['misinformation_vectorizer'] = joblib.load(TFIDF_PATH)
        models['misinformation_classifier'] = joblib.load(LOGREG_PATH)
        models['hate_speech_vectorizer'] = TfidfVectorizer()
        models['hate_speech_classifier'] = LinearSVC(C=1.0)
        
        X_dummy = ["i hate you", "you are a fool", "this is great", "I love a good movie", "I hate that group"]
        y_dummy = [0, 1, 2, 2, 0]
        
        X_vectorized = models['hate_speech_vectorizer'].fit_transform(X_dummy)
        models['hate_speech_classifier'].fit(X_vectorized, y_dummy)

    except FileNotFoundError as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load models: {e}")

# API endpoint for hate speech detection
@app.post("/api/hate-speech")
async def predict_hate_speech(data: TextIn):
    try:
        vectorized_text = models['hate_speech_vectorizer'].transform([data.text])
        prediction_label = models['hate_speech_classifier'].predict(vectorized_text)[0]
        
        # Access the specific confidence score for the predicted label
        confidence_scores = models['hate_speech_classifier'].decision_function(vectorized_text)[0]
        confidence_score = confidence_scores[prediction_label]

        if prediction_label == 0:
            label = "Hate Speech"
            explanation = "The model detected patterns of hate speech. This is a severe flag."
        elif prediction_label == 1:
            label = "Offensive Language"
            explanation = "The model found language that is offensive but not hate speech."
        else:
            label = "Neither"
            explanation = "The content is safe and does not contain hate speech or offensive language."

        return {
            "prediction": label,
            "confidence": float(abs(confidence_score)),
            "explanation": explanation
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/api/misinformation")
async def predict_misinformation(data: TextIn):
    try:
        vectorized_text = models['misinformation_vectorizer'].transform([data.text])
        prediction_index = models['misinformation_classifier'].predict(vectorized_text)[0]
        probabilities = models['misinformation_classifier'].predict_proba(vectorized_text)[0]
        confidence = probabilities[prediction_index]
        
        if prediction_index == 1:
            prediction_label = "Misinformation"
        else:
            prediction_label = "Reliable Content"

        return {
            "prediction": prediction_label,
            "confidence": float(confidence),
            "explanation": f"The model predicts '{prediction_label}' with {confidence:.2f} confidence."
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

